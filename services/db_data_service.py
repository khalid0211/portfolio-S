"""Database-backed data service for Portfolio Analysis Suite

This module provides database CRUD operations as an alternative to session state.
Parallel to data_service.py but uses DuckDB instead of st.session_state.

Phase B: Data Abstraction Layer
"""

from typing import Dict, Any, List, Optional, Tuple
import pandas as pd
from datetime import datetime, date
import logging

from database.db_connection import get_db
from database.queries import (
    get_effective_baseline,
    get_projects_by_portfolio,
    get_project_by_id,
    get_project_by_name,
    get_latest_status_report,
    get_status_report_on_date,
    get_all_baselines,
    get_active_baseline,
    get_portfolios_by_user,
    check_planned_start_lock,
    get_portfolio_snapshot_for_batch_calculation
)
from models.project import ProjectData, ColumnMapping

logger = logging.getLogger(__name__)


class DatabaseDataManager:
    """Database-backed data operations (alternative to session state DataManager)"""

    def __init__(self):
        self._db = None  # Lazy initialization

    @property
    def db(self):
        """Lazy-load database connection to avoid initialization before user auth"""
        if self._db is None:
            # Try to get user email from session state for user-specific config
            try:
                import streamlit as st
                user_email = None
                if hasattr(st, 'session_state') and 'user_info' in st.session_state:
                    user_email = st.session_state.user_info.get('email')
                self._db = get_db(user_email=user_email)
            except Exception:
                self._db = get_db()
        return self._db

    def reset_connection(self):
        """Reset database connection (call after user changes database preference)"""
        self._db = None

    # ========================================================================
    # PORTFOLIO OPERATIONS
    # ========================================================================

    def create_portfolio(
        self,
        name: str,
        managing_org: Optional[str] = None,
        manager: Optional[str] = None,
        owner_user_id: int = 1  # Default owner (will be from auth in Phase D)
    ) -> int:
        """Create new portfolio and return ID

        Args:
            name: Portfolio name (unique among active portfolios)
            managing_org: Managing organization (optional)
            manager: Portfolio manager name (optional)
            owner_user_id: Owner user ID (FK to app_user)

        Returns:
            portfolio_id of created portfolio

        Raises:
            ValueError: If an active portfolio with this name already exists
        """
        try:
            # Check if an active portfolio with this name already exists
            existing = self.db.fetch_one("""
                SELECT portfolio_id, portfolio_name
                FROM portfolio
                WHERE portfolio_name = ? AND is_active = TRUE
            """, (name,))

            if existing:
                raise ValueError(f"An active portfolio with the name '{name}' already exists (ID: {existing[0]})")

            # Generate portfolio ID (simple auto-increment for now)
            result = self.db.fetch_one("SELECT COALESCE(MAX(portfolio_id), 0) + 1 FROM portfolio")
            if result is None:
                raise RuntimeError("Database connection error: could not generate portfolio_id")
            portfolio_id = result[0]

            # Insert into portfolio table (without owner_user_id - that's in portfolio_ownership now)
            self.db.execute("""
                INSERT INTO portfolio (
                    portfolio_id, portfolio_name, managing_organization,
                    portfolio_manager, is_active, created_at
                ) VALUES (?, ?, ?, ?, TRUE, now())
            """, (portfolio_id, name, managing_org, manager))

            # Insert into portfolio_ownership table
            self.db.execute("""
                INSERT INTO portfolio_ownership (
                    portfolio_id, owner_user_id, is_active, assigned_at
                ) VALUES (?, ?, TRUE, now())
            """, (portfolio_id, owner_user_id))

            logger.info(f"Portfolio created: {portfolio_id} - {name} (owner: {owner_user_id})")
            return portfolio_id

        except Exception as e:
            logger.error(f"Error creating portfolio: {e}")
            raise

    def get_portfolio(self, portfolio_id: int) -> Optional[Dict]:
        """Get portfolio by ID (with owner information from portfolio_ownership)

        Args:
            portfolio_id: Portfolio ID

        Returns:
            Dictionary with portfolio data including owner_user_id, or None if not found
        """
        try:
            result = self.db.fetch_one("""
                SELECT
                    p.*,
                    po.owner_user_id,
                    po.created_at as owner_assigned_at
                FROM portfolio p
                LEFT JOIN portfolio_ownership po ON p.portfolio_id = po.portfolio_id
                WHERE p.portfolio_id = ?
            """, (portfolio_id,))

            if result:
                columns = ['portfolio_id', 'portfolio_name', 'managing_organization',
                          'portfolio_manager', 'is_active', 'created_at', 'description',
                          'owner_user_id', 'owner_assigned_at']
                return dict(zip(columns, result))
            return None

        except Exception as e:
            logger.error(f"Error getting portfolio: {e}")
            return None

    def list_portfolios(self, user_id: Optional[int] = None) -> pd.DataFrame:
        """List all portfolios (optionally filtered by owner)

        Args:
            user_id: Owner user ID (if None, returns all portfolios)

        Returns:
            DataFrame with portfolio data
        """
        return get_portfolios_by_user(user_id)

    def update_portfolio(self, portfolio_id: int, updates: Dict[str, Any]):
        """Update portfolio fields (owner updates go to portfolio_ownership table)

        Args:
            portfolio_id: Portfolio ID
            updates: Dictionary of field: value pairs to update
        """
        try:
            # Separate owner_user_id from other updates
            owner_user_id = updates.pop('owner_user_id', None)

            # Update portfolio table fields
            if updates:
                set_clauses = []
                values = []

                for field, value in updates.items():
                    if field not in ['portfolio_id', 'created_at']:  # Protect immutable fields
                        set_clauses.append(f"{field} = ?")
                        values.append(value)

                if set_clauses:
                    values.append(portfolio_id)
                    query = f"UPDATE portfolio SET {', '.join(set_clauses)} WHERE portfolio_id = ?"
                    self.db.execute(query, tuple(values))
                    logger.info(f"Portfolio updated: {portfolio_id}")

            # Update owner in portfolio_ownership table (no FK constraint issues!)
            if owner_user_id is not None:
                # Check if ownership record exists
                existing = self.db.fetch_one(
                    "SELECT 1 FROM portfolio_ownership WHERE portfolio_id = ?",
                    (portfolio_id,)
                )

                if existing:
                    # Update existing ownership
                    self.db.execute("""
                        UPDATE portfolio_ownership
                        SET owner_user_id = ?, assigned_at = now()
                        WHERE portfolio_id = ?
                    """, (owner_user_id, portfolio_id))
                else:
                    # Insert new ownership record
                    self.db.execute("""
                        INSERT INTO portfolio_ownership (portfolio_id, owner_user_id, is_active, assigned_at)
                        VALUES (?, ?, TRUE, now())
                    """, (portfolio_id, owner_user_id))

                logger.info(f"Portfolio {portfolio_id} owner updated to user {owner_user_id}")

        except Exception as e:
            logger.error(f"Error updating portfolio: {e}")
            raise

    def delete_portfolio(self, portfolio_id: int, cascade: bool = True):
        """Soft delete portfolio (set is_active = FALSE) and cascade to ALL child records

        This marks the portfolio and all related records as inactive (is_active = FALSE).
        The records remain in the database until purged via Database Diagnostics.

        Cascade sequence (respects FK dependencies):
        1. project_status_report
        2. project_sdg
        3. project_factor_score
        4. project_baseline
        5. project
        6. portfolio_ownership
        7. portfolio_factor
        8. portfolio

        Args:
            portfolio_id: Portfolio ID
            cascade: If True, cascades to all child records (default: True)

        Returns:
            dict: Summary of soft-deleted records
        """
        try:
            deleted_summary = {
                'portfolio': 0,
                'project_status_report': 0,
                'project_sdg': 0,
                'project_factor_score': 0,
                'project_baseline': 0,
                'projects': 0,
                'portfolio_access': 0,
                'portfolio_ownership': 0,
                'portfolio_factors': 0
            }

            if cascade:
                # Soft delete in correct FK dependency order
                # All updates use portfolio_id to catch all related records

                # 1. project_status_report
                self.db.execute(
                    "UPDATE project_status_report SET is_active = FALSE WHERE portfolio_id = ?",
                    (portfolio_id,)
                )
                count = self.db.fetch_one(
                    "SELECT COUNT(*) FROM project_status_report WHERE portfolio_id = ? AND is_active = FALSE",
                    (portfolio_id,)
                )[0]
                deleted_summary['project_status_report'] = count
                if count > 0:
                    logger.info(f"  Soft-deleted {count} status report(s) for portfolio {portfolio_id}")

                # 2. project_sdg
                self.db.execute(
                    "UPDATE project_sdg SET is_active = FALSE WHERE portfolio_id = ?",
                    (portfolio_id,)
                )
                count = self.db.fetch_one(
                    "SELECT COUNT(*) FROM project_sdg WHERE portfolio_id = ? AND is_active = FALSE",
                    (portfolio_id,)
                )[0]
                deleted_summary['project_sdg'] = count
                if count > 0:
                    logger.info(f"  Soft-deleted {count} project-SDG association(s) for portfolio {portfolio_id}")

                # 3. project_factor_score
                self.db.execute(
                    "UPDATE project_factor_score SET is_active = FALSE WHERE portfolio_id = ?",
                    (portfolio_id,)
                )
                count = self.db.fetch_one(
                    "SELECT COUNT(*) FROM project_factor_score WHERE portfolio_id = ? AND is_active = FALSE",
                    (portfolio_id,)
                )[0]
                deleted_summary['project_factor_score'] = count
                if count > 0:
                    logger.info(f"  Soft-deleted {count} factor score(s) for portfolio {portfolio_id}")

                # 4. project_baseline
                self.db.execute(
                    "UPDATE project_baseline SET is_active = FALSE WHERE portfolio_id = ?",
                    (portfolio_id,)
                )
                count = self.db.fetch_one(
                    "SELECT COUNT(*) FROM project_baseline WHERE portfolio_id = ? AND is_active = FALSE",
                    (portfolio_id,)
                )[0]
                deleted_summary['project_baseline'] = count
                if count > 0:
                    logger.info(f"  Soft-deleted {count} baseline(s) for portfolio {portfolio_id}")

                # 5. project
                self.db.execute(
                    "UPDATE project SET is_active = FALSE WHERE portfolio_id = ?",
                    (portfolio_id,)
                )
                count = self.db.fetch_one(
                    "SELECT COUNT(*) FROM project WHERE portfolio_id = ? AND is_active = FALSE",
                    (portfolio_id,)
                )[0]
                deleted_summary['projects'] = count
                if count > 0:
                    logger.info(f"  Soft-deleted {count} project(s) for portfolio {portfolio_id}")

                # 6. portfolio_access
                self.db.execute(
                    "UPDATE portfolio_access SET is_active = FALSE WHERE portfolio_id = ?",
                    (portfolio_id,)
                )
                count = self.db.fetch_one(
                    "SELECT COUNT(*) FROM portfolio_access WHERE portfolio_id = ? AND is_active = FALSE",
                    (portfolio_id,)
                )[0]
                deleted_summary['portfolio_access'] = count
                if count > 0:
                    logger.info(f"  Soft-deleted {count} access record(s) for portfolio {portfolio_id}")

                # 7. portfolio_ownership
                self.db.execute(
                    "UPDATE portfolio_ownership SET is_active = FALSE WHERE portfolio_id = ?",
                    (portfolio_id,)
                )
                count = self.db.fetch_one(
                    "SELECT COUNT(*) FROM portfolio_ownership WHERE portfolio_id = ? AND is_active = FALSE",
                    (portfolio_id,)
                )[0]
                deleted_summary['portfolio_ownership'] = count
                if count > 0:
                    logger.info(f"  Soft-deleted {count} ownership record(s) for portfolio {portfolio_id}")

                # 7. portfolio_factor
                self.db.execute(
                    "UPDATE portfolio_factor SET is_active = FALSE WHERE portfolio_id = ?",
                    (portfolio_id,)
                )
                count = self.db.fetch_one(
                    "SELECT COUNT(*) FROM portfolio_factor WHERE portfolio_id = ? AND is_active = FALSE",
                    (portfolio_id,)
                )[0]
                deleted_summary['portfolio_factors'] = count
                if count > 0:
                    logger.info(f"  Soft-deleted {count} portfolio factor(s) for portfolio {portfolio_id}")

            # 8. Soft delete the portfolio itself
            self.db.execute(
                "UPDATE portfolio SET is_active = FALSE WHERE portfolio_id = ?",
                (portfolio_id,)
            )
            deleted_summary['portfolio'] = 1
            logger.info(f"Portfolio soft-deleted: {portfolio_id}")

            return deleted_summary

        except Exception as e:
            logger.error(f"Error deleting portfolio: {e}")
            raise

    # ========================================================================
    # PROJECT OPERATIONS
    # ========================================================================

    def create_project(
        self,
        portfolio_id: int,
        project_data: Dict[str, Any],
        create_baseline: bool = True
    ) -> int:
        """Create project and optionally create baseline v0

        Args:
            portfolio_id: Portfolio ID (FK)
            project_data: Dictionary with project fields
            create_baseline: If True, creates baseline version 0

        Returns:
            project_id of created project
        """
        try:
            # Generate project ID
            result = self.db.fetch_one("SELECT COALESCE(MAX(project_id), 0) + 1 FROM project")
            if result is None:
                raise RuntimeError("Database connection error: could not generate project_id")
            project_id = result[0]

            # Extract project fields
            project_name = project_data.get('project_name')
            project_code = project_data.get('project_code')
            responsible_org = project_data.get('responsible_organization')
            project_manager = project_data.get('project_manager')
            project_status = project_data.get('project_status', 'Ongoing')
            planned_start = project_data.get('planned_start_date')
            planned_finish = project_data.get('planned_finish_date')
            initial_budget = project_data.get('initial_budget', 0)
            current_budget = project_data.get('current_budget', initial_budget)

            # Insert project (planned dates go in baseline table, not here)
            self.db.execute("""
                INSERT INTO project (
                    project_id, portfolio_id, project_code, project_name,
                    responsible_organization, project_manager, project_status,
                    initial_budget, current_budget,
                    is_active, created_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, TRUE, now())
            """, (
                project_id, portfolio_id, project_code, project_name,
                responsible_org, project_manager, project_status,
                initial_budget, current_budget
            ))

            # Create baseline v0 if requested
            if create_baseline:
                try:
                    # Ensure dates are date objects (not strings)
                    baseline_start = planned_start or date.today()
                    if isinstance(baseline_start, str):
                        import pandas as pd
                        baseline_start = pd.to_datetime(baseline_start, dayfirst=True, errors='coerce')
                        if pd.isna(baseline_start):
                            baseline_start = date.today()
                        else:
                            baseline_start = baseline_start.date()

                    self._create_baseline_no_transaction(
                        project_id=project_id,
                        baseline_data={
                            'baseline_version': 0,
                            'baseline_start_date': baseline_start,
                            'baseline_end_date': None,  # Active baseline
                            'planned_start_date': planned_start,
                            'planned_finish_date': planned_finish,
                            'budget_at_completion': initial_budget,
                            'project_status': project_status,
                            'baseline_reason': 'Initial baseline'
                        }
                    )
                except Exception as baseline_error:
                    logger.error(f"Error creating baseline for project {project_id}: {baseline_error}")
                    import traceback
                    logger.error(f"Baseline error traceback: {traceback.format_exc()}")
                    # Continue without baseline rather than failing the entire project creation
                    logger.warning(f"Project {project_id} created without baseline")

            logger.info(f"Project created: {project_id} - {project_name}")
            return project_id

        except Exception as e:
            logger.error(f"Error creating project: {e}")
            import traceback
            logger.error(f"Traceback: {traceback.format_exc()}")
            raise

    def get_project(self, project_id: int) -> Optional[Dict]:
        """Get project by ID

        Args:
            project_id: Project ID

        Returns:
            Dictionary with project data, or None if not found
        """
        return get_project_by_id(project_id)

    def update_project(self, project_id: int, updates: Dict[str, Any]):
        """Update project fields

        Args:
            project_id: Project ID
            updates: Dictionary of field: value pairs to update
        """
        try:
            # Build UPDATE query
            set_clauses = []
            values = []

            # Filter out fields that don't exist in project table
            valid_fields = [
                'project_code', 'project_name', 'responsible_organization',
                'project_manager', 'project_status', 'initial_budget',
                'current_budget', 'is_active'
            ]

            for field, value in updates.items():
                if field in valid_fields:
                    set_clauses.append(f"{field} = ?")
                    values.append(value)

            if not set_clauses:
                logger.warning(f"No valid fields to update for project {project_id}")
                return

            values.append(project_id)
            query = f"UPDATE project SET {', '.join(set_clauses)} WHERE project_id = ?"

            self.db.execute(query, tuple(values))
            logger.info(f"Project updated: {project_id}")

        except Exception as e:
            logger.error(f"Error updating project: {e}")
            raise

    def delete_project(self, project_id: int, cascade: bool = True):
        """Soft delete project (set is_active = FALSE) and cascade to ALL child records

        This marks the project and all related records as inactive (is_active = FALSE).
        The records remain in the database until purged via Database Diagnostics.

        Cascade sequence (respects FK dependencies):
        1. project_status_report (checking both portfolio_id AND project_id)
        2. project_sdg (checking both portfolio_id AND project_id)
        3. project_factor_score (checking both portfolio_id AND project_id)
        4. project_baseline (checking both portfolio_id AND project_id)
        5. project

        Args:
            project_id: Project ID
            cascade: If True, cascades to all child records (default: True)

        Returns:
            dict: Summary of soft-deleted records
        """
        try:
            deleted_summary = {
                'project': 0,
                'project_status_report': 0,
                'project_sdg': 0,
                'project_factor_score': 0,
                'project_baseline': 0
            }

            # Get portfolio_id for this project
            result = self.db.fetch_one(
                "SELECT portfolio_id FROM project WHERE project_id = ?",
                (project_id,)
            )
            if not result:
                raise ValueError(f"Project {project_id} not found")
            portfolio_id = result[0]

            if cascade:
                # Soft delete in correct FK dependency order
                # Always check BOTH portfolio_id AND project_id

                # 1. project_status_report
                self.db.execute(
                    "UPDATE project_status_report SET is_active = FALSE WHERE portfolio_id = ? AND project_id = ?",
                    (portfolio_id, project_id)
                )
                count = self.db.fetch_one(
                    "SELECT COUNT(*) FROM project_status_report WHERE portfolio_id = ? AND project_id = ? AND is_active = FALSE",
                    (portfolio_id, project_id)
                )[0]
                deleted_summary['project_status_report'] = count
                if count > 0:
                    logger.info(f"  Soft-deleted {count} status report(s) for project {project_id}")

                # 2. project_sdg
                self.db.execute(
                    "UPDATE project_sdg SET is_active = FALSE WHERE portfolio_id = ? AND project_id = ?",
                    (portfolio_id, project_id)
                )
                count = self.db.fetch_one(
                    "SELECT COUNT(*) FROM project_sdg WHERE portfolio_id = ? AND project_id = ? AND is_active = FALSE",
                    (portfolio_id, project_id)
                )[0]
                deleted_summary['project_sdg'] = count
                if count > 0:
                    logger.info(f"  Soft-deleted {count} SDG association(s) for project {project_id}")

                # 3. project_factor_score
                self.db.execute(
                    "UPDATE project_factor_score SET is_active = FALSE WHERE portfolio_id = ? AND project_id = ?",
                    (portfolio_id, project_id)
                )
                count = self.db.fetch_one(
                    "SELECT COUNT(*) FROM project_factor_score WHERE portfolio_id = ? AND project_id = ? AND is_active = FALSE",
                    (portfolio_id, project_id)
                )[0]
                deleted_summary['project_factor_score'] = count
                if count > 0:
                    logger.info(f"  Soft-deleted {count} factor score(s) for project {project_id}")

                # 4. project_baseline
                self.db.execute(
                    "UPDATE project_baseline SET is_active = FALSE WHERE portfolio_id = ? AND project_id = ?",
                    (portfolio_id, project_id)
                )
                count = self.db.fetch_one(
                    "SELECT COUNT(*) FROM project_baseline WHERE portfolio_id = ? AND project_id = ? AND is_active = FALSE",
                    (portfolio_id, project_id)
                )[0]
                deleted_summary['project_baseline'] = count
                if count > 0:
                    logger.info(f"  Soft-deleted {count} baseline(s) for project {project_id}")

            # 5. Soft delete the project itself
            self.db.execute(
                "UPDATE project SET is_active = FALSE, updated_at = now() WHERE project_id = ?",
                (project_id,)
            )
            deleted_summary['project'] = 1
            logger.info(f"Project soft-deleted: {project_id}")

            return deleted_summary

        except Exception as e:
            logger.error(f"Error deleting project: {e}")
            raise

    # ========================================================================
    # BASELINE OPERATIONS
    # ========================================================================

    def _create_baseline_no_transaction(self, project_id: int, baseline_data: Dict[str, Any]) -> int:
        """Internal method to create baseline without transaction wrapper

        Used when already inside a transaction (e.g., from create_project)
        """
        try:
            # Check planned start lock
            planned_start = baseline_data.get('planned_start_date')
            if planned_start:
                is_locked = check_planned_start_lock(project_id)
                if is_locked:
                    # Get current planned start
                    active = get_active_baseline(project_id)
                    if active and active['planned_start_date'] != planned_start:
                        raise ValueError(
                            "Cannot change planned start date - project has actual costs recorded "
                            "(planned start is locked)"
                        )

            # Generate baseline ID
            result = self.db.fetch_one("SELECT COALESCE(MAX(baseline_id), 0) + 1 FROM project_baseline")
            if result is None:
                raise RuntimeError("Database connection error: could not generate baseline_id")
            baseline_id = result[0]

            # Get version number
            baseline_version = baseline_data.get('baseline_version')
            if baseline_version is None:
                # Auto-increment version
                result = self.db.fetch_one(
                    "SELECT COALESCE(MAX(baseline_version), -1) + 1 FROM project_baseline WHERE project_id = ?",
                    (project_id,)
                )
                baseline_version = result[0]

            # Close previous active baseline if this is a new active baseline
            baseline_end_date = baseline_data.get('baseline_end_date')
            if baseline_end_date is None:  # This is the new active baseline
                baseline_start_date = baseline_data.get('baseline_start_date')

                # Set end_date of previous active baseline to (new start_date - 1 day)
                if baseline_start_date:
                    from datetime import timedelta
                    import pandas as pd

                    # Ensure baseline_start_date is a date object (not string)
                    if isinstance(baseline_start_date, str):
                        baseline_start_date = pd.to_datetime(baseline_start_date, dayfirst=True, errors='coerce')
                    if isinstance(baseline_start_date, pd.Timestamp):
                        baseline_start_date = baseline_start_date.date()

                    prev_end_date = baseline_start_date - timedelta(days=1)

                    self.db.execute("""
                        UPDATE project_baseline
                        SET baseline_end_date = ?
                        WHERE project_id = ? AND baseline_end_date IS NULL
                    """, (prev_end_date, project_id))

            # Get portfolio_id for the project
            result = self.db.fetch_one("SELECT portfolio_id FROM project WHERE project_id = ?", (project_id,))
            if not result:
                raise ValueError(f"Project {project_id} not found")
            portfolio_id = result[0]

            # Insert baseline
            self.db.execute("""
                INSERT INTO project_baseline (
                    baseline_id, project_id, portfolio_id, baseline_version,
                    baseline_start_date, baseline_end_date,
                    planned_start_date, planned_finish_date,
                    budget_at_completion, project_status,
                    baseline_reason, approved_by, approved_date,
                    is_active, created_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, TRUE, now())
            """, (
                baseline_id,
                project_id,
                portfolio_id,
                baseline_version,
                baseline_data.get('baseline_start_date'),
                baseline_data.get('baseline_end_date'),
                baseline_data.get('planned_start_date'),
                baseline_data.get('planned_finish_date'),
                baseline_data.get('budget_at_completion'),
                baseline_data.get('project_status'),
                baseline_data.get('baseline_reason'),
                baseline_data.get('approved_by'),
                baseline_data.get('approved_date')
            ))

            logger.info(f"Baseline created: {baseline_id} (project {project_id}, version {baseline_version})")
            return baseline_id

        except Exception as e:
            logger.error(f"Error creating baseline: {e}")
            raise

    def create_baseline(self, project_id: int, baseline_data: Dict[str, Any]) -> int:
        """Create new baseline version

        Validates:
        - Planned start lock (cannot change planned_start_date if AC > 0)
        - Only one active baseline per project

        Args:
            project_id: Project ID
            baseline_data: Dictionary with baseline fields

        Returns:
            baseline_id of created baseline
        """
        # Simply call the internal method (no transaction wrapper here)
        return self._create_baseline_no_transaction(project_id, baseline_data)

    def get_baseline(self, baseline_id: int) -> Optional[Dict]:
        """Get baseline by ID

        Args:
            baseline_id: Baseline ID

        Returns:
            Dictionary with baseline data, or None if not found
        """
        try:
            result = self.db.fetch_one(
                "SELECT * FROM project_baseline WHERE baseline_id = ?",
                (baseline_id,)
            )

            if result:
                columns = self.db.get_table_columns('project_baseline')
                return dict(zip(columns, result))
            return None

        except Exception as e:
            logger.error(f"Error getting baseline: {e}")
            return None

    def get_effective_baseline_for_project(
        self,
        project_id: int,
        status_date: date
    ) -> Optional[Dict]:
        """Get effective baseline on a date

        Args:
            project_id: Project ID
            status_date: Status date

        Returns:
            Dictionary with effective baseline data
        """
        return get_effective_baseline(project_id, status_date)

    def list_baselines(self, project_id: int) -> pd.DataFrame:
        """Get all baselines for a project

        Args:
            project_id: Project ID

        Returns:
            DataFrame with all baselines, ordered by version
        """
        return get_all_baselines(project_id)

    def update_baseline(self, baseline_id: int, baseline_data: Dict[str, Any]) -> bool:
        """Update existing baseline

        Args:
            baseline_id: Baseline ID to update
            baseline_data: Dictionary with fields to update

        Returns:
            True if updated successfully

        Raises:
            ValueError: If baseline_id doesn't exist or if trying to update baseline 0
        """
        try:
            # Check if baseline exists and get version
            existing = self.get_baseline(baseline_id)
            if not existing:
                raise ValueError(f"Baseline {baseline_id} not found")

            # Prevent modification of baseline 0
            if existing.get('baseline_version') == 0:
                raise ValueError("Baseline 0 (initial baseline) cannot be edited")

            # Build update query dynamically based on provided fields
            update_fields = []
            params = []

            allowed_fields = [
                'planned_start_date', 'planned_finish_date', 'budget_at_completion',
                'baseline_start_date', 'baseline_end_date', 'project_status',
                'baseline_reason', 'approved_by', 'approved_date'
            ]

            for field in allowed_fields:
                if field in baseline_data:
                    update_fields.append(f"{field} = ?")
                    params.append(baseline_data[field])

            if not update_fields:
                logger.warning("No valid fields to update")
                return False

            # Add baseline_id to params
            params.append(baseline_id)

            query = f"""
                UPDATE project_baseline
                SET {', '.join(update_fields)}
                WHERE baseline_id = ?
            """

            self.db.execute(query, tuple(params))
            logger.info(f"Baseline {baseline_id} updated successfully")
            return True

        except Exception as e:
            logger.error(f"Error updating baseline: {e}")
            raise

    def delete_baseline(self, baseline_id: int) -> bool:
        """Delete baseline

        Args:
            baseline_id: Baseline ID to delete

        Returns:
            True if deleted successfully

        Raises:
            ValueError: If baseline doesn't exist or if trying to delete baseline 0
        """
        try:
            # Check if baseline exists and get version
            existing = self.get_baseline(baseline_id)
            if not existing:
                raise ValueError(f"Baseline {baseline_id} not found")

            # Prevent deletion of baseline 0
            if existing.get('baseline_version') == 0:
                raise ValueError("Baseline 0 (initial baseline) cannot be deleted")

            # Check if this is the active baseline
            is_active = existing.get('baseline_end_date') is None

            # Delete the baseline
            self.db.execute(
                "DELETE FROM project_baseline WHERE baseline_id = ?",
                (baseline_id,)
            )

            # If we deleted the active baseline, make the previous one active
            if is_active:
                project_id = existing.get('project_id')
                # Find the most recent baseline with end_date set
                self.db.execute("""
                    UPDATE project_baseline
                    SET baseline_end_date = NULL
                    WHERE project_id = ?
                      AND baseline_version = (
                          SELECT MAX(baseline_version)
                          FROM project_baseline
                          WHERE project_id = ?
                      )
                """, (project_id, project_id))

            logger.info(f"Baseline {baseline_id} deleted successfully")
            return True

        except Exception as e:
            logger.error(f"Error deleting baseline: {e}")
            raise

    # ========================================================================
    # STATUS REPORT OPERATIONS
    # ========================================================================

    def create_status_report(
        self,
        project_id: int,
        portfolio_id: int,
        status_data: Dict[str, Any]
    ) -> int:
        """Create status report

        Args:
            project_id: Project ID
            portfolio_id: Portfolio ID (denormalized for query speed)
            status_data: Dictionary with status report fields

        Returns:
            status_report_id of created report
        """
        try:
            # Generate status report ID
            result = self.db.fetch_one("SELECT COALESCE(MAX(status_report_id), 0) + 1 FROM project_status_report")
            if result is None:
                raise RuntimeError("Database connection error: could not generate status_report_id")
            status_report_id = result[0]

            # Insert status report
            self.db.execute("""
                INSERT INTO project_status_report (
                    status_report_id, portfolio_id, project_id, status_date,
                    actual_cost, planned_value, earned_value, notes, is_active, created_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, TRUE, now())
            """, (
                status_report_id,
                portfolio_id,
                project_id,
                status_data.get('status_date'),
                status_data.get('actual_cost', 0),
                status_data.get('planned_value'),
                status_data.get('earned_value'),
                status_data.get('notes')
            ))

            logger.info(f"Status report created: {status_report_id} (project {project_id})")
            return status_report_id

        except Exception as e:
            logger.error(f"Error creating status report: {e}")
            raise

    def get_status_reports(self, project_id: int) -> pd.DataFrame:
        """Get all status reports for a project (time series)

        Args:
            project_id: Project ID

        Returns:
            DataFrame with all status reports, ordered by date
        """
        try:
            query = """
                SELECT * FROM project_status_report
                WHERE project_id = ?
                ORDER BY status_date ASC
            """
            return self.db.execute(query, (project_id,)).df()

        except Exception as e:
            logger.error(f"Error getting status reports: {e}")
            return pd.DataFrame()

    def get_portfolio_status_on_date(self, portfolio_id: int, status_date: date) -> pd.DataFrame:
        """Get all project statuses for a portfolio on a specific date

        Args:
            portfolio_id: Portfolio ID
            status_date: Status date

        Returns:
            DataFrame with status reports for all projects on that date
        """
        from database.queries import get_status_reports_by_portfolio_date
        return get_status_reports_by_portfolio_date(portfolio_id, status_date)

    # ========================================================================
    # DATA LOADING (CSV → DB)
    # ========================================================================

    def load_csv_to_db(
        self,
        df: pd.DataFrame,
        portfolio_id: int,
        status_date: date,
        column_mapping: Dict[str, str]
    ) -> Dict[str, int]:
        """Load CSV data into database tables

        For each row in DataFrame:
        1. Check if project exists (by project_name within portfolio)
        2. If new → INSERT into project + create baseline v0
        3. If exists → UPDATE project (if needed)
        4. INSERT status report for this status_date

        Args:
            df: DataFrame with project data (mapped columns)
            portfolio_id: Portfolio ID to assign projects to
            status_date: Status date for status reports
            column_mapping: Column name mappings

        Returns:
            Dictionary with counts:
            {
                'projects_created': int,
                'projects_updated': int,
                'status_reports_created': int,
                'errors': int
            }
        """
        stats = {
            'projects_created': 0,
            'projects_updated': 0,
            'status_reports_created': 0,
            'errors': 0
        }

        total_rows = len(df)
        reconnect_interval = 25  # Reconnect every N rows to prevent connection issues

        # Process each row individually (no transaction wrapper to avoid transaction abort issues)
        for idx, row in df.iterrows():
            try:
                # Periodic reconnection to prevent connection corruption during large batch loads
                if idx > 0 and idx % reconnect_interval == 0:
                    logger.info(f"Processing row {idx}/{total_rows} - refreshing database connection...")
                    try:
                        self.db._reconnect()
                    except Exception as reconnect_err:
                        logger.warning(f"Reconnection warning: {reconnect_err}")

                # Extract project data
                project_name = str(row[column_mapping.get('pname_col', 'Project')])
                project_code = row.get(column_mapping.get('pid_col', 'Project ID'))
                responsible_org = row.get(column_mapping.get('org_col', 'Organization'), '')
                project_manager = row.get(column_mapping.get('pm_col', 'Project Manager'), '')

                try:
                    bac = float(row[column_mapping.get('bac_col', 'BAC')])
                    ac = float(row[column_mapping.get('ac_col', 'AC')])
                except (ValueError, TypeError) as e:
                    logger.error(f"Row {idx} ({project_name}): Invalid numeric value for BAC or AC: {e}")
                    stats['errors'] += 1
                    continue

                # Convert dates to date objects (they may come as strings from CSV)
                plan_start = row[column_mapping.get('st_col', 'Plan Start')]
                plan_finish = row[column_mapping.get('fn_col', 'Plan Finish')]

                # Convert plan_start to date object
                if isinstance(plan_start, str):
                    plan_start = pd.to_datetime(plan_start, dayfirst=True, errors='coerce')
                if isinstance(plan_start, pd.Timestamp):
                    # Check for NaT (Not a Time)
                    if pd.isna(plan_start):
                        logger.warning(f"Row {idx}: Invalid plan_start date, using today's date")
                        plan_start = date.today()
                    else:
                        plan_start = plan_start.date()
                elif plan_start is None or (isinstance(plan_start, float) and pd.isna(plan_start)):
                    logger.warning(f"Row {idx}: Missing plan_start date, using today's date")
                    plan_start = date.today()
                elif not isinstance(plan_start, date):
                    logger.warning(f"Row {idx}: plan_start is type {type(plan_start)}, using today's date")
                    plan_start = date.today()

                # Convert plan_finish to date object
                if isinstance(plan_finish, str):
                    plan_finish = pd.to_datetime(plan_finish, dayfirst=True, errors='coerce')
                if isinstance(plan_finish, pd.Timestamp):
                    # Check for NaT (Not a Time)
                    if pd.isna(plan_finish):
                        logger.warning(f"Row {idx}: Invalid plan_finish date, using 1 year from start")
                        from datetime import timedelta
                        plan_finish = plan_start + timedelta(days=365)
                    else:
                        plan_finish = plan_finish.date()
                elif plan_finish is None or (isinstance(plan_finish, float) and pd.isna(plan_finish)):
                    logger.warning(f"Row {idx}: Missing plan_finish date, using 1 year from start")
                    from datetime import timedelta
                    plan_finish = plan_start + timedelta(days=365)
                elif not isinstance(plan_finish, date):
                    logger.warning(f"Row {idx}: plan_finish is type {type(plan_finish)}, using 1 year from start")
                    from datetime import timedelta
                    plan_finish = plan_start + timedelta(days=365)

                # Manual PV/EV if provided
                pv = row.get(column_mapping.get('pv_col', 'PV'))
                ev = row.get(column_mapping.get('ev_col', 'EV'))

                # Check if project exists (with retry on connection error)
                try:
                    existing = get_project_by_name(portfolio_id, project_name)
                except Exception as lookup_err:
                    logger.warning(f"Project lookup failed, retrying after reconnection: {lookup_err}")
                    self.db._reconnect()
                    existing = get_project_by_name(portfolio_id, project_name)

                if existing:
                    # Update project (planned dates are in baseline, not here)
                    project_id = existing['project_id']
                    try:
                        self.update_project(project_id, {
                            'project_code': project_code,
                            'responsible_organization': responsible_org,
                            'project_manager': project_manager,
                            'current_budget': bac
                        })
                    except RuntimeError as update_err:
                        logger.warning(f"Update failed, retrying: {update_err}")
                        self.db._reconnect()
                        self.update_project(project_id, {
                            'project_code': project_code,
                            'responsible_organization': responsible_org,
                            'project_manager': project_manager,
                            'current_budget': bac
                        })
                    stats['projects_updated'] += 1

                else:
                    # Create new project with baseline v0
                    try:
                        project_id = self.create_project(
                            portfolio_id=portfolio_id,
                            project_data={
                                'project_name': project_name,
                                'project_code': project_code,
                                'responsible_organization': responsible_org,
                                'project_manager': project_manager,
                                'planned_start_date': plan_start,
                                'planned_finish_date': plan_finish,
                                'initial_budget': bac,
                                'current_budget': bac
                            },
                            create_baseline=True
                        )
                    except RuntimeError as create_err:
                        logger.warning(f"Create failed, retrying: {create_err}")
                        self.db._reconnect()
                        project_id = self.create_project(
                            portfolio_id=portfolio_id,
                            project_data={
                                'project_name': project_name,
                                'project_code': project_code,
                                'responsible_organization': responsible_org,
                                'project_manager': project_manager,
                                'planned_start_date': plan_start,
                                'planned_finish_date': plan_finish,
                                'initial_budget': bac,
                                'current_budget': bac
                            },
                            create_baseline=True
                        )
                    stats['projects_created'] += 1

                # Create status report (with retry)
                try:
                    self.create_status_report(
                        project_id=project_id,
                        portfolio_id=portfolio_id,
                        status_data={
                            'status_date': status_date,
                            'actual_cost': ac,
                            'planned_value': pv if pd.notna(pv) else None,
                            'earned_value': ev if pd.notna(ev) else None
                        }
                    )
                except RuntimeError as report_err:
                    logger.warning(f"Status report creation failed, retrying: {report_err}")
                    self.db._reconnect()
                    self.create_status_report(
                        project_id=project_id,
                        portfolio_id=portfolio_id,
                        status_data={
                            'status_date': status_date,
                            'actual_cost': ac,
                            'planned_value': pv if pd.notna(pv) else None,
                            'earned_value': ev if pd.notna(ev) else None
                        }
                    )
                stats['status_reports_created'] += 1

            except Exception as e:
                logger.error(f"Error processing row {idx} ({project_name}): {e}")
                import traceback
                logger.error(f"Traceback: {traceback.format_exc()}")
                stats['errors'] += 1
                continue

        logger.info(f"CSV load complete: {stats}")
        return stats

    # ========================================================================
    # SDG OPERATIONS
    # ========================================================================

    def get_all_sdgs(self) -> pd.DataFrame:
        """Get all SDG goals

        Returns:
            DataFrame with columns: sdg_id, sdg_name
        """
        try:
            return self.db.execute("SELECT sdg_id, sdg_name FROM sdg ORDER BY sdg_id").df()
        except Exception as e:
            logger.error(f"Error getting SDGs: {e}")
            return pd.DataFrame(columns=['sdg_id', 'sdg_name'])

    def get_project_sdgs(self, project_id: int) -> List[int]:
        """Get SDG IDs assigned to a project

        Args:
            project_id: Project ID

        Returns:
            List of SDG IDs
        """
        try:
            result = self.db.execute(
                "SELECT sdg_id FROM project_sdg WHERE project_id = ? ORDER BY sdg_id",
                (project_id,)
            ).fetchall()
            return [row[0] for row in result]
        except Exception as e:
            logger.error(f"Error getting project SDGs: {e}")
            return []

    def assign_sdg_to_project(self, project_id: int, sdg_id: int) -> bool:
        """Assign SDG to project

        Args:
            project_id: Project ID
            sdg_id: SDG ID (1-17)

        Returns:
            True if successful
        """
        try:
            self.db.execute("""
                INSERT OR IGNORE INTO project_sdg (project_id, sdg_id)
                VALUES (?, ?)
            """, (project_id, sdg_id))
            logger.info(f"SDG {sdg_id} assigned to project {project_id}")
            return True
        except Exception as e:
            logger.error(f"Error assigning SDG: {e}")
            return False

    def remove_sdg_from_project(self, project_id: int, sdg_id: int) -> bool:
        """Remove SDG from project

        Args:
            project_id: Project ID
            sdg_id: SDG ID

        Returns:
            True if successful
        """
        try:
            self.db.execute("""
                DELETE FROM project_sdg
                WHERE project_id = ? AND sdg_id = ?
            """, (project_id, sdg_id))
            logger.info(f"SDG {sdg_id} removed from project {project_id}")
            return True
        except Exception as e:
            logger.error(f"Error removing SDG: {e}")
            return False

    def get_sdg_statistics(self, portfolio_id: Optional[int] = None) -> pd.DataFrame:
        """Get SDG statistics (project count per SDG)

        Args:
            portfolio_id: Optional portfolio filter

        Returns:
            DataFrame with columns: sdg_id, sdg_name, project_count
        """
        try:
            if portfolio_id is not None:
                query = """
                    SELECT
                        s.sdg_id,
                        s.sdg_name,
                        COUNT(DISTINCT ps.project_id) as project_count
                    FROM sdg s
                    LEFT JOIN project_sdg ps ON s.sdg_id = ps.sdg_id
                    LEFT JOIN project p ON ps.project_id = p.project_id
                    WHERE p.portfolio_id = ? OR p.portfolio_id IS NULL
                    GROUP BY s.sdg_id, s.sdg_name
                    ORDER BY s.sdg_id
                """
                return self.db.execute(query, (portfolio_id,)).df()
            else:
                query = """
                    SELECT
                        s.sdg_id,
                        s.sdg_name,
                        COUNT(DISTINCT ps.project_id) as project_count
                    FROM sdg s
                    LEFT JOIN project_sdg ps ON s.sdg_id = ps.sdg_id
                    GROUP BY s.sdg_id, s.sdg_name
                    ORDER BY s.sdg_id
                """
                return self.db.execute(query).df()
        except Exception as e:
            logger.error(f"Error getting SDG statistics: {e}")
            return pd.DataFrame(columns=['sdg_id', 'sdg_name', 'project_count'])

    # ========================================================================
    # STRATEGIC FACTOR OPERATIONS
    # ========================================================================

    def create_factor(
        self,
        portfolio_id: int,
        factor_name: str,
        factor_weight_percent: float,
        likert_min: int = 1,
        likert_max: int = 5
    ) -> int:
        """Create strategic factor for portfolio

        Args:
            portfolio_id: Portfolio ID
            factor_name: Factor name
            factor_weight_percent: Weight percentage (should sum to 100 across all factors)
            likert_min: Minimum score (default 1)
            likert_max: Maximum score (default 5)

        Returns:
            factor_id of created factor
        """
        try:
            # Generate factor ID
            result = self.db.fetch_one("SELECT COALESCE(MAX(factor_id), 0) + 1 FROM portfolio_factor")
            if result is None:
                raise RuntimeError("Database connection error: could not generate factor_id")
            factor_id = result[0]

            self.db.execute("""
                INSERT INTO portfolio_factor (
                    factor_id, portfolio_id, factor_name,
                    factor_weight_percent, likert_min, likert_max,
                    is_active, created_at
                ) VALUES (?, ?, ?, ?, ?, ?, TRUE, now())
            """, (factor_id, portfolio_id, factor_name, factor_weight_percent, likert_min, likert_max))

            logger.info(f"Factor created: {factor_id} - {factor_name}")
            return factor_id
        except Exception as e:
            logger.error(f"Error creating factor: {e}")
            raise

    def list_factors(self, portfolio_id: int) -> pd.DataFrame:
        """List all factors for portfolio

        Args:
            portfolio_id: Portfolio ID

        Returns:
            DataFrame with factor details
        """
        try:
            return self.db.execute("""
                SELECT
                    factor_id,
                    portfolio_id,
                    factor_name,
                    factor_weight_percent,
                    likert_min,
                    likert_max,
                    is_active,
                    created_at
                FROM portfolio_factor
                WHERE portfolio_id = ? AND is_active = TRUE
                ORDER BY factor_name
            """, (portfolio_id,)).df()
        except Exception as e:
            logger.error(f"Error listing factors: {e}")
            return pd.DataFrame()

    def update_factor(
        self,
        factor_id: int,
        factor_name: str = None,
        factor_weight_percent: float = None,
        likert_min: int = None,
        likert_max: int = None
    ) -> bool:
        """Update strategic factor

        Args:
            factor_id: Factor ID to update
            factor_name: New factor name (optional)
            factor_weight_percent: New weight percentage (optional)
            likert_min: New minimum score (optional)
            likert_max: New maximum score (optional)

        Returns:
            True if successful

        Raises:
            ValueError: If factor not found or validation fails
        """
        try:
            # Verify factor exists
            existing = self.db.fetch_one(
                "SELECT likert_min, likert_max FROM portfolio_factor WHERE factor_id = ?",
                (factor_id,)
            )
            if not existing:
                raise ValueError(f"Factor {factor_id} not found")

            current_min, current_max = existing

            # Only validate if score range is being changed
            if likert_min is not None or likert_max is not None:
                new_min = likert_min if likert_min is not None else current_min
                new_max = likert_max if likert_max is not None else current_max

                # Check if there are existing scores outside new range
                invalid_scores = self.db.fetch_one("""
                    SELECT COUNT(*) FROM project_factor_score
                    WHERE factor_id = ?
                    AND (score < ? OR score > ?)
                """, (factor_id, new_min, new_max))

                if invalid_scores and invalid_scores[0] > 0:
                    raise ValueError(
                        f"Cannot change score range: {invalid_scores[0]} existing scores "
                        f"would fall outside the new range ({new_min}-{new_max})"
                    )

            # Build update query for all fields at once
            # Name and weight changes never affect foreign keys
            updates = []
            params = []

            if factor_name is not None:
                updates.append("factor_name = ?")
                params.append(factor_name)

            if factor_weight_percent is not None:
                updates.append("factor_weight_percent = ?")
                params.append(factor_weight_percent)

            if likert_min is not None:
                updates.append("likert_min = ?")
                params.append(likert_min)

            if likert_max is not None:
                updates.append("likert_max = ?")
                params.append(likert_max)

            if not updates:
                logger.warning("No fields to update")
                return False

            # Execute update (no FK constraint issues anymore)
            params.append(factor_id)
            query = f"UPDATE portfolio_factor SET {', '.join(updates)} WHERE factor_id = ?"

            self.db.execute(query, tuple(params))
            logger.info(f"Factor {factor_id} updated: {', '.join(updates)}")
            return True

        except Exception as e:
            logger.error(f"Error updating factor {factor_id}: {e}")
            import traceback
            logger.error(traceback.format_exc())
            raise

    def delete_factor(self, factor_id: int) -> bool:
        """Delete (soft delete) strategic factor

        Args:
            factor_id: Factor ID to delete

        Returns:
            True if successful

        Note:
            This is a soft delete - sets is_active = FALSE
            Associated project scores are preserved
        """
        try:
            # Soft delete
            self.db.execute(
                "UPDATE portfolio_factor SET is_active = FALSE WHERE factor_id = ?",
                (factor_id,)
            )
            logger.info(f"Factor {factor_id} deleted (soft)")
            return True

        except Exception as e:
            logger.error(f"Error deleting factor: {e}")
            raise

    def score_project_factor(
        self,
        project_id: int,
        factor_id: int,
        score: int,
        scored_by_user_id: Optional[int] = None
    ) -> bool:
        """Score project against strategic factor

        Args:
            project_id: Project ID
            factor_id: Factor ID
            score: Score value (must be within likert range)
            scored_by_user_id: Optional user ID who scored

        Returns:
            True if successful
        """
        try:
            # Get portfolio_id from project
            portfolio_id = self.db.fetch_one(
                "SELECT portfolio_id FROM project WHERE project_id = ?",
                (project_id,)
            )[0]

            self.db.execute("""
                INSERT INTO project_factor_score (
                    portfolio_id, project_id, factor_id, score, is_active, scored_at, scored_by_user_id
                ) VALUES (?, ?, ?, ?, TRUE, now(), ?)
                ON CONFLICT (project_id, factor_id) DO UPDATE SET
                    score = EXCLUDED.score,
                    scored_at = EXCLUDED.scored_at,
                    scored_by_user_id = EXCLUDED.scored_by_user_id
            """, (portfolio_id, project_id, factor_id, score, scored_by_user_id))

            logger.info(f"Project {project_id} scored {score} for factor {factor_id}")
            return True
        except Exception as e:
            logger.error(f"Error scoring project factor: {e}")
            return False

    def get_project_factor_scores(self, project_id: int) -> pd.DataFrame:
        """Get all factor scores for a project

        Args:
            project_id: Project ID

        Returns:
            DataFrame with factor scores
        """
        try:
            return self.db.execute("""
                SELECT
                    pf.factor_id,
                    pf.factor_name,
                    pf.factor_weight_percent,
                    pf.likert_min,
                    pf.likert_max,
                    pfs.score,
                    pfs.scored_at,
                    u.display_name as scored_by
                FROM portfolio_factor pf
                LEFT JOIN project_factor_score pfs ON pf.factor_id = pfs.factor_id AND pfs.project_id = ?
                LEFT JOIN app_user u ON pfs.scored_by_user_id = u.user_id
                WHERE pf.is_active = TRUE
                ORDER BY pf.factor_name
            """, (project_id,)).df()
        except Exception as e:
            logger.error(f"Error getting project factor scores: {e}")
            return pd.DataFrame()

    def get_portfolio_strategic_alignment(self, portfolio_id: int) -> pd.DataFrame:
        """Get strategic alignment scores for all projects in portfolio

        Args:
            portfolio_id: Portfolio ID

        Returns:
            DataFrame with project names and weighted alignment scores
        """
        try:
            query = """
                SELECT
                    p.project_id,
                    p.project_name,
                    SUM(
                        (CAST(pfs.score AS FLOAT) / CAST(pf.likert_max AS FLOAT)) *
                        (pf.factor_weight_percent / 100.0)
                    ) * 100 as alignment_score,
                    COUNT(DISTINCT pfs.factor_id) as factors_scored,
                    COUNT(DISTINCT pf.factor_id) as total_factors
                FROM project p
                LEFT JOIN portfolio_factor pf ON p.portfolio_id = pf.portfolio_id AND pf.is_active = TRUE
                LEFT JOIN project_factor_score pfs ON p.project_id = pfs.project_id AND pf.factor_id = pfs.factor_id
                WHERE p.portfolio_id = ? AND p.is_active = TRUE
                GROUP BY p.project_id, p.project_name
                ORDER BY alignment_score DESC
            """
            return self.db.execute(query, (portfolio_id,)).df()
        except Exception as e:
            logger.error(f"Error getting strategic alignment: {e}")
            return pd.DataFrame()

    # ========================================================================
    # BATCH QUERY FOR EVM CALCULATION
    # ========================================================================

    def get_projects_for_batch_calculation(
        self,
        portfolio_id: int,
        status_date: Optional[date] = None
    ) -> pd.DataFrame:
        """Get portfolio snapshot ready for batch EVM calculation

        Returns DataFrame with columns matching perform_batch_calculation() input format:
        - Project ID, Project, Organization, Project Manager
        - BAC, AC, Plan Start, Plan Finish, Data Date
        - Optional: PV, EV (if manual values provided)

        Args:
            portfolio_id: Portfolio ID
            status_date: Status date (if None, uses latest status for each project)

        Returns:
            DataFrame ready for perform_batch_calculation()
        """
        return get_portfolio_snapshot_for_batch_calculation(portfolio_id, status_date)



    # ========================================================================
    # EVM RESULTS STORAGE (Multi-Period Support)
    # ========================================================================

    def save_batch_evm_results(
        self,
        portfolio_id: int,
        status_date: date,
        results_df: pd.DataFrame
    ) -> int:
        """Save batch EVM calculation results to database

        Stores all calculated EVM metrics in project_status_report table
        so they don't need to be recalculated when viewing historical periods.

        Args:
            portfolio_id: Portfolio ID
            status_date: Status date for this batch
            results_df: DataFrame from perform_batch_calculation() with EVM results

        Returns:
            Number of records updated
        """
        updated_count = 0

        try:
            for _, row in results_df.iterrows():
                project_id = row.get('project_id')

                if not project_id:
                    continue

                # Helper function to safely get numeric value
                def safe_numeric(value, default=None):
                    if pd.isna(value):
                        return default
                    if value == float('inf') or value == float('-inf'):
                        return default
                    try:
                        return float(value)
                    except (ValueError, TypeError):
                        return default

                # Helper function to safely get date value
                def safe_date(value):
                    if pd.isna(value):
                        return None
                    if isinstance(value, (date, datetime)):
                        return value if isinstance(value, date) else value.date()
                    try:
                        return pd.to_datetime(value, format='%d/%m/%Y', dayfirst=True).date()
                    except:
                        return None

                # Update the status report with calculated values
                update_query = """
                    UPDATE project_status_report
                    SET
                        calculation_performed = TRUE,
                        calculation_timestamp = now(),
                        curve_type = ?,
                        alpha = ?,
                        beta = ?,
                        inflation_rate = ?,
                        actual_duration_months = ?,
                        original_duration_months = ?,
                        calculated_pv = ?,
                        calculated_ev = ?,
                        present_value = ?,
                        percent_complete = ?,
                        percent_budget_used = ?,
                        percent_time_used = ?,
                        percent_present_value_project = ?,
                        percent_likely_value_project = ?,
                        cv = ?,
                        cpi = ?,
                        eac = ?,
                        etc = ?,
                        vac = ?,
                        tcpi = ?,
                        sv = ?,
                        spi = ?,
                        es = ?,
                        espi = ?,
                        likely_duration = ?,
                        likely_completion = ?,
                        planned_value_project = ?,
                        likely_value_project = ?
                    WHERE project_id = ? AND status_date = ?
                """

                params = (
                    row.get('curve_type'),
                    safe_numeric(row.get('alpha')),
                    safe_numeric(row.get('beta')),
                    safe_numeric(row.get('inflation_rate')),
                    safe_numeric(row.get('actual_duration_months')),
                    safe_numeric(row.get('original_duration_months')),
                    safe_numeric(row.get('pv')),
                    safe_numeric(row.get('ev')),
                    safe_numeric(row.get('present_value')),
                    safe_numeric(row.get('percent_complete')),
                    safe_numeric(row.get('percent_budget_used')),
                    safe_numeric(row.get('percent_time_used')),
                    safe_numeric(row.get('percent_present_value_project')),
                    safe_numeric(row.get('percent_likely_value_project')),
                    safe_numeric(row.get('cv')),
                    safe_numeric(row.get('cpi')),
                    safe_numeric(row.get('eac')),
                    safe_numeric(row.get('etc')),
                    safe_numeric(row.get('vac')),
                    safe_numeric(row.get('tcpi')),
                    safe_numeric(row.get('sv')),
                    safe_numeric(row.get('spi')),
                    safe_numeric(row.get('es')),
                    safe_numeric(row.get('spie')),  # Fixed: was 'espi', should be 'spie'
                    safe_numeric(row.get('ld')),
                    safe_date(row.get('likely_completion')),
                    safe_numeric(row.get('planned_value_project')),
                    safe_numeric(row.get('likely_value_project')),
                    project_id,
                    status_date
                )

                self.db.execute(update_query, params)
                updated_count += 1

            logger.info(f"Saved EVM results for {updated_count} projects (portfolio={portfolio_id}, date={status_date})")
            return updated_count

        except Exception as e:
            logger.error(f"Error saving batch EVM results: {e}")
            raise

    def get_evm_results_for_period(
        self,
        portfolio_id: int,
        status_date: date
    ) -> pd.DataFrame:
        """Get stored EVM calculation results for a period

        Retrieves pre-calculated EVM metrics from database, avoiding recalculation.

        Args:
            portfolio_id: Portfolio ID
            status_date: Status date

        Returns:
            DataFrame with all EVM metrics already calculated
            Empty DataFrame if no calculated results found
        """
        try:
            query = """
                SELECT
                    p.project_id as project_id,
                    p.project_name as project_name,
                    p.project_code as project,
                    p.responsible_organization as organization,
                    p.project_manager,
                    pb.budget_at_completion as bac,
                    psr.actual_cost as ac,
                    pb.planned_start_date as plan_start,
                    pb.planned_finish_date as plan_finish,
                    psr.status_date as data_date,
                    COALESCE(psr.calculated_pv, psr.planned_value, 0) as pv,
                    COALESCE(psr.calculated_ev, psr.earned_value, 0) as ev,
                    psr.calculated_pv,
                    psr.calculated_ev,
                    psr.present_value,
                    psr.percent_complete,
                    psr.percent_budget_used,
                    psr.percent_time_used,
                    psr.cv,
                    psr.cpi,
                    psr.eac,
                    psr.etc,
                    psr.vac,
                    psr.tcpi,
                    psr.sv,
                    psr.spi,
                    psr.es,
                    psr.espi as spie,  -- Alias to match calculation results
                    psr.likely_duration,
                    psr.likely_completion,
                    psr.planned_value_project,
                    psr.likely_value_project,
                    psr.percent_present_value_project,
                    psr.percent_likely_value_project,
                    psr.curve_type,
                    psr.alpha,
                    psr.beta,
                    psr.inflation_rate,
                    psr.actual_duration_months,
                    psr.original_duration_months
                FROM project_status_report psr
                JOIN project p ON psr.project_id = p.project_id
                LEFT JOIN project_baseline pb ON p.project_id = pb.project_id
                WHERE psr.portfolio_id = ?
                  AND psr.status_date = ?
                  AND psr.calculation_performed = TRUE
                  AND p.is_active = TRUE
                  AND (pb.baseline_id IS NULL OR pb.baseline_end_date IS NULL OR pb.baseline_end_date > ?)
                ORDER BY p.project_name
            """

            result = self.db.execute(query, (
                portfolio_id,
                status_date,
                status_date
            ))

            df = result.df()
            return df

        except Exception as e:
            logger.error(f"Error retrieving EVM results: {e}")
            return pd.DataFrame()

    def check_evm_results_exist(
        self,
        portfolio_id: int,
        status_date: date
    ) -> bool:
        """Check if EVM results have been calculated for a period

        Args:
            portfolio_id: Portfolio ID
            status_date: Status date

        Returns:
            True if calculated results exist, False otherwise
        """
        try:
            query = """
                SELECT COUNT(*) as count
                FROM project_status_report
                WHERE portfolio_id = ?
                  AND status_date = ?
                  AND calculation_performed = TRUE
            """

            result = self.db.fetch_one(query, (portfolio_id, status_date))
            count = result[0] if result else 0

            return count > 0

        except Exception as e:
            logger.error(f"Error checking EVM results: {e}")
            return False


# Global instance for easy access (similar to data_service.py pattern)
db_data_manager = DatabaseDataManager()

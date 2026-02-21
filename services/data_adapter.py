"""Unified data adapter - serves data from session state OR database

This adapter provides a single interface that routes data requests to either:
- Session state (st.session_state) - Current behavior
- Database (DuckDB) - New behavior

The adapter ensures data format consistency regardless of source, so protected
calculation and chart functions receive data in the expected format.

Phase B: Data Abstraction Layer
"""

from typing import Dict, Any, Optional, List
import pandas as pd
from datetime import date, datetime
import logging

# Import configuration
from config.constants import USE_DATABASE

# Import both data managers
from services.data_service import DataManager
from services.db_data_service import DatabaseDataManager

# Import for session state access
try:
    import streamlit as st
    STREAMLIT_AVAILABLE = True
except ImportError:
    STREAMLIT_AVAILABLE = False
    st = None

logger = logging.getLogger(__name__)


class UnifiedDataAdapter:
    """Adapter that serves data from session state OR database based on config

    This is the PRIMARY interface for data access throughout the application.
    It ensures that regardless of data source, the format matches what
    protected calculation and chart functions expect.
    """

    def __init__(self):
        """Initialize adapter with both data managers"""
        self.session_manager = DataManager()
        self.db_manager = DatabaseDataManager()
        self.use_db = USE_DATABASE

    def get_mode(self) -> str:
        """Get current data source mode

        Returns:
            'database' or 'session_state'
        """
        return 'database' if self.use_db else 'session_state'

    # ========================================================================
    # PROJECT DATA FOR EVM ANALYSIS
    # ========================================================================

    def get_projects_for_analysis(
        self,
        portfolio_id: Optional[int] = None,
        status_date: Optional[date] = None
    ) -> pd.DataFrame:
        """Get projects ready for EVM analysis

        This is the KEY function that adapters use to provide data to
        perform_batch_calculation() in the exact format it expects.

        Returns DataFrame with columns:
        - Project ID, Project, Organization, Project Manager
        - BAC, AC, Plan Start, Plan Finish, Data Date
        - Optional: PV, EV (manual values)

        Args:
            portfolio_id: Portfolio ID (required in database mode, ignored in session mode)
            status_date: Status date (optional - uses latest if None)

        Returns:
            DataFrame ready for perform_batch_calculation()
        """
        if self.use_db:
            # DATABASE MODE
            if portfolio_id is None:
                # Try to get from session state if available
                if STREAMLIT_AVAILABLE and hasattr(st.session_state, 'selected_portfolio_id'):
                    portfolio_id = st.session_state.selected_portfolio_id
                else:
                    raise ValueError("portfolio_id required when using database mode")

            logger.debug(f"Fetching projects from DATABASE (portfolio={portfolio_id}, date={status_date})")

            # Query database with baseline effectivity
            df = self.db_manager.get_projects_for_batch_calculation(
                portfolio_id=portfolio_id,
                status_date=status_date
            )

            # Transform to ensure column names match expected format
            df = self._ensure_column_format(df)

            return df

        else:
            # SESSION STATE MODE (existing behavior)
            logger.debug("Fetching projects from SESSION STATE")

            if not STREAMLIT_AVAILABLE:
                raise RuntimeError("Streamlit not available - cannot access session state")

            # Load from session state
            df = self.session_manager.load_table("dataset")

            return df

    def get_project_for_analysis(
        self,
        project_id: str,
        status_date: Optional[date] = None
    ) -> Optional[Dict[str, Any]]:
        """Get single project data for analysis

        Returns data in format expected by perform_complete_evm_analysis():
        {
            'bac': float,
            'ac': float,
            'plan_start': datetime,
            'plan_finish': datetime,
            'data_date': datetime,
            'annual_inflation_rate': float,
            'curve_type': str,
            'alpha': float,
            'beta': float,
            'manual_pv': Optional[float],
            'use_manual_pv': bool,
            'manual_ev': Optional[float],
            'use_manual_ev': bool
        }

        Args:
            project_id: Project ID
            status_date: Status date (optional - uses latest if None)

        Returns:
            Dictionary ready for perform_complete_evm_analysis(), or None if not found
        """
        if self.use_db:
            # DATABASE MODE
            logger.debug(f"Fetching project from DATABASE (project_id={project_id})")

            try:
                from database.queries import get_project_with_effective_baseline_and_status

                # Get combined data (project + baseline + status)
                data = get_project_with_effective_baseline_and_status(
                    project_id=int(project_id),
                    status_date=status_date
                )

                if not data:
                    return None

                # Transform to expected format
                project = data['project']
                baseline = data['baseline']
                status = data['status']

                return {
                    'bac': float(baseline['budget_at_completion']),
                    'ac': float(status['actual_cost']),
                    'plan_start': self._ensure_datetime(baseline['planned_start_date']),
                    'plan_finish': self._ensure_datetime(baseline['planned_finish_date']),
                    'data_date': self._ensure_datetime(status['status_date']),
                    'manual_pv': status.get('planned_value'),
                    'use_manual_pv': status.get('planned_value') is not None,
                    'manual_ev': status.get('earned_value'),
                    'use_manual_ev': status.get('earned_value') is not None,
                    # These would come from portfolio or project settings (Phase C)
                    'annual_inflation_rate': 0.03,  # Default for now
                    'curve_type': 'linear',         # Default for now
                    'alpha': 2.0,
                    'beta': 2.0
                }

            except Exception as e:
                logger.error(f"Error fetching project from database: {e}")
                return None

        else:
            # SESSION STATE MODE
            logger.debug(f"Fetching project from SESSION STATE (project_id={project_id})")

            if not STREAMLIT_AVAILABLE:
                return None

            return self.session_manager.get_project_record(project_id)

    # ========================================================================
    # BATCH RESULTS (Calculated EVM Metrics)
    # ========================================================================

    def get_batch_results(
        self,
        portfolio_id: Optional[int] = None,
        status_date: Optional[date] = None
    ) -> pd.DataFrame:
        """Get EVM calculation results (either pre-calculated or query + calculate)

        This returns the results of perform_batch_calculation() - either from
        cache (session state) or by querying and calculating on-the-fly.

        Args:
            portfolio_id: Portfolio ID (required in database mode)
            status_date: Status date (optional)

        Returns:
            DataFrame with EVM calculation results
        """
        if self.use_db:
            # DATABASE MODE
            logger.debug("Getting batch results from DATABASE mode")

            # Get portfolio_id and status_date from context or parameters
            if portfolio_id is None:
                if STREAMLIT_AVAILABLE and hasattr(st.session_state, 'current_portfolio_id'):
                    portfolio_id = st.session_state.current_portfolio_id
                else:
                    logger.warning("No portfolio_id provided for get_batch_results")
                    return pd.DataFrame()

            if status_date is None:
                if STREAMLIT_AVAILABLE and hasattr(st.session_state, 'current_status_date'):
                    status_date = st.session_state.current_status_date
                else:
                    logger.warning("No status_date provided for get_batch_results")
                    return pd.DataFrame()

            # Query database for pre-calculated EVM results
            try:
                results_df = self.db_manager.get_evm_results_for_period(portfolio_id, status_date)
                logger.info(f"Loaded {len(results_df)} calculated EVM results from database")
                return results_df
            except Exception as e:
                logger.error(f"Error loading batch results from database: {e}")
                return pd.DataFrame()

        else:
            # SESSION STATE MODE
            logger.debug("Getting batch results from SESSION STATE")

            if not STREAMLIT_AVAILABLE:
                return pd.DataFrame()

            # Return from session state
            if hasattr(st.session_state, 'batch_results') and st.session_state.batch_results is not None:
                return st.session_state.batch_results

            return pd.DataFrame()

    def save_batch_results(
        self,
        results_df: pd.DataFrame,
        portfolio_id: Optional[int] = None
    ):
        """Save EVM calculation results

        Args:
            results_df: DataFrame with calculated results
            portfolio_id: Portfolio ID (for database mode)
        """
        if self.use_db:
            # DATABASE MODE
            logger.debug("Saving batch results to DATABASE mode")

            # For now, don't store in DB (implement in Phase C)
            # Could store in a separate evm_results_cache table

            # Still save to session for compatibility during transition
            if STREAMLIT_AVAILABLE:
                st.session_state.batch_results = results_df

        else:
            # SESSION STATE MODE
            logger.debug("Saving batch results to SESSION STATE")

            if not STREAMLIT_AVAILABLE:
                return

            self.session_manager.save_table_replace(results_df, "batch_results")

    # ========================================================================
    # PORTFOLIO OPERATIONS
    # ========================================================================

    def list_portfolios(self, user_id: Optional[int] = None) -> pd.DataFrame:
        """List portfolios (database mode only)

        Args:
            user_id: Owner user ID filter (optional)

        Returns:
            DataFrame with portfolios
        """
        if self.use_db:
            return self.db_manager.list_portfolios(user_id)
        else:
            # Session mode doesn't have multi-portfolio support
            # Return single implicit portfolio
            return pd.DataFrame([{
                'portfolio_id': 1,
                'portfolio_name': 'Default Portfolio',
                'owner_user_id': 1
            }])

    # ========================================================================
    # DATA LOADING
    # ========================================================================

    def load_csv_data(
        self,
        df: pd.DataFrame,
        portfolio_id: Optional[int] = None,
        status_date: Optional[date] = None,
        column_mapping: Optional[Dict[str, str]] = None
    ) -> Dict[str, Any]:
        """Load CSV data (session state or database)

        Args:
            df: DataFrame with project data
            portfolio_id: Portfolio ID (required for database mode)
            status_date: Status date (required for database mode)
            column_mapping: Column mapping (required for database mode)

        Returns:
            Dictionary with load statistics
        """
        if self.use_db:
            # DATABASE MODE - Load into database tables
            logger.info("Loading CSV to DATABASE")

            if portfolio_id is None or status_date is None or column_mapping is None:
                raise ValueError("portfolio_id, status_date, and column_mapping required for database mode")

            stats = self.db_manager.load_csv_to_db(
                df=df,
                portfolio_id=portfolio_id,
                status_date=status_date,
                column_mapping=column_mapping
            )

            # Also save to session state for compatibility during transition
            if STREAMLIT_AVAILABLE:
                self.session_manager.save_table_replace(df, "dataset")

            return stats

        else:
            # SESSION STATE MODE - Save to session state
            logger.info("Loading CSV to SESSION STATE")

            if not STREAMLIT_AVAILABLE:
                raise RuntimeError("Streamlit not available")

            self.session_manager.save_table_replace(df, "dataset")

            return {
                'projects_loaded': len(df),
                'mode': 'session_state'
            }

    # ========================================================================
    # HELPER METHODS
    # ========================================================================

    def _ensure_column_format(self, df: pd.DataFrame) -> pd.DataFrame:
        """Ensure DataFrame has expected column names

        Maps database column names to expected names for calculations.

        Args:
            df: DataFrame from database query

        Returns:
            DataFrame with standardized column names
        """
        # Database queries already return correct column names
        # (see get_portfolio_snapshot_for_batch_calculation query)
        # Just verify key columns exist

        required_columns = ['Project ID', 'BAC', 'AC', 'Plan Start', 'Plan Finish', 'Data Date']
        missing = [col for col in required_columns if col not in df.columns]

        if missing:
            logger.warning(f"Missing expected columns: {missing}")

        return df

    def _ensure_datetime(self, d: Any) -> datetime:
        """Convert date to datetime if needed

        Args:
            d: Date value (date, datetime, string, etc.)

        Returns:
            datetime object
        """
        if d is None:
            return None

        if isinstance(d, datetime):
            return d

        if isinstance(d, date):
            return datetime.combine(d, datetime.min.time())

        # Try to parse string
        if isinstance(d, str):
            from core.evm_engine import parse_date_any
            return parse_date_any(d)

        return d

    # ========================================================================
    # COMPATIBILITY METHODS (For existing code)
    # ========================================================================

    def load_table(self, table_name: str) -> pd.DataFrame:
        """Load table (compatibility with session state DataManager)

        Args:
            table_name: Table name

        Returns:
            DataFrame
        """
        if table_name == "dataset":
            return self.get_projects_for_analysis()
        elif table_name == "batch_results":
            return self.get_batch_results()
        else:
            # Fall back to session manager
            return self.session_manager.load_table(table_name)

    def save_table_replace(self, df: pd.DataFrame, table_name: str):
        """Save table (compatibility with session state DataManager)

        Args:
            df: DataFrame to save
            table_name: Table name
        """
        if table_name == "dataset":
            self.load_csv_data(df)
        elif table_name == "batch_results":
            self.save_batch_results(df)
        else:
            # Fall back to session manager
            self.session_manager.save_table_replace(df, table_name)


# Global instance for easy access
unified_adapter = UnifiedDataAdapter()

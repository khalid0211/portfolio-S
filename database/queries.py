"""Database query helpers for Portfolio Analysis Suite

This module provides high-level query functions for common database operations.
Implements the baseline effectivity logic and other complex queries.
"""

from typing import Optional, List, Dict, Any, Tuple
import pandas as pd
from datetime import date, datetime
from .db_connection import get_db
import logging

logger = logging.getLogger(__name__)


# ============================================================================
# BASELINE QUERIES (Critical for EVM calculations)
# ============================================================================

def get_effective_baseline(project_id: int, status_date: date) -> Optional[Dict[str, Any]]:
    """Get effective baseline for a project on a given status date

    This implements the critical baseline effectivity logic:
    - baseline_start_date <= status_date
    - AND (baseline_end_date IS NULL OR status_date < baseline_end_date)

    Args:
        project_id: Project ID
        status_date: Date to query baseline for

    Returns:
        Dictionary with baseline data, or None if no effective baseline found:
        {
            'baseline_id': int,
            'project_id': int,
            'baseline_version': int,
            'planned_start_date': date,
            'planned_finish_date': date,
            'budget_at_completion': float,
            'project_status': str,
            'baseline_start_date': date,
            'baseline_end_date': date (or None)
        }
    """
    db = get_db()
    query = """
        SELECT
            baseline_id,
            project_id,
            baseline_version,
            planned_start_date,
            planned_finish_date,
            budget_at_completion,
            project_status,
            baseline_start_date,
            baseline_end_date,
            baseline_reason,
            approved_by,
            approved_date
        FROM project_baseline
        WHERE project_id = ?
          AND baseline_start_date <= ?
          AND (baseline_end_date IS NULL OR ? < baseline_end_date)
        ORDER BY baseline_version DESC
        LIMIT 1
    """

    try:
        result = db.fetch_one(query, (project_id, status_date, status_date))

        if result:
            return {
                'baseline_id': result[0],
                'project_id': result[1],
                'baseline_version': result[2],
                'planned_start_date': result[3],
                'planned_finish_date': result[4],
                'budget_at_completion': float(result[5]) if result[5] is not None else 0.0,
                'project_status': result[6],
                'baseline_start_date': result[7],
                'baseline_end_date': result[8],
                'baseline_reason': result[9],
                'approved_by': result[10],
                'approved_date': result[11]
            }
        return None

    except Exception as e:
        logger.error(f"Error getting effective baseline: {e}")
        return None


def get_all_baselines(project_id: int) -> pd.DataFrame:
    """Get all baselines for a project, ordered by version

    Args:
        project_id: Project ID

    Returns:
        DataFrame with all baselines for the project
    """
    db = get_db()
    query = """
        SELECT * FROM project_baseline
        WHERE project_id = ?
        ORDER BY baseline_version ASC
    """
    try:
        return db.execute(query, (project_id,)).df()
    except Exception as e:
        logger.error(f"Error getting baselines: {e}")
        return pd.DataFrame()


def get_active_baseline(project_id: int) -> Optional[Dict[str, Any]]:
    """Get the currently active baseline (baseline_end_date IS NULL)

    Args:
        project_id: Project ID

    Returns:
        Dictionary with active baseline data, or None if no active baseline
    """
    db = get_db()
    query = """
        SELECT
            baseline_id, project_id, baseline_version,
            planned_start_date, planned_finish_date,
            budget_at_completion, project_status,
            baseline_start_date, baseline_end_date
        FROM project_baseline
        WHERE project_id = ?
          AND baseline_end_date IS NULL
        LIMIT 1
    """

    try:
        result = db.fetch_one(query, (project_id,))

        if result:
            return {
                'baseline_id': result[0],
                'project_id': result[1],
                'baseline_version': result[2],
                'planned_start_date': result[3],
                'planned_finish_date': result[4],
                'budget_at_completion': float(result[5]) if result[5] is not None else 0.0,
                'project_status': result[6],
                'baseline_start_date': result[7],
                'baseline_end_date': result[8]
            }
        return None

    except Exception as e:
        logger.error(f"Error getting active baseline: {e}")
        return None


# ============================================================================
# PROJECT QUERIES
# ============================================================================

def get_projects_by_portfolio(portfolio_id: int, include_inactive: bool = False) -> pd.DataFrame:
    """Get all projects in a portfolio

    Args:
        portfolio_id: Portfolio ID
        include_inactive: If True, includes inactive projects

    Returns:
        DataFrame with project data
    """
    db = get_db()

    if include_inactive:
        query = """
            SELECT * FROM project
            WHERE portfolio_id = ?
            ORDER BY project_name
        """
    else:
        query = """
            SELECT * FROM project
            WHERE portfolio_id = ?
              AND is_active = TRUE
            ORDER BY project_name
        """

    try:
        return db.execute(query, (portfolio_id,)).df()
    except Exception as e:
        logger.error(f"Error getting projects: {e}")
        return pd.DataFrame()


def get_project_by_id(project_id: int) -> Optional[Dict[str, Any]]:
    """Get project by ID

    Args:
        project_id: Project ID

    Returns:
        Dictionary with project data, or None if not found
    """
    db = get_db()
    query = "SELECT * FROM project WHERE project_id = ?"

    try:
        result = db.fetch_one(query, (project_id,))
        if result:
            columns = db.get_table_columns('project')
            return dict(zip(columns, result))
        return None
    except Exception as e:
        logger.error(f"Error getting project: {e}")
        return None


def get_project_by_name(portfolio_id: int, project_name: str) -> Optional[Dict[str, Any]]:
    """Get project by name within a portfolio

    Args:
        portfolio_id: Portfolio ID
        project_name: Project name

    Returns:
        Dictionary with project data, or None if not found
    """
    db = get_db()
    query = """
        SELECT * FROM project
        WHERE portfolio_id = ? AND project_name = ?
    """

    try:
        result = db.fetch_one(query, (portfolio_id, project_name))
        if result:
            columns = db.get_table_columns('project')
            return dict(zip(columns, result))
        return None
    except Exception as e:
        logger.error(f"Error getting project by name: {e}")
        return None


# ============================================================================
# STATUS REPORT QUERIES
# ============================================================================

def get_latest_status_report(project_id: int) -> Optional[Dict[str, Any]]:
    """Get most recent status report for a project

    Args:
        project_id: Project ID

    Returns:
        Dictionary with status report data, or None if no reports found
    """
    db = get_db()
    query = """
        SELECT * FROM project_status_report
        WHERE project_id = ?
        ORDER BY status_date DESC
        LIMIT 1
    """

    try:
        result = db.fetch_one(query, (project_id,))
        if result:
            columns = db.get_table_columns('project_status_report')
            return dict(zip(columns, result))
        return None
    except Exception as e:
        logger.error(f"Error getting latest status report: {e}")
        return None


def get_status_report_on_date(project_id: int, status_date: date) -> Optional[Dict[str, Any]]:
    """Get status report for a specific date

    Args:
        project_id: Project ID
        status_date: Status date

    Returns:
        Dictionary with status report data, or None if not found
    """
    db = get_db()
    query = """
        SELECT * FROM project_status_report
        WHERE project_id = ? AND status_date = ?
    """

    try:
        result = db.fetch_one(query, (project_id, status_date))
        if result:
            columns = db.get_table_columns('project_status_report')
            return dict(zip(columns, result))
        return None
    except Exception as e:
        logger.error(f"Error getting status report: {e}")
        return None


def get_status_reports_by_portfolio_date(portfolio_id: int, status_date: date) -> pd.DataFrame:
    """Get all status reports for a portfolio on a specific date

    Args:
        portfolio_id: Portfolio ID
        status_date: Status date

    Returns:
        DataFrame with status reports for all projects on that date
    """
    db = get_db()
    query = """
        SELECT
            p.project_id,
            p.project_name,
            p.responsible_organization,
            p.project_manager,
            sr.status_date,
            sr.actual_cost,
            sr.planned_value,
            sr.earned_value,
            sr.notes,
            sr.created_at
        FROM project p
        JOIN project_status_report sr ON p.project_id = sr.project_id
        WHERE p.portfolio_id = ?
          AND sr.status_date = ?
          AND p.is_active = TRUE
        ORDER BY p.project_name
    """

    try:
        return db.execute(query, (portfolio_id, status_date)).df()
    except Exception as e:
        logger.error(f"Error getting portfolio status reports: {e}")
        return pd.DataFrame()


def get_all_status_reports(project_id: int) -> pd.DataFrame:
    """Get all status reports for a project (time series)

    Args:
        project_id: Project ID

    Returns:
        DataFrame with all status reports, ordered by date
    """
    db = get_db()
    query = """
        SELECT * FROM project_status_report
        WHERE project_id = ?
        ORDER BY status_date ASC
    """

    try:
        return db.execute(query, (project_id,)).df()
    except Exception as e:
        logger.error(f"Error getting status reports: {e}")
        return pd.DataFrame()


# ============================================================================
# COMBINED QUERIES (Project + Baseline + Status)
# ============================================================================

def get_project_with_effective_baseline_and_status(
    project_id: int,
    status_date: Optional[date] = None
) -> Optional[Dict[str, Any]]:
    """Get project with effective baseline and latest/specific status report

    This is a convenience function that combines:
    - Project master data
    - Effective baseline (based on status_date)
    - Status report (either specific date or latest)

    Args:
        project_id: Project ID
        status_date: Status date (if None, uses latest status report)

    Returns:
        Combined dictionary with all data, or None if project not found
    """
    # Get project
    project = get_project_by_id(project_id)
    if not project:
        return None

    # Get status report
    if status_date:
        status = get_status_report_on_date(project_id, status_date)
    else:
        status = get_latest_status_report(project_id)

    if not status:
        logger.warning(f"No status report found for project {project_id}")
        return None

    # Use status date from report
    report_date = status['status_date']

    # Get effective baseline
    baseline = get_effective_baseline(project_id, report_date)
    if not baseline:
        logger.warning(f"No effective baseline for project {project_id} on {report_date}")
        return None

    # Combine all data
    return {
        'project': project,
        'baseline': baseline,
        'status': status
    }


def get_portfolio_snapshot_for_batch_calculation(
    portfolio_id: int,
    status_date: Optional[date] = None
) -> pd.DataFrame:
    """Get portfolio snapshot ready for batch EVM calculation

    This query combines projects, effective baselines, and status reports
    into the format expected by perform_batch_calculation().

    Args:
        portfolio_id: Portfolio ID
        status_date: Status date (if None, uses latest status for each project)

    Returns:
        DataFrame with columns matching perform_batch_calculation() input format
    """
    db = get_db()

    if status_date:
        # Specific date snapshot
        query = """
            SELECT
                p.project_id AS "Project ID",
                p.project_name AS "Project",
                p.responsible_organization AS "Organization",
                p.project_manager AS "Project Manager",
                b.budget_at_completion AS "BAC",
                sr.actual_cost AS "AC",
                b.planned_start_date AS "Plan Start",
                b.planned_finish_date AS "Plan Finish",
                sr.status_date AS "Data Date",
                sr.planned_value AS "PV",
                sr.earned_value AS "EV"
            FROM project p
            JOIN project_baseline b ON p.project_id = b.project_id
            JOIN project_status_report sr ON p.project_id = sr.project_id
            WHERE p.portfolio_id = ?
              AND p.is_active = TRUE
              AND sr.status_date = ?
              AND (b.baseline_end_date IS NULL OR b.baseline_end_date > sr.status_date)
            ORDER BY p.project_name
        """
        params = (portfolio_id, status_date)
    else:
        # Latest status for each project (more complex - use subquery)
        query = """
            SELECT
                p.project_id AS "Project ID",
                p.project_name AS "Project",
                p.responsible_organization AS "Organization",
                p.project_manager AS "Project Manager",
                b.budget_at_completion AS "BAC",
                sr.actual_cost AS "AC",
                b.planned_start_date AS "Plan Start",
                b.planned_finish_date AS "Plan Finish",
                sr.status_date AS "Data Date",
                sr.planned_value AS "PV",
                sr.earned_value AS "EV"
            FROM project p
            JOIN project_baseline b ON p.project_id = b.project_id
            JOIN project_status_report sr ON p.project_id = sr.project_id
            JOIN (
                SELECT project_id, MAX(status_date) as latest_date
                FROM project_status_report
                GROUP BY project_id
            ) latest ON sr.project_id = latest.project_id AND sr.status_date = latest.latest_date
            WHERE p.portfolio_id = ?
              AND p.is_active = TRUE
              AND (b.baseline_end_date IS NULL OR b.baseline_end_date > sr.status_date)
            ORDER BY p.project_name
        """
        params = (portfolio_id,)

    try:
        return db.execute(query, params).df()
    except Exception as e:
        logger.error(f"Error getting portfolio snapshot: {e}")
        return pd.DataFrame()


# ============================================================================
# PORTFOLIO QUERIES
# ============================================================================

def get_portfolios_by_user(user_id: Optional[int] = None) -> pd.DataFrame:
    """Get portfolios (optionally filtered by owner) with ownership from portfolio_ownership table

    Args:
        user_id: Owner user ID (if None, returns all portfolios)

    Returns:
        DataFrame with portfolio data including owner_user_id from portfolio_ownership
    """
    db = get_db()

    if user_id:
        query = """
            SELECT
                p.portfolio_id,
                p.portfolio_name,
                p.managing_organization,
                p.portfolio_manager,
                p.is_active,
                p.created_at,
                p.description,
                po.owner_user_id,
                po.created_at as owner_assigned_at
            FROM portfolio p
            LEFT JOIN portfolio_ownership po ON p.portfolio_id = po.portfolio_id
            WHERE po.owner_user_id = ? AND p.is_active = TRUE
            ORDER BY p.portfolio_name
        """
        params = (user_id,)
    else:
        query = """
            SELECT
                p.portfolio_id,
                p.portfolio_name,
                p.managing_organization,
                p.portfolio_manager,
                p.is_active,
                p.created_at,
                p.description,
                po.owner_user_id,
                po.created_at as owner_assigned_at
            FROM portfolio p
            LEFT JOIN portfolio_ownership po ON p.portfolio_id = po.portfolio_id
            WHERE p.is_active = TRUE
            ORDER BY p.portfolio_name
        """
        params = None

    try:
        if params:
            return db.execute(query, params).df()
        else:
            return db.execute(query).df()
    except Exception as e:
        logger.error(f"Error getting portfolios: {e}")
        return pd.DataFrame()


# ============================================================================
# UTILITY QUERIES
# ============================================================================

def check_planned_start_lock(project_id: int) -> bool:
    """Check if planned start is locked (any AC > 0 exists)

    Args:
        project_id: Project ID

    Returns:
        True if planned start is locked (AC > 0 recorded), False otherwise
    """
    db = get_db()
    query = """
        SELECT COUNT(*) FROM project_status_report
        WHERE project_id = ? AND actual_cost > 0
    """

    try:
        result = db.fetch_one(query, (project_id,))
        return result[0] > 0 if result else False
    except Exception as e:
        logger.error(f"Error checking planned start lock: {e}")
        return False


def get_database_stats() -> Dict[str, int]:
    """Get database statistics (record counts)

    Returns:
        Dictionary with table record counts
    """
    db = get_db()
    stats = {}

    tables = [
        'app_user', 'portfolio', 'project', 'project_baseline',
        'project_status_report', 'sdg', 'project_sdg',
        'portfolio_factor', 'project_factor_score'
    ]

    for table in tables:
        try:
            result = db.fetch_one(f"SELECT COUNT(*) FROM {table}")
            stats[table] = result[0] if result else 0
        except Exception:
            stats[table] = 0

    return stats

"""Portfolio Context Component

Reusable component for portfolio and period selection across all pages.
Provides a consistent UI for selecting active portfolio and data date.

Usage:
    from utils.portfolio_context import render_portfolio_context

    # Show both portfolio and period selectors
    portfolio_id, status_date = render_portfolio_context()

    # Show only portfolio selector
    portfolio_id, _ = render_portfolio_context(show_period_selector=False)
"""

import streamlit as st
import pandas as pd
from datetime import date
from typing import Optional, Tuple, List
import logging

from config.constants import USE_DATABASE
from database.db_connection import get_db

logger = logging.getLogger(__name__)


def get_current_user_email() -> Optional[str]:
    """Get email of currently logged-in user

    Checks multiple session state keys because different auth flows
    store the email in different locations:
    - st.session_state.user (Streamlit native OAuth)
    - st.session_state.user_email (set by utils/auth.py)
    - st.session_state.user_info['email'] (set by utils/auth.py)

    Returns:
        User email from session state, or None if not logged in
    """
    # Check Streamlit native OAuth first
    if hasattr(st.session_state, 'user') and st.session_state.user:
        email = st.session_state.user.get('email')
        if email:
            return email

    # Check user_email set by auth.py
    email = st.session_state.get('user_email')
    if email:
        return email

    # Check user_info dict set by auth.py
    user_info = st.session_state.get('user_info')
    if user_info and isinstance(user_info, dict):
        email = user_info.get('email')
        if email:
            return email

    return None


def get_user_portfolios(user_email: Optional[str] = None) -> pd.DataFrame:
    """Get portfolios accessible to user

    Args:
        user_email: User email (if None, gets from current session)

    Returns:
        DataFrame with columns: portfolio_id, portfolio_name, managing_organization,
                                portfolio_manager, owner_email, project_count
    """
    if not USE_DATABASE:
        # Session state mode - return single default portfolio
        return pd.DataFrame([{
            'portfolio_id': 1,
            'portfolio_name': 'Default Portfolio',
            'managing_organization': '',
            'portfolio_manager': '',
            'owner_email': '',
            'project_count': 0
        }])

    if user_email is None:
        user_email = get_current_user_email()

    try:
        db = get_db()

        # Check if user is super admin (sees all portfolios)
        is_superadmin = False
        if user_email:
            try:
                from utils.firestore_client import get_firestore_client
                from utils.user_manager import is_super_admin
                fs = get_firestore_client()
                if fs:
                    is_superadmin = is_super_admin(fs, user_email)
            except Exception:
                pass

        if not user_email:
            # No user email available - return empty (don't show all portfolios)
            logger.warning("No user email available, returning empty portfolio list")
            return pd.DataFrame()

        if is_superadmin:
            # Super admin: show all active portfolios
            query = """
                SELECT
                    p.portfolio_id,
                    p.portfolio_name,
                    p.managing_organization,
                    p.portfolio_manager,
                    p.description,
                    u.email as owner_email,
                    COUNT(DISTINCT pr.project_id) as project_count
                FROM portfolio p
                LEFT JOIN portfolio_ownership po ON p.portfolio_id = po.portfolio_id AND po.is_active = TRUE
                LEFT JOIN app_user u ON po.owner_user_id = u.user_id
                LEFT JOIN project pr ON p.portfolio_id = pr.portfolio_id AND pr.is_active = TRUE
                WHERE p.is_active = TRUE
                GROUP BY p.portfolio_id, p.portfolio_name, p.managing_organization,
                         p.portfolio_manager, p.description, u.email
                ORDER BY p.portfolio_name
            """
            result = db.execute(query)
            df = result.df()
        else:
            # Regular user: show owned + shared portfolios
            query = """
                SELECT
                    p.portfolio_id,
                    p.portfolio_name,
                    p.managing_organization,
                    p.portfolio_manager,
                    p.description,
                    u_owner.email as owner_email,
                    COUNT(DISTINCT pr.project_id) as project_count
                FROM portfolio p
                LEFT JOIN portfolio_ownership po ON p.portfolio_id = po.portfolio_id AND po.is_active = TRUE
                LEFT JOIN app_user u_owner ON po.owner_user_id = u_owner.user_id
                LEFT JOIN project pr ON p.portfolio_id = pr.portfolio_id AND pr.is_active = TRUE
                WHERE p.is_active = TRUE
                  AND (
                      u_owner.email = ?
                      OR p.portfolio_id IN (
                          SELECT pa.portfolio_id
                          FROM portfolio_access pa
                          JOIN app_user u2 ON pa.user_id = u2.user_id
                          WHERE u2.email = ? AND pa.is_active = TRUE
                      )
                  )
                GROUP BY p.portfolio_id, p.portfolio_name, p.managing_organization,
                         p.portfolio_manager, p.description, u_owner.email
                ORDER BY p.portfolio_name
            """
            result = db.execute(query, (user_email, user_email))
            df = result.df()

        return df

    except Exception as e:
        logger.error(f"Error fetching user portfolios: {e}")
        return pd.DataFrame()


def get_available_periods(portfolio_id: int) -> List[date]:
    """Get all data dates (periods) available for a portfolio

    Args:
        portfolio_id: Portfolio ID

    Returns:
        List of dates, sorted newest first
    """
    if not USE_DATABASE:
        # Session state mode - check if we have batch results
        if hasattr(st.session_state, 'batch_results') and st.session_state.batch_results is not None:
            # Try to extract date from session state
            if hasattr(st.session_state, 'current_status_date'):
                return [st.session_state.current_status_date]
        return []

    try:
        db = get_db()

        query = """
            SELECT DISTINCT status_date
            FROM project_status_report
            WHERE portfolio_id = ?
            ORDER BY status_date DESC
        """

        result = db.execute(query, (portfolio_id,))
        dates = [row[0] for row in result.fetchall()]

        return dates

    except Exception as e:
        logger.error(f"Error fetching available periods: {e}")
        return []


def render_portfolio_selector() -> Optional[int]:
    """Render portfolio selection dropdown

    Returns:
        Selected portfolio_id, or None if no selection
    """
    # Get user's portfolios
    portfolios_df = get_user_portfolios()

    if portfolios_df.empty:
        st.warning("⚠️ No portfolios found. Please create a portfolio first in Portfolio Management.")
        return None

    # Create portfolio options
    portfolio_options = {}
    for _, row in portfolios_df.iterrows():
        label = f"{row['portfolio_name']}"
        if row.get('project_count', 0) > 0:
            label += f" ({int(row['project_count'])} projects)"
        portfolio_options[label] = row['portfolio_id']

    # Get current selection from session state
    current_portfolio_id = st.session_state.get('current_portfolio_id')

    # Find index of current selection
    default_index = 0
    if current_portfolio_id:
        portfolio_ids = list(portfolio_options.values())
        if current_portfolio_id in portfolio_ids:
            default_index = portfolio_ids.index(current_portfolio_id)

    # Render dropdown
    selected_label = st.selectbox(
        "📂 Select Portfolio",
        options=list(portfolio_options.keys()),
        index=default_index,
        key="portfolio_selector"
    )

    selected_portfolio_id = portfolio_options[selected_label]

    # Update session state
    st.session_state.current_portfolio_id = selected_portfolio_id

    return selected_portfolio_id


def render_period_selector(portfolio_id: int) -> Optional[date]:
    """Render period (data date) selection dropdown

    Args:
        portfolio_id: Portfolio ID to get periods for

    Returns:
        Selected status_date, or None if no periods available
    """
    if not portfolio_id:
        return None

    # Get available periods
    periods = get_available_periods(portfolio_id)

    if not periods:
        st.info("ℹ️ No data periods available for this portfolio. Upload data in File Management to create the first period.")
        return None

    # Format dates for display
    period_options = {}
    for period_date in periods:
        if isinstance(period_date, str):
            period_date = pd.to_datetime(period_date).date()
        label = period_date.strftime('%d-%b-%Y')  # e.g., "01-Sep-2025"
        period_options[label] = period_date

    # Get current selection from session state
    current_status_date = st.session_state.get('current_status_date')

    # Find index of current selection
    default_index = 0
    if current_status_date:
        period_dates = list(period_options.values())
        if current_status_date in period_dates:
            default_index = period_dates.index(current_status_date)

    # Render dropdown
    selected_label = st.selectbox(
        f"📅 Select Period ({len(periods)} available)",
        options=list(period_options.keys()),
        index=default_index,
        key="period_selector"
    )

    selected_date = period_options[selected_label]

    # Update session state
    st.session_state.current_status_date = selected_date

    return selected_date


def render_portfolio_context(
    show_period_selector: bool = True,
    show_header: bool = True,
    show_progress_filter: bool = False
) -> Tuple[Optional[int], Optional[date]]:
    """Render portfolio context component (portfolio + period selectors)

    This is the main component to use in pages. It shows:
    1. Portfolio dropdown (filtered by user)
    2. Period dropdown (if show_period_selector=True)
    3. Progress filter toggle (if show_progress_filter=True)
    4. Summary information

    Args:
        show_period_selector: If True, show period dropdown. If False, only portfolio.
        show_header: If True, show section header
        show_progress_filter: If True, show toggle to filter projects with AC > 0

    Returns:
        Tuple of (portfolio_id, status_date)
        - portfolio_id: Selected portfolio ID, or None
        - status_date: Selected period date, or None

    Example:
        >>> portfolio_id, status_date = render_portfolio_context()
        >>> if not portfolio_id:
        >>>     st.warning("Please select a portfolio")
        >>>     st.stop()
    """
    if show_header:
        st.markdown("### 🎯 Portfolio Context")

    col1, col2 = st.columns(2)

    with col1:
        portfolio_id = render_portfolio_selector()

    status_date = None
    if portfolio_id and show_period_selector:
        with col2:
            status_date = render_period_selector(portfolio_id)

    # Show progress filter toggle if enabled
    if show_progress_filter and portfolio_id and status_date:
        filter_col1, filter_col2 = st.columns([3, 1])
        with filter_col2:
            show_only_started = st.toggle(
                "Show only started projects",
                value=st.session_state.get('filter_started_projects', False),
                help="When enabled, only shows projects where Actual Cost (AC) > 0",
                key="filter_started_projects_toggle"
            )
            st.session_state['filter_started_projects'] = show_only_started

    # Compute and store access level in session state
    if portfolio_id:
        try:
            from utils.portfolio_access import get_portfolio_access_level
            user_email = get_current_user_email()
            access_level = get_portfolio_access_level(portfolio_id, user_email)
            st.session_state['portfolio_access_level'] = access_level
        except Exception as e:
            logger.warning(f"Could not determine portfolio access level: {e}")
            st.session_state['portfolio_access_level'] = None

    # Show summary info
    if portfolio_id:
        portfolios_df = get_user_portfolios()
        portfolio_info = portfolios_df[portfolios_df['portfolio_id'] == portfolio_id]

        if not portfolio_info.empty:
            portfolio_name = portfolio_info.iloc[0]['portfolio_name']
            project_count = int(portfolio_info.iloc[0].get('project_count', 0))

            if show_period_selector and status_date:
                st.caption(f"📊 Viewing: **{portfolio_name}** | Period: **{status_date.strftime('%d-%b-%Y')}** | {project_count} projects")
            else:
                st.caption(f"📊 Active Portfolio: **{portfolio_name}** | {project_count} projects")

    return portfolio_id, status_date


def apply_progress_filter(df: pd.DataFrame, ac_column: str = 'AC') -> Tuple[pd.DataFrame, str]:
    """Apply progress filter to DataFrame if enabled in session state

    Args:
        df: DataFrame to filter
        ac_column: Name of the AC column

    Returns:
        Tuple of (filtered_df, filter_message)
        - filtered_df: Filtered DataFrame (or original if filter not enabled)
        - filter_message: Message describing filter status (empty if not filtered)
    """
    if not st.session_state.get('filter_started_projects', False):
        return df, ""

    if df.empty or ac_column not in df.columns:
        return df, ""

    original_count = len(df)
    filtered_df = df[df[ac_column].fillna(0).astype(float) > 0].copy()
    filtered_count = len(filtered_df)

    if filtered_count < original_count:
        message = f"📊 Showing {filtered_count} of {original_count} projects (filtered by AC > 0)"
    else:
        message = f"📊 All {original_count} projects have AC > 0"

    return filtered_df, message


def get_portfolio_summary(portfolio_id: int) -> dict:
    """Get summary statistics for a portfolio

    Args:
        portfolio_id: Portfolio ID

    Returns:
        Dictionary with summary stats:
        - total_projects: Total number of projects
        - total_periods: Number of data periods
        - latest_period: Most recent period date
        - earliest_period: Oldest period date
    """
    if not USE_DATABASE:
        return {
            'total_projects': 0,
            'total_periods': 0,
            'latest_period': None,
            'earliest_period': None
        }

    try:
        db = get_db()

        # Get project count
        project_query = """
            SELECT COUNT(*) as count
            FROM project
            WHERE portfolio_id = ? AND is_active = TRUE
        """
        project_result = db.execute(project_query, (portfolio_id,)).fetchone()
        total_projects = project_result[0] if project_result else 0

        # Get period info
        period_query = """
            SELECT
                COUNT(DISTINCT status_date) as period_count,
                MIN(status_date) as earliest,
                MAX(status_date) as latest
            FROM project_status_report
            WHERE portfolio_id = ?
        """
        period_result = db.execute(period_query, (portfolio_id,)).fetchone()

        if period_result:
            return {
                'total_projects': total_projects,
                'total_periods': period_result[0] or 0,
                'earliest_period': period_result[1],
                'latest_period': period_result[2]
            }

        return {
            'total_projects': total_projects,
            'total_periods': 0,
            'latest_period': None,
            'earliest_period': None
        }

    except Exception as e:
        logger.error(f"Error fetching portfolio summary: {e}")
        return {
            'total_projects': 0,
            'total_periods': 0,
            'latest_period': None,
            'earliest_period': None
        }

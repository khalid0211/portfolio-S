"""
SDG Management - Portfolio Management Suite
Assign UN Sustainable Development Goals to projects and track SDG contributions
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from typing import List
import tempfile
import os
from datetime import datetime
from utils.auth import check_authentication, require_page_access
from config.constants import USE_DATABASE
from services.db_data_service import DatabaseDataManager
from database.db_connection import get_db
from services.sdg_csv_service import (
    export_sdg_to_csv,
    import_sdg_from_csv,
    get_sdg_template_csv,
    validate_sdg_csv
)

# Page configuration
st.set_page_config(
    page_title="SDG Management - Portfolio Suite",
    page_icon="🌍",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# Check authentication and page access
if not check_authentication():
    st.stop()

require_page_access('sdg_management', 'SDG Management')

# Custom CSS
st.markdown("""
<style>
    .main-header {
        text-align: center;
        color: #2c3e50;
        font-size: 2.2rem;
        font-weight: 600;
        margin-bottom: 1rem;
        border-bottom: 3px solid #27ae60;
        padding-bottom: 0.5rem;
    }
    .sdg-card {
        background: rgba(255, 255, 255, 0.95);
        padding: 1rem;
        border-radius: 10px;
        margin: 0.5rem 0;
        border-left: 5px solid #27ae60;
        box-shadow: 0 2px 4px rgba(0, 0, 0, 0.1);
    }
    .sdg-card.assigned {
        background: rgba(39, 174, 96, 0.1);
        border-left-color: #27ae60;
    }
    .sdg-card.unassigned {
        background: rgba(255, 255, 255, 0.95);
        border-left-color: #95a5a6;
    }
    .sdg-badge {
        display: inline-block;
        padding: 0.25rem 0.5rem;
        border-radius: 15px;
        font-size: 0.85rem;
        font-weight: 600;
        margin-right: 0.5rem;
    }
    .badge-assigned {
        background: #27ae60;
        color: white;
    }
    .badge-unassigned {
        background: #95a5a6;
        color: white;
    }
    .stat-box {
        background: linear-gradient(135deg, #27ae60 0%, #229954 100%);
        color: white;
        padding: 1.5rem;
        border-radius: 10px;
        text-align: center;
        margin: 0.5rem;
    }
    .stat-number {
        font-size: 2.5rem;
        font-weight: 700;
        margin: 0;
    }
    .stat-label {
        font-size: 0.9rem;
        opacity: 0.9;
        margin-top: 0.5rem;
    }
</style>
""", unsafe_allow_html=True)

# Main header
st.markdown('<h1 class="main-header">🌍 SDG Management</h1>', unsafe_allow_html=True)
st.markdown("Assign UN Sustainable Development Goals to projects and track portfolio SDG impact")

# Check database mode
if not USE_DATABASE:
    st.warning("⚠️ Database mode is not enabled")
    st.info("""
    **Database Mode Required:**
    SDG management requires database mode to be enabled.
    Set `USE_DATABASE = True` in `config/constants.py`
    """)
    st.stop()

# Initialize database connection with user-specific config
user_email = st.session_state.get('user_email')
db = get_db(user_email=user_email)

# Initialize database manager (uses the singleton db instance)
db_manager = DatabaseDataManager()

# SDG colors (official UN colors for each SDG)
SDG_COLORS = {
    1: "#E5243B", 2: "#DDA63A", 3: "#4C9F38", 4: "#C5192D",
    5: "#FF3A21", 6: "#26BDE2", 7: "#FCC30B", 8: "#A21942",
    9: "#FD6925", 10: "#DD1367", 11: "#FD9D24", 12: "#BF8B2E",
    13: "#3F7E44", 14: "#0A97D9", 15: "#56C02B", 16: "#00689D",
    17: "#19486A"
}


def select_portfolio():
    """UI for selecting portfolio"""
    portfolios_df = db.execute("""
        SELECT portfolio_id, portfolio_name
        FROM portfolio
        WHERE is_active = TRUE
        ORDER BY portfolio_name
    """).df()

    if portfolios_df.empty:
        st.warning("⚠️ No portfolios found. Create a portfolio first in Portfolio Management.")
        return None

    portfolio_options = {
        row['portfolio_name']: row['portfolio_id']
        for _, row in portfolios_df.iterrows()
    }

    selected_portfolio_name = st.selectbox("Select Portfolio", options=list(portfolio_options.keys()))
    return portfolio_options[selected_portfolio_name]


def display_sdg_overview(portfolio_id):
    """Display SDG overview with statistics"""
    st.markdown("### 📊 SDG Overview")

    # Get SDG statistics for portfolio
    sdg_stats = db_manager.get_sdg_statistics(portfolio_id)

    if sdg_stats.empty:
        st.info("No SDG data available")
        return

    # Display summary statistics
    total_projects = db.execute("""
        SELECT COUNT(DISTINCT project_id)
        FROM project
        WHERE portfolio_id = ? AND is_active = TRUE
    """, (portfolio_id,)).fetchone()[0]

    projects_with_sdgs = db.execute("""
        SELECT COUNT(DISTINCT p.project_id)
        FROM project p
        JOIN project_sdg ps ON p.project_id = ps.project_id
        WHERE p.portfolio_id = ? AND p.is_active = TRUE
    """, (portfolio_id,)).fetchone()[0]

    total_assignments = sdg_stats['project_count'].sum()

    col1, col2, col3 = st.columns(3)

    with col1:
        st.markdown(f"""
        <div class="stat-box">
            <div class="stat-number">{total_projects}</div>
            <div class="stat-label">Total Projects</div>
        </div>
        """, unsafe_allow_html=True)

    with col2:
        st.markdown(f"""
        <div class="stat-box">
            <div class="stat-number">{projects_with_sdgs}</div>
            <div class="stat-label">Projects with SDGs</div>
        </div>
        """, unsafe_allow_html=True)

    with col3:
        coverage_pct = (projects_with_sdgs / total_projects * 100) if total_projects > 0 else 0
        st.markdown(f"""
        <div class="stat-box">
            <div class="stat-number">{coverage_pct:.1f}%</div>
            <div class="stat-label">SDG Coverage</div>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("---")

    # Display SDG distribution chart in expander with department filter
    with st.expander("📈 Projects per SDG Chart", expanded=True):
        col_metric, col_filter, col_dept = st.columns(3)

        with col_metric:
            # Toggle between project count and budget
            metric_type = st.radio(
                "Show by:",
                ["Number of Projects", "Total Budget"],
                horizontal=True,
                key="sdg_metric_toggle"
            )

        with col_filter:
            # Toggle between all SDGs or only those with values
            sdg_filter = st.radio(
                "Display:",
                ["Only with Values", "All SDGs"],
                horizontal=True,
                key="sdg_filter_toggle"
            )

        with col_dept:
            # Department filter
            dept_query = """
                SELECT DISTINCT responsible_organization
                FROM project
                WHERE portfolio_id = ? AND is_active = TRUE
                ORDER BY responsible_organization
            """
            dept_df = db.execute(dept_query, (portfolio_id,)).df()

            if not dept_df.empty:
                # Fill null departments
                dept_df['responsible_organization'] = dept_df['responsible_organization'].fillna('Unassigned')
                departments = ['All Departments'] + dept_df['responsible_organization'].tolist()

                selected_dept = st.selectbox(
                    "Filter by department:",
                    options=departments,
                    key="sdg_dept_filter"
                )
            else:
                selected_dept = 'All Departments'

        # Get SDG statistics with department filter
        if metric_type == "Total Budget":
            # Query for budget allocation (only count projects WITH weight_percent set)
            if selected_dept != 'All Departments':
                if selected_dept == 'Unassigned':
                    # Handle NULL department case
                    budget_query = """
                        SELECT
                            s.sdg_id,
                            s.sdg_name,
                            COUNT(DISTINCT p.project_id) as project_count,
                            SUM(CASE
                                WHEN ps.weight_percent IS NOT NULL
                                THEN ps.weight_percent / 100.0 * COALESCE(p.current_budget, 0)
                                ELSE 0
                            END) as total_budget
                        FROM sdg s
                        LEFT JOIN project_sdg ps ON s.sdg_id = ps.sdg_id
                        LEFT JOIN project p ON ps.project_id = p.project_id
                            AND p.portfolio_id = ?
                            AND p.is_active = TRUE
                            AND p.responsible_organization IS NULL
                        GROUP BY s.sdg_id, s.sdg_name
                        ORDER BY s.sdg_id
                    """
                    filtered_sdg_stats = db.execute(budget_query, (portfolio_id,)).df()
                else:
                    # Handle specific department case
                    budget_query = """
                        SELECT
                            s.sdg_id,
                            s.sdg_name,
                            COUNT(DISTINCT p.project_id) as project_count,
                            SUM(CASE
                                WHEN ps.weight_percent IS NOT NULL
                                THEN ps.weight_percent / 100.0 * COALESCE(p.current_budget, 0)
                                ELSE 0
                            END) as total_budget
                        FROM sdg s
                        LEFT JOIN project_sdg ps ON s.sdg_id = ps.sdg_id
                        LEFT JOIN project p ON ps.project_id = p.project_id
                            AND p.portfolio_id = ?
                            AND p.is_active = TRUE
                            AND p.responsible_organization = ?
                        GROUP BY s.sdg_id, s.sdg_name
                        ORDER BY s.sdg_id
                    """
                    filtered_sdg_stats = db.execute(budget_query, (portfolio_id, selected_dept)).df()
                st.info(f"📍 Showing data for: **{selected_dept}**")
            else:
                budget_query = """
                    SELECT
                        s.sdg_id,
                        s.sdg_name,
                        COUNT(DISTINCT p.project_id) as project_count,
                        SUM(CASE
                            WHEN ps.weight_percent IS NOT NULL
                            THEN ps.weight_percent / 100.0 * COALESCE(p.current_budget, 0)
                            ELSE 0
                        END) as total_budget
                    FROM sdg s
                    LEFT JOIN project_sdg ps ON s.sdg_id = ps.sdg_id
                    LEFT JOIN project p ON ps.project_id = p.project_id
                        AND p.portfolio_id = ?
                        AND p.is_active = TRUE
                    GROUP BY s.sdg_id, s.sdg_name
                    ORDER BY s.sdg_id
                """
                filtered_sdg_stats = db.execute(budget_query, (portfolio_id,)).df()
        else:
            # Query for project count - always get all 17 SDGs
            if selected_dept != 'All Departments':
                if selected_dept == 'Unassigned':
                    # Handle NULL department case
                    dept_sdg_query = """
                        SELECT
                            s.sdg_id,
                            s.sdg_name,
                            COUNT(DISTINCT p.project_id) as project_count
                        FROM sdg s
                        LEFT JOIN project_sdg ps ON s.sdg_id = ps.sdg_id
                        LEFT JOIN project p ON ps.project_id = p.project_id
                            AND p.portfolio_id = ?
                            AND p.is_active = TRUE
                            AND p.responsible_organization IS NULL
                        GROUP BY s.sdg_id, s.sdg_name
                        ORDER BY s.sdg_id
                    """
                    filtered_sdg_stats = db.execute(dept_sdg_query, (portfolio_id,)).df()
                else:
                    # Handle specific department case
                    dept_sdg_query = """
                        SELECT
                            s.sdg_id,
                            s.sdg_name,
                            COUNT(DISTINCT p.project_id) as project_count
                        FROM sdg s
                        LEFT JOIN project_sdg ps ON s.sdg_id = ps.sdg_id
                        LEFT JOIN project p ON ps.project_id = p.project_id
                            AND p.portfolio_id = ?
                            AND p.is_active = TRUE
                            AND p.responsible_organization = ?
                        GROUP BY s.sdg_id, s.sdg_name
                        ORDER BY s.sdg_id
                    """
                    filtered_sdg_stats = db.execute(dept_sdg_query, (portfolio_id, selected_dept)).df()
                st.info(f"📍 Showing data for: **{selected_dept}**")
            else:
                # Get all 17 SDGs with counts for "All Departments"
                all_sdg_query = """
                    SELECT
                        s.sdg_id,
                        s.sdg_name,
                        COUNT(DISTINCT p.project_id) as project_count
                    FROM sdg s
                    LEFT JOIN project_sdg ps ON s.sdg_id = ps.sdg_id
                    LEFT JOIN project p ON ps.project_id = p.project_id
                        AND p.portfolio_id = ?
                        AND p.is_active = TRUE
                    GROUP BY s.sdg_id, s.sdg_name
                    ORDER BY s.sdg_id
                """
                filtered_sdg_stats = db.execute(all_sdg_query, (portfolio_id,)).df()

        # Filter SDGs based on toggle selection
        if sdg_filter == "Only with Values":
            if metric_type == "Total Budget":
                # Filter based on budget when showing budget
                active_sdgs = filtered_sdg_stats[filtered_sdg_stats['total_budget'] > 0].copy()
            else:
                # Filter based on project count when showing projects
                active_sdgs = filtered_sdg_stats[filtered_sdg_stats['project_count'] > 0].copy()
        else:
            # Show all SDGs
            active_sdgs = filtered_sdg_stats.copy()

        if not active_sdgs.empty:
            # Create labels in format "SDG-X Name"
            active_sdgs['sdg_label'] = active_sdgs.apply(
                lambda row: f"SDG-{row['sdg_id']} {row['sdg_name']}",
                axis=1
            )

            # Create horizontal bar chart
            active_sdgs['color'] = active_sdgs['sdg_id'].map(SDG_COLORS)

            fig = go.Figure()

            # Determine which metric to display
            if metric_type == "Total Budget":
                x_values = active_sdgs['total_budget']
                text_values = [f"${v:,.0f}" for v in active_sdgs['total_budget']]
                hover_template = '<b>%{y}</b><br>Total Budget: $%{x:,.0f}<extra></extra>'
                x_axis_title = "Total Budget ($)"
            else:
                x_values = active_sdgs['project_count']
                text_values = active_sdgs['project_count']
                hover_template = '<b>%{y}</b><br>Projects: %{x}<extra></extra>'
                x_axis_title = "Number of Projects"

            fig.add_trace(go.Bar(
                y=active_sdgs['sdg_label'],
                x=x_values,
                orientation='h',
                marker=dict(
                    color=active_sdgs['color'],
                    line=dict(color='white', width=2)
                ),
                text=text_values,
                textposition='outside',
                hovertemplate=hover_template
            ))

            fig.update_layout(
                xaxis_title=x_axis_title,
                yaxis_title="",
                height=max(400, len(active_sdgs) * 35),
                showlegend=False,
                yaxis=dict(autorange="reversed")
            )

            st.plotly_chart(fig, width='stretch')

            # Add info about budget calculation if showing budget
            if metric_type == "Total Budget":
                st.info("""
                ℹ️ **Budget Calculation Note:** Budget is only shown for SDGs where projects have explicit percentage allocations set.
                Projects with SDG assignments but no percentage allocation (NULL weight) are not included in budget totals.
                Use the Manage SDG tab to set percentage allocations for projects.
                """)
        else:
            st.info("No SDGs assigned yet. Assign SDGs to projects below.")

    # Display SDG distribution table in expander with department filter
    with st.expander("📊 Projects per SDG Table", expanded=False):
        col_metric_tbl, col_filter_tbl, col_dept_tbl = st.columns(3)

        with col_metric_tbl:
            # Toggle between project count and budget
            metric_type_tbl = st.radio(
                "Show by:",
                ["Number of Projects", "Total Budget"],
                horizontal=True,
                key="sdg_metric_toggle_tbl"
            )

        with col_filter_tbl:
            # Toggle between all SDGs or only those with values
            sdg_filter_tbl = st.radio(
                "Display:",
                ["Only with Values", "All SDGs"],
                horizontal=True,
                key="sdg_filter_toggle_tbl"
            )

        with col_dept_tbl:
            # Department filter
            dept_query_tbl = """
                SELECT DISTINCT responsible_organization
                FROM project
                WHERE portfolio_id = ? AND is_active = TRUE
                ORDER BY responsible_organization
            """
            dept_df_tbl = db.execute(dept_query_tbl, (portfolio_id,)).df()

            if not dept_df_tbl.empty:
                # Fill null departments
                dept_df_tbl['responsible_organization'] = dept_df_tbl['responsible_organization'].fillna('Unassigned')
                departments_tbl = ['All Departments'] + dept_df_tbl['responsible_organization'].tolist()

                selected_dept_tbl = st.selectbox(
                    "Filter by department:",
                    options=departments_tbl,
                    key="sdg_dept_filter_tbl"
                )
            else:
                selected_dept_tbl = 'All Departments'

        # Get SDG statistics with department filter
        if metric_type_tbl == "Total Budget":
            # Query for budget allocation
            if selected_dept_tbl != 'All Departments':
                if selected_dept_tbl == 'Unassigned':
                    budget_query_tbl = """
                        SELECT
                            s.sdg_id,
                            s.sdg_name,
                            COUNT(DISTINCT p.project_id) as project_count,
                            SUM(CASE
                                WHEN ps.weight_percent IS NOT NULL
                                THEN ps.weight_percent / 100.0 * COALESCE(p.current_budget, 0)
                                ELSE 0
                            END) as total_budget
                        FROM sdg s
                        LEFT JOIN project_sdg ps ON s.sdg_id = ps.sdg_id
                        LEFT JOIN project p ON ps.project_id = p.project_id
                            AND p.portfolio_id = ?
                            AND p.is_active = TRUE
                            AND p.responsible_organization IS NULL
                        GROUP BY s.sdg_id, s.sdg_name
                        ORDER BY s.sdg_id
                    """
                    filtered_sdg_stats_tbl = db.execute(budget_query_tbl, (portfolio_id,)).df()
                else:
                    budget_query_tbl = """
                        SELECT
                            s.sdg_id,
                            s.sdg_name,
                            COUNT(DISTINCT p.project_id) as project_count,
                            SUM(CASE
                                WHEN ps.weight_percent IS NOT NULL
                                THEN ps.weight_percent / 100.0 * COALESCE(p.current_budget, 0)
                                ELSE 0
                            END) as total_budget
                        FROM sdg s
                        LEFT JOIN project_sdg ps ON s.sdg_id = ps.sdg_id
                        LEFT JOIN project p ON ps.project_id = p.project_id
                            AND p.portfolio_id = ?
                            AND p.is_active = TRUE
                            AND p.responsible_organization = ?
                        GROUP BY s.sdg_id, s.sdg_name
                        ORDER BY s.sdg_id
                    """
                    filtered_sdg_stats_tbl = db.execute(budget_query_tbl, (portfolio_id, selected_dept_tbl)).df()
                st.info(f"📍 Showing data for: **{selected_dept_tbl}**")
            else:
                budget_query_tbl = """
                    SELECT
                        s.sdg_id,
                        s.sdg_name,
                        COUNT(DISTINCT p.project_id) as project_count,
                        SUM(CASE
                            WHEN ps.weight_percent IS NOT NULL
                            THEN ps.weight_percent / 100.0 * COALESCE(p.current_budget, 0)
                            ELSE 0
                        END) as total_budget
                    FROM sdg s
                    LEFT JOIN project_sdg ps ON s.sdg_id = ps.sdg_id
                    LEFT JOIN project p ON ps.project_id = p.project_id
                        AND p.portfolio_id = ?
                        AND p.is_active = TRUE
                    GROUP BY s.sdg_id, s.sdg_name
                    ORDER BY s.sdg_id
                """
                filtered_sdg_stats_tbl = db.execute(budget_query_tbl, (portfolio_id,)).df()
        else:
            # Query for project count
            if selected_dept_tbl != 'All Departments':
                if selected_dept_tbl == 'Unassigned':
                    dept_sdg_query_tbl = """
                        SELECT
                            s.sdg_id,
                            s.sdg_name,
                            COUNT(DISTINCT p.project_id) as project_count
                        FROM sdg s
                        LEFT JOIN project_sdg ps ON s.sdg_id = ps.sdg_id
                        LEFT JOIN project p ON ps.project_id = p.project_id
                            AND p.portfolio_id = ?
                            AND p.is_active = TRUE
                            AND p.responsible_organization IS NULL
                        GROUP BY s.sdg_id, s.sdg_name
                        ORDER BY s.sdg_id
                    """
                    filtered_sdg_stats_tbl = db.execute(dept_sdg_query_tbl, (portfolio_id,)).df()
                else:
                    dept_sdg_query_tbl = """
                        SELECT
                            s.sdg_id,
                            s.sdg_name,
                            COUNT(DISTINCT p.project_id) as project_count
                        FROM sdg s
                        LEFT JOIN project_sdg ps ON s.sdg_id = ps.sdg_id
                        LEFT JOIN project p ON ps.project_id = p.project_id
                            AND p.portfolio_id = ?
                            AND p.is_active = TRUE
                            AND p.responsible_organization = ?
                        GROUP BY s.sdg_id, s.sdg_name
                        ORDER BY s.sdg_id
                    """
                    filtered_sdg_stats_tbl = db.execute(dept_sdg_query_tbl, (portfolio_id, selected_dept_tbl)).df()
                st.info(f"📍 Showing data for: **{selected_dept_tbl}**")
            else:
                all_sdg_query_tbl = """
                    SELECT
                        s.sdg_id,
                        s.sdg_name,
                        COUNT(DISTINCT p.project_id) as project_count
                    FROM sdg s
                    LEFT JOIN project_sdg ps ON s.sdg_id = ps.sdg_id
                    LEFT JOIN project p ON ps.project_id = p.project_id
                        AND p.portfolio_id = ?
                        AND p.is_active = TRUE
                    GROUP BY s.sdg_id, s.sdg_name
                    ORDER BY s.sdg_id
                """
                filtered_sdg_stats_tbl = db.execute(all_sdg_query_tbl, (portfolio_id,)).df()

        # Filter SDGs based on toggle selection
        if sdg_filter_tbl == "Only with Values":
            if metric_type_tbl == "Total Budget":
                active_sdgs_tbl = filtered_sdg_stats_tbl[filtered_sdg_stats_tbl['total_budget'] > 0].copy()
            else:
                active_sdgs_tbl = filtered_sdg_stats_tbl[filtered_sdg_stats_tbl['project_count'] > 0].copy()
        else:
            active_sdgs_tbl = filtered_sdg_stats_tbl.copy()

        if not active_sdgs_tbl.empty:
            # Create display dataframe
            display_df = active_sdgs_tbl.copy()
            display_df['SDG'] = display_df.apply(
                lambda row: f"SDG-{row['sdg_id']} {row['sdg_name']}",
                axis=1
            )

            # Select columns based on metric type
            if metric_type_tbl == "Total Budget":
                display_df['Number of Projects'] = display_df['project_count'].astype(int)
                display_df['Total Budget'] = display_df['total_budget'].apply(lambda x: f"${x:,.2f}")
                table_df = display_df[['SDG', 'Number of Projects', 'Total Budget']]
            else:
                display_df['Number of Projects'] = display_df['project_count'].astype(int)
                table_df = display_df[['SDG', 'Number of Projects']]

            # Display table
            st.dataframe(table_df, width='stretch', hide_index=True)

            # CSV Export button
            csv_data = active_sdgs_tbl.copy()
            csv_data['SDG'] = csv_data.apply(
                lambda row: f"SDG-{row['sdg_id']} {row['sdg_name']}",
                axis=1
            )

            if metric_type_tbl == "Total Budget":
                export_df = csv_data[['SDG', 'project_count', 'total_budget']]
                export_df.columns = ['SDG', 'Number of Projects', 'Total Budget']
            else:
                export_df = csv_data[['SDG', 'project_count']]
                export_df.columns = ['SDG', 'Number of Projects']

            csv_string = export_df.to_csv(index=False)

            # Create filename
            dept_suffix = f"_{selected_dept_tbl.replace(' ', '_')}" if selected_dept_tbl != 'All Departments' else ""
            metric_suffix = "_budget" if metric_type_tbl == "Total Budget" else "_projects"
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"sdg_table{dept_suffix}{metric_suffix}_{timestamp}.csv"

            st.download_button(
                label="📥 Export to CSV",
                data=csv_string,
                file_name=filename,
                mime="text/csv",
                width='stretch',
                key="export_sdg_table_csv"
            )

            # Add info about budget calculation if showing budget
            if metric_type_tbl == "Total Budget":
                st.info("""
                ℹ️ **Budget Calculation Note:** Budget is only shown for SDGs where projects have explicit percentage allocations set.
                Projects with SDG assignments but no percentage allocation (NULL weight) are not included in budget totals.
                Use the Manage SDG tab to set percentage allocations for projects.
                """)
        else:
            st.info("No SDGs assigned yet. Assign SDGs to projects below.")

    # Performance vs SDG Analysis
    with st.expander("📊 Performance vs SDG Analysis", expanded=False):
        st.markdown("""
        This analysis shows how project performance metrics vary across different SDG assignments.
        All metrics are budget-weighted within each SDG group.
        """)

        # Get all SDGs
        all_sdgs_perf = db_manager.get_all_sdgs()

        if not all_sdgs_perf.empty:
            # Get available status dates for the portfolio
            status_dates_query_sdg = """
                SELECT DISTINCT psr.status_date
                FROM project_status_report psr
                JOIN project p ON psr.project_id = p.project_id
                WHERE p.portfolio_id = ? AND p.is_active = TRUE
                ORDER BY psr.status_date DESC
            """
            status_dates_df_sdg = db.execute(status_dates_query_sdg, (portfolio_id,)).df()

            # Controls
            col_status_sdg, col_dept_sdg = st.columns(2)

            with col_status_sdg:
                if not status_dates_df_sdg.empty:
                    # Convert dates to strings for display
                    status_date_options_sdg = status_dates_df_sdg['status_date'].dt.strftime('%Y-%m-%d').tolist()

                    selected_status_date_str_sdg = st.selectbox(
                        "Select reporting period:",
                        options=status_date_options_sdg,
                        index=0,  # Default to latest
                        key="perf_sdg_status_date_selector"
                    )

                    # Convert back to date for query
                    from datetime import datetime as dt
                    selected_status_date_sdg = dt.strptime(selected_status_date_str_sdg, '%Y-%m-%d').date()
                else:
                    selected_status_date_sdg = None
                    st.warning("No status reports found")

            # Get comprehensive project data with performance metrics for selected date
            if selected_status_date_sdg:
                perf_query_sdg = """
                    SELECT
                        p.project_id,
                        p.project_name,
                        p.responsible_organization,
                        p.current_budget,
                        psr.spi,
                        psr.cpi,
                        psr.percent_time_used,
                        psr.percent_budget_used
                    FROM project p
                    LEFT JOIN project_status_report psr
                        ON p.project_id = psr.project_id
                        AND psr.status_date = ?
                    WHERE p.portfolio_id = ? AND p.is_active = TRUE
                """

                perf_df_sdg = db.execute(perf_query_sdg, (selected_status_date_sdg, portfolio_id)).df()
            else:
                perf_df_sdg = pd.DataFrame()

            with col_dept_sdg:
                if not perf_df_sdg.empty:
                    # Get unique departments
                    perf_df_sdg['responsible_organization'] = perf_df_sdg['responsible_organization'].fillna('Unassigned')
                    departments_perf_sdg = sorted(perf_df_sdg['responsible_organization'].unique())
                    dept_options_perf_sdg = ['All Departments'] + departments_perf_sdg

                    selected_dept_perf_sdg = st.selectbox(
                        "Filter by department:",
                        options=dept_options_perf_sdg,
                        key="dept_filter_perf_sdg"
                    )

                    # Filter by department
                    if selected_dept_perf_sdg != 'All Departments':
                        perf_df_sdg = perf_df_sdg[perf_df_sdg['responsible_organization'] == selected_dept_perf_sdg].copy()
                        st.info(f"📍 Showing data for: **{selected_dept_perf_sdg}**")

            if not perf_df_sdg.empty:
                # Rename columns for consistency
                perf_df_sdg['pct_duration_used'] = perf_df_sdg['percent_time_used']
                perf_df_sdg['pct_budget_used'] = perf_df_sdg['percent_budget_used']

                # Get all SDG assignments for the portfolio
                all_sdg_assignments_query = """
                    SELECT DISTINCT sdg_id, project_id
                    FROM project_sdg
                    WHERE portfolio_id = ?
                """
                all_sdg_assignments_df = db.execute(all_sdg_assignments_query, (portfolio_id,)).df()

                # Calculate budget-weighted averages for each SDG
                summary_rows_sdg = []

                for _, sdg_row in all_sdgs_perf.iterrows():
                    sdg_id = sdg_row['sdg_id']
                    sdg_name = sdg_row['sdg_name']
                    sdg_label = f"SDG-{sdg_id} {sdg_name}"

                    # Get projects with this SDG
                    if not all_sdg_assignments_df.empty:
                        projects_with_sdg = all_sdg_assignments_df[
                            all_sdg_assignments_df['sdg_id'] == sdg_id
                        ]['project_id'].tolist()
                    else:
                        projects_with_sdg = []

                    if projects_with_sdg:
                        # Filter performance data to projects with this SDG
                        sdg_perf_data = perf_df_sdg[perf_df_sdg['project_id'].isin(projects_with_sdg)].copy()

                        if not sdg_perf_data.empty:
                            # Filter to projects with budget for weighting
                            sdg_perf_with_budget = sdg_perf_data[
                                (sdg_perf_data['current_budget'].notna()) &
                                (sdg_perf_data['current_budget'] > 0)
                            ]

                            if not sdg_perf_with_budget.empty:
                                total_budget = sdg_perf_with_budget['current_budget'].sum()

                                # Budget-weighted SPI (filter out NaN and infinity)
                                spi_data = sdg_perf_with_budget[
                                    sdg_perf_with_budget['spi'].notna() &
                                    np.isfinite(sdg_perf_with_budget['spi'])
                                ]
                                if not spi_data.empty:
                                    weighted_spi = (spi_data['spi'] * spi_data['current_budget']).sum() / spi_data['current_budget'].sum()
                                else:
                                    weighted_spi = None

                                # Budget-weighted CPI (filter out NaN and infinity)
                                cpi_data = sdg_perf_with_budget[
                                    sdg_perf_with_budget['cpi'].notna() &
                                    np.isfinite(sdg_perf_with_budget['cpi'])
                                ]
                                if not cpi_data.empty:
                                    weighted_cpi = (cpi_data['cpi'] * cpi_data['current_budget']).sum() / cpi_data['current_budget'].sum()
                                else:
                                    weighted_cpi = None

                                # Budget-weighted % Duration Used
                                duration_data = sdg_perf_with_budget[sdg_perf_with_budget['pct_duration_used'].notna()]
                                if not duration_data.empty:
                                    weighted_duration = (duration_data['pct_duration_used'] * duration_data['current_budget']).sum() / duration_data['current_budget'].sum()
                                else:
                                    weighted_duration = None

                                # Budget-weighted % Budget Used
                                budget_used_data = sdg_perf_with_budget[sdg_perf_with_budget['pct_budget_used'].notna()]
                                if not budget_used_data.empty:
                                    weighted_budget_used = (budget_used_data['pct_budget_used'] * budget_used_data['current_budget']).sum() / budget_used_data['current_budget'].sum()
                                else:
                                    weighted_budget_used = None

                                summary_rows_sdg.append({
                                    'SDG': sdg_label,
                                    'Project Count': len(sdg_perf_data),
                                    'Total Budget': total_budget,
                                    'Avg SPI': weighted_spi,
                                    'Avg CPI': weighted_cpi,
                                    'Avg % Duration Used': weighted_duration,
                                    'Avg % Budget Used': weighted_budget_used
                                })

                if summary_rows_sdg:
                    summary_df_sdg = pd.DataFrame(summary_rows_sdg)

                    # Format the dataframe for display
                    display_df_sdg = summary_df_sdg.copy()
                    display_df_sdg['Total Budget'] = display_df_sdg['Total Budget'].apply(lambda x: f"${x:,.0f}" if pd.notna(x) else "N/A")
                    display_df_sdg['Avg SPI'] = display_df_sdg['Avg SPI'].apply(lambda x: f"{x:.2f}" if pd.notna(x) else "N/A")
                    display_df_sdg['Avg CPI'] = display_df_sdg['Avg CPI'].apply(lambda x: f"{x:.2f}" if pd.notna(x) else "N/A")
                    display_df_sdg['Avg % Duration Used'] = display_df_sdg['Avg % Duration Used'].apply(lambda x: f"{x:.1f}%" if pd.notna(x) else "N/A")
                    display_df_sdg['Avg % Budget Used'] = display_df_sdg['Avg % Budget Used'].apply(lambda x: f"{x:.1f}%" if pd.notna(x) else "N/A")

                    st.markdown(f"### Performance Analysis by SDG")
                    st.markdown(f"*As of: {selected_status_date_str_sdg} | All averages are budget-weighted*")

                    # Display as table
                    st.dataframe(
                        display_df_sdg,
                        width='stretch',
                        hide_index=True
                    )

                    # Add explanatory notes
                    st.markdown("---")
                    st.markdown(f"""
                    **Metric Definitions (as of {selected_status_date_str_sdg}):**
                    - **SPI (Schedule Performance Index)**: Earned Value / Planned Value. >1.0 is ahead of schedule.
                    - **CPI (Cost Performance Index)**: Earned Value / Actual Cost. >1.0 is under budget.
                    - **% Duration Used**: Percentage of project timeline elapsed as of the selected reporting date.
                    - **% Budget Used**: Percentage of budget consumed as of the selected reporting date.

                    **Budget Weighting**: Larger projects have proportionally more influence on the averages.

                    **Interpretation**: Compare performance metrics across different SDGs to identify which sustainability goals
                    are associated with better or worse project performance. Only SDGs with assigned projects are shown.
                    """)
                else:
                    st.info("No SDG assignments with performance data found for the selected department and reporting period.")
            else:
                st.info("No project performance data available for the selected reporting period.")
        else:
            st.info("No SDGs defined in the system.")


def manage_project_sdgs_ui(portfolio_id):
    """UI for assigning SDGs to a project"""
    st.markdown("### 🎯 Assign SDGs to Project")

    # Get projects for selected portfolio
    projects_df = db.execute("""
        SELECT project_id, project_name
        FROM project
        WHERE portfolio_id = ? AND is_active = TRUE
        ORDER BY project_name
    """, (portfolio_id,)).df()

    if projects_df.empty:
        st.info("📭 No projects found in this portfolio")
        return

    # Project selector
    project_options = {
        row['project_name']: row['project_id']
        for _, row in projects_df.iterrows()
    }

    selected_project_name = st.selectbox("Select Project", options=list(project_options.keys()))
    project_id = project_options[selected_project_name]

    st.markdown("---")

    # Get all SDGs and project's current SDG assignments
    all_sdgs = db_manager.get_all_sdgs()
    assigned_sdgs = db_manager.get_project_sdgs(project_id)

    st.markdown(f"#### SDG Goals for '{selected_project_name}'")
    st.caption(f"Currently assigned: {len(assigned_sdgs)} of 17 SDGs")

    # Display SDGs in a grid with checkboxes
    col1, col2 = st.columns(2)

    changes_made = False

    for idx, sdg_row in all_sdgs.iterrows():
        sdg_id = sdg_row['sdg_id']
        sdg_name = sdg_row['sdg_name']
        is_assigned = sdg_id in assigned_sdgs

        # Alternate between columns
        col = col1 if idx % 2 == 0 else col2

        with col:
            card_class = "assigned" if is_assigned else "unassigned"
            st.markdown(f'<div class="sdg-card {card_class}">', unsafe_allow_html=True)

            # Checkbox for assignment
            col_check, col_label = st.columns([1, 5])

            with col_check:
                new_state = st.checkbox(
                    f"Select SDG {sdg_id}",
                    value=is_assigned,
                    key=f"sdg_{sdg_id}",
                    label_visibility="collapsed"
                )

            with col_label:
                badge_class = "badge-assigned" if is_assigned else "badge-unassigned"
                st.markdown(
                    f'<span class="sdg-badge {badge_class}">SDG {sdg_id}</span>',
                    unsafe_allow_html=True
                )
                st.markdown(f"**{sdg_name}**")

            st.markdown('</div>', unsafe_allow_html=True)

            # Handle assignment changes (write access required)
            _can_write = st.session_state.get('portfolio_access_level') in ('owner', 'write')
            if _can_write:
                if new_state and not is_assigned:
                    # Assign SDG
                    if db_manager.assign_sdg_to_project(project_id, sdg_id):
                        changes_made = True
                elif not new_state and is_assigned:
                    # Remove SDG
                    if db_manager.remove_sdg_from_project(project_id, sdg_id):
                        changes_made = True

    # Refresh page if changes were made
    if changes_made:
        st.rerun()


def view_all_sdgs_ui():
    """UI for viewing all SDG goals"""
    st.markdown("### 🌍 UN Sustainable Development Goals")
    st.caption("All 17 United Nations Sustainable Development Goals")

    all_sdgs = db_manager.get_all_sdgs()

    for _, sdg_row in all_sdgs.iterrows():
        sdg_id = sdg_row['sdg_id']
        sdg_name = sdg_row['sdg_name']
        color = SDG_COLORS.get(sdg_id, "#95a5a6")

        st.markdown(
            f'<div style="padding: 0.75rem; margin: 0.5rem 0; border-left: 5px solid {color}; '
            f'background: rgba(255,255,255,0.95); border-radius: 5px;">'
            f'<span style="font-weight: 600; color: {color};">SDG {sdg_id}:</span> {sdg_name}'
            f'</div>',
            unsafe_allow_html=True
        )


def manage_sdg_ui(portfolio_id):
    """UI for importing and exporting SDG weights"""
    st.markdown("### 📊 Manage SDG Weights")
    st.markdown("Import and export Sustainable Development Goals (SDG) weights for projects")

    # Get portfolio name
    portfolio_df = db.execute("""
        SELECT portfolio_name
        FROM portfolio
        WHERE portfolio_id = ?
    """, (portfolio_id,)).df()

    portfolio_name = portfolio_df.iloc[0]['portfolio_name'] if not portfolio_df.empty else "Unknown"

    # Create sub-tabs for export, import, template
    export_tab, import_tab, template_tab = st.tabs(["📤 Export", "📥 Import", "📋 Template"])

    # EXPORT TAB
    with export_tab:
        st.markdown("#### Export SDG Weights")
        st.markdown("""
        Export SDG weights to a CSV file with the following format:
        - `portfolio_id, project_id, project_name, organization, project_manager, sdg1, sdg2, ..., sdg17`
        - Each SDG column contains the weight percentage (0-100)
        - Note: `project_name`, `organization`, `project_manager` are for reference only
        """)

        col1, col2 = st.columns([2, 1])

        with col1:
            export_scope = st.radio(
                "Export scope:",
                ["Current portfolio only", "All portfolios"],
                key="export_scope"
            )

        with col2:
            st.markdown("##### Quick Stats")
            # Get count of projects with SDG weights
            if export_scope == "Current portfolio only":
                count_query = """
                    SELECT COUNT(DISTINCT project_id)
                    FROM project_sdg
                    WHERE portfolio_id = ?
                """
                params = (portfolio_id,)
            else:
                count_query = "SELECT COUNT(DISTINCT project_id) FROM project_sdg"
                params = ()

            result = db.fetch_one(count_query, params if params else None)
            project_count = result[0] if result else 0

            st.metric("Projects with SDG weights", project_count)

        if st.button("📥 Export to CSV", type="primary", width='stretch', key="export_btn"):
            with st.spinner("Exporting..."):
                # Create temp file
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                filename = f"sdg_export_{portfolio_name}_{timestamp}.csv" if export_scope == "Current portfolio only" else f"sdg_export_all_{timestamp}.csv"

                temp_dir = tempfile.gettempdir()
                output_path = os.path.join(temp_dir, filename)

                # Export
                export_portfolio_id = portfolio_id if export_scope == "Current portfolio only" else None
                success, message, count = export_sdg_to_csv(output_path, export_portfolio_id)

                if success:
                    st.success(message)

                    # Read the file for download
                    with open(output_path, 'r') as f:
                        csv_data = f.read()

                    # Provide download button
                    st.download_button(
                        label="💾 Download CSV",
                        data=csv_data,
                        file_name=filename,
                        mime="text/csv",
                        width='stretch',
                        key="download_export_btn"
                    )

                    # Show preview
                    st.markdown("##### Preview")
                    df = pd.read_csv(output_path)
                    st.dataframe(df.head(10), width='stretch')

                    if len(df) > 10:
                        st.info(f"Showing first 10 of {len(df)} rows")

                    # Cleanup
                    try:
                        os.remove(output_path)
                    except:
                        pass
                else:
                    st.error(message)

    # IMPORT TAB
    with import_tab:
        st.markdown("#### Import SDG Weights")
        st.markdown("""
        Upload a CSV file with SDG weights in the following format:
        - Required: `portfolio_id, project_id, sdg1, sdg2, sdg3, ..., sdg17`
        - Each SDG column should contain the weight percentage (0-100)
        - Weights for each project should sum to 100
        - Note: `project_name`, `organization`, `project_manager` columns are ignored if present
        """)

        st.warning("⚠️ **Important:** This will replace existing SDG weights for the projects in the CSV file.")

        # Upload file
        uploaded_file = st.file_uploader(
            "Choose a CSV file",
            type=['csv'],
            help="Upload a CSV file with SDG weights",
            key="import_file_uploader"
        )

        if uploaded_file is not None:
            # Save to temp file
            temp_dir = tempfile.gettempdir()
            input_path = os.path.join(temp_dir, "sdg_import_temp.csv")

            with open(input_path, 'wb') as f:
                f.write(uploaded_file.getvalue())

            # Show preview
            st.markdown("##### File Preview")
            df = pd.read_csv(input_path)
            st.dataframe(df.head(10), width='stretch')

            if len(df) > 10:
                st.info(f"File contains {len(df)} rows (showing first 10)")

            # Validate
            st.markdown("##### Validation")
            with st.spinner("Validating..."):
                is_valid, errors = validate_sdg_csv(input_path)

            if is_valid:
                st.success("✓ CSV file is valid and ready to import")
            else:
                st.error("✗ CSV file has validation errors:")
                for error in errors:
                    st.markdown(f"- {error}")

            # Import options
            st.markdown("##### Import Options")
            col1, col2 = st.columns(2)

            with col1:
                import_scope = st.radio(
                    "Import scope:",
                    ["Current portfolio only", "All portfolios in file"],
                    key="import_scope"
                )

            with col2:
                validate_totals = st.checkbox(
                    "Validate that weights sum to 100",
                    value=True,
                    help="If enabled, will reject projects where SDG weights don't sum to 100",
                    key="validate_totals_cb"
                )

            # Import button (requires write access)
            _import_write = st.session_state.get('portfolio_access_level') in ('owner', 'write')
            if not _import_write:
                st.warning("Write access required to import SDG data.")
            elif st.button("📤 Import from CSV", type="primary", width='stretch', disabled=not is_valid, key="import_btn"):
                with st.spinner("Importing..."):
                    import_portfolio_id = portfolio_id if import_scope == "Current portfolio only" else None
                    success, message, stats = import_sdg_from_csv(
                        input_path,
                        portfolio_id=import_portfolio_id,
                        validate_totals=validate_totals,
                        replace_existing=True
                    )

                    if success:
                        st.success(message)

                        # Show stats
                        col1, col2, col3, col4 = st.columns(4)
                        with col1:
                            st.metric("Imported", stats['imported'])
                        with col2:
                            st.metric("Updated", stats['updated'])
                        with col3:
                            st.metric("Skipped", stats['skipped'])
                        with col4:
                            st.metric("Errors", stats['errors'])

                        st.balloons()
                    else:
                        st.error(message)
                        st.json(stats)

            # Cleanup
            try:
                if os.path.exists(input_path):
                    os.remove(input_path)
            except:
                pass

    # TEMPLATE TAB
    with template_tab:
        st.markdown("#### Download Template")
        st.markdown(f"""
        Download a template CSV file with all active projects from **{portfolio_name}**.

        The template will include:
        - All active projects with their details (name, organization, project manager) for reference
        - Empty SDG columns (sdg1 to sdg17) ready to fill in
        """)

        # Show project count preview
        project_count_result = db.execute(
            "SELECT COUNT(*) FROM project WHERE portfolio_id = ? AND is_active = TRUE",
            (portfolio_id,)
        ).fetchone()
        project_count = project_count_result[0] if project_count_result else 0

        st.info(f"📊 Template will include **{project_count}** active project(s) from **{portfolio_name}**")

        if st.button("📋 Generate Template", type="primary", width='stretch', key="template_btn"):
            with st.spinner("Generating template..."):
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                filename = f"sdg_template_{portfolio_name}_{timestamp}.csv"

                temp_dir = tempfile.gettempdir()
                output_path = os.path.join(temp_dir, filename)

                # Generate template for selected portfolio
                success, message = get_sdg_template_csv(output_path, portfolio_id)

                if success:
                    st.success(message)

                    # Read the file for download
                    with open(output_path, 'r') as f:
                        csv_data = f.read()

                    # Provide download button
                    st.download_button(
                        label="💾 Download Template",
                        data=csv_data,
                        file_name=filename,
                        mime="text/csv",
                        width='stretch',
                        key="download_template_btn"
                    )

                    # Show preview
                    st.markdown("##### Template Preview")
                    df = pd.read_csv(output_path)
                    st.dataframe(df.head(10), width='stretch')

                    if len(df) > 10:
                        st.info(f"Template contains {len(df)} projects (showing first 10)")

                    # Cleanup
                    try:
                        os.remove(output_path)
                    except:
                        pass
                else:
                    st.error(message)

    # SDG Reference section
    st.markdown("---")
    st.markdown("""
    ##### 📚 SDG Reference

    The 17 UN Sustainable Development Goals:
    1. No Poverty | 2. Zero Hunger | 3. Good Health and Well-being | 4. Quality Education | 5. Gender Equality | 6. Clean Water and Sanitation
    7. Affordable and Clean Energy | 8. Decent Work and Economic Growth | 9. Industry, Innovation and Infrastructure
    10. Reduced Inequalities | 11. Sustainable Cities and Communities | 12. Responsible Consumption and Production
    13. Climate Action | 14. Life Below Water | 15. Life on Land | 16. Peace, Justice and Strong Institutions | 17. Partnerships for the Goals
    """)


# Main page layout
portfolio_id = select_portfolio()

# Compute write access
_has_write = False
if portfolio_id:
    try:
        from utils.portfolio_access import get_portfolio_access_level
        from utils.portfolio_context import get_current_user_email
        _user_email = get_current_user_email()
        _access_level = get_portfolio_access_level(portfolio_id, _user_email)
        _has_write = _access_level in ('owner', 'write')
        st.session_state['portfolio_access_level'] = _access_level
    except Exception:
        _has_write = True

    if not _has_write:
        st.info("You have **read-only** access to this portfolio. SDG editing is disabled.")

if portfolio_id:
    st.markdown("---")

    tab1, tab2, tab3, tab4, tab5 = st.tabs(["📊 Overview", "🎯 Assign SDGs", "📤 Manage SDG", "🌍 All SDGs", "ℹ️ Help"])

    with tab1:
        display_sdg_overview(portfolio_id)

    with tab2:
        manage_project_sdgs_ui(portfolio_id)

    with tab3:
        manage_sdg_ui(portfolio_id)

    with tab4:
        view_all_sdgs_ui()

    with tab5:
        st.markdown("""
        ## SDG Management Help

        ### 🌍 What are SDGs?

        The **Sustainable Development Goals (SDGs)** are 17 global goals established by the United Nations in 2015 as part of the 2030 Agenda for Sustainable Development. These goals provide a shared blueprint for peace and prosperity for people and the planet.

        ### 📋 The 17 SDGs

        1. **No Poverty** - End poverty in all its forms everywhere
        2. **Zero Hunger** - End hunger, achieve food security and improved nutrition
        3. **Good Health and Well-being** - Ensure healthy lives and promote well-being for all
        4. **Quality Education** - Ensure inclusive and equitable quality education
        5. **Gender Equality** - Achieve gender equality and empower all women and girls
        6. **Clean Water and Sanitation** - Ensure availability and sustainable management of water
        7. **Affordable and Clean Energy** - Ensure access to affordable, reliable, sustainable energy
        8. **Decent Work and Economic Growth** - Promote sustained, inclusive economic growth
        9. **Industry, Innovation and Infrastructure** - Build resilient infrastructure
        10. **Reduced Inequalities** - Reduce inequality within and among countries
        11. **Sustainable Cities and Communities** - Make cities inclusive, safe, resilient
        12. **Responsible Consumption and Production** - Ensure sustainable consumption patterns
        13. **Climate Action** - Take urgent action to combat climate change
        14. **Life Below Water** - Conserve and sustainably use oceans, seas, marine resources
        15. **Life on Land** - Protect, restore, promote sustainable use of terrestrial ecosystems
        16. **Peace, Justice and Strong Institutions** - Promote peaceful and inclusive societies
        17. **Partnerships for the Goals** - Strengthen means of implementation and partnerships

        ### 🎯 Using SDG Management

        **Assigning SDGs to Projects:**
        1. Select your portfolio
        2. Go to "Assign SDGs" tab
        3. Select a project
        4. Check/uncheck SDG goals that the project contributes to
        5. Changes are saved automatically

        **Viewing SDG Coverage:**
        1. Go to "Overview" tab
        2. View statistics on how many projects have SDG assignments
        3. See distribution chart showing which SDGs are most addressed

        ### 📊 Benefits of SDG Tracking

        **Strategic Alignment:**
        - Demonstrate alignment with global sustainability goals
        - Show corporate social responsibility
        - Track ESG (Environmental, Social, Governance) contributions

        **Portfolio Analysis:**
        - Identify gaps in SDG coverage
        - Balance portfolio across different sustainability areas
        - Prioritize projects with high SDG impact

        **Reporting:**
        - Generate SDG impact reports for stakeholders
        - Show contribution to UN 2030 Agenda
        - Support sustainability disclosures

        ### ✅ Best Practices

        1. **Be Specific**: Only assign SDGs that the project directly contributes to
        2. **Primary vs Secondary**: Consider which SDGs are primary goals vs secondary benefits
        3. **Review Regularly**: Update SDG assignments as project scope changes
        4. **Portfolio Balance**: Aim for coverage across multiple SDGs, not just one or two
        5. **Document Impact**: Use project notes to explain how each SDG is addressed

        ### 🔗 Integration

        SDG assignments are used in:
        - Portfolio reporting and dashboards
        - Strategic alignment analysis (coming in Strategic Factors)
        - ESG impact assessments
        - Project prioritization and selection

        ### 📚 Learn More

        For more information about the UN SDGs:
        - Official SDG website: https://sdgs.un.org
        - SDG indicators: https://unstats.un.org/sdgs
        - Corporate SDG guide: https://sdgcompass.org
        """)

# Footer
st.markdown("---")
st.caption("SDG Management - Portfolio Management Suite v1.0")

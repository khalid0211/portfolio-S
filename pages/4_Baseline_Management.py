"""
Baseline Management - Portfolio Management Suite
View and manage project baselines with versioning and effectivity dates
"""

import streamlit as st
import pandas as pd
from datetime import datetime, date, timedelta
from typing import Optional
from utils.auth import check_authentication, require_page_access
from config.constants import USE_DATABASE
from services.db_data_service import DatabaseDataManager
from database.db_connection import get_db

# Page configuration
st.set_page_config(
    page_title="Baseline Management - Portfolio Suite",
    page_icon="📐",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# Check authentication and page access
if not check_authentication():
    st.stop()

require_page_access('baseline_management', 'Baseline Management')

# Custom CSS
st.markdown("""
<style>
    .main-header {
        text-align: center;
        color: #2c3e50;
        font-size: 2.2rem;
        font-weight: 600;
        margin-bottom: 1rem;
        border-bottom: 3px solid #3498db;
        padding-bottom: 0.5rem;
    }
    .section-header {
        background: linear-gradient(90deg, #3498db 0%, #2c3e50 100%);
        color: white;
        padding: 0.75rem 1rem;
        border-radius: 8px;
        font-size: 1.2rem;
        font-weight: 600;
        margin: 1rem 0 0.5rem 0;
    }
    div[data-testid="stExpander"] div[role="button"] p {
        font-size: 1rem;
        font-weight: 600;
    }
</style>
""", unsafe_allow_html=True)

# Main header
st.markdown('<h1 class="main-header">📐 Baseline Management</h1>', unsafe_allow_html=True)

# Check database mode
if not USE_DATABASE:
    st.warning("⚠️ Database mode is not enabled")
    st.info("""
    **Database Mode Required:**
    Baseline management requires database mode to be enabled.
    Set `USE_DATABASE = True` in `config/constants.py`
    """)
    st.stop()

# Initialize database connection with user-specific config
user_email = st.session_state.get('user_email')
db = get_db(user_email=user_email)

# Initialize database manager (uses the singleton db instance)
db_manager = DatabaseDataManager()


def select_portfolio_and_project():
    """UI for selecting portfolio and project with Project ID"""
    col1, col2 = st.columns(2)

    with col1:
        # Get portfolios
        portfolios_df = db.execute("""
            SELECT portfolio_id, portfolio_name
            FROM portfolio
            WHERE is_active = TRUE
            ORDER BY portfolio_name
        """).df()

        if portfolios_df.empty:
            st.warning("⚠️ No portfolios found. Create a portfolio first in Portfolio Management.")
            return None, None, None

        portfolio_options = {
            row['portfolio_name']: row['portfolio_id']
            for _, row in portfolios_df.iterrows()
        }

        selected_portfolio_name = st.selectbox("Select Portfolio", options=list(portfolio_options.keys()))
        portfolio_id = portfolio_options[selected_portfolio_name]

    with col2:
        # Get projects for selected portfolio
        projects_df = db.execute("""
            SELECT project_id, project_name
            FROM project
            WHERE portfolio_id = ? AND is_active = TRUE
            ORDER BY project_name
        """, (portfolio_id,)).df()

        if projects_df.empty:
            st.info(f"📭 No projects found in portfolio '{selected_portfolio_name}'")
            return portfolio_id, None, None

        # Create options showing Project ID and Name
        project_options = {
            f"{row['project_id']} - {row['project_name']}": (row['project_id'], row['project_name'])
            for _, row in projects_df.iterrows()
        }

        selected_project_display = st.selectbox("Select Project", options=list(project_options.keys()))
        project_id, project_name = project_options[selected_project_display]

    return portfolio_id, project_id, project_name


def edit_baseline_modal(baseline_row):
    """Display edit form for a baseline"""
    baseline_id = baseline_row['baseline_id']
    baseline_version = baseline_row['baseline_version']

    st.markdown(f"### ✏️ Edit Baseline {baseline_version}")

    if baseline_version == 0:
        st.warning("⚠️ Baseline 0 (initial baseline) cannot be edited")
        return

    with st.form(f"edit_baseline_{baseline_id}"):
        col1, col2 = st.columns(2)

        with col1:
            st.markdown("**Budget:**")
            new_bac = st.number_input(
                "BAC (Budget at Completion)",
                min_value=0.0,
                value=float(baseline_row.get('budget_at_completion', 0)),
                step=1000.0,
                key=f"edit_bac_{baseline_id}"
            )

            st.markdown("**Schedule:**")
            current_start = pd.to_datetime(baseline_row['planned_start_date']).date()
            new_start = st.date_input(
                "Planned Start Date",
                value=current_start,
                key=f"edit_start_{baseline_id}"
            )

            current_finish = pd.to_datetime(baseline_row['planned_finish_date']).date()
            new_finish = st.date_input(
                "Planned Finish Date",
                value=current_finish,
                key=f"edit_finish_{baseline_id}"
            )

        with col2:
            st.markdown("**Effectivity:**")
            current_baseline_start = pd.to_datetime(baseline_row['baseline_start_date']).date()
            baseline_start = st.date_input(
                "Baseline Effective From",
                value=current_baseline_start,
                key=f"edit_eff_start_{baseline_id}"
            )

            current_baseline_end = baseline_row.get('baseline_end_date')
            if current_baseline_end and not pd.isna(current_baseline_end):
                current_baseline_end = pd.to_datetime(current_baseline_end).date()
                baseline_end = st.date_input(
                    "Baseline Effective Until (leave blank for active)",
                    value=current_baseline_end,
                    key=f"edit_eff_end_{baseline_id}"
                )
            else:
                st.info("This is the active baseline (no end date)")
                baseline_end = None

        notes = st.text_area(
            "Reason for Change",
            value=baseline_row.get('baseline_reason', ''),
            placeholder="Document reason for baseline modification",
            key=f"edit_notes_{baseline_id}"
        )

        col_submit, col_cancel = st.columns([1, 1])

        with col_submit:
            submitted = st.form_submit_button("💾 Save Changes", type="primary", width='stretch')

        with col_cancel:
            cancelled = st.form_submit_button("❌ Cancel", width='stretch')

        if submitted:
            if new_bac <= 0:
                st.error("❌ BAC must be greater than zero")
            elif new_start >= new_finish:
                st.error("❌ Planned start date must be before finish date")
            else:
                try:
                    baseline_data = {
                        'budget_at_completion': new_bac,
                        'planned_start_date': new_start,
                        'planned_finish_date': new_finish,
                        'baseline_start_date': baseline_start,
                        'baseline_end_date': baseline_end,
                        'baseline_reason': notes
                    }

                    db_manager.update_baseline(baseline_id, baseline_data)
                    st.success(f"✅ Baseline {baseline_version} updated successfully")
                    st.rerun()

                except Exception as e:
                    st.error(f"❌ Error updating baseline: {e}")

        if cancelled:
            st.rerun()


def delete_baseline_action(baseline_id, baseline_version, project_id):
    """Delete a baseline with confirmation"""
    if baseline_version == 0:
        st.error("❌ Baseline 0 (initial baseline) cannot be deleted")
        return

    if st.button(f"🗑️ Confirm Delete Baseline {baseline_version}", key=f"confirm_delete_{baseline_id}"):
        try:
            db_manager.delete_baseline(baseline_id)
            st.success(f"✅ Baseline {baseline_version} deleted successfully")
            st.rerun()
        except Exception as e:
            st.error(f"❌ Error deleting baseline: {e}")


def create_baseline_form(project_id, project_name):
    """Form to create new baseline"""
    st.markdown('<div class="section-header">➕ Create New Baseline</div>', unsafe_allow_html=True)

    st.info("""
    **Creating a New Baseline:**
    - This will supersede the current active baseline
    - Use this when project scope, budget, or schedule changes significantly
    """)

    with st.form("create_baseline_form"):
        col1, col2 = st.columns(2)

        with col1:
            st.markdown("**Budget:**")
            new_bac = st.number_input("BAC (Budget at Completion)", min_value=0.0, value=0.0, step=1000.0)

            st.markdown("**Schedule:**")
            new_start = st.date_input("Planned Start Date", value=date.today())
            new_finish = st.date_input("Planned Finish Date", value=date.today() + timedelta(days=365))

        with col2:
            st.markdown("**Effectivity:**")
            baseline_start = st.date_input(
                "Baseline Effective From",
                value=date.today(),
                help="Date from which this baseline becomes effective"
            )

            notes = st.text_area(
                "Reason for New Baseline",
                placeholder="Document reason (e.g., scope change, budget revision, contract modification)",
                height=100
            )

        submitted = st.form_submit_button("🚀 Create Baseline", type="primary", width='stretch')

        if submitted:
            if new_bac <= 0:
                st.error("❌ BAC must be greater than zero")
            elif new_start >= new_finish:
                st.error("❌ Planned start date must be before finish date")
            else:
                try:
                    baseline_data = {
                        'budget_at_completion': new_bac,
                        'planned_start_date': new_start,
                        'planned_finish_date': new_finish,
                        'baseline_start_date': baseline_start,
                        'baseline_reason': notes
                    }

                    baseline_id = db_manager.create_baseline(project_id, baseline_data)
                    st.success(f"✅ New baseline created successfully (ID: {baseline_id})")
                    st.info("The previous baseline has been superseded and is now historical.")
                    st.rerun()

                except Exception as e:
                    st.error(f"❌ Error creating baseline: {e}")


def display_baselines_table(project_id, project_name):
    """Display baselines in table format with edit/delete actions"""
    st.markdown('<div class="section-header">📋 Baselines</div>', unsafe_allow_html=True)

    try:
        # Get all baselines for project
        baselines_df = db_manager.list_baselines(project_id)

        if baselines_df.empty:
            st.info("📭 No baselines found for this project. Baselines are created automatically when loading CSV data.")
            return

        # Prepare display dataframe
        display_df = pd.DataFrame({
            'Baseline No': baselines_df['baseline_version'],
            'Effective Date': pd.to_datetime(baselines_df['baseline_start_date']).dt.strftime('%Y-%m-%d'),
            'End Date': baselines_df['baseline_end_date'].apply(
                lambda x: pd.to_datetime(x).strftime('%Y-%m-%d') if pd.notna(x) else 'Active'
            ),
            'Project ID': baselines_df['project_id'],
            'Project Name': project_name,
            'BAC': baselines_df['budget_at_completion'].apply(lambda x: f"${x:,.2f}"),
            'Plan Start': pd.to_datetime(baselines_df['planned_start_date']).dt.strftime('%Y-%m-%d'),
            'Plan Finish': pd.to_datetime(baselines_df['planned_finish_date']).dt.strftime('%Y-%m-%d'),
            'Status': baselines_df['baseline_end_date'].apply(lambda x: '🟢 Active' if pd.isna(x) else '⚪ Historical')
        })

        # Display summary
        active_count = baselines_df['baseline_end_date'].isna().sum()
        st.markdown(f"**Total Baselines:** {len(baselines_df)} | **Active:** {active_count} | **Historical:** {len(baselines_df) - active_count}")

        # Display table
        st.dataframe(
            display_df,
            width='stretch',
            hide_index=True,
            height=min(400, (len(display_df) + 1) * 35 + 3)
        )

        st.markdown("---")
        st.markdown("### 🔧 Manage Baselines")
        st.caption("Click on a baseline to edit or delete it")

        # Create expander for each baseline
        for idx, baseline_row in baselines_df.iterrows():
            baseline_version = baseline_row['baseline_version']
            baseline_id = baseline_row['baseline_id']
            is_active = pd.isna(baseline_row['baseline_end_date'])

            status_icon = "🟢" if is_active else "⚪"
            title = f"{status_icon} Baseline {baseline_version}"
            if baseline_version == 0:
                title += " (Initial - View Only)"

            with st.expander(title):
                col_info, col_actions = st.columns([3, 1])

                with col_info:
                    st.markdown(f"""
                    **Baseline Information:**
                    - **BAC:** ${baseline_row['budget_at_completion']:,.2f}
                    - **Plan Start:** {pd.to_datetime(baseline_row['planned_start_date']).strftime('%Y-%m-%d')}
                    - **Plan Finish:** {pd.to_datetime(baseline_row['planned_finish_date']).strftime('%Y-%m-%d')}
                    - **Effective From:** {pd.to_datetime(baseline_row['baseline_start_date']).strftime('%Y-%m-%d')}
                    - **Effective Until:** {pd.to_datetime(baseline_row['baseline_end_date']).strftime('%Y-%m-%d') if pd.notna(baseline_row['baseline_end_date']) else 'Current'}
                    """)

                    if baseline_row.get('baseline_reason'):
                        st.markdown(f"**Reason:** {baseline_row['baseline_reason']}")

                with col_actions:
                    _write_ok = st.session_state.get('portfolio_access_level') in ('owner', 'write')
                    if baseline_version == 0:
                        st.info("Cannot edit or delete Baseline 0")
                    elif not _write_ok:
                        st.caption("Read-only")
                    else:
                        if st.button(f"✏️ Edit", key=f"edit_btn_{baseline_id}", width='stretch'):
                            st.session_state[f'editing_{baseline_id}'] = True

                        if st.button(f"🗑️ Delete", key=f"delete_btn_{baseline_id}", type="secondary", width='stretch'):
                            st.session_state[f'deleting_{baseline_id}'] = True

                # Show edit form if edit button clicked
                if st.session_state.get(f'editing_{baseline_id}', False):
                    st.markdown("---")
                    edit_baseline_modal(baseline_row)
                    st.session_state[f'editing_{baseline_id}'] = False

                # Show delete confirmation if delete button clicked
                if st.session_state.get(f'deleting_{baseline_id}', False):
                    st.markdown("---")
                    st.warning(f"⚠️ Are you sure you want to delete Baseline {baseline_version}? This action cannot be undone.")
                    delete_baseline_action(baseline_id, baseline_version, project_id)
                    if st.button("❌ Cancel Delete", key=f"cancel_delete_{baseline_id}"):
                        st.session_state[f'deleting_{baseline_id}'] = False
                        st.rerun()

    except Exception as e:
        st.error(f"❌ Error loading baselines: {e}")


# Main page layout
portfolio_id, project_id, project_name = select_portfolio_and_project()

# Compute write access for this portfolio
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
        _has_write = True  # Fallback: allow write if check fails

    if not _has_write:
        st.info("You have **read-only** access to this portfolio. Baseline editing is disabled.")

if portfolio_id and project_id and project_name:
    st.markdown("---")

    # Display baselines table
    display_baselines_table(project_id, project_name)

    st.markdown("---")

    # Create baseline form (only if user has write access)
    if _has_write:
        create_baseline_form(project_id, project_name)

    # Help section
    st.markdown("---")
    with st.expander("ℹ️ Baseline Management Help"):
        st.markdown("""
        ## Baseline Management Help

        ### 📐 What is a Baseline?

        A **baseline** is a fixed reference point for measuring project performance. It captures:
        - **Budget at Completion (BAC)**: Total planned budget
        - **Planned Schedule**: Start and finish dates
        - **Effectivity Period**: When this baseline is valid

        ### 🔄 Baseline Versioning

        Projects can have multiple baseline versions over time:
        - **Version 0**: Initial baseline (created when project is first loaded) - **Cannot be edited or deleted**
        - **Version 1+**: Subsequent baselines (created for re-baselining) - **Can be edited or deleted**

        Each baseline has:
        - **Effective From Date**: When it becomes active
        - **Effective Until Date**: When it's superseded (NULL if current)

        ### ⏰ Baseline Effectivity

        The **effective baseline** for a given date is determined by:
        ```
        WHERE baseline_start_date <= status_date
          AND (baseline_end_date IS NULL OR status_date < baseline_end_date)
        ```

        **Example:**
        - Baseline v0: Effective 2024-01-01 to 2024-06-30
        - Baseline v1: Effective 2024-07-01 to *current*

        When calculating EVM for 2024-05-15, v0 is used.
        When calculating EVM for 2024-08-01, v1 is used.

        ### 🎯 When to Create a New Baseline

        Create a new baseline when:
        1. **Scope Change**: Major project scope revision
        2. **Budget Revision**: Significant budget increase/decrease
        3. **Schedule Change**: Major timeline adjustment
        4. **Contractual Change**: Change order or contract modification
        5. **Re-baselining**: Periodic re-baseline (e.g., annual)

        **Do NOT create a new baseline for:**
        - Minor adjustments
        - Regular progress updates
        - Temporary delays
        - Cost variances within tolerance

        ### 📊 Using Baselines

        **EVM Calculations:**
        - EV and PV are calculated against the **effective baseline** for the status date
        - This ensures consistent performance measurement over time
        - Historical analysis uses the baseline that was active at that time

        ### ✅ Best Practices

        1. **Document Changes**: Always add notes explaining why a baseline was created/modified
        2. **Timing**: Create new baselines at logical project milestones
        3. **Approval**: Ensure stakeholder approval before re-baselining
        4. **Communication**: Inform team when baselines change
        5. **History**: Preserve old baselines for audit trail (don't delete unless necessary)
        """)

# Footer
st.markdown("---")
st.caption("Baseline Management - Portfolio Management Suite v1.0")

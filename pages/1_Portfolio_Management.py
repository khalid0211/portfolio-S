"""
Portfolio Management - Portfolio Management Suite
Manage portfolios, assign owners, and configure portfolio settings
"""

import streamlit as st
import pandas as pd
from datetime import datetime, date
from typing import Optional, Dict, Any
from utils.auth import check_authentication, require_page_access
from config.constants import USE_DATABASE
from services.data_service import data_manager
from services.db_data_service import DatabaseDataManager
from database.db_connection import get_db
from utils.portfolio_context import render_portfolio_selector, get_user_portfolios, get_portfolio_summary
from utils.user_sync import sync_authenticated_user_to_db, get_current_user_id

# Page configuration
st.set_page_config(
    page_title="Portfolio Management - Portfolio Suite",
    page_icon="📂",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# Check authentication and page access
if not check_authentication():
    st.stop()

require_page_access('portfolio_management', 'Portfolio Management')

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
    .portfolio-card {
        background: rgba(255, 255, 255, 0.95);
        padding: 1.5rem;
        border-radius: 15px;
        margin: 1rem 0;
        border: 1px solid rgba(52, 152, 219, 0.3);
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
    }
    .portfolio-card:hover {
        box-shadow: 0 6px 12px rgba(0, 0, 0, 0.15);
        transform: translateY(-2px);
        transition: all 0.3s ease;
    }
    .stat-box {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
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
st.markdown('<h1 class="main-header">📂 Portfolio Management</h1>', unsafe_allow_html=True)
st.markdown("Create and manage project portfolios")

# Check database mode
if not USE_DATABASE:
    st.warning("⚠️ Database mode is not enabled")
    st.info("""
    **Database Mode Required:**
    Portfolio management requires database mode to be enabled.
    Set `USE_DATABASE = True` in `config/constants.py`
    """)
    st.stop()

# Initialize database connection with user-specific config
user_email = st.session_state.get('user_email')
db = get_db(user_email=user_email)

# Initialize database manager (uses the singleton db instance)
db_manager = DatabaseDataManager()

# Sync authenticated user to database
try:
    current_user_id = sync_authenticated_user_to_db()
    if not current_user_id:
        st.error("⚠️ Unable to sync user to database.")

        # Debug info
        with st.expander("🔍 Debug Info"):
            st.write("**Session State:**")
            st.write(f"- Authenticated: {st.session_state.get('authenticated', False)}")
            st.write(f"- User Info: {st.session_state.get('user_info', {})}")

            # Try to check if user exists
            user_info = st.session_state.get('user_info', {})
            if user_info and user_info.get('email'):
                try:
                    existing = db.execute(
                        "SELECT user_id, email, display_name FROM app_user WHERE email = ?",
                        (user_info['email'],)
                    ).fetchone()
                    if existing:
                        st.write(f"- User exists in DB: {existing}")
                        st.info("User exists but sync failed. Try refreshing the page.")
                    else:
                        st.write("- User does not exist in app_user table")
                        st.info("Creating user manually...")

                        # Try to create user manually
                        max_id = db.execute("SELECT COALESCE(MAX(user_id), 0) + 1 FROM app_user").fetchone()
                        user_id = max_id[0]
                        name = user_info.get('name', user_info.get('email', 'Unknown User'))

                        db.execute("""
                            INSERT INTO app_user (user_id, display_name, email, is_active)
                            VALUES (?, ?, ?, TRUE)
                        """, (user_id, name, user_info['email']))

                        st.success(f"✅ User created with ID: {user_id}. Please refresh the page.")
                        current_user_id = user_id
                except Exception as e:
                    st.error(f"Database error: {e}")
                    import traceback
                    st.code(traceback.format_exc())

        if not current_user_id:
            st.stop()
except Exception as e:
    st.error(f"⚠️ Error during user sync: {e}")
    import traceback
    st.code(traceback.format_exc())
    st.stop()

# Portfolio selection section
st.markdown("---")
st.markdown("### 🎯 Select Active Portfolio")
st.caption("Choose which portfolio you want to work with")

portfolio_id = render_portfolio_selector()

# Store access level in session state for this portfolio
if portfolio_id:
    try:
        from utils.portfolio_access import get_portfolio_access_level
        _user_email = st.session_state.get('user_email') or (
            st.session_state.get('user_info', {}).get('email') if st.session_state.get('user_info') else None
        )
        st.session_state['portfolio_access_level'] = get_portfolio_access_level(portfolio_id, _user_email)
    except Exception:
        st.session_state['portfolio_access_level'] = None

if portfolio_id:
    # Show portfolio info
    portfolios_df = get_user_portfolios()
    portfolio_info = portfolios_df[portfolios_df['portfolio_id'] == portfolio_id]

    if not portfolio_info.empty:
        info = portfolio_info.iloc[0]

        # Determine write access for this portfolio
        has_write = st.session_state.get('portfolio_access_level') in ('owner', 'write')
        if not has_write:
            st.info("You have **read-only** access to this portfolio.")

        col1, col2, col3 = st.columns(3)

        with col1:
            st.metric("Portfolio", info['portfolio_name'])
        with col2:
            st.metric("Projects", int(info.get('project_count', 0)))
        with col3:
            summary = get_portfolio_summary(portfolio_id)
            st.metric("Data Periods", summary.get('total_periods', 0))

        # Status Dates Management Section
        st.markdown("---")
        st.markdown("### 📅 Status Dates / Data Periods")
        st.caption("View and manage available status dates for this portfolio")

        try:
            # Query available status dates with project counts
            status_dates_query = """
                SELECT
                    status_date,
                    COUNT(DISTINCT project_id) as project_count
                FROM project_status_report psr
                WHERE portfolio_id = ?
                GROUP BY status_date
                ORDER BY status_date DESC
            """

            status_dates_df = db.execute(status_dates_query, (portfolio_id,)).df()

            if status_dates_df.empty:
                st.info("📭 No status dates found for this portfolio")
            else:
                # Format dates for display
                status_dates_df['Data Date'] = pd.to_datetime(status_dates_df['status_date']).dt.strftime('%d-%b-%Y')
                status_dates_df['Projects'] = status_dates_df['project_count'].astype(int)

                # Display table with selection
                st.dataframe(
                    status_dates_df[['Data Date', 'Projects']],
                    hide_index=True,
                    width='stretch',
                    column_config={
                        "Data Date": st.column_config.TextColumn("Data Date", width="medium"),
                        "Projects": st.column_config.NumberColumn("Projects", width="small")
                    }
                )

                # Delete status date section (write access required)
                if has_write:
                    st.markdown("#### 🗑️ Delete Status Date")
                    st.caption("Remove a status date and all its associated project data")

                    col_select, col_delete = st.columns([3, 1])

                    with col_select:
                        # Create options from status_dates_df
                        date_options = status_dates_df['Data Date'].tolist()
                        selected_date_display = st.selectbox(
                            "Select Status Date to Delete",
                            options=date_options,
                            key="delete_status_date_select"
                        )

                    with col_delete:
                        st.write("")  # Spacing
                        st.write("")  # Spacing
                        if st.button("🗑️ Delete", type="secondary", key="delete_status_date_btn"):
                            st.session_state['confirm_delete_status_date'] = selected_date_display
                            st.rerun()

                    # Confirmation dialog
                    if st.session_state.get('confirm_delete_status_date'):
                        selected_display = st.session_state['confirm_delete_status_date']

                        # Find the actual date value
                        selected_row = status_dates_df[status_dates_df['Data Date'] == selected_display]
                        if not selected_row.empty:
                            actual_date = selected_row.iloc[0]['status_date']
                            num_projects = int(selected_row.iloc[0]['project_count'])

                            st.warning(f"⚠️ **Confirm Deletion**")
                            st.markdown(f"**Status Date:** {selected_display}")
                            st.markdown(f"**Projects Affected:** {num_projects}")
                            st.markdown("""
                            **This will delete:**
                            - All status reports for this date
                            - All calculated EVM results for this date
                            - This action cannot be undone
                            """)

                            col_confirm, col_cancel = st.columns(2)

                            with col_confirm:
                                if st.button("✅ Yes, Delete This Status Date", type="primary", key="confirm_delete_status"):
                                    try:
                                        # Delete status reports for this date
                                        delete_query = """
                                            DELETE FROM project_status_report
                                            WHERE portfolio_id = ? AND status_date = ?
                                        """
                                        db.execute(delete_query, (portfolio_id, actual_date))

                                        st.success(f"✅ Status date {selected_display} deleted successfully ({num_projects} project records removed)")
                                        del st.session_state['confirm_delete_status_date']
                                        st.rerun()
                                    except Exception as e:
                                        st.error(f"❌ Error deleting status date: {e}")

                            with col_cancel:
                                if st.button("❌ Cancel", key="cancel_delete_status"):
                                    del st.session_state['confirm_delete_status_date']
                                    st.rerun()

        except Exception as e:
            st.error(f"❌ Error loading status dates: {e}")

        # Run Batch EVM Section
        st.markdown("---")
        st.markdown("### ⚡ Run Batch EVM Calculations")
        st.caption("Calculate or recalculate EVM metrics for all projects in this portfolio")

        # Check if we have projects for this portfolio
        if not status_dates_df.empty:
            # Calculation scope selector
            st.markdown("#### Calculation Scope")
            calc_scope = st.radio(
                "Select calculation scope",
                options=["Single Date", "All Dates"],
                index=0,
                horizontal=True,
                help="Choose whether to calculate for one status date or all available dates"
            )

            if calc_scope == "Single Date":
                # Let user select which status date to calculate for
                st.markdown("**Select Status Date:**")

                calc_date_options = status_dates_df['Data Date'].tolist()
                selected_calc_date = st.selectbox(
                    "Status Date",
                    options=calc_date_options,
                    key="batch_calc_status_date",
                    help="Select the status date to run EVM calculations for"
                )

                # Find the actual date value
                calc_row = status_dates_df[status_dates_df['Data Date'] == selected_calc_date]
                if not calc_row.empty:
                    calc_status_date = calc_row.iloc[0]['status_date']
                    num_calc_projects = int(calc_row.iloc[0]['project_count'])

                    st.info(f"📊 **{num_calc_projects} projects** will be calculated for status date **{selected_calc_date}**")
                else:
                    calc_status_date = None
                    num_calc_projects = 0
            else:
                # All dates mode
                total_dates = len(status_dates_df)
                total_records = status_dates_df['project_count'].sum()
                st.info(f"📊 **{int(total_records)} total project records** across **{total_dates} status dates** will be calculated")
                calc_status_date = "all"  # Special marker for all dates mode
                num_calc_projects = int(total_records)

            # Calculation mode (common for both single and all dates)
            calc_mode = st.radio(
                "Calculation Mode",
                options=["Portfolio Settings", "Project Settings"],
                index=0,
                key="batch_calc_mode_portfolio",
                horizontal=True,
                help="""
**Portfolio Settings**: Use portfolio default settings for all projects (from Portfolio Settings tab)

**Project Settings**: Use each project's own settings (falls back to portfolio defaults if project settings are blank)
                """
            )

            if calc_mode == "Portfolio Settings":
                st.info("📊 All projects will use portfolio default settings")
            else:
                st.info("🎯 Each project will use its own settings (or portfolio defaults as fallback)")

            # Run button (requires write access)
            button_text = "🚀 **Calculate All Dates**" if calc_scope == "All Dates" else "🚀 **Run Batch EVM Calculations**"
            if not has_write:
                st.warning("Write access required to run batch EVM calculations.")
            elif st.button(button_text, type="primary", key="run_batch_evm_portfolio"):
                try:
                    # Load portfolio settings
                    from utils.portfolio_settings import load_portfolio_settings
                    portfolio_settings = load_portfolio_settings(portfolio_id)

                    # Extract EVM parameters
                    curve_type = portfolio_settings.get('curve_type', 'linear')
                    alpha = float(portfolio_settings.get('alpha', 2.0))
                    beta = float(portfolio_settings.get('beta', 2.0))
                    inflation_rate = float(portfolio_settings.get('inflation_rate', 0.0))

                    # Get projects from database
                    from services.data_service import data_manager
                    adapter = data_manager.get_data_adapter()

                    # Determine dates to process
                    if calc_status_date == "all":
                        # Process all dates
                        dates_to_process = []
                        for _, row in status_dates_df.iterrows():
                            dates_to_process.append({
                                'date': row['status_date'],
                                'display': row['Data Date'],
                                'count': int(row['project_count'])
                            })
                        st.info(f"🔄 Processing {len(dates_to_process)} status dates...")
                    else:
                        # Process single date
                        dates_to_process = [{
                            'date': calc_status_date,
                            'display': selected_calc_date,
                            'count': num_calc_projects
                        }]

                    # Process each date
                    total_updated = 0
                    batch_results = None  # Initialize to avoid NameError
                    progress_bar = st.progress(0)
                    status_text = st.empty()

                    for idx, date_info in enumerate(dates_to_process):
                        current_date = date_info['date']
                        display_date = date_info['display']
                        project_count = date_info['count']

                        # Update progress
                        progress = (idx + 1) / len(dates_to_process)
                        progress_bar.progress(progress)
                        status_text.text(f"Processing {display_date} ({idx + 1}/{len(dates_to_process)})...")

                        with st.spinner(f"⚡ Loading projects for {display_date}..."):
                            projects_df = adapter.get_projects_for_analysis(
                                portfolio_id=portfolio_id,
                                status_date=current_date
                            )

                        if projects_df.empty:
                            st.warning(f"⚠️ No projects found for {display_date}")
                            continue

                        st.info(f"✓ Loaded {len(projects_df)} projects for {display_date}")

                        # Set up column mapping
                        column_mapping = {
                            'pid_col': 'Project ID',
                            'pname_col': 'Project',
                            'org_col': 'Organization',
                            'pm_col': 'Project Manager',
                            'bac_col': 'BAC',
                            'ac_col': 'AC',
                            'st_col': 'Plan Start',
                            'fn_col': 'Plan Finish',
                            'cp_col': 'Completion %'
                        }

                        # Add optional columns if they exist
                        df_columns = projects_df.columns.tolist()
                        if 'Manual_PV' in df_columns:
                            column_mapping['pv_col'] = 'Manual_PV'
                        if 'Manual_EV' in df_columns:
                            column_mapping['ev_col'] = 'Manual_EV'
                        if 'Curve Type' in df_columns or 'curve_type' in df_columns:
                            column_mapping['curve_type_col'] = 'Curve Type' if 'Curve Type' in df_columns else 'curve_type'
                        if 'Alpha' in df_columns:
                            column_mapping['alpha_col'] = 'Alpha'
                        if 'Beta' in df_columns:
                            column_mapping['beta_col'] = 'Beta'
                        if 'Inflation Rate' in df_columns:
                            column_mapping['inflation_rate_col'] = 'Inflation Rate'

                        # Run batch EVM calculation for this date
                        from core.evm_engine import perform_batch_calculation
                        mode_str = 'global' if calc_mode == "Portfolio Settings" else 'project'

                        batch_results = perform_batch_calculation(
                            projects_df, column_mapping,
                            curve_type, alpha, beta, current_date, inflation_rate,  # Use current_date
                            mode=mode_str
                        )

                        # Save results to database
                        try:
                            updated = db_manager.save_batch_evm_results(
                                portfolio_id=portfolio_id,
                                status_date=current_date,  # Use current_date
                                results_df=batch_results
                            )
                            total_updated += updated

                            # Store in session state (last calculated date)
                            st.session_state.batch_results = batch_results
                            st.session_state.batch_results_ready = True

                        except Exception as e:
                            st.error(f"❌ Error saving EVM results for {display_date}: {e}")
                            import traceback
                            st.code(traceback.format_exc())

                    # Clear progress indicators
                    progress_bar.empty()
                    status_text.empty()

                    # Show final summary
                    if calc_status_date == "all":
                        st.success(f"✅ **All calculations complete!** Processed {len(dates_to_process)} status dates, updated {total_updated} total project records")
                    else:
                        if batch_results is not None and not batch_results.empty:
                            st.success(f"✅ EVM calculations completed for {len(batch_results)} projects!")
                            st.success(f"✅ Saved EVM results for {total_updated} projects to database")
                        elif total_updated > 0:
                            st.success(f"✅ Saved EVM results for {total_updated} projects to database")
                        else:
                            st.warning(f"⚠️ No projects were found to calculate for the selected date")

                except Exception as e:
                    st.error(f"❌ Error running batch calculation: {e}")
                    import traceback
                    st.code(traceback.format_exc())
        else:
            st.info("📭 No status dates available. Load project data first in File Management.")

        # Configuration Summary
        st.markdown("---")
        st.markdown("### 📋 Portfolio Configuration Summary")

        col1, col2, col3, col4 = st.columns(4)

        with col1:
            # Portfolio info - get portfolio name
            portfolio_query = "SELECT portfolio_name FROM portfolio WHERE portfolio_id = ?"
            portfolio_result = db.execute(portfolio_query, (portfolio_id,)).fetchone()
            current_portfolio_name = portfolio_result[0] if portfolio_result else "Unknown"

            st.success(f"✅ Portfolio: {current_portfolio_name}")
            st.caption(f"ID: {portfolio_id}")

        with col2:
            # Data status
            if not status_dates_df.empty:
                st.success(f"✅ Data: {len(status_dates_df)} status dates")
                total_projects = status_dates_df['project_count'].sum()
                st.caption(f"{int(total_projects)} total project records")
            else:
                st.warning("⚠️ No data loaded")

        with col3:
            # Settings status
            from utils.portfolio_settings import load_portfolio_settings
            portfolio_settings = load_portfolio_settings(portfolio_id)

            curve_type = portfolio_settings.get('curve_type', 'linear')
            currency = portfolio_settings.get('currency_symbol', '$')
            st.success("✅ Settings: Configured")
            st.caption(f"{curve_type} curve, {currency} currency")

        with col4:
            # LLM status
            llm_config = portfolio_settings.get('llm_config', {})
            if llm_config.get('provider'):
                st.success("✅ LLM: Configured")
                st.caption(f"{llm_config.get('provider', 'N/A')} - {llm_config.get('model', 'N/A')}")
            else:
                st.info("💡 LLM: Not configured")
                st.caption("Configure in Portfolio Settings")

st.markdown("---")


def create_portfolio_ui():
    """UI for creating a new portfolio"""
    st.markdown("### ➕ Create New Portfolio")

    with st.form("create_portfolio_form"):
        col1, col2 = st.columns(2)

        with col1:
            name = st.text_input("Portfolio Name *", placeholder="e.g., Digital Transformation 2024")
            managing_org = st.text_input("Managing Organization *", placeholder="e.g., PMO")

        with col2:
            manager = st.text_input("Portfolio Manager *", placeholder="e.g., John Doe")

            # Get list of users for owner selection
            try:
                users_df = db.execute("SELECT user_id, display_name, email FROM app_user WHERE is_active = TRUE ORDER BY display_name").df()
                if users_df.empty:
                    st.warning("⚠️ No users found. Please contact administrator.")
                    owner_user_id = None
                else:
                    user_options = {
                        f"{row['display_name']} ({row['email']})": row['user_id']
                        for _, row in users_df.iterrows()
                    }

                    # Find current user's index for default selection
                    current_user_display = None
                    for key, uid in user_options.items():
                        if uid == current_user_id:
                            current_user_display = key
                            break

                    default_index = list(user_options.keys()).index(current_user_display) if current_user_display else 0

                    selected_user = st.selectbox(
                        "Portfolio Owner *",
                        options=list(user_options.keys()),
                        index=default_index,
                        help="The user who owns this portfolio (defaults to you)"
                    )
                    # Convert numpy.int64 to Python int for DuckDB
                    owner_user_id = int(user_options[selected_user]) if user_options else None
            except Exception as e:
                st.error(f"Error loading users: {e}")
                owner_user_id = current_user_id  # Fallback to current user

        description = st.text_area("Description", placeholder="Optional description of the portfolio")

        submitted = st.form_submit_button("🚀 Create Portfolio", type="primary")

        if submitted:
            if not name or not managing_org or not manager:
                st.error("❌ Please fill in all required fields (*)")
            elif not owner_user_id:
                st.error("❌ Please select a portfolio owner")
            else:
                try:
                    portfolio_id = db_manager.create_portfolio(
                        name=name,
                        managing_org=managing_org,
                        manager=manager,
                        owner_user_id=owner_user_id
                    )

                    # Update description if provided
                    if description:
                        db.execute("""
                            UPDATE portfolio
                            SET description = ?
                            WHERE portfolio_id = ?
                        """, (description, portfolio_id))

                    st.success(f"✅ Portfolio '{name}' created successfully (ID: {portfolio_id})")
                    st.rerun()

                except Exception as e:
                    st.error(f"❌ Error creating portfolio: {e}")


def list_portfolios_ui():
    """UI for listing all portfolios in table format"""
    st.markdown("### 📋 Existing Portfolios")

    try:
        # Get all portfolios with owner information from portfolio_ownership table
        query = """
            SELECT
                p.portfolio_id,
                p.portfolio_name,
                p.managing_organization,
                p.portfolio_manager,
                p.description,
                p.created_at as created_date,
                p.is_active,
                po.owner_user_id,
                u.display_name as owner_name,
                u.email as owner_email,
                COUNT(DISTINCT pr.project_id) as project_count
            FROM portfolio p
            LEFT JOIN portfolio_ownership po ON p.portfolio_id = po.portfolio_id
            LEFT JOIN app_user u ON po.owner_user_id = u.user_id
            LEFT JOIN project pr ON p.portfolio_id = pr.portfolio_id AND pr.is_active = TRUE
            WHERE p.is_active = TRUE
            GROUP BY p.portfolio_id, p.portfolio_name, p.managing_organization,
                     p.portfolio_manager, p.description, p.created_at, p.is_active, po.owner_user_id,
                     u.display_name, u.email
            ORDER BY p.created_at DESC
        """

        portfolios_df = db.execute(query).df()

        if portfolios_df.empty:
            st.info("📭 No portfolios found. Create your first portfolio above!")
            return

        # Display summary statistics
        col1, col2, col3 = st.columns(3)

        with col1:
            st.markdown(f"""
            <div class="stat-box">
                <div class="stat-number">{len(portfolios_df)}</div>
                <div class="stat-label">Total Portfolios</div>
            </div>
            """, unsafe_allow_html=True)

        with col2:
            total_projects = portfolios_df['project_count'].sum()
            st.markdown(f"""
            <div class="stat-box">
                <div class="stat-number">{int(total_projects)}</div>
                <div class="stat-label">Total Projects</div>
            </div>
            """, unsafe_allow_html=True)

        with col3:
            avg_projects = portfolios_df['project_count'].mean()
            st.markdown(f"""
            <div class="stat-box">
                <div class="stat-number">{avg_projects:.1f}</div>
                <div class="stat-label">Avg Projects/Portfolio</div>
            </div>
            """, unsafe_allow_html=True)

        st.markdown("---")

        # Get list of users for owner selection
        users_df = db.execute("SELECT user_id, display_name, email FROM app_user WHERE is_active = TRUE ORDER BY display_name").df()
        # Convert numpy.int64 to Python int immediately to avoid DuckDB errors
        user_options = {
            f"{row['display_name']} ({row['email']})": int(row['user_id'])
            for _, row in users_df.iterrows()
        }

        # Prepare table data
        table_data = []
        for _, portfolio in portfolios_df.iterrows():
            created_date = portfolio['created_date']
            if isinstance(created_date, str):
                created_date = datetime.fromisoformat(created_date)

            table_data.append({
                'ID': int(portfolio['portfolio_id']),
                'Portfolio Name': portfolio['portfolio_name'],
                'Organization': portfolio['managing_organization'],
                'Manager': portfolio['portfolio_manager'],
                'Owner': f"{portfolio['owner_name']} ({portfolio['owner_email']})" if portfolio['owner_name'] else "N/A",
                'Projects': int(portfolio['project_count']),
                'Created': created_date.strftime('%Y-%m-%d')
            })

        display_df = pd.DataFrame(table_data)

        # Display table
        st.dataframe(
            display_df,
            hide_index=True,
            width='stretch',
            column_config={
                "ID": st.column_config.NumberColumn("ID", width="small"),
                "Portfolio Name": st.column_config.TextColumn("Portfolio Name", width="medium"),
                "Organization": st.column_config.TextColumn("Organization", width="medium"),
                "Manager": st.column_config.TextColumn("Manager", width="medium"),
                "Owner": st.column_config.TextColumn("Owner", width="medium"),
                "Projects": st.column_config.NumberColumn("Projects", width="small"),
                "Created": st.column_config.TextColumn("Created", width="small")
            }
        )

        st.markdown("---")

        # Portfolio Details Section Below Table
        st.markdown("### 📝 Portfolio Details - Add / Edit / Delete")

        # Select portfolio to work with
        portfolio_options = ["-- Select Portfolio --"] + [f"{row['portfolio_id']}: {row['portfolio_name']}" for _, row in portfolios_df.iterrows()]
        selected_option = st.selectbox(
            "Select Portfolio to View/Edit",
            options=portfolio_options,
            key="selected_portfolio_for_edit"
        )

        if selected_option != "-- Select Portfolio --":
            # Extract portfolio ID from selection
            selected_portfolio_id = int(selected_option.split(":")[0])

            # Set current portfolio in session state for other pages
            st.session_state.current_portfolio_id = selected_portfolio_id

            portfolio = portfolios_df[portfolios_df['portfolio_id'] == selected_portfolio_id].iloc[0]

            # Display current details in an editable form
            with st.form("portfolio_details_form"):
                st.markdown(f"#### Editing Portfolio: {portfolio['portfolio_name']}")

                col1, col2 = st.columns(2)

                with col1:
                    edit_name = st.text_input("Portfolio Name *", value=portfolio['portfolio_name'])
                    edit_org = st.text_input("Managing Organization *", value=portfolio['managing_organization'])
                    edit_manager = st.text_input("Portfolio Manager *", value=portfolio['portfolio_manager'])

                with col2:
                    # Owner selection
                    current_owner_display = f"{portfolio['owner_name']} ({portfolio['owner_email']})" if portfolio['owner_name'] else None
                    current_owner_index = list(user_options.keys()).index(current_owner_display) if current_owner_display in user_options else 0

                    selected_owner_display = st.selectbox(
                        "Portfolio Owner *",
                        options=list(user_options.keys()),
                        index=current_owner_index
                    )
                    # Convert numpy.int64 to Python int for DuckDB
                    new_owner_user_id = int(user_options[selected_owner_display])

                edit_desc = st.text_area("Description", value=portfolio['description'] or "")

                # Action buttons
                col_save, col_delete = st.columns(2)

                with col_save:
                    save = st.form_submit_button("💾 Save Changes", type="primary")

                with col_delete:
                    delete = st.form_submit_button("🗑️ Delete Portfolio", type="secondary")

                if save:
                    if not edit_name or not edit_org or not edit_manager:
                        st.error("❌ Please fill in all required fields (*)")
                    else:
                        try:
                            # Convert numpy types to Python types for DuckDB
                            portfolio_id = int(portfolio['portfolio_id'])

                            # Verify the new owner exists
                            owner_exists = db.fetch_one(
                                "SELECT user_id FROM app_user WHERE user_id = ? AND is_active = TRUE",
                                (new_owner_user_id,)
                            )

                            if not owner_exists:
                                st.error(f"❌ Selected owner (ID: {new_owner_user_id}) does not exist in the database")
                                return

                            # Update using portfolio_ownership table for owner (no FK constraint issues!)
                            try:
                                conn = db.get_connection()

                                # Update portfolio details
                                conn.execute("""
                                    UPDATE portfolio
                                    SET portfolio_name = ?,
                                        managing_organization = ?,
                                        portfolio_manager = ?,
                                        description = ?
                                    WHERE portfolio_id = ?
                                """, (edit_name, edit_org, edit_manager, edit_desc, portfolio_id))

                                # Update owner in portfolio_ownership table (separate table - no FK issues!)
                                # Check if ownership record exists
                                existing_ownership = db.fetch_one(
                                    "SELECT 1 FROM portfolio_ownership WHERE portfolio_id = ?",
                                    (portfolio_id,)
                                )

                                if existing_ownership:
                                    # Update existing ownership
                                    conn.execute("""
                                        UPDATE portfolio_ownership
                                        SET owner_user_id = ?, assigned_at = now()
                                        WHERE portfolio_id = ?
                                    """, (new_owner_user_id, portfolio_id))
                                else:
                                    # Insert new ownership record
                                    conn.execute("""
                                        INSERT INTO portfolio_ownership (portfolio_id, owner_user_id, is_active, assigned_at)
                                        VALUES (?, ?, TRUE, now())
                                    """, (portfolio_id, new_owner_user_id))

                                st.success(f"✅ Portfolio '{edit_name}' updated successfully")
                                st.rerun()

                            except Exception as update_err:
                                st.error(f"Update operation failed: {update_err}")
                                import traceback
                                st.code(traceback.format_exc())
                        except Exception as e:
                            st.error(f"❌ Error updating portfolio: {e}")
                            import traceback
                            st.code(traceback.format_exc())

                if delete:
                    st.session_state['confirming_delete_portfolio'] = selected_portfolio_id
                    st.rerun()

        # Delete confirmation (outside form)
        if st.session_state.get('confirming_delete_portfolio'):
            delete_portfolio_id = st.session_state['confirming_delete_portfolio']
            delete_portfolio = portfolios_df[portfolios_df['portfolio_id'] == delete_portfolio_id].iloc[0]

            st.markdown("---")
            st.warning(f"⚠️ **Confirm Deletion**")
            st.markdown(f"**Portfolio:** {delete_portfolio['portfolio_name']}")
            st.markdown(f"**Projects:** {int(delete_portfolio['project_count'])}")
            st.info(
                "**What will be deleted:**\n"
                "- Portfolio (soft-deleted, can be purged later)\n"
                "- All projects in this portfolio (soft-deleted)\n"
                "- All strategic factors for this portfolio (soft-deleted)\n"
                "- Project SDG associations and factor scores (removed)\n\n"
                "**What will be preserved for audit:**\n"
                "- Project baselines and status reports (historical data)"
            )

            col_confirm, col_cancel = st.columns(2)

            with col_confirm:
                if st.button("✅ Yes, Delete Portfolio", key="confirm_delete_portfolio", type="primary"):
                    try:
                        # Use the new delete_portfolio method with cascade
                        db_manager = DatabaseDataManager()
                        deleted_summary = db_manager.delete_portfolio(delete_portfolio_id, cascade=True)

                        # Show summary of what was soft-deleted
                        summary_msg = f"✅ Portfolio '{delete_portfolio['portfolio_name']}' soft-deleted successfully!\n\n"
                        summary_msg += "**Marked as inactive (is_active = FALSE):**\n"
                        summary_msg += f"- Portfolio: {deleted_summary['portfolio']}\n"
                        summary_msg += f"- Projects: {deleted_summary['projects']}\n"
                        summary_msg += f"- Portfolio Factors: {deleted_summary['portfolio_factors']}\n"
                        summary_msg += f"- Portfolio Access: {deleted_summary.get('portfolio_access', 0)}\n"
                        summary_msg += f"- Portfolio Ownership: {deleted_summary['portfolio_ownership']}\n"
                        summary_msg += f"- Project Baselines: {deleted_summary['project_baseline']}\n"
                        summary_msg += f"- Project Status Reports: {deleted_summary['project_status_report']}\n"
                        summary_msg += f"- Project SDG Associations: {deleted_summary['project_sdg']}\n"
                        summary_msg += f"- Project Factor Scores: {deleted_summary['project_factor_score']}\n\n"
                        summary_msg += "💡 **Note:** Records are soft-deleted (hidden). Use Database Diagnostics to permanently purge."

                        st.success(summary_msg)
                        del st.session_state['confirming_delete_portfolio']
                        st.rerun()
                    except Exception as e:
                        st.error(f"❌ Error deleting portfolio: {e}")

            with col_cancel:
                if st.button("❌ Cancel", key="cancel_delete_portfolio"):
                    del st.session_state['confirming_delete_portfolio']
                    st.rerun()

    except Exception as e:
        st.error(f"❌ Error loading portfolios: {e}")


# Main page layout
tab1, tab_access, tab3, tab4 = st.tabs(["📂 Portfolios", "🔐 Portfolio Access", "⚙️ Portfolio Settings", "ℹ️ Help"])

with tab1:
    create_portfolio_ui()
    st.markdown("---")
    list_portfolios_ui()

with tab_access:
    st.markdown("### 🔐 Portfolio Access Control")
    st.caption("Grant other users read-only or read/write access to your portfolios")

    from utils.portfolio_access import (
        get_portfolio_access_level, list_portfolio_users,
        add_portfolio_access, remove_portfolio_access, update_portfolio_access_level
    )

    # Portfolio selector within this tab
    try:
        access_portfolios_df = db.execute("""
            SELECT p.portfolio_id, p.portfolio_name
            FROM portfolio p
            WHERE p.is_active = TRUE
            ORDER BY p.portfolio_name
        """).df()
    except Exception as e:
        st.error(f"Error loading portfolios: {e}")
        access_portfolios_df = pd.DataFrame()

    if access_portfolios_df.empty:
        st.warning("No portfolios found. Create a portfolio first.")
    else:
        access_portfolio_options = {
            row['portfolio_name']: int(row['portfolio_id'])
            for _, row in access_portfolios_df.iterrows()
        }

        selected_access_portfolio_name = st.selectbox(
            "Select Portfolio to Manage Access",
            options=list(access_portfolio_options.keys()),
            key="access_tab_portfolio_select"
        )
        access_portfolio_id = access_portfolio_options[selected_access_portfolio_name]

        # Check current user's access level for this portfolio
        _access_email = st.session_state.get('user_email') or (
            st.session_state.get('user_info', {}).get('email') if st.session_state.get('user_info') else None
        )
        current_level = get_portfolio_access_level(access_portfolio_id, _access_email)
        is_owner_or_admin = current_level == 'owner'

        st.info(f"Portfolio: **{selected_access_portfolio_name}** (ID: {access_portfolio_id})")

        # Show current users with access in a table
        st.markdown("#### Current Access")
        access_df = list_portfolio_users(access_portfolio_id)

        if access_df.empty:
            st.info("No users found with access to this portfolio.")
        else:
            # Display as a formatted table
            display_access = access_df[['display_name', 'email', 'access_level']].copy()
            display_access.columns = ['Name', 'Email', 'Access Level']
            display_access['Access Level'] = display_access['Access Level'].str.title()
            st.dataframe(
                display_access,
                hide_index=True,
                width='stretch',
                column_config={
                    "Name": st.column_config.TextColumn("Name", width="medium"),
                    "Email": st.column_config.TextColumn("Email", width="medium"),
                    "Access Level": st.column_config.TextColumn("Access Level", width="small"),
                }
            )

            # Edit/remove controls for each non-owner user (only if current user is owner/admin)
            if is_owner_or_admin:
                non_owner_users = access_df[access_df['access_level'] != 'owner']
                if not non_owner_users.empty:
                    st.markdown("#### Modify User Access")

                    for idx, row in non_owner_users.iterrows():
                        with st.container():
                            col_user, col_level, col_update, col_remove = st.columns([3, 2, 1, 1])

                            with col_user:
                                st.write(f"**{row['display_name']}** ({row['email']})")

                            with col_level:
                                new_level = st.selectbox(
                                    "Level",
                                    options=['read', 'write'],
                                    index=0 if row['access_level'] == 'read' else 1,
                                    key=f"access_level_{access_portfolio_id}_{row['user_id']}",
                                    label_visibility="collapsed"
                                )

                            with col_update:
                                if new_level != row['access_level']:
                                    if st.button("Save", key=f"update_access_{access_portfolio_id}_{row['user_id']}", type="primary"):
                                        update_portfolio_access_level(
                                            access_portfolio_id, int(row['user_id']), new_level
                                        )
                                        st.rerun()

                            with col_remove:
                                if st.button("Remove", key=f"remove_access_{access_portfolio_id}_{row['user_id']}", type="secondary"):
                                    remove_portfolio_access(access_portfolio_id, int(row['user_id']))
                                    st.success(f"Removed access for {row['display_name']}")
                                    st.rerun()

        # Add user form (only for owners/admins)
        if is_owner_or_admin:
            st.markdown("---")
            st.markdown("#### Grant Access to User")

            # Get users not already in the access list
            try:
                all_users_df = db.execute(
                    "SELECT user_id, display_name, email FROM app_user WHERE is_active = TRUE ORDER BY display_name"
                ).df()

                # Filter out users who already have access
                existing_user_ids = set()
                if not access_df.empty:
                    existing_user_ids = set(access_df['user_id'].astype(int).tolist())

                available_users = all_users_df[~all_users_df['user_id'].isin(existing_user_ids)]

                if available_users.empty:
                    st.info("All registered users already have access to this portfolio.")
                else:
                    with st.form("add_access_form"):
                        user_options = {
                            f"{row['display_name']} ({row['email']})": int(row['user_id'])
                            for _, row in available_users.iterrows()
                        }

                        selected_user = st.selectbox(
                            "Select User",
                            options=list(user_options.keys())
                        )

                        access_level = st.selectbox(
                            "Access Level",
                            options=['Read Only', 'Read & Write'],
                            help="Read Only: Can view portfolio data. Read & Write: Can also modify data."
                        )

                        submitted = st.form_submit_button("Grant Access", type="primary")

                        if submitted:
                            target_user_id = user_options[selected_user]
                            level = 'read' if access_level == 'Read Only' else 'write'
                            success = add_portfolio_access(
                                access_portfolio_id,
                                target_user_id,
                                level,
                                granted_by_user_id=current_user_id
                            )
                            if success:
                                st.success(f"Granted {access_level} access to {selected_user}")
                                st.rerun()
                            else:
                                st.error("Failed to grant access. User may already be the owner.")

            except Exception as e:
                st.error(f"Error loading users: {e}")
        elif current_level:
            st.info("Only portfolio owners can manage access.")
        else:
            st.warning("You do not have access to this portfolio.")

with tab3:
    # Portfolio Settings Tab
    st.markdown("### ⚙️ Portfolio Settings")
    st.caption("Configure default settings for this portfolio")

    # Check if a portfolio is selected
    selected_portfolio_id = st.session_state.get('current_portfolio_id')

    if not selected_portfolio_id:
        st.warning("⚠️ Please select a portfolio from the dropdown above first")
        st.info("Go to the **Portfolios** tab and select a portfolio to configure its settings")
    else:
        # Import settings functions
        from utils.portfolio_settings import load_portfolio_settings, save_portfolio_settings

        # Load current settings
        current_settings = load_portfolio_settings(selected_portfolio_id)

        # Get portfolio name for display
        portfolio_query = "SELECT portfolio_name FROM portfolio WHERE portfolio_id = ?"
        portfolio_name_result = db.fetch_one(portfolio_query, (selected_portfolio_id,))
        portfolio_name = portfolio_name_result[0] if portfolio_name_result else "Unknown"

        st.info(f"📂 Configuring settings for: **{portfolio_name}** (ID: {selected_portfolio_id})")

        # Create two columns for the settings sections
        st.markdown("---")

        # Section 1: EVM Controls
        st.markdown("### 📊 EVM Calculation Controls")
        st.caption("Default settings for EVM calculations in this portfolio")

        with st.form("controls_form"):
            # Curve settings
            curve_value = current_settings.get('curve_type', 'linear').lower()
            curve_index = 0 if curve_value == 'linear' else 1

            curve_type = st.selectbox(
                "Curve Type (PV)",
                ["Linear", "S-Curve"],
                index=curve_index
            )

            alpha = 2.0
            beta = 2.0

            if curve_type == "S-Curve":
                col1, col2 = st.columns(2)
                with col1:
                    alpha = st.number_input("S-Curve α", min_value=0.1, max_value=10.0,
                                          value=float(current_settings.get('alpha', 2.0)), step=0.1)
                with col2:
                    beta = st.number_input("S-Curve β", min_value=0.1, max_value=10.0,
                                         value=float(current_settings.get('beta', 2.0)), step=0.1)

            # Currency settings
            col1, col2 = st.columns(2)
            with col1:
                currency_symbol = st.text_input("Currency Symbol",
                                               value=current_settings.get('currency_symbol', '$'))
            with col2:
                currency_postfix = st.text_input("Currency Postfix",
                                                value=current_settings.get('currency_postfix', ''))

            # Additional settings
            col1, col2 = st.columns(2)
            with col1:
                date_format = st.selectbox(
                    "Date Format",
                    ["YYYY-MM-DD", "MM/DD/YYYY", "DD/MM/YYYY"],
                    index=["YYYY-MM-DD", "MM/DD/YYYY", "DD/MM/YYYY"].index(current_settings.get('date_format', 'YYYY-MM-DD'))
                )
            with col2:
                data_date = st.date_input(
                    "Default Data Date",
                    value=pd.to_datetime(current_settings.get('data_date', '2024-01-01')).date(),
                    min_value=datetime(1990, 1, 1).date(),
                    help="Default project data date for EVM calculations"
                )

            # Inflation rate
            inflation_rate = st.number_input(
                "Annual Inflation Rate (%)",
                min_value=0.0,
                max_value=100.0,
                value=float(current_settings.get('inflation_rate', 0.0)),
                step=0.1
            )

            # Budget Tier Configuration
            st.markdown("---")
            st.markdown("#### 🎯 Budget Tier Configuration")

            tier_config = current_settings.get('tier_config', {
                'cutoff_points': [4000, 8000, 15000],
                'tier_names': ['Tier 1', 'Tier 2', 'Tier 3', 'Tier 4']
            })

            default_cutoffs = tier_config.get('cutoff_points', [4000, 8000, 15000])
            default_names = tier_config.get('tier_names', ['Tier 1', 'Tier 2', 'Tier 3', 'Tier 4'])

            # Cutoff points
            st.markdown("**Cutoff Points**")
            col1, col2, col3 = st.columns(3)
            with col1:
                cutoff1 = st.number_input(
                    f"Cutoff 1 ({currency_symbol})",
                    min_value=0.0,
                    value=float(default_cutoffs[0]),
                    step=1000.0,
                    help="Boundary between Tier 1 and Tier 2"
                )
            with col2:
                cutoff2 = st.number_input(
                    f"Cutoff 2 ({currency_symbol})",
                    min_value=cutoff1,
                    value=max(float(default_cutoffs[1]), cutoff1),
                    step=1000.0,
                    help="Boundary between Tier 2 and Tier 3"
                )
            with col3:
                cutoff3 = st.number_input(
                    f"Cutoff 3 ({currency_symbol})",
                    min_value=cutoff2,
                    value=max(float(default_cutoffs[2]), cutoff2),
                    step=1000.0,
                    help="Boundary between Tier 3 and Tier 4"
                )

            # Tier names
            st.markdown("**Tier Names**")
            col1, col2, col3, col4 = st.columns(4)
            with col1:
                tier1_name = st.text_input(
                    "Tier 1 Name",
                    value=default_names[0],
                    help=f"Projects < {currency_symbol}{cutoff1:,.0f}"
                )
            with col2:
                tier2_name = st.text_input(
                    "Tier 2 Name",
                    value=default_names[1],
                    help=f"Projects {currency_symbol}{cutoff1:,.0f} - {currency_symbol}{cutoff2:,.0f}"
                )
            with col3:
                tier3_name = st.text_input(
                    "Tier 3 Name",
                    value=default_names[2],
                    help=f"Projects {currency_symbol}{cutoff2:,.0f} - {currency_symbol}{cutoff3:,.0f}"
                )
            with col4:
                tier4_name = st.text_input(
                    "Tier 4 Name",
                    value=default_names[3],
                    help=f"Projects ≥ {currency_symbol}{cutoff3:,.0f}"
                )

            # Duration Tier Configuration
            st.markdown("---")
            st.markdown("#### ⏱️ Duration Tier Configuration")

            duration_tier_config = current_settings.get('duration_tier_config', {
                'cutoff_points': [6, 12, 24],
                'tier_names': ['Short', 'Medium', 'Long', 'Extra Long']
            })

            default_duration_cutoffs = duration_tier_config.get('cutoff_points', [6, 12, 24])
            default_duration_names = duration_tier_config.get('tier_names', ['Short', 'Medium', 'Long', 'Extra Long'])

            # Cutoff points (in months)
            st.markdown("**Cutoff Points (months)**")
            col1, col2, col3 = st.columns(3)
            with col1:
                duration_cutoff1 = st.number_input(
                    "Cutoff 1 (months)",
                    min_value=1,
                    value=int(default_duration_cutoffs[0]),
                    step=1,
                    help="Boundary between Tier 1 and Tier 2"
                )
            with col2:
                duration_cutoff2 = st.number_input(
                    "Cutoff 2 (months)",
                    min_value=duration_cutoff1,
                    value=max(int(default_duration_cutoffs[1]), duration_cutoff1),
                    step=1,
                    help="Boundary between Tier 2 and Tier 3"
                )
            with col3:
                duration_cutoff3 = st.number_input(
                    "Cutoff 3 (months)",
                    min_value=duration_cutoff2,
                    value=max(int(default_duration_cutoffs[2]), duration_cutoff2),
                    step=1,
                    help="Boundary between Tier 3 and Tier 4"
                )

            # Duration tier names
            st.markdown("**Duration Tier Names**")
            col1, col2, col3, col4 = st.columns(4)
            with col1:
                duration_tier1_name = st.text_input(
                    "Duration Tier 1 Name",
                    value=default_duration_names[0],
                    help=f"Projects < {duration_cutoff1} months"
                )
            with col2:
                duration_tier2_name = st.text_input(
                    "Duration Tier 2 Name",
                    value=default_duration_names[1],
                    help=f"Projects {duration_cutoff1} - {duration_cutoff2} months"
                )
            with col3:
                duration_tier3_name = st.text_input(
                    "Duration Tier 3 Name",
                    value=default_duration_names[2],
                    help=f"Projects {duration_cutoff2} - {duration_cutoff3} months"
                )
            with col4:
                duration_tier4_name = st.text_input(
                    "Duration Tier 4 Name",
                    value=default_duration_names[3],
                    help=f"Projects ≥ {duration_cutoff3} months"
                )

            # Save button
            st.markdown("---")
            save_controls = st.form_submit_button("💾 Save EVM Controls", type="primary")

            if save_controls:
                # Build settings dict
                settings = {
                    'curve_type': curve_type.lower(),
                    'alpha': alpha,
                    'beta': beta,
                    'currency_symbol': currency_symbol,
                    'currency_postfix': currency_postfix,
                    'date_format': date_format,
                    'data_date': data_date.strftime('%Y-%m-%d'),
                    'inflation_rate': inflation_rate,
                    'tier_config': {
                        'cutoff_points': [cutoff1, cutoff2, cutoff3],
                        'tier_names': [tier1_name, tier2_name, tier3_name, tier4_name],
                        'colors': ['#3498db', '#27ae60', '#f39c12', '#e74c3c']
                    },
                    'duration_tier_config': {
                        'cutoff_points': [duration_cutoff1, duration_cutoff2, duration_cutoff3],
                        'tier_names': [duration_tier1_name, duration_tier2_name, duration_tier3_name, duration_tier4_name],
                        'colors': ['#3498db', '#27ae60', '#f39c12', '#e74c3c']
                    },
                    'llm_config': current_settings.get('llm_config', {})  # Preserve LLM config
                }

                # Save to database
                try:
                    save_portfolio_settings(selected_portfolio_id, settings)
                    st.success("✅ EVM Controls saved successfully!")
                    st.rerun()
                except Exception as e:
                    st.error(f"❌ Error saving settings: {e}")

        # Section 2: LLM Provider Configuration
        st.markdown("---")
        st.markdown("### 🤖 LLM Configuration (Executive Reports & Chat)")
        st.caption("Configure AI provider for Executive Brief generation and chat")

        with st.form("llm_config_form"):
            llm_config = current_settings.get('llm_config', {})

            # Provider selection
            provider_options = ["OpenAI", "Gemini", "Claude", "Kimi"]
            saved_provider = llm_config.get('provider', 'OpenAI')
            provider_index = provider_options.index(saved_provider) if saved_provider in provider_options else 0

            provider = st.radio(
                "Choose Provider",
                provider_options,
                index=provider_index
            )

            # API Key input (persisted in portfolio settings)
            saved_api_key = llm_config.get('api_key', '')
            api_key_input = st.text_input(
                "API Key",
                value=saved_api_key,
                type="password",
                help="Enter your API key for the selected provider. The key is saved in portfolio settings."
            )

            # Model selection based on provider
            if provider == "OpenAI":
                openai_models = ["gpt-4o", "gpt-4o-mini", "gpt-4-turbo", "gpt-3.5-turbo"]
                saved_model = llm_config.get('model', 'gpt-4o-mini') if saved_provider == 'OpenAI' else 'gpt-4o-mini'
                model_index = openai_models.index(saved_model) if saved_model in openai_models else 1
                selected_model = st.selectbox("OpenAI Model", openai_models, index=model_index)
            elif provider == "Gemini":
                gemini_models = ["gemini-1.5-pro", "gemini-1.5-flash", "gemini-pro"]
                saved_model = llm_config.get('model', 'gemini-1.5-flash') if saved_provider == 'Gemini' else 'gemini-1.5-flash'
                model_index = gemini_models.index(saved_model) if saved_model in gemini_models else 1
                selected_model = st.selectbox("Gemini Model", gemini_models, index=model_index)
            elif provider == "Claude":
                claude_models = ["claude-3-5-sonnet-20241022", "claude-3-5-haiku-20241022", "claude-3-opus-20240229"]
                saved_model = llm_config.get('model', 'claude-3-5-sonnet-20241022') if saved_provider == 'Claude' else 'claude-3-5-sonnet-20241022'
                model_index = claude_models.index(saved_model) if saved_model in claude_models else 0
                selected_model = st.selectbox("Claude Model", claude_models, index=model_index)
            else:  # Kimi
                kimi_models = ["kimi-k2-0711-preview", "moonshot-v1-128k", "moonshot-v1-8k"]
                saved_model = llm_config.get('model', 'kimi-k2-0711-preview') if saved_provider == 'Kimi' else 'kimi-k2-0711-preview'
                model_index = kimi_models.index(saved_model) if saved_model in kimi_models else 0
                selected_model = st.selectbox("Kimi Model", kimi_models, index=model_index)

            # Model parameters
            st.markdown("**Model Parameters**")
            col1, col2 = st.columns(2)
            with col1:
                temperature = st.slider(
                    "Temperature",
                    min_value=0.0, max_value=1.0,
                    value=float(llm_config.get('temperature', 0.2)),
                    step=0.01,
                    help="Lower values = more focused/deterministic"
                )
            with col2:
                timeout = st.slider(
                    "Timeout (seconds)",
                    min_value=30, max_value=300,
                    value=int(llm_config.get('timeout', 60)),
                    step=5,
                    help="Maximum time to wait for LLM response"
                )

            # Save button
            st.markdown("---")
            save_llm = st.form_submit_button("💾 Save LLM Configuration", type="primary")

            if save_llm:
                # Build LLM config
                new_llm_config = {
                    'provider': provider,
                    'model': selected_model,
                    'temperature': temperature,
                    'timeout': timeout,
                    'api_key': api_key_input.strip(),
                    'has_api_key': bool(api_key_input.strip())
                }

                # Merge with existing settings
                updated_settings = current_settings.copy()
                updated_settings['llm_config'] = new_llm_config

                # Save to database
                try:
                    save_portfolio_settings(selected_portfolio_id, updated_settings)
                    st.success("✅ LLM Configuration saved successfully!")
                    st.rerun()
                except Exception as e:
                    st.error(f"❌ Error saving LLM configuration: {e}")

        # Section 3: Voice Configuration (Transcription & Read Aloud)
        st.markdown("---")
        st.markdown("### 🎤 Voice Configuration (Transcription & Read Aloud)")
        st.caption("Configure voice transcription and text-to-speech for Executive Brief Q&A")

        with st.form("voice_config_form"):
            voice_config = current_settings.get('voice_config', {})
            llm_config = current_settings.get('llm_config', {})

            # Enable Voice Features toggle
            voice_enabled = st.checkbox(
                "Enable Voice Features",
                value=voice_config.get('voice_enabled', False),
                help="When enabled, voice input and read-aloud features appear in Executive Brief Q&A"
            )

            # Provider selection
            voice_provider_options = ["OpenAI", "Replicate"]
            saved_voice_provider = voice_config.get('provider', 'OpenAI')
            voice_provider_index = voice_provider_options.index(saved_voice_provider) if saved_voice_provider in voice_provider_options else 0

            voice_provider = st.radio(
                "Voice Provider",
                voice_provider_options,
                index=voice_provider_index,
                horizontal=True,
                help="OpenAI (Whisper + TTS) is faster and recommended. Replicate is an alternative."
            )

            # Key sharing and input logic
            use_llm_key = False
            voice_openai_key = ''
            voice_replicate_key = ''

            if voice_provider == "OpenAI":
                # Show option to use LLM's OpenAI key if LLM is also OpenAI
                llm_is_openai = llm_config.get('provider') == 'OpenAI'
                llm_has_key = bool(llm_config.get('api_key', '').strip())

                if llm_is_openai and llm_has_key:
                    use_llm_key = st.checkbox(
                        "Use LLM's OpenAI API Key",
                        value=voice_config.get('use_llm_openai_key', True),
                        help="Use the same OpenAI API key configured for LLM above"
                    )
                    if use_llm_key:
                        st.info("Using the OpenAI API key from LLM Configuration above")
                    else:
                        voice_openai_key = st.text_input(
                            "OpenAI API Key for Voice",
                            value=voice_config.get('openai_api_key', ''),
                            type="password",
                            help="Separate OpenAI API key for voice features (Whisper ASR + TTS)"
                        )
                elif llm_is_openai and not llm_has_key:
                    st.warning("LLM is set to OpenAI but has no API key. Configure LLM API key above, or enter a separate key below.")
                    voice_openai_key = st.text_input(
                        "OpenAI API Key for Voice",
                        value=voice_config.get('openai_api_key', ''),
                        type="password",
                        help="OpenAI API key for voice features (Whisper ASR + TTS)"
                    )
                else:
                    st.info("LLM is not using OpenAI. Enter an OpenAI API key for voice features.")
                    voice_openai_key = st.text_input(
                        "OpenAI API Key for Voice",
                        value=voice_config.get('openai_api_key', ''),
                        type="password",
                        help="OpenAI API key for voice features (Whisper ASR + TTS)"
                    )
            else:  # Replicate
                voice_replicate_key = st.text_input(
                    "Replicate API Token",
                    value=voice_config.get('replicate_api_key', ''),
                    type="password",
                    help="Replicate API token for voice features. Get from replicate.com/account/api-tokens"
                )

            save_voice = st.form_submit_button("💾 Save Voice Configuration", type="primary")

            if save_voice:
                new_voice_config = {
                    'voice_enabled': voice_enabled,
                    'provider': voice_provider,
                    'use_llm_openai_key': use_llm_key,
                    'openai_api_key': voice_openai_key.strip() if voice_openai_key else '',
                    'replicate_api_key': voice_replicate_key.strip() if voice_replicate_key else ''
                }
                updated_settings = current_settings.copy()
                updated_settings['voice_config'] = new_voice_config
                try:
                    save_portfolio_settings(selected_portfolio_id, updated_settings)
                    st.success("✅ Voice configuration saved!")
                    st.rerun()
                except Exception as e:
                    st.error(f"❌ Error saving voice configuration: {e}")

        # Section 4: Infographic Configuration (AI Visual Summaries)
        st.markdown("---")
        st.markdown("### 🎨 Infographic Configuration (AI Visual Summaries)")
        st.caption("Configure AI-powered infographic generation for portfolio and project summaries")

        with st.form("infographic_config_form"):
            infographic_config = current_settings.get('infographic_config', {})
            voice_config = current_settings.get('voice_config', {})

            # Enable Infographics toggle
            infographic_enabled = st.checkbox(
                "Enable Infographics",
                value=infographic_config.get('enabled', False),
                help="When enabled, AI infographic generation appears in Executive Brief sections"
            )

            # Key sharing logic
            voice_has_replicate = bool(voice_config.get('replicate_api_key', '').strip())

            use_voice_key = False
            infographic_replicate_key = ''

            if voice_has_replicate:
                use_voice_key = st.checkbox(
                    "Use Voice's Replicate API Key",
                    value=infographic_config.get('use_voice_replicate_key', True),
                    help="Use the same Replicate API key configured for Voice above"
                )
                if use_voice_key:
                    st.info("Using the Replicate API key from Voice Configuration above")
                else:
                    infographic_replicate_key = st.text_input(
                        "Replicate API Token for Infographics",
                        value=infographic_config.get('replicate_api_key', ''),
                        type="password",
                        help="Separate Replicate API token for infographic generation"
                    )
            else:
                st.info("Voice is not using Replicate. Enter a Replicate API key for infographics.")
                infographic_replicate_key = st.text_input(
                    "Replicate API Token",
                    value=infographic_config.get('replicate_api_key', ''),
                    type="password",
                    help="Replicate API token for AI infographic generation. Get from replicate.com/account/api-tokens"
                )

            # Model selection
            model_options = {
                "Nano Banana Pro (Recommended)": "nano-banana",
                "FLUX Schnell (Fast)": "schnell",
                "FLUX Dev (High Quality)": "dev"
            }
            saved_model = infographic_config.get('model', 'nano-banana')
            # Find the display name for the saved model
            model_display_names = list(model_options.keys())
            model_values = list(model_options.values())
            model_index = model_values.index(saved_model) if saved_model in model_values else 0

            selected_model_display = st.selectbox(
                "Image Generation Model",
                options=model_display_names,
                index=model_index,
                help="Nano Banana Pro is fast and cost-effective. FLUX Dev produces higher quality but takes longer."
            )
            selected_infographic_model = model_options[selected_model_display]

            save_infographic = st.form_submit_button("💾 Save Infographic Configuration", type="primary")

            if save_infographic:
                new_infographic_config = {
                    'enabled': infographic_enabled,
                    'use_voice_replicate_key': use_voice_key,
                    'replicate_api_key': infographic_replicate_key.strip() if infographic_replicate_key else '',
                    'model': selected_infographic_model
                }
                updated_settings = current_settings.copy()
                updated_settings['infographic_config'] = new_infographic_config
                try:
                    save_portfolio_settings(selected_portfolio_id, updated_settings)
                    st.success("✅ Infographic configuration saved!")
                    st.rerun()
                except Exception as e:
                    st.error(f"❌ Error saving infographic configuration: {e}")

with tab4:
    st.markdown("""
    ## Portfolio Management Help

    ### 📂 Portfolios

    **What is a Portfolio?**
    A portfolio is a collection of related projects managed together. Portfolios help organize projects by:
    - Strategic initiative
    - Department or organization
    - Time period (e.g., "2024 Projects")
    - Portfolio manager

    **Creating a Portfolio:**
    1. Enter a descriptive name
    2. Specify the managing organization
    3. Assign a portfolio manager
    4. Select the portfolio owner (must be a database user)
    5. Optionally add a description

    **Managing Portfolios:**
    - **Edit**: Update portfolio details
    - **Delete**: Soft-delete portfolio (projects remain but are unlinked)
    - Projects can be assigned to portfolios during CSV upload or project creation

    ### 🔐 Portfolio Access

    **Sharing Portfolios:**
    - Portfolio owners can grant other users access to their portfolios
    - **Read Only**: User can view portfolio data but cannot modify it
    - **Read & Write**: User can view and modify portfolio data
    - Owners always have full control over their portfolios

    ### 🔗 Integration

    **Using Portfolios:**
    - When uploading CSV data in File Management, select the target portfolio
    - Projects are linked to the selected portfolio
    - Reports and charts can be filtered by portfolio

    **Best Practices:**
    1. Create portfolios before uploading project data
    2. Use consistent naming conventions
    3. Assign clear ownership
    4. Regularly review and update portfolio information
    """)

# Footer
st.markdown("---")
st.caption("Portfolio Management - Portfolio Management Suite v1.0")

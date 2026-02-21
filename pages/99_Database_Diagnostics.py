"""
Database Diagnostics - Check what's in the database
"""

import streamlit as st
import pandas as pd
import re
from pathlib import Path
from database.db_connection import get_db, reset_db_instance
from database.db_schema import create_schema, create_indexes, seed_sdg_data
from services.db_data_service import db_data_manager
from utils.auth import check_authentication
from utils.user_manager import is_super_admin
from utils.firestore_client import get_firestore_client
from utils.database_config import get_user_database_config, set_user_database_config
from utils.db_validator import test_connection, validate_local_connection

# Page configuration
st.set_page_config(
    page_title="Database Diagnostics",
    page_icon="🔍",
    layout="wide"
)

# Check authentication
if not check_authentication():
    st.stop()

st.title("🔍 Database Diagnostics")

user_email = st.session_state.get('user_email')
db = get_db(user_email=user_email)

# Database Connections
st.header("🔗 Database Connections")

col1, col2 = st.columns(2)

with col1:
    st.subheader("Primary Database (DuckDB)")
    try:
        # Get the ACTUAL user configuration from Firestore (not local JSON)
        user_config = get_user_database_config(user_email)
        connection_type = user_config.get('connection_type', 'local')

        if connection_type == 'motherduck':
            database_name = user_config.get('motherduck_database', 'N/A')
            st.success("✅ Connected to MotherDuck (Cloud)")
            st.markdown(f"**Database:** `{database_name}`")
            st.markdown(f"**Connection:** `md:{database_name}`")
            st.caption("☁️ Data stored in MotherDuck cloud")
        else:
            local_path = user_config.get('local_path', 'N/A')
            st.success("✅ Connected to Local DuckDB")
            st.markdown(f"**Path:** `{local_path}`")
            st.caption("💾 Data stored locally on disk")

        # Test connection and show table count
        try:
            table_count = db.execute("SELECT COUNT(*) FROM information_schema.tables WHERE table_schema = 'main'").fetchone()[0]
            st.metric("Tables in Database", table_count)
        except Exception as e:
            st.error(f"Connection test failed: {e}")

    except Exception as e:
        st.error(f"Error getting DuckDB connection info: {e}")

with col2:
    st.subheader("User Auth Database (Firestore)")
    try:
        from utils.firestore_client import is_firestore_configured, get_firestore_client as get_fs_client

        if is_firestore_configured():
            fs_client = get_fs_client()
            project_id = fs_client.project if fs_client else "Unknown"
            st.success("✅ Connected to Firestore")
            st.markdown(f"**Project:** `{project_id}`")
            st.markdown(f"**Connection:** `firestore.googleapis.com`")
            st.caption("🔐 Stores user accounts & preferences")
        else:
            st.warning("⚠️ Firestore not configured")
            st.caption("Optional - used for multi-user authentication")

    except Exception as e:
        st.warning("⚠️ Firestore not available")
        st.caption(f"Error: {str(e)[:100]}")

st.markdown("---")

# ⚙️ Database Configuration Section (Super Admin Only)
st.header("⚙️ Database Configuration")

# Check if user is super admin
user_email = st.session_state.get('user_email')
firestore_db = get_firestore_client()

if not is_super_admin(firestore_db, user_email):
    st.info("🔒 Database configuration is only available to Super Administrators")
else:
    st.success("✅ Super Admin Access Granted")
    st.info("""
    **Database Configuration Switcher**

    Switch between MotherDuck (cloud) and Local DuckDB databases. This affects your database connection only.

    ⚠️ **Warning:** Data does NOT sync between local and cloud databases. They are independent data stores.
    """)

    # Load current user configuration
    current_config = get_user_database_config(user_email)
    current_type = current_config.get('connection_type', 'local')

    col1, col2 = st.columns([1, 2])

    with col1:
        st.subheader("Current Connection")
        if current_type == 'motherduck':
            st.info(f"☁️ **MotherDuck (Cloud)**")
            st.caption(f"Database: {current_config.get('motherduck_database', 'N/A')}")
        else:
            st.info(f"💾 **Local DuckDB**")
            st.caption(f"Path: {current_config.get('local_path', 'N/A')}")

    with col2:
        st.subheader("Switch Database Connection")

        # Connection type selector
        new_connection_type = st.radio(
            "Select Connection Type",
            options=["local", "motherduck"],
            format_func=lambda x: "💾 Local DuckDB File" if x == "local" else "☁️ MotherDuck (Cloud)",
            index=0 if current_type == "local" else 1,
            horizontal=True,
            key="db_connection_type_selector"
        )

        # Configuration inputs based on selection
        new_config = {
            'connection_type': new_connection_type
        }

        if new_connection_type == 'local':
            st.markdown("**Local Database Configuration**")

            # Default path in DuckDB directory
            default_path = str(Path(__file__).parent.parent / "DuckDB" / "portfolio.duckdb")
            current_local_path = current_config.get('local_path', default_path)

            # Option 1: Select existing file
            st.markdown("**Option 1: Select Existing Database**")
            existing_path = st.text_input(
                "Path to existing .duckdb file (must be in DuckDB/ directory)",
                value=current_local_path,
                key="local_db_path",
                help="Enter the full path to an existing .duckdb file within the application's DuckDB directory"
            )

            # Option 2: Create new file
            st.markdown("**Option 2: Create New Database**")
            col_a, col_b = st.columns([3, 1])
            with col_a:
                new_db_name = st.text_input(
                    "New database filename",
                    value="",
                    placeholder="my_portfolio.duckdb",
                    key="new_db_name",
                    help="Enter filename only (will be created in DuckDB directory)"
                )
            with col_b:
                create_new = st.button("Create New", type="secondary", disabled=not new_db_name)

            # Determine which path to use
            if create_new and new_db_name:
                # Create new database path
                if not new_db_name.endswith('.duckdb'):
                    new_db_name += '.duckdb'
                new_config['local_path'] = str(Path(__file__).parent.parent / "DuckDB" / new_db_name)
                new_config['_create_new'] = True
            else:
                new_config['local_path'] = existing_path
                new_config['_create_new'] = False

        else:  # motherduck
            st.markdown("**MotherDuck Configuration**")

            # Get org token from secrets (if available)
            has_org_token = False
            try:
                if hasattr(st, 'secrets') and 'motherduck' in st.secrets:
                    org_token = st.secrets['motherduck'].get('token', '')
                    if org_token:
                        has_org_token = True
                        st.success("✅ Organization token configured")
            except:
                pass

            if not has_org_token:
                st.warning("⚠️ MotherDuck organization token not found in secrets")
                st.info("Add MotherDuck token to `.streamlit/secrets.toml` under `[motherduck]` section")

            new_config['motherduck_database'] = st.text_input(
                "MotherDuck Database Name",
                value=current_config.get('motherduck_database', 'portfolio_cloud'),
                help="Name of your MotherDuck database (without 'md:' prefix)"
            )

            # Optional: Allow user-specific token (overrides org token)
            use_custom_token = st.checkbox("Use custom MotherDuck token (overrides org token)")
            if use_custom_token:
                new_config['motherduck_token'] = st.text_input(
                    "MotherDuck Token",
                    type="password",
                    value=current_config.get('motherduck_token', ''),
                    help="Your personal MotherDuck access token"
                )
            else:
                new_config['motherduck_token'] = ''

    # Validation and Save Section
    st.markdown("---")

    col_test, col_save = st.columns(2)

    with col_test:
        if st.button("🔍 Test Connection", type="secondary", width='stretch'):
            with st.spinner("Testing connection..."):
                success, message = test_connection(new_config)

                if success:
                    st.success(f"✅ {message}")
                    st.session_state['connection_validated'] = True
                    st.session_state['validated_config'] = new_config
                else:
                    st.error(f"❌ {message}")
                    st.session_state['connection_validated'] = False

    with col_save:
        # Only enable save if connection was validated or it's the same config
        config_changed = (
            new_config.get('connection_type') != current_config.get('connection_type') or
            new_config.get('local_path') != current_config.get('local_path') or
            new_config.get('motherduck_database') != current_config.get('motherduck_database')
        )

        save_enabled = (
            not config_changed or  # No change
            st.session_state.get('connection_validated', False)  # Or validated
        )

        save_button_label = "💾 Save & Reconnect"
        if new_config.get('_create_new'):
            save_button_label = "💾 Create & Initialize Database"

        if st.button(save_button_label, type="primary", width='stretch', disabled=not save_enabled):
            with st.spinner("Saving configuration and reconnecting..."):
                try:
                    # Handle new database creation
                    if new_config.get('_create_new'):
                        st.info("Creating new database file...")
                        db_path = new_config['local_path']

                        # Validate we can create it
                        success, message = validate_local_connection(db_path, create_if_missing=True)

                        if not success:
                            st.error(f"Failed to create database: {message}")
                            st.stop()

                        st.success(f"✅ {message}")

                        # Initialize the new database
                        st.info("Initializing database schema...")

                        # Temporarily set the new database path and reset singleton
                        reset_db_instance()
                        from database.db_connection import DatabaseConnection
                        temp_db = DatabaseConnection(db_path)

                        progress = st.progress(0)
                        status_text = st.empty()

                        status_text.text("Creating schema...")
                        create_schema()
                        progress.progress(33)

                        status_text.text("Creating indexes...")
                        create_indexes()
                        progress.progress(66)

                        status_text.text("Seeding SDG data...")
                        seed_sdg_data()
                        progress.progress(100)

                        status_text.text("Initialization complete!")
                        temp_db.close()

                        st.success("✅ Database initialized successfully!")

                        # Clean up progress indicators
                        progress.empty()
                        status_text.empty()

                    # Save configuration to Firestore
                    config_to_save = {
                        'connection_type': new_config['connection_type'],
                        'local_path': new_config.get('local_path', ''),
                        'motherduck_database': new_config.get('motherduck_database', ''),
                        'motherduck_token': new_config.get('motherduck_token', '')
                    }

                    success = set_user_database_config(user_email, config_to_save)

                    if success:
                        st.success("✅ Configuration saved successfully!")

                        # Reset database instances to force reconnection
                        reset_db_instance()
                        db_data_manager.reset_connection()

                        st.info("🔄 Reconnecting to new database...")
                        st.session_state['connection_validated'] = False

                        # Rerun to apply changes
                        st.rerun()
                    else:
                        st.error("❌ Failed to save configuration")

                except Exception as e:
                    st.error(f"❌ Error during save: {str(e)}")
                    import traceback
                    with st.expander("Error Details"):
                        st.code(traceback.format_exc())

    # Warnings
    if config_changed:
        st.warning("⚠️ **Important Notes:**")
        st.markdown("""
        - **Data Independence:** Local and MotherDuck databases are completely independent. Data does not sync automatically.
        - **Test First:** Always test the connection before saving to ensure connectivity.
        - **Schema Requirement:** The target database must have the same schema. New databases are automatically initialized.
        - **Active Sessions:** Switching will disconnect all active database sessions.
        """)

st.markdown("---")

# Database Overview - All Tables
st.header("📊 Database Overview - All Tables")
try:
    # Get all tables in the database
    tables_query = """
        SELECT table_name
        FROM information_schema.tables
        WHERE table_schema = 'main'
        AND table_type = 'BASE TABLE'
        ORDER BY table_name
    """

    tables_df = db.execute(tables_query).df()

    if not tables_df.empty:
        st.success(f"Found {len(tables_df)} table(s) in the database")

        # Create a summary with row counts
        table_info = []
        for table_name in tables_df['table_name']:
            try:
                count_result = db.execute(f"SELECT COUNT(*) as count FROM {table_name}").fetchone()
                row_count = count_result[0] if count_result else 0
                table_info.append({
                    'Table Name': table_name,
                    'Row Count': row_count
                })
            except Exception as e:
                table_info.append({
                    'Table Name': table_name,
                    'Row Count': f'Error: {str(e)}'
                })

        # Display summary table
        summary_df = pd.DataFrame(table_info)
        st.dataframe(summary_df, width='stretch', hide_index=True)

        # Option to view data from each table
        st.markdown("---")
        st.subheader("🔍 Explore Table Data")

        selected_table = st.selectbox(
            "Select a table to view its data:",
            options=tables_df['table_name'].tolist()
        )

        if selected_table:
            col1, col2 = st.columns([3, 1])

            with col1:
                limit = st.slider("Number of rows to display", min_value=5, max_value=1000, value=100, step=5)

            with col2:
                show_schema = st.checkbox("Show Schema", value=False)

            if st.button(f"📋 View {selected_table}", type="primary"):
                try:
                    # Show schema if requested
                    if show_schema:
                        schema_query = f"""
                            SELECT column_name, data_type, is_nullable
                            FROM information_schema.columns
                            WHERE table_name = '{selected_table}'
                            ORDER BY ordinal_position
                        """
                        schema_df = db.execute(schema_query).df()

                        st.markdown("**Table Schema:**")
                        st.dataframe(schema_df, width='stretch', hide_index=True)
                        st.markdown("---")

                    # Show data
                    data_df = db.execute(f"SELECT * FROM {selected_table} LIMIT {limit}").df()

                    if not data_df.empty:
                        st.success(f"Showing {len(data_df)} row(s) from {selected_table}")
                        st.dataframe(data_df, width='stretch')

                        # Download option
                        csv = data_df.to_csv(index=False)
                        st.download_button(
                            label=f"📥 Download {selected_table} data as CSV",
                            data=csv,
                            file_name=f"{selected_table}_export.csv",
                            mime="text/csv"
                        )
                    else:
                        st.info(f"Table '{selected_table}' is empty")

                except Exception as e:
                    st.error(f"Error querying {selected_table}: {e}")
    else:
        st.warning("No tables found in database")

except Exception as e:
    st.error(f"Error retrieving database tables: {e}")
    st.info("This might happen if the database connection is not established properly")

st.markdown("---")

# Quick Action
st.header("🔧 Quick Checks")

col1, col2 = st.columns(2)

with col1:
    st.subheader("Check Specific Portfolio")
    try:
        portfolio_options = db.execute(
            "SELECT portfolio_id, portfolio_name FROM portfolio WHERE is_active = TRUE"
        ).fetchall()

        if portfolio_options:
            selected = st.selectbox(
                "Select Portfolio",
                options=[(p[0], p[1]) for p in portfolio_options],
                format_func=lambda x: f"{x[1]} (ID: {x[0]})"
            )

            if selected:
                portfolio_id = selected[0]

                # Count projects
                project_count = db.execute(
                    "SELECT COUNT(*) FROM project WHERE portfolio_id = ? AND is_active = TRUE",
                    (portfolio_id,)
                ).fetchone()[0]

                # Get status dates
                status_dates = db.execute("""
                    SELECT DISTINCT status_date
                    FROM project_status_report
                    WHERE portfolio_id = ?
                    ORDER BY status_date DESC
                """, (portfolio_id,)).fetchall()

                st.metric("Projects", project_count)
                if status_dates:
                    st.success(f"Status dates available: {len(status_dates)}")
                    for sd in status_dates:
                        st.text(f"  • {sd[0]}")
                else:
                    st.warning("No status reports for this portfolio")
    except Exception as e:
        st.error(f"Error: {e}")

with col2:
    st.subheader("Check Specific Date")
    check_date = st.date_input("Select Date to Check")

    if st.button("Check This Date"):
        try:
            date_data = db.execute("""
                SELECT
                    portfolio_id,
                    COUNT(*) as project_count
                FROM project_status_report
                WHERE status_date = ?
                GROUP BY portfolio_id
            """, (check_date,)).fetchall()

            if date_data:
                st.success(f"Found data for {check_date}")
                for row in date_data:
                    st.info(f"Portfolio {row[0]}: {row[1]} project(s)")
            else:
                st.warning(f"❌ No data found for {check_date}")
                st.info("Try checking the available dates in the Status Reports section above")
        except Exception as e:
            st.error(f"Error: {e}")

st.markdown("---")

# Fix Missing Baselines
st.header("🔧 Fix Missing Baselines")
st.info("If projects were created without baselines, use this tool to fix them")

if st.button("🔍 Check for Projects Missing Baselines", type="primary"):
    try:
        missing_baselines = db.execute("""
            SELECT
                p.project_id,
                p.project_name,
                p.portfolio_id,
                p.initial_budget,
                p.project_status
            FROM project p
            LEFT JOIN project_baseline pb ON p.project_id = pb.project_id
            WHERE p.is_active = TRUE
              AND pb.baseline_id IS NULL
        """).fetchall()

        if not missing_baselines:
            st.success("✅ All active projects have baselines!")
        else:
            st.warning(f"⚠️ Found {len(missing_baselines)} project(s) without baselines:")

            # Show list
            for p in missing_baselines:
                st.text(f"  • Project {p[0]}: {p[1]}")

            st.markdown("---")

            if st.button("🔨 Create Missing Baselines", type="secondary"):
                from datetime import date, datetime

                created_count = 0
                error_count = 0

                progress_bar = st.progress(0)
                status_text = st.empty()

                for idx, p in enumerate(missing_baselines):
                    project_id = p[0]
                    project_name = p[1]
                    portfolio_id = p[2]
                    initial_budget = p[3] or 0
                    project_status = p[4] or 'Ongoing'

                    status_text.text(f"Processing {project_name}...")

                    try:
                        # Get the earliest status report for this project
                        earliest_status = db.execute("""
                            SELECT status_date, actual_cost
                            FROM project_status_report
                            WHERE project_id = ?
                            ORDER BY status_date ASC
                            LIMIT 1
                        """, (project_id,)).fetchone()

                        if earliest_status:
                            status_date = earliest_status[0]
                            # Convert string date to date object if needed
                            if isinstance(status_date, str):
                                status_date = datetime.strptime(status_date, '%Y-%m-%d').date()
                        else:
                            status_date = date.today()

                        # Generate baseline ID
                        result = db.fetch_one("SELECT COALESCE(MAX(baseline_id), 0) + 1 FROM project_baseline")
                        baseline_id = result[0]

                        # Create baseline v0
                        db.execute("""
                            INSERT INTO project_baseline (
                                baseline_id, project_id, portfolio_id, baseline_version,
                                baseline_start_date, baseline_end_date,
                                planned_start_date, planned_finish_date,
                                budget_at_completion, project_status,
                                baseline_reason, is_active, created_at
                            ) VALUES (?, ?, ?, 0, ?, NULL, ?, ?, ?, ?, 'Initial baseline (auto-created)', TRUE, now())
                        """, (
                            baseline_id,
                            project_id,
                            portfolio_id,
                            status_date,
                            status_date,
                            date(2099, 12, 31),  # placeholder finish date
                            initial_budget,
                            project_status,
                        ))

                        created_count += 1

                    except Exception as e:
                        error_count += 1
                        st.error(f"Error creating baseline for {project_name}: {e}")

                    progress_bar.progress((idx + 1) / len(missing_baselines))

                status_text.empty()
                progress_bar.empty()

                if created_count > 0:
                    st.success(f"✅ Created {created_count} baseline(s)")
                if error_count > 0:
                    st.error(f"❌ {error_count} error(s) occurred")

                st.rerun()

    except Exception as e:
        st.error(f"Error checking for missing baselines: {e}")

st.markdown("---")

# Purge Database Records
st.header("🗑️ Purge Database Records")
st.warning("⚠️ **CAUTION**: Purging permanently deletes records from the database. This action cannot be undone!")
st.info("""
This tool performs a three-step cleanup:
1. **Step 1**: Get list of portfolio_ids with is_active = TRUE (valid portfolios)
2. **Step 2**: Mark records as is_active = FALSE where portfolio_id is NOT in the valid list
3. **Step 3**: Delete all records with is_active = FALSE in this sequence:

**Deletion sequence (respects FK constraints):**
project_status_report → project_sdg → project_factor_score → project_baseline → project → portfolio_ownership → portfolio_factor → portfolio → app_user
""")

# Preview mode
st.subheader("📊 Scan Database")

if st.button("🔍 Scan for Records to Purge", type="primary"):
    try:
        purge_stats = {
            'inactive_records': {},
            'orphaned_records': {}
        }

        # STEP 1: Find all inactive records (is_active = FALSE)
        st.info("Step 1: Scanning for inactive records (is_active = FALSE)...")

        # Tables with is_active column (in deletion sequence order)
        tables_with_is_active = [
            'project_status_report',
            'project_sdg',
            'project_factor_score',
            'project_baseline',
            'project',
            'portfolio_ownership',
            'portfolio_factor',
            'portfolio',
            'app_user'
        ]

        for table in tables_with_is_active:
            inactive_count = db.execute(f"SELECT COUNT(*) FROM {table} WHERE is_active = FALSE").fetchone()[0]
            if inactive_count > 0:
                purge_stats['inactive_records'][table] = inactive_count

        # STEP 2: Find orphaned records
        st.info("Step 2: Scanning for orphaned records...")

        # Get list of active portfolio_ids
        active_portfolios = db.execute("SELECT portfolio_id FROM portfolio WHERE is_active = TRUE").fetchall()
        active_portfolio_ids = [p[0] for p in active_portfolios]

        # Get list of active project_ids
        active_projects = db.execute("SELECT project_id FROM project WHERE is_active = TRUE").fetchall()
        active_project_ids = [p[0] for p in active_projects]

        # Check for orphaned records in each table (in deletion sequence)
        if active_portfolio_ids:
            portfolio_ids_str = ','.join(str(pid) for pid in active_portfolio_ids)

            # project_status_report - orphaned by portfolio_id
            orphaned_status = db.execute(
                f"SELECT COUNT(*) FROM project_status_report WHERE portfolio_id NOT IN ({portfolio_ids_str})"
            ).fetchone()[0]
            if orphaned_status > 0:
                purge_stats['orphaned_records']['project_status_report (by portfolio_id)'] = orphaned_status

            # project_sdg - orphaned by portfolio_id
            orphaned_sdg_portfolio = db.execute(
                f"SELECT COUNT(*) FROM project_sdg WHERE portfolio_id NOT IN ({portfolio_ids_str})"
            ).fetchone()[0]
            if orphaned_sdg_portfolio > 0:
                purge_stats['orphaned_records']['project_sdg (by portfolio_id)'] = orphaned_sdg_portfolio

            # project_factor_score - orphaned by portfolio_id
            orphaned_scores_portfolio = db.execute(
                f"SELECT COUNT(*) FROM project_factor_score WHERE portfolio_id NOT IN ({portfolio_ids_str})"
            ).fetchone()[0]
            if orphaned_scores_portfolio > 0:
                purge_stats['orphaned_records']['project_factor_score (by portfolio_id)'] = orphaned_scores_portfolio

            # project_baseline - orphaned by portfolio_id
            orphaned_baseline_portfolio = db.execute(
                f"SELECT COUNT(*) FROM project_baseline WHERE portfolio_id NOT IN ({portfolio_ids_str})"
            ).fetchone()[0]
            if orphaned_baseline_portfolio > 0:
                purge_stats['orphaned_records']['project_baseline (by portfolio_id)'] = orphaned_baseline_portfolio

            # portfolio_ownership - orphaned by portfolio_id
            orphaned_ownership = db.execute(
                f"SELECT COUNT(*) FROM portfolio_ownership WHERE portfolio_id NOT IN ({portfolio_ids_str})"
            ).fetchone()[0]
            if orphaned_ownership > 0:
                purge_stats['orphaned_records']['portfolio_ownership'] = orphaned_ownership

            # portfolio_factor - orphaned by portfolio_id (excluding inactive ones already counted)
            orphaned_factors = db.execute(
                f"SELECT COUNT(*) FROM portfolio_factor WHERE portfolio_id NOT IN ({portfolio_ids_str}) AND is_active = TRUE"
            ).fetchone()[0]
            if orphaned_factors > 0:
                purge_stats['orphaned_records']['portfolio_factor (active but orphaned)'] = orphaned_factors

        if active_project_ids:
            project_ids_str = ','.join(str(pid) for pid in active_project_ids)

            # project_status_report - orphaned by project_id
            orphaned_status_proj = db.execute(
                f"SELECT COUNT(*) FROM project_status_report WHERE project_id NOT IN ({project_ids_str})"
            ).fetchone()[0]
            if orphaned_status_proj > 0:
                purge_stats['orphaned_records']['project_status_report (by project_id)'] = orphaned_status_proj

            # project_sdg - orphaned by project_id
            orphaned_sdg_proj = db.execute(
                f"SELECT COUNT(*) FROM project_sdg WHERE project_id NOT IN ({project_ids_str})"
            ).fetchone()[0]
            if orphaned_sdg_proj > 0:
                purge_stats['orphaned_records']['project_sdg (by project_id)'] = orphaned_sdg_proj

            # project_factor_score - orphaned by project_id
            orphaned_scores_proj = db.execute(
                f"SELECT COUNT(*) FROM project_factor_score WHERE project_id NOT IN ({project_ids_str})"
            ).fetchone()[0]
            if orphaned_scores_proj > 0:
                purge_stats['orphaned_records']['project_factor_score (by project_id)'] = orphaned_scores_proj

            # project_baseline - orphaned by project_id
            orphaned_baseline_proj = db.execute(
                f"SELECT COUNT(*) FROM project_baseline WHERE project_id NOT IN ({project_ids_str})"
            ).fetchone()[0]
            if orphaned_baseline_proj > 0:
                purge_stats['orphaned_records']['project_baseline (by project_id)'] = orphaned_baseline_proj

            # project - orphaned by portfolio_id (active projects in non-existent portfolios)
            if active_portfolio_ids:
                orphaned_projects = db.execute(
                    f"SELECT COUNT(*) FROM project WHERE is_active = TRUE AND portfolio_id NOT IN ({portfolio_ids_str})"
                ).fetchone()[0]
                if orphaned_projects > 0:
                    purge_stats['orphaned_records']['project (active but orphaned)'] = orphaned_projects

        # Display results
        total_inactive = sum(purge_stats['inactive_records'].values())
        total_orphaned = sum(purge_stats['orphaned_records'].values())
        grand_total = total_inactive + total_orphaned

        if grand_total == 0:
            st.success("✅ No records to purge. Database is clean!")
        else:
            st.warning(f"⚠️ Found **{grand_total} total records** to purge")

            col1, col2 = st.columns(2)

            with col1:
                st.markdown("**Inactive Records (is_active = FALSE):**")
                if total_inactive > 0:
                    for table, count in purge_stats['inactive_records'].items():
                        st.metric(table.replace('_', ' ').title(), count)
                else:
                    st.info("None found")

            with col2:
                st.markdown("**Orphaned Records:**")
                if total_orphaned > 0:
                    for table, count in purge_stats['orphaned_records'].items():
                        st.metric(table, count)
                else:
                    st.info("None found")

            # Store in session state
            st.session_state['purge_stats'] = purge_stats

    except Exception as e:
        st.error(f"Error scanning database: {e}")
        import traceback
        st.code(traceback.format_exc())

# Purge execution section
if 'purge_stats' in st.session_state:
    purge_stats = st.session_state['purge_stats']
    total_inactive = sum(purge_stats['inactive_records'].values())
    total_orphaned = sum(purge_stats['orphaned_records'].values())
    grand_total = total_inactive + total_orphaned

    if grand_total > 0:
        st.markdown("---")
        st.subheader("🔥 Execute Purge")

        # Dry run mode
        dry_run = st.checkbox(
            "🧪 **Dry Run Mode** (simulate only, don't actually delete)",
            value=True,
            help="When enabled, shows what would be deleted without actually deleting anything. **Uncheck this to enable actual deletion.**"
        )

        if dry_run:
            st.info("ℹ️ Dry Run is enabled. The purge will only be simulated. **Uncheck the box above to perform actual deletion.**")
        else:
            st.warning("⚠️ Dry Run is disabled. The purge will **permanently delete** records from the database!")

        st.info(f"📊 **Total records to purge**: {total_inactive} inactive + {total_orphaned} orphaned = **{grand_total} total**")

        st.markdown("---")

        # Final confirmation
        if not dry_run:
            st.error("🔴 **FINAL WARNING**: You are about to permanently delete records!")
            st.markdown("**This action cannot be undone!**")
            confirm_text = st.text_input(
                f"Type 'DELETE {grand_total}' to confirm:",
                key="purge_confirm",
                placeholder=f"DELETE {grand_total}"
            )

            execute_enabled = confirm_text == f"DELETE {grand_total}"

            if not execute_enabled and confirm_text:
                st.error(f"❌ Confirmation text doesn't match. Please type exactly: DELETE {grand_total}")
        else:
            execute_enabled = True

        # Execute button
        if dry_run:
            button_label = "🧪 Run Dry Run (Preview Only)"
            button_type = "primary"
        else:
            button_label = "🔥 EXECUTE PURGE (Permanent Deletion)"
            button_type = "secondary"

        if st.button(button_label, type=button_type, disabled=not execute_enabled):
            try:
                # Use dictionary to track state for proper scoping
                state = {
                    'purge_log': [],
                    'error_count': 0,
                    'deleted_count': 0
                }

                progress_bar = st.progress(0)
                status_text = st.empty()

                def log(message, is_error=False):
                    state['purge_log'].append(('ERROR' if is_error else 'INFO', message))
                    if is_error:
                        state['error_count'] += 1
                    status_text.text(message)

                try:
                    # STEP 1: Get valid portfolio list (is_active = TRUE)
                    print("")
                    print("="*80)
                    print("STEP 1: Get valid portfolio list (is_active = TRUE)")
                    print("="*80)

                    log("=" * 60)
                    log("STEP 1: Get valid portfolio list (is_active = TRUE)")
                    log("=" * 60)

                    valid_portfolios = db.execute("SELECT portfolio_id FROM portfolio WHERE is_active = TRUE").fetchall()
                    valid_portfolio_ids = [p[0] for p in valid_portfolios]

                    print(f"Valid portfolios (is_active = TRUE): {valid_portfolio_ids}")
                    print(f"Count: {len(valid_portfolio_ids)}")
                    log(f"Valid portfolios (is_active = TRUE): {valid_portfolio_ids}")

                    progress_bar.progress(0.2)

                    # STEP 2: Mark records as inactive where portfolio_id is NOT in valid list
                    print("")
                    print("="*80)
                    print("STEP 2: Mark records as inactive where portfolio_id NOT in valid list")
                    print("="*80)

                    log("=" * 60)
                    log("STEP 2: Mark records as inactive where portfolio_id NOT in valid list")
                    log("=" * 60)

                    if valid_portfolio_ids:
                        portfolio_ids_str = ','.join(str(pid) for pid in valid_portfolio_ids)

                        # All tables with both portfolio_id and is_active columns
                        # Mark in reverse order of deletion (children first, then parents)
                        tables_to_mark = [
                            ('project_status_report', 'portfolio_id'),
                            ('project_sdg', 'portfolio_id'),
                            ('project_factor_score', 'portfolio_id'),
                            ('project_baseline', 'portfolio_id'),
                            ('project', 'portfolio_id'),
                            ('portfolio_ownership', 'portfolio_id'),
                            ('portfolio_factor', 'portfolio_id'),
                        ]

                        for table, fk_column in tables_to_mark:
                            if not dry_run:
                                # Mark as inactive
                                result = db.execute(f"UPDATE {table} SET is_active = FALSE WHERE {fk_column} NOT IN ({portfolio_ids_str}) AND is_active = TRUE")
                                marked_count = db.execute(f"SELECT COUNT(*) FROM {table} WHERE {fk_column} NOT IN ({portfolio_ids_str}) AND is_active = FALSE").fetchone()[0]
                                if marked_count > 0:
                                    print(f"  ✓ Marked {marked_count} records as inactive in {table}")
                                    log(f"  ✓ Marked {marked_count} records as inactive in {table}")
                            else:
                                check_count = db.execute(f"SELECT COUNT(*) FROM {table} WHERE {fk_column} NOT IN ({portfolio_ids_str}) AND is_active = TRUE").fetchone()[0]
                                if check_count > 0:
                                    print(f"  ↳ Would mark {check_count} records as inactive in {table}")
                                    log(f"  ↳ Would mark {check_count} records as inactive in {table}")
                    else:
                        print("  ⚠ No valid portfolios found - all records will be marked inactive")
                        log("  ⚠ No valid portfolios found - all records will be marked inactive")

                    progress_bar.progress(0.4)

                    # STEP 3: Delete all records with is_active = FALSE
                    print("")
                    print("="*80)
                    print("STEP 3: Delete all records with is_active = FALSE")
                    print("="*80)
                    print("Deletion sequence: project_status_report → project_sdg → project_factor_score → project_baseline → project → portfolio_ownership → portfolio_factor → portfolio → app_user")
                    print("="*80)

                    log("=" * 60)
                    log("STEP 3: Delete all records with is_active = FALSE")
                    log("=" * 60)
                    log("Deletion sequence: project_status_report → project_sdg → project_factor_score → project_baseline → project → portfolio_ownership → portfolio_factor → portfolio → app_user")
                    log("=" * 60)

                    # Deletion sequence (respects FK constraints)
                    deletion_sequence = [
                        'project_status_report',
                        'project_sdg',
                        'project_factor_score',
                        'project_baseline',
                        'project',
                        'portfolio_ownership',
                        'portfolio_factor',
                        'portfolio',
                        'app_user'
                    ]

                    for idx, table in enumerate(deletion_sequence):
                        # Terminal output (visible in console)
                        print("")
                        print(f"{'='*80}")
                        print(f"[{idx+1}/{len(deletion_sequence)}] TABLE: {table}")
                        print(f"{'='*80}")

                        # UI log output
                        log("")
                        log(f"{'='*60}")
                        log(f"Processing Table [{idx+1}/{len(deletion_sequence)}]: {table}")
                        log(f"{'='*60}")

                        # All tables in the deletion sequence now have is_active column
                        has_is_active = table in [
                            'project_status_report',
                            'project_sdg',
                            'project_factor_score',
                            'project_baseline',
                            'project',
                            'portfolio_ownership',
                            'portfolio_factor',
                            'portfolio',
                            'app_user'
                        ]

                        if has_is_active:
                            # Count total records
                            total_count = db.execute(f"SELECT COUNT(*) FROM {table}").fetchone()[0]
                            # Count inactive records
                            inactive_count = db.execute(f"SELECT COUNT(*) FROM {table} WHERE is_active = FALSE").fetchone()[0]
                            # Count active records
                            active_count = db.execute(f"SELECT COUNT(*) FROM {table} WHERE is_active = TRUE").fetchone()[0]

                            # Terminal output
                            print(f"Total records: {total_count}")
                            print(f"Active records (is_active = TRUE): {active_count}")
                            print(f"Inactive records (is_active = FALSE): {inactive_count}")

                            # UI log output
                            log(f"  Total records in {table}: {total_count}")
                            log(f"  Active records (is_active = TRUE): {active_count}")
                            log(f"  Inactive records (is_active = FALSE): {inactive_count}")

                            if inactive_count > 0:
                                # Terminal output
                                print(f"→ {'[DRY RUN] ' if dry_run else ''}Attempting to delete {inactive_count} inactive records...")

                                # UI log output
                                log(f"  → {'[DRY RUN] ' if dry_run else ''}Attempting to delete {inactive_count} inactive records from {table}...")

                                if not dry_run:
                                    # Start transaction for this table
                                    try:
                                        print(f"→ Starting transaction for {table}...")
                                        db.execute("BEGIN TRANSACTION")

                                        # Special handling for app_user
                                        if table == 'app_user':
                                            # Check for active portfolio ownership
                                            inactive_users = db.execute("SELECT user_id, display_name FROM app_user WHERE is_active = FALSE").fetchall()
                                            deleted_users = 0
                                            skipped_users = 0

                                            print(f"→ Processing {len(inactive_users)} inactive users...")
                                            log(f"  → Processing {len(inactive_users)} inactive users...")

                                            for user_id, user_name in inactive_users:
                                                # Check portfolio_ownership table (owner_user_id is there now)
                                                active_portfolios_owned = db.execute("""
                                                    SELECT COUNT(*)
                                                    FROM portfolio_ownership po
                                                    JOIN portfolio p ON po.portfolio_id = p.portfolio_id
                                                    WHERE po.owner_user_id = ?
                                                    AND p.is_active = TRUE
                                                    AND po.is_active = TRUE
                                                """, (user_id,)).fetchone()[0]

                                                if active_portfolios_owned > 0:
                                                    print(f"⚠ Skipping user '{user_name}' (ID: {user_id}) - owns {active_portfolios_owned} active portfolios")
                                                    log(f"  ⚠ Skipping user '{user_name}' (ID: {user_id}) - owns {active_portfolios_owned} active portfolios", is_error=True)
                                                    skipped_users += 1
                                                    continue

                                                # Nullify references in project_factor_score
                                                print(f"→ Nullifying project_factor_score references for user '{user_name}' (ID: {user_id})...")
                                                log(f"  → Nullifying project_factor_score references for user '{user_name}' (ID: {user_id})...")
                                                db.execute("UPDATE project_factor_score SET scored_by_user_id = NULL WHERE scored_by_user_id = ?", (user_id,))

                                                # Delete the user
                                                print(f"→ Deleting user '{user_name}' (ID: {user_id})...")
                                                log(f"  → Deleting user '{user_name}' (ID: {user_id})...")
                                                db.execute("DELETE FROM app_user WHERE user_id = ?", (user_id,))
                                                deleted_users += 1

                                            state['deleted_count'] += deleted_users
                                            print(f"✓ Successfully deleted {deleted_users} inactive users")
                                            log(f"  ✓ Successfully deleted {deleted_users} inactive users")
                                            if skipped_users > 0:
                                                print(f"⚠ Skipped {skipped_users} users (own active portfolios)")
                                                log(f"  ⚠ Skipped {skipped_users} users (own active portfolios)")
                                        else:
                                            # Regular deletion
                                            print(f"→ Executing: DELETE FROM {table} WHERE is_active = FALSE")
                                            log(f"  → Executing: DELETE FROM {table} WHERE is_active = FALSE")

                                            db.execute(f"DELETE FROM {table} WHERE is_active = FALSE")

                                            # Verify deletion
                                            remaining_inactive = db.execute(f"SELECT COUNT(*) FROM {table} WHERE is_active = FALSE").fetchone()[0]
                                            if remaining_inactive > 0:
                                                print(f"⚠ WARNING: {remaining_inactive} inactive records still remain in {table}!")
                                                log(f"  ⚠ WARNING: {remaining_inactive} inactive records still remain in {table}!", is_error=True)

                                            state['deleted_count'] += inactive_count
                                            print(f"✓ Successfully deleted {inactive_count} inactive records from {table}")
                                            log(f"  ✓ Successfully deleted {inactive_count} inactive records from {table}")

                                        # Commit transaction for this table
                                        db.execute("COMMIT")
                                        print(f"✓ Transaction committed for {table}")
                                        log(f"  ✓ Transaction committed for {table}")

                                    except Exception as table_error:
                                        # Rollback only this table's transaction
                                        try:
                                            db.execute("ROLLBACK")
                                            print(f"✗ Transaction rolled back for {table}")
                                            log(f"  ✗ Transaction rolled back for {table}", is_error=True)
                                        except:
                                            pass  # Rollback might fail if transaction wasn't started

                                        print(f"✗ ERROR deleting from {table}: {str(table_error)}")
                                        log(f"  ✗ ERROR deleting from {table}: {str(table_error)}", is_error=True)
                                        # Don't re-raise - continue to next table
                                else:
                                    print(f"[DRY RUN] Would delete {inactive_count} inactive records from {table}")
                                    log(f"  ↳ [DRY RUN] Would delete {inactive_count} inactive records from {table}")
                            else:
                                print(f"✓ No inactive records - skipping deletion")
                                log(f"  ✓ No inactive records in {table} - skipping deletion")
                        else:
                            print(f"⚠ Table has no is_active column - skipping")
                            log(f"  ⚠ {table} has no is_active column - skipping")

                        # Update progress
                        progress_bar.progress(0.4 + (0.6 * (idx + 1) / len(deletion_sequence)))

                        print(f"{'='*80}")
                        print(f"Completed: {table}")
                        print(f"{'='*80}")

                        log(f"{'='*60}")
                        log(f"Completed processing {table}")
                        log(f"{'='*60}")

                    print("")
                    print("="*80)
                    print("ALL DELETIONS COMPLETE")
                    print("="*80)

                    log("=" * 60)
                    log("All deletions complete")
                    log("=" * 60)

                    if dry_run:
                        print("🧪 Dry run completed - no changes made")
                        log("🧪 Dry run completed - no changes made")
                    else:
                        print("✅ All table transactions completed")
                        log("✅ All table transactions completed")

                except Exception as e:
                    log(f"❌ Unexpected error occurred: {e}", is_error=True)
                    import traceback
                    print(traceback.format_exc())

                # Clear progress
                progress_bar.empty()
                status_text.empty()

                # Display results
                if state['error_count'] == 0:
                    if dry_run:
                        st.success(f"🧪 Dry run completed! {grand_total} record(s) would be purged.")
                    else:
                        st.success(f"✅ Successfully purged {state['deleted_count']} record(s)!")
                        # Clear session state
                        del st.session_state['purge_stats']
                else:
                    st.error(f"❌ Completed with {state['error_count']} error(s)")

                # Show log
                with st.expander("📜 View Operation Log", expanded=True):
                    log_text = "\n".join([f"[{level}] {msg}" for level, msg in state['purge_log']])
                    st.text_area("Log", log_text, height=400)

                    st.download_button(
                        "📥 Download Log",
                        log_text,
                        f"purge_log_{'dry_run_' if dry_run else ''}{pd.Timestamp.now().strftime('%Y%m%d_%H%M%S')}.txt",
                        "text/plain"
                    )

                if not dry_run and state['error_count'] == 0:
                    st.info("💡 Click 'Scan for Records to Purge' again to verify the purge")

            except Exception as e:
                st.error(f"Fatal error during purge operation: {e}")
                import traceback
                st.code(traceback.format_exc())

st.markdown("---")
st.caption("💡 Use this page to diagnose data loading issues and verify database contents")

# Enter Progress Data — Enhanced Project Data Management Interface
# Professional interface for adding, editing, and managing project data

from __future__ import annotations
import io
import json
import logging
import math
import os
import re
import textwrap
from datetime import datetime, date, timedelta
from pathlib import Path
from typing import Optional, Dict, Any, List, Union

import pandas as pd
import numpy as np
import streamlit as st
from dateutil import parser as date_parser
from utils.auth import check_authentication, require_page_access
from utils.portfolio_context import render_portfolio_context
from services.db_data_service import DatabaseDataManager
from config.constants import USE_DATABASE

# =============================================================================
# CONSTANTS
# =============================================================================

# Application constants
APP_TITLE = "Enter Progress Data 📝"
DEFAULT_DATASET_TABLE = "dataset"

# =============================================================================
# DATA PERSISTENCE FUNCTIONS (copied from main app)
# =============================================================================

def load_table(table_name: str) -> pd.DataFrame:
    """Load table from session state."""
    try:
        # Load main dataset table
        if table_name == DEFAULT_DATASET_TABLE:
            if st.session_state.data_df is not None:
                return st.session_state.data_df.copy()
            else:
                return pd.DataFrame()

        # Load from config tables
        if ("tables" in st.session_state.config_dict and
            table_name in st.session_state.config_dict["tables"]):
            table_records = st.session_state.config_dict["tables"][table_name]
            return pd.DataFrame(table_records)

        # Table not found
        return pd.DataFrame()

    except Exception as e:
        st.error(f"Failed to load table {table_name}: {e}")
        return pd.DataFrame()

def save_table_replace(df: pd.DataFrame, table_name: str):
    """Save table to session state."""
    try:
        # Save to appropriate session location
        if table_name == DEFAULT_DATASET_TABLE:
            st.session_state.data_df = df.copy()
            st.session_state.data_loaded = True
        else:
            # Save to config tables
            if "tables" not in st.session_state.config_dict:
                st.session_state.config_dict["tables"] = {}

            st.session_state.config_dict["tables"][table_name] = df.to_dict('records')

    except Exception as e:
        st.error(f"Failed to save table {table_name}: {e}")
        raise

# =============================================================================
# PAGE CONFIGURATION
# =============================================================================

st.set_page_config(
    page_title=APP_TITLE,
    page_icon="📝",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# Check authentication and page access
if not check_authentication():
    st.stop()

require_page_access('manual_data_entry', 'Enter Progress Data')

# Custom CSS for professional appearance
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
</style>
""", unsafe_allow_html=True)

# =============================================================================
# SESSION STATE INITIALIZATION
# =============================================================================

def initialize_session_state():
    """Initialize session state variables."""
    if "data_df" not in st.session_state:
        st.session_state.data_df = None
    if "data_loaded" not in st.session_state:
        st.session_state.data_loaded = False
    if "config_dict" not in st.session_state:
        st.session_state.config_dict = {}
    if "selected_row_index" not in st.session_state:
        st.session_state.selected_row_index = None
    if "edit_mode" not in st.session_state:
        st.session_state.edit_mode = False
    if "active_tab" not in st.session_state:
        st.session_state.active_tab = 0  # Default to first tab

    # Synchronize data_loaded flag with actual data state
    # This ensures File Management sees the correct data status
    if st.session_state.data_df is not None and not st.session_state.data_df.empty:
        st.session_state.data_loaded = True
    else:
        st.session_state.data_loaded = False

# =============================================================================
# UTILITY FUNCTIONS
# =============================================================================

def validate_project_data(data: Dict[str, Any]) -> List[str]:
    """Validate project data and return list of errors."""
    errors = []

    # Required fields
    if not data.get("Project ID", "").strip():
        errors.append("Project ID is required")
    if not data.get("Project", "").strip():
        errors.append("Project name is required")

    # Numeric validations
    try:
        bac = float(data.get("BAC", 0))
        if bac <= 0:
            errors.append("BAC must be greater than 0")
    except (ValueError, TypeError):
        errors.append("BAC must be a valid number")

    try:
        ac = float(data.get("AC", 0))
        if ac < 0:
            errors.append("AC cannot be negative")
    except (ValueError, TypeError):
        errors.append("AC must be a valid number")

    # Date validations
    try:
        start_date = date_parser.parse(data.get("Plan Start", ""))
        finish_date = date_parser.parse(data.get("Plan Finish", ""))
        if start_date >= finish_date:
            errors.append("Plan Start must be before Plan Finish")
    except:
        errors.append("Invalid date format. Use DD/MM/YYYY")

    return errors

def format_currency(value: float) -> str:
    """Format currency value with proper formatting."""
    if value is None:
        return ""
    return f"{value:,.2f}"

# =============================================================================
# DATA MANAGEMENT FUNCTIONS
# =============================================================================

def add_new_project(project_data: Dict[str, Any]) -> bool:
    """Add a new project to the dataframe."""
    try:
        # Load existing data
        current_df = load_table(DEFAULT_DATASET_TABLE)

        if current_df.empty:
            # Create new dataframe with proper schema
            current_df = pd.DataFrame(columns=[
                "Project ID", "Project", "Organization", "Project Manager",
                "BAC", "AC", "Plan Start", "Plan Finish",
                "Use_Manual_PV", "Manual_PV", "Use_Manual_EV", "Manual_EV",
                "Curve Type", "Alpha", "Beta", "Inflation Rate"
            ])

        # Check for duplicate Project ID
        if not current_df.empty and project_data["Project ID"] in current_df["Project ID"].values:
            st.error(f"Project ID '{project_data['Project ID']}' already exists!")
            return False

        # In database mode, also save to the database
        if USE_DATABASE:
            portfolio_id = st.session_state.get('current_portfolio_id')
            status_date = st.session_state.get('current_status_date')

            if portfolio_id and status_date:
                try:
                    db_manager = DatabaseDataManager()

                    # Parse dates
                    plan_start = project_data.get("Plan Start")
                    plan_finish = project_data.get("Plan Finish")

                    if isinstance(plan_start, str):
                        plan_start = pd.to_datetime(plan_start, dayfirst=True, errors='coerce')
                        if pd.notna(plan_start):
                            plan_start = plan_start.date()
                        else:
                            plan_start = None

                    if isinstance(plan_finish, str):
                        plan_finish = pd.to_datetime(plan_finish, dayfirst=True, errors='coerce')
                        if pd.notna(plan_finish):
                            plan_finish = plan_finish.date()
                        else:
                            plan_finish = None

                    # Create project in database
                    db_project_id = db_manager.create_project(
                        portfolio_id=portfolio_id,
                        project_data={
                            'project_name': project_data.get("Project"),
                            'project_code': project_data.get("Project ID"),
                            'responsible_organization': project_data.get("Organization"),
                            'project_manager': project_data.get("Project Manager"),
                            'planned_start_date': plan_start,
                            'planned_finish_date': plan_finish,
                            'initial_budget': project_data.get("BAC", 0),
                            'current_budget': project_data.get("BAC", 0)
                        },
                        create_baseline=True
                    )

                    # Create status report for this period
                    db_manager.create_status_report(
                        project_id=db_project_id,
                        portfolio_id=portfolio_id,
                        status_data={
                            'status_date': status_date,
                            'actual_cost': project_data.get("AC", 0),
                            'planned_value': project_data.get("Manual_PV") if project_data.get("Use_Manual_PV") else None,
                            'earned_value': project_data.get("Manual_EV") if project_data.get("Use_Manual_EV") else None
                        }
                    )

                except Exception as db_error:
                    st.error(f"Database error: {db_error}")
                    import logging
                    logging.error(f"Database add project error: {db_error}")
                    return False

        # Add new row to session state - handle empty DataFrame case to avoid FutureWarning
        new_row = pd.DataFrame([project_data])

        if current_df.empty:
            # If current_df is empty, just use the new row
            updated_df = new_row.copy()
        else:
            # Only concat if current_df has data
            updated_df = pd.concat([current_df, new_row], ignore_index=True)

        # Save using the proper persistence mechanism
        save_table_replace(updated_df, DEFAULT_DATASET_TABLE)

        # Set file type for batch processing compatibility
        st.session_state.file_type = "manual"

        # Set flags for success message
        st.session_state.project_just_added = True
        st.session_state.new_project_id = project_data['Project ID']

        return True

    except Exception as e:
        st.error(f"Error adding project: {e}")
        return False

def update_project(index: int, project_data: Dict[str, Any]) -> bool:
    """Update an existing project."""
    try:
        current_df = load_table(DEFAULT_DATASET_TABLE)

        if current_df.empty or index >= len(current_df):
            st.error("Project not found!")
            return False

        # Check for duplicate Project ID (excluding current record)
        mask = current_df.index != index
        if project_data["Project ID"] in current_df.loc[mask, "Project ID"].values:
            st.error(f"Project ID '{project_data['Project ID']}' already exists!")
            return False

        # Get the project_id for database update
        old_project_id = current_df.at[index, "Project ID"]

        # Update row in session state
        for key, value in project_data.items():
            current_df.at[index, key] = value

        # Save to session state
        save_table_replace(current_df, DEFAULT_DATASET_TABLE)

        # In database mode, also update the database
        if USE_DATABASE:
            portfolio_id = st.session_state.get('current_portfolio_id')
            if portfolio_id:
                try:
                    db_manager = DatabaseDataManager()

                    # Get the database project_id (numeric)
                    from database.queries import get_project_by_name
                    db_project = get_project_by_name(portfolio_id, str(old_project_id))

                    if db_project:
                        db_project_id = db_project['project_id']

                        # Update project table fields
                        db_manager.update_project(db_project_id, {
                            'project_code': project_data.get("Project ID"),
                            'project_name': project_data.get("Project"),
                            'responsible_organization': project_data.get("Organization"),
                            'project_manager': project_data.get("Project Manager"),
                            'current_budget': project_data.get("BAC")
                        })

                        # Update baseline for Plan Start/Finish
                        # Get active baseline
                        from database.queries import get_active_baseline, check_planned_start_lock
                        active_baseline = get_active_baseline(db_project_id)

                        if active_baseline:
                            baseline_id = active_baseline['baseline_id']
                            baseline_version = active_baseline.get('baseline_version', 0)

                            # Parse dates
                            import pandas as pd
                            plan_start = project_data.get("Plan Start")
                            plan_finish = project_data.get("Plan Finish")

                            if isinstance(plan_start, str):
                                plan_start = pd.to_datetime(plan_start, dayfirst=True, errors='coerce')
                                if pd.notna(plan_start):
                                    plan_start = plan_start.date()
                                else:
                                    plan_start = None

                            if isinstance(plan_finish, str):
                                plan_finish = pd.to_datetime(plan_finish, dayfirst=True, errors='coerce')
                                if pd.notna(plan_finish):
                                    plan_finish = plan_finish.date()
                                else:
                                    plan_finish = None

                            # Check if plan start is locked (AC > 0 exists)
                            is_locked = check_planned_start_lock(db_project_id)

                            # For baseline 0, allow update if no AC recorded yet
                            if baseline_version == 0 and not is_locked:
                                # Direct update for baseline 0 when no AC recorded
                                db_manager.db.execute("""
                                    UPDATE project_baseline
                                    SET planned_start_date = ?,
                                        planned_finish_date = ?,
                                        budget_at_completion = ?
                                    WHERE baseline_id = ?
                                """, (plan_start, plan_finish, project_data.get("BAC"), baseline_id))
                            elif baseline_version > 0:
                                # For other baselines, use the standard update
                                db_manager.update_baseline(baseline_id, {
                                    'planned_start_date': plan_start,
                                    'planned_finish_date': plan_finish,
                                    'budget_at_completion': project_data.get("BAC")
                                })
                            elif is_locked:
                                st.warning("⚠️ Plan Start date is locked because actual costs have been recorded.")

                except Exception as db_error:
                    st.warning(f"⚠️ Session updated but database sync failed: {db_error}")
                    import logging
                    logging.error(f"Database update error: {db_error}")

        st.success(f"✅ Project '{project_data['Project ID']}' updated successfully!")
        return True

    except Exception as e:
        st.error(f"Error updating project: {e}")
        return False

def delete_project(index: int) -> bool:
    """Delete a project."""
    try:
        current_df = load_table(DEFAULT_DATASET_TABLE)

        if current_df.empty or index >= len(current_df):
            st.error("Project not found!")
            return False

        project_id = current_df.at[index, "Project ID"]
        updated_df = current_df.drop(index).reset_index(drop=True)

        # Save back to persistence
        save_table_replace(updated_df, DEFAULT_DATASET_TABLE)

        st.success(f"✅ Project '{project_id}' deleted successfully!")
        return True

    except Exception as e:
        st.error(f"Error deleting project: {e}")
        return False

# =============================================================================
# UI COMPONENTS
# =============================================================================

def render_project_form(project_data: Dict[str, Any] = None, form_key: str = "project_form") -> Dict[str, Any]:
    """Render project input form."""

    # Check write access
    has_write_access = st.session_state.get('portfolio_access_level') in ('owner', 'write')

    # Default values
    if project_data is None:
        project_data = {
            "Project ID": "",
            "Project": "",
            "Organization": "",
            "Project Manager": "",
            "BAC": 1000.0,
            "AC": 0.0,
            "Plan Start": "01/01/2025",
            "Plan Finish": "31/12/2025",
            "Use_Manual_PV": False,
            "Manual_PV": 0.0,
            "Use_Manual_EV": False,
            "Manual_EV": 0.0,
            # Per-project EVM settings (optional)
            "Curve Type": None,
            "Alpha": None,
            "Beta": None,
            "Inflation Rate": None
        }
    else:
        # Convert date columns from datetime to DD/MM/YYYY format if needed
        for date_col in ["Plan Start", "Plan Finish"]:
            if date_col in project_data:
                date_val = project_data[date_col]
                # If it's a pandas timestamp or datetime, convert to DD/MM/YYYY string
                if pd.notna(date_val) and hasattr(date_val, 'strftime'):
                    project_data[date_col] = date_val.strftime('%d/%m/%Y')
                elif isinstance(date_val, str):
                    # If it's already a string, try to parse and reformat to ensure DD/MM/YYYY
                    try:
                        parsed_date = date_parser.parse(date_val, dayfirst=True)
                        project_data[date_col] = parsed_date.strftime('%d/%m/%Y')
                    except:
                        # If parsing fails, leave as is
                        pass

    with st.form(form_key):
        # Row 1: Project ID and Project Name
        col1, col2 = st.columns(2)
        with col1:
            project_id = st.text_input("Project ID *", value=project_data.get("Project ID", ""))
        with col2:
            project_name = st.text_input("Project Name *", value=project_data.get("Project", ""))

        # Row 2: Organization and Project Manager
        col3, col4 = st.columns(2)
        with col3:
            organization = st.text_input("Organization", value=project_data.get("Organization", ""))
        with col4:
            project_manager = st.text_input("Project Manager", value=project_data.get("Project Manager", ""))

        # Row 3: Financial Data Header
        st.subheader("Financial Data")

        # Row 4: BAC and AC
        col5, col6 = st.columns(2)
        with col5:
            bac = st.number_input("BAC (Budget) *", min_value=0.0, value=float(project_data.get("BAC", 1000.0)), step=100.0)
        with col6:
            ac = st.number_input("AC (Actual Cost)", min_value=0.0, value=float(project_data.get("AC", 0.0)), step=50.0)

        # Row 5: Manual PV and Manual EV
        col7, col8 = st.columns(2)
        with col7:
            manual_pv = st.number_input("Manual PV (Optional)", min_value=0.0, value=float(project_data.get("Manual_PV") or 0.0), step=50.0, help="Leave at 0 for automatic calculation")
        with col8:
            manual_ev = st.number_input("Manual EV (Optional)", min_value=0.0, value=float(project_data.get("Manual_EV") or 0.0), step=50.0, help="Leave at 0 for automatic calculation")

        # Row 6: EVM Settings Header
        st.subheader("EVM Settings (Optional)")
        st.caption("Leave these fields empty (0 or default) to use global settings from File Management")

        # Row 7: Curve Type and Inflation Rate
        col_curve, col_inflation = st.columns(2)
        with col_curve:
            # Get the curve type value, handling None case
            curve_val = project_data.get("Curve Type")
            if curve_val is None or curve_val == "":
                curve_index = 0  # Empty/None = index 0 (empty option)
            elif str(curve_val).lower() == "linear":
                curve_index = 1  # "linear" = index 1
            else:
                curve_index = 2  # "s-curve" = index 2

            curve_type = st.selectbox(
                "Curve Type",
                options=["", "linear", "s-curve"],
                index=curve_index,
                key=f"{form_key}_curve_type",
                help="Leave empty to use global setting"
            )

        with col_inflation:
            inflation_rate = st.number_input(
                "Inflation Rate (%)",
                min_value=0.0,
                max_value=100.0,
                value=float(project_data.get("Inflation Rate") or 0.0),
                step=0.1,
                key=f"{form_key}_inflation_rate",
                help="Set to 0 to use global setting"
            )

        # Row 8: Alpha and Beta
        col_alpha, col_beta = st.columns(2)
        with col_alpha:
            alpha = st.number_input(
                "Alpha (S-curve parameter)",
                min_value=0.0,
                max_value=5.0,
                value=float(project_data.get("Alpha") or 0.0),
                step=0.1,
                key=f"{form_key}_alpha",
                help="Set to 0 to use global setting. Only used for S-curve."
            )
        with col_beta:
            beta = st.number_input(
                "Beta (S-curve parameter)",
                min_value=0.0,
                max_value=5.0,
                value=float(project_data.get("Beta") or 0.0),
                step=0.1,
                key=f"{form_key}_beta",
                help="Set to 0 to use global setting. Only used for S-curve."
            )

        # Row 9: Schedule Header
        st.subheader("Schedule")

        # Row 7: Plan Start and Plan Finish
        col9, col10 = st.columns(2)
        with col9:
            plan_start = st.text_input("Plan Start (DD/MM/YYYY) *", value=project_data.get("Plan Start", "01/01/2025"))
        with col10:
            plan_finish = st.text_input("Plan Finish (DD/MM/YYYY) *", value=project_data.get("Plan Finish", "31/12/2025"))

        submitted = st.form_submit_button("💾 Save Project", type="primary")

        if submitted and not has_write_access:
            st.error("You have read-only access to this portfolio. Cannot save project data.")
        elif submitted:
            # Auto-detect manual values based on non-zero input
            use_manual_pv = manual_pv > 0
            use_manual_ev = manual_ev > 0

            # Apply "use global" logic - convert empty/zero values to None
            # Empty string for curve_type means use global
            final_curve_type = None if (not curve_type or curve_type == "") else curve_type

            # Zero for numeric fields means use global
            final_inflation_rate = None if inflation_rate == 0.0 else inflation_rate
            final_alpha = None if alpha == 0.0 else alpha
            final_beta = None if beta == 0.0 else beta

            form_data = {
                "Project ID": project_id.strip(),
                "Project": project_name.strip(),
                "Organization": organization.strip(),
                "Project Manager": project_manager.strip(),
                "BAC": bac,
                "AC": ac,
                "Plan Start": plan_start.strip(),
                "Plan Finish": plan_finish.strip(),
                "Use_Manual_PV": use_manual_pv,
                "Manual_PV": manual_pv,
                "Use_Manual_EV": use_manual_ev,
                "Manual_EV": manual_ev,
                # Per-project EVM settings (None = use global)
                "Curve Type": final_curve_type,
                "Alpha": final_alpha,
                "Beta": final_beta,
                "Inflation Rate": final_inflation_rate
            }

            # Validate data
            errors = validate_project_data(form_data)
            if errors:
                for error in errors:
                    st.error(f"❌ {error}")
                return None

            return form_data

    return None

def render_data_overview(portfolio_id=None, status_date=None):
    """Render data overview metrics."""
    # Load from database if in database mode
    if USE_DATABASE and portfolio_id and status_date:
        from services.data_service import data_manager
        adapter = data_manager.get_data_adapter()
        df = adapter.get_projects_for_analysis(portfolio_id=portfolio_id, status_date=status_date)
    else:
        # Fallback to session state
        df = load_table(DEFAULT_DATASET_TABLE)

    if df.empty:
        st.info("📊 No project data available. Add your first project below!")
        return

    col1, col2, col3, col4 = st.columns(4)

    with col1:
        st.metric("Total Projects", len(df))

    with col2:
        total_bac = df["BAC"].sum()
        st.metric("Total BAC", f"{format_currency(total_bac)}")

    with col3:
        total_ac = df["AC"].sum()
        st.metric("Total AC", f"{format_currency(total_ac)}")

    with col4:
        if total_bac > 0:
            progress = (total_ac / total_bac) * 100
            st.metric("Overall Progress", f"{progress:.1f}%")
        else:
            st.metric("Overall Progress", "0%")

# =============================================================================
# MAIN APPLICATION
# =============================================================================

def main():
    """Main application function."""
    initialize_session_state()

    # Header
    st.markdown('<h1 class="main-header">📝 Enter Progress Data</h1>', unsafe_allow_html=True)
    st.markdown("Managing your project & portfolio data")

    # Portfolio & Period Selection
    st.markdown("---")
    portfolio_id, status_date = render_portfolio_context(show_period_selector=True)

    # Check write access
    _has_write = st.session_state.get('portfolio_access_level') in ('owner', 'write')
    if portfolio_id and not _has_write:
        st.info("You have **read-only** access to this portfolio. Data entry is disabled.")

    st.markdown("---")

    if not portfolio_id:
        st.warning("⚠️ Please select a portfolio to continue")
        st.info("Go to **Portfolio Management** to create or select a portfolio")
        st.stop()

    if not status_date:
        st.info("ℹ️ No data periods available for this portfolio")
        st.info("Go to **File Management** to upload data and create the first period")
        st.stop()

    # Check if EVM results exist for this period (informational only - don't block)
    if USE_DATABASE:
        db_manager = DatabaseDataManager()
        has_results = db_manager.check_evm_results_exist(portfolio_id, status_date)

        if not has_results:
            st.info("ℹ️ EVM calculations have not been run for this period yet. SPI/CPI metrics will not be available until you run batch calculations.")
            st.caption("💡 You can still view and edit project data. Run batch calculations from **Portfolio Management** when ready.")

    # Check if data exists - load from database or session state
    if USE_DATABASE and portfolio_id and status_date:
        # Load raw project data from database (not dependent on EVM results)
        try:
            from services.data_service import data_manager
            adapter = data_manager.get_data_adapter()
            current_df = adapter.get_projects_for_analysis(portfolio_id=portfolio_id, status_date=status_date)

            if current_df.empty:
                st.warning("⚠️ No project data found for this period.")
                st.info("Go to **Load Progress Data** to upload data for this period")
                st.stop()

            st.success(f"✅ Loaded {len(current_df)} projects for {status_date.strftime('%d-%b-%Y')}")
            has_data = True

        except Exception as e:
            logging.error(f"Error loading project data: {e}")
            st.error(f"Error loading data: {e}")
            st.stop()
    else:
        # Session state mode: Load from session state
        current_df = load_table(DEFAULT_DATASET_TABLE)
        has_data = (current_df is not None and not current_df.empty)

    if not has_data:
        st.warning("⚠️ No project data found. Please create a demo project or load data from the main page first.")

        if st.button("🚀 Create Demo Project", type="primary"):
            # Create 5 demo projects with different configurations
            demo_projects = [
                {
                    "Project ID": "Demo-1",
                    "Project": "Demo Project 1",
                    "Organization": "Demo Org",
                    "Project Manager": "Demo Manager",
                    "BAC": 1000.0,
                    "AC": 500.0,
                    "Plan Start": "01/01/2025",
                    "Plan Finish": "31/10/2025",
                    "Use_Manual_PV": False,
                    "Manual_PV": 0.0,
                    "Use_Manual_EV": False,
                    "Manual_EV": 0.0,
                    "Curve Type": "linear",
                    "Alpha": None,
                    "Beta": None,
                    "Inflation Rate": 14.0
                },
                {
                    "Project ID": "Demo-2",
                    "Project": "Demo Project 2",
                    "Organization": "Demo Org",
                    "Project Manager": "Demo Manager",
                    "BAC": 1000.0,
                    "AC": 500.0,
                    "Plan Start": "01/01/2025",
                    "Plan Finish": "31/10/2025",
                    "Use_Manual_PV": False,
                    "Manual_PV": 0.0,
                    "Use_Manual_EV": False,
                    "Manual_EV": 0.0,
                    "Curve Type": "s-curve",
                    "Alpha": 2.0,
                    "Beta": 2.0,
                    "Inflation Rate": 20.0
                },
                {
                    "Project ID": "Demo-3",
                    "Project": "Demo Project 3",
                    "Organization": "Demo Org",
                    "Project Manager": "Demo Manager",
                    "BAC": 1000.0,
                    "AC": 500.0,
                    "Plan Start": "01/01/2025",
                    "Plan Finish": "31/10/2025",
                    "Use_Manual_PV": True,
                    "Manual_PV": 600.0,
                    "Use_Manual_EV": True,
                    "Manual_EV": 500.0,
                    "Curve Type": None,
                    "Alpha": None,
                    "Beta": None,
                    "Inflation Rate": None
                },
                {
                    "Project ID": "Demo-4",
                    "Project": "Demo Project 4",
                    "Organization": "Demo Org",
                    "Project Manager": "Demo Manager",
                    "BAC": 1000.0,
                    "AC": 500.0,
                    "Plan Start": "01/01/2025",
                    "Plan Finish": "31/10/2025",
                    "Use_Manual_PV": False,
                    "Manual_PV": 0.0,
                    "Use_Manual_EV": True,
                    "Manual_EV": 500.0,
                    "Curve Type": "s-curve",
                    "Alpha": 1.0,
                    "Beta": 3.0,
                    "Inflation Rate": None
                },
                {
                    "Project ID": "Demo-5",
                    "Project": "Demo Project 5",
                    "Organization": "Demo Org",
                    "Project Manager": "Demo Manager",
                    "BAC": 1000.0,
                    "AC": 500.0,
                    "Plan Start": "01/01/2025",
                    "Plan Finish": "31/10/2025",
                    "Use_Manual_PV": False,
                    "Manual_PV": 0.0,
                    "Use_Manual_EV": True,
                    "Manual_EV": 500.0,
                    "Curve Type": "s-curve",
                    "Alpha": 1.0,
                    "Beta": 3.0,
                    "Inflation Rate": None
                }
            ]

            # Use proper persistence mechanism
            demo_df = pd.DataFrame(demo_projects)
            save_table_replace(demo_df, DEFAULT_DATASET_TABLE)

            # Set file type for batch processing
            st.session_state.file_type = "demo"

            # Update config
            st.session_state.config_dict.update({
                "curve_type": "S-curve",
                "data_date": "2025-05-30",
                "currency_symbol": "PKR",
                "currency_postfix": "Million"
            })

            st.success("✅ 5 Demo projects created successfully!")
            st.rerun()

        return

    # Data overview
    render_data_overview(portfolio_id, status_date)
    st.divider()

    # Create tabs
    tab1, tab2, tab3 = st.tabs(["📋 View & Edit Data", "➕ Add New Project", "📦 Bulk Operations"])

    with tab1:  # View & Edit Data
        st.subheader("Project Data")

        # Load and display dataframe with selection
        # Load from database if in database mode
        if USE_DATABASE and portfolio_id and status_date:
            from services.data_service import data_manager
            adapter = data_manager.get_data_adapter()
            current_df = adapter.get_projects_for_analysis(portfolio_id=portfolio_id, status_date=status_date)
        else:
            current_df = load_table(DEFAULT_DATASET_TABLE)

        if not current_df.empty:
            # Try to get SPI and CPI from batch results or database
            current_df = current_df.copy()

            # In database mode, try to get EVM results from database first
            if USE_DATABASE and portfolio_id and status_date:
                try:
                    evm_results = adapter.db_manager.get_evm_results_for_period(
                        portfolio_id=portfolio_id,
                        status_date=status_date
                    )

                    if not evm_results.empty and 'project_id' in evm_results.columns:
                        # Select only the EVM metrics we need
                        evm_columns = ['project_id']
                        if 'spi' in evm_results.columns:
                            evm_columns.append('spi')
                        if 'cpi' in evm_results.columns:
                            evm_columns.append('cpi')
                        if 'ev' in evm_results.columns:
                            evm_columns.append('ev')
                        if 'pv' in evm_results.columns:
                            evm_columns.append('pv')

                        if len(evm_columns) > 1:  # More than just project_id
                            evm_df = evm_results[evm_columns].copy()

                            # Rename columns to match display format
                            evm_df = evm_df.rename(columns={
                                'project_id': 'Project ID',
                                'spi': 'SPI',
                                'cpi': 'CPI',
                                'ev': 'EV',
                                'pv': 'PV'
                            })

                            # Ensure both Project ID columns are strings
                            current_df['Project ID'] = current_df['Project ID'].astype(str)
                            evm_df['Project ID'] = evm_df['Project ID'].astype(str)

                            # Merge the EVM data
                            current_df = current_df.merge(
                                evm_df,
                                on='Project ID',
                                how='left'
                            )
                            st.info("📊 Using calculated EVM results from database")
                        else:
                            st.warning("⚠️ No calculated EVM metrics found in database. Please run 'Batch EVM Calculation' first.")
                            current_df['SPI'] = 0.0
                            current_df['CPI'] = 0.0
                    else:
                        st.warning("⚠️ No calculated EVM results found in database. Please run 'Batch EVM Calculation' first.")
                        current_df['SPI'] = 0.0
                        current_df['CPI'] = 0.0
                except Exception as e:
                    st.warning(f"⚠️ Could not load EVM results from database: {e}")
                    current_df['SPI'] = 0.0
                    current_df['CPI'] = 0.0
            # Check if batch results are available with EVM calculations (session mode)
            elif hasattr(st.session_state, 'batch_results') and st.session_state.batch_results is not None:
                batch_df = st.session_state.batch_results.copy()

                # Merge EVM metrics from batch results
                # The batch results use 'project_id' (lowercase) as the column name
                project_id_col = None
                if 'project_id' in batch_df.columns:
                    project_id_col = 'project_id'
                elif 'Project ID' in batch_df.columns:
                    project_id_col = 'Project ID'

                if project_id_col:
                    # Select relevant EVM columns from batch results
                    evm_columns = [project_id_col]

                    # Rename batch results columns to match our display names
                    column_renames = {}
                    if 'spi' in batch_df.columns:
                        evm_columns.append('spi')
                        column_renames['spi'] = 'SPI'
                    if 'cpi' in batch_df.columns:
                        evm_columns.append('cpi')
                        column_renames['cpi'] = 'CPI'
                    if 'ev' in batch_df.columns:
                        evm_columns.append('ev')
                        column_renames['ev'] = 'EV'
                    if 'pv' in batch_df.columns:
                        evm_columns.append('pv')
                        column_renames['pv'] = 'PV'

                    if len(evm_columns) > 1:  # More than just project ID
                        # Select only the columns we need
                        batch_evm_df = batch_df[evm_columns].copy()

                        # Rename columns
                        if column_renames:
                            batch_evm_df = batch_evm_df.rename(columns=column_renames)

                        # Rename project_id column to match current_df
                        if project_id_col == 'project_id':
                            batch_evm_df = batch_evm_df.rename(columns={'project_id': 'Project ID'})

                        # Ensure both Project ID columns are strings to avoid merge type errors
                        current_df['Project ID'] = current_df['Project ID'].astype(str)
                        batch_evm_df['Project ID'] = batch_evm_df['Project ID'].astype(str)

                        # Merge the EVM data
                        current_df = current_df.merge(
                            batch_evm_df,
                            on='Project ID',
                            how='left'
                        )
                        st.info("📊 Using EVM calculations from Portfolio Analysis batch results")
                    else:
                        st.warning("⚠️ No EVM metrics found in batch results. Please run 'Batch EVM Calculation' in Portfolio Analysis first.")
                        # Add empty columns for consistency
                        current_df['SPI'] = 0.0
                        current_df['CPI'] = 0.0
                else:
                    st.warning("⚠️ Batch results don't contain project identifier column. Please run 'Batch EVM Calculation' in Portfolio Analysis first.")
                    # Add empty columns for consistency
                    current_df['SPI'] = 0.0
                    current_df['CPI'] = 0.0
            else:
                st.warning("⚠️ No batch results found. Please run 'Batch EVM Calculation' in Portfolio Analysis to get SPI and CPI values.")
                # Add empty columns for consistency
                current_df['SPI'] = 0.0
                current_df['CPI'] = 0.0

            # Add comprehensive filters
            filter_header_col1, filter_header_col2 = st.columns([3, 1])
            with filter_header_col1:
                st.subheader("🔧 Filters")
            with filter_header_col2:
                if st.button("🔄 Clear All Filters", key="clear_filters"):
                    st.rerun()

            filter_col1, filter_col2, filter_col3, filter_col4 = st.columns(4)

            with filter_col1:
                # Project ID search filter
                search_project_id = st.text_input("🔍 Project ID", placeholder="Search by Project ID...")

            with filter_col2:
                # Organization filter
                org_options = ["All"] + sorted(current_df["Organization"].dropna().unique().tolist())
                selected_org = st.selectbox("🏢 Organization", options=org_options)

            with filter_col3:
                # Plan Start date range filter
                if 'Plan Start' in current_df.columns:
                    try:
                        # Convert Plan Start to datetime if it's not already
                        if current_df['Plan Start'].dtype == 'object':
                            current_df['Plan Start'] = pd.to_datetime(current_df['Plan Start'], format='%d/%m/%Y', errors='coerce')

                        # Get min and max dates, ignoring NaT values
                        valid_dates = current_df['Plan Start'].dropna()
                        if len(valid_dates) > 0:
                            min_date = valid_dates.min()
                            max_date = valid_dates.max()

                            # Ensure we have valid datetime objects
                            if pd.notna(min_date) and pd.notna(max_date) and hasattr(min_date, 'date'):
                                date_range = st.date_input(
                                    "📅 Plan Start Range",
                                    value=(min_date.date(), max_date.date()),
                                    min_value=min_date.date(),
                                    max_value=max_date.date()
                                )
                            else:
                                st.text("📅 Plan Start Range")
                                st.caption("No valid dates found")
                                date_range = None
                        else:
                            st.text("📅 Plan Start Range")
                            st.caption("No valid dates found")
                            date_range = None
                    except Exception as e:
                        st.text("📅 Plan Start Range")
                        st.caption(f"Date parsing error: {str(e)}")
                        date_range = None
                else:
                    date_range = None

            with filter_col4:
                # SPI/CPI range filters
                if 'SPI' in current_df.columns and 'CPI' in current_df.columns:
                    metric_filter = st.selectbox("📊 Performance Filter",
                        options=["All", "SPI < 1.0 (Behind Schedule)", "SPI >= 1.0 (On/Ahead Schedule)",
                                "CPI < 1.0 (Over Budget)", "CPI >= 1.0 (On/Under Budget)",
                                "SPI < 1.0 AND CPI < 1.0 (Critical)", "SPI >= 1.0 AND CPI >= 1.0 (Healthy)"])
                else:
                    metric_filter = "All"

            # Store original dataframe and indices for mapping
            original_df = current_df.copy()

            # Apply filters
            filtered_df = current_df.copy()

            # Project ID filter
            if search_project_id:
                filtered_df = filtered_df[filtered_df["Project ID"].astype(str).str.contains(search_project_id, case=False, na=False)]

            # Organization filter
            if selected_org != "All":
                filtered_df = filtered_df[filtered_df["Organization"] == selected_org]

            # Date range filter
            if date_range and len(date_range) == 2:
                try:
                    start_date, end_date = date_range
                    # Ensure Plan Start is datetime before filtering
                    if 'Plan Start' in filtered_df.columns:
                        if filtered_df['Plan Start'].dtype == 'object':
                            filtered_df['Plan Start'] = pd.to_datetime(filtered_df['Plan Start'], format='%d/%m/%Y', errors='coerce')

                        # Filter only rows with valid dates
                        valid_date_mask = filtered_df['Plan Start'].notna()
                        if valid_date_mask.any():
                            filtered_df = filtered_df[
                                valid_date_mask &
                                (filtered_df['Plan Start'].dt.date >= start_date) &
                                (filtered_df['Plan Start'].dt.date <= end_date)
                            ]
                except Exception as e:
                    st.error(f"Error filtering by date range: {str(e)}")

            # Performance metric filter
            if metric_filter != "All" and 'SPI' in filtered_df.columns and 'CPI' in filtered_df.columns:
                if metric_filter == "SPI < 1.0 (Behind Schedule)":
                    filtered_df = filtered_df[filtered_df['SPI'] < 1.0]
                elif metric_filter == "SPI >= 1.0 (On/Ahead Schedule)":
                    filtered_df = filtered_df[filtered_df['SPI'] >= 1.0]
                elif metric_filter == "CPI < 1.0 (Over Budget)":
                    filtered_df = filtered_df[filtered_df['CPI'] < 1.0]
                elif metric_filter == "CPI >= 1.0 (On/Under Budget)":
                    filtered_df = filtered_df[filtered_df['CPI'] >= 1.0]
                elif metric_filter == "SPI < 1.0 AND CPI < 1.0 (Critical)":
                    filtered_df = filtered_df[(filtered_df['SPI'] < 1.0) & (filtered_df['CPI'] < 1.0)]
                elif metric_filter == "SPI >= 1.0 AND CPI >= 1.0 (Healthy)":
                    filtered_df = filtered_df[(filtered_df['SPI'] >= 1.0) & (filtered_df['CPI'] >= 1.0)]

            # Update current_df to filtered results
            current_df = filtered_df

            # Show filter results
            if len(current_df) != len(original_df):
                st.info(f"📊 Showing {len(current_df)} of {len(original_df)} projects after filtering")

            if current_df.empty:
                st.warning("No projects match the selected filters. Please adjust your filter criteria.")
            else:
                # Create a display version with formatted numbers
                display_df = current_df.copy()

                # Format currency columns
                display_df["BAC"] = display_df["BAC"].apply(format_currency)
                display_df["AC"] = display_df["AC"].apply(format_currency)
                if "Manual_PV" in display_df.columns:
                    display_df["Manual_PV"] = display_df["Manual_PV"].apply(format_currency)
                if "Manual_EV" in display_df.columns:
                    display_df["Manual_EV"] = display_df["Manual_EV"].apply(format_currency)
                if "PV" in display_df.columns:
                    display_df["PV"] = display_df["PV"].apply(format_currency)
                if "EV" in display_df.columns:
                    display_df["EV"] = display_df["EV"].apply(format_currency)

                # Format SPI and CPI as ratios with 3 decimal places
                if "SPI" in display_df.columns:
                    display_df["SPI"] = display_df["SPI"].apply(lambda x: f"{x:.3f}" if pd.notna(x) else "N/A")
                if "CPI" in display_df.columns:
                    display_df["CPI"] = display_df["CPI"].apply(lambda x: f"{x:.3f}" if pd.notna(x) else "N/A")

                # Format dates back to string for display
                if 'Plan Start' in display_df.columns:
                    try:
                        # Only use dt accessor if column is actually datetime
                        if pd.api.types.is_datetime64_any_dtype(display_df['Plan Start']):
                            display_df['Plan Start'] = display_df['Plan Start'].dt.strftime('%d/%m/%Y')
                        else:
                            # If it's already string or other type, keep as is
                            display_df['Plan Start'] = display_df['Plan Start'].astype(str)
                    except:
                        # Fallback to string conversion
                        display_df['Plan Start'] = display_df['Plan Start'].astype(str)

                if 'Plan Finish' in display_df.columns:
                    try:
                        # Only use dt accessor if column is actually datetime
                        if pd.api.types.is_datetime64_any_dtype(display_df['Plan Finish']):
                            display_df['Plan Finish'] = display_df['Plan Finish'].dt.strftime('%d/%m/%Y')
                        else:
                            # If it's already string or other type, keep as is
                            display_df['Plan Finish'] = display_df['Plan Finish'].astype(str)
                    except:
                        # Fallback to string conversion
                        display_df['Plan Finish'] = display_df['Plan Finish'].astype(str)

                # Select key columns for display (avoid showing too many columns)
                display_columns = ['Project ID', 'Project', 'Organization', 'BAC', 'AC', 'Plan Start', 'Plan Finish']
                if 'SPI' in display_df.columns:
                    display_columns.append('SPI')
                if 'CPI' in display_df.columns:
                    display_columns.append('CPI')

                # Only show columns that exist in the dataframe
                display_columns = [col for col in display_columns if col in display_df.columns]
                display_df = display_df[display_columns]

                # Display table
                event = st.dataframe(
                    display_df,
                    width='stretch',
                    on_select="rerun",
                    selection_mode="single-row"
                )

                # Handle row selection
                if event.selection.rows:
                    selected_display_idx = event.selection.rows[0]
                    # Map the display index back to the original dataframe index
                    actual_idx = current_df.index[selected_display_idx]
                    st.session_state.selected_row_index = actual_idx

                    if _has_write:
                        col1, col2 = st.columns(2)
                        with col1:
                            if st.button("✏️ Edit Selected Project", type="primary"):
                                st.session_state.edit_mode = True
                                st.rerun()

                        with col2:
                            if st.button("🗑️ Delete Selected Project", type="secondary"):
                                if delete_project(actual_idx):
                                    st.session_state.selected_row_index = None
                                    st.rerun()

                    # Show edit form if in edit mode
                    if st.session_state.edit_mode and st.session_state.selected_row_index is not None:
                        st.divider()
                        st.subheader("Edit Project")

                        current_data = original_df.iloc[st.session_state.selected_row_index].to_dict()
                        updated_data = render_project_form(current_data, "edit_form")

                        if updated_data:
                            if update_project(st.session_state.selected_row_index, updated_data):
                                st.session_state.edit_mode = False
                                st.session_state.selected_row_index = None
                                st.rerun()

                        if st.button("❌ Cancel Edit"):
                            st.session_state.edit_mode = False
                            st.rerun()

    with tab2:  # Add New Project
        st.subheader("Add New Project")

        # Clear stale flags when tab is first viewed (not during a button click rerun)
        # Use a unique key to detect tab switches
        current_tab_key = "tab2_add_new_project"
        if st.session_state.get("last_active_tab") != current_tab_key:
            # Tab just switched, clear stale flags
            st.session_state.project_just_added = False
            st.session_state.new_project_id = None
            st.session_state.last_active_tab = current_tab_key

        # Don't render form if project was just added - show message instead
        if st.session_state.get("project_just_added", False):
            project_id = st.session_state.get("new_project_id", "")
            st.success(f"🎉 Project '{project_id}' added successfully!")
            st.info("👈 Please switch to the 'View & Edit Data' tab to see your new project.")

            if st.button("➕ Add Another Project"):
                # Reset flags to show form again
                st.session_state.project_just_added = False
                st.session_state.new_project_id = None
                st.rerun()
        else:
            # Show the form only if no project was just added
            new_project_data = render_project_form(form_key="add_form")

            if new_project_data:
                if add_new_project(new_project_data):
                    st.rerun()

    with tab3:  # Bulk Operations
        st.subheader("Bulk Operations")

        # Export CSV section
        st.write("**Export Data to CSV**")
        export_file_name = st.text_input(
            "Enter file name for export",
            value="project_data_export",
            help="Enter the desired file name for the export (with or without .csv extension)"
        )

        # Prepare CSV data for single-click download
        # Load from database if in database mode
        if USE_DATABASE and portfolio_id and status_date:
            from services.data_service import data_manager
            adapter = data_manager.get_data_adapter()
            export_df = adapter.get_projects_for_analysis(portfolio_id=portfolio_id, status_date=status_date)
        else:
            export_df = load_table(DEFAULT_DATASET_TABLE)

        if not export_df.empty:
            # Use the exact file name provided by user
            final_file_name = export_file_name.strip()

            # Add .csv extension only if not already present
            if not final_file_name.endswith('.csv'):
                final_file_name = f"{final_file_name}.csv"

            # Create comprehensive export with all available fields (same as File Management)
            export_df_complete = export_df.copy()

            # If batch results are available, merge them to include EVM calculations
            if hasattr(st.session_state, 'batch_results') and st.session_state.batch_results is not None and not st.session_state.batch_results.empty:
                batch_df = st.session_state.batch_results.copy()

                # Identify the project ID column in batch results
                project_id_col = None
                if 'project_id' in batch_df.columns:
                    project_id_col = 'project_id'
                elif 'Project ID' in batch_df.columns:
                    project_id_col = 'Project ID'

                if project_id_col:
                    # Rename to standard column name if needed
                    if project_id_col == 'project_id':
                        batch_df = batch_df.rename(columns={'project_id': 'Project ID'})

                    # Remove duplicate columns from batch_df that already exist in export_df
                    # Keep only the EVM calculation columns from batch results
                    duplicate_cols = []
                    for col in batch_df.columns:
                        # Convert column names to lowercase for comparison
                        col_lower = col.lower().replace(' ', '_')
                        export_cols_lower = [c.lower().replace(' ', '_') for c in export_df_complete.columns]

                        # Skip the merge key (Project ID) and duplicate columns
                        if col != 'Project ID' and col_lower in export_cols_lower:
                            duplicate_cols.append(col)

                    # Drop duplicate columns from batch_df
                    if duplicate_cols:
                        batch_df = batch_df.drop(columns=duplicate_cols)

                    # Ensure both are strings for merge
                    export_df_complete['Project ID'] = export_df_complete['Project ID'].astype(str)
                    batch_df['Project ID'] = batch_df['Project ID'].astype(str)

                    # Merge batch results into export
                    # Use 'left' join to keep all original projects even if no batch results
                    export_df_complete = export_df_complete.merge(
                        batch_df,
                        on='Project ID',
                        how='left',
                        suffixes=('', '_batch')
                    )

            # Convert to CSV
            csv = export_df_complete.to_csv(index=False)

            # Single-click download button
            st.download_button(
                label="📥 Export to CSV",
                data=csv,
                file_name=final_file_name,
                mime="text/csv",
                help="Download complete data with EVM calculations if batch processing was run"
            )

        st.divider()

        # Clear All Data section
        st.write("**Clear All Data**")
        if not _has_write:
            st.info("Write access required to clear data.")
        else:
            st.warning("⚠️ This action will permanently delete all project data. This cannot be undone!")

        # Initialize confirmation state if not exists
        if 'confirm_clear_data' not in st.session_state:
            st.session_state.confirm_clear_data = False

        if _has_write and st.button("🗑️ Clear All Data", type="secondary"):
            st.session_state.confirm_clear_data = True

        if st.session_state.confirm_clear_data:
            st.error("🚨 Are you sure you want to delete all project data?")
            col1, col2 = st.columns(2)

            with col1:
                if st.button("✅ Yes, Delete All", type="primary"):
                    current_df = load_table(DEFAULT_DATASET_TABLE)
                    if not current_df.empty:
                        # Save empty dataframe to clear data
                        empty_df = pd.DataFrame()
                        save_table_replace(empty_df, DEFAULT_DATASET_TABLE)
                        st.session_state.confirm_clear_data = False
                        st.success("✅ All data cleared successfully!")
                        st.rerun()
                    else:
                        st.info("No data to clear.")
                        st.session_state.confirm_clear_data = False

            with col2:
                if st.button("❌ Cancel"):
                    st.session_state.confirm_clear_data = False
                    st.rerun()

if __name__ == "__main__":
    main()
# Show user info in sidebar
from utils.auth import show_user_info_sidebar
show_user_info_sidebar()

"""
Load Progress Data - Portfolio Management Suite
Upload and load project progress data from CSV files.
"""

import streamlit as st
import pandas as pd
import json
import logging
from datetime import datetime
from typing import Dict, List, Tuple, Any
import os
import traceback
from utils.auth import check_authentication, require_page_access
from config.constants import USE_DATABASE
from services.data_service import data_manager
from services.db_data_service import DatabaseDataManager
from database.db_connection import get_db
from utils.portfolio_context import render_portfolio_selector, get_user_portfolios

# Setup logging
logger = logging.getLogger(__name__)

# Page configuration
st.set_page_config(
    page_title="Load Progress Data - Portfolio Suite",
    page_icon="📁",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# Check authentication and page access
if not check_authentication():
    st.stop()

require_page_access('file_management', 'Load Progress Data')

# Note: Some advanced functionality (like database operations)
# will be handled through session state integration with Portfolio Analysis

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
    .file-section {
        background: rgba(255, 255, 255, 0.1);
        padding: 1rem;
        border-radius: 15px;
        margin: 0.5rem 0;
        border: 1px solid rgba(255, 255, 255, 0.2);
    }
    .section-header {
        font-size: 1.2em;
        font-weight: 600;
        color: #4A90E2;
        margin-bottom: 0.75rem;
        padding-bottom: 0.5rem;
        border-bottom: 2px solid #4A90E2;
    }
    .nav-button {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        padding: 1rem 2rem;
        border-radius: 10px;
        text-decoration: none;
        font-weight: 600;
        margin: 0.5rem;
        display: inline-block;
        transition: all 0.3s ease;
        border: none;
        cursor: pointer;
    }
    .nav-button:hover {
        transform: translateY(-2px);
        box-shadow: 0 12px 30px rgba(102, 126, 234, 0.4);
    }
</style>
""", unsafe_allow_html=True)

# Initialize session state - only initialize if they don't exist
# This prevents overwriting data added from other pages (like Manual Data Entry)
if 'session_initialized' not in st.session_state:
    st.session_state.session_initialized = False

if 'data_df' not in st.session_state:
    st.session_state.data_df = None

if 'data_loaded' not in st.session_state:
    st.session_state.data_loaded = False

if 'raw_csv_df' not in st.session_state:
    st.session_state.raw_csv_df = None

if 'batch_results' not in st.session_state:
    st.session_state.batch_results = None

if 'config_dict' not in st.session_state:
    st.session_state.config_dict = {}

if 'config_loaded_flag' not in st.session_state:
    st.session_state.config_loaded_flag = False

if 'original_filename' not in st.session_state:
    st.session_state.original_filename = None

if 'file_type' not in st.session_state:
    st.session_state.file_type = None

if 'processed_file_info' not in st.session_state:
    st.session_state.processed_file_info = None

# Synchronize data_loaded flag with actual data state
# This ensures consistency when navigating between pages
if st.session_state.data_df is not None and not st.session_state.data_df.empty:
    st.session_state.data_loaded = True
else:
    st.session_state.data_loaded = False

# CRITICAL: Validate config persistence after rerun
if st.session_state.config_loaded_flag and 'controls' not in st.session_state.config_dict:
    st.error("🚨 CRITICAL BUG: Config was marked as loaded but is missing from session state!")
    st.error("This is a session state persistence issue on the hosted environment.")
    st.info("Workaround: Re-upload your JSON file to reload configuration.")

def render_data_source_section():
    """Render A. Data Source section."""
    st.markdown('<div class="file-section">', unsafe_allow_html=True)
    st.markdown('<div class="section-header">A. Data Source</div>', unsafe_allow_html=True)

    # Simple data source selection
    data_source = st.radio(
        "Select Data Source",
        options=["Load JSON", "Load CSV"],
        index=0,
        key="data_source_radio",
        help="Choose JSON for comprehensive data with configuration, or CSV for data-only imports"
    )

    selected_table = None
    column_mapping = {}
    df = None

    # Dynamic file upload info based on selection
    if data_source == "Load JSON":
        st.info("💡 Upload JSON files exported from this application or compatible EVM tools")
        file_types = ["json"]
        help_text = "JSON files contain both data and configuration settings"
    elif data_source == "Load CSV":
        st.info("💡 Upload CSV files containing project data. Configuration will be set to defaults")
        file_types = ["csv"]
        help_text = "CSV files contain data only - you'll need to map columns to EVM fields"

    # File uploader for JSON and CSV options
    uploaded_file = st.file_uploader(
        "Choose file",
        type=file_types,
        key="unified_file_uploader",
        help=help_text,
        label_visibility="visible"
    )

    # CSV Column Mapping Interface (always visible when CSV is selected)
    if data_source == "Load CSV":
        st.markdown("### 🔗 Column Mapping")
        st.info("Map your CSV columns to the required EVM fields below:")

        # Get available columns from uploaded file or session state
        csv_columns = ['']  # Empty option first
        if uploaded_file is not None:
            try:
                # Try different parsing options for problematic CSV files
                temp_df = None

                # Try standard parsing first
                try:
                    uploaded_file.seek(0)  # Reset file pointer to beginning
                    temp_df = pd.read_csv(uploaded_file)
                except pd.errors.EmptyDataError:
                    st.error("❌ CSV file appears to be empty or has no columns")
                    return None, None, {}
                except Exception as e:
                    # Try with different encoding
                    try:
                        uploaded_file.seek(0)  # Reset file pointer
                        temp_df = pd.read_csv(uploaded_file, encoding='latin-1')
                    except Exception as e2:
                        # Try with different separator
                        try:
                            uploaded_file.seek(0)  # Reset file pointer
                            temp_df = pd.read_csv(uploaded_file, sep=';')
                        except Exception as e3:
                            st.error(f"❌ Could not parse CSV file. Error: {str(e)}")
                            st.info("💡 Try saving your file as UTF-8 encoded CSV with comma separators")
                            return None, None, {}

                if temp_df is not None and not temp_df.empty:
                    csv_columns.extend(list(temp_df.columns))
                else:
                    st.warning("⚠️ CSV file contains no data")
                    return None, None, {}

            except Exception as e:
                st.error(f"❌ Error reading CSV file: {str(e)}")
                return None, None, {}
        elif st.session_state.get('raw_csv_df') is not None:
            csv_columns.extend(list(st.session_state.raw_csv_df.columns))

        if len(csv_columns) > 1:  # More than just empty option
            # Raw File Preview - BEFORE column mapping
            with st.expander("📄 Preview Raw CSV File", expanded=False):
                st.caption("View the first 10 rows of your uploaded CSV file as-is")
                try:
                    preview_raw_df = None
                    if uploaded_file is not None:
                        uploaded_file.seek(0)
                        preview_raw_df = pd.read_csv(uploaded_file)
                    elif st.session_state.get('raw_csv_df') is not None:
                        preview_raw_df = st.session_state.raw_csv_df

                    if preview_raw_df is not None and not preview_raw_df.empty:
                        st.info(f"📊 Total: **{len(preview_raw_df)}** rows × **{len(preview_raw_df.columns)}** columns")

                        # Show column names
                        st.markdown("**Available Columns:**")
                        st.code(", ".join(preview_raw_df.columns.tolist()))

                        # Show preview of first 10 rows
                        st.markdown("**Data Preview (first 10 rows):**")
                        st.dataframe(preview_raw_df.head(10), width='stretch')
                    else:
                        st.info("No file uploaded yet")
                except Exception as e:
                    st.error(f"Error reading file: {str(e)}")

            st.markdown("---")

            col1, col2 = st.columns(2)

            with col1:
                st.markdown("**Required Fields:**")
                pid_col = st.selectbox("Project ID", csv_columns, key="csv_pid_col")
                pname_col = st.selectbox("Project Name", csv_columns, key="csv_pname_col")
                org_col = st.selectbox("Organization", csv_columns, key="csv_org_col")
                pm_col = st.selectbox("Project Manager", csv_columns, key="csv_pm_col")

            with col2:
                st.markdown("**Date & Financial Fields:**")
                st_col = st.selectbox("Plan Start Date", csv_columns, key="csv_st_col")
                fn_col = st.selectbox("Plan Finish Date", csv_columns, key="csv_fn_col")
                bac_col = st.selectbox("BAC (Budget at Completion)", csv_columns, key="csv_bac_col")
                ac_col = st.selectbox("AC (Actual Cost)", csv_columns, key="csv_ac_col")

            # Optional Fields Section
            st.markdown("---")
            st.markdown("**Optional Fields:**")
            st.caption("Leave empty if not present in your CSV. These will use default/global settings if not mapped.")

            col3, col4 = st.columns(2)

            with col3:
                st.markdown("*Manual EVM Values:*")
                manual_pv_col = st.selectbox("Manual PV", csv_columns, key="csv_manual_pv_col", help="Manually entered Planned Value (overrides calculated PV)")
                manual_ev_col = st.selectbox("Manual EV", csv_columns, key="csv_manual_ev_col", help="Manually entered Earned Value (overrides calculated EV)")

            with col4:
                st.markdown("*Per-Project Settings:*")
                curve_type_col = st.selectbox("Curve Type", csv_columns, key="csv_curve_type_col", help="linear or s-curve (empty = use global)")
                alpha_col = st.selectbox("Alpha (S-curve)", csv_columns, key="csv_alpha_col", help="S-curve parameter (0 or empty = use global)")
                beta_col = st.selectbox("Beta (S-curve)", csv_columns, key="csv_beta_col", help="S-curve parameter (0 or empty = use global)")
                inflation_rate_col = st.selectbox("Inflation Rate (%)", csv_columns, key="csv_inflation_rate_col", help="Per-project inflation rate (0 or empty = use global)")

            # Build column mapping
            if pid_col: column_mapping['pid_col'] = pid_col
            if pname_col: column_mapping['pname_col'] = pname_col
            if org_col: column_mapping['org_col'] = org_col
            if pm_col: column_mapping['pm_col'] = pm_col
            if st_col: column_mapping['st_col'] = st_col
            if fn_col: column_mapping['fn_col'] = fn_col
            if bac_col: column_mapping['bac_col'] = bac_col
            if ac_col: column_mapping['ac_col'] = ac_col

            # Optional mappings
            if manual_pv_col and manual_pv_col != '': column_mapping['manual_pv_col'] = manual_pv_col
            if manual_ev_col and manual_ev_col != '': column_mapping['manual_ev_col'] = manual_ev_col
            if curve_type_col and curve_type_col != '': column_mapping['curve_type_col'] = curve_type_col
            if alpha_col and alpha_col != '': column_mapping['alpha_col'] = alpha_col
            if beta_col and beta_col != '': column_mapping['beta_col'] = beta_col
            if inflation_rate_col and inflation_rate_col != '': column_mapping['inflation_rate_col'] = inflation_rate_col

            # Validation
            required_fields = ['pid_col', 'pname_col', 'org_col', 'pm_col', 'st_col', 'fn_col', 'bac_col', 'ac_col']
            missing_fields = [field for field in required_fields if field not in column_mapping]

            if missing_fields:
                st.warning(f"⚠️ Please map all required fields. Missing: {', '.join(missing_fields)}")
            else:
                st.success("✅ All required fields mapped successfully!")

                # Data Preview Expander
                with st.expander("🔍 Preview Data Before Loading", expanded=False):
                    st.caption("Preview the first 5 rows of your data with the current column mapping")

                    try:
                        # Get the uploaded file or session state data
                        preview_df = None
                        if uploaded_file is not None:
                            uploaded_file.seek(0)
                            preview_df = pd.read_csv(uploaded_file).head(5)
                        elif st.session_state.get('raw_csv_df') is not None:
                            preview_df = st.session_state.raw_csv_df.head(5)

                        if preview_df is not None and not preview_df.empty:
                            # Create a mapped preview dataframe
                            mapped_preview_df = pd.DataFrame()

                            # Map each required field
                            field_display_names = {
                                'pid_col': 'Project ID',
                                'pname_col': 'Project Name',
                                'org_col': 'Organization',
                                'pm_col': 'Project Manager',
                                'st_col': 'Plan Start',
                                'fn_col': 'Plan Finish',
                                'bac_col': 'BAC',
                                'ac_col': 'AC',
                                'manual_pv_col': 'Manual PV',
                                'manual_ev_col': 'Manual EV',
                                'curve_type_col': 'Curve Type',
                                'alpha_col': 'Alpha',
                                'beta_col': 'Beta',
                                'inflation_rate_col': 'Inflation Rate'
                            }

                            # Build the preview with mapped columns
                            for field_key, display_name in field_display_names.items():
                                if field_key in column_mapping and column_mapping[field_key] in preview_df.columns:
                                    mapped_preview_df[display_name] = preview_df[column_mapping[field_key]]

                            # Display the preview
                            if not mapped_preview_df.empty:
                                st.dataframe(mapped_preview_df, width='stretch')
                                st.caption(f"Showing {len(preview_df)} of {len(temp_df) if 'temp_df' in locals() else 'unknown'} total rows")
                            else:
                                st.warning("No mapped columns to preview")
                        else:
                            st.info("No data available for preview")

                    except Exception as e:
                        st.error(f"Error generating preview: {str(e)}")

                # Database Mode: Additional fields
                portfolio_id = None
                status_date = None

                if USE_DATABASE:
                    st.markdown("---")
                    st.markdown("**📊 Database Mode Settings:**")
                    st.info("Database mode is enabled. Please select portfolio and status date.")

                    col_db1, col_db2 = st.columns(2)

                    with col_db1:
                        # Portfolio selector - use the same function as portfolio context
                        from utils.portfolio_context import get_user_portfolios

                        portfolios_df = get_user_portfolios()

                        if portfolios_df.empty:
                            st.warning("⚠️ No portfolios found. Please create a portfolio first.")
                            st.info("Go to **Portfolio Management** to create a portfolio")
                        else:
                            portfolio_options = {
                                f"{row['portfolio_name']} (ID: {row['portfolio_id']})": row['portfolio_id']
                                for _, row in portfolios_df.iterrows()
                            }
                            selected_portfolio = st.selectbox(
                                "Select Portfolio",
                                options=list(portfolio_options.keys()),
                                key="csv_portfolio_selector"
                            )
                            portfolio_id = portfolio_options[selected_portfolio]

                    with col_db2:
                        # Status date input
                        status_date = st.date_input(
                            "Status Date",
                            value=datetime.now().date(),
                            min_value=datetime(2000, 1, 1).date(),
                            max_value=datetime(2035, 12, 31).date(),
                            key="csv_status_date",
                            help="The date for which this data represents project status"
                        )

                # Load Data button (requires write access)
                _has_write_access = st.session_state.get('portfolio_access_level') in ('owner', 'write')
                if not _has_write_access:
                    st.warning("Write access required to load data into this portfolio.")
                elif st.button("📊 Load Data with Mapping", type="primary", key="load_csv_with_mapping"):
                    if uploaded_file is not None or st.session_state.get('raw_csv_df') is not None:
                        try:
                            # Use uploaded file or session state data
                            if uploaded_file is not None:
                                # Try robust parsing again
                                df = None
                                try:
                                    uploaded_file.seek(0)  # Reset file pointer to beginning
                                    df = pd.read_csv(uploaded_file)
                                except pd.errors.EmptyDataError:
                                    st.error("❌ CSV file appears to be empty or has no columns")
                                    st.stop()
                                except Exception as e:
                                    # Try with different encoding
                                    try:
                                        uploaded_file.seek(0)
                                        df = pd.read_csv(uploaded_file, encoding='latin-1')
                                    except Exception as e2:
                                        # Try with different separator
                                        try:
                                            uploaded_file.seek(0)
                                            df = pd.read_csv(uploaded_file, sep=';')
                                        except Exception as e3:
                                            st.error(f"❌ Could not parse CSV file. Error: {str(e)}")
                                            st.stop()
                            else:
                                df = st.session_state.raw_csv_df.copy()

                            if df is not None and not df.empty:
                                # Apply column mapping to create standardized dataframe
                                mapped_df = df.copy()

                                # Create reverse mapping to rename columns to standard names
                                column_rename_map = {}
                                standard_names = {
                                    'pid_col': 'Project ID',
                                    'pname_col': 'Project',
                                    'org_col': 'Organization',
                                    'pm_col': 'Project Manager',
                                    'st_col': 'Plan Start',
                                    'fn_col': 'Plan Finish',
                                    'bac_col': 'BAC',
                                    'ac_col': 'AC',
                                    'pv_col': 'PV',
                                    'ev_col': 'EV',
                                    # Optional fields
                                    'manual_pv_col': 'Manual_PV',
                                    'manual_ev_col': 'Manual_EV',
                                    'curve_type_col': 'Curve Type',
                                    'alpha_col': 'Alpha',
                                    'beta_col': 'Beta',
                                    'inflation_rate_col': 'Inflation Rate'
                                }

                                # Build rename mapping from selected columns to standard names
                                for field_key, csv_column in column_mapping.items():
                                    if field_key in standard_names:
                                        column_rename_map[csv_column] = standard_names[field_key]

                                # Rename columns to standard names
                                mapped_df = mapped_df.rename(columns=column_rename_map)

                                # Add missing optional columns with default values if not mapped
                                if 'Manual_PV' not in mapped_df.columns:
                                    mapped_df['Manual_PV'] = 0.0
                                if 'Manual_EV' not in mapped_df.columns:
                                    mapped_df['Manual_EV'] = 0.0
                                if 'Curve Type' not in mapped_df.columns:
                                    mapped_df['Curve Type'] = None
                                if 'Alpha' not in mapped_df.columns:
                                    mapped_df['Alpha'] = None
                                if 'Beta' not in mapped_df.columns:
                                    mapped_df['Beta'] = None
                                if 'Inflation Rate' not in mapped_df.columns:
                                    mapped_df['Inflation Rate'] = None

                                # Auto-detect Use_Manual_PV and Use_Manual_EV based on values
                                mapped_df['Use_Manual_PV'] = mapped_df['Manual_PV'] > 0
                                mapped_df['Use_Manual_EV'] = mapped_df['Manual_EV'] > 0

                                # Clean up unwanted columns (like "Unnamed:" from blank CSV headers)
                                columns_to_keep = [col for col in mapped_df.columns if not col.startswith('Unnamed:')]
                                mapped_df = mapped_df[columns_to_keep]

                                # Store the loaded data - Route to database OR session state
                                if USE_DATABASE and portfolio_id is not None:
                                    # Database mode: Load CSV data into database
                                    st.info("🔄 Loading data to database...")
                                    adapter = data_manager.get_data_adapter()

                                    # Create column mapping dict for database service
                                    # Use same format as batch calculation mapping
                                    db_column_mapping = {
                                        'pid_col': 'Project ID',
                                        'pname_col': 'Project',
                                        'org_col': 'Organization',
                                        'pm_col': 'Project Manager',
                                        'st_col': 'Plan Start',
                                        'fn_col': 'Plan Finish',
                                        'bac_col': 'BAC',
                                        'ac_col': 'AC'
                                    }

                                    # Load to database
                                    stats = adapter.load_csv_data(
                                        df=mapped_df,
                                        portfolio_id=portfolio_id,
                                        status_date=status_date,
                                        column_mapping=db_column_mapping
                                    )

                                    # Also store to session state for immediate viewing
                                    st.session_state.data_df = mapped_df
                                    if uploaded_file:
                                        st.session_state.original_filename = uploaded_file.name
                                    st.session_state.file_type = "csv"

                                    # Store database context for batch calculation
                                    st.session_state.current_portfolio_id = portfolio_id
                                    st.session_state.current_status_date = status_date

                                    # Mark that data is already in database (don't need to reload)
                                    st.session_state.data_already_in_db = True

                                    # Update controls with the status_date so Section B can access it
                                    if 'controls' not in st.session_state.config_dict:
                                        st.session_state.config_dict['controls'] = {}
                                    st.session_state.config_dict['controls']['data_date'] = status_date.strftime('%Y-%m-%d')

                                    # Show database load stats
                                    st.success(f"✅ CSV data loaded to database!")
                                    st.info(f"📊 **Database Load Statistics:**\n"
                                           f"- Projects created: {stats.get('projects_created', 0)}\n"
                                           f"- Projects updated: {stats.get('projects_updated', 0)}\n"
                                           f"- Status reports created: {stats.get('status_reports_created', 0)}\n"
                                           f"- Errors: {stats.get('errors', 0)}")
                                    st.info(f"💡 **Next Steps:**\n"
                                           f"- Data has been loaded for status date: {status_date.strftime('%d-%b-%Y')}\n"
                                           f"- Go to **Portfolio Management** to run Batch EVM Calculations\n"
                                           f"- Then view results in **Portfolio Analysis** or **Portfolio Charts**")

                                else:
                                    # Session state mode (existing behavior)
                                    st.session_state.data_df = mapped_df
                                    if uploaded_file:
                                        st.session_state.original_filename = uploaded_file.name
                                    st.session_state.file_type = "csv"

                                    st.success(f"✅ CSV data loaded successfully: {len(mapped_df)} projects")

                                # Preview mapped data
                                with st.expander("🔍 Preview Mapped Data"):
                                    preview_df = mapped_df.head(3)
                                    mapped_preview = {}
                                    for field, col in column_mapping.items():
                                        if col in preview_df.columns:
                                            mapped_preview[field] = preview_df[col].tolist()
                                    st.json(mapped_preview)
                            else:
                                st.error("❌ CSV file contains no data")

                        except Exception as e:
                            st.error(f"❌ Error loading CSV data: {str(e)}")
                    else:
                        st.error("No CSV file available. Please upload a file first.")
        else:
            st.info("📁 Upload a CSV file above to see available columns for mapping")

    # Process uploaded file based on selection
    if uploaded_file is not None:
        # Create a unique identifier for the current file to avoid reprocessing
        current_file_info = (uploaded_file.name, uploaded_file.size)

        # Process only if it's a new file
        if st.session_state.get('processed_file_info') != current_file_info:
            try:
                if data_source == "Load JSON":
                    # Process JSON file
                    content = uploaded_file.read().decode('utf-8')
                    json_data = json.loads(content)

                    # Debug: Show the actual JSON structure
                    st.markdown("🔍 **Debug: JSON File Structure**")
                    st.markdown(f"**Top-level keys:** {list(json_data.keys()) if isinstance(json_data, dict) else 'Not a dictionary'}")

                    if isinstance(json_data, dict):
                        if 'config' in json_data:
                            st.markdown(f"**Config keys:** {list(json_data['config'].keys()) if json_data['config'] else 'Config is empty/null'}")
                            if json_data['config'] and 'controls' in json_data['config']:
                                st.markdown(f"**Controls:** {json_data['config']['controls']}")
                        else:
                            st.markdown("**No 'config' key found**")

                    with st.expander("🔍 Show Full JSON Structure (First 1000 chars)"):
                        st.code(str(json_data)[:1000])

                    # Extract data and config - handle multiple formats
                    df = None
                    config_loaded = False

                    if isinstance(json_data, dict):
                        # Check for controls at top level (your format)
                        if 'controls' in json_data:
                            # CRITICAL: Force explicit copy to session state
                            st.session_state.config_dict['controls'] = json_data['controls'].copy() if isinstance(json_data['controls'], dict) else json_data['controls']
                            controls = json_data['controls']
                            config_loaded = True
                            st.info(f"🔧 Controls loaded from top-level: {controls.get('curve_type', 'N/A')} curve, {controls.get('currency_symbol', '$')} currency")

                        # Handle other top-level config items (but avoid enable_batch)
                        for key, value in json_data.items():
                            if key in ['enable_batch']:
                                # Store batch setting info but don't set it directly (widget manages this)
                                st.session_state.config_dict['batch_setting_from_json'] = value
                                if value:
                                    st.info("📋 JSON file indicates batch mode was enabled")
                            elif key not in ['controls', 'data', 'export_date', 'batch_results']:
                                st.session_state.config_dict[key] = value

                        # Get data - could be at top level or in 'data' key
                        if 'data' in json_data:
                            df = pd.DataFrame(json_data['data'])
                        elif any(key for key in json_data.keys() if key not in ['controls', 'config', 'export_date', 'batch_results']):
                            # Assume the JSON itself contains the data (legacy format)
                            df = pd.DataFrame([json_data])  # Single record

                        # Also check for nested config format
                        if 'config' in json_data and json_data['config']:
                            loaded_config = json_data['config']
                            if 'controls' in loaded_config:
                                # CRITICAL: Force explicit copy to session state
                                st.session_state.config_dict['controls'] = loaded_config['controls'].copy() if isinstance(loaded_config['controls'], dict) else loaded_config['controls']
                                controls = loaded_config['controls']
                                config_loaded = True
                                st.info(f"🔧 Controls loaded from config section: {controls.get('curve_type', 'N/A')} curve, {controls.get('currency_symbol', '$')} currency")

                            # Load other config sections (but avoid enable_batch which is managed by widget)
                            for key, value in loaded_config.items():
                                if key not in ['controls', 'enable_batch']:
                                    st.session_state.config_dict[key] = value

                    # If no data found, treat entire JSON as data
                    if df is None:
                        df = pd.DataFrame(json_data)

                    # Clean up unwanted columns (like "Unnamed:" from blank headers)
                    if df is not None and not df.empty:
                        columns_to_keep = [col for col in df.columns if not col.startswith('Unnamed:')]
                        df = df[columns_to_keep]

                    # Success message
                    st.success(f"✅ JSON file loaded: {len(df)} projects")
                    if not config_loaded:
                        st.warning("⚠️ No controls configuration found in JSON file")

                    st.session_state.file_type = "json"

                elif data_source == "Load CSV":
                    # Store raw CSV file for mapping interface with error handling
                    try:
                        # Try different parsing options for problematic CSV files
                        df = None

                        # Try standard parsing first
                        try:
                            uploaded_file.seek(0)  # Reset file pointer to beginning
                            df = pd.read_csv(uploaded_file)
                        except pd.errors.EmptyDataError:
                            st.error("❌ CSV file appears to be empty or has no columns")
                            return
                        except Exception as e:
                            # Try with different encoding
                            try:
                                uploaded_file.seek(0)  # Reset file pointer
                                df = pd.read_csv(uploaded_file, encoding='latin-1')
                                st.info("ℹ️ File loaded with Latin-1 encoding")
                            except Exception as e2:
                                # Try with different separator
                                try:
                                    uploaded_file.seek(0)  # Reset file pointer
                                    df = pd.read_csv(uploaded_file, sep=';')
                                    st.info("ℹ️ File loaded with semicolon separator")
                                except Exception as e3:
                                    st.error(f"❌ Could not parse CSV file. Error: {str(e)}")
                                    st.info("💡 Try saving your file as UTF-8 encoded CSV with comma separators")
                                    return

                        if df is not None and not df.empty:
                            # Clean up unwanted columns (like "Unnamed:" from blank CSV headers)
                            columns_to_keep = [col for col in df.columns if not col.startswith('Unnamed:')]
                            df = df[columns_to_keep]

                            st.session_state.raw_csv_df = df.copy()
                            st.session_state.processed_file_info = current_file_info
                            st.info(f"📁 CSV file uploaded: {len(df)} rows, {len(df.columns)} columns. Configure mapping below and click 'Load Data with Mapping'.")
                        else:
                            st.error("❌ CSV file contains no data")

                    except Exception as e:
                        st.error(f"❌ Error processing CSV file: {str(e)}")
                        logging.error(f"CSV processing error: {e}")

                # Store processed data (only for JSON, not CSV)
                if data_source != "Load CSV":
                    st.session_state.data_df = df
                    st.session_state.original_filename = uploaded_file.name
                    st.session_state.processed_file_info = current_file_info

                    # CRITICAL: Mark config as loaded to prevent loss during rerun
                    if data_source == "Load JSON" and config_loaded:
                        st.session_state.config_loaded_flag = True
                        # Debug: Verify config is actually in session state before rerun
                        if 'controls' in st.session_state.config_dict:
                            st.info(f"✅ Config verified in session before rerun: {len(st.session_state.config_dict['controls'])} settings")
                        else:
                            st.error("❌ Config missing from session state before rerun!")

                # Force rerun to update UI with new config values
                if data_source == "Load JSON" and config_loaded:
                    st.rerun()

            except Exception as e:
                st.error(f"❌ Error processing file: {str(e)}")
                logging.error(f"File processing error: {e}")


    st.markdown('</div>', unsafe_allow_html=True)
    return df, selected_table, column_mapping

def render_load_to_database_section():
    """Render B. Load Projects to Database / Select Existing Data section."""
    if not USE_DATABASE:
        return  # Skip this section if not in database mode

    st.markdown('<div class="file-section">', unsafe_allow_html=True)
    st.markdown('<div class="section-header">B. Database Data Management</div>', unsafe_allow_html=True)
    st.caption("View and manage data already loaded to the database, or load existing session state data")

    # Get portfolio from portfolio context
    portfolio_id = st.session_state.get('current_portfolio_id')

    if not portfolio_id:
        st.warning("⚠️ Please select a portfolio first")
        st.info("📌 Go to the **portfolio selector** at the top of this page or in **Portfolio Management**")
        st.markdown('</div>', unsafe_allow_html=True)
        return

    # Get status date from Controls section (Data Date)
    controls = st.session_state.config_dict.get('controls', {})
    data_date_str = controls.get('data_date')

    if not data_date_str:
        st.info("💡 **This section is for managing existing database data**")
        st.markdown("""
        **To load new CSV data:**
        1. Use **Section A** above to upload and map your CSV
        2. Select portfolio and status date in the CSV mapping interface
        3. Click "Load Data with Mapping"

        **Once data is loaded**, this section will show available status dates for batch EVM calculations.
        """)
        st.markdown('</div>', unsafe_allow_html=True)
        return

    # Convert data_date string to date object
    from datetime import datetime as dt
    status_date = dt.strptime(data_date_str, '%Y-%m-%d').date()

    # Update session state for consistency
    st.session_state.current_status_date = status_date

    st.markdown(f"**Portfolio:** {portfolio_id} | **Status Date:** {status_date.strftime('%d-%b-%Y')}")

    # Check if we have NEW data in session state
    has_new_data = (st.session_state.data_df is not None and not st.session_state.data_df.empty)
    data_already_in_db = st.session_state.get('data_already_in_db', False)

    try:
        # Initialize database connection with user-specific config
        user_email = st.session_state.get('user_email')
        db = get_db(user_email=user_email)

        # Initialize database manager (uses the singleton db instance)
        db_manager = DatabaseDataManager()

        # Check what data exists in database for this portfolio/date
        existing_count = db.execute("""
            SELECT COUNT(*) as count
            FROM project_status_report
            WHERE portfolio_id = ? AND status_date = ?
        """, (portfolio_id, status_date)).fetchone()[0]

        if data_already_in_db and existing_count > 0:
            # Data was just loaded from CSV in Section A
            st.success(f"✅ **Data loaded successfully!**")
            st.info(f"📊 Database contains **{existing_count} project(s)** for {status_date.strftime('%d-%b-%Y')}")
            st.markdown("---")
            st.markdown("**Next Steps:**")
            st.markdown("1. Go to **Portfolio Management** to run **Batch EVM Calculations**")
            st.markdown("2. View results in **Portfolio Analysis** or **Portfolio Charts**")

            # Clear the flag
            if st.button("🔄 Load Different Data"):
                st.session_state.data_already_in_db = False
                st.session_state.data_df = None
                st.rerun()

        elif has_new_data and not data_already_in_db:
            # PATH 1: User has loaded new CSV/JSON data that hasn't been saved to DB yet
            st.markdown(f"**New Projects in Session State:** {len(st.session_state.data_df)}")
            st.info("📤 You have new data loaded from CSV/JSON. You can load it to the database below.")

            # Check if data already exists in database for this portfolio/status date
            existing_check = db.execute("""
                SELECT COUNT(*) as count
                FROM project_status_report
                WHERE portfolio_id = ? AND status_date = ?
            """, (portfolio_id, status_date)).fetchone()

            existing_count = existing_check[0] if existing_check else 0

            if existing_count > 0:
                st.warning(f"⚠️ Database already contains {existing_count} project(s) for this portfolio and status date")
                st.markdown("**Choose an action:**")

                overwrite = st.checkbox(
                    "✅ Overwrite existing data",
                    key="overwrite_existing",
                    help="This will delete all existing projects and status reports for this portfolio/status date and load the new data"
                )

                if not overwrite:
                    st.info("💡 Check the box above to confirm overwrite, or select a different status date")
                    st.markdown('</div>', unsafe_allow_html=True)
                    return

            # Load to Database button (requires write access)
            _has_write_db = st.session_state.get('portfolio_access_level') in ('owner', 'write')
            if not _has_write_db:
                st.warning("Write access required to load data into this portfolio.")
            elif st.button("📥 Load Projects to Database", type="primary", key="load_to_db"):
                with st.spinner("🔄 Loading projects to database..."):
                    try:
                        # If overwriting, delete existing data first
                        if existing_count > 0:
                            st.info(f"🗑️ Deleting {existing_count} existing records...")

                            # Get list of projects for this portfolio/status to delete completely
                            projects_to_delete = db.execute("""
                                SELECT DISTINCT p.project_id
                                FROM project p
                                JOIN project_status_report sr ON p.project_id = sr.project_id
                                WHERE p.portfolio_id = ? AND sr.status_date = ?
                            """, (portfolio_id, status_date)).fetchall()

                            project_ids = [row[0] for row in projects_to_delete]

                            if project_ids:
                                # Delete in correct order (foreign key constraints)
                                # 1. Delete status reports
                                db.execute("""
                                    DELETE FROM project_status_report
                                    WHERE portfolio_id = ? AND status_date = ?
                                """, (portfolio_id, status_date))

                                # 2. Delete baselines for these projects
                                placeholders = ','.join(['?'] * len(project_ids))
                                db.execute(f"""
                                    DELETE FROM project_baseline
                                    WHERE project_id IN ({placeholders})
                                """, project_ids)

                                # 3. Delete projects
                                db.execute(f"""
                                    DELETE FROM project
                                    WHERE project_id IN ({placeholders})
                                """, project_ids)

                                st.success(f"✅ Deleted {len(project_ids)} projects, their baselines, and status reports")

                        # Get data from session state
                        df = st.session_state.data_df.copy()

                        # Convert date columns to datetime objects (keep as date objects, not strings)
                        date_columns = ['Plan Start', 'Plan Finish']
                        for col in date_columns:
                            if col in df.columns:
                                # Convert to datetime, handling various formats
                                df[col] = pd.to_datetime(df[col], dayfirst=True, errors='coerce')
                                # Convert to date objects (not strings)
                                df[col] = df[col].dt.date

                        # Prepare data adapter
                        adapter = data_manager.get_data_adapter()

                        # Column mapping
                        column_mapping = {
                            'pid_col': 'Project ID',
                            'pname_col': 'Project',
                            'org_col': 'Organization',
                            'pm_col': 'Project Manager',
                            'st_col': 'Plan Start',
                            'fn_col': 'Plan Finish',
                            'bac_col': 'BAC',
                            'ac_col': 'AC'
                        }

                        # Load to database
                        stats = adapter.load_csv_data(
                            df=df,
                            portfolio_id=portfolio_id,
                            status_date=status_date,
                            column_mapping=column_mapping
                        )

                        st.success("✅ Projects loaded to database!")
                        st.info(f"""📊 **Database Load Statistics:**
- Projects created: {stats.get('projects_created', 0)}
- Projects updated: {stats.get('projects_updated', 0)}
- Status reports created: {stats.get('status_reports_created', 0)}
- Errors: {stats.get('errors', 0)}""")

                        # Update session state flag
                        st.session_state['projects_loaded_to_db'] = True
                        st.session_state['db_load_portfolio_id'] = portfolio_id
                        st.session_state['db_load_status_date'] = status_date

                        st.success("✅ Ready for batch EVM calculation!")

                    except Exception as e:
                        st.error(f"❌ Error loading to database: {e}")
                        logger.error(f"Database load error: {e}")

        else:
            # PATH 2: No new data loaded - work with existing database data
            st.info("📊 No new CSV/JSON data loaded. Select an existing status date to run Batch EVM calculations.")

            # Get available status dates for this portfolio
            available_dates = db.execute("""
                SELECT DISTINCT status_date
                FROM project_status_report
                WHERE portfolio_id = ?
                ORDER BY status_date DESC
            """, (portfolio_id,)).fetchall()

            if not available_dates:
                st.warning("⚠️ No data found in database for this portfolio. Please load CSV/JSON data first.")
                st.markdown('</div>', unsafe_allow_html=True)
                return

            # Convert to list of dates
            date_options = [row[0] for row in available_dates]

            # Format for display
            date_display = [d.strftime('%d-%b-%Y') if hasattr(d, 'strftime') else str(d) for d in date_options]

            # Create a mapping between display and actual dates
            date_map = dict(zip(date_display, date_options))

            st.markdown(f"**Available status dates:** {len(date_options)}")

            # Select status date
            selected_display = st.selectbox(
                "Select Status Date",
                options=date_display,
                key="select_existing_status_date",
                help="Choose a status date to load projects for Batch EVM calculation"
            )

            if selected_display:
                selected_date = date_map[selected_display]

                # Count projects for this date
                count_check = db.execute("""
                    SELECT COUNT(*) as count
                    FROM project_status_report
                    WHERE portfolio_id = ? AND status_date = ?
                """, (portfolio_id, selected_date)).fetchone()

                project_count = count_check[0] if count_check else 0

                st.success(f"✅ Found {project_count} project(s) for {selected_display}")

                # Update the Controls data_date to match selected date
                if st.button("🔄 Load Data for Batch EVM", type="primary", key="load_existing_data"):
                    # Convert selected_date to datetime.date if it's a string
                    if isinstance(selected_date, str):
                        from datetime import datetime as dt
                        selected_date = dt.strptime(selected_date, '%Y-%m-%d').date()

                    # Update Controls section data_date
                    st.session_state.config_dict['controls']['data_date'] = selected_date.strftime('%Y-%m-%d')
                    st.session_state.current_status_date = selected_date

                    # Load data from database into session state
                    try:
                        db_manager = DatabaseDataManager()
                        projects_df = db_manager.get_projects_for_batch_calculation(portfolio_id, selected_date)

                        if projects_df is not None and not projects_df.empty:
                            st.session_state.data_df = projects_df
                            st.session_state.data_loaded = True
                            st.session_state.file_type = 'database'

                            st.success(f"✅ Loaded {len(projects_df)} projects from database for {selected_display}")
                            st.info("📊 Data Date in Controls has been updated. You can now run Batch EVM in Section C below.")
                        else:
                            st.warning(f"⚠️ No project data found for {selected_display}")
                            st.session_state.data_df = None
                            st.session_state.data_loaded = False
                    except Exception as e:
                        st.error(f"❌ Error loading data from database: {e}")
                        logger.error(f"Error in get_projects_for_batch_calculation: {e}")
                        st.session_state.data_df = None
                        st.session_state.data_loaded = False

                    st.rerun()

    except Exception as e:
        st.error(f"❌ Error accessing database: {e}")
        logger.error(f"Database access error: {e}")

    st.markdown('</div>', unsafe_allow_html=True)

def render_save_download_section():
    """Render C. Save & Download section."""
    st.markdown('<div class="file-section">', unsafe_allow_html=True)
    st.markdown('<div class="section-header">C. Save & Download</div>', unsafe_allow_html=True)

    # Check if there's data to save
    has_data = False
    data_sources = []

    # Check session state data
    if st.session_state.data_df is not None and not st.session_state.data_df.empty:
        has_data = True

    # Check batch results
    if st.session_state.batch_results is not None and not st.session_state.batch_results.empty:
        has_data = True

    # Check config data
    config_items = len(st.session_state.config_dict) if st.session_state.config_dict else 0

    if has_data or config_items > 0:

        # Filename input
        st.markdown("**📝 Download Filename**")

        # Initialize default filename in session state if not exists
        if 'custom_filename_input' not in st.session_state:
            st.session_state.custom_filename_input = f"portfolio_package_{datetime.now().strftime('%Y%m%d_%H%M%S')}"

        st.text_input(
            "Base filename (without extension)",
            key="custom_filename_input",
            help="Enter filename without extension. Extensions (.json, .csv) will be added automatically."
        )

        # Helper function to get current filename from session state
        def get_sanitized_filename():
            """Get and sanitize the current filename from session state."""
            import re
            # Read directly from session state to get the most current value
            current_value = st.session_state.get('custom_filename_input', '')
            sanitized = re.sub(r'[<>:"/\\|?*]', '_', current_value.strip())
            return sanitized if sanitized else f"portfolio_package_{datetime.now().strftime('%Y%m%d_%H%M%S')}"

        st.markdown("")  # Add spacing

        # Export options
        col1, col2 = st.columns(2)

        with col1:
            # Prepare JSON data for single-click download
            # Define required and optional fields for export
            required_fields = {
                'project id': 'Project ID',
                'project': 'Project',
                'organization': 'Organization',
                'project manager': 'Project Manager',
                'plan start': 'Plan Start',
                'plan finish': 'Plan Finish',
                'bac': 'BAC',
                'ac': 'AC'
            }

            optional_fields = {
                'manual_pv': 'Manual_PV',
                'manual_ev': 'Manual_EV',
                'use_manual_pv': 'Use_Manual_PV',
                'use_manual_ev': 'Use_Manual_EV',
                'curve type': 'Curve Type',
                'alpha': 'Alpha',
                'beta': 'Beta',
                'inflation rate': 'Inflation Rate',
                'completion %': 'Completion %'
            }

            all_allowed_fields = {**required_fields, **optional_fields}

            # Clean the data - keep only whitelisted input fields
            clean_data = []
            if st.session_state.data_df is not None:
                for record in st.session_state.data_df.to_dict('records'):
                    record_map = {k.lower(): (k, v) for k, v in record.items()}
                    clean_record = {}

                    # Add required fields
                    for field_lower, standard_name in required_fields.items():
                        if field_lower in record_map:
                            clean_record[standard_name] = record_map[field_lower][1]
                        else:
                            clean_record[standard_name] = None

                    # Add optional fields
                    for field_lower, standard_name in optional_fields.items():
                        if field_lower in record_map:
                            clean_record[standard_name] = record_map[field_lower][1]
                        else:
                            if 'use_manual' in field_lower:
                                clean_record[standard_name] = False
                            elif 'manual_' in field_lower:
                                clean_record[standard_name] = 0.0
                            else:
                                clean_record[standard_name] = None

                    clean_data.append(clean_record)

            # Clean config
            import copy
            clean_config = {}

            if 'controls' in st.session_state.config_dict:
                clean_config['controls'] = copy.deepcopy(st.session_state.config_dict['controls'])

            if 'llm_config' in st.session_state.config_dict:
                clean_config['llm_config'] = copy.deepcopy(st.session_state.config_dict['llm_config'])
                if 'api_key' in clean_config['llm_config']:
                    clean_config['llm_config']['api_key'] = ""
                    clean_config['llm_config']['has_api_key'] = False

            if 'controls' in clean_config and 'llm_config' in clean_config['controls']:
                if 'api_key' in clean_config['controls']['llm_config']:
                    clean_config['controls']['llm_config']['api_key'] = ""
                    clean_config['controls']['llm_config']['has_api_key'] = False

            # Create optimized package
            package = {
                'config': clean_config,
                'data': clean_data
            }

            json_str = json.dumps(package, indent=2, default=str)

            # Single-click download button
            st.download_button(
                "📥 Download JSON Package",
                json_str,
                file_name=f"{get_sanitized_filename()}.json",
                mime="application/json",
                help=f"Download {len(clean_data)} projects with all {len(all_allowed_fields)} required and optional input fields"
            )

        with col2:
            if has_data:
                st.markdown("**CSV Export Options**")

                # Prepare base data
                if st.session_state.data_df is not None and not st.session_state.data_df.empty:
                    base_df = st.session_state.data_df.copy()

                    # Define input-only columns (Columns 1-18 from documentation)
                    INPUT_ONLY_COLUMNS = [
                        # Columns 1-17: Input Data
                        'Project ID',
                        'Project',
                        'Organization',
                        'Project Manager',
                        'Plan Start',
                        'Plan Finish',
                        'BAC',
                        'AC',
                        'Manual_PV',
                        'Manual_EV',
                        'Use_Manual_PV',
                        'Use_Manual_EV',
                        'Curve Type',
                        'Alpha',
                        'Beta',
                        'Inflation Rate',
                        'Completion %',
                        # Column 18: Basic Calculation
                        'data_date'
                    ]

                    # Create input-only dataframe
                    input_cols_available = [col for col in INPUT_ONLY_COLUMNS if col in base_df.columns]
                    input_only_df = base_df[input_cols_available].copy()

                    # If data_date is not in base_df but exists in batch_results, add it
                    if 'data_date' not in input_only_df.columns:
                        if hasattr(st.session_state, 'batch_results') and st.session_state.batch_results is not None:
                            batch_df = st.session_state.batch_results
                            if 'data_date' in batch_df.columns and 'Project ID' in base_df.columns:
                                # Merge just the data_date column
                                temp_df = base_df[['Project ID']].copy()
                                temp_df['Project ID'] = temp_df['Project ID'].astype(str)

                                date_df = batch_df[['project_id', 'data_date']].copy() if 'project_id' in batch_df.columns else batch_df[['Project ID', 'data_date']].copy()
                                date_df.columns = ['Project ID', 'data_date']
                                date_df['Project ID'] = date_df['Project ID'].astype(str)

                                temp_df = temp_df.merge(date_df, on='Project ID', how='left')
                                input_only_df['data_date'] = temp_df['data_date']

                    # Reorder columns to match INPUT_ONLY_COLUMNS order
                    final_cols = [col for col in INPUT_ONLY_COLUMNS if col in input_only_df.columns]
                    input_only_df = input_only_df[final_cols]

                    # Filter out unwanted columns
                    input_only_df = input_only_df[[col for col in input_only_df.columns if not col.startswith('Unnamed:')]]

                    # Convert to CSV
                    csv_input_only = input_only_df.to_csv(index=False)

                    # Download button for Input Data Only
                    st.download_button(
                        "📥 Download CSV (Input Data Only)",
                        csv_input_only,
                        file_name=f"{get_sanitized_filename()}_input.csv",
                        mime="text/csv",
                        help="Download only input data columns (for re-import or sharing)",
                        key="download_csv_input_only"
                    )

                    st.markdown("")  # Spacing

                    # Prepare full export dataframe
                    export_df = base_df.copy()

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
                            duplicate_cols = []
                            for col in batch_df.columns:
                                col_lower = col.lower().replace(' ', '_')
                                export_cols_lower = [c.lower().replace(' ', '_') for c in export_df.columns]

                                if col != 'Project ID' and col_lower in export_cols_lower:
                                    duplicate_cols.append(col)

                            # Drop duplicate columns from batch_df
                            if duplicate_cols:
                                batch_df = batch_df.drop(columns=duplicate_cols)

                            # Ensure both are strings for merge
                            export_df['Project ID'] = export_df['Project ID'].astype(str)
                            batch_df['Project ID'] = batch_df['Project ID'].astype(str)

                            # Merge batch results into export
                            export_df = export_df.merge(
                                batch_df,
                                on='Project ID',
                                how='left',
                                suffixes=('', '_batch')
                            )

                    # Filter out unwanted columns before export
                    # Remove "Unnamed:" columns that come from blank CSV headers
                    columns_to_keep = [col for col in export_df.columns if not col.startswith('Unnamed:')]
                    export_df = export_df[columns_to_keep]

                    # Convert to CSV
                    csv_full = export_df.to_csv(index=False)

                    # Download button for Full Export
                    st.download_button(
                        "📊 Download CSV (Full Export)",
                        csv_full,
                        file_name=f"{get_sanitized_filename()}_full.csv",
                        mime="text/csv",
                        help="Download complete data with all EVM calculations",
                        key="download_csv_full"
                    )
    else:
        st.info("💡 No data available for export. Load or create data first.")

    st.markdown('</div>', unsafe_allow_html=True)

# Main header
st.markdown('<h1 class="main-header">📁 Load Progress Data</h1>', unsafe_allow_html=True)
st.markdown("Centralized Data Import, Configuration & Export Hub")

# Portfolio Selection
st.markdown("---")
st.markdown("### 🎯 Select Active Portfolio")
st.caption("Choose which portfolio you want to work with")

portfolio_id = render_portfolio_selector()

if portfolio_id:
    # Compute and store access level
    try:
        from utils.portfolio_access import get_portfolio_access_level
        from utils.portfolio_context import get_current_user_email
        _user_email = get_current_user_email()
        st.session_state['portfolio_access_level'] = get_portfolio_access_level(portfolio_id, _user_email)
    except Exception:
        st.session_state['portfolio_access_level'] = None

    _has_write = st.session_state.get('portfolio_access_level') in ('owner', 'write')
    if not _has_write:
        st.info("You have **read-only** access to this portfolio. Data import and loading operations are disabled.")

    # Show portfolio info
    portfolios_df = get_user_portfolios()
    portfolio_info = portfolios_df[portfolios_df['portfolio_id'] == portfolio_id]

    if not portfolio_info.empty:
        info = portfolio_info.iloc[0]
        st.success(f"✅ Active Portfolio: **{info['portfolio_name']}** ({int(info.get('project_count', 0))} projects)")
else:
    st.warning("⚠️ Please select a portfolio to continue")
    st.info("The portfolio selection will be used for loading projects and batch calculations")

st.markdown("---")

# Render all sections
df, selected_table, column_mapping = render_data_source_section()
render_load_to_database_section()
render_save_download_section()

# Quick help
st.markdown("---")
with st.expander("ℹ️ Load Progress Data Help"):
    st.markdown("""
    **Workflow:**
    1. **A. Data Source**: Load JSON/CSV files or initialize manual entry
    2. **B. Load Projects to Database**: Load or select existing project data
    3. **C. Save & Download**: Export your data and configuration

    **Portfolio Settings:**
    - EVM calculation parameters (curves, currency, tiers) are configured per-portfolio in **Portfolio Management > Portfolio Settings**
    - LLM provider settings for executive reports are configured per-portfolio in **Portfolio Management > Portfolio Settings**

    **Run Batch EVM Calculations:**
    - EVM batch calculations are now performed in **Portfolio Management** page
    - Select your portfolio and status date, then click "Run Batch EVM Calculations"
    - Results are automatically saved to the database

    **Next Steps:**
    - After loading data, go to **Portfolio Management** to run calculations
    - Then proceed to **Portfolio Analysis** to view results and charts

    **Data Persistence:**
    - All project data is saved in the database
    - Use JSON export to create portable data packages
    """)

# Show user info in sidebar
from utils.auth import show_user_info_sidebar
show_user_info_sidebar()
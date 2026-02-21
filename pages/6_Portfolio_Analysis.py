import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import numpy as np
from datetime import datetime
import warnings
import sys
import os
warnings.filterwarnings('ignore')

# Import core utilities
from core.utils import safe_divide
from utils.auth import check_authentication, require_page_access
from config.constants import USE_DATABASE
from services.data_service import data_manager
from utils.portfolio_context import render_portfolio_context
from services.db_data_service import DatabaseDataManager

# Add pages directory to path for imports
sys.path.insert(0, os.path.dirname(__file__))

# Import functions from Project Analysis for Executive Brief
create_portfolio_executive_summary = None
safe_llm_request = None

try:
    import importlib.util
    spec = importlib.util.spec_from_file_location(
        "project_analysis",
        os.path.join(os.path.dirname(__file__), "5_Project_Analysis.py")
    )
    project_analysis = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(project_analysis)
    create_portfolio_executive_summary = project_analysis.create_portfolio_executive_summary
    safe_llm_request = project_analysis.safe_llm_request
except Exception as e:
    print(f"Warning: Could not import from Project Analysis: {e}")
    create_portfolio_executive_summary = None
    safe_llm_request = None

# Page configuration
st.set_page_config(
    page_title="CPO Executive Dashboard",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Check authentication and page access
if not check_authentication():
    st.stop()

require_page_access('portfolio_analysis', 'Portfolio Analysis')

# Custom CSS for executive styling
st.markdown("""
<style>
    /* Import professional fonts */
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');
    
    /* Global styling */
    .stApp {
        background: #ffffff;
        font-family: 'Inter', sans-serif;
    }
    
    /* Main header simple style */
    .main-header {
        text-align: center;
        color: #2c3e50;
        font-size: 2.2rem;
        font-weight: 600;
        margin-bottom: 1rem;
        border-bottom: 3px solid #3498db;
        padding-bottom: 0.5rem;
    }
    
    /* Executive metric cards */
    .metric-card {
        background: rgba(255, 255, 255, 0.95);
        backdrop-filter: blur(20px);
        border: 1px solid rgba(255,255,255,0.3);
        padding: 2rem;
        border-radius: 15px;
        box-shadow: 0 20px 40px rgba(0, 0, 0, 0.1);
        text-align: center;
        transition: all 0.3s ease;
        position: relative;
        overflow: hidden;
    }
    
    .metric-card:hover {
        transform: translateY(-5px);
        box-shadow: 0 30px 60px rgba(0, 0, 0, 0.15);
    }
    
    .metric-card::before {
        content: '';
        position: absolute;
        top: 0;
        left: 0;
        width: 100%;
        height: 4px;
        background: linear-gradient(90deg, #667eea, #764ba2);
        border-radius: 15px 15px 0 0;
    }
    
    /* Alert banners with modern styling */
    .alert-banner {
        background: linear-gradient(135deg, #ff6b6b 0%, #ee5a52 100%);
        color: white;
        padding: 1.5rem 2rem;
        border-radius: 15px;
        text-align: center;
        font-weight: 600;
        margin: 1.5rem 0;
        box-shadow: 0 15px 35px rgba(238, 90, 82, 0.3);
        position: relative;
        overflow: hidden;
        animation: alertPulse 3s ease-in-out infinite;
    }
    
    @keyframes alertPulse {
        0%, 100% { box-shadow: 0 15px 35px rgba(238, 90, 82, 0.3); }
        50% { box-shadow: 0 20px 45px rgba(238, 90, 82, 0.5); }
    }
    
    .alert-banner::before {
        content: '';
        position: absolute;
        top: 0;
        left: -100%;
        width: 100%;
        height: 100%;
        background: linear-gradient(90deg, transparent, rgba(255,255,255,0.2), transparent);
        animation: alertShine 2s infinite;
    }
    
    @keyframes alertShine {
        0% { left: -100%; }
        100% { left: 100%; }
    }
    
    /* Section cards with premium styling */
    .section-card {
        background: rgba(255, 255, 255, 0.95);
        backdrop-filter: blur(20px);
        border: 1px solid rgba(255,255,255,0.3);
        padding: 2rem;
        border-radius: 20px;
        box-shadow: 0 20px 40px rgba(0, 0, 0, 0.08);
        margin-bottom: 2rem;
        transition: all 0.3s ease;
    }
    
    .section-card:hover {
        transform: translateY(-2px);
        box-shadow: 0 25px 50px rgba(0, 0, 0, 0.12);
    }
    
    /* Modern metric styling */
    [data-testid="metric-container"] {
        background: rgba(255, 255, 255, 0.9);
        border: 1px solid rgba(255,255,255,0.3);
        padding: 0.8rem;
        border-radius: 12px;
        box-shadow: 0 10px 25px rgba(0, 0, 0, 0.08);
        transition: all 0.3s ease;
        min-height: 130px;
        max-height: 160px;
        overflow: visible;
        width: 100%;
        box-sizing: border-box;
        display: flex;
        flex-direction: column;
        justify-content: center;
    }
    
    [data-testid="metric-container"]:hover {
        transform: translateY(-3px);
        box-shadow: 0 15px 35px rgba(0, 0, 0, 0.12);
    }
    
    [data-testid="metric-container"] > div > div > div > div {
        font-size: 1.1rem !important;
        font-weight: 700 !important;
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        word-wrap: break-word !important;
        white-space: normal !important;
        line-height: 1.2 !important;
        max-width: 100% !important;
        overflow-wrap: break-word !important;
        hyphens: auto !important;
        display: block !important;
        padding: 0 !important;
        margin: 0 !important;
        overflow: visible !important;
    }

    /* Additional responsive sizing for very long numbers */
    @media (max-width: 1400px) {
        [data-testid="metric-container"] > div > div > div > div {
            font-size: 1rem !important;
        }
    }

    @media (max-width: 1200px) {
        [data-testid="metric-container"] > div > div > div > div {
            font-size: 0.95rem !important;
        }
    }

    @media (max-width: 768px) {
        [data-testid="metric-container"] > div > div > div > div {
            font-size: 0.9rem !important;
        }
    }
    
    /* Executive section headers */
    .section-header {
        font-size: 1.5rem;
        font-weight: 600;
        color: #2d3748;
        margin-bottom: 1.5rem;
        padding-bottom: 0.5rem;
        border-bottom: 3px solid;
        border-image: linear-gradient(90deg, #667eea, #764ba2) 1;
        display: flex;
        align-items: center;
        gap: 0.5rem;
    }
    
    /* Status indicators with professional styling */
    .status-critical {
        background: linear-gradient(135deg, #fed7d7 0%, #feb2b2 100%);
        border-left: 5px solid #e53e3e;
        padding: 1rem 1.5rem;
        border-radius: 10px;
        margin: 0.5rem 0;
        box-shadow: 0 4px 12px rgba(229, 62, 62, 0.15);
    }
    
    .status-warning {
        background: linear-gradient(135deg, #fef5e7 0%, #feebc8 100%);
        border-left: 5px solid #d69e2e;
        padding: 1rem 1.5rem;
        border-radius: 10px;
        margin: 0.5rem 0;
        box-shadow: 0 4px 12px rgba(214, 158, 46, 0.15);
    }
    
    .status-success {
        background: linear-gradient(135deg, #f0fff4 0%, #c6f6d5 100%);
        border-left: 5px solid #38a169;
        padding: 1rem 1.5rem;
        border-radius: 10px;
        margin: 0.5rem 0;
        box-shadow: 0 4px 12px rgba(56, 161, 105, 0.15);
    }
    
    /* Action items with executive styling */
    .action-item {
        background: linear-gradient(135deg, rgba(255,243,205,0.9) 0%, rgba(254,235,200,0.9) 100%);
        border-left: 4px solid #d69e2e;
        padding: 1.5rem;
        margin: 1rem 0;
        border-radius: 12px;
        font-weight: 500;
        box-shadow: 0 8px 20px rgba(214, 158, 46, 0.1);
        transition: all 0.3s ease;
    }
    
    .action-item:hover {
        transform: translateX(5px);
        box-shadow: 0 12px 30px rgba(214, 158, 46, 0.2);
    }
    
    /* Executive button styling */
    .stButton > button {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        border: none;
        padding: 0.75rem 2rem;
        border-radius: 10px;
        font-weight: 600;
        font-size: 1rem;
        transition: all 0.3s ease;
        box-shadow: 0 8px 20px rgba(102, 126, 234, 0.3);
    }
    
    .stButton > button:hover {
        transform: translateY(-2px);
        box-shadow: 0 12px 30px rgba(102, 126, 234, 0.4);
        background: linear-gradient(135deg, #5a6fd8 0%, #6b4190 100%);
    }
    
    /* Sidebar styling */
    .css-1d391kg {
        background: rgba(255, 255, 255, 0.95);
        backdrop-filter: blur(20px);
        border-right: 1px solid rgba(255,255,255,0.3);
    }
    
    /* Table styling */
    .stDataFrame {
        border-radius: 15px;
        overflow: hidden;
        box-shadow: 0 15px 35px rgba(0, 0, 0, 0.1);
    }
    
    /* Expander styling */
    .streamlit-expanderHeader {
        background: rgba(255, 255, 255, 0.9);
        backdrop-filter: blur(10px);
        border-radius: 10px;
        font-weight: 600;
        color: #2d3748;
    }
    
    /* Selectbox and slider styling */
    .stSelectbox > div > div > div {
        background: rgba(255, 255, 255, 0.9);
        backdrop-filter: blur(10px);
        border-radius: 8px;
        border: 1px solid rgba(255,255,255,0.3);
    }
    
    .stSlider > div > div > div > div {
        background: linear-gradient(90deg, #667eea, #764ba2);
    }
    
    /* Footer styling */
    .footer {
        text-align: center;
        padding: 2rem;
        color: #718096;
        font-size: 0.9rem;
        margin-top: 3rem;
    }
    
    /* Hide Streamlit branding */
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    header {visibility: hidden;}
</style>
""", unsafe_allow_html=True)


def format_currency(amount, currency_symbol='$', currency_postfix='', thousands=True):
    """Format currency values with proper symbol and postfix"""
    if thousands:
        formatted_amount = f"{amount/1000:.0f}K"
    else:
        formatted_amount = f"{amount:,.0f}"

    if currency_postfix:
        return f"{currency_symbol} {formatted_amount} {currency_postfix}"
    else:
        return f"{currency_symbol} {formatted_amount}"

def categorize_project_health(row):
    """Categorize project health based on CPI and SPI"""
    cpi = row.get('CPI', 0)
    spi = row.get('SPI', 0)

    if cpi >= 0.95 and spi >= 0.95:
        return 'Healthy'
    elif cpi >= 0.85 and spi >= 0.85:
        return 'At Risk'
    else:
        return 'Critical'

def map_columns_to_standard(df):
    """Map Portfolio Analysis columns to dashboard expected columns"""
    # Create a mapping from Portfolio Analysis format to Dashboard format
    column_mapping = {
        # Real batch calculation results (from Project Analysis)
        'bac': 'Budget',
        'ac': 'Actual Cost',
        'ev': 'Earned Value',
        'pv': 'Plan Value',
        'etc': 'ETC',
        'eac': 'EAC',
        'cpi': 'CPI',
        'spi': 'SPI',
        'spie': 'SPIe',
        'project_name': 'Project Name',
        'project_id': 'Project ID',
        'organization': 'Organization',
        'project_manager': 'Project Manager',
        'plan_start': 'Plan Start',
        'plan_finish': 'Plan Finish',
        'cv': 'Cost Variance',
        'sv': 'Schedule Variance',
        'vac': 'VAC',
        'percent_complete': 'Percent Complete',
        'actual_duration_months': 'Actual Duration',
        'original_duration_months': 'Original Duration',
        'forecast_duration': 'Forecast Duration',
        'present_value': 'Present Value',
        # File Management data format (uppercase) - for fallback
        'BAC': 'Budget',
        'AC': 'Actual Cost',
        'Project ID': 'Project ID',
        'Project': 'Project Name',
        'Organization': 'Organization',
        'Project Manager': 'Project Manager',
        'Plan Start': 'Plan Start',
        'Plan Finish': 'Plan Finish',
        'CPI': 'CPI',
        'SPI': 'SPI'
    }

    # Create a copy of the dataframe
    mapped_df = df.copy()

    # Rename columns if they exist in the source format
    for source_col, target_col in column_mapping.items():
        if source_col in mapped_df.columns and target_col not in mapped_df.columns:
            mapped_df[target_col] = mapped_df[source_col]

    # Ensure all expected columns exist with default values
    expected_columns = {
        'Budget': 0,
        'Actual Cost': 0,
        'Earned Value': 0,
        'Plan Value': 0,
        'CPI': 1.0,
        'SPI': 1.0,
        'SPIe': 1.0,
        'ETC': 0,
        'EAC': 0,
        'Project Name': 'Unknown',
        'Project ID': 'Unknown',
        'Organization': 'Unknown',
        'Project Manager': 'Unknown'
    }

    for col, default_value in expected_columns.items():
        if col not in mapped_df.columns:
            if col in ['Project Name', 'Project ID', 'Organization', 'Project Manager']:
                mapped_df[col] = default_value
            else:
                mapped_df[col] = default_value

    return mapped_df

def calculate_portfolio_metrics(df):
    """Calculate key portfolio-level metrics"""
    # First map columns to expected format
    df = map_columns_to_standard(df)

    metrics = {}

    # Basic counts and totals
    metrics['total_projects'] = len(df)

    # Use get() with default values for missing columns
    metrics['total_budget'] = df.get('Budget', pd.Series([0])).sum()
    metrics['total_actual_cost'] = df.get('Actual Cost', pd.Series([0])).sum()
    metrics['total_earned_value'] = df.get('Earned Value', pd.Series([0])).sum()
    metrics['total_planned_value'] = df.get('Plan Value', pd.Series([0])).sum()
    metrics['total_etc'] = df.get('ETC', pd.Series([0])).sum()
    metrics['total_eac'] = df.get('EAC', pd.Series([0])).sum()
    
    # Portfolio performance indices - use portfolio-level sums to avoid unrealistic individual values
    # CPI = SUM(EV)/SUM(AC), SPI = SUM(EV)/SUM(PV)
    metrics['portfolio_cpi'] = metrics['total_earned_value'] / metrics['total_actual_cost'] if metrics['total_actual_cost'] > 0 else 0
    metrics['portfolio_spi'] = metrics['total_earned_value'] / metrics['total_planned_value'] if metrics['total_planned_value'] > 0 else 0

    # Portfolio TCPI (To Complete Performance Index)
    # TCPI = (BAC - EV) / (BAC - AC)
    # Represents the cost performance needed for remaining work to meet budget
    work_remaining = metrics['total_budget'] - metrics['total_earned_value']
    budget_remaining = metrics['total_budget'] - metrics['total_actual_cost']
    if budget_remaining <= 0:
        if work_remaining == 0:
            metrics['portfolio_tcpi'] = 0  # Project complete
        else:
            metrics['portfolio_tcpi'] = float('inf')  # Work remaining but no budget
    else:
        metrics['portfolio_tcpi'] = work_remaining / budget_remaining if budget_remaining > 0 else 0

    # Forecast metrics
    metrics['forecast_overrun'] = metrics['total_eac'] - metrics['total_budget']
    metrics['overrun_percentage'] = (metrics['forecast_overrun'] / metrics['total_budget']) * 100 if metrics['total_budget'] > 0 else 0
    
    # Health distribution
    df['Health_Category'] = df.apply(categorize_project_health, axis=1)
    health_counts = df['Health_Category'].value_counts()
    metrics['critical_projects'] = health_counts.get('Critical', 0)
    metrics['at_risk_projects'] = health_counts.get('At Risk', 0)
    metrics['healthy_projects'] = health_counts.get('Healthy', 0)
    
    return metrics

def main():
    # Initialize session state data structures if not already done
    if "config_dict" not in st.session_state:
        st.session_state.config_dict = {}
    if "data_loaded" not in st.session_state:
        st.session_state.data_loaded = False

    # Executive Header
    st.markdown('<h1 class="main-header">📊 Portfolio Executive Dashboard</h1>', unsafe_allow_html=True)
    st.markdown("Portfolio Analytics and graphs")

    # Portfolio & Period Selection
    st.markdown("---")
    portfolio_id, status_date = render_portfolio_context(show_period_selector=True, show_progress_filter=True)
    st.markdown("---")

    if not portfolio_id:
        st.warning("⚠️ Please select a portfolio to continue")
        st.info("Go to **Portfolio Management** to create or select a portfolio")
        st.stop()

    # Load portfolio settings including tier configurations
    if USE_DATABASE and portfolio_id:
        from utils.portfolio_settings import load_portfolio_settings
        portfolio_settings = load_portfolio_settings(portfolio_id)

        # Build controls dictionary with all settings including tier configs
        controls = {
            'curve_type': portfolio_settings.get('curve_type', 'linear'),
            'alpha': portfolio_settings.get('alpha', 2.0),
            'beta': portfolio_settings.get('beta', 2.0),
            'currency_symbol': portfolio_settings.get('currency_symbol', '$'),
            'currency_postfix': portfolio_settings.get('currency_postfix', ''),
            'date_format': portfolio_settings.get('date_format', 'YYYY-MM-DD'),
            'tier_config': portfolio_settings.get('tier_config', {
                'cutoff_points': [4000, 8000, 15000],
                'tier_names': ['Tier 1', 'Tier 2', 'Tier 3', 'Tier 4'],
                'colors': ['#3498db', '#27ae60', '#f39c12', '#e74c3c']
            }),
            'duration_tier_config': portfolio_settings.get('duration_tier_config', {
                'cutoff_points': [6, 12, 24],
                'tier_names': ['Short', 'Medium', 'Long', 'Extra Long'],
                'colors': ['#3498db', '#27ae60', '#f39c12', '#e74c3c']
            })
        }
        st.session_state.config_dict['controls'] = controls

        # Also store llm_config separately if it exists
        if 'llm_config' in portfolio_settings:
            st.session_state.config_dict['llm_config'] = portfolio_settings['llm_config']

        # Also store voice_config if it exists
        if 'voice_config' in portfolio_settings:
            st.session_state.config_dict['voice_config'] = portfolio_settings['voice_config']

        # Also store infographic_config if it exists
        if 'infographic_config' in portfolio_settings:
            st.session_state.config_dict['infographic_config'] = portfolio_settings['infographic_config']

    if not status_date:
        st.info("ℹ️ No data periods available for this portfolio")
        st.info("Go to **File Management** to upload data and create the first period")
        st.stop()

    # Check if EVM results exist for this period
    if USE_DATABASE:
        db_manager = DatabaseDataManager()
        has_results = db_manager.check_evm_results_exist(portfolio_id, status_date)

        if not has_results:
            st.warning("⚠️ EVM calculations not found for this period")

            col1, col2 = st.columns(2)
            with col1:
                st.info("📊 To view analysis, EVM calculations must be run first")
            with col2:
                if st.button("🔄 Calculate Now", type="primary"):
                    st.info("Please run batch calculation from File Management page")

            st.stop()

    # Get currency settings from Portfolio Analysis if available
    # Try multiple sources in order of preference:
    # 1. Dashboard-specific settings (from Generate Executive Dashboard button)
    # 2. Widget session state (from current Portfolio Analysis inputs)
    # 3. Saved controls (from config_dict)
    # 4. Default values

    saved_controls = getattr(st.session_state, 'config_dict', {}).get('controls', {})

    currency_symbol = (
        getattr(st.session_state, 'dashboard_currency_symbol', None) or
        getattr(st.session_state, 'currency_symbol', None) or
        saved_controls.get('currency_symbol', '$')
    )

    currency_postfix = (
        getattr(st.session_state, 'dashboard_currency_postfix', None) or
        getattr(st.session_state, 'currency_postfix', None) or
        saved_controls.get('currency_postfix', '')
    )



    # Check for data from Portfolio Analysis first
    if hasattr(st.session_state, 'dashboard_data') and st.session_state.dashboard_data is not None:
        df = st.session_state.dashboard_data.copy()
        st.sidebar.success("✅ Using data from Portfolio Analysis")
        st.sidebar.info(f"📊 {len(df)} projects loaded")
        if currency_symbol != '$' or currency_postfix != '':
            st.sidebar.info(f"💱 Currency: {currency_symbol} {currency_postfix}")
    else:
        # Retrieve stored EVM results
        if USE_DATABASE and portfolio_id and status_date:
            # Database mode: Get EVM results from database
            try:
                df = db_manager.get_evm_results_for_period(portfolio_id, status_date)

                if df is not None and not df.empty:
                    st.sidebar.success(f"✅ Loaded {len(df)} projects for {status_date.strftime('%d-%b-%Y')}")

            except Exception as e:
                import logging
                logging.error(f"Error retrieving EVM results: {e}")
                st.error(f"Error loading data: {e}")
                df = pd.DataFrame()
        else:
            # Session state mode: Use adapter
            adapter = data_manager.get_data_adapter()
            df = adapter.get_batch_results()

            if df is not None and not df.empty:
                st.sidebar.success("✅ Using batch results from session state")
                st.sidebar.info(f"📊 {len(df)} projects loaded")

        # Apply progress filter if enabled
        if df is not None and not df.empty:
            from utils.portfolio_context import apply_progress_filter
            # Try common AC column names
            ac_col = 'ac' if 'ac' in df.columns else ('AC' if 'AC' in df.columns else 'Actual Cost')
            if ac_col in df.columns:
                df, filter_message = apply_progress_filter(df, ac_col)
                if filter_message:
                    st.caption(filter_message)

    # If no data from any source, show file upload option
    if df is None or (isinstance(df, pd.DataFrame) and df.empty):
        # File upload option as fallback
        st.sidebar.info("💡 **Recommended Workflow:**")
        st.sidebar.markdown("1. Go to **Portfolio Analysis**")
        st.sidebar.markdown("2. Upload data & run calculations")
        st.sidebar.markdown("3. Click **Generate Executive Dashboard**")
        st.sidebar.markdown("---")

        uploaded_file = st.sidebar.file_uploader(
            "Or Upload Portfolio Data Directly",
            type=['csv'],
            help="Upload your batch_evm_results.csv file or use Portfolio Analysis to generate data"
        )

        if uploaded_file is not None:
            df = pd.read_csv(uploaded_file)
            # Clean and process the uploaded data
            df.columns = df.columns.str.strip()
            numeric_columns = ['Budget', 'Actual Cost', 'Earned Value', 'Plan Value',
                              'CPI', 'SPI', 'ETC', 'EAC', '% Budget Used', '% Time Used']
            for col in numeric_columns:
                if col in df.columns:
                    df[col] = pd.to_numeric(df[col], errors='coerce').fillna(0)
        else:
            # No data available - inform user of proper workflow
            st.sidebar.warning("⚠️ No portfolio data available")
            st.sidebar.markdown("**Please use one of these options:**")
            st.sidebar.markdown("• **File Management** → Run Batch EVM → **Portfolio Analysis**")
            st.sidebar.markdown("• **Upload CSV file** using the uploader above")
            df = None
    
    if df is None:
        # Show helpful guidance when no data is available
        st.markdown("## 🚀 Get Started")
        st.info("To view your portfolio analysis dashboard, you'll need to process your project data first.")

        col1, col2 = st.columns(2)
        with col1:
            st.markdown("""
            ### 📁 Recommended Workflow
            1. **File Management** → Upload your project data
            2. **Run Batch EVM** → Process all projects
            3. **Portfolio Analysis** → Return here for insights
            """)
            if st.button("📁 Go to File Management", type="primary"):
                st.switch_page("pages/1_File_Management.py")

        with col2:
            st.markdown("""
            ### 📊 Direct Upload
            Use the **CSV file uploader** in the sidebar to directly upload your batch EVM results.
            """)

        st.stop()

    # Map columns to expected format and calculate metrics
    df = map_columns_to_standard(df)
    metrics = calculate_portfolio_metrics(df)

    # Create budget tier categories early for use in charts
    tier_config = st.session_state.config_dict.get('controls', {}).get('tier_config', {})
    default_tier_config = {
        'cutoff_points': [4000, 8000, 15000],
        'tier_names': ['Tier 1', 'Tier 2', 'Tier 3', 'Tier 4'],
        'colors': ['#3498db', '#27ae60', '#f39c12', '#e74c3c']
    }
    cutoffs = tier_config.get('cutoff_points', default_tier_config['cutoff_points'])
    tier_names = tier_config.get('tier_names', default_tier_config['tier_names'])

    def categorize_budget_by_tier(budget):
        """Assign tier based on configurable budget ranges"""
        if pd.isna(budget):
            return "Unknown"
        elif budget >= cutoffs[2]:  # Tier 4 (highest)
            return tier_names[3]
        elif budget >= cutoffs[1]:  # Tier 3
            return tier_names[2]
        elif budget >= cutoffs[0]:  # Tier 2
            return tier_names[1]
        else:  # Tier 1 (lowest)
            return tier_names[0]

    df['Budget_Category'] = df['Budget'].apply(categorize_budget_by_tier)

    # Use portfolio-level calculations for CPI/SPI, keep weighted average for SPIe
    portfolio_cpi_weighted = metrics['portfolio_cpi']  # Already calculated as SUM(EV)/SUM(AC)
    portfolio_spi_weighted = metrics['portfolio_spi']  # Already calculated as SUM(EV)/SUM(PV)
    portfolio_spie_weighted = (df['SPIe'] * df['Budget']).sum() / df['Budget'].sum() if df['Budget'].sum() > 0 else 0  # Keep weighted for SPIe

    # Critical Alert Banner
    if portfolio_cpi_weighted < 0.85 or portfolio_spi_weighted < 0.85:
        st.markdown(f"""
        <div class="alert-banner">
            🚨 CRITICAL PORTFOLIO ALERT: Immediate intervention required •
            Portfolio CPI: {portfolio_cpi_weighted:.2f} •
            Portfolio SPI: {portfolio_spi_weighted:.2f}
        </div>
        """, unsafe_allow_html=True)

    # Key Performance Indicators
    st.markdown('<div class="section-header">📈 Executive Portfolio Overview</div>', unsafe_allow_html=True)

    col1, col2, col3, col4, col5 = st.columns(5)

    with col1:
        st.metric(
            label="Total Projects",
            value=f"{metrics['total_projects']:,}",
            delta=None
        )

    with col2:
        st.metric(
            label="Portfolio CPI",
            value=f"{portfolio_cpi_weighted:.3f}",
            delta=f"{(portfolio_cpi_weighted - 1) * 100:.1f}%",
            delta_color="inverse" if portfolio_cpi_weighted < 1 else "normal"
        )

    with col3:
        portfolio_tcpi = metrics.get('portfolio_tcpi', 0)
        # Handle infinite TCPI display
        if portfolio_tcpi == float('inf'):
            tcpi_display = "∞"
            tcpi_delta = None
        else:
            tcpi_display = f"{portfolio_tcpi:.3f}"
            tcpi_delta = f"{(portfolio_tcpi - 1) * 100:.1f}%"

        st.metric(
            label="Portfolio TCPI",
            value=tcpi_display,
            delta=tcpi_delta,
            delta_color="inverse" if portfolio_tcpi > 1 else "normal",
            help="To-Complete Performance Index: Cost performance needed for remaining work to meet budget"
        )

    with col4:
        st.metric(
            label="Portfolio SPI",
            value=f"{portfolio_spi_weighted:.3f}",
            delta=f"{(portfolio_spi_weighted - 1) * 100:.1f}%",
            delta_color="inverse" if portfolio_spi_weighted < 1 else "normal"
        )

    with col5:
        st.metric(
            label="Portfolio SPIe",
            value=f"{portfolio_spie_weighted:.3f}",
            delta=f"{(portfolio_spie_weighted - 1) * 100:.1f}%",
            delta_color="inverse" if portfolio_spie_weighted < 1 else "normal"
        )
    
    # Performance Metrics Section
    st.markdown('<div class="section-header">⚡ Strategic Performance Indicators</div>', unsafe_allow_html=True)

    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        # CPI Gauge - Dial with Needle
        fig_cpi = go.Figure(go.Indicator(
            mode = "gauge+number",
            value = portfolio_cpi_weighted,
            domain = {'x': [0, 1], 'y': [0, 1]},
            title = {'text': "Cost Performance Index (CPI)", 'font': {'size': 14}},
            number = {'font': {'size': 16}},
            gauge = {
                'axis': {'range': [0, 2.0], 'tickwidth': 1, 'tickcolor': "darkblue"},
                'bar': {'color': "rgba(0,0,0,0)"},  # Make bar transparent to show only needle
                'bgcolor': "white",
                'borderwidth': 2,
                'bordercolor': "gray",
                'steps': [
                    {'range': [0, 0.8], 'color': "#ff6b6b"},      # Red - Poor
                    {'range': [0.8, 1.0], 'color': "#ffd93d"},    # Yellow - Caution
                    {'range': [1.0, 1.5], 'color': "#6bcf7f"},    # Green - Good
                    {'range': [1.5, 2.0], 'color': "#4ecdc4"}     # Teal - Excellent
                ],
                'threshold': {
                    'line': {'color': "black", 'width': 3},
                    'thickness': 0.8,
                    'value': portfolio_cpi_weighted
                }
            }
        ))
        fig_cpi.update_layout(height=250, margin=dict(l=20, r=20, t=40, b=20))
        st.plotly_chart(fig_cpi, width='stretch')
    
    with col2:
        # SPI Gauge - Dial with Needle
        fig_spi = go.Figure(go.Indicator(
            mode = "gauge+number",
            value = portfolio_spi_weighted,
            domain = {'x': [0, 1], 'y': [0, 1]},
            title = {'text': "Schedule Performance Index (SPI)", 'font': {'size': 14}},
            number = {'font': {'size': 16}},
            gauge = {
                'axis': {'range': [0, 2.0], 'tickwidth': 1, 'tickcolor': "darkblue"},
                'bar': {'color': "rgba(0,0,0,0)"},  # Make bar transparent to show only needle
                'bgcolor': "white",
                'borderwidth': 2,
                'bordercolor': "gray",
                'steps': [
                    {'range': [0, 0.8], 'color': "#ff6b6b"},      # Red - Behind Schedule
                    {'range': [0.8, 1.0], 'color': "#ffd93d"},    # Yellow - Slightly Behind
                    {'range': [1.0, 1.5], 'color': "#6bcf7f"},    # Green - On/Ahead Schedule
                    {'range': [1.5, 2.0], 'color': "#4ecdc4"}     # Teal - Excellent
                ],
                'threshold': {
                    'line': {'color': "black", 'width': 3},
                    'thickness': 0.8,
                    'value': portfolio_spi_weighted
                }
            }
        ))
        fig_spi.update_layout(height=250, margin=dict(l=20, r=20, t=40, b=20))
        st.plotly_chart(fig_spi, width='stretch')

    with col3:
        # % Budget Used (AC/BAC) Gauge - Dial with Needle
        portfolio_budget_used = (metrics['total_actual_cost'] / metrics['total_budget']) * 100 if metrics['total_budget'] > 0 else 0
        fig_budget_used = go.Figure(go.Indicator(
            mode = "gauge+number",
            value = portfolio_budget_used,
            domain = {'x': [0, 1], 'y': [0, 1]},
            title = {'text': "% Budget Used (AC/BAC)", 'font': {'size': 14}},
            number = {'suffix': "%", 'font': {'size': 16}},
            gauge = {
                'axis': {'range': [0, 150], 'tickwidth': 1, 'tickcolor': "darkblue"},
                'bar': {'color': "rgba(0,0,0,0)"},  # Make bar transparent to show only needle
                'bgcolor': "white",
                'borderwidth': 2,
                'bordercolor': "gray",
                'steps': [
                    {'range': [0, 70], 'color': "#6bcf7f"},       # Green - Under Budget
                    {'range': [70, 90], 'color': "#ffd93d"},      # Yellow - Approaching Budget
                    {'range': [90, 110], 'color': "#ff9500"},     # Orange - At Budget
                    {'range': [110, 150], 'color': "#ff6b6b"}     # Red - Over Budget
                ],
                'threshold': {
                    'line': {'color': "black", 'width': 3},
                    'thickness': 0.8,
                    'value': portfolio_budget_used
                }
            }
        ))
        fig_budget_used.update_layout(height=250, margin=dict(l=20, r=20, t=40, b=20))
        st.plotly_chart(fig_budget_used, width='stretch')

    with col4:
        # % Earned Value (EV/BAC) Gauge - Dial with Needle
        portfolio_earned_value_pct = (metrics['total_earned_value'] / metrics['total_budget']) * 100 if metrics['total_budget'] > 0 else 0
        fig_earned_value = go.Figure(go.Indicator(
            mode = "gauge+number",
            value = portfolio_earned_value_pct,
            domain = {'x': [0, 1], 'y': [0, 1]},
            title = {'text': "% Earned Value (EV/BAC)", 'font': {'size': 14}},
            number = {'suffix': "%", 'font': {'size': 16}},
            gauge = {
                'axis': {'range': [0, 120], 'tickwidth': 1, 'tickcolor': "darkblue"},
                'bar': {'color': "rgba(0,0,0,0)"},  # Make bar transparent to show only needle
                'bgcolor': "white",
                'borderwidth': 2,
                'bordercolor': "gray",
                'steps': [
                    {'range': [0, 60], 'color': "#ff6b6b"},       # Red - Low Performance
                    {'range': [60, 80], 'color': "#ffd93d"},      # Yellow - Below Target
                    {'range': [80, 100], 'color': "#6bcf7f"},     # Green - Good Performance
                    {'range': [100, 120], 'color': "#4ecdc4"}     # Teal - Excellent Performance
                ],
                'threshold': {
                    'line': {'color': "black", 'width': 3},
                    'thickness': 0.8,
                    'value': portfolio_earned_value_pct
                }
            }
        ))
        fig_earned_value.update_layout(height=250, margin=dict(l=20, r=20, t=40, b=20))
        st.plotly_chart(fig_earned_value, width='stretch')

    # Project Health Distribution
    col1, col2 = st.columns([1, 1])
    
    with col1:
        st.markdown('<div class="section-header">🏥 Portfolio Health Analytics</div>', unsafe_allow_html=True)
        
        df['Health_Category'] = df.apply(categorize_project_health, axis=1)
        health_counts = df['Health_Category'].value_counts()
        
        colors = {'Critical': '#e74c3c', 'At Risk': '#f39c12', 'Healthy': '#27ae60'}
        
        fig_health = px.pie(
            values=health_counts.values,
            names=health_counts.index,
            color=health_counts.index,
            color_discrete_map=colors,
            title="Project Health Distribution"
        )
        fig_health.update_traces(textposition='inside', textinfo='percent+label')
        fig_health.update_layout(height=400)
        st.plotly_chart(fig_health, width='stretch')
    
    with col2:
        st.markdown('<div class="section-header">💰 Financial Performance Intelligence</div>', unsafe_allow_html=True)
        
        # Financial comparison chart
        financial_data = {
            'Metric': ['Budget', 'Actual Cost', 'Earned Value', 'Planned Value', 'EAC'],
            'Value': [
                metrics['total_budget'],
                metrics['total_actual_cost'],
                metrics['total_earned_value'],
                metrics['total_planned_value'],
                metrics['total_eac']
            ]
        }
        
        fig_financial = px.bar(
            financial_data,
            x='Metric',
            y='Value',
            title="Portfolio Financial Overview",
            color='Metric',
            color_discrete_sequence=['#3498db', '#e74c3c', '#f39c12', '#9b59b6', '#e74c3c']
        )
        fig_financial.update_layout(
            height=400,
            showlegend=False,
            yaxis=dict(
                tickformat=',.0f',
                title=f'Amount ({currency_symbol}{" " + currency_postfix if currency_postfix else ""})'
            )
        )
        st.plotly_chart(fig_financial, width='stretch')

    # Portfolio by Tier Analysis
    st.markdown('<div class="section-header">🎯 Portfolio Distribution</div>', unsafe_allow_html=True)

    # Toggle between Budget and Duration distribution
    distribution_view = st.radio(
        "Distribution By:",
        options=["Budget Tiers", "Duration Tiers"],
        horizontal=True,
        key="portfolio_distribution_view"
    )

    col1, col2 = st.columns([1, 1])

    if distribution_view == "Budget Tiers":
        with col1:
            # Portfolio Budget by Tier
            if 'Budget_Category' in df.columns and 'Budget' in df.columns:
                tier_budget = df.groupby('Budget_Category')['Budget'].sum().sort_values(ascending=False)

                # Get tier colors from config
                tier_config = st.session_state.config_dict.get('controls', {}).get('tier_config', {})
                tier_colors = tier_config.get('colors', ['#3498db', '#27ae60', '#f39c12', '#e74c3c'])
                tier_names = tier_config.get('tier_names', ['Tier 1', 'Tier 2', 'Tier 3', 'Tier 4'])

                # Create color map based on tier names
                color_map = {tier_names[i]: tier_colors[i] for i in range(len(tier_names))}

                fig_budget_tier = px.pie(
                    values=tier_budget.values,
                    names=tier_budget.index,
                    title="Portfolio Value by Budget Category",
                    color=tier_budget.index,
                    color_discrete_map=color_map
                )
                fig_budget_tier.update_traces(textposition='inside', textinfo='percent+label')
                fig_budget_tier.update_layout(height=400)
                st.plotly_chart(fig_budget_tier, width='stretch')
            else:
                st.info("Budget tier data not available")

        with col2:
            # Portfolio Projects by Tier
            if 'Budget_Category' in df.columns:
                tier_count = df['Budget_Category'].value_counts().sort_index()

                # Get tier colors from config
                tier_config = st.session_state.config_dict.get('controls', {}).get('tier_config', {})
                tier_colors = tier_config.get('colors', ['#3498db', '#27ae60', '#f39c12', '#e74c3c'])
                tier_names = tier_config.get('tier_names', ['Tier 1', 'Tier 2', 'Tier 3', 'Tier 4'])

                # Create color map based on tier names
                color_map = {tier_names[i]: tier_colors[i] for i in range(len(tier_names))}

                fig_project_tier = px.pie(
                    values=tier_count.values,
                    names=tier_count.index,
                    title="Portfolio Projects by Budget Category",
                    color=tier_count.index,
                    color_discrete_map=color_map
                )
                fig_project_tier.update_traces(textposition='inside', textinfo='percent+label')
                fig_project_tier.update_layout(height=400)
                st.plotly_chart(fig_project_tier, width='stretch')
            else:
                st.info("Project tier data not available")

    else:  # Duration Tiers
        # First, create Duration_Category if not present
        if 'Duration_Category' not in df.columns and 'original_duration_months' in df.columns:
            # Get duration tier config
            duration_tier_config = st.session_state.config_dict.get('controls', {}).get('duration_tier_config', {})
            default_duration_tier_config = {
                'cutoff_points': [6, 12, 24],
                'tier_names': ['Short', 'Medium', 'Long', 'Extra Long']
            }
            cutoffs = duration_tier_config.get('cutoff_points', default_duration_tier_config['cutoff_points'])
            tier_names = duration_tier_config.get('tier_names', default_duration_tier_config['tier_names'])

            def get_duration_tier(duration):
                """Determine tier based on duration and cutoff points."""
                if pd.isna(duration):
                    return "Unknown"
                duration = int(duration)
                # Use >= logic like budget tiers (check from high to low)
                if duration >= cutoffs[2]:
                    return tier_names[3]
                elif duration >= cutoffs[1]:
                    return tier_names[2]
                elif duration >= cutoffs[0]:
                    return tier_names[1]
                else:
                    return tier_names[0]

            df['Duration_Category'] = df['original_duration_months'].apply(get_duration_tier)

        with col1:
            # Portfolio Budget by Duration Tier
            if 'Duration_Category' in df.columns and 'Budget' in df.columns:
                duration_budget = df.groupby('Duration_Category')['Budget'].sum().sort_values(ascending=False)

                # Get tier colors from config
                duration_tier_config = st.session_state.config_dict.get('controls', {}).get('duration_tier_config', {})
                tier_colors = duration_tier_config.get('colors', ['#3498db', '#27ae60', '#f39c12', '#e74c3c'])
                tier_names = duration_tier_config.get('tier_names', ['Short', 'Medium', 'Long', 'Extra Long'])

                # Create color map based on tier names
                color_map = {tier_names[i]: tier_colors[i] for i in range(len(tier_names))}

                fig_duration_budget = px.pie(
                    values=duration_budget.values,
                    names=duration_budget.index,
                    title="Portfolio Value by Duration Category",
                    color=duration_budget.index,
                    color_discrete_map=color_map
                )
                fig_duration_budget.update_traces(textposition='inside', textinfo='percent+label')
                fig_duration_budget.update_layout(height=400)
                st.plotly_chart(fig_duration_budget, width='stretch')
            else:
                st.info("Duration tier data not available. Run batch EVM calculation first.")

        with col2:
            # Portfolio Projects by Duration Tier
            if 'Duration_Category' in df.columns:
                duration_count = df['Duration_Category'].value_counts().sort_index()

                # Get tier colors from config
                duration_tier_config = st.session_state.config_dict.get('controls', {}).get('duration_tier_config', {})
                tier_colors = duration_tier_config.get('colors', ['#3498db', '#27ae60', '#f39c12', '#e74c3c'])
                tier_names = duration_tier_config.get('tier_names', ['Short', 'Medium', 'Long', 'Extra Long'])

                # Create color map based on tier names
                color_map = {tier_names[i]: tier_colors[i] for i in range(len(tier_names))}

                fig_duration_projects = px.pie(
                    values=duration_count.values,
                    names=duration_count.index,
                    title="Portfolio Projects by Duration Category",
                    color=duration_count.index,
                    color_discrete_map=color_map
                )
                fig_duration_projects.update_traces(textposition='inside', textinfo='percent+label')
                fig_duration_projects.update_layout(height=400)
                st.plotly_chart(fig_duration_projects, width='stretch')
            else:
                st.info("Duration tier data not available. Run batch EVM calculation first.")

    # ═══════════════════════════════════════════════════════════════════
    # FILTERS EXPANDER (Applied to all sections below)
    # ═══════════════════════════════════════════════════════════════════
    with st.expander("🔍 Filters", expanded=False):
        # ═══════════════════════════════════════════════════════════════════
        # ORGANIZATIONAL FILTERS
        # ═══════════════════════════════════════════════════════════════════
        st.markdown("**🏢 Organizational**")

        # Organization filter with toggle
        org_columns = [col for col in df.columns if 'org' in col.lower() or 'department' in col.lower() or 'division' in col.lower()]
        if org_columns:
            org_options = df[org_columns[0]].dropna().unique().tolist()
        else:
            # Create dummy organizations for demonstration
            np.random.seed(42)
            orgs = ['Engineering', 'Infrastructure', 'IT', 'Construction', 'Energy', 'Healthcare']
            df['Organization'] = np.random.choice(orgs, len(df))
            org_options = orgs

        org_toggle = st.toggle("Filter by Organization", value=False, key="analysis_org_toggle")
        if org_toggle:
            organization_filter = st.multiselect(
                "Organization",
                options=org_options,
                default=org_options,
                placeholder="Select organization(s)" if org_options else "No organizations available",
                key="analysis_org_filter"
            ) if org_options else []
        else:
            organization_filter = org_options

        # Health Status filter with toggle
        health_toggle = st.toggle("Filter by Health Status", value=False, key="analysis_health_toggle")
        if health_toggle:
            health_filter = st.multiselect(
                "Health Status",
                options=['Critical', 'At Risk', 'Healthy'],
                default=['Critical', 'At Risk', 'Healthy'],
                placeholder="Select health status(es)",
                key="analysis_health_filter"
            )
        else:
            health_filter = ['Critical', 'At Risk', 'Healthy']

        st.markdown("")  # Spacing

        # ═══════════════════════════════════════════════════════════════════
        # DATE RANGE FILTERS
        # ═══════════════════════════════════════════════════════════════════
        st.markdown("**📅 Date Ranges**")

        plan_start_col = 'plan_start'
        if plan_start_col in df.columns:
            temp_dates = pd.to_datetime(df[plan_start_col], errors='coerce').dropna()
            if len(temp_dates) > 0:
                min_start = temp_dates.min()
                max_start = temp_dates.max()

                col1, col2 = st.columns([1, 1])
                with col1:
                    plan_start_later_toggle = st.toggle("Plan Start Later Than", value=False, key="analysis_plan_start_later_toggle") if pd.notna(min_start) else False
                    if plan_start_later_toggle and pd.notna(min_start):
                        plan_start_later_value = st.date_input(
                            "Plan Start Later Than",
                            value=min_start.date(),
                            min_value=min_start.date(),
                            max_value=max_start.date() if pd.notna(max_start) else min_start.date(),
                            key="analysis_plan_start_later_value"
                        )
                    else:
                        plan_start_later_value = None
                with col2:
                    plan_start_earlier_toggle = st.toggle("Plan Start Earlier Than", value=False, key="analysis_plan_start_earlier_toggle") if pd.notna(max_start) else False
                    if plan_start_earlier_toggle and pd.notna(max_start):
                        plan_start_earlier_value = st.date_input(
                            "Plan Start Earlier Than",
                            value=max_start.date(),
                            min_value=min_start.date() if pd.notna(min_start) else max_start.date(),
                            max_value=max_start.date(),
                            key="analysis_plan_start_earlier_value"
                        )
                    else:
                        plan_start_earlier_value = None

                selected_date_column = plan_start_col
            else:
                plan_start_later_value = None
                plan_start_earlier_value = None
                selected_date_column = None
        else:
            plan_start_later_value = None
            plan_start_earlier_value = None
            selected_date_column = None

        st.markdown("")  # Spacing

        # ═══════════════════════════════════════════════════════════════════
        # PROJECT CHARACTERISTICS
        # ═══════════════════════════════════════════════════════════════════
        st.markdown("**📊 Project Characteristics**")

        # Calculate budget and duration ranges
        if 'Budget' in df.columns:
            portfolio_min_budget = float(df['Budget'].min())
            portfolio_max_budget = float(df['Budget'].max())
        else:
            portfolio_min_budget = 0.0
            portfolio_max_budget = 1000000.0

        if 'original_duration_months' in df.columns:
            portfolio_min_duration = float(df['original_duration_months'].min())
            portfolio_max_duration = float(df['original_duration_months'].max())
        else:
            portfolio_min_duration = 0.0
            portfolio_max_duration = 60.0

        col1, col2 = st.columns(2)
        with col1:
            enable_lower_budget = st.toggle("Set Min Budget", value=False, key="analysis_min_budget_toggle")
            if enable_lower_budget:
                min_budget = st.number_input(
                    f"Min Budget ({currency_symbol})",
                    value=portfolio_min_budget,
                    step=max(1000.0, (portfolio_max_budget - portfolio_min_budget) / 10) if portfolio_max_budget > portfolio_min_budget else 1000.0,
                    min_value=0.0,
                    key="analysis_min_budget_value"
                )
            else:
                min_budget = 0

        with col2:
            enable_upper_budget = st.toggle("Set Max Budget", value=False, key="analysis_max_budget_toggle")
            if enable_upper_budget:
                max_budget = st.number_input(
                    f"Max Budget ({currency_symbol})",
                    value=portfolio_max_budget,
                    step=max(1000.0, (portfolio_max_budget - portfolio_min_budget) / 10) if portfolio_max_budget > portfolio_min_budget else 1000.0,
                    min_value=0.0,
                    key="analysis_max_budget_value"
                )
            else:
                max_budget = float('inf')

        col1, col2 = st.columns(2)
        with col1:
            enable_min_duration = st.toggle("Set Min Duration", value=False, key="analysis_min_duration_toggle", help="Filter by minimum Original Duration (months)")
            if enable_min_duration and 'original_duration_months' in df.columns:
                min_duration = st.number_input(
                    "Min Duration (months)",
                    value=portfolio_min_duration,
                    step=max(1.0, (portfolio_max_duration - portfolio_min_duration) / 10) if portfolio_max_duration > portfolio_min_duration else 1.0,
                    min_value=0.0,
                    key="analysis_min_duration_value"
                )
            else:
                min_duration = 0.0

        with col2:
            enable_max_duration = st.toggle("Set Max Duration", value=False, key="analysis_max_duration_toggle", help="Filter by maximum Original Duration (months)")
            if enable_max_duration and 'original_duration_months' in df.columns:
                max_duration = st.number_input(
                    "Max Duration (months)",
                    value=portfolio_max_duration,
                    step=max(1.0, (portfolio_max_duration - portfolio_min_duration) / 10) if portfolio_max_duration > portfolio_min_duration else 1.0,
                    min_value=0.0,
                    key="analysis_max_duration_value"
                )
            else:
                max_duration = float('inf')

        st.markdown("")  # Spacing

        # ═══════════════════════════════════════════════════════════════════
        # PERFORMANCE INDICES
        # ═══════════════════════════════════════════════════════════════════
        st.markdown("**📈 Performance Indices**")
        col1, col2 = st.columns([1, 1])
        with col1:
            enable_cpi_filter = st.toggle("Filter by CPI Range", value=False, key="analysis_cpi_filter_toggle", help="Cost Performance Index: >1.0 = under budget, <1.0 = over budget")
            if enable_cpi_filter and 'CPI' in df.columns:
                cpi_range = st.slider(
                    "CPI Range",
                    min_value=0.0,
                    max_value=3.0,
                    value=(0.0, 3.0),
                    step=0.01,
                    key="analysis_cpi_range_value"
                )
            else:
                cpi_range = (0.0, 3.0)
        with col2:
            enable_spi_filter = st.toggle("Filter by SPI Range", value=False, key="analysis_spi_filter_toggle", help="Schedule Performance Index: >1.0 = ahead of schedule, <1.0 = behind schedule")
            if enable_spi_filter and 'SPI' in df.columns:
                spi_range = st.slider(
                    "SPI Range",
                    min_value=0.0,
                    max_value=3.0,
                    value=(0.0, 3.0),
                    step=0.01,
                    key="analysis_spi_range_value"
                )
            else:
                spi_range = (0.0, 3.0)

        # SPIe filter (unique to Portfolio Analysis)
        enable_spie_filter = st.toggle("Filter by SPIe Range", value=False, key="analysis_spie_filter_toggle", help="Schedule Performance Index Estimate")
        if enable_spie_filter and 'SPIe' in df.columns:
            spie_range = st.slider(
                "SPIe Range",
                min_value=0.0,
                max_value=3.0,
                value=(0.0, 3.0),
                step=0.01,
                key="analysis_spie_range_value"
            )
        else:
            spie_range = (0.0, 3.0)

    # Apply filters to create filtered_df
    filtered_df = df.copy()

    # Organization filter
    if org_toggle and org_options and organization_filter:
        if org_columns:
            filtered_df = filtered_df[filtered_df[org_columns[0]].isin(organization_filter)]
        elif 'Organization' in filtered_df.columns:
            filtered_df = filtered_df[filtered_df['Organization'].isin(organization_filter)]

    # Health status filter
    if health_toggle and health_filter:
        filtered_df = filtered_df[filtered_df['Health_Category'].isin(health_filter)]

    # Plan Start date filters
    if plan_start_later_value is not None and selected_date_column:
        plan_start_later_dt = pd.to_datetime(plan_start_later_value)
        filtered_df = filtered_df[pd.to_datetime(filtered_df[selected_date_column], errors='coerce') >= plan_start_later_dt]

    if plan_start_earlier_value is not None and selected_date_column:
        plan_start_earlier_dt = pd.to_datetime(plan_start_earlier_value)
        filtered_df = filtered_df[pd.to_datetime(filtered_df[selected_date_column], errors='coerce') <= plan_start_earlier_dt]

    # Budget filters
    if 'Budget' in filtered_df.columns:
        if enable_lower_budget:
            filtered_df = filtered_df[filtered_df['Budget'] >= min_budget]
        if enable_upper_budget:
            filtered_df = filtered_df[filtered_df['Budget'] <= max_budget]

    # Duration filters
    if 'original_duration_months' in filtered_df.columns:
        if enable_min_duration:
            filtered_df = filtered_df[filtered_df['original_duration_months'] >= min_duration]
        if enable_max_duration:
            filtered_df = filtered_df[filtered_df['original_duration_months'] <= max_duration]

    # CPI filter
    if enable_cpi_filter and 'CPI' in filtered_df.columns:
        filtered_df = filtered_df[(filtered_df['CPI'] >= cpi_range[0]) & (filtered_df['CPI'] <= cpi_range[1])]

    # SPI filter
    if enable_spi_filter and 'SPI' in filtered_df.columns:
        filtered_df = filtered_df[(filtered_df['SPI'] >= spi_range[0]) & (filtered_df['SPI'] <= spi_range[1])]

    # SPIe filter
    if enable_spie_filter and 'SPIe' in filtered_df.columns:
        filtered_df = filtered_df[(filtered_df['SPIe'] >= spie_range[0]) & (filtered_df['SPIe'] <= spie_range[1])]

    st.markdown(f"**Projects Displayed:** {len(filtered_df)}")

    # Project Spotlight Section
    with st.expander("🎯 Project Spotlight", expanded=False):
        # Dropdown for view selection
        view_options = [
            "1. Critical Projects",
            "2. Top 10 Budget",
            "3. Top 10 Earliest",
            "4. Top 10 Latest",
            "5. Top 10 Highest SPI",
            "6. Top 10 Highest CPI",
            "7. Top 10 Lowest SPI",
            "8. Top 10 Lowest CPI",
            "9. Top 10 Longest Duration",
            "10. Top 10 Shortest Duration",
            "11. Top 10 Longest Delay",
            "12. Top 10 Highest ETC"
        ]

        selected_view = st.selectbox("Select View:", view_options, key="project_spotlight_view")

        # Initialize variables
        spotlight_projects = df.copy()
        view_description = ""

        # Filter and sort based on selected view
        if selected_view == "1. Critical Projects":
            spotlight_projects = df[df['Health_Category'] == 'Critical'].copy()
            spotlight_projects = spotlight_projects.sort_values('CPI').head(10)
            view_description = "⚠️ These projects require immediate executive intervention due to critical performance issues."

        elif selected_view == "2. Top 10 Budget":
            spotlight_projects = df.nlargest(10, 'Budget')
            view_description = "💰 Projects with the highest budgets in the portfolio."

        elif selected_view == "3. Top 10 Earliest":
            # Check for possible Plan Start column names
            plan_start_col = None
            for col in ['Plan Start', 'plan_start', 'Start Date', 'start_date']:
                if col in df.columns:
                    plan_start_col = col
                    break

            if plan_start_col:
                # Convert to datetime and sort for earliest dates (ascending)
                df_temp = df.copy()
                df_temp[f'{plan_start_col}_datetime'] = pd.to_datetime(df_temp[plan_start_col], errors='coerce')
                valid_dates = df_temp.dropna(subset=[f'{plan_start_col}_datetime'])
                if not valid_dates.empty:
                    spotlight_projects = valid_dates.nsmallest(10, f'{plan_start_col}_datetime')
                else:
                    spotlight_projects = pd.DataFrame()
            else:
                spotlight_projects = pd.DataFrame()
            view_description = "📅 Projects with the earliest planned start dates."

        elif selected_view == "4. Top 10 Latest":
            # Check for possible Plan Start column names
            plan_start_col = None
            for col in ['Plan Start', 'plan_start', 'Start Date', 'start_date']:
                if col in df.columns:
                    plan_start_col = col
                    break

            if plan_start_col:
                # Convert to datetime and sort for latest dates (descending)
                df_temp = df.copy()
                df_temp[f'{plan_start_col}_datetime'] = pd.to_datetime(df_temp[plan_start_col], errors='coerce')
                valid_dates = df_temp.dropna(subset=[f'{plan_start_col}_datetime'])
                if not valid_dates.empty:
                    spotlight_projects = valid_dates.nlargest(10, f'{plan_start_col}_datetime')
                else:
                    spotlight_projects = pd.DataFrame()
            else:
                spotlight_projects = pd.DataFrame()
            view_description = "📅 Projects with the latest planned start dates."

        elif selected_view == "5. Top 10 Highest SPI":
            spotlight_projects = df.nlargest(10, 'SPI')
            view_description = "🚀 Projects with the best schedule performance (ahead of schedule)."

        elif selected_view == "6. Top 10 Highest CPI":
            spotlight_projects = df.nlargest(10, 'CPI')
            view_description = "💎 Projects with the best cost performance (under budget)."

        elif selected_view == "7. Top 10 Lowest SPI":
            spotlight_projects = df.nsmallest(10, 'SPI')
            view_description = "⏰ Projects with the worst schedule performance (behind schedule)."

        elif selected_view == "8. Top 10 Lowest CPI":
            spotlight_projects = df.nsmallest(10, 'CPI')
            view_description = "💸 Projects with the worst cost performance (over budget)."

        elif selected_view == "9. Top 10 Longest Duration":
            # Look for duration columns - try multiple possible names
            duration_col = None
            duration_types = ['Original Dur', 'original_duration_months', 'OD', 'Actual Dur', 'actual_duration_months', 'AD', 'Likely Dur', 'forecast_duration', 'LD']
            for col in duration_types:
                if col in df.columns:
                    duration_col = col
                    break

            if duration_col:
                spotlight_projects = df.nlargest(10, duration_col)
                view_description = f"⏳ Projects with the longest duration ({duration_col})."
            else:
                spotlight_projects = pd.DataFrame()
                view_description = "⏳ Duration data not available."

        elif selected_view == "10. Top 10 Shortest Duration":
            # Look for duration columns
            duration_col = None
            duration_types = ['Original Dur', 'original_duration_months', 'OD', 'Actual Dur', 'actual_duration_months', 'AD', 'Likely Dur', 'forecast_duration', 'LD']
            for col in duration_types:
                if col in df.columns:
                    duration_col = col
                    break

            if duration_col:
                spotlight_projects = df.nsmallest(10, duration_col)
                view_description = f"⚡ Projects with the shortest duration ({duration_col})."
            else:
                spotlight_projects = pd.DataFrame()
                view_description = "⚡ Duration data not available."

        elif selected_view == "11. Top 10 Longest Delay":
            # Calculate delay as LD-OD (Likely Duration - Original Duration)
            df_temp = df.copy()

            # Find LD (Likely Duration) column
            ld_col = None
            for col in ['Likely Dur', 'forecast_duration', 'LD']:
                if col in df_temp.columns:
                    ld_col = col
                    break

            # Find OD (Original Duration) column
            od_col = None
            for col in ['Original Dur', 'original_duration_months', 'OD']:
                if col in df_temp.columns:
                    od_col = col
                    break

            if ld_col and od_col:
                # Calculate delay = LD - OD
                df_temp['Delay'] = df_temp[ld_col] - df_temp[od_col]
                # Filter projects: positive delay AND delay ≤ 3 × OD
                delayed_projects = df_temp[
                    (df_temp['Delay'] > 0) &
                    (df_temp['Delay'] <= 3 * df_temp[od_col])
                ]
                if not delayed_projects.empty:
                    spotlight_projects = delayed_projects.nlargest(10, 'Delay')
                    view_description = f"🐌 Projects with the longest delays (LD - OD ≤ 3×OD, in months)."
                else:
                    spotlight_projects = pd.DataFrame()
                    view_description = "🐌 No projects with valid delays found (within 3×OD limit)."
            else:
                # Fallback to lowest SPI if duration columns not available
                spotlight_projects = df.nsmallest(10, 'SPI')
                view_description = "🐌 Projects with the most significant schedule delays (using SPI)."

        elif selected_view == "12. Top 10 Highest ETC":
            # Calculate ETC if not available (ETC = EAC - AC)
            df_temp = df.copy()
            if 'ETC' not in df_temp.columns:
                if 'EAC' in df_temp.columns and 'Actual Cost' in df_temp.columns:
                    df_temp['ETC'] = df_temp['EAC'] - df_temp['Actual Cost']
                elif 'EAC' in df_temp.columns and 'AC' in df_temp.columns:
                    df_temp['ETC'] = df_temp['EAC'] - df_temp['AC']
                else:
                    df_temp['ETC'] = 0  # Default if we can't calculate

            spotlight_projects = df_temp.nlargest(10, 'ETC')
            view_description = "🔮 Projects with the highest remaining costs to complete (ETC = EAC - AC)."

        # Ensure ETC column is available if needed (calculate if missing)
        if 'ETC' not in spotlight_projects.columns and selected_view == "12. Top 10 Highest ETC":
            if 'EAC' in spotlight_projects.columns and 'Actual Cost' in spotlight_projects.columns:
                spotlight_projects['ETC'] = spotlight_projects['EAC'] - spotlight_projects['Actual Cost']
            elif 'EAC' in spotlight_projects.columns and 'AC' in spotlight_projects.columns:
                spotlight_projects['ETC'] = spotlight_projects['EAC'] - spotlight_projects['AC']

        # Ensure Delay column is available if needed (calculate if missing)
        if 'Delay' not in spotlight_projects.columns and selected_view == "11. Top 10 Longest Delay":
            # Find LD (Likely Duration) column
            ld_col = None
            for col in ['Likely Dur', 'forecast_duration', 'LD']:
                if col in spotlight_projects.columns:
                    ld_col = col
                    break

            # Find OD (Original Duration) column
            od_col = None
            for col in ['Original Dur', 'original_duration_months', 'OD']:
                if col in spotlight_projects.columns:
                    od_col = col
                    break

            if ld_col and od_col:
                # Calculate delay and apply 3×OD restriction
                spotlight_projects['Delay'] = spotlight_projects[ld_col] - spotlight_projects[od_col]
                # Cap delay at 3 times the original duration
                spotlight_projects['Delay'] = spotlight_projects['Delay'].where(
                    spotlight_projects['Delay'] <= 3 * spotlight_projects[od_col],
                    3 * spotlight_projects[od_col]
                )

        # Display the results
        if not spotlight_projects.empty:
            # Prepare data for table display with appropriate columns for each view
            # Base columns that should appear in all views
            base_columns = ['Project Name']

            # Financial columns (using different possible names)
            financial_columns = []
            # BAC (Budget at Completion)
            for col in ['Budget', 'BAC', 'bac']:
                if col in spotlight_projects.columns:
                    financial_columns.append(col)
                    break

            # AC (Actual Cost)
            for col in ['Actual Cost', 'AC', 'ac']:
                if col in spotlight_projects.columns:
                    financial_columns.append(col)
                    break

            # EV (Earned Value)
            for col in ['Earned Value', 'EV', 'ev']:
                if col in spotlight_projects.columns:
                    financial_columns.append(col)
                    break

            # Performance columns
            performance_columns = ['CPI', 'SPI', 'SPIe']

            # View-specific columns
            if selected_view in ["2. Top 10 Budget", "12. Top 10 Highest ETC"]:
                specific_columns = ['ETC', 'EAC']
                display_columns = base_columns + financial_columns + performance_columns + specific_columns

            elif selected_view in ["3. Top 10 Earliest", "4. Top 10 Latest"]:
                # Find the plan start column
                plan_start_col = None
                for col in ['Plan Start', 'plan_start', 'Start Date', 'start_date']:
                    if col in spotlight_projects.columns:
                        plan_start_col = col
                        break

                specific_columns = [plan_start_col] if plan_start_col else []
                display_columns = base_columns + specific_columns + financial_columns + performance_columns

            elif selected_view in ["9. Top 10 Longest Duration", "10. Top 10 Shortest Duration"]:
                # Dynamically determine which duration column to show
                duration_display_col = None
                duration_types = ['Original Dur', 'original_duration_months', 'OD', 'Actual Dur', 'actual_duration_months', 'AD', 'Likely Dur', 'forecast_duration', 'LD']
                for col in duration_types:
                    if col in spotlight_projects.columns:
                        duration_display_col = col
                        break

                specific_columns = [duration_display_col] if duration_display_col else []
                display_columns = base_columns + specific_columns + financial_columns + performance_columns

            elif selected_view == "11. Top 10 Longest Delay":
                # For delay view, show Delay column plus original and likely durations
                delay_columns = []

                # Add Delay column if it exists
                if 'Delay' in spotlight_projects.columns:
                    delay_columns.append('Delay')

                # Add original duration column
                for col in ['Original Dur', 'original_duration_months', 'OD']:
                    if col in spotlight_projects.columns:
                        delay_columns.append(col)
                        break

                # Add actual duration column
                for col in ['Actual Dur', 'actual_duration_months', 'AD']:
                    if col in spotlight_projects.columns:
                        delay_columns.append(col)
                        break

                # Add likely duration column
                for col in ['Likely Dur', 'forecast_duration', 'LD']:
                    if col in spotlight_projects.columns:
                        delay_columns.append(col)
                        break

                display_columns = base_columns + delay_columns + financial_columns + performance_columns

            else:
                # Default view - include EAC
                specific_columns = ['EAC']
                display_columns = base_columns + financial_columns + performance_columns + specific_columns

            # Remove None values and duplicates while preserving order
            display_columns = [col for col in display_columns if col is not None]
            display_columns = list(dict.fromkeys(display_columns))  # Remove duplicates while preserving order

            # Check which columns are available
            available_columns = [col for col in display_columns if col in spotlight_projects.columns]

            if available_columns:
                spotlight_table = spotlight_projects[available_columns].copy()

                # Update column headers to include currency information instead of formatting values
                currency_columns = ['Budget', 'BAC', 'bac', 'Actual Cost', 'AC', 'ac', 'Earned Value', 'EV', 'ev', 'Planned Value', 'PV', 'pv', 'EAC', 'ETC']
                spotlight_column_renames = {}
                for col in currency_columns:
                    if col in spotlight_table.columns:
                        if col in ['Budget', 'BAC', 'bac']:
                            spotlight_column_renames[col] = f'BAC ({currency_symbol}{currency_postfix})' if currency_postfix else f'BAC ({currency_symbol})'
                        elif col in ['Actual Cost', 'AC', 'ac']:
                            spotlight_column_renames[col] = f'AC ({currency_symbol}{currency_postfix})' if currency_postfix else f'AC ({currency_symbol})'
                        elif col in ['Earned Value', 'EV', 'ev']:
                            spotlight_column_renames[col] = f'EV ({currency_symbol}{currency_postfix})' if currency_postfix else f'EV ({currency_symbol})'
                        elif col in ['Planned Value', 'PV', 'pv']:
                            spotlight_column_renames[col] = f'PV ({currency_symbol}{currency_postfix})' if currency_postfix else f'PV ({currency_symbol})'
                        elif col == 'EAC':
                            spotlight_column_renames[col] = f'EAC ({currency_symbol}{currency_postfix})' if currency_postfix else f'EAC ({currency_symbol})'
                        elif col == 'ETC':
                            spotlight_column_renames[col] = f'ETC ({currency_symbol}{currency_postfix})' if currency_postfix else f'ETC ({currency_symbol})'

                # Apply column renames for spotlight table
                spotlight_table = spotlight_table.rename(columns=spotlight_column_renames)

                # Create column configuration for spotlight table
                spotlight_column_config = {}

                # Configure currency columns for spotlight table
                spotlight_currency_original_columns = ['Budget', 'BAC', 'bac', 'Actual Cost', 'AC', 'ac', 'Earned Value', 'EV', 'ev', 'Planned Value', 'PV', 'pv', 'EAC', 'ETC']
                for col in spotlight_currency_original_columns:
                    if col in spotlight_projects.columns:  # Check original column names before renaming
                        renamed_col = spotlight_column_renames.get(col, col)  # Get the renamed column name
                        if renamed_col in spotlight_table.columns:
                            spotlight_column_config[renamed_col] = st.column_config.NumberColumn(
                                renamed_col,
                                format="%.2f",
                                help=f"Values in {currency_symbol}{currency_postfix}" if currency_postfix else f"Values in {currency_symbol}"
                            )

                # Configure performance indices columns for spotlight table
                for col in ['CPI', 'SPI', 'SPIe']:
                    if col in spotlight_table.columns:
                        spotlight_column_config[col] = st.column_config.NumberColumn(
                            col,
                            format="%.3f",
                            help=f"{col} performance index"
                        )

                # Configure duration columns
                duration_format_cols = ['Original Dur', 'original_duration_months', 'OD', 'Actual Dur', 'actual_duration_months', 'AD', 'Likely Dur', 'forecast_duration', 'LD']
                for col in duration_format_cols:
                    if col in spotlight_table.columns:
                        spotlight_column_config[col] = st.column_config.NumberColumn(
                            col,
                            format="%.1f",
                            help="Duration in months"
                        )

                # Configure Delay column
                if 'Delay' in spotlight_table.columns:
                    spotlight_column_config['Delay'] = st.column_config.NumberColumn(
                        'Delay',
                        format="%.1f",
                        help="Schedule delay in months"
                    )

                # Format dates if present - handle multiple possible date column names
                date_columns = ['Plan Start', 'plan_start', 'Start Date', 'start_date']
                for col in date_columns:
                    if col in spotlight_table.columns:
                        spotlight_table[col] = pd.to_datetime(spotlight_table[col], errors='coerce').dt.strftime('%Y-%m-%d')

                # Duration and delay columns are now handled by column configuration
                # (Remove string formatting to preserve numeric sorting)

                # Apply conditional styling based on view type
                if selected_view == "1. Critical Projects":
                    # Style critical projects with red background
                    def highlight_critical_projects(val):
                        return 'background-color: #ffebee; color: #d32f2f; font-weight: bold;'
                    try:
                        styled_table = spotlight_table.style.applymap(highlight_critical_projects)
                        st.dataframe(styled_table, width='stretch', height=300, column_config=spotlight_column_config)
                    except:
                        st.dataframe(spotlight_table, width='stretch', height=300, column_config=spotlight_column_config)
                else:
                    # Standard display for other views
                    st.dataframe(spotlight_table, width='stretch', height=300, column_config=spotlight_column_config)

                st.markdown(f"**{view_description}**")
            else:
                st.info("Required data columns not available for this view.")
        else:
            st.info(f"No data available for {selected_view.split('.')[1].strip()}.")


    # Interactive Data Explorer
    with st.expander("Advanced Portfolio Analytics", expanded=False):
        # Budget_Category already created earlier in the code
        # Filters are now in a separate expander above
        # filtered_df is already created and filtered above

        # Projects Expander
        with st.expander("📋 Projects", expanded=False):
            # Display filtered data with enhanced columns
            if org_columns:
                display_columns = ['Project Name', org_columns[0], 'Budget_Category', 'Budget', 'CPI', 'SPI', 'SPIe', 'Health_Category', 'Actual Cost', 'Plan Value', 'Earned Value', 'EAC']
            else:
                display_columns = ['Project Name', 'Organization', 'Budget_Category', 'Budget', 'CPI', 'SPI', 'SPIe', 'Health_Category', 'Actual Cost', 'Plan Value', 'Earned Value', 'EAC']

            available_columns = [col for col in display_columns if col in filtered_df.columns]

            if available_columns:
                # Format the dataframe for better display
                display_df = filtered_df[available_columns].copy()

                # Update column names to include currency information
                column_renames = {}
                if 'Budget' in display_df.columns:
                    column_renames['Budget'] = f'Budget ({currency_symbol}{currency_postfix})' if currency_postfix else f'Budget ({currency_symbol})'
                if 'Actual Cost' in display_df.columns:
                    column_renames['Actual Cost'] = f'AC ({currency_symbol}{currency_postfix})' if currency_postfix else f'AC ({currency_symbol})'
                if 'Plan Value' in display_df.columns:
                    column_renames['Plan Value'] = f'PV ({currency_symbol}{currency_postfix})' if currency_postfix else f'PV ({currency_symbol})'
                if 'Earned Value' in display_df.columns:
                    column_renames['Earned Value'] = f'EV ({currency_symbol}{currency_postfix})' if currency_postfix else f'EV ({currency_symbol})'
                if 'EAC' in display_df.columns:
                    column_renames['EAC'] = f'EAC ({currency_symbol}{currency_postfix})' if currency_postfix else f'EAC ({currency_symbol})'

                # Apply column renames
                display_df = display_df.rename(columns=column_renames)

                # Create column configuration for proper numeric formatting and alignment
                column_config = {}

                # Configure currency columns
                currency_value_columns = ['Budget', 'Actual Cost', 'Plan Value', 'Earned Value', 'EAC']
                for col in currency_value_columns:
                    if col in filtered_df.columns:  # Check original column names
                        renamed_col = column_renames.get(col, col)  # Get the renamed column name
                        if renamed_col in display_df.columns:
                            column_config[renamed_col] = st.column_config.NumberColumn(
                                renamed_col,
                                format="%.2f",
                                help=f"Values in {currency_symbol}{currency_postfix}" if currency_postfix else f"Values in {currency_symbol}"
                            )

                # Configure performance indices columns
                for col in ['CPI', 'SPI', 'SPIe']:
                    if col in display_df.columns:
                        column_config[col] = st.column_config.NumberColumn(
                            col,
                            format="%.3f",
                            help=f"{col} performance index"
                        )

                # Configure other numeric columns
                if 'Project Count' in display_df.columns:
                    column_config['Project Count'] = st.column_config.NumberColumn(
                        'Project Count',
                        format="%d"
                    )

                # Color-code the health status
                def highlight_health(val):
                    if val == 'Critical':
                        return 'background-color: #ffebee'
                    elif val == 'At Risk':
                        return 'background-color: #fff3e0'
                    elif val == 'Healthy':
                        return 'background-color: #e8f5e8'
                    return ''

                if 'Health_Category' in display_df.columns:
                    styled_df = display_df.style.applymap(highlight_health, subset=['Health_Category'])
                    st.dataframe(styled_df, width='stretch', height=400, column_config=column_config)
                else:
                    st.dataframe(display_df, width='stretch', height=400, column_config=column_config)

        # Organizations Expander
        with st.expander("🏢 Organizations", expanded=False):
            # Get organization column name
            org_col = org_columns[0] if org_columns else 'Organization'

            if org_col in filtered_df.columns and len(filtered_df) > 0:
                # Group by organization and calculate consolidated metrics
                try:
                    # Define aggregation columns dynamically
                    agg_dict = {
                        'Project Name': 'count',  # Project count
                        'Budget': 'sum',          # Total budget
                        'Actual Cost': 'sum',     # Total actual cost
                        'EAC': 'sum',            # Total EAC
                        'Health_Category': lambda x: x.mode().iloc[0] if len(x.mode()) > 0 else 'Unknown'  # Most common health status
                    }

                    # Add Plan Value and Earned Value if they exist
                    if 'Plan Value' in filtered_df.columns:
                        agg_dict['Plan Value'] = 'sum'
                    if 'Earned Value' in filtered_df.columns:
                        agg_dict['Earned Value'] = 'sum'

                    org_summary = filtered_df.groupby(org_col).agg(agg_dict).rename(columns={'Project Name': 'Project Count'})

                    # Calculate weighted averages for CPI, SPI, SPIe
                    org_weighted_metrics = filtered_df.groupby(org_col).apply(
                        lambda group: pd.Series({
                            'CPI': (group['CPI'] * group['Budget']).sum() / group['Budget'].sum() if group['Budget'].sum() > 0 else 0,
                            'SPI': (group['SPI'] * group['Budget']).sum() / group['Budget'].sum() if group['Budget'].sum() > 0 else 0,
                            'SPIe': (group['SPIe'] * group['Budget']).sum() / group['Budget'].sum() if group['Budget'].sum() > 0 else 0
                        })
                    )

                    # Combine the metrics
                    org_display = pd.concat([org_summary, org_weighted_metrics], axis=1)

                    # Check if we have valid data
                    if len(org_display) > 0:
                        # Reorder columns to match project view (without Budget_Category)
                        base_org_columns = ['Project Count', 'Budget', 'CPI', 'SPI', 'SPIe', 'Health_Category', 'Actual Cost']
                        optional_columns = []
                        if 'Plan Value' in org_display.columns:
                            optional_columns.append('Plan Value')
                        if 'Earned Value' in org_display.columns:
                            optional_columns.append('Earned Value')
                        if 'EAC' in org_display.columns:
                            optional_columns.append('EAC')

                        org_display_columns = base_org_columns + optional_columns
                        available_org_columns = [col for col in org_display_columns if col in org_display.columns]
                        org_display = org_display[available_org_columns]

                        # Reset index to ensure unique indices for styling
                        org_display = org_display.reset_index()

                        # Format the organizational data for display
                        org_display_formatted = org_display.copy()

                        # Update column names to include currency information for organizations
                        org_column_renames = {}
                        if 'Budget' in org_display_formatted.columns:
                            org_column_renames['Budget'] = f'Budget ({currency_symbol}{currency_postfix})' if currency_postfix else f'Budget ({currency_symbol})'
                        if 'Actual Cost' in org_display_formatted.columns:
                            org_column_renames['Actual Cost'] = f'AC ({currency_symbol}{currency_postfix})' if currency_postfix else f'AC ({currency_symbol})'
                        if 'Plan Value' in org_display_formatted.columns:
                            org_column_renames['Plan Value'] = f'PV ({currency_symbol}{currency_postfix})' if currency_postfix else f'PV ({currency_symbol})'
                        if 'Earned Value' in org_display_formatted.columns:
                            org_column_renames['Earned Value'] = f'EV ({currency_symbol}{currency_postfix})' if currency_postfix else f'EV ({currency_symbol})'
                        if 'EAC' in org_display_formatted.columns:
                            org_column_renames['EAC'] = f'EAC ({currency_symbol}{currency_postfix})' if currency_postfix else f'EAC ({currency_symbol})'

                        # Apply column renames for organizations
                        org_display_formatted = org_display_formatted.rename(columns=org_column_renames)

                        # Create column configuration for organizations table
                        org_column_config = {}

                        # Configure currency columns for organizations
                        org_currency_value_columns = ['Budget', 'Actual Cost', 'Plan Value', 'Earned Value', 'EAC']
                        for col in org_currency_value_columns:
                            if col in org_display.columns:  # Check original column names before renaming
                                renamed_col = org_column_renames.get(col, col)  # Get the renamed column name
                                if renamed_col in org_display_formatted.columns:
                                    org_column_config[renamed_col] = st.column_config.NumberColumn(
                                        renamed_col,
                                        format="%.2f",
                                        help=f"Values in {currency_symbol}{currency_postfix}" if currency_postfix else f"Values in {currency_symbol}"
                                    )

                        # Configure performance indices columns for organizations
                        for col in ['CPI', 'SPI', 'SPIe']:
                            if col in org_display_formatted.columns:
                                org_column_config[col] = st.column_config.NumberColumn(
                                    col,
                                    format="%.3f",
                                    help=f"{col} performance index"
                                )

                        # Configure Project Count column
                        if 'Project Count' in org_display_formatted.columns:
                            org_column_config['Project Count'] = st.column_config.NumberColumn(
                                'Project Count',
                                format="%d"
                            )

                        # Apply health status styling with error handling
                        try:
                            if 'Health_Category' in org_display_formatted.columns and len(org_display_formatted) > 0:
                                styled_org_df = org_display_formatted.style.applymap(highlight_health, subset=['Health_Category'])
                                st.dataframe(styled_org_df, width='stretch', height=300, column_config=org_column_config)
                            else:
                                st.dataframe(org_display_formatted, width='stretch', height=300, column_config=org_column_config)
                        except (KeyError, ValueError) as e:
                            # Fallback to unstyled dataframe if styling fails
                            st.dataframe(org_display_formatted, width='stretch', height=300)
                    else:
                        st.info("No organization data available with current filters.")

                except Exception as e:
                    st.error(f"Error processing organization data: {str(e)}")
                    st.info("Unable to display organizational consolidation with current data.")
            else:
                if len(filtered_df) == 0:
                    st.info("No projects available for organizational consolidation.")
                else:
                    st.info("Organization data not available for consolidation.")

        # Budget Tiers Expander
        with st.expander("🎯 Budget Tiers", expanded=False):
            if 'Budget_Category' in filtered_df.columns and len(filtered_df) > 0:
                # Group by budget tier and calculate consolidated metrics
                try:
                    # Define aggregation columns dynamically
                    tier_agg_dict = {
                        'Project Name': 'count',  # Project count
                        'Budget': 'sum',          # Total budget
                        'Actual Cost': 'sum',     # Total actual cost
                        'EAC': 'sum',            # Total EAC
                        'Health_Category': lambda x: x.mode().iloc[0] if len(x.mode()) > 0 else 'Unknown'  # Most common health status
                    }

                    # Add Plan Value and Earned Value if they exist
                    if 'Plan Value' in filtered_df.columns:
                        tier_agg_dict['Plan Value'] = 'sum'
                    if 'Earned Value' in filtered_df.columns:
                        tier_agg_dict['Earned Value'] = 'sum'

                    tier_summary = filtered_df.groupby('Budget_Category').agg(tier_agg_dict).rename(columns={'Project Name': 'Project Count'})

                    # Calculate weighted averages for CPI, SPI, SPIe
                    tier_weighted_metrics = filtered_df.groupby('Budget_Category').apply(
                        lambda group: pd.Series({
                            'CPI': (group['CPI'] * group['Budget']).sum() / group['Budget'].sum() if group['Budget'].sum() > 0 else 0,
                            'SPI': (group['SPI'] * group['Budget']).sum() / group['Budget'].sum() if group['Budget'].sum() > 0 else 0,
                            'SPIe': (group['SPIe'] * group['Budget']).sum() / group['Budget'].sum() if group['Budget'].sum() > 0 else 0
                        })
                    )

                    # Combine the metrics
                    tier_display = pd.concat([tier_summary, tier_weighted_metrics], axis=1)

                    # Check if we have valid data
                    if len(tier_display) > 0:
                        # Reorder columns to match organization view
                        base_tier_columns = ['Project Count', 'Budget', 'CPI', 'SPI', 'SPIe', 'Health_Category', 'Actual Cost']
                        optional_columns = []
                        if 'Plan Value' in tier_display.columns:
                            optional_columns.append('Plan Value')
                        if 'Earned Value' in tier_display.columns:
                            optional_columns.append('Earned Value')
                        if 'EAC' in tier_display.columns:
                            optional_columns.append('EAC')

                        tier_display_columns = base_tier_columns + optional_columns
                        available_tier_columns = [col for col in tier_display_columns if col in tier_display.columns]
                        tier_display = tier_display[available_tier_columns]

                        # Reset index to ensure unique indices for styling
                        tier_display = tier_display.reset_index()

                        # Sort by tier order (Tier 4 -> Tier 1 for descending budget)
                        tier_config = st.session_state.config_dict.get('controls', {}).get('tier_config', {})
                        tier_names = tier_config.get('tier_names', ['Tier 1', 'Tier 2', 'Tier 3', 'Tier 4'])

                        # Create a custom sort order (reverse tier order so Tier 4 appears first)
                        tier_order = {name: i for i, name in enumerate(reversed(tier_names))}
                        tier_display['_sort_order'] = tier_display['Budget_Category'].map(tier_order)
                        tier_display = tier_display.sort_values('_sort_order').drop('_sort_order', axis=1)

                        # Format the tier data for display
                        tier_display_formatted = tier_display.copy()

                        # Update column names to include currency information
                        tier_column_renames = {}
                        if 'Budget' in tier_display_formatted.columns:
                            tier_column_renames['Budget'] = f'Budget ({currency_symbol}{currency_postfix})' if currency_postfix else f'Budget ({currency_symbol})'
                        if 'Actual Cost' in tier_display_formatted.columns:
                            tier_column_renames['Actual Cost'] = f'AC ({currency_symbol}{currency_postfix})' if currency_postfix else f'AC ({currency_symbol})'
                        if 'Plan Value' in tier_display_formatted.columns:
                            tier_column_renames['Plan Value'] = f'PV ({currency_symbol}{currency_postfix})' if currency_postfix else f'PV ({currency_symbol})'
                        if 'Earned Value' in tier_display_formatted.columns:
                            tier_column_renames['Earned Value'] = f'EV ({currency_symbol}{currency_postfix})' if currency_postfix else f'EV ({currency_symbol})'
                        if 'EAC' in tier_display_formatted.columns:
                            tier_column_renames['EAC'] = f'EAC ({currency_symbol}{currency_postfix})' if currency_postfix else f'EAC ({currency_symbol})'

                        # Apply column renames
                        tier_display_formatted = tier_display_formatted.rename(columns=tier_column_renames)

                        # Create column configuration for tier table
                        tier_column_config = {}

                        # Configure currency columns
                        tier_currency_value_columns = ['Budget', 'Actual Cost', 'Plan Value', 'Earned Value', 'EAC']
                        for col in tier_currency_value_columns:
                            if col in tier_display.columns:
                                renamed_col = tier_column_renames.get(col, col)
                                if renamed_col in tier_display_formatted.columns:
                                    tier_column_config[renamed_col] = st.column_config.NumberColumn(
                                        renamed_col,
                                        format="%.2f",
                                        help=f"Values in {currency_symbol}{currency_postfix}" if currency_postfix else f"Values in {currency_symbol}"
                                    )

                        # Configure performance indices columns
                        for col in ['CPI', 'SPI', 'SPIe']:
                            if col in tier_display_formatted.columns:
                                tier_column_config[col] = st.column_config.NumberColumn(
                                    col,
                                    format="%.3f",
                                    help=f"{col} performance index"
                                )

                        # Configure Project Count column
                        if 'Project Count' in tier_display_formatted.columns:
                            tier_column_config['Project Count'] = st.column_config.NumberColumn(
                                'Project Count',
                                format="%d"
                            )

                        # Display summary statistics
                        st.markdown("### 📊 Budget Tier Summary")

                        # Show tier configuration info
                        if tier_config:
                            cutoff_points = tier_config.get('cutoff_points', [])
                            if cutoff_points and len(cutoff_points) >= 3:
                                st.caption(f"**Tier Boundaries:** < {currency_symbol}{cutoff_points[0]:,.0f} | {currency_symbol}{cutoff_points[0]:,.0f}-{currency_symbol}{cutoff_points[1]:,.0f} | {currency_symbol}{cutoff_points[1]:,.0f}-{currency_symbol}{cutoff_points[2]:,.0f} | ≥ {currency_symbol}{cutoff_points[2]:,.0f}")

                        # Calculate height based on number of rows (header + 4 tiers)
                        num_rows = len(tier_display_formatted)
                        table_height = min((num_rows + 1) * 35 + 3, 200)  # 35px per row + 3px padding, max 200px

                        # Apply health status styling with error handling
                        try:
                            if 'Health_Category' in tier_display_formatted.columns and len(tier_display_formatted) > 0:
                                styled_tier_df = tier_display_formatted.style.applymap(highlight_health, subset=['Health_Category'])
                                st.dataframe(styled_tier_df, width='stretch', height=table_height, column_config=tier_column_config)
                            else:
                                st.dataframe(tier_display_formatted, width='stretch', height=table_height, column_config=tier_column_config)
                        except (KeyError, ValueError) as e:
                            # Fallback to unstyled dataframe if styling fails
                            st.dataframe(tier_display_formatted, width='stretch', height=table_height, column_config=tier_column_config)

                        # Add tier ranges information
                        st.markdown("---")
                        st.markdown(f'<p style="color: black; margin-bottom: 5px;"><strong>Budget Tier Definitions:</strong></p>', unsafe_allow_html=True)

                        # Display tier ranges
                        tier_config = st.session_state.config_dict.get('controls', {}).get('tier_config', {})
                        tier_names = tier_config.get('tier_names', ['Tier 1', 'Tier 2', 'Tier 3', 'Tier 4'])
                        cutoff_points = tier_config.get('cutoff_points', [4000, 8000, 15000])

                        if cutoff_points and len(cutoff_points) >= 3:
                            st.markdown(f'<p style="color: black; font-size: 14px;">{tier_names[0]}: &lt; {currency_symbol}{cutoff_points[0]:,.0f}  |  {tier_names[1]}: {currency_symbol}{cutoff_points[0]:,.0f}-{currency_symbol}{cutoff_points[1]:,.0f}  |  {tier_names[2]}: {currency_symbol}{cutoff_points[1]:,.0f}-{currency_symbol}{cutoff_points[2]:,.0f}  |  {tier_names[3]}: ≥ {currency_symbol}{cutoff_points[2]:,.0f}</p>', unsafe_allow_html=True)

                    else:
                        st.info("No budget tier data available with current filters.")

                except Exception as e:
                    st.error(f"Error processing budget tier data: {str(e)}")
                    st.info("Unable to display budget tier consolidation with current data.")
            else:
                if len(filtered_df) == 0:
                    st.info("No projects available for budget tier consolidation.")
                else:
                    st.info("Budget tier data not available. Ensure budget tiers are configured in File Management.")

        # Duration Tiers Expander
        with st.expander("⏱️ Duration Tiers", expanded=False):
            # First ensure Duration_Category exists
            if 'Duration_Category' not in filtered_df.columns and 'original_duration_months' in filtered_df.columns:
                # Get duration tier config
                duration_tier_config = st.session_state.config_dict.get('controls', {}).get('duration_tier_config', {})
                default_duration_tier_config = {
                    'cutoff_points': [6, 12, 24],
                    'tier_names': ['Short', 'Medium', 'Long', 'Extra Long']
                }
                cutoffs = duration_tier_config.get('cutoff_points', default_duration_tier_config['cutoff_points'])
                tier_names = duration_tier_config.get('tier_names', default_duration_tier_config['tier_names'])

                def get_duration_tier(duration):
                    """Determine tier based on duration and cutoff points."""
                    if pd.isna(duration):
                        return "Unknown"
                    duration = int(duration)
                    # Use >= logic like budget tiers (check from high to low)
                    if duration >= cutoffs[2]:
                        return tier_names[3]
                    elif duration >= cutoffs[1]:
                        return tier_names[2]
                    elif duration >= cutoffs[0]:
                        return tier_names[1]
                    else:
                        return tier_names[0]

                filtered_df['Duration_Category'] = filtered_df['original_duration_months'].apply(get_duration_tier)

            if 'Duration_Category' in filtered_df.columns and len(filtered_df) > 0:
                # Group by duration tier and calculate consolidated metrics
                try:
                    # Define aggregation columns dynamically
                    duration_tier_agg_dict = {
                        'Project Name': 'count',  # Project count
                        'Budget': 'sum',          # Total budget
                        'Actual Cost': 'sum',     # Total actual cost
                        'EAC': 'sum',            # Total EAC
                        'Health_Category': lambda x: x.mode().iloc[0] if len(x.mode()) > 0 else 'Unknown'  # Most common health status
                    }

                    # Add Plan Value and Earned Value if they exist
                    if 'Plan Value' in filtered_df.columns:
                        duration_tier_agg_dict['Plan Value'] = 'sum'
                    if 'Earned Value' in filtered_df.columns:
                        duration_tier_agg_dict['Earned Value'] = 'sum'

                    # Add original_duration_months for average calculation
                    if 'original_duration_months' in filtered_df.columns:
                        duration_tier_agg_dict['original_duration_months'] = 'mean'

                    duration_tier_summary = filtered_df.groupby('Duration_Category').agg(duration_tier_agg_dict).rename(columns={'Project Name': 'Project Count'})

                    # Calculate weighted averages for CPI, SPI, SPIe
                    duration_tier_weighted_metrics = filtered_df.groupby('Duration_Category').apply(
                        lambda group: pd.Series({
                            'CPI': (group['CPI'] * group['Budget']).sum() / group['Budget'].sum() if group['Budget'].sum() > 0 else 0,
                            'SPI': (group['SPI'] * group['Budget']).sum() / group['Budget'].sum() if group['Budget'].sum() > 0 else 0,
                            'SPIe': (group['SPIe'] * group['Budget']).sum() / group['Budget'].sum() if group['Budget'].sum() > 0 else 0
                        })
                    )

                    # Combine the metrics
                    duration_tier_display = pd.concat([duration_tier_summary, duration_tier_weighted_metrics], axis=1)

                    # Check if we have valid data
                    if len(duration_tier_display) > 0:
                        # Reorder columns
                        base_duration_tier_columns = ['Project Count', 'Budget', 'CPI', 'SPI', 'SPIe', 'Health_Category', 'Actual Cost']
                        optional_columns = []
                        if 'Plan Value' in duration_tier_display.columns:
                            optional_columns.append('Plan Value')
                        if 'Earned Value' in duration_tier_display.columns:
                            optional_columns.append('Earned Value')
                        if 'EAC' in duration_tier_display.columns:
                            optional_columns.append('EAC')
                        if 'original_duration_months' in duration_tier_display.columns:
                            optional_columns.append('original_duration_months')

                        duration_tier_display_columns = base_duration_tier_columns + optional_columns
                        available_duration_tier_columns = [col for col in duration_tier_display_columns if col in duration_tier_display.columns]
                        duration_tier_display = duration_tier_display[available_duration_tier_columns]

                        # Reset index
                        duration_tier_display = duration_tier_display.reset_index()

                        # Sort by tier order (Extra Long -> Short for descending duration)
                        duration_tier_config = st.session_state.config_dict.get('controls', {}).get('duration_tier_config', {})
                        duration_tier_names = duration_tier_config.get('tier_names', ['Short', 'Medium', 'Long', 'Extra Long'])

                        # Create a custom sort order (reverse tier order so Extra Long appears first)
                        duration_tier_order = {name: i for i, name in enumerate(reversed(duration_tier_names))}
                        duration_tier_display['_sort_order'] = duration_tier_display['Duration_Category'].map(duration_tier_order)
                        duration_tier_display = duration_tier_display.sort_values('_sort_order').drop('_sort_order', axis=1)

                        # Format the tier data for display
                        duration_tier_display_formatted = duration_tier_display.copy()

                        # Update column names to include currency information
                        duration_tier_column_renames = {}
                        if 'Budget' in duration_tier_display_formatted.columns:
                            duration_tier_column_renames['Budget'] = f'Budget ({currency_symbol}{currency_postfix})' if currency_postfix else f'Budget ({currency_symbol})'
                        if 'Actual Cost' in duration_tier_display_formatted.columns:
                            duration_tier_column_renames['Actual Cost'] = f'AC ({currency_symbol}{currency_postfix})' if currency_postfix else f'AC ({currency_symbol})'
                        if 'Plan Value' in duration_tier_display_formatted.columns:
                            duration_tier_column_renames['Plan Value'] = f'PV ({currency_symbol}{currency_postfix})' if currency_postfix else f'PV ({currency_symbol})'
                        if 'Earned Value' in duration_tier_display_formatted.columns:
                            duration_tier_column_renames['Earned Value'] = f'EV ({currency_symbol}{currency_postfix})' if currency_postfix else f'EV ({currency_symbol})'
                        if 'EAC' in duration_tier_display_formatted.columns:
                            duration_tier_column_renames['EAC'] = f'EAC ({currency_symbol}{currency_postfix})' if currency_postfix else f'EAC ({currency_symbol})'
                        if 'original_duration_months' in duration_tier_display_formatted.columns:
                            duration_tier_column_renames['original_duration_months'] = 'Avg Duration (months)'

                        # Apply column renames
                        duration_tier_display_formatted = duration_tier_display_formatted.rename(columns=duration_tier_column_renames)

                        # Create column configuration for duration tier table
                        duration_tier_column_config = {}

                        # Configure currency columns
                        duration_tier_currency_value_columns = ['Budget', 'Actual Cost', 'Plan Value', 'Earned Value', 'EAC']
                        for col in duration_tier_currency_value_columns:
                            if col in duration_tier_display.columns:
                                renamed_col = duration_tier_column_renames.get(col, col)
                                if renamed_col in duration_tier_display_formatted.columns:
                                    duration_tier_column_config[renamed_col] = st.column_config.NumberColumn(
                                        renamed_col,
                                        format="%.2f",
                                        help=f"Values in {currency_symbol}{currency_postfix}" if currency_postfix else f"Values in {currency_symbol}"
                                    )

                        # Configure performance indices columns
                        for col in ['CPI', 'SPI', 'SPIe']:
                            if col in duration_tier_display_formatted.columns:
                                duration_tier_column_config[col] = st.column_config.NumberColumn(
                                    col,
                                    format="%.3f",
                                    help=f"{col} performance index"
                                )

                        # Configure Project Count column
                        if 'Project Count' in duration_tier_display_formatted.columns:
                            duration_tier_column_config['Project Count'] = st.column_config.NumberColumn(
                                'Project Count',
                                format="%d"
                            )

                        # Configure Average Duration column
                        if 'Avg Duration (months)' in duration_tier_display_formatted.columns:
                            duration_tier_column_config['Avg Duration (months)'] = st.column_config.NumberColumn(
                                'Avg Duration (months)',
                                format="%.1f",
                                help="Average project duration in months"
                            )

                        # Display summary statistics
                        st.markdown("### 📊 Duration Tier Summary")

                        # Show tier configuration info
                        if duration_tier_config:
                            cutoff_points = duration_tier_config.get('cutoff_points', [])
                            if cutoff_points and len(cutoff_points) >= 3:
                                st.caption(f"**Tier Boundaries:** < {cutoff_points[0]} months | {cutoff_points[0]}-{cutoff_points[1]} months | {cutoff_points[1]}-{cutoff_points[2]} months | ≥ {cutoff_points[2]} months")

                        # Calculate height based on number of rows (header + 4 tiers)
                        num_duration_rows = len(duration_tier_display_formatted)
                        duration_table_height = min((num_duration_rows + 1) * 35 + 3, 200)  # 35px per row + 3px padding, max 200px

                        # Apply health status styling with error handling
                        try:
                            if 'Health_Category' in duration_tier_display_formatted.columns and len(duration_tier_display_formatted) > 0:
                                styled_duration_tier_df = duration_tier_display_formatted.style.applymap(highlight_health, subset=['Health_Category'])
                                st.dataframe(styled_duration_tier_df, width='stretch', height=duration_table_height, column_config=duration_tier_column_config)
                            else:
                                st.dataframe(duration_tier_display_formatted, width='stretch', height=duration_table_height, column_config=duration_tier_column_config)
                        except (KeyError, ValueError) as e:
                            # Fallback to unstyled dataframe if styling fails
                            st.dataframe(duration_tier_display_formatted, width='stretch', height=duration_table_height, column_config=duration_tier_column_config)

                        # Add tier ranges information
                        st.markdown("---")
                        st.markdown(f'<p style="color: black; margin-bottom: 5px;"><strong>Duration Tier Definitions:</strong></p>', unsafe_allow_html=True)

                        # Display tier ranges
                        duration_tier_config = st.session_state.config_dict.get('controls', {}).get('duration_tier_config', {})
                        duration_tier_names = duration_tier_config.get('tier_names', ['Short', 'Medium', 'Long', 'Extra Long'])
                        duration_cutoff_points = duration_tier_config.get('cutoff_points', [6, 12, 24])

                        if duration_cutoff_points and len(duration_cutoff_points) >= 3:
                            st.markdown(f'<p style="color: black; font-size: 14px;">{duration_tier_names[0]}: &lt; {duration_cutoff_points[0]} mo  |  {duration_tier_names[1]}: {duration_cutoff_points[0]}-{duration_cutoff_points[1]} mo  |  {duration_tier_names[2]}: {duration_cutoff_points[1]}-{duration_cutoff_points[2]} mo  |  {duration_tier_names[3]}: ≥ {duration_cutoff_points[2]} mo</p>', unsafe_allow_html=True)

                    else:
                        st.info("No duration tier data available with current filters.")

                except Exception as e:
                    st.error(f"Error processing duration tier data: {str(e)}")
                    st.info("Unable to display duration tier consolidation with current data.")
            else:
                if len(filtered_df) == 0:
                    st.info("No projects available for duration tier consolidation.")
                elif 'original_duration_months' not in filtered_df.columns:
                    st.info("Duration data not available. Run batch EVM calculation first to generate duration metrics.")
                else:
                    st.info("Duration tier data not available. Ensure duration tiers are configured in File Management.")

        # Budget Tiers × Duration Tiers Cross-Tab Expander
        with st.expander("🎯⏱️ Budget Tiers × Duration Tiers Cross-Tab", expanded=False):
            # Prepare data for cross-tab (make a copy first to avoid modifying filtered_df)
            crosstab_df = filtered_df.copy()

            # Check if both Budget_Category and Duration_Category exist
            has_budget_tiers = 'Budget_Category' in crosstab_df.columns
            has_duration_tiers = 'Duration_Category' in crosstab_df.columns

            # If Duration_Category doesn't exist but original_duration_months does, create it
            if not has_duration_tiers and 'original_duration_months' in crosstab_df.columns:
                duration_tier_config = st.session_state.config_dict.get('controls', {}).get('duration_tier_config', {})
                default_duration_tier_config = {
                    'cutoff_points': [6, 12, 24],
                    'tier_names': ['Short', 'Medium', 'Long', 'Extra Long']
                }
                cutoffs = duration_tier_config.get('cutoff_points', default_duration_tier_config['cutoff_points'])
                tier_names = duration_tier_config.get('tier_names', default_duration_tier_config['tier_names'])

                def get_duration_tier(duration):
                    if pd.isna(duration):
                        return "Unknown"
                    duration = int(duration)
                    # Use >= logic like budget tiers (check from high to low)
                    if duration >= cutoffs[2]:
                        return tier_names[3]
                    elif duration >= cutoffs[1]:
                        return tier_names[2]
                    elif duration >= cutoffs[0]:
                        return tier_names[1]
                    else:
                        return tier_names[0]

                crosstab_df['Duration_Category'] = crosstab_df['original_duration_months'].apply(get_duration_tier)
                has_duration_tiers = True

            if has_budget_tiers and has_duration_tiers and len(crosstab_df) > 0:
                try:
                    # Metric selector
                    st.markdown("**Select Metric to Display:**")
                    metric_options = {
                        'Count (Number of Projects)': 'COUNT',
                        'BAC (Budget at Completion)': 'BAC',
                        'AC (Actual Cost)': 'AC',
                        'PV (Planned Value)': 'PV',
                        'ETC (Estimate to Complete)': 'ETC',
                        'EAC (Estimate at Completion)': 'EAC',
                        'CPI (Cost Performance Index)': 'CPI',
                        'SPI (Schedule Performance Index)': 'SPI'
                    }

                    selected_metric_label = st.selectbox(
                        "Metric",
                        options=list(metric_options.keys()),
                        key="crosstab_metric_selector",
                        label_visibility="collapsed"
                    )

                    selected_metric = metric_options[selected_metric_label]

                    # Calculate ETC if needed
                    if selected_metric == 'ETC' and 'ETC' not in crosstab_df.columns:
                        if 'EAC' in crosstab_df.columns and 'Actual Cost' in crosstab_df.columns:
                            crosstab_df['ETC'] = crosstab_df['EAC'] - crosstab_df['Actual Cost']
                        elif 'EAC' in crosstab_df.columns and 'AC' in crosstab_df.columns:
                            crosstab_df['ETC'] = crosstab_df['EAC'] - crosstab_df['AC']
                        else:
                            crosstab_df['ETC'] = 0

                    # Handle COUNT metric
                    if selected_metric == 'COUNT':
                        # Count projects in each Budget × Duration category
                        crosstab_result = pd.crosstab(
                            crosstab_df['Budget_Category'],
                            crosstab_df['Duration_Category']
                        )
                        is_performance_index = False
                        is_count_metric = True
                    else:
                        is_count_metric = False

                        # Map metric to column name
                        metric_column_map = {
                            'BAC': 'Budget',
                            'AC': 'Actual Cost',
                            'PV': 'Plan Value',
                            'ETC': 'ETC',
                            'EAC': 'EAC',
                            'CPI': 'CPI',
                            'SPI': 'SPI'
                        }

                        metric_column = metric_column_map[selected_metric]

                        # Check if metric column exists
                        if metric_column not in crosstab_df.columns:
                            st.warning(f"{selected_metric_label} data is not available in the current dataset.")
                            metric_column = None
                        else:
                            metric_column = metric_column

                    # Only proceed if we have COUNT or a valid metric column
                    if selected_metric == 'COUNT' or (selected_metric != 'COUNT' and metric_column is not None and metric_column in crosstab_df.columns):
                        if selected_metric != 'COUNT':
                            # Determine aggregation method
                            is_performance_index = selected_metric in ['CPI', 'SPI']

                            if is_performance_index:
                                # Calculate weighted average by Budget for performance indices
                                def weighted_avg(group):
                                    if group['Budget'].sum() > 0:
                                        return (group[metric_column] * group['Budget']).sum() / group['Budget'].sum()
                                    else:
                                        return 0

                                crosstab_result = crosstab_df.groupby(['Budget_Category', 'Duration_Category']).apply(weighted_avg).unstack(fill_value=0)
                            else:
                                # Sum for monetary values
                                crosstab_result = pd.pivot_table(
                                    crosstab_df,
                                    values=metric_column,
                                    index='Budget_Category',
                                    columns='Duration_Category',
                                    aggfunc='sum',
                                    fill_value=0
                                )

                        # Get tier configurations
                        budget_tier_config = st.session_state.config_dict.get('controls', {}).get('tier_config', {})
                        budget_tier_names = budget_tier_config.get('tier_names', ['Tier 1', 'Tier 2', 'Tier 3', 'Tier 4'])

                        duration_tier_config = st.session_state.config_dict.get('controls', {}).get('duration_tier_config', {})
                        duration_tier_names = duration_tier_config.get('tier_names', ['Short', 'Medium', 'Long', 'Extra Long'])

                        # Reorder rows (Budget Tiers: descending from Tier 4 to Tier 1)
                        budget_order = list(reversed(budget_tier_names))
                        crosstab_result = crosstab_result.reindex([t for t in budget_order if t in crosstab_result.index])

                        # Reorder columns (Duration Tiers: descending from Extra Long to Short)
                        duration_order = list(reversed(duration_tier_names))
                        crosstab_result = crosstab_result.reindex(columns=[t for t in duration_order if t in crosstab_result.columns])

                        # Add row and column totals
                        crosstab_result['Total'] = crosstab_result.sum(axis=1)
                        crosstab_result.loc['Total'] = crosstab_result.sum(axis=0)

                        # If performance index, recalculate totals as weighted averages
                        if is_performance_index:
                            # Recalculate row totals (weighted by budget for each budget tier across all durations)
                            for budget_tier in budget_tier_names:
                                if budget_tier in crosstab_df['Budget_Category'].values:
                                    tier_data = crosstab_df[crosstab_df['Budget_Category'] == budget_tier]
                                    if tier_data['Budget'].sum() > 0:
                                        crosstab_result.loc[budget_tier, 'Total'] = (tier_data[metric_column] * tier_data['Budget']).sum() / tier_data['Budget'].sum()

                            # Recalculate column totals (weighted by budget for each duration tier across all budgets)
                            for duration_tier in duration_tier_names:
                                if duration_tier in crosstab_df['Duration_Category'].values:
                                    tier_data = crosstab_df[crosstab_df['Duration_Category'] == duration_tier]
                                    if tier_data['Budget'].sum() > 0:
                                        crosstab_result.loc['Total', duration_tier] = (tier_data[metric_column] * tier_data['Budget']).sum() / tier_data['Budget'].sum()

                            # Recalculate grand total (overall weighted average)
                            if crosstab_df['Budget'].sum() > 0:
                                crosstab_result.loc['Total', 'Total'] = (crosstab_df[metric_column] * crosstab_df['Budget']).sum() / crosstab_df['Budget'].sum()

                        # Format for display
                        crosstab_display = crosstab_result.copy()
                        crosstab_display = crosstab_display.reset_index()
                        crosstab_display = crosstab_display.rename(columns={'Budget_Category': 'Budget Tier'})

                        # Format monetary values with thousands separators as strings
                        if not is_count_metric and not is_performance_index:
                            # Apply formatting to all columns except 'Budget Tier'
                            for col in crosstab_display.columns:
                                if col != 'Budget Tier':
                                    crosstab_display[col] = crosstab_display[col].apply(lambda x: f"{x:,.2f}" if pd.notna(x) else "")

                        # Create column configuration
                        column_config = {}

                        # Configure Budget Tier column
                        column_config['Budget Tier'] = st.column_config.TextColumn(
                            'Budget Tier',
                            help="Budget tier categories"
                        )

                        # Configure data columns based on metric type
                        if is_count_metric:
                            # Count metric: integer format
                            for col in crosstab_display.columns:
                                if col != 'Budget Tier':
                                    column_config[col] = st.column_config.NumberColumn(
                                        col,
                                        format="%d",
                                        help="Number of projects"
                                    )
                        elif is_performance_index:
                            # Performance indices: 3 decimal places
                            for col in crosstab_display.columns:
                                if col != 'Budget Tier':
                                    column_config[col] = st.column_config.NumberColumn(
                                        col,
                                        format="%.3f",
                                        help=f"{selected_metric} value"
                                    )
                        else:
                            # Monetary values: displayed as formatted text (already formatted above)
                            for col in crosstab_display.columns:
                                if col != 'Budget Tier':
                                    column_config[col] = st.column_config.TextColumn(
                                        col,
                                        help=f"Values in {currency_symbol}{currency_postfix}" if currency_postfix else f"Values in {currency_symbol}",
                                        width="medium"
                                    )

                        # Display the cross-tab
                        st.markdown(f"### 📊 {selected_metric_label} Cross-Tab")
                        st.caption("Rows: Budget Tiers | Columns: Duration Tiers")

                        # Calculate table height dynamically
                        num_rows = len(crosstab_display)
                        table_height = min((num_rows + 1) * 35 + 3, 300)

                        st.dataframe(
                            crosstab_display,
                            width='stretch',
                            height=table_height,
                            column_config=column_config,
                            hide_index=True
                        )

                        # Add tier definitions
                        st.markdown("---")
                        col1, col2 = st.columns(2)

                        with col1:
                            st.markdown('<p style="color: black; margin-bottom: 5px;"><strong>Budget Tier Definitions:</strong></p>', unsafe_allow_html=True)
                            budget_cutoff_points = budget_tier_config.get('cutoff_points', [4000, 8000, 15000])
                            if budget_cutoff_points and len(budget_cutoff_points) >= 3:
                                st.markdown(f'<p style="color: black; font-size: 13px;">{budget_tier_names[0]}: &lt; {currency_symbol}{budget_cutoff_points[0]:,.0f}<br>{budget_tier_names[1]}: {currency_symbol}{budget_cutoff_points[0]:,.0f}-{currency_symbol}{budget_cutoff_points[1]:,.0f}<br>{budget_tier_names[2]}: {currency_symbol}{budget_cutoff_points[1]:,.0f}-{currency_symbol}{budget_cutoff_points[2]:,.0f}<br>{budget_tier_names[3]}: ≥ {currency_symbol}{budget_cutoff_points[2]:,.0f}</p>', unsafe_allow_html=True)

                        with col2:
                            st.markdown('<p style="color: black; margin-bottom: 5px;"><strong>Duration Tier Definitions:</strong></p>', unsafe_allow_html=True)
                            duration_cutoff_points = duration_tier_config.get('cutoff_points', [6, 12, 24])
                            if duration_cutoff_points and len(duration_cutoff_points) >= 3:
                                st.markdown(f'<p style="color: black; font-size: 13px;">{duration_tier_names[0]}: &lt; {duration_cutoff_points[0]} mo<br>{duration_tier_names[1]}: {duration_cutoff_points[0]}-{duration_cutoff_points[1]} mo<br>{duration_tier_names[2]}: {duration_cutoff_points[1]}-{duration_cutoff_points[2]} mo<br>{duration_tier_names[3]}: ≥ {duration_cutoff_points[2]} mo</p>', unsafe_allow_html=True)

                except Exception as e:
                    st.error(f"Error creating cross-tab: {str(e)}")
                    st.info("Unable to display cross-tab with current data.")
            else:
                if len(crosstab_df) == 0:
                    st.info("No projects available for cross-tab analysis.")
                elif not has_budget_tiers:
                    st.info("Budget tier data not available. Ensure budget tiers are configured in File Management.")
                elif not has_duration_tiers:
                    st.info("Duration tier data not available. Ensure duration tiers are configured in File Management or run batch EVM calculation.")

        # Portfolio Budget Chart Expander
        # Portfolio Budget Chart has been moved to 5_Portfolio_Charts.py

        # Cash Flow Chart has been moved to 5_Portfolio_Charts.py

        # Approvals Chart has been moved to 5_Portfolio_Charts.py

        # Financial Summary Expander
        with st.expander("💰 Financial Summary", expanded=False):
            if len(filtered_df) > 0:
                # Enhanced styling for Financial Summary
                st.markdown("""
                <style>
                .financial-metric {
                    font-size: 1.1rem;
                    line-height: 1.4;
                    margin-bottom: 15px;
                    padding: 10px;
                    border-radius: 8px;
                    background-color: #f8f9fa;
                    border-left: 4px solid #007bff;
                }
                .financial-value {
                    font-size: 1.3rem;
                    font-weight: bold;
                    color: #1f77b4;
                    margin-top: 8px;
                }
                </style>
                """, unsafe_allow_html=True)

                # Calculate portfolio totals with safe column access
                total_budget = filtered_df.get('Budget', pd.Series([0])).sum()
                total_actual_cost = filtered_df.get('Actual Cost', pd.Series([0])).sum()
                total_earned_value = filtered_df.get('Earned Value', pd.Series([0])).sum()
                total_planned_value = filtered_df.get('Plan Value', pd.Series([0])).sum()
                total_eac = filtered_df.get('EAC', pd.Series([0])).sum()

                # Calculate ETC as EAC - AC
                total_etc = total_eac - total_actual_cost

                # Row 1: Budget (BAC) and Actual Cost (AC)
                col1, col2 = st.columns(2)
                with col1:
                    if 'Budget' in filtered_df.columns and total_budget > 0:
                        bac_formatted = format_currency(total_budget, currency_symbol, currency_postfix, thousands=False)
                        st.markdown(f'<div class="financial-metric">💰 **Portfolio Budget (BAC)**<br><span class="financial-value">{bac_formatted}</span></div>', unsafe_allow_html=True)
                    else:
                        st.markdown(f'<div class="financial-metric">💰 **Portfolio Budget (BAC)**<br><span class="financial-value">Not Available</span></div>', unsafe_allow_html=True)
                with col2:
                    if 'Actual Cost' in filtered_df.columns:
                        ac_formatted = format_currency(total_actual_cost, currency_symbol, currency_postfix, thousands=False)
                        st.markdown(f'<div class="financial-metric">💸 **Portfolio Actual Cost (AC)**<br><span class="financial-value">{ac_formatted}</span></div>', unsafe_allow_html=True)
                    else:
                        st.markdown(f'<div class="financial-metric">💸 **Portfolio Actual Cost (AC)**<br><span class="financial-value">Not Available</span></div>', unsafe_allow_html=True)

                st.markdown("<br>", unsafe_allow_html=True)

                # Row 2: Planned Value (PV) and Earned Value (EV)
                col3, col4 = st.columns(2)
                with col3:
                    if 'Plan Value' in filtered_df.columns:
                        pv_formatted = format_currency(total_planned_value, currency_symbol, currency_postfix, thousands=False)
                        st.markdown(f'<div class="financial-metric">📊 **Portfolio Planned Value (PV)**<br><span class="financial-value">{pv_formatted}</span></div>', unsafe_allow_html=True)
                    else:
                        st.markdown(f'<div class="financial-metric">📊 **Portfolio Planned Value (PV)**<br><span class="financial-value">Not Available</span></div>', unsafe_allow_html=True)
                with col4:
                    if 'Earned Value' in filtered_df.columns:
                        ev_formatted = format_currency(total_earned_value, currency_symbol, currency_postfix, thousands=False)
                        st.markdown(f'<div class="financial-metric">💎 **Portfolio Earned Value (EV)**<br><span class="financial-value">{ev_formatted}</span></div>', unsafe_allow_html=True)
                    else:
                        st.markdown(f'<div class="financial-metric">💎 **Portfolio Earned Value (EV)**<br><span class="financial-value">Not Available</span></div>', unsafe_allow_html=True)

                st.markdown("<br>", unsafe_allow_html=True)

                # Row 3: ETC and EAC
                col5, col6 = st.columns(2)
                with col5:
                    if 'EAC' in filtered_df.columns and 'Actual Cost' in filtered_df.columns:
                        etc_formatted = format_currency(total_etc, currency_symbol, currency_postfix, thousands=False)
                        st.markdown(f'<div class="financial-metric">🔧 **Portfolio Estimate to Complete (ETC)**<br><span class="financial-value">{etc_formatted}</span></div>', unsafe_allow_html=True)
                    else:
                        st.markdown(f'<div class="financial-metric">🔧 **Portfolio Estimate to Complete (ETC)**<br><span class="financial-value">Not Available</span></div>', unsafe_allow_html=True)
                with col6:
                    if 'EAC' in filtered_df.columns:
                        eac_formatted = format_currency(total_eac, currency_symbol, currency_postfix, thousands=False)
                        st.markdown(f'<div class="financial-metric">🎯 **Portfolio Estimate at Completion (EAC)**<br><span class="financial-value">{eac_formatted}</span></div>', unsafe_allow_html=True)
                    else:
                        st.markdown(f'<div class="financial-metric">🎯 **Portfolio Estimate at Completion (EAC)**<br><span class="financial-value">Not Available</span></div>', unsafe_allow_html=True)

                st.markdown("<br>", unsafe_allow_html=True)

                # Portfolio EAC Status Alert
                forecast_overrun = total_eac - total_budget
                if forecast_overrun > 0:
                    eac_fmt = format_currency(total_eac, currency_symbol, currency_postfix, thousands=False)
                    overrun_fmt = format_currency(forecast_overrun, currency_symbol, currency_postfix, thousands=False)
                    st.markdown(
                        f"""<div style="background-color: #fee; border-left: 4px solid #c33; padding: 1rem; border-radius: 4px; margin: 1rem 0; font-family: sans-serif; letter-spacing: normal; word-spacing: normal;">
                        📢 <strong>Portfolio EAC Status:</strong> {eac_fmt} (+{overrun_fmt} over budget)
                        </div>""",
                        unsafe_allow_html=True
                    )
                else:
                    eac_fmt = format_currency(total_eac, currency_symbol, currency_postfix, thousands=False)
                    st.markdown(
                        f"""<div style="background-color: #efe; border-left: 4px solid #3c3; padding: 1rem; border-radius: 4px; margin: 1rem 0; font-family: sans-serif; letter-spacing: normal; word-spacing: normal;">
                        ✅ <strong>Portfolio EAC Status:</strong> {eac_fmt} (Under budget)
                        </div>""",
                        unsafe_allow_html=True
                    )

            else:
                st.info("No data available for financial summary.")

        # Durations Expander
        with st.expander("⏱️ Durations", expanded=False):
            if len(filtered_df) > 0:
                # Enhanced styling for Duration metrics (same as Financial Summary)
                st.markdown("""
                <style>
                .duration-metric {
                    font-size: 1.1rem;
                    line-height: 1.4;
                    margin-bottom: 15px;
                    padding: 10px;
                    border-radius: 8px;
                    background-color: #f8f9fa;
                    border-left: 4px solid #28a745;
                }
                .duration-value {
                    font-size: 1.3rem;
                    font-weight: bold;
                    color: #28a745;
                    margin-top: 8px;
                }
                </style>
                """, unsafe_allow_html=True)

                # Check for duration columns and calculate metrics
                od_col = 'original_duration_months'
                ad_col = 'actual_duration_months'
                ld_col = 'likely_duration'  # Fixed: database returns 'likely_duration', not 'ld'
                bac_col = 'bac' if 'bac' in filtered_df.columns else 'Budget'

                # Initialize variables
                avg_od = wt_avg_od = avg_ad = wt_avg_ad = avg_ld = wt_avg_ld = None
                avg_delay = wt_avg_delay = None

                # Calculate OD metrics
                if od_col in filtered_df.columns:
                    od_data = filtered_df[filtered_df[od_col].notna() & (filtered_df[od_col] > 0)]
                    if len(od_data) > 0:
                        avg_od = od_data[od_col].mean()
                        if bac_col in od_data.columns:
                            total_bac = od_data[bac_col].sum()
                            if total_bac > 0:
                                wt_avg_od = (od_data[od_col] * od_data[bac_col]).sum() / total_bac

                # Calculate AD metrics
                if ad_col in filtered_df.columns:
                    ad_data = filtered_df[filtered_df[ad_col].notna() & (filtered_df[ad_col] > 0)]
                    if len(ad_data) > 0:
                        avg_ad = ad_data[ad_col].mean()
                        if bac_col in ad_data.columns:
                            total_bac = ad_data[bac_col].sum()
                            if total_bac > 0:
                                wt_avg_ad = (ad_data[ad_col] * ad_data[bac_col]).sum() / total_bac

                # Calculate LD metrics with upper limit check
                if ld_col in filtered_df.columns:
                    ld_data = filtered_df[filtered_df[ld_col].notna() & (filtered_df[ld_col] > 0)].copy()
                    if len(ld_data) > 0:
                        # Apply upper limit cap: min(LD, OD + 48)
                        if od_col in ld_data.columns:
                            # For projects with OD data, cap LD to OD + 48
                            ld_data_with_od = ld_data[ld_data[od_col].notna() & (ld_data[od_col] > 0)].copy()
                            if len(ld_data_with_od) > 0:
                                ld_data_with_od[ld_col + '_capped'] = ld_data_with_od.apply(
                                    lambda row: min(row[ld_col], row[od_col] + 48), axis=1
                                )
                                # Use capped values for calculation
                                avg_ld = ld_data_with_od[ld_col + '_capped'].mean()
                                if bac_col in ld_data_with_od.columns:
                                    total_bac = ld_data_with_od[bac_col].sum()
                                    if total_bac > 0:
                                        wt_avg_ld = (ld_data_with_od[ld_col + '_capped'] * ld_data_with_od[bac_col]).sum() / total_bac
                            else:
                                # No OD data available, use original LD values
                                avg_ld = ld_data[ld_col].mean()
                                if bac_col in ld_data.columns:
                                    total_bac = ld_data[bac_col].sum()
                                    if total_bac > 0:
                                        wt_avg_ld = (ld_data[ld_col] * ld_data[bac_col]).sum() / total_bac
                        else:
                            # No OD column available, use original LD values
                            avg_ld = ld_data[ld_col].mean()
                            if bac_col in ld_data.columns:
                                total_bac = ld_data[bac_col].sum()
                                if total_bac > 0:
                                    wt_avg_ld = (ld_data[ld_col] * ld_data[bac_col]).sum() / total_bac

                # Calculate Delay metrics
                if avg_ld is not None and avg_od is not None:
                    avg_delay = avg_ld - avg_od
                if wt_avg_ld is not None and wt_avg_od is not None:
                    wt_avg_delay = wt_avg_ld - wt_avg_od

                # Row 1: Plan Duration (OD)
                col1, col2 = st.columns(2)
                with col1:
                    if avg_od is not None:
                        st.markdown(f'<div class="duration-metric">📅 **Avg Plan Duration (OD)**<br><span class="duration-value">{round(avg_od)} months</span></div>', unsafe_allow_html=True)
                    else:
                        st.markdown(f'<div class="duration-metric">📅 **Avg Plan Duration (OD)**<br><span class="duration-value">Not Available</span></div>', unsafe_allow_html=True)
                with col2:
                    if wt_avg_od is not None:
                        st.markdown(f'<div class="duration-metric">⚖️ **Wt Avg Plan Duration (wt OD)**<br><span class="duration-value">{round(wt_avg_od)} months</span></div>', unsafe_allow_html=True)
                    else:
                        st.markdown(f'<div class="duration-metric">⚖️ **Wt Avg Plan Duration (wt OD)**<br><span class="duration-value">Not Available</span></div>', unsafe_allow_html=True)

                st.markdown("<br>", unsafe_allow_html=True)

                # Row 2: Actual Duration (AD)
                col3, col4 = st.columns(2)
                with col3:
                    if avg_ad is not None:
                        st.markdown(f'<div class="duration-metric">📊 **Avg Actual Duration (AD)**<br><span class="duration-value">{round(avg_ad)} months</span></div>', unsafe_allow_html=True)
                    else:
                        st.markdown(f'<div class="duration-metric">📊 **Avg Actual Duration (AD)**<br><span class="duration-value">Not Available</span></div>', unsafe_allow_html=True)
                with col4:
                    if wt_avg_ad is not None:
                        st.markdown(f'<div class="duration-metric">⚖️ **Wt Avg Actual Duration (Wt AD)**<br><span class="duration-value">{round(wt_avg_ad)} months</span></div>', unsafe_allow_html=True)
                    else:
                        st.markdown(f'<div class="duration-metric">⚖️ **Wt Avg Actual Duration (Wt AD)**<br><span class="duration-value">Not Available</span></div>', unsafe_allow_html=True)

                st.markdown("<br>", unsafe_allow_html=True)

                # Row 3: Likely Duration (LD)
                col5, col6 = st.columns(2)
                with col5:
                    if avg_ld is not None:
                        st.markdown(f'<div class="duration-metric">🔮 **Avg Likely Duration (LD)**<br><span class="duration-value">{round(avg_ld)} months</span></div>', unsafe_allow_html=True)
                    else:
                        st.markdown(f'<div class="duration-metric">🔮 **Avg Likely Duration (LD)**<br><span class="duration-value">Not Available</span></div>', unsafe_allow_html=True)
                with col6:
                    if wt_avg_ld is not None:
                        st.markdown(f'<div class="duration-metric">⚖️ **Wt Avg Likely Duration (wt LD)**<br><span class="duration-value">{round(wt_avg_ld)} months</span></div>', unsafe_allow_html=True)
                    else:
                        st.markdown(f'<div class="duration-metric">⚖️ **Wt Avg Likely Duration (wt LD)**<br><span class="duration-value">Not Available</span></div>', unsafe_allow_html=True)

                st.markdown("<br>", unsafe_allow_html=True)

                # Row 4: Delay
                col7, col8 = st.columns(2)
                with col7:
                    if avg_delay is not None:
                        delay_color = "#dc3545" if avg_delay > 0 else "#28a745"
                        delay_sign = "+" if avg_delay > 0 else ""
                        st.markdown(f'<div class="duration-metric">⏰ **Avg Delay (Avg LD - Avg OD)**<br><span class="duration-value" style="color: {delay_color};">{delay_sign}{round(avg_delay)} months</span></div>', unsafe_allow_html=True)
                    else:
                        st.markdown(f'<div class="duration-metric">⏰ **Avg Delay (Avg LD - Avg OD)**<br><span class="duration-value">Not Available</span></div>', unsafe_allow_html=True)
                with col8:
                    if wt_avg_delay is not None:
                        delay_color = "#dc3545" if wt_avg_delay > 0 else "#28a745"
                        delay_sign = "+" if wt_avg_delay > 0 else ""
                        st.markdown(f'<div class="duration-metric">⚖️ **Wt Avg Delay (wt Avg LD - wt Avg OD)**<br><span class="duration-value" style="color: {delay_color};">{delay_sign}{round(wt_avg_delay)} months</span></div>', unsafe_allow_html=True)
                    else:
                        st.markdown(f'<div class="duration-metric">⚖️ **Wt Avg Delay (wt Avg LD - wt Avg OD)**<br><span class="duration-value">Not Available</span></div>', unsafe_allow_html=True)

            else:
                st.info("No data available for duration analysis.")

        # Time/Budget Performance Chart
        # Time/Budget Performance has been moved to 5_Portfolio_Charts.py

        # Portfolio Treemap
        # Portfolio Treemap has been moved to 5_Portfolio_Charts.py

    # Helper function for formatting project details in prompts
    def format_project_details_for_prompt(project_details, currency='$', postfix=''):
        lines = []
        for i, p in enumerate(project_details, 1):
            budget_fmt = f"{currency}{p['budget']:,.0f}{' ' + postfix if postfix else ''}"
            ac_fmt = f"{currency}{p['actual_cost']:,.0f}{' ' + postfix if postfix else ''}"
            lines.append(
                f"{i}. **{p['name']}** ({p['organization']})\n"
                f"   - Start: {p['start_date']} | Finish: {p['finish_date']}\n"
                f"   - Plan Duration: {p['plan_duration']:.1f} months | Likely Duration: {p['likely_duration']:.1f} months\n"
                f"   - Budget (BAC): {budget_fmt} | % Budget Used: {p['percent_budget_used']:.1f}%\n"
                f"   - Actual Duration: {p['actual_duration']:.1f} months | % Time Used: {p['percent_time_used']:.1f}%\n"
                f"   - SPI: {p['spi']:.2f} | CPI: {p['cpi']:.2f}\n"
                f"   - Actual Cost: {ac_fmt}"
            )
        return '\n'.join(lines)

    # Executive Portfolio Brief Section (Enhanced with Project Details)
    with st.expander("📊 Executive Portfolio Brief (AI-Generated)", expanded=False):
        st.markdown("""
        ### 🤖 AI-Powered Portfolio Executive Report

        Generate a comprehensive executive briefing of your portfolio health using AI with deep analysis of hidden trends.

        **Advanced Analysis includes:**
        - Portfolio dashboard with key metrics
        - Executive summary with overall health rating
        - Performance analysis (cost and schedule)
        - **Trend Analysis**: Patterns by organization, project size, tier, and timeline
        - **Root Cause Analysis**: Identify systemic issues affecting performance
        - Risk assessment across all dimensions
        - Strategic recommendations and corrective actions
        - Decision points requiring executive approval
        """)

        # Check if LLM is configured
        llm_config = st.session_state.config_dict.get('llm_config', {})

        # Generate button (requires API key)
        has_api_key = bool(llm_config.get('api_key', '').strip()) if llm_config else False
        show_generate_button = has_api_key and safe_llm_request

        if not has_api_key:
            st.warning("⚠️ API key not configured. Please set it in Portfolio Management > LLM Provider Configuration to enable AI-generated reports.")
        elif not safe_llm_request:
            st.error("❌ Required functions not available. Please check imports.")

        # Two separate buttons
        generate_clicked = False
        get_prompt_clicked = False

        if show_generate_button:
            generate_clicked = st.button("🚀 Generate Executive Portfolio Brief", type="primary", key="gen_portfolio_brief")

        get_prompt_clicked = st.button("📋 Get Prompt", key="get_prompt_brief", help="Get the prompt to paste into ChatGPT, Claude, or any LLM")

        # Handle Generate button
        if show_generate_button and generate_clicked:
                with st.spinner("Analyzing portfolio data and generating comprehensive brief..."):
                    try:
                        # Calculate quadrants with correct key names
                        on_budget_on_schedule = len(df[(df['CPI'] >= 0.95) & (df['SPI'] >= 0.95)]) if 'CPI' in df.columns and 'SPI' in df.columns else 0
                        over_budget_on_schedule = len(df[(df['CPI'] < 0.95) & (df['SPI'] >= 0.95)]) if 'CPI' in df.columns and 'SPI' in df.columns else 0
                        on_budget_behind_schedule = len(df[(df['CPI'] >= 0.95) & (df['SPI'] < 0.95)]) if 'CPI' in df.columns and 'SPI' in df.columns else 0
                        over_budget_behind_schedule = len(df[(df['CPI'] < 0.95) & (df['SPI'] < 0.95)]) if 'CPI' in df.columns and 'SPI' in df.columns else 0

                        # Get controls for currency formatting
                        controls = st.session_state.config_dict.get('controls', {
                            'currency_symbol': '$',
                            'currency_postfix': ''
                        })

                        currency = controls.get('currency_symbol', '$')
                        postfix = controls.get('currency_postfix', '')

                        # Prepare detailed project data for trend analysis
                        project_details = []
                        for _, row in df.iterrows():
                            budget_val = row.get('Budget', row.get('BAC', 0)) or 0
                            ac_val = row.get('Actual Cost', row.get('AC', 0)) or 0
                            od_val = row.get('Original Duration', 0) or 0
                            ld_val = row.get('Forecast Duration', 0) or 0
                            ad_val = row.get('Actual Duration', 0) or 0
                            project_info = {
                                'name': row.get('Project', row.get('Project Name', 'Unknown')),
                                'organization': row.get('Organization', row.get('Org', 'Unknown')),
                                'start_date': str(row['Plan Start'])[:10] if 'Plan Start' in row and pd.notna(row.get('Plan Start')) else 'N/A',
                                'finish_date': str(row['Plan Finish'])[:10] if 'Plan Finish' in row and pd.notna(row.get('Plan Finish')) else 'N/A',
                                'plan_duration': float(od_val),
                                'likely_duration': float(ld_val),
                                'budget': float(budget_val),
                                'percent_budget_used': (float(ac_val) / float(budget_val) * 100) if budget_val else 0,
                                'actual_duration': float(ad_val),
                                'percent_time_used': (float(ad_val) / float(od_val) * 100) if od_val else 0,
                                'spi': row.get('SPI', 0) or 0,
                                'cpi': row.get('CPI', 0) or 0,
                                'actual_cost': float(ac_val),
                                'progress': row.get('Completion %', 0) or 0,
                                'tier': row.get('Budget_Category', 'Unknown'),
                                'health': row.get('Health_Category', 'Unknown'),
                            }

                            project_details.append(project_info)

                        # Analyze trends by organization
                        org_analysis = df.groupby('Organization').agg({
                            'CPI': 'mean',
                            'SPI': 'mean',
                            'Budget': 'sum'
                        }).round(2).to_dict('index') if 'Organization' in df.columns else {}

                        # Analyze trends by tier
                        tier_analysis = df.groupby('Budget_Category').agg({
                            'CPI': 'mean',
                            'SPI': 'mean',
                            'Budget': 'sum'
                        }).round(2).to_dict('index') if 'Budget_Category' in df.columns else {}

                        # Create enhanced prompt with detailed project data
                        prompt = f"""
You are a seasoned Chief Portfolio Officer preparing an executive briefing for the C-Suite. Create a comprehensive, narrative-style report that tells the story of this portfolio's performance with actionable insights.

Write in a professional yet conversational tone that executives can quickly scan but also dive deep into. Use clear headings, bullet points, and specific numbers to support your analysis.

**IMPORTANT CONTEXT:**
- All projects in this portfolio are currently in their execution stage (actively being worked on)
- AD (Actual Duration) represents the time already spent on each project, NOT the total project duration
- Projects are ongoing and not yet completed

---

# PORTFOLIO EXECUTIVE BRIEFING
**Report Date:** {datetime.now().strftime('%B %d, %Y')}

## DATA SNAPSHOT

**Portfolio Scale:**
- {metrics['total_projects']} active projects under management
- Total Budget at Completion (BAC): {currency}{metrics['total_budget']:,.0f}{' ' + postfix if postfix else ''}
- Current Actual Cost (AC): {currency}{metrics['total_actual_cost']:,.0f}{' ' + postfix if postfix else ''}
- Earned Value (EV): {currency}{metrics['total_earned_value']:,.0f}{' ' + postfix if postfix else ''}
- Average Portfolio Progress: {df['Completion %'].mean() if 'Completion %' in df.columns else 0:.1f}%

**Key Performance Indicators:**
- Portfolio CPI: {metrics['portfolio_cpi']:.2f} (Every dollar spent delivers {currency}{metrics['portfolio_cpi']:.2f} of value)
- Portfolio TCPI: {metrics['portfolio_tcpi']:.2f} (Cost performance needed to meet budget: {('must improve' if metrics['portfolio_tcpi'] > 1 else 'on track')})
- Portfolio SPI: {metrics['portfolio_spi']:.2f} (Portfolio is {'ahead of' if metrics['portfolio_spi'] > 1 else 'behind'} schedule)
- Forecast at Completion (EAC): {currency}{metrics['total_eac']:,.0f}{' ' + postfix if postfix else ''}
- Projected Overrun: {currency}{metrics['forecast_overrun']:,.0f}{' ' + postfix if postfix else ''} ({'over' if metrics['forecast_overrun'] > 0 else 'under'} budget)

**Portfolio Health Status:**
- 🟢 Healthy Projects: {metrics['healthy_projects']} ({(metrics['healthy_projects']/metrics['total_projects'])*100:.0f}%)
- 🟡 At Risk Projects: {metrics['at_risk_projects']} ({(metrics['at_risk_projects']/metrics['total_projects'])*100:.0f}%)
- 🔴 Critical Projects: {metrics['critical_projects']} ({(metrics['critical_projects']/metrics['total_projects'])*100:.0f}%)

**Performance Quadrant Distribution:**
- ✅ On Budget & On Schedule: {on_budget_on_schedule} projects ({(on_budget_on_schedule/metrics['total_projects'])*100:.0f}%)
- ⚠️ Over Budget but On Schedule: {over_budget_on_schedule} projects ({(over_budget_on_schedule/metrics['total_projects'])*100:.0f}%)
- ⏰ On Budget but Behind Schedule: {on_budget_behind_schedule} projects ({(on_budget_behind_schedule/metrics['total_projects'])*100:.0f}%)
- 🚨 Over Budget & Behind Schedule: {over_budget_behind_schedule} projects ({(over_budget_behind_schedule/metrics['total_projects'])*100:.0f}%)

**Performance by Organization:**
{chr(10).join([f"- **{org}**: CPI {data['CPI']:.2f}, SPI {data['SPI']:.2f}, Managing {currency}{data['Budget']:,.0f}{' ' + postfix if postfix else ''} ({(data['Budget']/metrics['total_budget'])*100:.0f}% of portfolio)" for org, data in sorted(org_analysis.items(), key=lambda x: x[1]['Budget'], reverse=True)]) if org_analysis else "Organization data not available"}

**Performance by Project Size (Tier):**
{chr(10).join([f"- **{tier}**: CPI {data['CPI']:.2f}, SPI {data['SPI']:.2f}, Total Value {currency}{data['Budget']:,.0f}{' ' + postfix if postfix else ''}" for tier, data in sorted(tier_analysis.items(), key=lambda x: x[1]['Budget'], reverse=True)]) if tier_analysis else "Tier data not available"}

**🚨 Projects Requiring Immediate Attention (Worst CPI):**
{chr(10).join([f"{i+1}. **{p['name']}** ({p['organization']}, {p['tier']})" + chr(10) + f"   - Cost Performance: CPI {p['cpi']:.2f} (Spending {currency}{(1/p['cpi'] if p['cpi'] > 0 else 0):.2f} for every {currency}1.00 of value)" + chr(10) + f"   - Schedule: SPI {p['spi']:.2f}" + chr(10) + f"   - Physical Progress: {p['progress']:.0f}%" + chr(10) + f"   - Status: {p['health']}" for i, p in enumerate(sorted(project_details, key=lambda x: x['cpi'])[:5])])}

**⏰ Projects with Significant Schedule Delays (Worst SPI):**
{chr(10).join([f"{i+1}. **{p['name']}** ({p['organization']}, {p['tier']})" + chr(10) + f"   - Schedule Performance: SPI {p['spi']:.2f} ({'Ahead' if p['spi'] > 1 else str(int((1-p['spi'])*100)) + '% behind'} schedule)" + chr(10) + f"   - Cost: CPI {p['cpi']:.2f}" + chr(10) + f"   - Progress: {p['progress']:.0f}%" + chr(10) + f"   - Status: {p['health']}" for i, p in enumerate(sorted(project_details, key=lambda x: x['spi'])[:5])])}

---

Now, based on this comprehensive data, provide your executive analysis:

## 1. EXECUTIVE SUMMARY
Write 3-5 concise bullets that capture the most important insights an executive needs to know. Each bullet should:
- State the finding clearly with specific numbers
- Explain what it means in business terms
- Indicate if action is required

Example: "Portfolio is tracking 8% over budget with EAC of $X.X billion, driven primarily by Organization Y's projects which represent 35% of overruns despite managing only 20% of portfolio value."

## 2. FINANCIAL PERFORMANCE NARRATIVE
Tell the story of the portfolio's financial health:
- How efficiently are we converting budget into value?
- Which organizations or project types are performing well vs. struggling?
- What does the CPI trend tell us about our project management capabilities?
- Are cost overruns concentrated or widespread?
- What's driving the forecast overrun/underrun?

Use specific examples and numbers. Compare performance across organizations and tiers.

## 3. SCHEDULE PERFORMANCE NARRATIVE
Tell the story of timeline adherence:
- Are we meeting our delivery commitments?
- Where are the delays concentrated? (organization, tier, project type?)
- Is poor schedule performance correlated with cost issues?
- What does this mean for our strategic timeline goals?

## 4. DEEP DIVE: PATTERNS & ROOT CAUSES
This is the most critical section. Analyze the data for hidden patterns:

**Organizational Performance:**
- Are certain organizations consistently outperforming or underperforming?
- Is this a capability issue, resource issue, or complexity issue?
- What can high-performing organizations teach struggling ones?

**Project Size/Complexity Patterns:**
- Do larger projects (higher tiers) perform differently than smaller ones?
- Is our PMO better equipped for certain project scales?
- Should we adjust governance based on tier?

**Systemic vs. Isolated Issues:**
- Are problems concentrated in a few projects or spread across the portfolio?
- Do we have systemic process issues or just a few troubled projects?
- What percentage of the budget is in healthy vs. troubled projects?

**Strategic Implications:**
- What does this performance pattern mean for our strategic objectives?
- Are we at risk of missing key business milestones?

## 5. PORTFOLIO HEALTH RATING & TRAJECTORY

Give an overall rating: **EXCELLENT** | **GOOD** | **FAIR** | **POOR** | **CRITICAL**

Explain your rating with specific rationale. Then discuss:
- Is performance improving, stable, or declining?
- What's the trajectory if current trends continue?
- How does this compare to industry benchmarks or past performance?

## 6. RISK ASSESSMENT & IMPACT QUANTIFICATION

Identify 3-5 specific risks with:
- **Risk Description:** What could go wrong?
- **Likelihood:** High/Medium/Low
- **Impact:** Quantify in dollars and timeline
- **Affected Projects/Organizations:** Be specific
- **Mitigation Status:** What's being done?

Example: "**Risk:** Organization X's consistent underperformance (CPI 0.72) may lead to $15M additional overrun by Q4. **Impact:** Would consume 40% of portfolio contingency reserve. **Mitigation:** Requires immediate intervention."

## 7. STRATEGIC RECOMMENDATIONS (Prioritized)

Provide 5-8 actionable recommendations ranked by impact. For each:
- **Recommendation:** What should be done?
- **Rationale:** Why this matters (tie to data)
- **Expected Impact:** Quantify the benefit
- **Owner:** Who should lead this?
- **Timeline:** When should this happen?
- **Success Metric:** How will we measure success?

Focus on strategic moves, not tactical project management. Think:
- Organizational restructuring
- Resource reallocation
- Process improvements
- Governance changes
- Portfolio rebalancing

## 8. IMMEDIATE ACTION PLAN (Next 60 Days)

Create a prioritized action list with:
- **Action Item:** Specific, clear directive
- **Owner:** Organization or role responsible
- **Deadline:** Specific date or timeframe
- **Success Criteria:** How we'll know it's done
- **Dependencies:** What's needed to execute

Focus on actions that will have the highest impact on portfolio health.

## 9. EXECUTIVE DECISION POINTS

List 2-4 decisions that require C-suite approval:
- Budget reallocation or contingency release
- Project cancellation/pause considerations
- Organizational restructuring
- Strategic priority changes
- Major resource shifts

For each, provide:
- The decision needed
- Options available
- Recommendation with rationale
- Consequences of inaction

---

**WRITING GUIDELINES:**
- **CRITICAL**: Use ONLY the data provided above. DO NOT add assumed information, make up details, or infer data that is not explicitly given.
- Use clear, confident language appropriate for senior executives
- Avoid jargon; explain technical terms when needed
- Use storytelling: connect the dots between data points
- Be specific: use actual numbers, names, and examples
- Be actionable: every insight should lead to a decision or action
- Be balanced: acknowledge both risks and opportunities
- Use formatting (bold, bullets, sections) to make it scannable
- Think like a trusted advisor: be honest about problems and realistic about solutions

Write this report as if you're presenting it to the CEO and Board. They trust your judgment and need you to tell them what's really happening and what they need to do about it.
"""

                        # Make LLM request
                        brief_response = safe_llm_request(
                            llm_config.get('provider', ''),
                            llm_config.get('model', ''),
                            llm_config.get('api_key', ''),
                            llm_config.get('temperature', 0.3),  # Slightly higher for more creative insights
                            llm_config.get('timeout', 180),  # Longer timeout for complex analysis
                            prompt
                        )

                        st.session_state.portfolio_executive_brief = brief_response
                        # Clear chat history and infographic when new brief is generated
                        from components.brief_chat import clear_chat_history
                        clear_chat_history('portfolio_brief_chat_history')
                        if 'portfolio_infographic' in st.session_state:
                            del st.session_state.portfolio_infographic

                    except Exception as e:
                        st.error(f"Failed to generate portfolio brief: {e}")
                        import traceback
                        st.error(f"Detailed error: {traceback.format_exc()}")

        # Handle Get Prompt button
        if get_prompt_clicked:
            with st.spinner("Preparing prompt..."):
                try:
                    # Calculate metrics (same as top of page)
                    # Health distribution
                    if 'Health_Category' not in df.columns:
                        df['Health_Category'] = df.apply(categorize_project_health, axis=1)
                    health_counts = df['Health_Category'].value_counts()

                    # Calculate all metrics needed for the prompt
                    metrics = {}
                    metrics['total_projects'] = len(df)
                    metrics['total_budget'] = df.get('Budget', pd.Series([0])).sum()
                    metrics['total_actual_cost'] = df.get('AC', pd.Series([0])).sum()
                    metrics['total_earned_value'] = df.get('EV', pd.Series([0])).sum()
                    metrics['total_planned_value'] = df.get('PV', pd.Series([0])).sum()
                    metrics['total_etc'] = df.get('ETC', pd.Series([0])).sum()
                    metrics['total_eac'] = df.get('EAC', pd.Series([0])).sum()

                    # Average metrics
                    metrics['avg_cpi'] = df['CPI'].mean() if 'CPI' in df.columns else 0
                    metrics['avg_spi'] = df['SPI'].mean() if 'SPI' in df.columns else 0
                    metrics['avg_completion'] = df['Completion %'].mean() if 'Completion %' in df.columns else 0

                    # Portfolio-level performance indices
                    metrics['portfolio_cpi'] = metrics['total_earned_value'] / metrics['total_actual_cost'] if metrics['total_actual_cost'] > 0 else 0
                    metrics['portfolio_spi'] = metrics['total_earned_value'] / metrics['total_planned_value'] if metrics['total_planned_value'] > 0 else 0

                    # TCPI calculation
                    work_remaining = metrics['total_budget'] - metrics['total_earned_value']
                    budget_remaining = metrics['total_budget'] - metrics['total_actual_cost']
                    if work_remaining <= 0:
                        metrics['portfolio_tcpi'] = 0  # Project complete
                    elif budget_remaining <= 0:
                        metrics['portfolio_tcpi'] = float('inf')  # Work remaining but no budget
                    else:
                        metrics['portfolio_tcpi'] = work_remaining / budget_remaining if budget_remaining > 0 else 0

                    # Forecast metrics
                    metrics['forecast_overrun'] = metrics['total_eac'] - metrics['total_budget']
                    metrics['overrun_percentage'] = (metrics['forecast_overrun'] / metrics['total_budget']) * 100 if metrics['total_budget'] > 0 else 0

                    # Health distribution
                    metrics['critical_projects'] = health_counts.get('Critical', 0)
                    metrics['at_risk_projects'] = health_counts.get('At Risk', 0)
                    metrics['healthy_projects'] = health_counts.get('Healthy', 0)

                    # Calculate quadrants with correct key names (same as generate button)
                    on_budget_on_schedule = len(df[(df['CPI'] >= 0.95) & (df['SPI'] >= 0.95)]) if 'CPI' in df.columns and 'SPI' in df.columns else 0
                    over_budget_on_schedule = len(df[(df['CPI'] < 0.95) & (df['SPI'] >= 0.95)]) if 'CPI' in df.columns and 'SPI' in df.columns else 0
                    on_budget_behind_schedule = len(df[(df['CPI'] >= 0.95) & (df['SPI'] < 0.95)]) if 'CPI' in df.columns and 'SPI' in df.columns else 0
                    over_budget_behind_schedule = len(df[(df['CPI'] < 0.95) & (df['SPI'] < 0.95)]) if 'CPI' in df.columns and 'SPI' in df.columns else 0

                    # Get controls for currency formatting
                    controls = st.session_state.config_dict.get('controls', {
                        'currency_symbol': '$',
                        'currency_postfix': ''
                    })

                    currency = controls.get('currency_symbol', '$')
                    postfix = controls.get('currency_postfix', '')

                    # Prepare detailed project data for trend analysis
                    project_details = []
                    for _, row in df.iterrows():
                        budget_val = row.get('Budget', row.get('BAC', 0)) or 0
                        ac_val = row.get('Actual Cost', row.get('AC', 0)) or 0
                        od_val = row.get('Original Duration', 0) or 0
                        ld_val = row.get('Forecast Duration', 0) or 0
                        ad_val = row.get('Actual Duration', 0) or 0
                        project_info = {
                            'name': row.get('Project', row.get('Project Name', 'Unknown')),
                            'organization': row.get('Organization', row.get('Org', 'Unknown')),
                            'start_date': str(row['Plan Start'])[:10] if 'Plan Start' in row and pd.notna(row.get('Plan Start')) else 'N/A',
                            'finish_date': str(row['Plan Finish'])[:10] if 'Plan Finish' in row and pd.notna(row.get('Plan Finish')) else 'N/A',
                            'plan_duration': float(od_val),
                            'likely_duration': float(ld_val),
                            'budget': float(budget_val),
                            'percent_budget_used': (float(ac_val) / float(budget_val) * 100) if budget_val else 0,
                            'actual_duration': float(ad_val),
                            'percent_time_used': (float(ad_val) / float(od_val) * 100) if od_val else 0,
                            'spi': row.get('SPI', 0) or 0,
                            'cpi': row.get('CPI', 0) or 0,
                            'actual_cost': float(ac_val),
                            'progress': row.get('Completion %', 0) or 0,
                            'tier': row.get('Budget_Category', 'Unknown'),
                            'health': row.get('Health_Category', 'Unknown'),
                        }

                        project_details.append(project_info)

                    # Analyze trends by organization
                    org_analysis = df.groupby('Organization').agg({
                        'CPI': 'mean',
                        'SPI': 'mean',
                        'Budget': 'sum'
                    }).round(2).to_dict('index') if 'Organization' in df.columns else {}

                    # Analyze trends by tier
                    tier_analysis = df.groupby('Budget_Category').agg({
                        'CPI': 'mean',
                        'SPI': 'mean',
                        'Budget': 'sum'
                    }).round(2).to_dict('index') if 'Budget_Category' in df.columns else {}

                    # Create the same prompt as the generate button
                    prompt = f"""
You are a seasoned Chief Portfolio Officer preparing an executive briefing for the C-Suite. Create a comprehensive, narrative-style report that tells the story of this portfolio's performance with actionable insights.

Write in a professional yet conversational tone that executives can quickly scan but also dive deep into. Use clear headings, bullet points, and specific numbers to support your analysis.

**IMPORTANT CONTEXT:**
- All projects in this portfolio are currently in their execution stage (actively being worked on)
- AD (Actual Duration) represents the time already spent on each project, NOT the total project duration
- Projects are ongoing and not yet completed

---

# PORTFOLIO EXECUTIVE BRIEFING
**Report Date:** {datetime.now().strftime('%B %d, %Y')}

## DATA SNAPSHOT

**Portfolio Scale:**
- {metrics['total_projects']} active projects under management
- Total Budget at Completion (BAC): {currency}{metrics['total_budget']:,.0f}{' ' + postfix if postfix else ''}
- Current Actual Cost (AC): {currency}{metrics['total_actual_cost']:,.0f}{' ' + postfix if postfix else ''}
- Portfolio Cost Performance Index (CPI): {metrics['avg_cpi']:.2f}
- Portfolio Schedule Performance Index (SPI): {metrics['avg_spi']:.2f}
- Average Completion: {metrics['avg_completion']:.1f}%

**Performance Distribution:**
- ✅ On Budget & On Schedule: {on_budget_on_schedule} projects ({on_budget_on_schedule/metrics['total_projects']*100:.1f}%)
- ⚠️ Over Budget but On Schedule: {over_budget_on_schedule} projects ({over_budget_on_schedule/metrics['total_projects']*100:.1f}%)
- ⚠️ On Budget but Behind Schedule: {on_budget_behind_schedule} projects ({on_budget_behind_schedule/metrics['total_projects']*100:.1f}%)
- 🚨 Over Budget & Behind Schedule: {over_budget_behind_schedule} projects ({over_budget_behind_schedule/metrics['total_projects']*100:.1f}%)

**Project Details:**
{format_project_details_for_prompt(project_details, currency, postfix)}

**Organization Performance:**
{org_analysis}

**Budget Tier Analysis:**
{tier_analysis}

---

## YOUR TASK

Based on this data, create an executive briefing that includes:

1. **Executive Summary** (3-4 sentences)
   - What's the overall portfolio health?
   - What's the single most important thing executives need to know?

2. **Key Insights** (3-5 bullet points)
   - What patterns or trends stand out?
   - Are there any concerning clusters (by org, tier, or timeline)?
   - What's driving performance (good or bad)?

3. **Deep Dive Analysis**
   - Which specific projects need immediate attention and why?
   - Are there systemic issues affecting multiple projects?
   - What hidden risks or opportunities do you see in the data?

4. **Strategic Recommendations** (3-5 prioritized actions)
   - What should leadership do now?
   - What resources need to be reallocated?
   - What decisions need to be made?

5. **Looking Ahead**
   - What will happen if we stay on the current path?
   - What early warning signs should we watch?

## WRITING GUIDELINES

- **CRITICAL**: Use ONLY the data provided above. DO NOT add assumed information, make up details, or infer data that is not explicitly given.
- Start with what matters most to the C-Suite (risk, money, reputation)
- Use specific project names and numbers to illustrate points
- Don't sugarcoat problems, but frame them with solutions
- Be balanced: acknowledge both risks and opportunities
- Use formatting (bold, bullets, sections) to make it scannable
- Think like a trusted advisor: be honest about problems and realistic about solutions

Write this report as if you're presenting it to the CEO and Board. They trust your judgment and need you to tell them what's really happening and what they need to do about it.
"""

                    # Store in session state and display
                    st.session_state.portfolio_prompt_only = prompt

                    st.success("✅ Prompt generated! Copy the text below and paste it into ChatGPT, Claude, or any LLM.")

                    st.text_area(
                        "Prompt for LLM",
                        value=prompt,
                        height=400,
                        key="prompt_text_area",
                        help="Copy this entire prompt and paste it into your preferred AI chat interface"
                    )

                    # Download button for the prompt
                    st.download_button(
                        "📥 Download Prompt as Text File",
                        prompt,
                        file_name=f"executive_brief_prompt_{datetime.now().strftime('%Y%m%d')}.txt",
                        mime="text/plain",
                        key="download_prompt_only"
                    )

                except Exception as e:
                    st.error(f"Failed to generate prompt: {e}")
                    import traceback
                    st.error(f"Detailed error: {traceback.format_exc()}")

        # Display brief if generated
        if "portfolio_executive_brief" in st.session_state:
            brief = st.session_state.portfolio_executive_brief
            if brief.startswith("Error:") or brief == "No API key available":
                if brief == "No API key available":
                    st.warning("⚠️ No API key available. Please upload an API key file in File Management.")
                else:
                    st.error(brief)
            else:
                st.markdown("#### 📄 Executive Portfolio Report")
                # Clean up LaTeX/math formatting issues from LLM response
                # Replace inline math delimiters that shouldn't be interpreted as LaTeX
                cleaned_brief = brief.replace('$', r'\$')  # Escape dollar signs to prevent LaTeX rendering
                st.markdown(cleaned_brief)

                # Download button
                st.download_button(
                    "📥 Download Portfolio Brief",
                    brief,
                    file_name=f"portfolio_executive_brief_{datetime.now().strftime('%Y%m%d')}.md",
                    mime="text/markdown",
                    key="download_portfolio_brief"
                )

                # Infographic Generation Section
                st.markdown("---")
                st.markdown("#### 🎨 Visual Summary")

                # Get infographic configuration
                from utils.portfolio_settings import get_infographic_replicate_key
                voice_config = st.session_state.get('config_dict', {}).get('voice_config', {})
                infographic_config = st.session_state.get('config_dict', {}).get('infographic_config', {})

                # Check if infographics are enabled and key is available
                infographic_enabled = infographic_config.get('enabled', False)
                replicate_key = get_infographic_replicate_key(voice_config, infographic_config)
                infographic_model = infographic_config.get('model', 'nano-banana')

                if infographic_enabled and replicate_key:
                    if st.button("🖼️ Generate Infographic", key="gen_portfolio_infographic", help="Generate an AI-powered visual summary of portfolio metrics"):
                        # Collect metrics for infographic
                        health_counts_info = df['Health_Category'].value_counts() if 'Health_Category' in df.columns else {}
                        total_projects = len(df)

                        infographic_metrics = {
                            'total_projects': total_projects,
                            'cpi': portfolio_cpi_weighted,
                            'spi': portfolio_spi_weighted,
                            'critical_projects': health_counts_info.get('Critical', 0),
                            'healthy_pct': (health_counts_info.get('Healthy', 0) / total_projects * 100) if total_projects > 0 else 0,
                            'at_risk_pct': (health_counts_info.get('At Risk', 0) / total_projects * 100) if total_projects > 0 else 0,
                            'critical_pct': (health_counts_info.get('Critical', 0) / total_projects * 100) if total_projects > 0 else 0,
                        }

                        with st.spinner("Generating infographic with AI..."):
                            try:
                                from services.infographic_service import InfographicService
                                service = InfographicService(replicate_key, model=infographic_model)
                                image_bytes, error = service.generate_portfolio_infographic(infographic_metrics)

                                if image_bytes:
                                    st.session_state.portfolio_infographic = image_bytes
                                    st.rerun()
                                else:
                                    st.error(f"Failed to generate infographic: {error}")
                            except Exception as e:
                                st.error(f"Infographic generation error: {e}")

                    # Display generated infographic
                    if 'portfolio_infographic' in st.session_state:
                        st.image(st.session_state.portfolio_infographic, caption="Portfolio Executive Summary Infographic")
                        st.download_button(
                            "📥 Download Infographic",
                            st.session_state.portfolio_infographic,
                            file_name=f"portfolio_infographic_{datetime.now().strftime('%Y%m%d')}.png",
                            mime="image/png",
                            key="download_portfolio_infographic"
                        )
                elif not infographic_enabled:
                    st.caption("Enable Infographics in Portfolio Management settings to generate AI visual summaries.")
                else:
                    st.caption("Configure Replicate API key in Portfolio Management to enable AI infographic generation.")

                # Chat with Brief Section
                st.markdown("---")
                llm_config = st.session_state.get('config_dict', {}).get('llm_config', {})

                from components.brief_chat import render_brief_chat
                render_brief_chat(
                    brief_content=brief,
                    history_key='portfolio_brief_chat_history',
                    llm_config=llm_config,
                    voice_config=voice_config,
                    expander_title="Chat with your Portfolio Brief"
                )


    # Sidebar with executive styling
    with st.sidebar:
        st.markdown('<div class="section-header">⚙️ Executive Controls</div>', unsafe_allow_html=True)
        
        st.markdown('<div class="section-header">📈 Portfolio KPIs</div>', unsafe_allow_html=True)

        # Use the same weighted calculations as the main dashboard for consistency
        st.metric("⭐ Portfolio CPI", f"{portfolio_cpi_weighted:.3f}", help="Weighted Portfolio Cost Performance Index (weighted by budget)")
        st.metric("🎯 Portfolio SPI", f"{portfolio_spi_weighted:.3f}", help="Weighted Portfolio Schedule Performance Index (weighted by budget)")
        st.metric("📊 Portfolio SPIe", f"{portfolio_spie_weighted:.3f}", help="Weighted Schedule Performance Index Estimate (weighted by budget)")
        st.metric("🏢 Total Projects", len(df), help="Active projects in portfolio")
        
        # Add executive summary
        st.markdown('<div class="section-header">📋 Executive Summary</div>', unsafe_allow_html=True)
        portfolio_health = "Critical" if portfolio_cpi_weighted < 0.8 else "At Risk" if portfolio_cpi_weighted < 0.95 else "Healthy"
        health_color = "🔴" if portfolio_health == "Critical" else "🟡" if portfolio_health == "At Risk" else "🟢"
        
        st.markdown(f"""
        **Portfolio Status:** {health_color} {portfolio_health}
        
        **Key Insights:**
        - {metrics['critical_projects']} projects need immediate attention
        - {format_currency(metrics['forecast_overrun'], currency_symbol, currency_postfix, thousands=False)} projected overrun
        - {metrics['overrun_percentage']:.1f}% budget variance
        """)

    # Professional Footer
    st.markdown("""
    <div class="footer">
        <div style="border-top: 1px solid rgba(0,0,0,0.1); padding-top: 1rem; margin-top: 2rem;">
            <strong>Portfolio Executive Dashboard</strong> • Real-time Intelligence for Strategic Decision Making<br>
            Generated on {date} • Confidential Executive Report
        </div>
    </div>
    """.format(date=datetime.now().strftime('%B %d, %Y at %I:%M %p')), unsafe_allow_html=True)

if __name__ == "__main__":
    main()
# Show user info in sidebar
from utils.auth import show_user_info_sidebar
show_user_info_sidebar()

"""
Portfolio Management Suite - Main Navigation
A comprehensive project portfolio analysis and executive dashboard system.
"""

# CRITICAL: Set environment variables FIRST, before ANY imports
# This must be the absolute first code that runs
import os
os.environ['GRPC_VERBOSITY'] = 'NONE'
os.environ['GRPC_TRACE'] = ''
os.environ['GRPC_LOG_SEVERITY_LEVEL'] = 'NONE'
os.environ['GLOG_minloglevel'] = '3'
os.environ['GLOG_logtostderr'] = '0'
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
os.environ['GRPC_ENABLE_FORK_SUPPORT'] = '0'
os.environ['GRPC_POLL_STRATEGY'] = 'poll'

import sys
import os
import warnings
import logging

# Force the root directory into the path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# Suppress Python warnings
warnings.filterwarnings('ignore', category=UserWarning, module='google')
warnings.filterwarnings('ignore', category=FutureWarning, module='google')

# Suppress absl logging
try:
    import absl.logging
    absl.logging.set_verbosity(absl.logging.ERROR)
    absl.logging.set_stderrthreshold(absl.logging.ERROR)
except ImportError:
    pass

# Suppress gRPC logging
logging.getLogger('grpc').setLevel(logging.ERROR)
logging.getLogger('google').setLevel(logging.ERROR)
logging.getLogger('google.auth').setLevel(logging.ERROR)

import streamlit as st
from utils.auth import check_authentication, show_user_info_sidebar
from utils.user_manager import check_page_access, is_admin
from utils.firestore_client import get_firestore_client

# Initialize Firestore client
db = get_firestore_client()

# Page configuration
st.set_page_config(
    page_title="Portfolio Management Suite",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Check authentication first
if not check_authentication():
    st.stop()  # Stop here if not authenticated

# Custom CSS for navigation
st.markdown("""
<style>
    .main-header {
        background: linear-gradient(135deg, rgba(255,255,255,0.25) 0%, rgba(255,255,255,0.1) 100%);
        backdrop-filter: blur(20px);
        border: 1px solid rgba(255,255,255,0.3);
        padding: 2rem;
        border-radius: 20px;
        text-align: center;
        margin-bottom: 2rem;
        box-shadow: 0 25px 50px rgba(0, 0, 0, 0.15);
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

# Main header
st.markdown("""
<div class="main-header">
    <h1>📊 Portfolio Analysis Suite</h1>
    <h3>Project Portfolio Analysis & Executive Dashboard</h3>
    <p style="margin-top: 1rem; font-size: 1.1em; color: #666; font-style: italic;">
        Smarter Projects and Portfolios with Earned Value Analysis and AI-Powered Executive Reporting<br>
        <strong>Beta Version 1.9 • Released Feb 19, 2026</strong><br>
        Developed by Dr. Khalid Ahmad Khan – <a href="https://www.linkedin.com/in/khalidahmadkhan/" target="_blank" style="color: #0066cc; text-decoration: none;">LinkedIn</a>
    </p>
</div>
""", unsafe_allow_html=True)

# Navigation
st.markdown("## Choose Your Tool")

# Get user email for access checks
user_email = st.session_state.get('user_email')

# Determine which pages to show
show_file_mgmt = check_page_access(db, user_email, 'file_management')
show_manual_entry = check_page_access(db, user_email, 'manual_data_entry')
show_project_analysis = check_page_access(db, user_email, 'project_analysis')
show_portfolio_analysis = check_page_access(db, user_email, 'portfolio_analysis')
show_portfolio_charts = check_page_access(db, user_email, 'portfolio_charts')
show_cash_flow = check_page_access(db, user_email, 'cash_flow_simulator')
show_evm_simulator = check_page_access(db, user_email, 'evm_simulator')

# ═══════════════════════════════════════════════════════════════════
# SECTION 1: File Management, Manual Data Entry, Project Analysis
# ═══════════════════════════════════════════════════════════════════
section1_pages = [show_file_mgmt, show_manual_entry, show_project_analysis]
if any(section1_pages):
    col1, col2, col3 = st.columns(3)

    with col1:
        if show_file_mgmt:
            st.markdown("""
            #### 📁 Portfolio Management
            - Data import (CSV/JSON)
            - Configuration settings
            - Batch calculations
            - Export & download options
            """)
            if st.button("📁 Open Portfolio Management", key="file_mgmt_btn", width="stretch"):
                st.switch_page("pages/1_Portfolio_Management.py")

    with col2:
        if show_manual_entry:
            st.markdown("""
            #### 📝 Enter Progress Data
            - Quick project data input
            - Direct data entry interface
            - Alternative to file upload
            - Instant data validation
            """)
            if st.button("✏️ Open Enter Progress Data", key="manual_btn", width="stretch"):
                st.switch_page("pages/3_Enter_Progress_Data.py")

    with col3:
        if show_project_analysis:
            st.markdown("""
            #### 🔍 Project Analysis
            - Single project EVM analysis
            - Individual project insights
            - Detailed calculations
            - Project-level charts
            """)
            if st.button("🚀 Open Project Analysis", key="project_btn", width="stretch"):
                st.switch_page("pages/5_Project_Analysis.py")

    st.markdown("---")

# ═══════════════════════════════════════════════════════════════════
# SECTION 2: Portfolio Analysis, Portfolio Charts
# ═══════════════════════════════════════════════════════════════════
section2_pages = [show_portfolio_analysis, show_portfolio_charts]
if any(section2_pages):
    col1, col2, col3 = st.columns(3)

    with col1:
        if show_portfolio_analysis:
            st.markdown("""
            #### 📈 Portfolio Analysis
            - Portfolio health metrics
            - Multi-project comparisons
            - Strategic performance indicators
            - Executive summary reports
            """)
            if st.button("📊 Open Portfolio Analysis", key="portfolio_btn", width="stretch"):
                st.switch_page("pages/6_Portfolio_Analysis.py")

    with col2:
        if show_portfolio_charts:
            st.markdown("""
            #### 📊 Portfolio Charts
            - Interactive baseline vs forecast timeline
            - Organization, budget, and date filtering
            - EV progress shading with forecast alerts
            - Hover insights for project detail
            """)
            if st.button("📊 Open Portfolio Charts", key="gantt_btn", width="stretch"):
                st.switch_page("pages/7_Portfolio_Charts.py")

    st.markdown("---")

# ═══════════════════════════════════════════════════════════════════
# SECTION 3: Cash Flow Simulator, EVM Simulator
# ═══════════════════════════════════════════════════════════════════
# NOTE: These pages are currently disabled as they don't exist
# Uncomment and create the pages if needed
# section3_pages = [show_cash_flow, show_evm_simulator]
# if any(section3_pages):
#     col1, col2, col3 = st.columns(3)
#
#     with col1:
#         if show_cash_flow:
#             st.markdown("""
#             #### 💸 Cash Flow Simulator
#             - Project delay impact analysis
#             - Multiple cash flow patterns (Linear, S-Curve, Highway, Building)
#             - Inflation and delay modeling
#             - Baseline comparison & export capabilities
#             """)
#             if st.button("📈 Open Cash Flow Simulator", key="cashflow_btn", width="stretch"):
#                 st.switch_page("pages/6_Cash_Flow_Simulator.py")
#
#     with col2:
#         if show_evm_simulator:
#             st.markdown("""
#             #### 🎯 EVM Simulator
#             - Interactive EVM scenario modeling
#             - Performance index simulations
#             - Schedule and cost impact analysis
#             - Advanced forecasting tools
#             """)
#             if st.button("🎯 Open EVM Simulator", key="evm_btn", width="stretch"):
#                 st.switch_page("pages/7_EVM_Simulator.py")
#
#     st.markdown("---")

# ═══════════════════════════════════════════════════════════════════
# SECTION 4: Video Tutorials & Documentation (Available to All Users)
# ═══════════════════════════════════════════════════════════════════
col1, col2, col3 = st.columns(3)

with col1:
    st.markdown("""
    #### 📺 Video Tutorials
    - Step-by-step instructions
    - Application walkthroughs
    - Feature demonstrations
    - Getting started guides
    """)
    st.markdown("""
    <a href="https://youtube.com/playlist?list=PLJyNrFUBTQvtbZbUmbPA3Ej-gSJX85Dfm&si=5ThkVaCRb8_cbwkB" target="_blank" style="text-decoration: none;">
        <button style="background: linear-gradient(135deg, #FF0000 0%, #CC0000 100%); color: white; padding: 0.75rem 1.5rem; border-radius: 10px; border: none; cursor: pointer; font-weight: 600; width: 100%; font-size: 1em;">
            📺 Watch Tutorial Playlist
        </button>
    </a>
    """, unsafe_allow_html=True)

with col2:
    st.markdown("""
    #### 📚 Documentation
    - User manuals & guides
    - Getting started tutorials
    - Feature documentation
    - Quick reference materials
    """)

    # Expandable section for multiple PDFs
    with st.expander("📥 Download Documentation", expanded=False):
        # Portfolio Overview
        try:
            with open("docs/Product_Portfolio_Management_Overview.pdf", "rb") as file:
                st.download_button(
                    label="📄 Portfolio Management Overview",
                    data=file,
                    file_name="Portfolio_Management_Overview.pdf",
                    mime="application/pdf",
                    width='stretch',
                    key="download_overview"
                )
        except FileNotFoundError:
            st.warning("Overview PDF not found")

        st.markdown("---")
        st.markdown("**User Manuals:**")

        # User Manuals
        manuals = [
            ("Settings", "User Manual - Settings.pdf", "settings"),
            ("Entering Data", "User Manual - Entering Data.pdf", "entering"),
            ("Loading CSV Data", "User Manual - Loading csv Data.pdf", "csv"),
            ("Loading JSON Data", "User Manual - Loading JSON data.pdf", "json"),
            ("Project Analysis", "User Manual - Project Analysis.pdf", "analysis")
        ]

        for label, filename, key_suffix in manuals:
            try:
                with open(f"docs/{filename}", "rb") as file:
                    st.download_button(
                        label=f"📘 {label}",
                        data=file,
                        file_name=filename,
                        mime="application/pdf",
                        width='stretch',
                        key=f"download_{key_suffix}"
                    )
            except FileNotFoundError:
                st.caption(f"⚠️ {label} manual not available")

st.markdown("---")

# ═══════════════════════════════════════════════════════════════════
# SECTION 5: User Management (Admin Only)
# ═══════════════════════════════════════════════════════════════════
if is_admin(db, user_email):
    col1, col2, col3 = st.columns(3)

    with col1:
        st.markdown("""
        #### 👥 User Management
        - Manage user access permissions
        - Grant/revoke page access
        - View user statistics
        - Admin-only access control
        """)
        if st.button("👥 Open User Management", key="user_mgmt_btn", width="stretch"):
            st.switch_page("pages/10_User_Management.py")

    st.markdown("---")

# Quick stats if data exists
if hasattr(st.session_state, 'batch_results') and st.session_state.batch_results is not None:
    st.markdown("---")
    st.markdown("### 📋 Current Session Data")
    col1, col2, col3 = st.columns(3)

    with col1:
        st.metric("Projects in Session", len(st.session_state.batch_results))
    with col2:
        if 'CPI' in st.session_state.batch_results.columns:
            avg_cpi = st.session_state.batch_results['CPI'].mean()
            st.metric("Average CPI", f"{avg_cpi:.2f}")
    with col3:
        st.success("✅ Data Ready for Dashboard")

# Help section
with st.expander("ℹ️ How to Use This System"):
    st.markdown("""
    ### 📋 Complete System Workflow

    #### 1. Loading Data
    Choose one of three methods to load your project data:
    - **Manual Entry:** Use Enter Progress Data for quick single-project input with instant validation
    - **CSV Upload:** Import structured project data via CSV files in Portfolio Management
    - **JSON Upload:** Load comprehensive project portfolios with JSON format in Portfolio Management

    #### 2. Portfolio Management: Configuration & Settings
    Before running analysis, configure your system in Portfolio Management:
    - **Set Global Settings:** Define default parameters (budget tiers, thresholds, date ranges)
    - **Budget Tier Configuration:** Establish budget categories for portfolio classification
    - **Run Batch EVM:** Process projects in either mode:
      - **Global Mode:** Apply uniform settings across all projects (standardized analysis)
      - **Project-Specific Mode:** Use individual project parameters (custom analysis)
    - **Optional - LLM Integration:**
      - Upload your LLM provider API key
      - Test connection to verify connectivity
      - Select preferred model and parameters for AI-powered insights

    #### 3. Analysis & Visualization
    After loading data and running batch calculations:
    - **Project Analysis:** Deep-dive into individual project metrics and detailed EVM calculations
    - **Portfolio Analysis:** View multi-project comparisons, health metrics, and executive summaries
    - **Portfolio Charts:** Interactive timeline visualizations with baseline vs forecast comparisons

    **Data Flow:**
    - Portfolio tools (Analysis/Charts) share session data automatically
    - All tools provide export capabilities for further analysis
    """)

# Show user info in sidebar
show_user_info_sidebar()
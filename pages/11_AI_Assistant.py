"""
AI Assistant - Direct chat interface for portfolio and project data.

Provides conversational access to portfolio data without requiring
executive report generation first.
"""

import streamlit as st
import pandas as pd
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

# Import core utilities
from utils.auth import check_authentication, require_page_access
from config.constants import USE_DATABASE
from utils.portfolio_context import render_portfolio_context
from utils.portfolio_settings import load_portfolio_settings
from services.db_data_service import DatabaseDataManager
from database.db_connection import get_db
from components.brief_chat import render_brief_chat, clear_chat_history

# Page configuration
st.set_page_config(
    page_title="AI Assistant",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Check authentication and page access
if not check_authentication():
    st.stop()

require_page_access('ai_assistant', 'AI Assistant')


def format_currency(value: float, symbol: str = "$", postfix: str = "") -> str:
    """Format currency value with symbol and postfix"""
    if pd.isna(value) or value == 0:
        return f"{symbol}0{postfix}"

    abs_value = abs(value)
    if abs_value >= 1_000_000_000:
        formatted = f"{value/1_000_000_000:.2f}B"
    elif abs_value >= 1_000_000:
        formatted = f"{value/1_000_000:.2f}M"
    elif abs_value >= 1_000:
        formatted = f"{value/1_000:.1f}K"
    else:
        formatted = f"{value:.0f}"

    return f"{symbol}{formatted}{postfix}"


def categorize_project_health(row) -> str:
    """Categorize project health based on CPI and SPI"""
    cpi = row.get('CPI', 1.0) if pd.notna(row.get('CPI')) else 1.0
    spi = row.get('SPI', 1.0) if pd.notna(row.get('SPI')) else 1.0

    if cpi < 0.9 or spi < 0.9:
        return 'Critical'
    elif cpi < 0.95 or spi < 0.95:
        return 'At Risk'
    else:
        return 'Healthy'


def get_project_list(portfolio_id: int) -> pd.DataFrame:
    """Get list of projects in portfolio for selection"""
    if not USE_DATABASE:
        return pd.DataFrame()

    try:
        db = get_db()
        query = """
            SELECT
                p.project_id,
                p.project_name,
                p.project_manager,
                p.responsible_organization,
                p.project_status
            FROM project p
            WHERE p.portfolio_id = ? AND p.is_active = TRUE
            ORDER BY p.project_name
        """
        return db.execute(query, (portfolio_id,)).df()
    except Exception as e:
        st.error(f"Error loading projects: {e}")
        return pd.DataFrame()


def get_project_details(project_id: int, status_date) -> dict:
    """Get detailed project data for AI context using pre-calculated EVM results"""
    if not USE_DATABASE:
        return {}

    try:
        db = get_db()

        # Get project with baseline and calculated EVM metrics from status report
        query = """
            SELECT
                p.project_id,
                p.project_name,
                p.project_manager,
                p.responsible_organization,
                p.project_status,
                b.budget_at_completion as bac,
                b.planned_start_date,
                b.planned_finish_date,
                sr.status_date,
                sr.actual_cost as ac,
                COALESCE(sr.calculated_ev, sr.earned_value, 0) as ev,
                COALESCE(sr.calculated_pv, sr.planned_value, 0) as pv,
                sr.cpi,
                sr.spi,
                sr.cv,
                sr.sv,
                sr.eac,
                sr.etc,
                sr.vac,
                sr.tcpi,
                sr.percent_complete,
                sr.likely_completion_date
            FROM project p
            LEFT JOIN project_baseline b ON p.project_id = b.project_id
                AND b.is_active = TRUE
                AND (b.baseline_end_date IS NULL OR b.baseline_end_date > ?)
            LEFT JOIN project_status_report sr ON p.project_id = sr.project_id
                AND sr.status_date = ?
                AND sr.is_active = TRUE
            WHERE p.project_id = ? AND p.is_active = TRUE
        """
        result = db.execute(query, (status_date, status_date, project_id)).fetchone()

        if result:
            columns = ['project_id', 'project_name', 'project_manager', 'organization',
                      'status', 'bac', 'plan_start', 'plan_finish', 'status_date',
                      'ac', 'ev', 'pv', 'cpi', 'spi', 'cv', 'sv', 'eac', 'etc',
                      'vac', 'tcpi', 'percent_complete', 'likely_completion']
            return dict(zip(columns, result))
        return {}
    except Exception as e:
        st.error(f"Error loading project details: {e}")
        return {}


def build_portfolio_context(portfolio_id: int, status_date, settings: dict) -> str:
    """Build a structured data context for the AI from portfolio data"""

    currency = settings.get('currency_symbol', '$')
    postfix = settings.get('currency_postfix', '')

    # Get pre-calculated EVM results
    db_manager = DatabaseDataManager()
    df = db_manager.get_evm_results_for_period(portfolio_id, status_date)

    if df.empty:
        return "No data available for this portfolio and period. Please ensure EVM calculations have been run."

    # Helper to safely get numeric value
    def safe_val(value, default=0):
        if pd.isna(value):
            return default
        try:
            return float(value)
        except:
            return default

    # Calculate portfolio totals from the results
    total_projects = len(df)
    total_bac = df['bac'].sum() if 'bac' in df.columns else 0
    total_ac = df['ac'].sum() if 'ac' in df.columns else 0
    total_ev = df['ev'].sum() if 'ev' in df.columns else 0
    total_pv = df['pv'].sum() if 'pv' in df.columns else 0
    total_eac = df['eac'].sum() if 'eac' in df.columns else 0

    # Portfolio performance indices
    portfolio_cpi = total_ev / total_ac if total_ac > 0 else 0
    portfolio_spi = total_ev / total_pv if total_pv > 0 else 0

    # TCPI calculation
    work_remaining = total_bac - total_ev
    budget_remaining = total_bac - total_ac
    if budget_remaining <= 0:
        portfolio_tcpi = float('inf') if work_remaining > 0 else 0
    else:
        portfolio_tcpi = work_remaining / budget_remaining

    # Forecast
    forecast_overrun = total_eac - total_bac

    # Completion percentages
    percent_complete = (total_ev / total_bac * 100) if total_bac > 0 else 0
    percent_spent = (total_ac / total_bac * 100) if total_bac > 0 else 0

    # Build project details table
    project_lines = []
    for _, row in df.iterrows():
        project_name = row.get('project_name', 'Unknown')
        bac = safe_val(row.get('bac', 0))
        ac = safe_val(row.get('ac', 0))
        ev = safe_val(row.get('ev', 0))
        pv = safe_val(row.get('pv', 0))
        proj_cpi = safe_val(row.get('cpi', 0))
        proj_spi = safe_val(row.get('spi', 0))
        proj_complete = safe_val(row.get('percent_complete', 0))
        proj_eac = safe_val(row.get('eac', 0))
        proj_pct_budget = safe_val(row.get('percent_budget_used', 0))
        proj_pct_time = safe_val(row.get('percent_time_used', 0))
        proj_orig_dur = safe_val(row.get('original_duration_months', 0))
        proj_actual_dur = safe_val(row.get('actual_duration_months', 0))
        proj_likely_dur = safe_val(row.get('likely_duration', 0))
        plan_start = row.get('plan_start', None)
        plan_finish = row.get('plan_finish', None)

        # Health status based on pre-calculated CPI/SPI
        if proj_cpi < 0.9 or proj_spi < 0.9:
            health = "CRITICAL"
        elif proj_cpi < 0.95 or proj_spi < 0.95:
            health = "AT RISK"
        else:
            health = "Healthy"

        # Format dates
        start_str = str(plan_start) if pd.notna(plan_start) else "N/A"
        finish_str = str(plan_finish) if pd.notna(plan_finish) else "N/A"

        project_lines.append(
            f"- {project_name}: [{health}]\n"
            f"    Budget (BAC): {format_currency(bac, currency, postfix)}, "
            f"AC: {format_currency(ac, currency, postfix)}, "
            f"EV: {format_currency(ev, currency, postfix)}, "
            f"PV: {format_currency(pv, currency, postfix)}\n"
            f"    CPI={proj_cpi:.2f}, SPI={proj_spi:.2f}, "
            f"{proj_complete:.1f}% complete, "
            f"EAC: {format_currency(proj_eac, currency, postfix)}\n"
            f"    Plan Start: {start_str}, Plan Finish: {finish_str}\n"
            f"    Original Duration: {proj_orig_dur:.1f} months, "
            f"Actual Duration: {proj_actual_dur:.1f} months, "
            f"Likely Duration: {proj_likely_dur:.1f} months\n"
            f"    Budget Used: {proj_pct_budget:.1f}%, Time Used: {proj_pct_time:.1f}%"
        )

    # Count health categories
    critical = sum(1 for line in project_lines if "[CRITICAL]" in line)
    at_risk = sum(1 for line in project_lines if "[AT RISK]" in line)
    healthy = total_projects - critical - at_risk

    # Build the context string
    context = f"""PORTFOLIO DATA SNAPSHOT
=======================
Status Date: {status_date}
Total Projects: {total_projects}

FINANCIAL SUMMARY
-----------------
Total Budget (BAC): {format_currency(total_bac, currency, postfix)}
Actual Cost (AC): {format_currency(total_ac, currency, postfix)}
Earned Value (EV): {format_currency(total_ev, currency, postfix)}
Planned Value (PV): {format_currency(total_pv, currency, postfix)}

PERFORMANCE METRICS
-------------------
Cost Performance Index (CPI): {portfolio_cpi:.3f} {'(Over budget)' if portfolio_cpi < 1 else '(Under budget)' if portfolio_cpi > 1 else '(On budget)'}
Schedule Performance Index (SPI): {portfolio_spi:.3f} {'(Behind schedule)' if portfolio_spi < 1 else '(Ahead of schedule)' if portfolio_spi > 1 else '(On schedule)'}
To-Complete Performance Index (TCPI): {portfolio_tcpi:.3f}

FORECAST
--------
Estimate at Completion (EAC): {format_currency(total_eac, currency, postfix)}
Forecast Variance: {format_currency(forecast_overrun, currency, postfix)} {'OVERRUN' if forecast_overrun > 0 else 'UNDERRUN' if forecast_overrun < 0 else ''}

PROGRESS
--------
Work Complete: {percent_complete:.1f}%
Budget Spent: {percent_spent:.1f}%

PORTFOLIO HEALTH
----------------
Critical Projects: {critical}
At Risk Projects: {at_risk}
Healthy Projects: {healthy}

PROJECT DETAILS
---------------
{chr(10).join(project_lines)}

You are a project controls expert specializing in Earned Value Management (EVM).
Answer questions about this portfolio based on the data above.
Provide specific numbers and project names when relevant.
"""

    return context


def build_project_context(project_data: dict, settings: dict) -> str:
    """Build a structured data context for a single project"""

    currency = settings.get('currency_symbol', '$')
    postfix = settings.get('currency_postfix', '')

    if not project_data:
        return "No data available for this project."

    # Extract values with defaults
    project_name = project_data.get('project_name', 'Unknown')
    pm = project_data.get('project_manager', 'Unknown')
    org = project_data.get('organization', 'Unknown')
    status = project_data.get('status', 'Unknown')

    bac = project_data.get('bac', 0) or 0
    ac = project_data.get('ac', 0) or 0
    ev = project_data.get('ev', 0) or 0
    pv = project_data.get('pv', 0) or 0

    cpi = project_data.get('cpi', 0) or 0
    spi = project_data.get('spi', 0) or 0
    cv = project_data.get('cv', 0) or 0
    sv = project_data.get('sv', 0) or 0
    eac = project_data.get('eac', 0) or 0
    etc = project_data.get('etc', 0) or 0
    vac = project_data.get('vac', 0) or 0
    tcpi = project_data.get('tcpi', 0) or 0

    percent_complete = project_data.get('percent_complete', 0) or 0
    plan_start = project_data.get('plan_start', 'N/A')
    plan_finish = project_data.get('plan_finish', 'N/A')
    likely_completion = project_data.get('likely_completion', 'N/A')
    status_date = project_data.get('status_date', 'N/A')

    # Health status
    if cpi < 0.9 or spi < 0.9:
        health = "CRITICAL - Immediate attention required"
    elif cpi < 0.95 or spi < 0.95:
        health = "AT RISK - Monitoring required"
    else:
        health = "HEALTHY - On track"

    context = f"""PROJECT DATA SNAPSHOT
=====================
Project: {project_name}
Project Manager: {pm}
Organization: {org}
Status: {status}
Status Date: {status_date}

SCHEDULE
--------
Planned Start: {plan_start}
Planned Finish: {plan_finish}
Likely Completion: {likely_completion}

FINANCIAL SUMMARY
-----------------
Budget at Completion (BAC): {format_currency(bac, currency, postfix)}
Actual Cost (AC): {format_currency(ac, currency, postfix)}
Earned Value (EV): {format_currency(ev, currency, postfix)}
Planned Value (PV): {format_currency(pv, currency, postfix)}

VARIANCES
---------
Cost Variance (CV): {format_currency(cv, currency, postfix)} {'(Over budget)' if cv < 0 else '(Under budget)' if cv > 0 else ''}
Schedule Variance (SV): {format_currency(sv, currency, postfix)} {'(Behind schedule)' if sv < 0 else '(Ahead)' if sv > 0 else ''}

PERFORMANCE INDICES
-------------------
Cost Performance Index (CPI): {cpi:.3f} {'(Inefficient)' if cpi < 1 else '(Efficient)' if cpi > 1 else ''}
Schedule Performance Index (SPI): {spi:.3f} {'(Behind)' if spi < 1 else '(Ahead)' if spi > 1 else ''}
To-Complete Performance Index (TCPI): {tcpi:.3f}

FORECAST
--------
Estimate at Completion (EAC): {format_currency(eac, currency, postfix)}
Estimate to Complete (ETC): {format_currency(etc, currency, postfix)}
Variance at Completion (VAC): {format_currency(vac, currency, postfix)}

PROGRESS
--------
Percent Complete: {percent_complete:.1f}%

HEALTH STATUS
-------------
{health}

You are a project controls expert specializing in Earned Value Management (EVM).
Answer questions about this project based on the data above.
Provide specific numbers and explain EVM concepts when relevant.
"""

    return context


def main():
    # Header
    st.markdown("## 🤖 AI Assistant")
    st.markdown("Chat directly with your portfolio and project data")

    # Portfolio & Period Selection
    st.markdown("---")
    portfolio_id, status_date = render_portfolio_context(show_period_selector=True)

    if not portfolio_id:
        st.warning("Please select a portfolio to continue")
        st.info("Go to **Portfolio Management** to create or select a portfolio")
        st.stop()

    if not status_date:
        st.warning("No data periods available for this portfolio")
        st.info("Upload data in **Load Progress Data** to create the first period")
        st.stop()

    # Load portfolio settings
    settings = load_portfolio_settings(portfolio_id)
    llm_config = settings.get('llm_config', {})
    voice_config = settings.get('voice_config', {})

    # Check if LLM is configured
    has_llm_key = bool(llm_config.get('api_key', '').strip())

    if not has_llm_key:
        st.error("LLM API key not configured")
        st.info("Configure your LLM settings in **Portfolio Management** → Settings tab")
        st.stop()

    st.markdown("---")

    # Chat mode selection
    col1, col2 = st.columns([1, 2])

    with col1:
        chat_mode = st.radio(
            "Chat Mode",
            options=["Portfolio Overview", "Specific Project"],
            index=0,
            help="Choose to chat about the entire portfolio or a specific project"
        )

    project_id = None
    if chat_mode == "Specific Project":
        with col2:
            projects_df = get_project_list(portfolio_id)
            if projects_df.empty:
                st.warning("No projects found in this portfolio")
                st.stop()

            project_options = dict(zip(
                projects_df['project_name'].tolist(),
                projects_df['project_id'].tolist()
            ))

            selected_project = st.selectbox(
                "Select Project",
                options=list(project_options.keys()),
                index=0
            )
            project_id = project_options[selected_project]

    st.markdown("---")

    # Build appropriate context based on mode
    if chat_mode == "Portfolio Overview":
        history_key = f"ai_assistant_portfolio_{portfolio_id}_{status_date}"
        data_context = build_portfolio_context(portfolio_id, status_date, settings)

        # Show quick stats
        with st.expander("📊 Current Data Summary", expanded=False):
            st.text(data_context[:5000] + "..." if len(data_context) > 5000 else data_context)
    else:
        history_key = f"ai_assistant_project_{project_id}_{status_date}"
        project_data = get_project_details(project_id, status_date)
        data_context = build_project_context(project_data, settings)

        # Show quick stats
        with st.expander("📊 Current Data Summary", expanded=False):
            st.text(data_context)

    # Clear history if context changed (different portfolio/project/date)
    context_key = f"ai_assistant_context_{history_key}"
    if context_key not in st.session_state:
        clear_chat_history(history_key)
        st.session_state[context_key] = True

    # Render the chat interface
    render_brief_chat(
        brief_content=data_context,
        history_key=history_key,
        llm_config=llm_config,
        voice_config=voice_config,
        expander_title="Ask questions about your data"
    )

    # Suggested questions
    st.markdown("---")
    st.markdown("##### 💡 Suggested Questions")

    if chat_mode == "Portfolio Overview":
        suggestions = [
            "What is the overall health of the portfolio?",
            "Which projects are at risk and why?",
            "What is the forecasted cost overrun?",
            "Summarize the portfolio performance",
            "Which projects need immediate attention?",
            "Compare the CPI and SPI across projects"
        ]
    else:
        suggestions = [
            "Is this project on track?",
            "What are the main risks for this project?",
            "Explain the cost performance",
            "What is the forecasted completion date?",
            "How much more budget is needed to complete?",
            "What does the TCPI indicate?"
        ]

    cols = st.columns(3)
    for i, suggestion in enumerate(suggestions):
        with cols[i % 3]:
            st.caption(f"• {suggestion}")


if __name__ == "__main__":
    main()

from typing import List, Dict

from datetime import datetime

import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
from utils.auth import check_authentication, require_page_access
from config.constants import USE_DATABASE
from services.data_service import data_manager
from utils.portfolio_context import render_portfolio_context
from services.db_data_service import DatabaseDataManager

# Page configuration
st.set_page_config(
    page_title="Portfolio Gantt Chart",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Check authentication and page access
if not check_authentication():
    st.stop()

require_page_access('portfolio_charts', 'Portfolio Charts')

st.markdown("""
<style>
    .footer {
        text-align: center;
        padding: 2rem;
        color: #718096;
        font-size: 0.9rem;
        margin-top: 3rem;
    }
    .main-header {
        text-align: center;
        color: #2c3e50;
        font-size: 2.2rem;
        font-weight: 600;
        margin-bottom: 1rem;
        border-bottom: 3px solid #3498db;
        padding-bottom: 0.5rem;
    }
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    header {visibility: hidden;}
</style>
""", unsafe_allow_html=True)

COLOR_MAP = {
    "Progress": "#40b57b",  # lighter green
    "Planned": "#4389d1",   # lighter blue
    "Predicted": "#6366f1", # purple for predicted completion
    "Overrun": "#d55454"    # lighter red
}

YEAR_LINE_COLOR = "#374151"  # dark grey for year boundaries
QUARTER_LINE_COLOR = "#9CA3AF"  # light grey for quarter boundaries

PERIOD_OPTIONS = {
    "Month": {"dtick": "M1", "delta": pd.Timedelta(days=30)},
    "Quarter": {"dtick": "M3", "delta": pd.Timedelta(days=91)},
    "Year": {"dtick": "M12", "delta": pd.Timedelta(days=365)}
}


def load_portfolio_dataframe(portfolio_id=None, status_date=None) -> pd.DataFrame | None:
    """Return the latest batch results DataFrame if available."""
    df = None

    # Retrieve stored EVM results
    if USE_DATABASE and portfolio_id and status_date:
        # Database mode: Get EVM results from database
        try:
            db_manager = DatabaseDataManager()
            df = db_manager.get_evm_results_for_period(portfolio_id, status_date)
        except Exception as e:
            import logging
            logging.error(f"Error retrieving EVM results: {e}")
            df = pd.DataFrame()
    else:
        # Session state mode: Use adapter
        adapter = data_manager.get_data_adapter()
        df = adapter.get_batch_results()

    # Fallback: check dashboard_data
    if (df is None or df.empty) and hasattr(st.session_state, "dashboard_data") and st.session_state.dashboard_data is not None:
        # dashboard_data is formatted for display (strings), but use as last resort
        df = st.session_state.dashboard_data.copy()

    if df is None or df.empty:
        return None

    if "error" in df.columns:
        df = df[df["error"].isna()]

    if df.empty:
        return None

    # Apply progress filter if enabled
    from utils.portfolio_context import apply_progress_filter
    # Try common AC column names
    ac_col = 'ac' if 'ac' in df.columns else ('AC' if 'AC' in df.columns else 'Actual Cost')
    if ac_col in df.columns:
        df, _ = apply_progress_filter(df, ac_col)

    if df.empty:
        return None

    return df


def _coerce_datetime(series: pd.Series) -> pd.Series:
    """Convert series to datetime with proper handling of different formats."""
    # First try standard conversion
    converted = pd.to_datetime(series, errors="coerce")

    # For any that failed, try specific date formats that EVM engine uses
    mask = pd.isna(converted) & pd.notna(series)
    if mask.any():
        # Try the EVM engine format: 'dd/mm/yyyy'
        for idx in series[mask].index:
            try:
                if isinstance(series[idx], str) and series[idx].strip():
                    # Try specific format used by EVM engine
                    converted.loc[idx] = pd.to_datetime(series[idx], format='%d/%m/%Y', errors='coerce')
                    if pd.isna(converted.loc[idx]):
                        # Try other common formats
                        converted.loc[idx] = pd.to_datetime(series[idx], format='%Y-%m-%d', errors='coerce')
            except:
                continue

    return converted


def _coerce_numeric(series: pd.Series) -> pd.Series:
    return pd.to_numeric(series, errors="coerce")


def is_valid_finite_number(x):
    """Check if a number is valid and finite."""
    try:
        return pd.notna(x) and np.isfinite(float(x)) and not pd.isna(x)
    except (ValueError, TypeError, OverflowError):
        return False


def format_currency(amount: float, symbol: str = '$', postfix: str = "", decimals: int = 2, thousands: bool = False) -> str:
    """Enhanced currency formatting with comma separators and postfix options."""
    if not is_valid_finite_number(amount):
        return "—"

    # Handle thousands parameter (for cash flow chart compatibility)
    if thousands:
        formatted_amount = f"{float(amount)/1000:.0f}K"
        if postfix:
            return f"{symbol} {formatted_amount} {postfix}"
        else:
            return f"{symbol} {formatted_amount}"

    # Format with comma separators and specified decimal places
    formatted_amount = f"{float(amount):,.{decimals}f}"

    # Map postfix to abbreviations
    postfix_map = {
        "Thousand": "K",
        "Million": "M",
        "Billion": "B"
    }

    if postfix in postfix_map:
        return f"{symbol}{formatted_amount} {postfix_map[postfix]}"
    elif postfix:
        return f"{symbol}{formatted_amount} {postfix}"
    else:
        return f"{symbol}{formatted_amount}"



def apply_filters(df: pd.DataFrame, start_col: str = None, finish_col: str = None) -> pd.DataFrame:
    """Render filter widgets and return the filtered DataFrame with control values."""
    organizations = sorted({str(org) for org in df.get("organization", pd.Series()).dropna() if str(org).strip()})

    numeric_bac = df["bac"].dropna().astype(float) if "bac" in df.columns else pd.Series(dtype=float)
    min_budget = float(numeric_bac.min()) if not numeric_bac.empty else 0.0
    max_budget = float(numeric_bac.max()) if not numeric_bac.empty else min_budget

    numeric_od = df["original_duration_months"].dropna().astype(float) if "original_duration_months" in df.columns else pd.Series(dtype=float)
    min_od = float(numeric_od.min()) if not numeric_od.empty else 0.0
    max_od = float(numeric_od.max()) if not numeric_od.empty else min_od

    # CPI and SPI ranges
    cpi_col = 'cpi' if 'cpi' in df.columns else 'CPI' if 'CPI' in df.columns else None
    spi_col = 'spi' if 'spi' in df.columns else 'SPI' if 'SPI' in df.columns else None

    numeric_cpi = df[cpi_col].dropna().astype(float) if cpi_col and cpi_col in df.columns else pd.Series(dtype=float)
    min_cpi = float(numeric_cpi.min()) if not numeric_cpi.empty else 0.0
    max_cpi = float(numeric_cpi.max()) if not numeric_cpi.empty else 3.0

    numeric_spi = df[spi_col].dropna().astype(float) if spi_col and spi_col in df.columns else pd.Series(dtype=float)
    min_spi = float(numeric_spi.min()) if not numeric_spi.empty else 0.0
    max_spi = float(numeric_spi.max()) if not numeric_spi.empty else 3.0

    min_start = df[start_col].min() if start_col and start_col in df.columns else pd.NaT
    max_start = df[start_col].max() if start_col and start_col in df.columns else pd.NaT
    min_finish = df[finish_col].min() if finish_col and finish_col in df.columns else pd.NaT
    max_finish = df[finish_col].max() if finish_col and finish_col in df.columns else pd.NaT

    with st.expander("🔍 Filters", expanded=False):
        # ═══════════════════════════════════════════════════════════════════
        # ORGANIZATIONAL FILTERS
        # ═══════════════════════════════════════════════════════════════════
        st.markdown("**🏢 Organizational**")
        org_toggle = st.toggle("Filter by Organization", value=False, key="gantt_org_toggle")
        if org_toggle:
            org_selection = st.multiselect(
                "Organization",
                options=organizations,
                default=organizations,
                placeholder="Select organization(s)" if organizations else "No organizations available"
            ) if organizations else []
        else:
            org_selection = organizations

        st.markdown("")  # Spacing

        # ═══════════════════════════════════════════════════════════════════
        # DATE RANGE FILTERS
        # ═══════════════════════════════════════════════════════════════════
        st.markdown("**📅 Date Ranges**")
        col1, col2 = st.columns([1, 1])
        with col1:
            plan_start_later_toggle = st.toggle("Plan Start Later Than", value=False, key="gantt_plan_start_later_toggle") if pd.notna(min_start) else False
            if plan_start_later_toggle and pd.notna(min_start):
                plan_start_later_value = st.date_input(
                    "Plan Start Later Than",
                    value=min_start.date(),
                    min_value=min_start.date(),
                    max_value=max_start.date() if pd.notna(max_start) else min_start.date(),
                    key="gantt_plan_start_later_value"
                )
            else:
                plan_start_later_value = None
        with col2:
            plan_start_earlier_toggle = st.toggle("Plan Start Earlier Than", value=False, key="gantt_plan_start_earlier_toggle") if pd.notna(max_start) else False
            if plan_start_earlier_toggle and pd.notna(max_start):
                plan_start_earlier_value = st.date_input(
                    "Plan Start Earlier Than",
                    value=max_start.date(),
                    min_value=min_start.date() if pd.notna(min_start) else max_start.date(),
                    max_value=max_start.date(),
                    key="gantt_plan_start_earlier_value"
                )
            else:
                plan_start_earlier_value = None

        col1, col2 = st.columns([1, 1])
        with col1:
            plan_finish_later_toggle = st.toggle("Plan Finish Later Than", value=False, key="gantt_plan_finish_later_toggle") if pd.notna(min_finish) else False
            if plan_finish_later_toggle and pd.notna(min_finish):
                plan_finish_later_value = st.date_input(
                    "Plan Finish Later Than",
                    value=min_finish.date(),
                    min_value=min_finish.date(),
                    max_value=max_finish.date() if pd.notna(max_finish) else min_finish.date(),
                    key="gantt_plan_finish_later_value"
                )
            else:
                plan_finish_later_value = None
        with col2:
            plan_finish_earlier_toggle = st.toggle("Plan Finish Earlier Than", value=False, key="gantt_plan_finish_earlier_toggle") if pd.notna(max_finish) else False
            if plan_finish_earlier_toggle and pd.notna(max_finish):
                plan_finish_earlier_value = st.date_input(
                    "Plan Finish Earlier Than",
                    value=max_finish.date(),
                    min_value=min_finish.date() if pd.notna(min_finish) else max_finish.date(),
                    max_value=max_finish.date(),
                    key="gantt_plan_finish_earlier_value"
                )
            else:
                plan_finish_earlier_value = None

        st.markdown("")  # Spacing

        # ═══════════════════════════════════════════════════════════════════
        # PROJECT CHARACTERISTICS
        # ═══════════════════════════════════════════════════════════════════
        st.markdown("**📊 Project Characteristics**")
        col1, col2 = st.columns([1, 1])
        with col1:
            min_budget_toggle = st.toggle("Set Min Budget", value=False, key="gantt_min_budget_toggle")
            if min_budget_toggle:
                min_budget_value = st.number_input(
                    "Min Budget",
                    value=min_budget,
                    step=max(1.0, (max_budget - min_budget) / 10) if max_budget > min_budget else 1.0,
                    min_value=0.0,
                    key="gantt_min_budget_value"
                )
            else:
                min_budget_value = min_budget
        with col2:
            max_budget_toggle = st.toggle("Set Max Budget", value=False, key="gantt_max_budget_toggle")
            if max_budget_toggle:
                max_budget_value = st.number_input(
                    "Max Budget",
                    value=max_budget,
                    step=max(1.0, (max_budget - min_budget) / 10) if max_budget > min_budget else 1.0,
                    min_value=0.0,
                    key="gantt_max_budget_value"
                )
            else:
                max_budget_value = max_budget

        col1, col2 = st.columns([1, 1])
        with col1:
            od_min_toggle = st.toggle("Set Min Duration", value=False, key="gantt_od_min_toggle", help="Filter by minimum Original Duration (months)")
            if od_min_toggle:
                od_min_value = st.number_input(
                    "Min Duration (months)",
                    value=min_od,
                    step=max(1.0, (max_od - min_od) / 10) if max_od > min_od else 1.0,
                    min_value=0.0,
                    key="gantt_od_min_value"
                )
            else:
                od_min_value = min_od
        with col2:
            od_max_toggle = st.toggle("Set Max Duration", value=False, key="gantt_od_max_toggle", help="Filter by maximum Original Duration (months)")
            if od_max_toggle:
                od_max_value = st.number_input(
                    "Max Duration (months)",
                    value=max_od,
                    step=max(1.0, (max_od - min_od) / 10) if max_od > min_od else 1.0,
                    min_value=0.0,
                    key="gantt_od_max_value"
                )
            else:
                od_max_value = max_od

        st.markdown("")  # Spacing

        # ═══════════════════════════════════════════════════════════════════
        # PERFORMANCE INDICES
        # ═══════════════════════════════════════════════════════════════════
        st.markdown("**📈 Performance Indices**")
        col1, col2 = st.columns([1, 1])
        with col1:
            cpi_filter_toggle = st.toggle("Filter by CPI Range", value=False, key="gantt_cpi_filter_toggle", help="Cost Performance Index: >1.0 = under budget, <1.0 = over budget")
            if cpi_filter_toggle and cpi_col:
                cpi_range = st.slider(
                    "CPI Range",
                    min_value=0.0,
                    max_value=3.0,
                    value=(min_cpi, max_cpi),
                    step=0.01,
                    key="gantt_cpi_range_value"
                )
            else:
                cpi_range = (0.0, 3.0)
        with col2:
            spi_filter_toggle = st.toggle("Filter by SPI Range", value=False, key="gantt_spi_filter_toggle", help="Schedule Performance Index: >1.0 = ahead of schedule, <1.0 = behind schedule")
            if spi_filter_toggle and spi_col:
                spi_range = st.slider(
                    "SPI Range",
                    min_value=0.0,
                    max_value=3.0,
                    value=(min_spi, max_spi),
                    step=0.01,
                    key="gantt_spi_range_value"
                )
            else:
                spi_range = (0.0, 3.0)

    # Apply filters
    filtered = df.copy()

    # Organization filter
    if org_toggle and organizations and org_selection:
        filtered = filtered[filtered["organization"].isin(org_selection)]

    # Plan Start date filters
    if plan_start_later_value is not None and start_col:
        plan_start_later_dt = pd.to_datetime(plan_start_later_value)
        filtered = filtered[filtered[start_col] >= plan_start_later_dt]

    if plan_start_earlier_value is not None and start_col:
        plan_start_earlier_dt = pd.to_datetime(plan_start_earlier_value)
        filtered = filtered[filtered[start_col] <= plan_start_earlier_dt]

    # Plan Finish date filters
    if plan_finish_later_value is not None and finish_col:
        plan_finish_later_dt = pd.to_datetime(plan_finish_later_value)
        filtered = filtered[filtered[finish_col] >= plan_finish_later_dt]

    if plan_finish_earlier_value is not None and finish_col:
        plan_finish_earlier_dt = pd.to_datetime(plan_finish_earlier_value)
        filtered = filtered[filtered[finish_col] <= plan_finish_earlier_dt]

    # Original Duration filters
    if "original_duration_months" in filtered.columns:
        if od_min_toggle:
            filtered = filtered[filtered["original_duration_months"] >= od_min_value]
        if od_max_toggle:
            filtered = filtered[filtered["original_duration_months"] <= od_max_value]

    # Budget filters
    if "bac" in filtered.columns:
        if min_budget_toggle:
            filtered = filtered[filtered["bac"] >= min_budget_value]
        if max_budget_toggle:
            filtered = filtered[filtered["bac"] <= max_budget_value]

    # CPI filter
    if cpi_filter_toggle and cpi_col and cpi_col in filtered.columns:
        filtered = filtered[(filtered[cpi_col] >= cpi_range[0]) & (filtered[cpi_col] <= cpi_range[1])]

    # SPI filter
    if spi_filter_toggle and spi_col and spi_col in filtered.columns:
        filtered = filtered[(filtered[spi_col] >= spi_range[0]) & (filtered[spi_col] <= spi_range[1])]

    return filtered



def build_segments(df: pd.DataFrame, show_predicted: bool) -> List[Dict]:
    segments: List[Dict] = []
    for _, row in df.iterrows():
        start = row.get("plan_start", row.get("Plan Start"))
        finish = row.get("plan_finish", row.get("Plan Finish"))
        if pd.isna(start) or pd.isna(finish):
            continue

        project_id = str(row.get("project_id", row.get("Project ID", ""))) or "Unknown"
        project_name = row.get("project_name", row.get("Project Name", ""))
        organization = row.get("organization", row.get("Organization", ""))

        bac = row.get("bac", row.get("BAC", 0.0))
        ac = row.get("ac", row.get("AC", 0.0))
        earned_value = row.get("ev", row.get("EV", 0.0))
        cpi = row.get("cpi", row.get("CPI", 0.0))
        spi = row.get("spi", row.get("SPI", 0.0))
        actual_duration = row.get("actual_duration_months", 0.0)
        original_duration = row.get("original_duration_months", 0.0)

        # Calculate percentages
        percent_budget_used = (ac / bac * 100) if is_valid_finite_number(bac) and bac > 0 else 0.0
        percent_time_used = (actual_duration / original_duration * 100) if is_valid_finite_number(original_duration) and original_duration > 0 else 0.0
        percent_work_completed = (earned_value / bac * 100) if is_valid_finite_number(bac) and bac > 0 else 0.0

        # Get currency settings from session state
        currency_symbol = (
            getattr(st.session_state, 'dashboard_currency_symbol', None) or
            getattr(st.session_state, 'currency_symbol', None) or
            st.session_state.get('config_dict', {}).get('controls', {}).get('currency_symbol', '$')
        )
        currency_postfix = (
            getattr(st.session_state, 'dashboard_currency_postfix', None) or
            getattr(st.session_state, 'currency_postfix', None) or
            st.session_state.get('config_dict', {}).get('controls', {}).get('currency_postfix', '')
        )

        # Format values for tooltip
        bac_formatted = format_currency(bac, currency_symbol, currency_postfix)
        plan_start_str = start.strftime('%Y-%m-%d') if pd.notna(start) else "N/A"
        plan_finish_str = finish.strftime('%Y-%m-%d') if pd.notna(finish) else "N/A"
        if pd.isna(bac) or bac <= 0:
            progress_ratio = 0.0
        else:
            progress_ratio = max(0.0, min(float(earned_value) / float(bac), 1.0))

        # Get OD (Original Duration) and LD (Likely Duration)
        od = original_duration
        ld = row.get("ld", row.get("likely_duration", 0.0))

        # Calculate dates based on OD, LD, and % Work Completed
        # Green segment: From Plan Start to (Plan Start + OD * % Work Completed)
        if is_valid_finite_number(od) and od > 0 and is_valid_finite_number(percent_work_completed):
            # Convert percent to decimal (e.g., 45% -> 0.45)
            work_completed_decimal = percent_work_completed / 100.0
            # Calculate progress duration in days
            od_days = pd.Timedelta(days=od * 30.44)  # Convert months to days (avg 30.44 days/month)
            progress_duration = od_days * work_completed_decimal
            progress_end = start + progress_duration
        else:
            # Fallback: use plan finish if OD or % work completed is not available
            progress_end = start

        # Ensure progress_end is within bounds
        if progress_end < start:
            progress_end = start
        if progress_end > finish:
            progress_end = finish

        # Add Progress segment (Green)
        segments.append({
            "Task": project_id,
            "Start": start,
            "Finish": progress_end,
            "Segment": "Progress",
            "project_name": project_name,
            "organization": organization,
            "bac_formatted": bac_formatted,
            "plan_start": plan_start_str,
            "plan_finish": plan_finish_str,
            "cpi": cpi,
            "spi": spi,
            "percent_budget_used": percent_budget_used,
            "percent_time_used": percent_time_used,
            "percent_work_completed": percent_work_completed
        })

        # Handle the remaining timeline based on view mode (Plan vs Predicted)
        if show_predicted and is_valid_finite_number(od) and od > 0 and is_valid_finite_number(ld) and ld > 0:
            # Predicted View: Use OD and LD to calculate segments

            # Blue segment: from (Plan Start + OD * % Work Completed) to Plan Finish
            # This represents the original planned remaining work
            if progress_end < finish:
                segments.append({
                    "Task": project_id,
                    "Start": progress_end,
                    "Finish": finish,
                    "Segment": "Planned",
                    "project_name": project_name,
                    "organization": organization,
                    "bac_formatted": bac_formatted,
                    "plan_start": plan_start_str,
                    "plan_finish": plan_finish_str,
                    "cpi": cpi,
                    "spi": spi,
                    "percent_budget_used": percent_budget_used,
                    "percent_time_used": percent_time_used,
                    "percent_work_completed": percent_work_completed
                })

            # Red segment: (if LD > OD) From Plan Start + OD to Plan Start + LD
            # This represents the schedule overrun
            if ld > od:
                od_end = start + pd.Timedelta(days=od * 30.44)
                ld_end = start + pd.Timedelta(days=ld * 30.44)

                segments.append({
                    "Task": project_id,
                    "Start": od_end,
                    "Finish": ld_end,
                    "Segment": "Overrun",
                    "project_name": project_name,
                    "organization": organization,
                    "bac_formatted": bac_formatted,
                    "plan_start": plan_start_str,
                    "plan_finish": plan_finish_str,
                    "cpi": cpi,
                    "spi": spi,
                    "percent_budget_used": percent_budget_used,
                    "percent_time_used": percent_time_used,
                    "percent_work_completed": percent_work_completed
                })
        else:
            # Plan View: Show planned completion (simple two-segment view)
            if progress_end < finish:
                segments.append({
                    "Task": project_id,
                    "Start": progress_end,
                    "Finish": finish,
                    "Segment": "Planned",
                    "project_name": project_name,
                    "organization": organization,
                    "bac_formatted": bac_formatted,
                    "plan_start": plan_start_str,
                    "plan_finish": plan_finish_str,
                    "cpi": cpi,
                    "spi": spi,
                    "percent_budget_used": percent_budget_used,
                    "percent_time_used": percent_time_used,
                    "percent_work_completed": percent_work_completed
                })

    return segments


def render_cash_flow_chart(filtered_df: pd.DataFrame, start_col: str = None, finish_col: str = None) -> None:
    """Render the cash flow chart with all controls and visualizations."""
    if len(filtered_df) > 0:
        # Get currency settings from session state
        currency_symbol = (
            getattr(st.session_state, 'dashboard_currency_symbol', None) or
            getattr(st.session_state, 'currency_symbol', None) or
            st.session_state.get('config_dict', {}).get('controls', {}).get('currency_symbol', '$')
        )
        currency_postfix = (
            getattr(st.session_state, 'dashboard_currency_postfix', None) or
            getattr(st.session_state, 'currency_postfix', None) or
            st.session_state.get('config_dict', {}).get('controls', {}).get('currency_postfix', '')
        )

        # Detect possible date columns
        date_columns = []
        for col in filtered_df.columns:
            if any(keyword in col.lower() for keyword in ['start', 'begin', 'finish', 'end', 'complete', 'date']):
                date_columns.append(col)

        # Look for specific date column patterns
        start_date_col = start_col
        plan_finish_col = finish_col
        likely_finish_col = None
        expected_finish_col = None

        for col in date_columns:
            col_lower = col.lower()
            if 'expected' in col_lower and ('finish' in col_lower or 'end' in col_lower or 'complete' in col_lower):
                expected_finish_col = col
            elif 'likely' in col_lower and ('finish' in col_lower or 'end' in col_lower or 'complete' in col_lower):
                likely_finish_col = col

        if start_date_col and (plan_finish_col or likely_finish_col or expected_finish_col):
            # Validate expected finish dates (max 4 years from plan finish)
            valid_expected_finish_col = None
            expected_date_info = ""
            if expected_finish_col and plan_finish_col:
                try:
                    # Parse dates for validation
                    temp_df = filtered_df.copy()
                    temp_df[plan_finish_col] = pd.to_datetime(temp_df[plan_finish_col], errors='coerce')
                    temp_df[expected_finish_col] = pd.to_datetime(temp_df[expected_finish_col], errors='coerce')

                    # Check if expected dates are within 4 years of plan dates
                    valid_rows = temp_df.dropna(subset=[plan_finish_col, expected_finish_col])
                    if len(valid_rows) > 0:
                        date_diff_years = (valid_rows[expected_finish_col] - valid_rows[plan_finish_col]).dt.days / 365.25
                        valid_dates = (date_diff_years <= 4) & (date_diff_years >= -1)
                        if valid_dates.all():
                            valid_expected_finish_col = expected_finish_col
                            expected_date_info = f"✅ Expected dates validated ({len(valid_rows)} projects)"
                        else:
                            invalid_count = (~valid_dates).sum()
                            expected_date_info = f"⚠️ Expected dates excluded ({invalid_count} projects exceed 4-year limit)"
                    else:
                        expected_date_info = "⚠️ No valid expected dates found"
                except Exception as e:
                    expected_date_info = f"❌ Expected date validation failed: {str(e)}"

            # Controls for cash flow chart
            col1, col2, col3, col4 = st.columns(4)

            with col1:
                time_period = st.selectbox("Time Period",
                                           options=["Month", "Quarter", "FY"],
                                           index=0,
                                           key="cash_flow_time_period",
                                           help="Month: Monthly view, Quarter: Quarterly view, FY: Financial Year (July-June)")

            with col2:
                # Determine available finish date options (prioritize expected over likely)
                finish_options = []
                if plan_finish_col:
                    finish_options.append("Plan Finish")
                if valid_expected_finish_col:
                    finish_options.append("Expected Finish")
                elif likely_finish_col:  # Only show likely if expected is not available
                    finish_options.append("Likely Finish")

                if len(finish_options) > 1:
                    finish_date_choice = st.selectbox("Finish Date Type", finish_options, key="cash_flow_finish_type")
                else:
                    finish_date_choice = finish_options[0] if finish_options else "Plan Finish"
                    st.write(f"**Finish Date:** {finish_date_choice}")

            with col3:
                cash_flow_type = st.radio("Cash Flow Type",
                                          options=["Plan", "Predicted", "Actual", "All"],
                                          index=0,
                                          key="cash_flow_type",
                                          help="Plan: BAC + OD, Predicted: BAC + LD, Actual: AC from Start to Data Date, All: Line chart comparing all three")

            with col4:
                st.write("**Configuration:**")
                st.write(f"📊 {time_period}")
                if cash_flow_type == "Plan":
                    st.write("💰 BAC/OD")
                elif cash_flow_type == "Predicted":
                    st.write("💰 BAC/LD")
                elif cash_flow_type == "Actual":
                    st.write("💰 AC to Data Date")
                else:  # All
                    st.write("💰 Plan vs Predicted vs Actual")

            # Show expected date validation info if available
            if expected_date_info:
                st.info(expected_date_info)

            # Select finish date column based on choice
            if finish_date_choice == "Expected Finish" and valid_expected_finish_col:
                finish_col_selected = valid_expected_finish_col
            elif finish_date_choice == "Likely Finish" and likely_finish_col:
                finish_col_selected = likely_finish_col
            else:
                finish_col_selected = plan_finish_col

            if finish_col_selected:
                try:
                    # Convert date columns to datetime
                    df_cash = filtered_df.copy()

                    # Parse dates with error handling
                    df_cash[start_date_col] = pd.to_datetime(df_cash[start_date_col], errors='coerce')
                    df_cash[finish_col_selected] = pd.to_datetime(df_cash[finish_col_selected], errors='coerce')

                    # Remove rows with invalid dates
                    df_cash = df_cash.dropna(subset=[start_date_col, finish_col_selected])

                    if len(df_cash) > 0:
                        def get_financial_year(date):
                            """Get financial year string for a date (FY starts July 1st)"""
                            if date.month >= 7:  # July onwards = start of FY
                                return f"FY{date.year + 1}"  # e.g., July 2024 = FY2025
                            else:  # Jan-June = end of previous FY
                                return f"FY{date.year}"  # e.g., March 2024 = FY2024

                        def get_period_key(date, time_period):
                            """Get period key based on time period selection"""
                            if time_period == "Month":
                                return date.strftime("%Y-%b")
                            elif time_period == "Quarter":
                                quarter = f"{date.year}-Q{((date.month - 1) // 3) + 1}"
                                return quarter
                            else:  # FY
                                return get_financial_year(date)

                        def calculate_cash_flow_for_scenario(df_cash, scenario_type):
                            """Calculate cash flow for Plan (BAC/OD) or Predicted (BAC/LD) scenario"""
                            cash_flow_data = []

                            for idx, row in df_cash.iterrows():
                                start_date = row[start_date_col]

                                if pd.notna(start_date):
                                    # Ensure start_date is a proper datetime object
                                    if not isinstance(start_date, pd.Timestamp):
                                        start_date = pd.to_datetime(start_date, errors='coerce')
                                        if pd.isna(start_date):
                                            continue

                                    # Always use BAC (Budget) for both scenarios
                                    budget = row.get('bac', row.get('Budget', 0))  # Try 'bac' first, then 'Budget'

                                    if scenario_type == "Plan":
                                        # Use Original Duration (OD)
                                        if 'original_duration_months' in row and pd.notna(row.get('original_duration_months')):
                                            duration_months = max(1, row['original_duration_months'])
                                        else:
                                            # Fallback: calculate from plan start to plan finish
                                            plan_finish = row.get('plan_finish', row.get('Plan Finish'))
                                            if pd.notna(plan_finish):
                                                plan_finish_date = pd.to_datetime(plan_finish, errors='coerce')
                                                if pd.notna(plan_finish_date):
                                                    duration_months = max(1, (plan_finish_date - start_date).days / 30.44)
                                                else:
                                                    continue
                                            else:
                                                continue
                                    else:  # Predicted
                                        # Use Likely Duration (LD) with cap check
                                        # Try multiple possible column names for LD
                                        ld = None
                                        if 'likely_duration' in row and pd.notna(row.get('likely_duration')):
                                            ld = row['likely_duration']
                                        elif 'ld' in row and pd.notna(row.get('ld')):
                                            ld = row['ld']
                                        elif 'Likely Duration' in row and pd.notna(row.get('Likely Duration')):
                                            ld = row['Likely Duration']

                                        if ld is not None:
                                            # Get OD for cap calculation
                                            if 'original_duration_months' in row and pd.notna(row.get('original_duration_months')):
                                                od = row['original_duration_months']
                                            else:
                                                # Fallback: calculate OD from plan dates
                                                plan_finish = row.get('plan_finish', row.get('Plan Finish'))
                                                if pd.notna(plan_finish):
                                                    plan_finish_date = pd.to_datetime(plan_finish, errors='coerce')
                                                    if pd.notna(plan_finish_date):
                                                        od = max(1, (plan_finish_date - start_date).days / 30.44)
                                                    else:
                                                        od = 12  # Default fallback
                                                else:
                                                    od = 12  # Default fallback

                                            # Cap LD to prevent timestamp overflow: min(LD, OD+48)
                                            duration_months = max(1, min(ld, od + 48))
                                        else:
                                            continue

                                    if budget > 0 and duration_months > 0:
                                        # Calculate monthly cash flow: BAC/Duration
                                        monthly_cash_flow = budget / duration_months

                                        # Generate monthly cash flow from plan start for duration months
                                        current_date = start_date.replace(day=1)

                                        for month in range(int(duration_months)):
                                            period_key = get_period_key(current_date, time_period)

                                            cash_flow_data.append({
                                                'Period': period_key,
                                                'Cash_Flow': monthly_cash_flow,
                                                'Project': row.get('project_name', row.get('Project Name', 'Unknown')),
                                                'Date': current_date,
                                                'Scenario': scenario_type
                                            })

                                            # Move to next month
                                            if current_date.month == 12:
                                                current_date = current_date.replace(year=current_date.year + 1, month=1)
                                            else:
                                                current_date = current_date.replace(month=current_date.month + 1)
                            return cash_flow_data

                        def calculate_actual_cash_flow(df_cash):
                            """Calculate actual cash flow using AC/AD from Plan Start for AD months"""
                            cash_flow_data = []

                            for idx, row in df_cash.iterrows():
                                start_date = row[start_date_col]

                                if pd.notna(start_date):
                                    # Ensure start_date is a proper datetime object
                                    if not isinstance(start_date, pd.Timestamp):
                                        start_date = pd.to_datetime(start_date, errors='coerce')
                                        if pd.isna(start_date):
                                            continue

                                    # Get AC (Actual Cost)
                                    ac = row.get('ac', row.get('AC', row.get('Actual Cost', 0)))

                                    # Get AD (Actual Duration)
                                    if 'actual_duration_months' in row and pd.notna(row.get('actual_duration_months')):
                                        duration_months = max(1, row['actual_duration_months'])
                                    else:
                                        # Fallback: calculate from plan start to data date
                                        data_date_col = None
                                        for col in df_cash.columns:
                                            if 'data' in col.lower() and 'date' in col.lower():
                                                data_date_col = col
                                                break

                                        if data_date_col:
                                            data_date = row.get(data_date_col)
                                            if pd.notna(data_date):
                                                data_date_parsed = pd.to_datetime(data_date, errors='coerce')
                                                if pd.notna(data_date_parsed):
                                                    duration_months = max(1, (data_date_parsed - start_date).days / 30.44)
                                                else:
                                                    continue
                                            else:
                                                continue
                                        else:
                                            continue

                                    if ac > 0 and duration_months > 0:
                                        # Calculate monthly cash flow: AC/AD
                                        monthly_cash_flow = ac / duration_months

                                        # Generate monthly cash flow from plan start for AD months
                                        current_date = start_date.replace(day=1)

                                        for month in range(int(duration_months)):
                                            period_key = get_period_key(current_date, time_period)

                                            cash_flow_data.append({
                                                'Period': period_key,
                                                'Cash_Flow': monthly_cash_flow,
                                                'Project': row.get('project_name', row.get('Project Name', 'Unknown')),
                                                'Date': current_date,
                                                'Scenario': 'Actual'
                                            })

                                            # Move to next month
                                            if current_date.month == 12:
                                                current_date = current_date.replace(year=current_date.year + 1, month=1)
                                            else:
                                                current_date = current_date.replace(month=current_date.month + 1)

                            return cash_flow_data

                        # Calculate cash flow based on selected type
                        if cash_flow_type == "Plan":
                            cash_flow_data = calculate_cash_flow_for_scenario(df_cash, "Plan")
                        elif cash_flow_type == "Predicted":
                            cash_flow_data = calculate_cash_flow_for_scenario(df_cash, "Predicted")
                        elif cash_flow_type == "Actual":
                            cash_flow_data = calculate_actual_cash_flow(df_cash)
                        else:  # All
                            plan_data = calculate_cash_flow_for_scenario(df_cash, "Plan")
                            predicted_data = calculate_cash_flow_for_scenario(df_cash, "Predicted")
                            actual_data = calculate_actual_cash_flow(df_cash)
                            # Ensure all are lists before concatenating
                            if not isinstance(plan_data, list):
                                plan_data = []
                            if not isinstance(predicted_data, list):
                                predicted_data = []
                            if not isinstance(actual_data, list):
                                actual_data = []
                            cash_flow_data = plan_data + predicted_data + actual_data

                        if cash_flow_data:
                            cash_df = pd.DataFrame(cash_flow_data)

                            # Ensure Cash_Flow column is numeric and handle any data type issues
                            try:
                                cash_df['Cash_Flow'] = pd.to_numeric(cash_df['Cash_Flow'], errors='coerce')
                                # Remove any rows with invalid cash flow values
                                cash_df = cash_df.dropna(subset=['Cash_Flow'])
                                cash_df = cash_df[cash_df['Cash_Flow'].notna() & (cash_df['Cash_Flow'] != float('inf')) & (cash_df['Cash_Flow'] != float('-inf'))]

                                # Check if we have valid data after cleaning
                                if cash_df.empty:
                                    st.warning("No valid cash flow data available after processing.")
                                    cash_df = pd.DataFrame()
                            except Exception as e:
                                st.error(f"Error processing cash flow data types: {str(e)}")
                                cash_df = pd.DataFrame()  # Empty dataframe to prevent further errors

                            def get_sort_key(period_str, time_period):
                                """Generate sort key for different time periods"""
                                if time_period == "Month":
                                    # For "2024-Jan" format
                                    try:
                                        year, month_abbr = period_str.split('-')
                                        month_num = {
                                            'Jan': 1, 'Feb': 2, 'Mar': 3, 'Apr': 4, 'May': 5, 'Jun': 6,
                                            'Jul': 7, 'Aug': 8, 'Sep': 9, 'Oct': 10, 'Nov': 11, 'Dec': 12
                                        }.get(month_abbr, 1)
                                        return (int(year), month_num)
                                    except:
                                        return (2000, 1)
                                elif time_period == "Quarter":
                                    # For "2024-Q1" format
                                    try:
                                        year, quarter = period_str.split('-')
                                        quarter_num = int(quarter[1:])  # Extract number from Q1, Q2, etc.
                                        return (int(year), quarter_num)
                                    except:
                                        return (2000, 1)
                                else:  # FY
                                    # For "FY2024" format
                                    try:
                                        return (int(period_str[2:]), 1)  # Extract year from FY2024
                                    except:
                                        return (2000, 1)

                            # Only proceed if we have valid cash flow data
                            if not cash_df.empty and len(cash_df) > 0:
                                if cash_flow_type == "All":
                                    # For All option, create line chart with three series
                                    period_cash_flow = cash_df.groupby(['Period', 'Scenario'])['Cash_Flow'].sum().reset_index()
                                    period_cash_flow['Sort_Key'] = period_cash_flow['Period'].apply(
                                        lambda x: get_sort_key(x, time_period)
                                    )
                                    period_cash_flow = period_cash_flow.sort_values('Sort_Key').drop('Sort_Key', axis=1)

                                    # For Actual in all views, remove the last point to create horizontal line effect
                                    actual_mask = period_cash_flow['Scenario'] == 'Actual'
                                    actual_data = period_cash_flow[actual_mask]
                                    if len(actual_data) > 1:
                                        # Remove the last point of Actual data
                                        last_actual_index = actual_data.index[-1]
                                        period_cash_flow = period_cash_flow.drop(last_actual_index)

                                    # Add final points for Plan and Predicted scenarios
                                    for scenario in period_cash_flow['Scenario'].unique():
                                        if scenario == 'Actual':
                                            # Skip Actual - no additional points needed
                                            continue

                                        scenario_data = period_cash_flow[period_cash_flow['Scenario'] == scenario]
                                        if not scenario_data.empty:
                                            last_period = scenario_data.iloc[-1]['Period']

                                            # For Plan and Predicted, add zero point at next period
                                            # Generate next period based on time_period type
                                            if time_period == "Month":
                                                # For Month: increment month (format: "2024-Jan")
                                                year, month_abbr = last_period.split('-')
                                                month_num = {'Jan': 1, 'Feb': 2, 'Mar': 3, 'Apr': 4, 'May': 5, 'Jun': 6,
                                                            'Jul': 7, 'Aug': 8, 'Sep': 9, 'Oct': 10, 'Nov': 11, 'Dec': 12}.get(month_abbr, 1)
                                                if month_num == 12:
                                                    next_period = f"{int(year) + 1}-Jan"
                                                else:
                                                    # month_num is 1-12, so we need index month_num (which gives us the next month)
                                                    month_names = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']
                                                    next_month = month_names[month_num]  # month_num is the current month (1-12), so index month_num gives next month
                                                    next_period = f"{year}-{next_month}"
                                            elif time_period == "Quarter":
                                                # For Quarter: increment quarter (format: "2024-Q1")
                                                year, quarter = last_period.split('-')
                                                quarter_num = int(quarter[1:])
                                                if quarter_num == 4:
                                                    next_period = f"{int(year) + 1}-Q1"
                                                else:
                                                    next_period = f"{year}-Q{quarter_num + 1}"
                                            else:  # FY
                                                # For FY: increment year
                                                fy_year = int(last_period[2:])
                                                next_period = f"FY{fy_year + 1}"

                                            # Add the zero point for Plan and Predicted
                                            final_row = pd.DataFrame({
                                                'Period': [next_period],
                                                'Scenario': [scenario],
                                                'Cash_Flow': [0]
                                            })
                                            period_cash_flow = pd.concat([period_cash_flow, final_row], ignore_index=True)

                                    # Re-sort after adding zero points
                                    period_cash_flow['Sort_Key'] = period_cash_flow['Period'].apply(
                                        lambda x: get_sort_key(x, time_period)
                                    )
                                    period_cash_flow = period_cash_flow.sort_values('Sort_Key').drop('Sort_Key', axis=1)

                                    chart_title = f"Portfolio Cash Flow Comparison (Plan vs Predicted vs Actual) - {time_period} View"

                                    # Define color mapping for scenarios
                                    color_map = {
                                        'Plan': 'blue',
                                        'Predicted': 'orange',
                                        'Actual': 'green'
                                    }

                                    fig_cash_flow = px.line(
                                        period_cash_flow,
                                        x='Period',
                                        y='Cash_Flow',
                                        color='Scenario',
                                        title=chart_title,
                                        labels={
                                            'Cash_Flow': f'Cash Flow ({currency_symbol})',
                                            'Period': 'Period',
                                            'Scenario': 'Scenario'
                                        },
                                        line_shape='spline',  # Makes the line smooth
                                        markers=True,
                                        color_discrete_map=color_map
                                    )

                                    # Reduce marker size to make them less distracting
                                    fig_cash_flow.update_traces(marker=dict(size=4))

                                    # Add vertical line at data date
                                    # Find the data date column and calculate average data date
                                    data_date_col = None
                                    for col in df_cash.columns:
                                        if 'data' in col.lower() and 'date' in col.lower():
                                            data_date_col = col
                                            break

                                    if data_date_col:
                                        # Get the most common data date (or average)
                                        valid_data_dates = df_cash[data_date_col].dropna()
                                        if len(valid_data_dates) > 0:
                                            # Use the most recent data date
                                            data_date_value = pd.to_datetime(valid_data_dates).max()
                                            if pd.notna(data_date_value):
                                                # Get the period key for the data date
                                                data_date_period = get_period_key(data_date_value, time_period)

                                                # Add vertical line at data date using add_shape (works with categorical x-axis)
                                                fig_cash_flow.add_shape(
                                                    type="line",
                                                    x0=data_date_period,
                                                    x1=data_date_period,
                                                    y0=0,
                                                    y1=1,
                                                    yref="paper",
                                                    line=dict(
                                                        color="red",
                                                        width=2,
                                                        dash="dash"
                                                    )
                                                )
                                                # Add annotation for the line
                                                fig_cash_flow.add_annotation(
                                                    x=data_date_period,
                                                    y=1,
                                                    yref="paper",
                                                    text="Data Date",
                                                    showarrow=False,
                                                    yshift=10,
                                                    font=dict(color="red")
                                                )
                                else:
                                    # For Plan, Predicted, or Actual, create bar chart
                                    period_cash_flow = cash_df.groupby('Period')['Cash_Flow'].sum().reset_index()
                                    period_cash_flow['Sort_Key'] = period_cash_flow['Period'].apply(
                                        lambda x: get_sort_key(x, time_period)
                                    )
                                    period_cash_flow = period_cash_flow.sort_values('Sort_Key').drop('Sort_Key', axis=1)

                                    if cash_flow_type == 'Plan':
                                        scenario_label = "(Plan: BAC/OD)"
                                    elif cash_flow_type == 'Predicted':
                                        scenario_label = "(Predicted: BAC/LD)"
                                    else:  # Actual
                                        scenario_label = "(Actual: AC to Data Date)"
                                    chart_title = f"Portfolio Cash Flow {scenario_label} - {time_period} View"

                                    # Choose color scale based on cash flow type
                                    if cash_flow_type == 'Actual':
                                        color_scale = 'greens'
                                    else:
                                        color_scale = 'blues'

                                    fig_cash_flow = px.bar(
                                        period_cash_flow,
                                        x='Period',
                                        y='Cash_Flow',
                                        title=chart_title,
                                        labels={
                                            'Cash_Flow': f'Cash Flow ({currency_symbol})',
                                            'Period': time_period
                                        },
                                        color='Cash_Flow',
                                        color_continuous_scale=color_scale
                                    )

                                # Update layout for better visualization
                                fig_cash_flow.update_layout(
                                    height=500,
                                    showlegend=True if cash_flow_type == "All" else False,
                                    xaxis=dict(
                                        title=time_period,
                                        tickangle=45
                                    ),
                                    yaxis=dict(
                                        title=f'Cash Flow ({currency_symbol}{" " + currency_postfix if currency_postfix else ""})',
                                        tickformat=',.0f'
                                    ),
                                    coloraxis_showscale=False if cash_flow_type != "All" else True
                                )

                                # Update traces for better appearance
                                if cash_flow_type != "All":
                                    fig_cash_flow.update_traces(
                                        texttemplate='%{y:,.0f}',
                                        textposition='outside'
                                    )

                                st.plotly_chart(fig_cash_flow, width="stretch")

                                # Display summary metrics
                                col1, col2 = st.columns(2)
                                with col1:
                                    if cash_flow_type != "Both":
                                        try:
                                            avg_monthly = period_cash_flow['Cash_Flow'].mean()
                                            if pd.isna(avg_monthly) or avg_monthly == float('inf') or avg_monthly == float('-inf'):
                                                st.metric("Average per Period", "N/A")
                                            else:
                                                st.metric("Average per Period", format_currency(avg_monthly, currency_symbol, currency_postfix, thousands=False))
                                        except Exception as e:
                                            st.metric("Average per Period", "Error")
                                            st.error(f"Error calculating average: {str(e)}")

                                with col2:
                                    if cash_flow_type != "Both":
                                        try:
                                            if len(period_cash_flow) > 0 and not period_cash_flow['Cash_Flow'].empty:
                                                peak_amount = period_cash_flow['Cash_Flow'].max()
                                                peak_period = period_cash_flow.loc[period_cash_flow['Cash_Flow'].idxmax(), 'Period']
                                                if pd.isna(peak_amount) or peak_amount == float('inf') or peak_amount == float('-inf'):
                                                    st.metric("Peak Period", "N/A")
                                                else:
                                                    st.metric(f"Peak Period: {peak_period}", format_currency(peak_amount, currency_symbol, currency_postfix, thousands=False))
                                            else:
                                                st.metric("Peak Period", "No Data")
                                        except Exception as e:
                                            st.metric("Peak Period", "Error")
                                            st.error(f"Error calculating peak: {str(e)}")

                                # Show detailed data table
                                with st.expander("📊 Detailed Cash Flow Data", expanded=False):
                                    if cash_flow_type == "All":
                                        # Show comparison table for all three scenarios
                                        # Create pivot table with numeric data first
                                        pivot_df = period_cash_flow.pivot_table(index='Period', columns='Scenario', values='Cash_Flow', aggfunc='sum', fill_value=0)

                                        # Reorder columns to: Plan, Actual, Predicted
                                        column_order = []
                                        if 'Plan' in pivot_df.columns:
                                            column_order.append('Plan')
                                        if 'Actual' in pivot_df.columns:
                                            column_order.append('Actual')
                                        if 'Predicted' in pivot_df.columns:
                                            column_order.append('Predicted')

                                        pivot_df = pivot_df[column_order]

                                        # Format the values for display
                                        for col in pivot_df.columns:
                                            pivot_df[col] = pivot_df[col].apply(
                                                lambda x: format_currency(x, currency_symbol, currency_postfix, thousands=False) if x != 0 else '—'
                                            )
                                        st.dataframe(pivot_df, width='stretch')
                                    elif cash_flow_type in ["Plan", "Predicted", "Actual"]:
                                        # Show single scenario table
                                        display_cash_flow = period_cash_flow[['Period', 'Cash_Flow']].copy()
                                        display_cash_flow['Cash_Flow'] = display_cash_flow['Cash_Flow'].apply(
                                            lambda x: format_currency(x, currency_symbol, currency_postfix, thousands=False)
                                        )
                                        st.dataframe(display_cash_flow, width='stretch')
                            else:
                                st.warning("No valid cash flow data could be generated from the selected projects.")
                                # Show debug information
                                with st.expander("🔍 Debug Information", expanded=False):
                                    st.write("**Issue:** Cash flow data was generated but became invalid after processing")
                                    st.write(f"**Cash flow type:** {cash_flow_type}")
                                    st.write(f"**Number of projects in data:** {len(df_cash)}")

                        else:
                            st.warning("No valid cash flow data could be generated from the selected projects.")
                            # Show debug information
                            with st.expander("🔍 Debug Information", expanded=False):
                                st.write("**Available columns in data:**")
                                st.write(list(df_cash.columns))
                                st.write(f"**Number of projects:** {len(df_cash)}")
                                if cash_flow_type == "Predicted":
                                    # Check for LD column
                                    ld_cols = [col for col in df_cash.columns if 'likely' in col.lower() or col.lower() == 'ld']
                                    st.write(f"**Likely Duration columns found:** {ld_cols if ld_cols else 'None'}")
                                    if ld_cols:
                                        for col in ld_cols:
                                            valid_count = df_cash[col].notna().sum()
                                            st.write(f"  - '{col}': {valid_count} projects with valid values")
                                # Check for BAC column
                                bac_cols = [col for col in df_cash.columns if 'bac' in col.lower() or 'budget' in col.lower()]
                                st.write(f"**Budget (BAC) columns found:** {bac_cols if bac_cols else 'None'}")
                                if bac_cols:
                                    for col in bac_cols:
                                        if col in df_cash.columns:
                                            valid_count = df_cash[col].notna().sum()
                                            total = df_cash[col].sum()
                                            st.write(f"  - '{col}': {valid_count} projects with valid values (total: {total:,.0f})")
                    else:
                        st.warning("No projects have valid start and finish dates.")

                except Exception as e:
                    st.error(f"Error processing cash flow data: {str(e)}")
                    st.info("Please check that date columns contain valid date formats.")
                    # Add detailed traceback for debugging
                    import traceback
                    st.code(traceback.format_exc())
            else:
                st.warning("Required finish date column not found.")
        else:
            st.info("Cash flow chart requires start date and finish date columns. Available columns:")
            if date_columns:
                for col in date_columns:
                    st.write(f"• {col}")
            else:
                st.write("No date columns detected in the data.")
    else:
        st.info("No data available for cash flow analysis.")


def render_time_budget_performance(filtered_df: pd.DataFrame) -> None:
    """Render the Time/Budget Performance chart."""
    if len(filtered_df) > 0:
        st.markdown("### Time/Budget Performance Analysis")
        st.markdown("This chart shows each project's performance relative to time and budget, with reference curves for comparison.")

        # Main view mode selector
        view_mode = st.radio(
            "View Mode:",
            options=["Project", "Department"],
            horizontal=True,
            key="time_budget_view_mode"
        )

        # Sub-options based on view mode
        if view_mode == "Project":
            # For Project mode, allow selection of coloring method
            tier_color_mode = st.radio(
                "Color Projects By:",
                options=["Budget Tiers", "Duration Tiers"],
                horizontal=True,
                key="time_budget_project_color_by"
            )
            bubble_size_mode = None
        else:  # Department
            # For Department mode, add bubble size selector
            bubble_size_mode = st.radio(
                "Bubble Size:",
                options=["Uniform", "Budget Proportional"],
                horizontal=True,
                key="time_budget_dept_bubble_size"
            )
            tier_color_mode = None  # Not used for department view

        # Calculate normalized values for each project
        performance_data = []

        # Get tier configuration or color mapping based on selected mode
        if view_mode == "Department":
            # For Department view, we'll color by organization
            # We'll create the color map after we know all departments
            tier_color_map = {}
            get_tier = None  # Will not use tier function for departments
            tier_names = []  # Not used in department mode
            tier_colors = []
            cutoff_points = []
        elif tier_color_mode == "Budget Tiers":
            tier_config = st.session_state.get('config_dict', {}).get('controls', {}).get('tier_config', {})
            default_tier_config = {
                'cutoff_points': [4000, 8000, 15000],
                'tier_names': ['Tier 1', 'Tier 2', 'Tier 3', 'Tier 4'],
                'colors': ['#3498db', '#27ae60', '#f39c12', '#e74c3c']
            }
            tier_colors = tier_config.get('colors', default_tier_config['colors'])
            tier_names = tier_config.get('tier_names', default_tier_config['tier_names'])
            cutoff_points = tier_config.get('cutoff_points', default_tier_config['cutoff_points'])

            # Create color mapping for tiers
            tier_color_map = {tier_names[i]: tier_colors[i] for i in range(len(tier_names))}

            # Function to determine tier based on budget
            def get_tier(project):
                """Determine budget tier based on budget and cutoff points."""
                budget = project.get('bac', project.get('BAC', project.get('Budget', 0)))
                if budget <= cutoff_points[0]:
                    return tier_names[0]
                elif budget <= cutoff_points[1]:
                    return tier_names[1]
                elif budget <= cutoff_points[2]:
                    return tier_names[2]
                else:
                    return tier_names[3]
        else:  # Duration Tiers
            tier_config = st.session_state.get('config_dict', {}).get('controls', {}).get('duration_tier_config', {})
            default_tier_config = {
                'cutoff_points': [6, 12, 24],
                'tier_names': ['Short', 'Medium', 'Long', 'Extra Long'],
                'colors': ['#3498db', '#27ae60', '#f39c12', '#e74c3c']
            }
            tier_colors = tier_config.get('colors', default_tier_config['colors'])
            tier_names = tier_config.get('tier_names', default_tier_config['tier_names'])
            cutoff_points = tier_config.get('cutoff_points', default_tier_config['cutoff_points'])

            # Create color mapping for tiers
            tier_color_map = {tier_names[i]: tier_colors[i] for i in range(len(tier_names))}

            # Function to determine tier based on duration
            def get_tier(project):
                """Determine duration tier based on original duration and cutoff points."""
                duration = project.get('original_duration_months', project.get('OD', project.get('Original Duration', 0)))
                if pd.isna(duration):
                    return "Unknown"
                duration = int(duration)
                # Use >= logic like budget tiers (check from high to low)
                if duration >= cutoff_points[2]:
                    return tier_names[3]
                elif duration >= cutoff_points[1]:
                    return tier_names[2]
                elif duration >= cutoff_points[0]:
                    return tier_names[1]
                else:
                    return tier_names[0]

        for _, project in filtered_df.iterrows():
            # Calculate % Time Used (AD/OD) - Actual Duration / Original Duration
            actual_duration = project.get('actual_duration_months', project.get('AD', project.get('Actual Duration', project.get('Actual Duration (months)', 0))))
            original_duration = project.get('original_duration_months', project.get('OD', project.get('Original Duration', project.get('Original Duration (months)', 0))))

            if pd.notna(actual_duration) and pd.notna(original_duration) and original_duration > 0:
                time_used_pct = actual_duration / original_duration
            else:
                continue  # Skip projects without duration data

            # Calculate % Budget Used (AC/BAC) - Actual Cost / Budget at Completion
            actual_cost = project.get('ac', project.get('AC', project.get('Actual Cost', 0)))
            budget = project.get('bac', project.get('BAC', project.get('Budget', 0)))

            if pd.notna(actual_cost) and pd.notna(budget) and budget > 0:
                budget_used_pct = actual_cost / budget
            else:
                continue  # Skip projects without budget/cost data

            # Determine tier/category and color based on selected mode
            if get_tier is not None:
                tier = get_tier(project)
                color = tier_color_map.get(tier, '#888888')
            else:
                tier = "N/A"
                color = '#888888'

            performance_data.append({
                'project_id': project.get('project_id', project.get('Project ID', 'Unknown')),
                'project_name': project.get('project_name', project.get('Project Name', 'Unknown')),
                'organization': project.get('organization', project.get('responsible_organization', project.get('Organization', 'Unknown'))),
                'time_used_pct': time_used_pct,
                'budget_used_pct': budget_used_pct,
                'tier': tier,
                'color': color,
                'bac': budget,
                'actual_duration': actual_duration,
                'original_duration': original_duration,
                'actual_cost': actual_cost,
                'spi': project.get('spi', project.get('SPI', 0)),
                'cpi': project.get('cpi', project.get('CPI', 0))
            })

        if performance_data:
            # Handle Department aggregation if in Department mode
            if view_mode == "Department":
                # Aggregate by department (organization)
                dept_aggregates = {}
                for p in performance_data:
                    org = p['organization']
                    if org not in dept_aggregates:
                        dept_aggregates[org] = {
                            'projects': [],
                            'total_bac': 0,
                            'total_ac': 0,
                            'total_ad_bac': 0,
                            'total_od_bac': 0
                        }

                    dept_aggregates[org]['projects'].append(p)
                    dept_aggregates[org]['total_bac'] += p['bac']
                    dept_aggregates[org]['total_ac'] += p['actual_cost']
                    dept_aggregates[org]['total_ad_bac'] += p['actual_duration'] * p['bac']
                    dept_aggregates[org]['total_od_bac'] += p['original_duration'] * p['bac']

                # Calculate weighted averages for each department
                dept_performance_data = []
                unique_depts = sorted(list(dept_aggregates.keys()))

                # Generate distinct colors for departments
                import plotly.colors as pc
                dept_colors = pc.qualitative.Plotly[:len(unique_depts)]
                if len(unique_depts) > len(dept_colors):
                    dept_colors = dept_colors * (len(unique_depts) // len(dept_colors) + 1)
                dept_color_map = {dept: dept_colors[i] for i, dept in enumerate(unique_depts)}

                for org, agg in dept_aggregates.items():
                    if agg['total_od_bac'] > 0 and agg['total_bac'] > 0:
                        time_used_pct = agg['total_ad_bac'] / agg['total_od_bac']
                        budget_used_pct = agg['total_ac'] / agg['total_bac']

                        # Calculate average SPI and CPI for the department
                        avg_spi = sum([p['spi'] for p in agg['projects']]) / len(agg['projects'])
                        avg_cpi = sum([p['cpi'] for p in agg['projects']]) / len(agg['projects'])

                        dept_performance_data.append({
                            'organization': org,
                            'time_used_pct': time_used_pct,
                            'budget_used_pct': budget_used_pct,
                            'total_bac': agg['total_bac'],
                            'project_count': len(agg['projects']),
                            'color': dept_color_map[org],
                            'spi': avg_spi,
                            'cpi': avg_cpi
                        })

                # Replace performance_data with department aggregates
                performance_data = dept_performance_data
                tier_color_map = dept_color_map

            # Calculate portfolio-level SPI and CPI for display
            spi_col = 'spi' if 'spi' in filtered_df.columns else 'SPI'
            cpi_col = 'cpi' if 'cpi' in filtered_df.columns else 'CPI'
            portfolio_spi = filtered_df[spi_col].mean() if spi_col in filtered_df.columns else 1.0
            portfolio_cpi = filtered_df[cpi_col].mean() if cpi_col in filtered_df.columns else 1.0

            # Calculate portfolio-level % time used and % budget used with weighted formulas
            # % time used = sum(AD*BAC)/sum(OD*BAC)
            # % budget used = sum(AC)/sum(BAC)
            total_ad_bac = 0
            total_od_bac = 0
            total_ac = 0
            total_bac = 0

            for _, project in filtered_df.iterrows():
                actual_duration = project.get('actual_duration_months', project.get('AD', project.get('Actual Duration', project.get('Actual Duration (months)', 0))))
                original_duration = project.get('original_duration_months', project.get('OD', project.get('Original Duration', project.get('Original Duration (months)', 0))))
                actual_cost = project.get('ac', project.get('AC', project.get('Actual Cost', 0)))
                budget = project.get('bac', project.get('BAC', project.get('Budget', 0)))

                if pd.notna(actual_duration) and pd.notna(original_duration) and pd.notna(budget) and budget > 0:
                    total_ad_bac += actual_duration * budget
                    total_od_bac += original_duration * budget

                if pd.notna(actual_cost) and pd.notna(budget):
                    total_ac += actual_cost
                    total_bac += budget

            portfolio_time_used = total_ad_bac / total_od_bac if total_od_bac > 0 else 1.0
            portfolio_budget_used = total_ac / total_bac if total_bac > 0 else 1.0

            # Create the plot using Plotly for interactivity

            fig = go.Figure()

            # Create normalized time array for reference curves
            T = np.linspace(0, 1, 101)

            # Define the performance curves (same as in project analysis)
            blue_curve = -0.794*T**3 + 0.632*T**2 + 1.162*T
            red_curve = -0.387*T**3 + 1.442*T**2 - 0.055*T

            # Plot the reference curves
            fig.add_trace(go.Scatter(
                x=T, y=blue_curve,
                mode='lines',
                name='Blue Curve (Good Performance)',
                line=dict(color='blue', width=2),
                opacity=0.7,
                hoverinfo='skip'
            ))

            fig.add_trace(go.Scatter(
                x=T, y=red_curve,
                mode='lines',
                name='Red Curve (Poor Performance)',
                line=dict(color='red', width=2),
                opacity=0.7,
                hoverinfo='skip'
            ))

            # Plot data based on view mode
            if view_mode == "Department":
                # Plot departments with budget-weighted aggregation
                # Calculate bubble sizes based on mode
                if bubble_size_mode == "Budget Proportional":
                    # Size bubbles proportionally to total budget
                    total_max_bac = max([p['total_bac'] for p in performance_data])
                    min_size = 15
                    max_size = 50
                    sizes = [min_size + (max_size - min_size) * (p['total_bac'] / total_max_bac) for p in performance_data]
                else:
                    # Uniform size
                    sizes = [20] * len(performance_data)

                for i, dept in enumerate(performance_data):
                    # Create hover text for department
                    hover_text = (
                        f"<b>{dept['organization']}</b><br>" +
                        f"Projects: {dept['project_count']}<br>" +
                        f"Total Budget: ${dept['total_bac']:,.0f}<br>" +
                        f"Avg SPI: {dept['spi']:.2f}<br>" +
                        f"Avg CPI: {dept['cpi']:.2f}<br>" +
                        f"% Time Used: {dept['time_used_pct'] * 100:.1f}%<br>" +
                        f"% Budget Used: {dept['budget_used_pct'] * 100:.1f}%"
                    )

                    fig.add_trace(go.Scatter(
                        x=[dept['time_used_pct']],
                        y=[dept['budget_used_pct']],
                        mode='markers',
                        name=dept['organization'],
                        marker=dict(
                            color=dept['color'],
                            size=sizes[i],
                            opacity=0.7,
                            line=dict(color='black', width=2)
                        ),
                        hovertemplate='%{hovertext}<extra></extra>',
                        hovertext=[hover_text]
                    ))
            else:
                # Plot individual projects colored by tier with tooltips
                # Get all unique tiers from the actual data
                actual_tiers = set([p['tier'] for p in performance_data])

                # Use actual tiers if they don't match configured tier names
                tiers_to_plot = tier_names if any(tier in tier_names for tier in actual_tiers) else list(actual_tiers)

                for tier in tiers_to_plot:
                    tier_projects = [p for p in performance_data if p['tier'] == tier]
                    if tier_projects:
                        x_vals = [p['time_used_pct'] for p in tier_projects]
                        y_vals = [p['budget_used_pct'] for p in tier_projects]
                        color = tier_color_map.get(tier, '#888888')  # Default gray if tier not in map

                        # Create custom hover text
                        hover_text = []
                        for p in tier_projects:
                            hover_text.append(
                                f"<b>{p['project_name']}</b><br>" +
                                f"Project ID: {p['project_id']}<br>" +
                                f"Organization: {p['organization']}<br>" +
                                f"Category: {p['tier']}<br>" +
                                f"BAC: ${p['bac']:,.0f}<br>" +
                                f"SPI: {p['spi']:.2f}<br>" +
                                f"CPI: {p['cpi']:.2f}<br>" +
                                f"% Time Used: {p['time_used_pct'] * 100:.1f}%<br>" +
                                f"% Budget Used: {p['budget_used_pct'] * 100:.1f}%"
                            )

                        fig.add_trace(go.Scatter(
                            x=x_vals, y=y_vals,
                            mode='markers',
                            name=tier,
                            marker=dict(
                                color=color,
                                size=10,
                                opacity=0.7,
                                line=dict(color='black', width=1)
                            ),
                            hovertemplate='%{hovertext}<extra></extra>',
                            hovertext=hover_text
                        ))

            # Add portfolio overall performance as large yellow star
            portfolio_hover = (
                f"<b>Portfolio Overall</b><br>" +
                f"% Time Used: {portfolio_time_used * 100:.1f}%<br>" +
                f"% Budget Used: {portfolio_budget_used * 100:.1f}%<br>" +
                f"Average SPI: {portfolio_spi:.2f}<br>" +
                f"Average CPI: {portfolio_cpi:.2f}"
            )

            fig.add_trace(go.Scatter(
                x=[portfolio_time_used], y=[portfolio_budget_used],
                mode='markers',
                name=f'Portfolio Overall',
                marker=dict(
                    color='yellow',
                    size=20,
                    opacity=0.9,
                    symbol='star',
                    line=dict(color='black', width=3)
                ),
                hovertemplate='%{hovertext}<extra></extra>',
                hovertext=[portfolio_hover]
            ))

            # Add reference lines
            fig.add_hline(y=1.0, line_dash="dash", line_color="gray", opacity=0.5, annotation_text="Budget Baseline")
            fig.add_vline(x=1.0, line_dash="dash", line_color="gray", opacity=0.5, annotation_text="Schedule Baseline")

            # Customize the layout
            fig.update_layout(
                title='Portfolio Time/Budget Performance Analysis',
                xaxis_title='% Time Used (AD/OD)',
                yaxis_title='% Budget Used (AC/BAC)',
                xaxis=dict(range=[0, 1.3], showgrid=True, gridwidth=1, gridcolor='lightgray'),
                yaxis=dict(range=[0, 1.3], showgrid=True, gridwidth=1, gridcolor='lightgray'),
                width=900,
                height=600,
                showlegend=True,
                legend=dict(
                    yanchor="top",
                    y=0.99,
                    xanchor="left",
                    x=1.01
                ),
                margin=dict(r=150)
            )

            st.plotly_chart(fig, width="stretch")

            # Add interpretation guide
            st.markdown("""
            **📊 Chart Interpretation:**
            - **X-axis**: % Time Used (AD/OD) - 1.0 = on schedule, <1.0 = ahead, >1.0 = delayed
            - **Y-axis**: % Budget Used (AC/BAC) - 1.0 = on budget, <1.0 = under budget, >1.0 = over budget
            - **Blue Curve**: Represents good performance trajectory
            - **Red Curve**: Represents poor performance trajectory
            - **Yellow Star**: Overall portfolio performance
            """)

            # Add color legend based on view mode
            if view_mode == "Project" and tier_color_mode == "Budget Tiers":
                # Get currency symbol
                currency_symbol = (
                    getattr(st.session_state, 'dashboard_currency_symbol', None) or
                    getattr(st.session_state, 'currency_symbol', None) or
                    st.session_state.get('config_dict', {}).get('controls', {}).get('currency_symbol', '$')
                )

                st.markdown("**🎨 Color Legend (Budget Tiers):**")
                legend_cols = st.columns(len(tier_names))
                for i, (tier_name, tier_color) in enumerate(zip(tier_names, tier_colors)):
                    with legend_cols[i]:
                        if i == 0:
                            range_text = f"< {currency_symbol}{cutoff_points[0]:,.0f}"
                        elif i == len(tier_names) - 1:
                            range_text = f"≥ {currency_symbol}{cutoff_points[i-1]:,.0f}"
                        else:
                            range_text = f"{currency_symbol}{cutoff_points[i-1]:,.0f}-{currency_symbol}{cutoff_points[i]:,.0f}"

                        st.markdown(
                            f'<div style="background-color: {tier_color}; padding: 10px; border-radius: 5px; text-align: center; color: white; font-weight: bold;">'
                            f'{tier_name}<br><span style="font-size: 0.8em;">{range_text}</span>'
                            f'</div>',
                            unsafe_allow_html=True
                        )
            elif view_mode == "Project" and tier_color_mode == "Duration Tiers":
                st.markdown("**🎨 Color Legend (Duration Tiers):**")
                legend_cols = st.columns(len(tier_names))
                for i, (tier_name, tier_color) in enumerate(zip(tier_names, tier_colors)):
                    with legend_cols[i]:
                        if i == 0:
                            range_text = f"< {cutoff_points[0]} mo"
                        elif i == len(tier_names) - 1:
                            range_text = f"≥ {cutoff_points[i-1]} mo"
                        else:
                            range_text = f"{cutoff_points[i-1]}-{cutoff_points[i]} mo"

                        st.markdown(
                            f'<div style="background-color: {tier_color}; padding: 10px; border-radius: 5px; text-align: center; color: white; font-weight: bold;">'
                            f'{tier_name}<br><span style="font-size: 0.8em;">{range_text}</span>'
                            f'</div>',
                            unsafe_allow_html=True
                        )
            elif view_mode == "Department":
                st.markdown(f"**📊 Showing {len(performance_data)} departments with budget-weighted aggregation**")
                if bubble_size_mode == "Budget Proportional":
                    st.info("💡 Bubble size is proportional to department's total budget")
                else:
                    st.info("💡 All departments shown with uniform bubble size")

            # Show summary statistics
            col1, col2, col3 = st.columns(3)
            with col1:
                if view_mode == "Department":
                    st.metric("Departments", len(performance_data))
                else:
                    st.metric("Projects Analyzed", len(performance_data))
            with col2:
                st.metric("Portfolio % Time Used", f"{portfolio_time_used * 100:.1f}%")
            with col3:
                st.metric("Portfolio % Budget Used", f"{portfolio_budget_used * 100:.1f}%")

        else:
            st.warning("Insufficient data for Time/Budget Performance analysis. Projects need both duration data (Actual Duration, Original Duration) and budget data (Actual Cost, Budget).")
    else:
        st.info("No projects available for Time/Budget Performance analysis.")


def render_portfolio_performance_curve(filtered_df: pd.DataFrame) -> None:
    """Render the Portfolio Performance Matrix chart."""
    # Get currency settings
    currency_symbol = (
        getattr(st.session_state, 'dashboard_currency_symbol', None) or
        getattr(st.session_state, 'currency_symbol', None) or
        st.session_state.get('config_dict', {}).get('controls', {}).get('currency_symbol', '$')
    )

    # Main view mode selector
    view_mode = st.radio(
        "View Mode:",
        options=["Project", "Department"],
        horizontal=True,
        key="perf_matrix_view_mode"
    )

    # Sub-options based on view mode
    if view_mode == "Project":
        # For Project mode, allow selection of coloring method
        perf_tier_color_mode = st.radio(
            "Color Projects By:",
            options=["Budget Tiers", "Duration Tiers"],
            horizontal=True,
            key="perf_curve_tier_color_by"
        )
        bubble_size_mode = None
    else:  # Department
        # For Department mode, add bubble size selector
        bubble_size_mode = st.radio(
            "Bubble Size:",
            options=["Uniform", "Budget Proportional"],
            horizontal=True,
            key="perf_matrix_bubble_size"
        )
        perf_tier_color_mode = None  # Not used for department view

    # Check for required columns (allow both lowercase and uppercase variants)
    cpi_col = 'cpi' if 'cpi' in filtered_df.columns else 'CPI'
    spi_col = 'spi' if 'spi' in filtered_df.columns else 'SPI'
    budget_col = 'bac' if 'bac' in filtered_df.columns else 'BAC' if 'BAC' in filtered_df.columns else 'Budget'
    project_name_col = 'project_name' if 'project_name' in filtered_df.columns else 'Project Name'

    if len(filtered_df) > 0 and cpi_col in filtered_df.columns and spi_col in filtered_df.columns and budget_col in filtered_df.columns:

        # Handle Department aggregation if in Department mode
        if view_mode == "Department":
            # Get organization column
            org_col = None
            if 'organization' in filtered_df.columns:
                org_col = 'organization'
            elif 'responsible_organization' in filtered_df.columns:
                org_col = 'responsible_organization'
            elif 'Organization' in filtered_df.columns:
                org_col = 'Organization'

            if org_col:
                # Group by department and calculate weighted averages
                dept_data = []
                for dept, group in filtered_df.groupby(org_col):
                    total_bac = group[budget_col].sum()
                    if total_bac > 0:
                        # Calculate weighted average SPI and CPI by budget
                        weighted_spi = (group[spi_col] * group[budget_col]).sum() / total_bac
                        weighted_cpi = (group[cpi_col] * group[budget_col]).sum() / total_bac

                        dept_data.append({
                            'Department': dept,
                            spi_col: weighted_spi,
                            cpi_col: weighted_cpi,
                            budget_col: total_bac,
                            'Project_Count': len(group),
                            project_name_col: dept  # Use department name for hover
                        })

                if dept_data:
                    # Replace filtered_df with department aggregates
                    filtered_df = pd.DataFrame(dept_data)
                    # Update project_name_col to use Department
                    project_name_col = 'Department'
                else:
                    st.warning("No department data available for aggregation.")
                    return
            else:
                st.warning("Organization column not found in data.")
                return

        # Get tier configuration based on selected mode
        has_data = False
        if view_mode == "Department":
            # For Department view, generate colors for departments
            import plotly.colors as pc
            unique_depts = sorted(filtered_df[project_name_col].unique())
            dept_colors = pc.qualitative.Plotly[:len(unique_depts)]
            if len(unique_depts) > len(dept_colors):
                dept_colors = dept_colors * (len(unique_depts) // len(dept_colors) + 1)

            # Create a single tier for all departments (we'll color them differently in plotting)
            df_scatter = filtered_df.copy()
            df_scatter['Tier_Range'] = 'Department'
            has_data = True
            tier_order = ['Department']
            tier_names = ['Department']
            tier_colors = ['#3498db']  # Not used, we'll use custom colors

        elif perf_tier_color_mode == "Budget Tiers":
            tier_config = st.session_state.get('config_dict', {}).get('controls', {}).get('tier_config', {})
            default_tier_config = {
                'cutoff_points': [4000, 8000, 15000],
                'tier_names': ['Tier 1', 'Tier 2', 'Tier 3', 'Tier 4'],
                'colors': ['#3498db', '#27ae60', '#f39c12', '#e74c3c']
            }
            cutoffs = tier_config.get('cutoff_points', default_tier_config['cutoff_points'])
            tier_names = tier_config.get('tier_names', default_tier_config['tier_names'])
            tier_colors = tier_config.get('colors', default_tier_config['colors'])

            # Create configurable BAC-based tier ranges
            budget_values = filtered_df[budget_col].dropna()
            if len(budget_values) > 0:
                has_data = True
                def get_category(row):
                    budget = row[budget_col]
                    if pd.isna(budget):
                        return "Unknown"
                    elif budget >= cutoffs[2]:  # Tier 4 (highest)
                        return f"{tier_names[3]}: (≥ {currency_symbol}{cutoffs[2]:,.0f})"
                    elif budget >= cutoffs[1]:  # Tier 3
                        return f"{tier_names[2]}: ({currency_symbol}{cutoffs[1]:,.0f} - {currency_symbol}{cutoffs[2]:,.0f})"
                    elif budget >= cutoffs[0]:  # Tier 2
                        return f"{tier_names[1]}: ({currency_symbol}{cutoffs[0]:,.0f} - {currency_symbol}{cutoffs[1]:,.0f})"
                    else:  # Tier 1 (lowest)
                        return f"{tier_names[0]}: (< {currency_symbol}{cutoffs[0]:,.0f})"

                # Add category to filtered dataframe
                df_scatter = filtered_df.copy()
                df_scatter['Tier_Range'] = df_scatter.apply(get_category, axis=1)

                # Define the tier order (Tier 4 to Tier 1 for legend display)
                tier_order = [
                    f"{tier_names[3]}: (≥ {currency_symbol}{cutoffs[2]:,.0f})",
                    f"{tier_names[2]}: ({currency_symbol}{cutoffs[1]:,.0f} - {currency_symbol}{cutoffs[2]:,.0f})",
                    f"{tier_names[1]}: ({currency_symbol}{cutoffs[0]:,.0f} - {currency_symbol}{cutoffs[1]:,.0f})",
                    f"{tier_names[0]}: (< {currency_symbol}{cutoffs[0]:,.0f})"
                ]

                # Convert to categorical with specific order
                df_scatter['Tier_Range'] = pd.Categorical(df_scatter['Tier_Range'], categories=tier_order, ordered=True)

        else:  # Duration Tiers
            tier_config = st.session_state.get('config_dict', {}).get('controls', {}).get('duration_tier_config', {})
            default_tier_config = {
                'cutoff_points': [6, 12, 24],
                'tier_names': ['Short', 'Medium', 'Long', 'Extra Long'],
                'colors': ['#3498db', '#27ae60', '#f39c12', '#e74c3c']
            }
            cutoffs = tier_config.get('cutoff_points', default_tier_config['cutoff_points'])
            tier_names = tier_config.get('tier_names', default_tier_config['tier_names'])
            tier_colors = tier_config.get('colors', default_tier_config['colors'])

            # Create duration-based tier ranges
            duration_col = 'original_duration_months'
            if duration_col in filtered_df.columns:
                has_data = True
                def get_category(row):
                    duration = row.get(duration_col, row.get('OD', row.get('Original Duration', 0)))
                    if pd.isna(duration):
                        return "Unknown"
                    duration = int(duration)
                    # Use >= logic like budget tiers (check from high to low)
                    if duration >= cutoffs[2]:
                        return f"{tier_names[3]}: (≥ {cutoffs[2]} mo)"
                    elif duration >= cutoffs[1]:
                        return f"{tier_names[2]}: ({cutoffs[1]}-{cutoffs[2]} mo)"
                    elif duration >= cutoffs[0]:
                        return f"{tier_names[1]}: ({cutoffs[0]}-{cutoffs[1]} mo)"
                    else:
                        return f"{tier_names[0]}: (< {cutoffs[0]} mo)"

                # Add category to filtered dataframe
                df_scatter = filtered_df.copy()
                df_scatter['Tier_Range'] = df_scatter.apply(get_category, axis=1)

                # Define the tier order (Extra Long to Short for legend display)
                tier_order = [
                    f"{tier_names[3]}: (≥ {cutoffs[2]} mo)",
                    f"{tier_names[2]}: ({cutoffs[1]}-{cutoffs[2]} mo)",
                    f"{tier_names[1]}: ({cutoffs[0]}-{cutoffs[1]} mo)",
                    f"{tier_names[0]}: (< {cutoffs[0]} mo)"
                ]

                # Convert to categorical with specific order
                df_scatter['Tier_Range'] = pd.Categorical(df_scatter['Tier_Range'], categories=tier_order, ordered=True)
            else:
                st.warning("Duration data not available. Run batch EVM calculation first.")
                return

        if has_data:

            # Create scatter plot based on view mode
            if view_mode == "Department":
                # Department mode with custom colors and sizes
                import plotly.graph_objects as go
                fig_performance = go.Figure()

                # Calculate bubble sizes
                if bubble_size_mode == "Budget Proportional":
                    total_max_bac = df_scatter[budget_col].max()
                    min_size = 15
                    max_size = 50
                    sizes = [min_size + (max_size - min_size) * (bac / total_max_bac) for bac in df_scatter[budget_col]]
                else:
                    sizes = [20] * len(df_scatter)

                # Add each department as a separate trace
                for i, (idx, dept) in enumerate(df_scatter.iterrows()):
                    dept_name = dept[project_name_col]
                    color = dept_colors[i % len(dept_colors)]

                    # Create hover text
                    hover_text = (
                        f"<b>{dept_name}</b><br>" +
                        f"Projects: {dept.get('Project_Count', 'N/A')}<br>" +
                        f"Total Budget: {currency_symbol}{dept[budget_col]:,.0f}<br>" +
                        f"SPI: {dept[spi_col]:.2f}<br>" +
                        f"CPI: {dept[cpi_col]:.2f}"
                    )

                    fig_performance.add_trace(go.Scatter(
                        x=[dept[spi_col]],
                        y=[dept[cpi_col]],
                        mode='markers',
                        name=dept_name,
                        marker=dict(
                            color=color,
                            size=sizes[i],
                            opacity=0.7,
                            line=dict(color='black', width=2)
                        ),
                        hovertemplate='%{hovertext}<extra></extra>',
                        hovertext=[hover_text]
                    ))

                # Update layout for Department mode
                fig_performance.update_layout(
                    title=f"Portfolio Performance Matrix - Department View ({len(df_scatter)} departments)",
                    xaxis_title='Schedule Performance Index (SPI)<br>← Behind Schedule | Ahead of Schedule →',
                    yaxis_title='Cost Performance Index (CPI)<br>← Over Budget | Under Budget →',
                    height=500,
                    showlegend=True,
                    legend=dict(
                        orientation="v",
                        yanchor="top",
                        y=1,
                        xanchor="left",
                        x=1.02
                    )
                )
            else:
                # Project mode with tier coloring
                tier_label = 'Budget Range' if perf_tier_color_mode == "Budget Tiers" else 'Duration Range'
                fig_performance = px.scatter(
                    df_scatter,
                    x=spi_col,
                    y=cpi_col,
                    color='Tier_Range',
                    hover_data=[project_name_col, budget_col],
                    title=f"Portfolio Performance Matrix ({len(filtered_df)} projects)",
                    labels={
                        spi_col: 'Schedule Performance Index (SPI)',
                        cpi_col: 'Cost Performance Index (CPI)',
                        'Tier_Range': tier_label
                    },
                    color_discrete_sequence=[tier_colors[3], tier_colors[2], tier_colors[1], tier_colors[0]],  # Tier 4 to Tier 1
                    category_orders={'Tier_Range': tier_order}
                )

                # Update layout for Project mode
                fig_performance.update_layout(
                    height=500,
                    showlegend=True,
                    legend=dict(
                        orientation="v",
                        yanchor="top",
                        y=1,
                        xanchor="left",
                        x=1.02
                    ),
                    xaxis=dict(title='Schedule Performance Index (SPI)<br>← Behind Schedule | Ahead of Schedule →'),
                    yaxis=dict(title='Cost Performance Index (CPI)<br>← Over Budget | Under Budget →')
                )

                # Update traces for consistent dot appearance in Project mode
                fig_performance.update_traces(
                    marker=dict(
                        size=10,
                        line=dict(width=1, color='rgba(0,0,0,0.3)')
                    )
                )

            # Add quadrant lines at 1.0 for both axes
            fig_performance.add_hline(y=1.0, line_dash="dash", line_color="gray", opacity=0.7)
            fig_performance.add_vline(x=1.0, line_dash="dash", line_color="gray", opacity=0.7)

            # Add quadrant labels
            fig_performance.add_annotation(
                x=1.3, y=1.3, text="✅ On Time<br>Under Budget",
                showarrow=False, font=dict(size=12, color="green"), bgcolor="rgba(255,255,255,0.8)"
            )
            fig_performance.add_annotation(
                x=0.7, y=1.3, text="⚠️ Behind Schedule<br>Under Budget",
                showarrow=False, font=dict(size=12, color="orange"), bgcolor="rgba(255,255,255,0.8)"
            )
            fig_performance.add_annotation(
                x=1.3, y=0.7, text="⚠️ On Time<br>Over Budget",
                showarrow=False, font=dict(size=12, color="orange"), bgcolor="rgba(255,255,255,0.8)"
            )
            fig_performance.add_annotation(
                x=0.7, y=0.7, text="🚨 Behind Schedule<br>Over Budget",
                showarrow=False, font=dict(size=12, color="red"), bgcolor="rgba(255,255,255,0.8)"
            )

            st.plotly_chart(fig_performance, width="stretch")

            # Filter summary for performance curve
            if view_mode == "Project":
                health_col = 'health_category' if 'health_category' in df_scatter.columns else 'Health_Category'
                if health_col in df_scatter.columns:
                    col1, col2, col3 = st.columns(3)
                    with col1:
                        healthy_count = len(df_scatter[df_scatter[health_col] == 'Healthy'])
                        st.metric("✅ Healthy Projects", healthy_count)
                    with col2:
                        at_risk_count = len(df_scatter[df_scatter[health_col] == 'At Risk'])
                        st.metric("⚠️ At Risk Projects", at_risk_count)
                    with col3:
                        critical_count = len(df_scatter[df_scatter[health_col] == 'Critical'])
                        st.metric("🚨 Critical Projects", critical_count)
            else:
                # Department mode - show department count and total budget
                col1, col2 = st.columns(2)
                with col1:
                    st.metric("Departments", len(df_scatter))
                with col2:
                    total_budget = df_scatter[budget_col].sum()
                    st.metric("Total Budget", f"{currency_symbol}{total_budget:,.0f}")

            # Add guidance text
            if view_mode == "Department":
                st.markdown(f"""
                **📊 How to Read This Chart:**
                - **X-axis (SPI):** Schedule Performance - Right is better (ahead of schedule)
                - **Y-axis (CPI):** Cost Performance - Up is better (under budget)
                - **Target Zone:** Upper right quadrant (SPI > 1.0, CPI > 1.0)
                - **Bubbles:** Each bubble represents a department with budget-weighted SPI/CPI
                - **Hover:** Click any bubble to see department details
                """)
                if bubble_size_mode == "Budget Proportional":
                    st.info("💡 Bubble size is proportional to department's total budget")
                else:
                    st.info("💡 All departments shown with uniform bubble size")
            else:
                st.markdown(f"""
                **📊 How to Read This Chart:**
                - **X-axis (SPI):** Schedule Performance - Right is better (ahead of schedule)
                - **Y-axis (CPI):** Cost Performance - Up is better (under budget)
                - **Target Zone:** Upper right quadrant (SPI > 1.0, CPI > 1.0)
                - **Hover:** Click any dot to see project name and budget details
                - **Chart updates automatically** based on your filter selections above
                """)

            # Add color legend based on view mode
            if view_mode == "Project" and perf_tier_color_mode == "Budget Tiers":
                st.markdown("**🎨 Color Legend (Budget Tiers):**")
                legend_cols = st.columns(len(tier_names))
                for i, (tier_name, tier_color) in enumerate(zip(tier_names, tier_colors)):
                    with legend_cols[i]:
                        if i == 0:
                            range_text = f"< {currency_symbol}{cutoffs[0]:,.0f}"
                        elif i == len(tier_names) - 1:
                            range_text = f"≥ {currency_symbol}{cutoffs[i-1]:,.0f}"
                        else:
                            range_text = f"{currency_symbol}{cutoffs[i-1]:,.0f}-{currency_symbol}{cutoffs[i]:,.0f}"

                        st.markdown(
                            f'<div style="background-color: {tier_color}; padding: 10px; border-radius: 5px; text-align: center; color: white; font-weight: bold;">'
                            f'{tier_name}<br><span style="font-size: 0.8em;">{range_text}</span>'
                            f'</div>',
                            unsafe_allow_html=True
                        )
            elif view_mode == "Project" and perf_tier_color_mode == "Duration Tiers":
                st.markdown("**🎨 Color Legend (Duration Tiers):**")
                legend_cols = st.columns(len(tier_names))
                for i, (tier_name, tier_color) in enumerate(zip(tier_names, tier_colors)):
                    with legend_cols[i]:
                        if i == 0:
                            range_text = f"< {cutoffs[0]} mo"
                        elif i == len(tier_names) - 1:
                            range_text = f"≥ {cutoffs[i-1]} mo"
                        else:
                            range_text = f"{cutoffs[i-1]}-{cutoffs[i]} mo"

                        st.markdown(
                            f'<div style="background-color: {tier_color}; padding: 10px; border-radius: 5px; text-align: center; color: white; font-weight: bold;">'
                            f'{tier_name}<br><span style="font-size: 0.8em;">{range_text}</span>'
                            f'</div>',
                            unsafe_allow_html=True
                        )
        else:
            st.info("No budget data available for performance curve.")
    else:
        st.info("Performance curve requires CPI, SPI, and Budget data.")


def render_portfolio_treemap(filtered_df: pd.DataFrame) -> None:
    """Render the Portfolio Treemap chart."""
    # Get currency settings
    currency_symbol = (
        getattr(st.session_state, 'dashboard_currency_symbol', None) or
        getattr(st.session_state, 'currency_symbol', None) or
        st.session_state.get('config_dict', {}).get('controls', {}).get('currency_symbol', '$')
    )

    if len(filtered_df) > 0:
        # Toggle for tier coloring
        treemap_tier_mode = st.radio(
            "Color Projects By:",
            options=["Budget Tiers", "Duration Tiers"],
            horizontal=True,
            key="treemap_tier_color_mode"
        )

        # Prepare data for treemap
        treemap_df = filtered_df.copy()

        # Get organization column name (check for different variants)
        org_col = None
        if 'organization' in treemap_df.columns:
            org_col = 'organization'
        elif 'Organization' in treemap_df.columns:
            org_col = 'Organization'

        # Get project name column
        project_name_col = 'project_name' if 'project_name' in treemap_df.columns else 'Project Name'

        # Get budget column
        budget_col = 'bac' if 'bac' in treemap_df.columns else 'BAC' if 'BAC' in treemap_df.columns else 'Budget'

        # Ensure we have the required columns
        if org_col and project_name_col in treemap_df.columns and budget_col in treemap_df.columns:
            # Add tier names to the dataframe based on selected mode
            if treemap_tier_mode == "Budget Tiers":
                # Calculate tier based on budget
                tier_config = st.session_state.get('config_dict', {}).get('controls', {}).get('tier_config', {})
                default_tier_config = {
                    'cutoff_points': [4000, 8000, 15000],
                    'tier_names': ['Tier 1', 'Tier 2', 'Tier 3', 'Tier 4']
                }
                cutoffs = tier_config.get('cutoff_points', default_tier_config['cutoff_points'])
                tier_names = tier_config.get('tier_names', default_tier_config['tier_names'])

                def get_tier(row):
                    budget = row[budget_col]
                    if pd.isna(budget):
                        return "Unknown"
                    elif budget <= cutoffs[0]:
                        return tier_names[0]
                    elif budget <= cutoffs[1]:
                        return tier_names[1]
                    elif budget <= cutoffs[2]:
                        return tier_names[2]
                    else:
                        return tier_names[3]

                treemap_df['tier_names'] = treemap_df.apply(get_tier, axis=1)

            else:  # Duration Tiers
                # Calculate tier based on duration
                tier_config = st.session_state.get('config_dict', {}).get('controls', {}).get('duration_tier_config', {})
                default_tier_config = {
                    'cutoff_points': [6, 12, 24],
                    'tier_names': ['Short', 'Medium', 'Long', 'Extra Long']
                }
                cutoffs = tier_config.get('cutoff_points', default_tier_config['cutoff_points'])
                tier_names = tier_config.get('tier_names', default_tier_config['tier_names'])

                def get_tier(row):
                    duration = row.get('original_duration_months', row.get('OD', row.get('Original Duration', 0)))
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

                treemap_df['tier_names'] = treemap_df.apply(get_tier, axis=1)

            # Create drill-down selection
            col1, col2 = st.columns([1, 3])

            with col1:
                drill_level = st.selectbox(
                    "View Level:",
                    ["Organization", "Project"],
                    help="Select Organization for high-level view, Project for detailed drill-down",
                    key="treemap_drill_level"
                )

            with col2:
                if drill_level == "Project":
                    # Show organization filter for project-level view
                    available_orgs = sorted(treemap_df[org_col].dropna().unique())
                    selected_org = st.selectbox(
                        "Filter by Organization:",
                        ["All"] + available_orgs,
                        help="Select an organization to focus on its projects",
                        key="treemap_org_filter"
                    )
                else:
                    selected_org = "All"

            # Filter data based on selection
            if drill_level == "Project" and selected_org != "All":
                plot_df = treemap_df[treemap_df[org_col] == selected_org].copy()
                if len(plot_df) == 0:
                    st.warning(f"No projects found for {selected_org}")
                    plot_df = treemap_df.copy()
            else:
                plot_df = treemap_df.copy()

            # Create treemap based on selected level
            if drill_level == "Organization":
                # Aggregate by organization
                org_summary = plot_df.groupby(org_col).agg({
                    budget_col: 'sum',
                    project_name_col: 'count'
                }).reset_index()
                org_summary = org_summary.rename(columns={project_name_col: 'Project_Count'})

                # Create organization-level treemap
                fig_treemap = px.treemap(
                    org_summary,
                    path=[org_col],
                    values=budget_col,
                    color=org_col,
                    title=f"Portfolio Treemap - Organization Level ({len(org_summary)} organizations, {len(plot_df)} projects)",
                    hover_data={'Project_Count': True},
                    labels={
                        budget_col: f'Total Budget ({currency_symbol})',
                        'Project_Count': 'Project Count'
                    }
                )

                # Update hover template for organizations
                fig_treemap.update_traces(
                    hovertemplate='<b>%{label}</b><br>' +
                                f'Budget: {currency_symbol}%{{value:,.0f}}<br>' +
                                'Projects: %{customdata[0]}<br>' +
                                '<extra></extra>'
                )

            else:  # Project level
                # Project-level treemap with tier colors
                if len(plot_df) > 0:
                    # Check for CPI and SPI columns
                    cpi_col = 'cpi' if 'cpi' in plot_df.columns else 'CPI'
                    spi_col = 'spi' if 'spi' in plot_df.columns else 'SPI'

                    # Get tier configuration for color mapping based on selected mode
                    if treemap_tier_mode == "Budget Tiers":
                        tier_config = st.session_state.get('config_dict', {}).get('controls', {}).get('tier_config', {})
                        default_tier_config = {
                            'tier_names': ['Tier 1', 'Tier 2', 'Tier 3', 'Tier 4'],
                            'colors': ['#3498db', '#27ae60', '#f39c12', '#e74c3c']
                        }
                    else:  # Duration Tiers
                        tier_config = st.session_state.get('config_dict', {}).get('controls', {}).get('duration_tier_config', {})
                        default_tier_config = {
                            'tier_names': ['Short', 'Medium', 'Long', 'Extra Long'],
                            'colors': ['#3498db', '#27ae60', '#f39c12', '#e74c3c']
                        }

                    tier_names_map = tier_config.get('tier_names', default_tier_config['tier_names'])
                    tier_colors_map = tier_config.get('colors', default_tier_config['colors'])
                    tier_color_discrete_map = {tier_names_map[i]: tier_colors_map[i] for i in range(len(tier_names_map))}

                    fig_treemap = px.treemap(
                        plot_df,
                        path=[org_col, project_name_col],
                        values=budget_col,
                        color='tier_names',
                        color_discrete_map=tier_color_discrete_map,
                        title=f"Portfolio Treemap - Project Level ({selected_org if selected_org != 'All' else 'All Organizations'}, {len(plot_df)} projects)",
                        hover_data={
                            budget_col: ':,.0f',
                            cpi_col: ':.3f' if cpi_col in plot_df.columns else False,
                            spi_col: ':.3f' if spi_col in plot_df.columns else False
                        },
                        labels={
                            budget_col: f'Budget ({currency_symbol})',
                            'tier_names': 'Budget Tier'
                        }
                    )

                    # Update hover template for projects
                    hover_template = '<b>%{label}</b><br>' + f'Budget: {currency_symbol}%{{value:,.0f}}<br>'
                    if cpi_col in plot_df.columns:
                        hover_template += 'CPI: %{customdata[1]:.3f}<br>'
                    if spi_col in plot_df.columns:
                        hover_template += 'SPI: %{customdata[2] if len(customdata) > 2 else "N/A"}<br>'
                    hover_template += '<extra></extra>'

                    fig_treemap.update_traces(hovertemplate=hover_template)

            # Update layout for better appearance
            fig_treemap.update_layout(
                height=600,
                font_size=12,
                margin=dict(t=80, l=10, r=10, b=10)
            )

            # Display the treemap
            st.plotly_chart(fig_treemap, width="stretch")

            # Add explanatory text
            if drill_level == "Organization":
                st.markdown("""
                **📊 How to Read This Treemap:**
                - **Size**: Represents total budget for each organization
                - **Color**: Different colors for each organization
                - **Hover**: Shows organization name, total budget, and project count
                - **Click**: Switch to "Project" view above to drill down into specific organizations
                """)
            else:
                tier_label = "budget tier" if treemap_tier_mode == "Budget Tiers" else "duration tier"
                st.markdown(f"""
                **📊 How to Read This Treemap:**
                - **Size**: Represents budget for each project
                - **Color**: Shows {tier_label} for each project
                - **Hierarchy**: Projects are grouped by organization
                - **Hover**: Shows project details including budget, CPI, and SPI
                - **Filter**: Use the organization dropdown above to focus on specific organizations
                {f"- **Current View**: Showing {selected_org}" if selected_org != "All" else "- **Current View**: Showing all organizations"}
                """)

                # Add color legend based on tier mode
                if treemap_tier_mode == "Budget Tiers":
                    tier_config = st.session_state.get('config_dict', {}).get('controls', {}).get('tier_config', {})
                    default_tier_config = {
                        'cutoff_points': [4000, 8000, 15000],
                        'tier_names': ['Tier 1', 'Tier 2', 'Tier 3', 'Tier 4'],
                        'colors': ['#3498db', '#27ae60', '#f39c12', '#e74c3c']
                    }
                    cutoffs = tier_config.get('cutoff_points', default_tier_config['cutoff_points'])
                    tier_names_legend = tier_config.get('tier_names', default_tier_config['tier_names'])
                    tier_colors_legend = tier_config.get('colors', default_tier_config['colors'])

                    st.markdown("**🎨 Color Legend (Budget Tiers):**")
                    legend_cols = st.columns(len(tier_names_legend))
                    for i, (tier_name, tier_color) in enumerate(zip(tier_names_legend, tier_colors_legend)):
                        with legend_cols[i]:
                            if i == 0:
                                range_text = f"< {currency_symbol}{cutoffs[0]:,.0f}"
                            elif i == len(tier_names_legend) - 1:
                                range_text = f"≥ {currency_symbol}{cutoffs[i-1]:,.0f}"
                            else:
                                range_text = f"{currency_symbol}{cutoffs[i-1]:,.0f}-{currency_symbol}{cutoffs[i]:,.0f}"

                            st.markdown(
                                f'<div style="background-color: {tier_color}; padding: 10px; border-radius: 5px; text-align: center; color: white; font-weight: bold;">'
                                f'{tier_name}<br><span style="font-size: 0.8em;">{range_text}</span>'
                                f'</div>',
                                unsafe_allow_html=True
                            )

                else:  # Duration Tiers
                    tier_config = st.session_state.get('config_dict', {}).get('controls', {}).get('duration_tier_config', {})
                    default_tier_config = {
                        'cutoff_points': [6, 12, 24],
                        'tier_names': ['Short', 'Medium', 'Long', 'Extra Long'],
                        'colors': ['#3498db', '#27ae60', '#f39c12', '#e74c3c']
                    }
                    cutoffs = tier_config.get('cutoff_points', default_tier_config['cutoff_points'])
                    tier_names_legend = tier_config.get('tier_names', default_tier_config['tier_names'])
                    tier_colors_legend = tier_config.get('colors', default_tier_config['colors'])

                    st.markdown("**🎨 Color Legend (Duration Tiers):**")
                    legend_cols = st.columns(len(tier_names_legend))
                    for i, (tier_name, tier_color) in enumerate(zip(tier_names_legend, tier_colors_legend)):
                        with legend_cols[i]:
                            if i == 0:
                                range_text = f"< {cutoffs[0]} mo"
                            elif i == len(tier_names_legend) - 1:
                                range_text = f"≥ {cutoffs[i-1]} mo"
                            else:
                                range_text = f"{cutoffs[i-1]}-{cutoffs[i]} mo"

                            st.markdown(
                                f'<div style="background-color: {tier_color}; padding: 10px; border-radius: 5px; text-align: center; color: white; font-weight: bold;">'
                                f'{tier_name}<br><span style="font-size: 0.8em;">{range_text}</span>'
                                f'</div>',
                                unsafe_allow_html=True
                            )

        else:
            st.warning("Treemap requires Organization, Project Name, and Budget columns.")
    else:
        st.info("No data available for treemap visualization.")


def render_portfolio_budget_chart(filtered_df: pd.DataFrame) -> None:
    """Render the Organization Budget Chart."""
    # Get currency settings
    currency_symbol = (
        getattr(st.session_state, 'dashboard_currency_symbol', None) or
        getattr(st.session_state, 'currency_symbol', None) or
        st.session_state.get('config_dict', {}).get('controls', {}).get('currency_symbol', '$')
    )
    currency_postfix = (
        getattr(st.session_state, 'dashboard_currency_postfix', None) or
        getattr(st.session_state, 'currency_postfix', None) or
        st.session_state.get('config_dict', {}).get('controls', {}).get('currency_postfix', '')
    )

    # Get organization column name
    org_col = None
    if 'organization' in filtered_df.columns:
        org_col = 'organization'
    elif 'Organization' in filtered_df.columns:
        org_col = 'Organization'

    # Get budget column
    budget_col = 'bac' if 'bac' in filtered_df.columns else 'BAC' if 'BAC' in filtered_df.columns else 'Budget'

    if org_col and org_col in filtered_df.columns and len(filtered_df) > 0:
        try:
            # Add chart type and tier mode selection
            col1, col2, col3 = st.columns([1, 1, 2])
            with col1:
                chart_type = st.selectbox(
                    "Chart Type:",
                    ["Total Budget", "Budget by Tier"],
                    help="Select chart type: Total Budget shows organization totals, Budget by Tier shows stacked bars by tier",
                    key="budget_chart_type"
                )
            with col2:
                tier_mode = st.radio(
                    "Tier By:",
                    options=["Budget", "Duration"],
                    horizontal=True,
                    key="org_budget_tier_mode"
                )

            if chart_type == "Total Budget":
                # Original implementation - Calculate total budget by organization
                org_budget_summary = filtered_df.groupby(org_col).agg({
                    budget_col: 'sum'
                }).reset_index()

                # Sort by budget in descending order
                org_budget_summary = org_budget_summary.sort_values(budget_col, ascending=True)  # ascending=True for horizontal bar chart

                if len(org_budget_summary) > 0:
                    # Create horizontal bar chart
                    fig_portfolio = px.bar(
                        org_budget_summary,
                        x=budget_col,
                        y=org_col,
                        orientation='h',
                        title="Total Budget by Organization",
                        labels={budget_col: f'Total Budget ({currency_symbol}{" " + currency_postfix if currency_postfix else ""})', org_col: 'Organization'},
                        color=budget_col,
                        color_continuous_scale='viridis'
                    )

                    # Update layout for better visualization
                    fig_portfolio.update_layout(
                        height=max(400, len(org_budget_summary) * 40),  # Dynamic height based on number of organizations
                        showlegend=False,
                        xaxis=dict(tickformat=',.0f'),
                        yaxis=dict(title=org_col),
                        coloraxis_showscale=False
                    )

                    # Update traces for better appearance
                    fig_portfolio.update_traces(
                        texttemplate='%{x:,.0f}',
                        textposition='outside',
                        marker_line_width=0
                    )

                    st.plotly_chart(fig_portfolio, width="stretch")

            else:  # Budget by Tier
                # Prepare data with tier information based on selected mode
                tier_df = filtered_df.copy()

                # Get tier configuration based on mode
                if tier_mode == "Budget":
                    tier_config = st.session_state.get('config_dict', {}).get('controls', {}).get('tier_config', {})
                    default_tier_config = {
                        'cutoff_points': [4000, 8000, 15000],
                        'tier_names': ['Tier 1', 'Tier 2', 'Tier 3', 'Tier 4']
                    }
                    cutoffs = tier_config.get('cutoff_points', default_tier_config['cutoff_points'])
                    tier_names_config = tier_config.get('tier_names', default_tier_config['tier_names'])

                    def get_tier(row):
                        budget = row[budget_col]
                        if pd.isna(budget):
                            return "Unknown"
                        elif budget <= cutoffs[0]:
                            return tier_names_config[0]
                        elif budget <= cutoffs[1]:
                            return tier_names_config[1]
                        elif budget <= cutoffs[2]:
                            return tier_names_config[2]
                        else:
                            return tier_names_config[3]

                    tier_df['tier_calculated'] = tier_df.apply(get_tier, axis=1)
                    tier_col = 'tier_calculated'

                else:  # Duration
                    tier_config = st.session_state.get('config_dict', {}).get('controls', {}).get('duration_tier_config', {})
                    default_tier_config = {
                        'cutoff_points': [6, 12, 24],
                        'tier_names': ['Short', 'Medium', 'Long', 'Extra Long']
                    }
                    cutoffs = tier_config.get('cutoff_points', default_tier_config['cutoff_points'])
                    tier_names_config = tier_config.get('tier_names', default_tier_config['tier_names'])

                    def get_tier(row):
                        duration = row.get('original_duration_months', row.get('OD', row.get('Original Duration', 0)))
                        if pd.isna(duration):
                            return "Unknown"
                        duration = int(duration)
                        # Use >= logic like budget tiers (check from high to low)
                        if duration >= cutoffs[2]:
                            return tier_names_config[3]
                        elif duration >= cutoffs[1]:
                            return tier_names_config[2]
                        elif duration >= cutoffs[0]:
                            return tier_names_config[1]
                        else:
                            return tier_names_config[0]

                    tier_df['tier_calculated'] = tier_df.apply(get_tier, axis=1)
                    tier_col = 'tier_calculated'

                if tier_col:
                    # Create pivot table for stacked bar chart
                    pivot_df = tier_df.groupby([org_col, tier_col])[budget_col].sum().reset_index()
                    pivot_wide = pivot_df.pivot(index=org_col, columns=tier_col, values=budget_col).fillna(0)

                    # Sort organizations by total budget
                    pivot_wide['Total'] = pivot_wide.sum(axis=1)
                    pivot_wide = pivot_wide.sort_values('Total', ascending=True)
                    pivot_wide = pivot_wide.drop('Total', axis=1)

                    # Reset index to make organization a column again
                    pivot_wide = pivot_wide.reset_index()

                    # Get tier configuration for colors based on mode
                    if tier_mode == "Budget":
                        tier_config = st.session_state.get('config_dict', {}).get('controls', {}).get('tier_config', {})
                        default_tier_config = {
                            'tier_names': ['Tier 1', 'Tier 2', 'Tier 3', 'Tier 4'],
                            'colors': ['#3498db', '#27ae60', '#f39c12', '#e74c3c']
                        }
                    else:  # Duration
                        tier_config = st.session_state.get('config_dict', {}).get('controls', {}).get('duration_tier_config', {})
                        default_tier_config = {
                            'tier_names': ['Short', 'Medium', 'Long', 'Extra Long'],
                            'colors': ['#3498db', '#27ae60', '#f39c12', '#e74c3c']
                        }

                    tier_names = tier_config.get('tier_names', default_tier_config['tier_names'])
                    tier_colors = tier_config.get('colors', default_tier_config['colors'])

                    # Create color mapping for tiers
                    color_map = {}
                    available_tiers = [col for col in pivot_wide.columns if col != org_col]
                    for i, tier in enumerate(available_tiers):
                        # Try to match tier with configured names
                        tier_index = 0
                        for j, configured_tier in enumerate(tier_names):
                            if configured_tier in tier:
                                tier_index = j
                                break
                        color_map[tier] = tier_colors[tier_index % len(tier_colors)]

                    # Create stacked horizontal bar chart
                    tier_label = f"{tier_mode} Tiers"
                    fig_portfolio = px.bar(
                        pivot_wide,
                        x=available_tiers,
                        y=org_col,
                        orientation='h',
                        title=f"Budget by Organization and {tier_mode} Tier",
                        labels={'value': f'Budget ({currency_symbol}{" " + currency_postfix if currency_postfix else ""})', org_col: 'Organization'},
                        color_discrete_map=color_map
                    )

                    # Update layout for stacked bar chart
                    fig_portfolio.update_layout(
                        height=max(400, len(pivot_wide) * 50),  # Slightly more height for stacked bars
                        showlegend=True,
                        xaxis=dict(tickformat=',.0f'),
                        yaxis=dict(title=org_col),
                        legend=dict(
                            orientation="h",
                            yanchor="bottom",
                            y=1.02,
                            xanchor="right",
                            x=1,
                            title=tier_label
                        ),
                        barmode='stack'
                    )

                    # Update traces for stacked appearance
                    fig_portfolio.update_traces(
                        textposition='inside',
                        texttemplate='%{x:,.0f}',
                        textfont_size=10
                    )

                    st.plotly_chart(fig_portfolio, width="stretch")

                    # Show tier distribution in simple text
                    # Calculate total budget for percentage calculations
                    total_portfolio_budget = sum(pivot_wide[tier].sum() for tier in available_tiers)

                    # Sort available tiers by their position in tier_names configuration
                    def get_tier_order(tier_name):
                        # Find the index of this tier in the configuration
                        for i, configured_tier in enumerate(tier_names):
                            if configured_tier in tier_name:
                                return i
                        return len(tier_names)  # Unknown tiers go to the end

                    ordered_tiers = sorted(available_tiers, key=get_tier_order)

                    # Display tiers in proper order with percentages
                    for tier in ordered_tiers:
                        tier_total = pivot_wide[tier].sum()
                        if tier_total > 0:
                            percentage = (tier_total / total_portfolio_budget * 100) if total_portfolio_budget > 0 else 0
                            st.text(f"{tier}: {format_currency(tier_total, currency_symbol, currency_postfix, thousands=False)} ({percentage:.1f}%)")

                else:
                    st.warning("Budget tier information not available. Please ensure tier configuration is set up.")

            if len(filtered_df) == 0:
                st.info("No organization budget data available to display.")

        except Exception as e:
            st.error(f"Error creating portfolio graph: {str(e)}")
            st.info("Unable to display portfolio graph with current data.")
    else:
        if len(filtered_df) == 0:
            st.info("No projects available for portfolio graph.")
        else:
            st.info("Organization data not available for portfolio graph.")


def render_approvals_chart(filtered_df: pd.DataFrame) -> None:
    """Render the Approvals Chart."""
    # Get currency settings
    currency_symbol = (
        getattr(st.session_state, 'dashboard_currency_symbol', None) or
        getattr(st.session_state, 'currency_symbol', None) or
        st.session_state.get('config_dict', {}).get('controls', {}).get('currency_symbol', '$')
    )
    currency_postfix = (
        getattr(st.session_state, 'dashboard_currency_postfix', None) or
        getattr(st.session_state, 'currency_postfix', None) or
        st.session_state.get('config_dict', {}).get('controls', {}).get('currency_postfix', '')
    )

    if len(filtered_df) > 0:
        # Use specific column names for approvals chart
        start_date_col = 'plan_start'
        budget_col = 'bac'

        if start_date_col in filtered_df.columns and budget_col in filtered_df.columns:
            # Controls for approvals chart
            col1, col2, col3, col4 = st.columns(4)

            with col1:
                approval_time_period = st.selectbox("Approval Time Period",
                                                  options=["Month", "Quarter", "FY"],
                                                  index=0,
                                                  key="approval_time_period",
                                                  help="Month: Monthly approvals, Quarter: Quarterly approvals, FY: Financial Year (July-June)")

            with col2:
                show_tiers = st.toggle("Show Tiers",
                                     value=False,
                                     key="approval_show_tiers",
                                     help="Show stacked chart by tiers")

            with col3:
                tier_mode = st.radio(
                    "Tier By:",
                    options=["Budget", "Duration"],
                    horizontal=True,
                    key="approval_tier_mode",
                    disabled=not show_tiers
                )

            try:
                # Prepare data for approvals chart
                df_approvals = filtered_df.copy()

                # Parse approval date (using Plan Start as proxy for approval)
                df_approvals[start_date_col] = pd.to_datetime(df_approvals[start_date_col], errors='coerce')
                df_approvals = df_approvals.dropna(subset=[start_date_col, budget_col])

                if len(df_approvals) > 0:
                    # Helper functions
                    def get_financial_year_approvals(date):
                        """Get financial year string for a date (FY starts July 1st)"""
                        if date.month >= 7:
                            return f"FY{date.year + 1}"
                        else:
                            return f"FY{date.year}"

                    def get_approval_period_key(date, time_period):
                        """Get period key based on time period selection"""
                        if time_period == "Month":
                            return date.strftime("%b-%Y")
                        elif time_period == "Quarter":
                            quarter = f"Q{((date.month - 1) // 3) + 1}-{date.year}"
                            return quarter
                        else:  # FY
                            return get_financial_year_approvals(date)

                    def get_approval_sort_key(period_str, time_period):
                        """Generate sort key for different time periods"""
                        if time_period == "Month":
                            try:
                                month_abbr, year = period_str.split('-')
                                month_num = {
                                    'Jan': 1, 'Feb': 2, 'Mar': 3, 'Apr': 4, 'May': 5, 'Jun': 6,
                                    'Jul': 7, 'Aug': 8, 'Sep': 9, 'Oct': 10, 'Nov': 11, 'Dec': 12
                                }.get(month_abbr, 1)
                                return (int(year), month_num)
                            except:
                                return (2000, 1)
                        elif time_period == "Quarter":
                            try:
                                quarter, year = period_str.split('-')
                                quarter_num = int(quarter[1:])
                                return (int(year), quarter_num)
                            except:
                                return (2000, 1)
                        else:  # FY
                            try:
                                return (int(period_str[2:]), 1)
                            except:
                                return (2000, 1)

                    # Get tier configuration based on mode
                    if tier_mode == "Budget":
                        tier_config = st.session_state.get('config_dict', {}).get('controls', {}).get('tier_config', {})
                        default_tier_config = {
                            'cutoff_points': [4000, 8000, 15000],
                            'tier_names': ['Tier 1', 'Tier 2', 'Tier 3', 'Tier 4'],
                            'colors': ['#3498db', '#27ae60', '#f39c12', '#e74c3c']
                        }
                        cutoffs = tier_config.get('cutoff_points', default_tier_config['cutoff_points'])
                        tier_names = tier_config.get('tier_names', default_tier_config['tier_names'])
                        tier_colors = tier_config.get('colors', default_tier_config['colors'])

                        def assign_tier(row):
                            """Assign tier based on budget"""
                            budget = row.get(budget_col, 0)
                            if budget >= cutoffs[2]:
                                return tier_names[3]
                            elif budget >= cutoffs[1]:
                                return tier_names[2]
                            elif budget >= cutoffs[0]:
                                return tier_names[1]
                            else:
                                return tier_names[0]

                    else:  # Duration
                        tier_config = st.session_state.get('config_dict', {}).get('controls', {}).get('duration_tier_config', {})
                        default_tier_config = {
                            'cutoff_points': [6, 12, 24],
                            'tier_names': ['Short', 'Medium', 'Long', 'Extra Long'],
                            'colors': ['#3498db', '#27ae60', '#f39c12', '#e74c3c']
                        }
                        cutoffs = tier_config.get('cutoff_points', default_tier_config['cutoff_points'])
                        tier_names = tier_config.get('tier_names', default_tier_config['tier_names'])
                        tier_colors = tier_config.get('colors', default_tier_config['colors'])

                        def assign_tier(row):
                            """Assign tier based on duration"""
                            duration = row.get('original_duration_months', row.get('OD', row.get('Original Duration', 0)))
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

                    # Calculate approvals data
                    approvals_data = []
                    for idx, row in df_approvals.iterrows():
                        approval_date = row[start_date_col]
                        budget = row.get(budget_col, 0)

                        if pd.notna(approval_date) and budget > 0:
                            period_key = get_approval_period_key(approval_date, approval_time_period)

                            # Get financial values
                            ac = row.get('ac', row.get('Actual Cost', 0))
                            ev = row.get('ev', row.get('Earned Value', 0))
                            eac = row.get('eac', row.get('EAC', 0))
                            project_name = row.get('project_name', row.get('Project Name', 'Unknown'))

                            # Assign tier
                            tier = assign_tier(row)

                            approvals_data.append({
                                'Period': period_key,
                                'BAC': budget,
                                'AC': ac,
                                'EV': ev,
                                'EAC': eac,
                                'Project': project_name,
                                'Date': approval_date,
                                'Tier': tier
                            })

                    if approvals_data:
                        # Create DataFrame
                        approvals_df = pd.DataFrame(approvals_data)

                        if show_tiers:
                            # Aggregate by period and tier
                            period_tier_approvals = approvals_df.groupby(['Period', 'Tier']).agg({
                                'BAC': 'sum',
                                'AC': 'sum',
                                'EV': 'sum',
                                'EAC': 'sum',
                                'Project': 'count'
                            }).rename(columns={'Project': 'Number of Projects'}).reset_index()

                            # Sort periods chronologically
                            period_tier_approvals['Sort_Key'] = period_tier_approvals['Period'].apply(
                                lambda x: get_approval_sort_key(x, approval_time_period)
                            )
                            period_tier_approvals = period_tier_approvals.sort_values('Sort_Key').drop('Sort_Key', axis=1)

                            # Create tier color mapping
                            tier_color_map = {tier_names[i]: tier_colors[i] for i in range(len(tier_names))}

                            # Create stacked bar chart by tiers
                            chart_title = f"Project Approvals by Tier - {approval_time_period} View"

                            fig_approvals = px.bar(
                                period_tier_approvals,
                                x='Period',
                                y='BAC',
                                color='Tier',
                                title=chart_title,
                                labels={
                                    'BAC': f'Total BAC ({currency_symbol})',
                                    'Period': approval_time_period,
                                    'Tier': 'Budget Tier'
                                },
                                color_discrete_map=tier_color_map,
                                category_orders={'Tier': tier_names}
                            )

                        else:
                            # Regular aggregation by period only
                            period_approvals = approvals_df.groupby('Period').agg({
                                'BAC': 'sum',
                                'AC': 'sum',
                                'EV': 'sum',
                                'EAC': 'sum',
                                'Project': 'count'
                            }).rename(columns={'Project': 'Number of Projects'}).reset_index()

                            # Calculate percentage columns
                            period_approvals['% AC/BAC'] = (period_approvals['AC'] / period_approvals['BAC'] * 100).fillna(0)
                            period_approvals['% EV/BAC'] = (period_approvals['EV'] / period_approvals['BAC'] * 100).fillna(0)
                            period_approvals['% EAC/BAC'] = (period_approvals['EAC'] / period_approvals['BAC'] * 100).fillna(0)

                            # Sort periods chronologically
                            period_approvals['Sort_Key'] = period_approvals['Period'].apply(
                                lambda x: get_approval_sort_key(x, approval_time_period)
                            )
                            period_approvals = period_approvals.sort_values('Sort_Key').drop('Sort_Key', axis=1)

                            # Create regular bar chart
                            chart_title = f"Project Approvals by BAC - {approval_time_period} View"

                            fig_approvals = px.bar(
                                period_approvals,
                                x='Period',
                                y='BAC',
                                title=chart_title,
                                labels={
                                    'BAC': f'Total BAC ({currency_symbol})',
                                    'Period': approval_time_period
                                },
                                color='BAC',
                                color_continuous_scale='greens'
                            )

                        # Update layout for better visualization
                        fig_approvals.update_layout(
                            height=450,
                            showlegend=False if not show_tiers else True,
                            xaxis=dict(
                                title=approval_time_period,
                                tickangle=45
                            ),
                            yaxis=dict(
                                title=f'Total BAC ({currency_symbol}{" " + currency_postfix if currency_postfix else ""})',
                                tickformat=',.0f'
                            ),
                            coloraxis_showscale=False,
                            margin=dict(t=80, b=60, l=60, r=60)
                        )

                        # Update traces for better appearance
                        fig_approvals.update_traces(
                            texttemplate='%{y:,.0f}',
                            textposition='auto',
                            textfont=dict(size=10),
                            cliponaxis=False
                        )

                        st.plotly_chart(fig_approvals, width="stretch")

                        # Add tier legend when tiers are enabled
                        if show_tiers:
                            st.markdown("**🎯 Tier Legend:**")
                            tier_legend_items = []
                            for i, tier_name in enumerate(tier_names):
                                if i == 0:
                                    range_text = f"< {currency_symbol}{cutoffs[0]:,.0f}"
                                elif i == len(tier_names) - 1:
                                    range_text = f"≥ {currency_symbol}{cutoffs[2]:,.0f}"
                                else:
                                    range_text = f"{currency_symbol}{cutoffs[i-1]:,.0f} - {currency_symbol}{cutoffs[i]:,.0f}"

                                color_emoji = ["🔵", "🟢", "🟠", "🔴"][i]
                                tier_legend_items.append(f"{color_emoji} {tier_name}: {range_text}")

                            st.text(" | ".join(tier_legend_items))

                        # Show detailed data table
                        with st.expander("📊 Detailed Approvals Data", expanded=False):
                            if show_tiers:
                                display_approvals = period_tier_approvals.copy()
                                for col in ['BAC', 'AC', 'EV', 'EAC']:
                                    if col in display_approvals.columns:
                                        display_approvals[col] = display_approvals[col].apply(
                                            lambda x: format_currency(x, currency_symbol, currency_postfix, thousands=False)
                                        )
                            else:
                                display_approvals = period_approvals.copy()
                                for col in ['BAC', 'AC', 'EV', 'EAC']:
                                    if col in display_approvals.columns:
                                        display_approvals[col] = display_approvals[col].apply(
                                            lambda x: format_currency(x, currency_symbol, currency_postfix, thousands=False)
                                        )
                                for col in ['% AC/BAC', '% EV/BAC', '% EAC/BAC']:
                                    if col in display_approvals.columns:
                                        display_approvals[col] = display_approvals[col].apply(lambda x: f"{x:.1f}%")

                            st.dataframe(display_approvals, width='stretch')

                    else:
                        st.warning("No valid approval data could be generated from the selected projects.")

                else:
                    st.warning("No projects have valid approval dates and budget values.")

            except Exception as e:
                st.error(f"Error processing approvals data: {str(e)}")
                st.info("Please check that date and budget columns contain valid values.")

        else:
            st.info("Approvals chart requires 'plan_start' and 'bac' columns.")
    else:
        st.info("No data available for approvals analysis.")


def render_gantt(df: pd.DataFrame, show_predicted: bool, period_choice: str, start_col: str = None) -> None:
    # Sort by start date if column is available
    if start_col and start_col in df.columns:
        df = df.sort_values(start_col).reset_index(drop=True)
    elif "plan_start" in df.columns:
        df = df.sort_values("plan_start").reset_index(drop=True)
    segments = build_segments(df, show_predicted)
    if not segments:
        st.info("No projects match the current filters.")
        return

    seg_df = pd.DataFrame(segments).sort_values(by=["Start", "Finish", "Segment"])

    if "project_id" in df.columns and start_col:
        project_order_df = df[["project_id", start_col]].dropna(subset=[start_col]).copy()
        project_order_df = project_order_df.sort_values(start_col, kind="mergesort")
        category_order = project_order_df["project_id"].astype(str).tolist()
    else:
        task_order = (
            seg_df.groupby("Task")["Start"]
            .min()
            .sort_values()
        )
        category_order = task_order.index.astype(str).tolist()

    if not category_order:
        category_order = seg_df["Task"].astype(str).unique().tolist()

    fig = px.timeline(
        seg_df,
        x_start="Start",
        x_end="Finish",
        y="Task",
        color="Segment",
        color_discrete_map=COLOR_MAP,
        category_orders={"Task": category_order},
        custom_data=["project_name", "organization", "Segment", "bac_formatted", "plan_start", "plan_finish", "cpi", "spi", "percent_budget_used", "percent_time_used", "percent_work_completed"]
    )

    fig.update_traces(
        hovertemplate=(
            "<b>Project ID:</b> %{y}<br>"
            "<b>Project Name:</b> %{customdata[0]}<br>"
            "<b>Organization:</b> %{customdata[1]}<br>"
            "<b>BAC:</b> %{customdata[3]}<br>"
            "<b>Plan Start:</b> %{customdata[4]}<br>"
            "<b>Plan Finish:</b> %{customdata[5]}<br>"
            "<b>CPI:</b> %{customdata[6]:.2f}<br>"
            "<b>SPI:</b> %{customdata[7]:.2f}<br>"
            "<b>% Budget Used:</b> %{customdata[8]:.1f}%<br>"
            "<b>% Time Used:</b> %{customdata[9]:.1f}%<br>"
            "<b>% Work Completed:</b> %{customdata[10]:.1f}%<br>"
            "<b>Segment:</b> %{customdata[2]}<extra></extra>"
        )
    )

    fig.update_yaxes(autorange="reversed")

    period_meta = PERIOD_OPTIONS.get(period_choice, PERIOD_OPTIONS["Month"])

    # Calculate min/max from segment data for end range
    valid_segment_finishes = seg_df["Finish"].dropna()
    if not valid_segment_finishes.empty:
        max_finish = valid_segment_finishes.max()
        # Validate the max finish date is reasonable
        try:
            max_finish_ts = pd.Timestamp(max_finish)
            if max_finish_ts.year < 1980:  # Invalid date
                max_finish = pd.NaT
        except (ValueError, TypeError, pd.errors.OutOfBoundsDatetime):
            max_finish = pd.NaT
    else:
        max_finish = pd.NaT

    # Calculate earliest plan start from original dataframe (not segments)
    earliest_plan_start = None
    if start_col and start_col in df.columns:
        # Get valid dates only (remove NaT values)
        valid_start_dates = df[start_col].dropna()
        if not valid_start_dates.empty:
            earliest_plan_start = valid_start_dates.min()

    # Fallback to segments if no valid start dates found
    if pd.isna(earliest_plan_start) or earliest_plan_start is None:
        valid_segment_starts = seg_df["Start"].dropna()
        if not valid_segment_starts.empty:
            earliest_plan_start = valid_segment_starts.min()

    # Timeline should start 1 quarter (3 months) before earliest plan start
    if pd.notna(earliest_plan_start) and earliest_plan_start is not None:
        try:
            # Ensure we have a valid timestamp
            earliest_plan_start_ts = pd.Timestamp(earliest_plan_start)
            # Check if the timestamp is reasonable (not epoch time)
            if earliest_plan_start_ts.year > 1980:  # Reasonable check for valid dates
                axis_range_start = (earliest_plan_start_ts - pd.DateOffset(months=3)).normalize()
            else:
                # Fallback to current date if we get unreasonable dates
                axis_range_start = (pd.Timestamp.now() - pd.DateOffset(months=3)).normalize()
        except (ValueError, TypeError, pd.errors.OutOfBoundsDatetime):
            # Fallback to current date if conversion fails
            axis_range_start = (pd.Timestamp.now() - pd.DateOffset(months=3)).normalize()
    else:
        # Ultimate fallback
        axis_range_start = (pd.Timestamp.now() - pd.DateOffset(months=3)).normalize()
    # Calculate end range with validation
    if pd.notna(max_finish):
        try:
            timeline_end = pd.Timestamp(max_finish)
            axis_range_end = timeline_end + period_meta["delta"]
            year_end_candidate = timeline_end.to_period('Y').end_time
            if year_end_candidate > axis_range_end:
                axis_range_end = year_end_candidate
            axis_range_end = pd.Timestamp(axis_range_end)
            axis_range_end = axis_range_end.normalize() + pd.Timedelta(days=1)
        except (ValueError, TypeError, pd.errors.OutOfBoundsDatetime):
            # Fallback to a reasonable end date
            axis_range_end = (pd.Timestamp.now() + pd.DateOffset(months=12)).normalize()
    else:
        # Ultimate fallback for end date
        axis_range_end = (pd.Timestamp.now() + pd.DateOffset(months=12)).normalize()
    fig.update_xaxes(
        type="date",
        dtick=period_meta["dtick"],
        tickformat="%b %Y",
        range=[axis_range_start, axis_range_end],
        showline=True,
        linecolor="#000000",
        linewidth=1,
        mirror=True,
        tickfont=dict(color="#000000"),
        title_font=dict(color="#000000")
    )

    if pd.notna(axis_range_start) and pd.notna(axis_range_end):
        axis_start_ts = pd.Timestamp(axis_range_start)
        axis_end_ts = pd.Timestamp(axis_range_end)
        start_year = int(axis_start_ts.year)
        end_year = int(axis_end_ts.year)
        year_boundaries = []
        for year in range(start_year, end_year + 1):
            year_end = pd.Timestamp(year=year, month=12, day=31)
            if year_end < axis_start_ts or year_end > axis_end_ts:
                continue
            year_boundaries.append(year_end)
            fig.add_shape(
                type="line",
                x0=year_end,
                x1=year_end,
                y0=0,
                y1=1,
                xref="x",
                yref="paper",
                line=dict(color=YEAR_LINE_COLOR, dash="dot", width=0.8)
            )
        if period_choice in ("Quarter", "Month"):
            quarter_months = (3, 6, 9, 12)
            for year in range(start_year, end_year + 1):
                for month in quarter_months:
                    quarter_end = pd.Timestamp(year=year, month=month, day=1) + pd.offsets.MonthEnd(0)
                    if quarter_end < axis_start_ts or quarter_end > axis_end_ts:
                        continue
                    if quarter_end in year_boundaries:
                        continue
                    fig.add_shape(
                        type="line",
                        x0=quarter_end,
                        x1=quarter_end,
                        y0=0,
                        y1=1,
                        xref="x",
                        yref="paper",
                        line=dict(color=QUARTER_LINE_COLOR, dash="dot", width=0.4)
                    )

    data_dates = df["data_date"].dropna().unique()
    if data_dates.size:
        data_date = pd.to_datetime(sorted(data_dates)[-1])
        fig.add_shape(
            type="line",
            x0=data_date,
            x1=data_date,
            y0=0,
            y1=1,
            xref="x",
            yref="paper",
            line=dict(color="#facc15", dash="dot", width=2.4)
        )
        fig.add_annotation(
            x=data_date,
            yref="paper",
            y=1.03,
            xref="x",
            text=f"<b>Data Date: {data_date.strftime('%Y-%m-%d')}</b>",
            showarrow=False,
            font=dict(color="#c53030"),
            align="center"
        )

    project_count = df.shape[0]
    if project_count <= 15:
        row_height = 36
    elif project_count <= 30:
        row_height = 24
    elif project_count <= 60:
        row_height = 16
    else:
        row_height = 10

    figure_height = 140 + project_count * row_height
    show_labels = row_height >= 16
    fig.update_layout(
        height=figure_height,
        legend_title_text="",
        margin=dict(l=120, r=40, t=60, b=40),
        plot_bgcolor="#ffffff",
        paper_bgcolor="#ffffff"
    )

    fig.update_yaxes(
        autorange="reversed",
        categoryorder="array",
        categoryarray=category_order,
        showticklabels=show_labels,
        title_text="Project ID" if show_labels else "",
        showline=True,
        linecolor="#000000",
        linewidth=1,
        mirror=True,
        tickfont=dict(color="#000000"),
        title_font=dict(color="#000000")
    )

    st.plotly_chart(fig, width="stretch")


def render_portfolio_heatmap(filtered_df: pd.DataFrame) -> None:
    """
    Render a heatmap showing the cross-tabulation of Budget Tiers × Duration Tiers
    with selectable metrics (Count, BAC, AC, PV, ETC, EAC, CPI, SPI).
    Supports both user-defined tiers (4×4) and custom grid sizes.
    """
    if filtered_df is None or len(filtered_df) == 0:
        st.info("No data available to generate heatmap.")
        return

    # Get currency settings
    currency_symbol = st.session_state.get('config_dict', {}).get('controls', {}).get('currency_symbol', '$')
    currency_postfix = st.session_state.get('config_dict', {}).get('controls', {}).get('currency_postfix', '')

    # Add toggle for grid mode
    st.markdown("**Grid Mode:**")
    grid_mode = st.radio(
        "Select Grid Mode",
        options=["Use Defined Tiers (4×4)", "Use Custom Grid"],
        key="heatmap_grid_mode",
        label_visibility="collapsed",
        horizontal=True
    )

    # Grid size selector for custom mode
    grid_size = 4  # Default
    if grid_mode == "Use Custom Grid":
        st.markdown("**Select Grid Size:**")
        grid_size = st.selectbox(
            "Grid Size",
            options=[4, 6, 8, 10, 12],
            index=2,  # Default to 8×8
            key="heatmap_grid_size",
            label_visibility="collapsed"
        )

    # Prepare data for heatmap (make a copy)
    heatmap_df = filtered_df.copy()

    # Find budget and duration columns
    budget_col = None
    if 'Budget' in heatmap_df.columns:
        budget_col = 'Budget'
    elif 'BAC' in heatmap_df.columns:
        budget_col = 'BAC'
    elif 'bac' in heatmap_df.columns:
        budget_col = 'bac'

    duration_col = 'original_duration_months' if 'original_duration_months' in heatmap_df.columns else None

    # Initialize variables for tier ranges (will be displayed at the end)
    budget_tier_ranges = {}
    duration_tier_ranges = {}

    # MODE 1: Use Defined Tiers (4×4)
    if grid_mode == "Use Defined Tiers (4×4)":
        # Check if both Budget_Category and Duration_Category exist
        has_budget_tiers = 'Budget_Category' in heatmap_df.columns
        has_duration_tiers = 'Duration_Category' in heatmap_df.columns

        # Create Budget_Category if needed
        if not has_budget_tiers and budget_col is not None:
            budget_tier_config = st.session_state.get('config_dict', {}).get('controls', {}).get('tier_config', {})
            default_tier_config = {
                'cutoff_points': [4000, 8000, 15000],
                'tier_names': ['Tier 1', 'Tier 2', 'Tier 3', 'Tier 4']
            }
            budget_cutoffs = budget_tier_config.get('cutoff_points', default_tier_config['cutoff_points'])
            budget_tier_names = budget_tier_config.get('tier_names', default_tier_config['tier_names'])

            def get_budget_tier(budget):
                if pd.isna(budget):
                    return "Unknown"
                if budget < budget_cutoffs[0]:
                    return budget_tier_names[0]
                elif budget < budget_cutoffs[1]:
                    return budget_tier_names[1]
                elif budget < budget_cutoffs[2]:
                    return budget_tier_names[2]
                else:
                    return budget_tier_names[3]

            heatmap_df['Budget_Category'] = heatmap_df[budget_col].apply(get_budget_tier)
            has_budget_tiers = True

        # Create Duration_Category if needed
        if not has_duration_tiers and duration_col is not None:
            duration_tier_config = st.session_state.get('config_dict', {}).get('controls', {}).get('duration_tier_config', {})
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

            heatmap_df['Duration_Category'] = heatmap_df['original_duration_months'].apply(get_duration_tier)
            has_duration_tiers = True

    # MODE 2: Use Custom Grid
    else:  # grid_mode == "Use Custom Grid"
        # Create percentile-based tiers
        if budget_col is None or duration_col is None:
            st.warning("Budget or Duration data not available for custom grid.")
            return

        # Calculate percentile cutoffs for budget
        budget_values = heatmap_df[budget_col].dropna()
        if len(budget_values) == 0:
            st.warning("No budget data available.")
            return

        # Generate percentiles for grid_size bins
        percentiles = [i / grid_size for i in range(1, grid_size)]
        budget_cutoffs = [budget_values.quantile(p) for p in percentiles]

        # Create budget tier labels and ranges
        budget_tier_names = [f'B{i+1}' for i in range(grid_size)]
        for i in range(grid_size):
            if i == 0:
                budget_tier_ranges[budget_tier_names[i]] = f'< {currency_symbol}{budget_cutoffs[0]:,.0f}'
            elif i == grid_size - 1:
                budget_tier_ranges[budget_tier_names[i]] = f'≥ {currency_symbol}{budget_cutoffs[-1]:,.0f}'
            else:
                budget_tier_ranges[budget_tier_names[i]] = f'{currency_symbol}{budget_cutoffs[i-1]:,.0f} - {currency_symbol}{budget_cutoffs[i]:,.0f}'

        # Apply budget tier assignment
        def get_budget_tier_custom(budget):
            if pd.isna(budget):
                return "Unknown"
            for i, cutoff in enumerate(budget_cutoffs):
                if budget < cutoff:
                    return budget_tier_names[i]
            return budget_tier_names[-1]

        heatmap_df['Budget_Category'] = heatmap_df[budget_col].apply(get_budget_tier_custom)

        # Calculate percentile cutoffs for duration
        duration_values = heatmap_df[duration_col].dropna()
        if len(duration_values) == 0:
            st.warning("No duration data available.")
            return

        duration_cutoffs = [duration_values.quantile(p) for p in percentiles]

        # Create duration tier labels and ranges
        duration_tier_names = [f'D{i+1}' for i in range(grid_size)]
        for i in range(grid_size):
            if i == 0:
                duration_tier_ranges[duration_tier_names[i]] = f'< {duration_cutoffs[0]:.0f} mo'
            elif i == grid_size - 1:
                duration_tier_ranges[duration_tier_names[i]] = f'≥ {duration_cutoffs[-1]:.0f} mo'
            else:
                duration_tier_ranges[duration_tier_names[i]] = f'{duration_cutoffs[i-1]:.0f} - {duration_cutoffs[i]:.0f} mo'

        # Apply duration tier assignment
        def get_duration_tier_custom(duration):
            if pd.isna(duration):
                return "Unknown"
            for i, cutoff in enumerate(duration_cutoffs):
                if duration < cutoff:
                    return duration_tier_names[i]
            return duration_tier_names[-1]

        heatmap_df['Duration_Category'] = heatmap_df[duration_col].apply(get_duration_tier_custom)

        has_budget_tiers = True
        has_duration_tiers = True

    # Normalize Budget column name to 'Budget' if it exists as BAC or bac
    if budget_col and budget_col != 'Budget' and 'Budget' not in heatmap_df.columns:
        heatmap_df['Budget'] = heatmap_df[budget_col]

    if not has_budget_tiers or not has_duration_tiers:
        if not has_budget_tiers:
            # Debug info
            available_cols = ', '.join(heatmap_df.columns.tolist()[:10])
            st.warning(f"Budget tier data not available. Budget column not found in data.")
            st.info(f"Available columns (first 10): {available_cols}...")
            st.info("Ensure your data has been loaded and budget tiers are configured in File Management.")
        elif not has_duration_tiers:
            st.info("Duration tier data not available. Ensure duration tiers are configured in File Management or run batch EVM calculation.")
        return

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
            key="heatmap_metric_selector",
            label_visibility="collapsed"
        )

        selected_metric = metric_options[selected_metric_label]

        # Helper function to find column names
        def find_column(possible_names):
            """Find the first matching column name from a list of possibilities"""
            for name in possible_names:
                if name in heatmap_df.columns:
                    return name
            return None

        # Calculate ETC if needed
        if selected_metric == 'ETC' and 'ETC' not in heatmap_df.columns and 'etc' not in heatmap_df.columns:
            eac_col = find_column(['EAC', 'eac', 'estimate_at_completion'])
            ac_col = find_column(['Actual Cost', 'AC', 'ac', 'actual_cost'])

            if eac_col and ac_col:
                heatmap_df['ETC'] = heatmap_df[eac_col] - heatmap_df[ac_col]
            else:
                heatmap_df['ETC'] = 0

        # Handle COUNT metric
        if selected_metric == 'COUNT':
            # Count projects in each Budget × Duration category
            heatmap_result = pd.crosstab(
                heatmap_df['Budget_Category'],
                heatmap_df['Duration_Category']
            )
            value_format = '.0f'
            is_performance_index = False
        else:
            # Map metric to possible column names
            metric_column = None
            if selected_metric == 'BAC':
                metric_column = find_column(['Budget', 'BAC', 'bac'])
            elif selected_metric == 'AC':
                metric_column = find_column(['Actual Cost', 'AC', 'ac', 'actual_cost'])
            elif selected_metric == 'PV':
                metric_column = find_column(['Plan Value', 'PV', 'pv', 'planned_value', 'Planned Value'])
            elif selected_metric == 'ETC':
                metric_column = find_column(['ETC', 'etc', 'estimate_to_complete'])
            elif selected_metric == 'EAC':
                metric_column = find_column(['EAC', 'eac', 'estimate_at_completion'])
            elif selected_metric == 'CPI':
                metric_column = find_column(['CPI', 'cpi', 'cost_performance_index'])
            elif selected_metric == 'SPI':
                metric_column = find_column(['SPI', 'spi', 'schedule_performance_index', 'SPIe', 'spie'])

            # Check if metric column exists
            if metric_column is None:
                st.warning(f"{selected_metric_label} data is not available in the current dataset.")
                return

            # Determine aggregation method
            is_performance_index = selected_metric in ['CPI', 'SPI']

            if is_performance_index:
                # Calculate weighted average by Budget for performance indices
                def weighted_avg(group):
                    if group['Budget'].sum() > 0:
                        return (group[metric_column] * group['Budget']).sum() / group['Budget'].sum()
                    else:
                        return 0

                heatmap_result = heatmap_df.groupby(['Budget_Category', 'Duration_Category']).apply(weighted_avg).unstack(fill_value=0)
                value_format = '.3f'
            else:
                # Sum for monetary values
                heatmap_result = pd.pivot_table(
                    heatmap_df,
                    values=metric_column,
                    index='Budget_Category',
                    columns='Duration_Category',
                    aggfunc='sum',
                    fill_value=0
                )
                value_format = ',.2f'

        # Get tier names and ordering based on mode
        if grid_mode == "Use Defined Tiers (4×4)":
            budget_tier_config = st.session_state.get('config_dict', {}).get('controls', {}).get('tier_config', {})
            budget_tier_names = budget_tier_config.get('tier_names', ['Tier 1', 'Tier 2', 'Tier 3', 'Tier 4'])

            duration_tier_config = st.session_state.get('config_dict', {}).get('controls', {}).get('duration_tier_config', {})
            duration_tier_names = duration_tier_config.get('tier_names', ['Short', 'Medium', 'Long', 'Extra Long'])

            # Reorder rows (Budget Tiers: descending from Tier 4 to Tier 1)
            budget_order = list(reversed(budget_tier_names))
            # Reorder columns (Duration Tiers: ascending)
            duration_order = duration_tier_names
        else:
            # Custom grid mode - tier names already defined earlier
            # budget_tier_names and duration_tier_names are already set as B1-Bn, D1-Dn
            # Reorder rows (Budget: descending from Bn to B1)
            budget_order = list(reversed(budget_tier_names))
            # Reorder columns (Duration: ascending from D1 to Dn)
            duration_order = duration_tier_names

        # Apply reordering
        heatmap_result = heatmap_result.reindex([t for t in budget_order if t in heatmap_result.index])
        heatmap_result = heatmap_result.reindex(columns=[t for t in duration_order if t in heatmap_result.columns])

        # Create the heatmap using plotly
        fig = go.Figure(data=go.Heatmap(
            z=heatmap_result.values,
            x=heatmap_result.columns,
            y=heatmap_result.index,
            colorscale='RdYlGn' if is_performance_index else 'Blues',
            hoverongaps=False,
            hovertemplate='<b>Budget Tier:</b> %{y}<br><b>Duration Tier:</b> %{x}<br><b>Value:</b> %{z:' + value_format + '}<extra></extra>',
            colorbar=dict(
                title=selected_metric_label.split('(')[0].strip(),
                thickness=15,
                len=0.7
            )
        ))

        # Update layout
        fig.update_layout(
            title=f'📊 Portfolio Heatmap: {selected_metric_label}',
            xaxis_title='Duration Tier',
            yaxis_title='Budget Tier',
            height=500,
            font=dict(size=12),
            xaxis=dict(side='bottom'),
            yaxis=dict(autorange='reversed')  # So Tier 4 appears at top
        )

        # Add annotations with values on cells
        annotations = []
        for i, budget_tier in enumerate(heatmap_result.index):
            for j, duration_tier in enumerate(heatmap_result.columns):
                value = heatmap_result.iloc[i, j]
                if is_performance_index:
                    text = f'{value:.3f}'
                elif selected_metric == 'COUNT':
                    text = f'{int(value)}'
                else:
                    text = f'{currency_symbol}{value:,.0f}'

                annotations.append(
                    dict(
                        x=duration_tier,
                        y=budget_tier,
                        text=text,
                        showarrow=False,
                        font=dict(color='white' if value > heatmap_result.values.mean() else 'black', size=11, family='Arial Black')
                    )
                )

        fig.update_layout(annotations=annotations)

        # Display the heatmap
        st.plotly_chart(fig, width='stretch')

        # Add tier definitions
        st.markdown("---")

        if grid_mode == "Use Defined Tiers (4×4)":
            # Display user-defined tier ranges
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

        else:
            # Display custom grid tier ranges
            st.markdown('<p style="color: black; margin-bottom: 5px;"><strong>Tier Definitions:</strong></p>', unsafe_allow_html=True)

            col1, col2 = st.columns(2)

            with col1:
                st.markdown('<p style="color: black; margin-bottom: 3px;"><em>Budget Tiers:</em></p>', unsafe_allow_html=True)
                # Build HTML for budget ranges
                budget_html = '<p style="color: black; font-size: 12px; line-height: 1.4;">'
                for tier_name in budget_tier_names:
                    budget_html += f'{tier_name}: {budget_tier_ranges[tier_name]}<br>'
                budget_html += '</p>'
                st.markdown(budget_html, unsafe_allow_html=True)

            with col2:
                st.markdown('<p style="color: black; margin-bottom: 3px;"><em>Duration Tiers:</em></p>', unsafe_allow_html=True)
                # Build HTML for duration ranges
                duration_html = '<p style="color: black; font-size: 12px; line-height: 1.4;">'
                for tier_name in duration_tier_names:
                    duration_html += f'{tier_name}: {duration_tier_ranges[tier_name]}<br>'
                duration_html += '</p>'
                st.markdown(duration_html, unsafe_allow_html=True)

    except Exception as e:
        st.error(f"Error creating heatmap: {str(e)}")
        st.info("Unable to display heatmap with current data.")


def render_comparison_histogram(filtered_df, metric_options, column_mapping, find_column, currency_symbol):
    """Render comparison view with overlapping histograms and scatter plot."""
    import numpy as np

    # Define logical comparison pairs - grouped by Budget, Duration, Performance
    comparison_pairs = {
        '💰 BAC vs EAC (Planned vs Forecasted Budget)': ('Project Budget (BAC)', 'Estimate at Completion (EAC)'),
        '💰 BAC vs AC (Planned vs Spent Budget)': ('Project Budget (BAC)', 'Actual Cost (AC)'),
        '💰 AC vs ETC (Spent vs Remaining Budget)': ('Actual Cost (AC)', 'Estimate to Complete (ETC)'),
        '⏱️ OD vs LD (Planned vs Forecasted Duration)': ('Project Duration (OD)', 'Likely Duration (LD)'),
        '⏱️ OD vs AD (Planned vs Current Duration)': ('Project Duration (OD)', 'Actual Duration (AD)'),
        '📊 CPI vs SPI (Cost vs Schedule Performance)': ('Cost Efficiency (CPI)', 'Schedule Efficiency (SPI)'),
        '📊 PV vs EV (Planned vs Earned Value)': ('Planned Value (PV)', 'Earned Value (EV)'),
        '🔧 Custom (Select Your Own)': ('custom', 'custom')
    }

    st.markdown("**Select Comparison Type:**")

    selected_pair = st.selectbox(
        "Comparison Pair",
        options=list(comparison_pairs.keys()),
        index=0,  # Default to OD vs LD
        key="histogram_comparison_pair_selector",
        label_visibility="collapsed"
    )

    # Get the metrics for the selected pair
    metric1_label, metric2_label = comparison_pairs[selected_pair]

    # If custom, show individual selectors
    if metric1_label == 'custom':
        st.markdown("**Select Two Metrics to Compare:**")
        col1, col2 = st.columns(2)

        with col1:
            metric1_label = st.selectbox(
                "First Metric",
                options=list(metric_options.keys()),
                index=0,
                key="histogram_metric1_selector"
            )

        with col2:
            metric2_label = st.selectbox(
                "Second Metric",
                options=list(metric_options.keys()),
                index=1,
                key="histogram_metric2_selector"
            )
    else:
        # Show selected pair info
        st.info(f"📊 Comparing: **{metric1_label}** vs **{metric2_label}**")

    metric1 = metric_options[metric1_label]
    metric2 = metric_options[metric2_label]

    # Find actual columns
    actual_column1 = find_column(column_mapping.get(metric1, [metric1]))
    actual_column2 = find_column(column_mapping.get(metric2, [metric2]))

    if not actual_column1:
        st.warning(f"⚠️ {metric1_label} data not available in the dataset.")
        return

    if not actual_column2:
        st.warning(f"⚠️ {metric2_label} data not available in the dataset.")
        return

    # Extract data
    data1 = filtered_df[actual_column1].dropna()
    data2 = filtered_df[actual_column2].dropna()

    if len(data1) == 0 or len(data2) == 0:
        st.warning(f"⚠️ Insufficient data for comparison.")
        return

    # Find common indices (projects that have both metrics)
    common_indices = data1.index.intersection(data2.index)
    if len(common_indices) == 0:
        st.warning("⚠️ No projects have both metrics available for comparison.")
        return

    data1_common = filtered_df.loc[common_indices, actual_column1]
    data2_common = filtered_df.loc[common_indices, actual_column2]

    # Calculate optimal bins
    def calculate_optimal_bins(data):
        n = len(data)
        if n < 2:
            return 10
        q75, q25 = np.percentile(data, [75, 25])
        iqr = q75 - q25
        if iqr == 0:
            return int(np.ceil(np.log2(n) + 1))
        bin_width = 2 * iqr * (n ** (-1/3))
        data_range = data.max() - data.min()
        if bin_width == 0:
            return 10
        num_bins = int(np.ceil(data_range / bin_width))
        return max(10, min(50, num_bins))

    # === OVERLAPPING HISTOGRAMS ===
    st.markdown("### 📊 Overlapping Distribution")

    try:
        fig_overlap = go.Figure()

        # Add first metric histogram
        fig_overlap.add_trace(go.Histogram(
            x=data1,
            name=metric1_label,
            opacity=0.6,
            marker=dict(color='#4A90E2'),
            nbinsx=calculate_optimal_bins(data1)
        ))

        # Add second metric histogram
        fig_overlap.add_trace(go.Histogram(
            x=data2,
            name=metric2_label,
            opacity=0.6,
            marker=dict(color='#E24A4A'),
            nbinsx=calculate_optimal_bins(data2)
        ))

        # Update layout
        fig_overlap.update_layout(
            title=f'Distribution Comparison: {metric1_label} vs {metric2_label}',
            xaxis_title='Value',
            yaxis_title='Number of Projects',
            barmode='overlay',
            height=450,
            hovermode='closest',
            plot_bgcolor='rgba(240, 240, 240, 0.5)',
            paper_bgcolor='white',
            font=dict(size=12),
            legend=dict(
                orientation="h",
                yanchor="bottom",
                y=1.02,
                xanchor="right",
                x=1
            )
        )

        fig_overlap.update_xaxes(showgrid=True, gridwidth=1, gridcolor='rgba(200, 200, 200, 0.3)')
        fig_overlap.update_yaxes(showgrid=True, gridwidth=1, gridcolor='rgba(200, 200, 200, 0.3)')

        st.plotly_chart(fig_overlap, width='stretch')

    except Exception as e:
        st.error(f"Error creating overlapping histogram: {str(e)}")

    # === SCATTER PLOT WITH 45° LINE ===
    st.markdown("### 📈 Project-by-Project Comparison")

    try:
        # Get project names if available
        project_col = find_column(['Project', 'project', 'Project Name', 'project_name'])
        project_names = filtered_df.loc[common_indices, project_col] if project_col else [f"Project {i+1}" for i in range(len(common_indices))]

        fig_scatter = go.Figure()

        # Add scatter points
        fig_scatter.add_trace(go.Scatter(
            x=data1_common,
            y=data2_common,
            mode='markers',
            name='Projects',
            marker=dict(
                size=10,
                color='#4A90E2',
                opacity=0.7,
                line=dict(width=1, color='white')
            ),
            text=project_names,
            hovertemplate=(
                '<b>%{text}</b><br>' +
                f'<b>{metric1_label}:</b> %{{x:,.2f}}<br>' +
                f'<b>{metric2_label}:</b> %{{y:,.2f}}<br>' +
                '<extra></extra>'
            )
        ))

        # Add 45° reference line (perfect match)
        min_val = min(data1_common.min(), data2_common.min())
        max_val = max(data1_common.max(), data2_common.max())

        fig_scatter.add_trace(go.Scatter(
            x=[min_val, max_val],
            y=[min_val, max_val],
            mode='lines',
            name='Perfect Match (45° line)',
            line=dict(color='red', width=2, dash='dash'),
            hoverinfo='skip'
        ))

        # Determine position labels based on comparison type
        if metric1_label == 'Project Budget (BAC)' and metric2_label == 'Estimate at Completion (EAC)':
            above_label = 'Over Budget'
            below_label = 'Under Budget'
        elif metric1_label == 'Project Budget (BAC)' and metric2_label == 'Actual Cost (AC)':
            above_label = 'Overspent'
            below_label = 'Under Spent'
        elif metric1_label == 'Actual Cost (AC)' and metric2_label == 'Estimate to Complete (ETC)':
            above_label = 'More Remaining'
            below_label = 'Less Remaining'
        elif metric1_label == 'Project Duration (OD)' and metric2_label == 'Likely Duration (LD)':
            above_label = 'Delayed'
            below_label = 'Ahead of Schedule'
        elif metric1_label == 'Project Duration (OD)' and metric2_label == 'Actual Duration (AD)':
            above_label = 'Duration Overrun'
            below_label = 'Within Planned Duration'
        elif metric1_label == 'Cost Efficiency (CPI)' and metric2_label == 'Schedule Efficiency (SPI)':
            above_label = 'Better Schedule'
            below_label = 'Better Cost'
        elif metric1_label == 'Planned Value (PV)' and metric2_label == 'Earned Value (EV)':
            above_label = 'Ahead of Plan'
            below_label = 'Behind Plan'
        else:
            above_label = f'{metric2_label} > {metric1_label}'
            below_label = f'{metric2_label} < {metric1_label}'

        # Add annotations for regions
        fig_scatter.add_annotation(
            x=max_val * 0.8,
            y=max_val * 0.95,
            text=above_label,
            showarrow=False,
            font=dict(size=11, color='red'),
            bgcolor='rgba(255, 200, 200, 0.3)',
            borderpad=4
        )

        fig_scatter.add_annotation(
            x=max_val * 0.95,
            y=max_val * 0.8,
            text=below_label,
            showarrow=False,
            font=dict(size=11, color='green'),
            bgcolor='rgba(200, 255, 200, 0.3)',
            borderpad=4
        )

        # Update layout
        fig_scatter.update_layout(
            title=f'{metric1_label} vs {metric2_label} ({len(common_indices)} Projects)',
            xaxis_title=metric1_label,
            yaxis_title=metric2_label,
            height=500,
            hovermode='closest',
            plot_bgcolor='rgba(240, 240, 240, 0.5)',
            paper_bgcolor='white',
            font=dict(size=12),
            showlegend=True,
            legend=dict(
                orientation="h",
                yanchor="bottom",
                y=-0.2,
                xanchor="center",
                x=0.5
            )
        )

        fig_scatter.update_xaxes(showgrid=True, gridwidth=1, gridcolor='rgba(200, 200, 200, 0.3)')
        fig_scatter.update_yaxes(showgrid=True, gridwidth=1, gridcolor='rgba(200, 200, 200, 0.3)')

        st.plotly_chart(fig_scatter, width='stretch')

        # Statistics
        col1, col2, col3, col4 = st.columns(4)

        with col1:
            st.metric("Projects Compared", len(common_indices))

        with col2:
            correlation = np.corrcoef(data1_common, data2_common)[0, 1]
            st.metric("Correlation", f"{correlation:.3f}")

        with col3:
            above_line = (data2_common > data1_common).sum()
            st.metric(above_label, f"{above_line} ({above_line/len(common_indices)*100:.1f}%)")

        with col4:
            below_line = (data2_common < data1_common).sum()
            st.metric(below_label, f"{below_line} ({below_line/len(common_indices)*100:.1f}%)")

    except Exception as e:
        st.error(f"Error creating scatter plot: {str(e)}")


def render_histogram(filtered_df: pd.DataFrame) -> None:
    """Render histogram for portfolio metrics distribution."""
    import numpy as np

    if filtered_df is None or filtered_df.empty:
        st.info("No data available for histogram.")
        return

    # Add toggle for Single vs Compare mode
    col1, col2 = st.columns([1, 3])
    with col1:
        comparison_mode = st.toggle("Compare Two Metrics", value=False, key="histogram_comparison_toggle")

    with col2:
        if comparison_mode:
            st.caption("📊 Compare distributions of two metrics side-by-side")
        else:
            st.caption("📊 Analyze distribution of a single metric")

    # Metric options
    metric_options = {
        'Project Duration (OD)': 'original_duration_months',
        'Actual Duration (AD)': 'actual_duration_months',
        'Likely Duration (LD)': 'ld',
        'Project Budget (BAC)': 'BAC',
        'Actual Cost (AC)': 'AC',
        'Estimate at Completion (EAC)': 'EAC',
        'Cost Efficiency (CPI)': 'CPI',
        'Schedule Efficiency (SPI)': 'SPI',
        'Estimate to Complete (ETC)': 'ETC',
        'Planned Value (PV)': 'PV',
        'Earned Value (EV)': 'EV'
    }

    # Helper function to find column names (case-insensitive)
    def find_column(possible_names):
        """Find the first matching column name from a list of possibilities"""
        for name in possible_names:
            if name in filtered_df.columns:
                return name
        return None

    # Map metric to possible column names
    column_mapping = {
        'original_duration_months': ['original_duration_months', 'OD', 'Original Duration'],
        'actual_duration_months': ['actual_duration_months', 'AD', 'Actual Duration'],
        'ld': ['ld', 'LD', 'likely_duration', 'Likely Duration'],
        'BAC': ['BAC', 'bac', 'Budget'],
        'AC': ['AC', 'ac', 'Actual Cost', 'actual_cost'],
        'EAC': ['eac', 'EAC', 'estimate_at_completion'],
        'CPI': ['cpi', 'CPI', 'cost_performance_index'],
        'SPI': ['spi', 'SPI', 'schedule_performance_index'],
        'ETC': ['etc', 'ETC', 'estimate_to_complete'],
        'PV': ['pv', 'PV', 'Planned Value', 'planned_value'],
        'EV': ['ev', 'EV', 'Earned Value', 'earned_value']
    }

    # Get currency symbol
    currency_symbol = '$'
    if 'config_dict' in st.session_state and 'controls' in st.session_state.config_dict:
        currency_symbol = st.session_state.config_dict['controls'].get('currency_symbol', '$')

    # Branch based on mode
    if comparison_mode:
        # COMPARISON MODE: Two metrics
        render_comparison_histogram(filtered_df, metric_options, column_mapping, find_column, currency_symbol)
    else:
        # SINGLE METRIC MODE: Original behavior
        st.markdown("**Select Metric for Distribution Analysis:**")

        selected_metric_label = st.selectbox(
            "Metric",
            options=list(metric_options.keys()),
            key="histogram_metric_selector",
            label_visibility="collapsed"
        )

        selected_metric = metric_options[selected_metric_label]

        # Find the actual column name
        actual_column = find_column(column_mapping.get(selected_metric, [selected_metric]))

        if not actual_column:
            st.warning(f"⚠️ {selected_metric_label} data not available in the dataset.")
            st.info("Ensure batch EVM calculation has been run to generate performance metrics.")
            return

        # Extract data and remove NaN values
        data = filtered_df[actual_column].dropna()

        if len(data) == 0:
            st.warning(f"⚠️ No valid data available for {selected_metric_label}.")
            return

        # Calculate optimal number of bins using Freedman-Diaconis rule
        # This is considered best practice for histogram binning
        def calculate_optimal_bins(data):
            """Calculate optimal number of bins using Freedman-Diaconis rule."""
            n = len(data)
            if n < 2:
                return 10  # Default if too few data points

            # Freedman-Diaconis rule: bin_width = 2 * IQR * n^(-1/3)
            q75, q25 = np.percentile(data, [75, 25])
            iqr = q75 - q25

            if iqr == 0:
                # Fallback to Sturges' rule if IQR is 0
                return int(np.ceil(np.log2(n) + 1))

            bin_width = 2 * iqr * (n ** (-1/3))
            data_range = data.max() - data.min()

            if bin_width == 0:
                return 10  # Default

            num_bins = int(np.ceil(data_range / bin_width))

            # Constrain between 10 and 50 bins for readability
            return max(10, min(50, num_bins))
    
        optimal_bins = calculate_optimal_bins(data)
    
        # Get currency symbol if needed
        currency_symbol = '$'
        if 'config_dict' in st.session_state and 'controls' in st.session_state.config_dict:
            currency_symbol = st.session_state.config_dict['controls'].get('currency_symbol', '$')
    
        # Find BAC column for tooltip
        bac_column = find_column(['BAC', 'bac', 'Budget'])
    
        # Create histogram
        try:
            # Create a temporary dataframe to align BAC values with the metric
            temp_df = pd.DataFrame({
                'metric_value': data.values,
                'bac': filtered_df.loc[data.index, bac_column].values if bac_column else np.zeros(len(data))
            })
    
            # Calculate bin edges
            bin_edges = np.histogram_bin_edges(data, bins=optimal_bins)
    
            # Assign each data point to a bin and calculate totals per bin
            temp_df['bin'] = pd.cut(temp_df['metric_value'], bins=bin_edges, include_lowest=True)
            bin_stats = temp_df.groupby('bin', observed=True).agg(
                count=('metric_value', 'size'),
                total_bac=('bac', 'sum')
            ).reset_index()
    
            # Get bin centers for plotting
            bin_stats['bin_center'] = bin_stats['bin'].apply(lambda x: x.mid)
            bin_stats['bin_left'] = bin_stats['bin'].apply(lambda x: x.left)
            bin_stats['bin_right'] = bin_stats['bin'].apply(lambda x: x.right)
    
            # Determine axis labels and formatting based on metric
            if selected_metric == 'original_duration_months':
                x_label = 'Project Duration (months)'
                hover_template = (
                    '<b>Duration Range:</b> %{customdata[0]:.1f} - %{customdata[1]:.1f} months<br>'
                    '<b>Number of Projects:</b> %{y}<br>'
                    '<b>Total BAC:</b> ' + currency_symbol + '%{customdata[2]:,.0f}<extra></extra>'
                )
            elif selected_metric == 'BAC':
                x_label = f'Project Budget ({currency_symbol})'
                hover_template = (
                    f'<b>Budget Range:</b> {currency_symbol}%{{customdata[0]:,.0f}} - {currency_symbol}%{{customdata[1]:,.0f}}<br>'
                    '<b>Number of Projects:</b> %{y}<br>'
                    '<b>Total BAC:</b> ' + currency_symbol + '%{customdata[2]:,.0f}<extra></extra>'
                )
            elif selected_metric in ['CPI', 'SPI']:
                x_label = selected_metric_label
                hover_template = (
                    f'<b>{selected_metric} Range:</b> %{{customdata[0]:.2f}} - %{{customdata[1]:.2f}}<br>'
                    '<b>Number of Projects:</b> %{y}<br>'
                    '<b>Total BAC:</b> ' + currency_symbol + '%{customdata[2]:,.0f}<extra></extra>'
                )
            else:  # ETC, EAC
                x_label = f'{selected_metric_label} ({currency_symbol})'
                hover_template = (
                    f'<b>{selected_metric} Range:</b> {currency_symbol}%{{customdata[0]:,.0f}} - {currency_symbol}%{{customdata[1]:,.0f}}<br>'
                    '<b>Number of Projects:</b> %{y}<br>'
                    '<b>Total BAC:</b> ' + currency_symbol + '%{customdata[2]:,.0f}<extra></extra>'
                )
    
            fig = go.Figure()
    
            # Add bar trace (instead of histogram) for better control over tooltips
            fig.add_trace(go.Bar(
                x=bin_stats['bin_center'],
                y=bin_stats['count'],
                width=(bin_edges[1] - bin_edges[0]) * 0.9,  # 90% of bin width
                name='Projects',
                marker=dict(
                    color='#4A90E2',
                    line=dict(color='white', width=1)
                ),
                customdata=np.column_stack((
                    bin_stats['bin_left'],
                    bin_stats['bin_right'],
                    bin_stats['total_bac']
                )),
                hovertemplate=hover_template
            ))
    
            # Calculate statistics
            mean_val = data.mean()
            median_val = data.median()
    
            # Add mean line
            fig.add_vline(
                x=mean_val,
                line_dash="dash",
                line_color="red",
                annotation_text=f"Mean: {mean_val:.2f}" if selected_metric in ['CPI', 'SPI'] else f"Mean: {mean_val:,.0f}",
                annotation_position="top"
            )
    
            # Add median line
            fig.add_vline(
                x=median_val,
                line_dash="dot",
                line_color="green",
                annotation_text=f"Median: {median_val:.2f}" if selected_metric in ['CPI', 'SPI'] else f"Median: {median_val:,.0f}",
                annotation_position="bottom"
            )
    
            # Update layout
            fig.update_layout(
                title=f'Distribution of {selected_metric_label} ({len(data)} Projects)',
                xaxis_title=x_label,
                yaxis_title='Number of Projects',
                showlegend=False,
                height=500,
                hovermode='closest',
                plot_bgcolor='rgba(240, 240, 240, 0.5)',
                paper_bgcolor='white',
                font=dict(size=12)
            )
    
            # Update axes
            fig.update_xaxes(showgrid=True, gridwidth=1, gridcolor='rgba(200, 200, 200, 0.3)')
            fig.update_yaxes(showgrid=True, gridwidth=1, gridcolor='rgba(200, 200, 200, 0.3)')
    
            # Display the histogram
            st.plotly_chart(fig, width='stretch')
    
            # Display statistics
            col1, col2, col3, col4 = st.columns(4)
    
            with col1:
                st.metric("Mean", f"{mean_val:.2f}" if selected_metric in ['CPI', 'SPI'] else f"{currency_symbol}{mean_val:,.0f}" if selected_metric in ['BAC', 'ETC', 'EAC'] else f"{mean_val:,.1f}")
    
            with col2:
                st.metric("Median", f"{median_val:.2f}" if selected_metric in ['CPI', 'SPI'] else f"{currency_symbol}{median_val:,.0f}" if selected_metric in ['BAC', 'ETC', 'EAC'] else f"{median_val:,.1f}")
    
            with col3:
                std_val = data.std()
                st.metric("Std Dev", f"{std_val:.2f}" if selected_metric in ['CPI', 'SPI'] else f"{currency_symbol}{std_val:,.0f}" if selected_metric in ['BAC', 'ETC', 'EAC'] else f"{std_val:,.1f}")
    
            with col4:
                st.metric("Projects", f"{len(data)}")
    
            # Additional insights
            with st.expander("📊 Distribution Insights"):
                st.markdown(f"**Range:** {data.min():.2f} to {data.max():.2f}" if selected_metric in ['CPI', 'SPI'] else
                           f"**Range:** {currency_symbol}{data.min():,.0f} to {currency_symbol}{data.max():,.0f}" if selected_metric in ['BAC', 'ETC', 'EAC'] else
                           f"**Range:** {data.min():.1f} to {data.max():.1f}")
                st.markdown(f"**Optimal Bins:** {optimal_bins} (calculated using Freedman-Diaconis rule)")
                st.markdown(f"**25th Percentile:** {data.quantile(0.25):.2f}" if selected_metric in ['CPI', 'SPI'] else
                           f"**25th Percentile:** {currency_symbol}{data.quantile(0.25):,.0f}" if selected_metric in ['BAC', 'ETC', 'EAC'] else
                           f"**25th Percentile:** {data.quantile(0.25):.1f}")
                st.markdown(f"**75th Percentile:** {data.quantile(0.75):.2f}" if selected_metric in ['CPI', 'SPI'] else
                           f"**75th Percentile:** {currency_symbol}{data.quantile(0.75):,.0f}" if selected_metric in ['BAC', 'ETC', 'EAC'] else
                           f"**75th Percentile:** {data.quantile(0.75):.1f}")
    
        except Exception as e:
            st.error(f"Error creating histogram: {str(e)}")
            st.info("Unable to display histogram with current data.")
    
    
def render_footer():
    st.markdown(
        """
        <div class="footer">
            <div style="border-top: 1px solid rgba(0,0,0,0.1); padding-top: 1rem; margin-top: 2rem;">
                <strong>Project Portfolio Intelligence Suite</strong> • Schedule Performance Overview<br>
                Generated on {date} • Confidential Executive Report
            </div>
        </div>
        """.format(date=datetime.now().strftime('%B %d, %Y at %I:%M %p')),
        unsafe_allow_html=True
    )


def create_performance_bins(df: pd.DataFrame, column: str, bin_type: str = 'budget') -> pd.DataFrame:
    """
    Categorize projects into performance bins based on % budget used or % time used.

    Args:
        df: DataFrame with project data
        column: Column name to use for binning ('percent_budget_used' or 'percent_time_used')
        bin_type: 'budget' (11 bins) or 'time' (12 bins with Over 100%)

    Returns:
        DataFrame with added 'bin_category' column
    """
    result_df = df.copy()

    # Ensure column exists and is numeric
    if column not in result_df.columns:
        result_df['bin_category'] = 'Unknown'
        return result_df

    result_df[column] = pd.to_numeric(result_df[column], errors='coerce')

    def assign_bin(value):
        if pd.isna(value):
            return 'Unknown'

        if value == 0:
            return '0%'
        elif value <= 10:
            return '1-10%'
        elif value <= 20:
            return '11-20%'
        elif value <= 30:
            return '21-30%'
        elif value <= 40:
            return '31-40%'
        elif value <= 50:
            return '41-50%'
        elif value <= 60:
            return '51-60%'
        elif value <= 70:
            return '61-70%'
        elif value <= 80:
            return '71-80%'
        elif value <= 90:
            return '81-90%'
        elif value <= 100:
            return '91-100%'
        else:
            # Over 100% only for time bins
            if bin_type == 'time':
                return 'Over 100%'
            else:
                return '91-100%'  # Cap budget at 100%

    result_df['bin_category'] = result_df[column].apply(assign_bin)
    return result_df


def render_budget_performance(filtered_df: pd.DataFrame) -> None:
    """Render Budget Performance expander with binned analysis."""
    if filtered_df is None or filtered_df.empty:
        st.info("No data available for budget performance analysis.")
        return

    st.markdown("**Analyze project distribution across budget utilization ranges**")

    # Controls
    col1, col2 = st.columns(2)
    with col1:
        metric_mode = st.radio(
            "Chart Metric:",
            options=["Count of Projects", "Sum of Project Budget"],
            horizontal=True,
            key="budget_perf_metric"
        )

    with col2:
        view_mode = st.radio(
            "View Mode:",
            options=["Table View", "Bar Chart"],
            horizontal=True,
            key="budget_perf_view"
        )

    # Get currency symbol
    currency_symbol = '$'
    if 'config_dict' in st.session_state and 'controls' in st.session_state.config_dict:
        currency_symbol = st.session_state.config_dict['controls'].get('currency_symbol', '$')

    # Create bins
    binned_df = create_performance_bins(filtered_df, 'percent_budget_used', 'budget')

    # Define bin order (11 bins for budget)
    bin_order = ['0%', '1-10%', '11-20%', '21-30%', '31-40%', '41-50%',
                 '51-60%', '61-70%', '71-80%', '81-90%', '91-100%', 'Unknown']

    # Calculate statistics for each bin
    bin_stats = []
    total_count = len(binned_df)

    # Ensure BAC column exists and is numeric
    bac_col = None
    for col in ['BAC', 'bac', 'budget_at_completion']:
        if col in binned_df.columns:
            bac_col = col
            break

    if bac_col:
        binned_df[bac_col] = pd.to_numeric(binned_df[bac_col], errors='coerce').fillna(0)
        total_budget = binned_df[bac_col].sum()
    else:
        total_budget = 0

    for bin_name in bin_order:
        bin_projects = binned_df[binned_df['bin_category'] == bin_name]
        count = len(bin_projects)
        count_pct = (count / total_count * 100) if total_count > 0 else 0

        if bac_col:
            budget_sum = bin_projects[bac_col].sum()
            budget_pct = (budget_sum / total_budget * 100) if total_budget > 0 else 0
        else:
            budget_sum = 0
            budget_pct = 0

        bin_stats.append({
            'Bin Range': bin_name,
            'Count': count,
            'Count %': count_pct,
            'Budget': budget_sum,
            'Budget %': budget_pct
        })

    stats_df = pd.DataFrame(bin_stats)

    # Remove 'Unknown' row if it has zero count
    stats_df = stats_df[stats_df['Count'] > 0]

    # Display based on view mode
    if view_mode == "Table View":
        # Format table for display
        display_df = stats_df.copy()
        display_df['Count %'] = display_df['Count %'].apply(lambda x: f"{x:.1f}%")
        display_df['Budget'] = display_df['Budget'].apply(lambda x: format_currency(x, symbol=currency_symbol))
        display_df['Budget %'] = display_df['Budget %'].apply(lambda x: f"{x:.1f}%")

        st.dataframe(display_df, width='stretch', hide_index=True)

        # CSV Export
        csv = stats_df.copy()
        csv['Count %'] = csv['Count %'].apply(lambda x: f"{x:.1f}")
        csv['Budget %'] = csv['Budget %'].apply(lambda x: f"{x:.1f}")
        csv_data = csv.to_csv(index=False)

        st.download_button(
            "📥 Download CSV",
            csv_data,
            file_name=f"budget_performance_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
            mime="text/csv"
        )

    else:  # Bar Chart
        # Prepare chart data
        chart_df = stats_df[stats_df['Bin Range'] != 'Unknown'].copy()

        if metric_mode == "Count of Projects":
            y_values = chart_df['Count']
            labels = chart_df['Count %'].apply(lambda x: f"{x:.1f}%")
            y_title = "Number of Projects"
            hover_template = "<b>%{x}</b><br>Count: %{y}<br>Percentage: %{text}<extra></extra>"
        else:  # Sum of Project Budget
            y_values = chart_df['Budget']
            labels = chart_df['Budget %'].apply(lambda x: f"{x:.1f}%")
            y_title = f"Total Budget ({currency_symbol})"
            hover_template = f"<b>%{{x}}</b><br>Budget: {currency_symbol}%{{y:,.0f}}<br>Percentage: %{{text}}<extra></extra>"

        fig = go.Figure()

        fig.add_trace(go.Bar(
            x=chart_df['Bin Range'],
            y=y_values,
            text=labels,
            textposition='outside',
            marker=dict(
                color='#3498db',
                line=dict(color='#2980b9', width=1)
            ),
            hovertemplate=hover_template
        ))

        fig.update_layout(
            title=f"Budget Performance Distribution - {metric_mode}",
            xaxis_title="Budget Used Range",
            yaxis_title=y_title,
            height=500,
            showlegend=False,
            hovermode='x',
            plot_bgcolor='rgba(0,0,0,0)',
            paper_bgcolor='rgba(0,0,0,0)'
        )

        fig.update_xaxes(showgrid=False)
        fig.update_yaxes(showgrid=True, gridcolor='rgba(128,128,128,0.2)')

        st.plotly_chart(fig, width='stretch')


def render_time_performance(filtered_df: pd.DataFrame) -> None:
    """Render Time Performance expander with binned analysis."""
    if filtered_df is None or filtered_df.empty:
        st.info("No data available for time performance analysis.")
        return

    st.markdown("**Analyze project distribution across time utilization ranges**")

    # Controls
    col1, col2 = st.columns(2)
    with col1:
        metric_mode = st.radio(
            "Chart Metric:",
            options=["Count of Projects", "Sum of Project Budget"],
            horizontal=True,
            key="time_perf_metric"
        )

    with col2:
        view_mode = st.radio(
            "View Mode:",
            options=["Table View", "Bar Chart"],
            horizontal=True,
            key="time_perf_view"
        )

    # Get currency symbol
    currency_symbol = '$'
    if 'config_dict' in st.session_state and 'controls' in st.session_state.config_dict:
        currency_symbol = st.session_state.config_dict['controls'].get('currency_symbol', '$')

    # Create bins
    binned_df = create_performance_bins(filtered_df, 'percent_time_used', 'time')

    # Define bin order (12 bins for time, including Over 100%)
    bin_order = ['0%', '1-10%', '11-20%', '21-30%', '31-40%', '41-50%',
                 '51-60%', '61-70%', '71-80%', '81-90%', '91-100%', 'Over 100%', 'Unknown']

    # Calculate statistics for each bin
    bin_stats = []
    total_count = len(binned_df)

    # Ensure BAC column exists and is numeric
    bac_col = None
    for col in ['BAC', 'bac', 'budget_at_completion']:
        if col in binned_df.columns:
            bac_col = col
            break

    if bac_col:
        binned_df[bac_col] = pd.to_numeric(binned_df[bac_col], errors='coerce').fillna(0)
        total_budget = binned_df[bac_col].sum()
    else:
        total_budget = 0

    for bin_name in bin_order:
        bin_projects = binned_df[binned_df['bin_category'] == bin_name]
        count = len(bin_projects)
        count_pct = (count / total_count * 100) if total_count > 0 else 0

        if bac_col:
            budget_sum = bin_projects[bac_col].sum()
            budget_pct = (budget_sum / total_budget * 100) if total_budget > 0 else 0
        else:
            budget_sum = 0
            budget_pct = 0

        bin_stats.append({
            'Bin Range': bin_name,
            'Count': count,
            'Count %': count_pct,
            'Budget': budget_sum,
            'Budget %': budget_pct
        })

    stats_df = pd.DataFrame(bin_stats)

    # Remove 'Unknown' row if it has zero count
    stats_df = stats_df[stats_df['Count'] > 0]

    # Display based on view mode
    if view_mode == "Table View":
        # Format table for display
        display_df = stats_df.copy()
        display_df['Count %'] = display_df['Count %'].apply(lambda x: f"{x:.1f}%")
        display_df['Budget'] = display_df['Budget'].apply(lambda x: format_currency(x, symbol=currency_symbol))
        display_df['Budget %'] = display_df['Budget %'].apply(lambda x: f"{x:.1f}%")

        st.dataframe(display_df, width='stretch', hide_index=True)

        # CSV Export
        csv = stats_df.copy()
        csv['Count %'] = csv['Count %'].apply(lambda x: f"{x:.1f}")
        csv['Budget %'] = csv['Budget %'].apply(lambda x: f"{x:.1f}")
        csv_data = csv.to_csv(index=False)

        st.download_button(
            "📥 Download CSV",
            csv_data,
            file_name=f"time_performance_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
            mime="text/csv"
        )

    else:  # Bar Chart
        # Prepare chart data
        chart_df = stats_df[stats_df['Bin Range'] != 'Unknown'].copy()

        if metric_mode == "Count of Projects":
            y_values = chart_df['Count']
            labels = chart_df['Count %'].apply(lambda x: f"{x:.1f}%")
            y_title = "Number of Projects"
            hover_template = "<b>%{x}</b><br>Count: %{y}<br>Percentage: %{text}<extra></extra>"
        else:  # Sum of Project Budget
            y_values = chart_df['Budget']
            labels = chart_df['Budget %'].apply(lambda x: f"{x:.1f}%")
            y_title = f"Total Budget ({currency_symbol})"
            hover_template = f"<b>%{{x}}</b><br>Budget: {currency_symbol}%{{y:,.0f}}<br>Percentage: %{{text}}<extra></extra>"

        fig = go.Figure()

        fig.add_trace(go.Bar(
            x=chart_df['Bin Range'],
            y=y_values,
            text=labels,
            textposition='outside',
            marker=dict(
                color='#9b59b6',
                line=dict(color='#8e44ad', width=1)
            ),
            hovertemplate=hover_template
        ))

        fig.update_layout(
            title=f"Time Performance Distribution - {metric_mode}",
            xaxis_title="Time Used Range",
            yaxis_title=y_title,
            height=500,
            showlegend=False,
            hovermode='x',
            plot_bgcolor='rgba(0,0,0,0)',
            paper_bgcolor='rgba(0,0,0,0)'
        )

        fig.update_xaxes(showgrid=False)
        fig.update_yaxes(showgrid=True, gridcolor='rgba(128,128,128,0.2)')

        st.plotly_chart(fig, width='stretch')


def render_budget_time_matrix(filtered_df: pd.DataFrame) -> None:
    """Render Budget-Time Performance Matrix expander with 2D binned analysis."""
    if filtered_df is None or filtered_df.empty:
        st.info("No data available for budget-time matrix analysis.")
        return

    st.markdown("**2D matrix showing project distribution across budget and time utilization**")

    # Controls
    col1, col2, col3 = st.columns(3)

    with col1:
        metric_mode = st.radio(
            "Metric:",
            options=["Count of Projects", "Sum of Project Budget"],
            horizontal=False,
            key="matrix_metric"
        )

    with col2:
        value_display = st.radio(
            "Cell Values:",
            options=["Absolute Values", "% of Portfolio", "% of Row", "% of Column"],
            horizontal=False,
            key="matrix_value_display"
        )

    with col3:
        granularity = st.radio(
            "Bin Granularity:",
            options=["Full (11×12)", "Simplified (6×6)"],
            horizontal=False,
            key="matrix_granularity"
        )

    # Get currency symbol
    currency_symbol = '$'
    if 'config_dict' in st.session_state and 'controls' in st.session_state.config_dict:
        currency_symbol = st.session_state.config_dict['controls'].get('currency_symbol', '$')

    # Ensure required columns exist and are numeric
    required_cols = ['percent_budget_used', 'percent_time_used']
    for col in required_cols:
        if col not in filtered_df.columns:
            st.warning(f"⚠️ Required column '{col}' not found in data.")
            return
        filtered_df[col] = pd.to_numeric(filtered_df[col], errors='coerce')

    # Get BAC column
    bac_col = None
    for col in ['BAC', 'bac', 'budget_at_completion']:
        if col in filtered_df.columns:
            bac_col = col
            break

    if bac_col:
        filtered_df[bac_col] = pd.to_numeric(filtered_df[bac_col], errors='coerce').fillna(0)

    # Define bins based on granularity
    if granularity == "Full (11×12)":
        # Budget bins (11 bins)
        budget_bins = ['0%', '1-10%', '11-20%', '21-30%', '31-40%', '41-50%',
                       '51-60%', '61-70%', '71-80%', '81-90%', '91-100%']
        # Time bins (12 bins)
        time_bins = ['0%', '1-10%', '11-20%', '21-30%', '31-40%', '41-50%',
                     '51-60%', '61-70%', '71-80%', '81-90%', '91-100%', 'Over 100%']
    else:  # Simplified
        budget_bins = ['0%', '1-25%', '26-50%', '51-75%', '76-100%', 'Over 100%']
        time_bins = ['0%', '1-25%', '26-50%', '51-75%', '76-100%', 'Over 100%']

    # Categorize projects
    def categorize_budget_simplified(value):
        if pd.isna(value):
            return 'Unknown'
        if value == 0:
            return '0%'
        elif value <= 25:
            return '1-25%'
        elif value <= 50:
            return '26-50%'
        elif value <= 75:
            return '51-75%'
        elif value <= 100:
            return '76-100%'
        else:
            return 'Over 100%'

    def categorize_time_simplified(value):
        if pd.isna(value):
            return 'Unknown'
        if value == 0:
            return '0%'
        elif value <= 25:
            return '1-25%'
        elif value <= 50:
            return '26-50%'
        elif value <= 75:
            return '51-75%'
        elif value <= 100:
            return '76-100%'
        else:
            return 'Over 100%'

    # Apply categorization
    if granularity == "Full (11×12)":
        df_matrix = create_performance_bins(filtered_df, 'percent_budget_used', 'budget')
        df_matrix = df_matrix.rename(columns={'bin_category': 'budget_bin'})
        df_matrix = create_performance_bins(df_matrix, 'percent_time_used', 'time')
        df_matrix = df_matrix.rename(columns={'bin_category': 'time_bin'})
    else:
        df_matrix = filtered_df.copy()
        df_matrix['budget_bin'] = df_matrix['percent_budget_used'].apply(categorize_budget_simplified)
        df_matrix['time_bin'] = df_matrix['percent_time_used'].apply(categorize_time_simplified)

    # Remove Unknown bins
    df_matrix = df_matrix[(df_matrix['budget_bin'] != 'Unknown') & (df_matrix['time_bin'] != 'Unknown')]

    # Create pivot table (without totals)
    if metric_mode == "Count of Projects":
        # Count projects in each cell
        matrix = pd.crosstab(
            df_matrix['time_bin'],
            df_matrix['budget_bin']
        )
    else:  # Sum of Project Budget
        if not bac_col:
            st.warning("⚠️ Budget column (BAC) not found in data.")
            return

        matrix = pd.pivot_table(
            df_matrix,
            values=bac_col,
            index='time_bin',
            columns='budget_bin',
            aggfunc='sum',
            fill_value=0
        )

    # Reorder rows and columns to match bin order
    matrix = matrix.reindex(index=[b for b in time_bins if b in matrix.index],
                            columns=[b for b in budget_bins if b in matrix.columns],
                            fill_value=0)

    # Calculate percentages if needed
    if value_display == "% of Portfolio":
        total_value = matrix.values.sum()
        display_matrix = (matrix / total_value * 100) if total_value > 0 else matrix
    elif value_display == "% of Row":
        row_sums = matrix.sum(axis=1)
        display_matrix = matrix.div(row_sums, axis=0) * 100
    elif value_display == "% of Column":
        col_sums = matrix.sum(axis=0)
        display_matrix = matrix.div(col_sums, axis=1) * 100
    else:  # Absolute Values
        display_matrix = matrix

    # Create heatmap
    heatmap_data = display_matrix.copy()

    # Calculate min/max for text color determination
    data_min = heatmap_data.values.min()
    data_max = heatmap_data.values.max()
    data_range = data_max - data_min if data_max > data_min else 1

    # Create annotations
    annotations = []
    for i, row_label in enumerate(display_matrix.index):
        for j, col_label in enumerate(display_matrix.columns):
            value = display_matrix.loc[row_label, col_label]

            # Skip annotation if value is 0 or NaN (hide zeros)
            if pd.isna(value) or value == 0 or abs(value) < 0.01:
                continue

            # Format text based on display mode and metric
            if value_display == "Absolute Values":
                if metric_mode == "Count of Projects":
                    text = f"{int(value)}"
                else:
                    text = f"{currency_symbol}{value/1e6:.1f}M" if value >= 1e6 else f"{currency_symbol}{value/1e3:.0f}K"
            else:  # Percentage
                text = f"{value:.1f}%"

            # Determine text color based on cell intensity
            # Normalize value to 0-1 range
            normalized_value = (value - data_min) / data_range if not pd.isna(value) else 0
            # Use black text for light cells (< 0.5), white text for dark cells (>= 0.5)
            text_color = 'black' if normalized_value < 0.5 else 'white'

            annotations.append(
                dict(
                    x=col_label,
                    y=row_label,
                    text=text,
                    font=dict(size=10, color=text_color),
                    showarrow=False
                )
            )

    # Create figure
    fig = go.Figure(data=go.Heatmap(
        z=heatmap_data.values,
        x=heatmap_data.columns.tolist(),
        y=heatmap_data.index.tolist(),
        colorscale='Blues',
        showscale=True,
        hovertemplate='Budget: %{x}<br>Time: %{y}<br>Value: %{z}<extra></extra>',
        xgap=1,  # Add gap between columns (1 pixel)
        ygap=1   # Add gap between rows (1 pixel)
    ))

    # Add annotations
    fig.update_layout(annotations=annotations)

    # Update layout
    title_text = f"Budget-Time Performance Matrix - {metric_mode}"
    if value_display != "Absolute Values":
        title_text += f" ({value_display})"

    fig.update_layout(
        title=title_text,
        xaxis_title="% Budget Used",
        yaxis_title="% Time Used",
        height=600,
        xaxis=dict(
            side='bottom',
            showgrid=False
        ),
        yaxis=dict(
            autorange='reversed',  # Time bins top to bottom
            showgrid=False
        ),
        plot_bgcolor='black'
    )

    st.plotly_chart(fig, width='stretch')

    # Show data table
    with st.expander("📊 View Matrix Data Table", expanded=False):
        # Format for display
        table_display = display_matrix.copy()

        if value_display == "Absolute Values" and metric_mode == "Sum of Project Budget":
            # Format currency
            for col in table_display.columns:
                table_display[col] = table_display[col].apply(lambda x: format_currency(x, symbol=currency_symbol))
        elif value_display != "Absolute Values":
            # Format percentages
            for col in table_display.columns:
                table_display[col] = table_display[col].apply(lambda x: f"{x:.1f}%" if not pd.isna(x) else "0.0%")
        else:
            # Count - just integers
            for col in table_display.columns:
                table_display[col] = table_display[col].apply(lambda x: f"{int(x)}" if not pd.isna(x) else "0")

        st.dataframe(table_display, width='stretch')

        # CSV Export
        csv_matrix = display_matrix.copy()
        if value_display != "Absolute Values":
            # Format percentages for CSV (no % symbol)
            for col in csv_matrix.columns:
                csv_matrix[col] = csv_matrix[col].apply(lambda x: f"{x:.2f}" if not pd.isna(x) else "0")

        csv_data = csv_matrix.to_csv()

        st.download_button(
            "📥 Download Matrix CSV",
            csv_data,
            file_name=f"budget_time_matrix_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
            mime="text/csv"
        )


def main():
    st.markdown('<h1 class="main-header">📈 Portfolio Gantt</h1>', unsafe_allow_html=True)
    st.markdown("Timeline visualization for your portfolio")

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
        # Initialize config_dict if it doesn't exist
        if 'config_dict' not in st.session_state:
            st.session_state.config_dict = {}
        st.session_state.config_dict['controls'] = controls

        # Also store llm_config separately if it exists
        if 'llm_config' in portfolio_settings:
            st.session_state.config_dict['llm_config'] = portfolio_settings['llm_config']

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

    df = load_portfolio_dataframe(portfolio_id, status_date)
    if df is None:
        st.warning("No portfolio data available. Run a batch analysis from the Portfolio Analysis page first.")
        render_footer()
        return

    # Detect available column names and coerce data types
    date_cols_to_check = ["plan_start", "plan_finish", "Plan Start", "Plan Finish", "data_date", "forecast_completion", "likely_completion"]
    for col in date_cols_to_check:
        if col in df.columns:
            df[col] = _coerce_datetime(df[col])

    numeric_cols_to_check = ["bac", "BAC", "ac", "AC", "earned_value", "original_duration_months", "actual_duration_months", "cost_performance_index", "schedule_performance_index"]
    for col in numeric_cols_to_check:
        if col in df.columns:
            df[col] = _coerce_numeric(df[col])

    # Find the correct start and finish date columns
    start_col = None
    if "plan_start" in df.columns:
        start_col = "plan_start"
    elif "Plan Start" in df.columns:
        start_col = "Plan Start"

    finish_col = None
    if "plan_finish" in df.columns:
        finish_col = "plan_finish"
    elif "Plan Finish" in df.columns:
        finish_col = "Plan Finish"

    if not start_col or not finish_col:
        st.warning("Project records are missing plan date columns. Please verify the source data.")
        render_footer()
        return

    # Filter out rows with missing or invalid dates
    df = df.dropna(subset=[start_col, finish_col])

    # Additional validation: remove rows with unreasonable dates (like epoch dates)
    if not df.empty:
        # Filter out dates before 1980 (likely invalid/epoch dates)
        valid_start_mask = df[start_col].apply(lambda x: pd.notna(x) and pd.Timestamp(x).year > 1980 if pd.notna(x) else False)
        valid_finish_mask = df[finish_col].apply(lambda x: pd.notna(x) and pd.Timestamp(x).year > 1980 if pd.notna(x) else False)
        df = df[valid_start_mask & valid_finish_mask]

    if df.empty:
        st.warning("Project records are missing valid plan dates. Please verify the source data.")
        render_footer()
        return

    filtered_df = apply_filters(df, start_col, finish_col)

    st.markdown(f"**Projects Displayed:** {len(filtered_df)}")

    with st.expander("📊 Portfolio Gantt Chart", expanded=False):
        # Period and View controls
        col1, col2 = st.columns([1, 1])
        with col1:
            period_choice = st.radio("Period", list(PERIOD_OPTIONS.keys()), index=2, horizontal=True, key="gantt_period_choice")  # Default to Year (index 2)
        with col2:
            show_predicted = st.toggle("Predicted View", value=False, key="gantt_predicted_view")  # Default to Plan view
            st.caption(f"Mode: {'Predicted' if show_predicted else 'Plan'}")

        render_gantt(filtered_df, show_predicted, period_choice, start_col)

    # Cash Flow Chart Expander
    with st.expander("💰 Cash Flow Chart", expanded=False):
        render_cash_flow_chart(filtered_df, start_col, finish_col)

    # Time/Budget Performance Expander
    with st.expander("📈 Time/Budget Performance", expanded=False):
        render_time_budget_performance(filtered_df)

    # Portfolio Performance Matrix Expander
    with st.expander("📈 Portfolio Performance Matrix", expanded=False):
        render_portfolio_performance_curve(filtered_df)

    # Portfolio Treemap Expander
    with st.expander("🗺️ Portfolio Treemap", expanded=False):
        render_portfolio_treemap(filtered_df)

    # Organization Budget Chart Expander
    with st.expander("📊 Organization Budget Chart", expanded=False):
        render_portfolio_budget_chart(filtered_df)

    # Approvals Chart Expander
    with st.expander("📈 Approvals Chart", expanded=False):
        render_approvals_chart(filtered_df)

    # Portfolio Heatmap Expander
    with st.expander("🔥 Portfolio Heatmap (Budget × Duration Tiers)", expanded=False):
        render_portfolio_heatmap(filtered_df)

    # Histogram Expander
    with st.expander("📊 Histogram (Distribution Analysis)", expanded=False):
        render_histogram(filtered_df)

    # Budget Performance Expander
    with st.expander("💰 Budget Performance (Binned Analysis)", expanded=False):
        render_budget_performance(filtered_df)

    # Time Performance Expander
    with st.expander("⏱️ Time Performance (Binned Analysis)", expanded=False):
        render_time_performance(filtered_df)

    # Budget-Time Performance Matrix Expander
    with st.expander("📊 Budget-Time Performance Matrix", expanded=False):
        render_budget_time_matrix(filtered_df)

    render_footer()


if __name__ == "__main__":
    main()


# Show user info in sidebar
from utils.auth import show_user_info_sidebar
show_user_info_sidebar()

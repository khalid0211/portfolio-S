# Project Analysis — Single Project EVM Analysis and Management
# Requirements:
#   pip install streamlit pandas matplotlib python-dateutil requests
#
# Notes:
# - Data now stored in session state (no database files needed)
# - S-curve uses Beta(2,2) closed-form CDF (no SciPy needed)
# - Charts use matplotlib; solid to Data Date, dotted to BAC/EAC at Forecast Completion
# - Enhanced UX with reorganized sidebar and professional styling

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
import requests
try:
    import matplotlib.pyplot as plt
    import matplotlib.patches as patches
    MATPLOTLIB_AVAILABLE = True
except ImportError:
    MATPLOTLIB_AVAILABLE = False
    # Create stub functions to prevent errors
    plt = None
    patches = None
import streamlit as st
from dateutil import parser as date_parser
from utils.auth import check_authentication, require_page_access
from config.constants import USE_DATABASE
from services.data_service import data_manager
from utils.portfolio_context import render_portfolio_context
from services.db_data_service import DatabaseDataManager

# Import utility functions from core.utils (centralized utilities)
from core.utils import *  # noqa: F401, F403

# Import EVM functions from centralized engine
from core.evm_engine import (
    perform_complete_evm_analysis,
    perform_batch_calculation,
    calculate_evm_metrics,
    calculate_pv_linear,
    calculate_pv_scurve,
)

# =============================================================================
# CONSTANTS
# =============================================================================

# Application constants
APP_TITLE = "Project Analysis - EVM Intelligence Suite 📊"
DEFAULT_DATASET_TABLE = "dataset"
RESULTS_TABLE = "evm_results"
CONFIG_TABLE = "app_config"

# Calculation constants
DAYS_PER_MONTH = 30.44
INTEGRATION_STEPS = 200
MAX_TIMEOUT_SECONDS = 120
MIN_TIMEOUT_SECONDS = 10
EXCEL_ORDINAL_BASE = datetime(1899, 12, 30)

# Config file paths for local storage
CONFIG_DIR = Path.home() / ".portfolio_suite"
MODEL_CONFIG_FILE = CONFIG_DIR / "model_config.json"

# Validation patterns
VALID_TABLE_NAME_PATTERN = re.compile(r'^[a-zA-Z0-9_-]+$')
VALID_COLUMN_NAME_PATTERN = re.compile(r'^[a-zA-Z0-9_\s-]+$')

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Streamlit page config with professional styling
st.set_page_config(
    page_title=APP_TITLE,
    layout="wide",
    page_icon="📊",
    initial_sidebar_state="expanded"
)

# Check authentication and page access
if not check_authentication():
    st.stop()

require_page_access('project_analysis', 'Project Analysis')

# Professional CSS styling
st.markdown("""
<style>
    .main > div {
        padding-top: 2rem;
    }
    .stSelectbox > div > div {
        background-color: #e9ecef;
        border: 1px solid #ced4da;
    }
    .metric-card {
        background-color: #f8f9fa;
        padding: 1rem;
        border-radius: 0.5rem;
        border-left: 4px solid #007bff;
        margin-bottom: 1rem;
    }
    .success-metric {
        border-left-color: #28a745;
    }
    .warning-metric {
        border-left-color: #ffc107;
    }
    .danger-metric {
        border-left-color: #dc3545;
    }
    .sidebar-section {
        background-color: #f8f9fa;
        padding: 0.8rem;
        border-radius: 0.5rem;
        margin-bottom: 1rem;
        border: 1px solid #dee2e6;
    }
    .section-header {
        font-weight: 600;
        font-size: 1.1rem;
        color: #495057;
        margin-bottom: 0.5rem;
        padding-bottom: 0.2rem;
        border-bottom: 2px solid #007bff;
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
    /* Input field styling for better visibility */
    .stNumberInput > div > div > input {
        background-color: #e3f2fd !important;
        color: #1a1a1a !important;
        border: 1px solid #90caf9 !important;
        font-weight: 500 !important;
    }
    .stTextInput > div > div > input {
        background-color: #e3f2fd !important;
        color: #1a1a1a !important;
        border: 1px solid #90caf9 !important;
        font-weight: 500 !important;
    }
    .stDateInput > div > div > input {
        background-color: #e3f2fd !important;
        color: #1a1a1a !important;
        border: 1px solid #90caf9 !important;
        font-weight: 500 !important;
    }
    /* Labels for input fields - make them black for clarity */
    .stNumberInput > label,
    .stTextInput > label,
    .stDateInput > label,
    .stSelectbox > label {
        color: #1a1a1a !important;
        font-weight: 500 !important;
    }
</style>
""", unsafe_allow_html=True)

# =============================================================================
# SECURITY AND VALIDATION UTILITIES
# =============================================================================

def validate_table_name(table_name: str) -> bool:
    """Validate table name to prevent SQL injection."""
    if not table_name or len(table_name) > 50:
        return False
    return bool(VALID_TABLE_NAME_PATTERN.match(table_name))

def validate_column_name(column_name: str) -> bool:
    """Validate column name to prevent SQL injection."""
    if not column_name or len(column_name) > 100:
        return False
    return bool(VALID_COLUMN_NAME_PATTERN.match(column_name))

def sanitize_sql_identifier(identifier: str) -> str:
    """Sanitize SQL identifier by quoting it properly."""
    if not identifier:
        raise ValueError("Empty identifier")
    cleaned = identifier.replace('"', '').replace('[', '').replace(']', '')
    return f'"{cleaned}"'

# validate_numeric_input() now imported from core.utils

# =============================================================================
# LOCAL STORAGE UTILITIES
# =============================================================================

def ensure_config_dir():
    """Ensure config directory exists."""
    CONFIG_DIR.mkdir(exist_ok=True)


def save_model_config(provider: str, model: str):
    """Save model configuration."""
    try:
        ensure_config_dir()
        config = {"provider": provider, "model": model}
        with open(MODEL_CONFIG_FILE, 'w') as f:
            json.dump(config, f)
    except Exception as e:
        logger.error(f"Failed to save model config: {e}")

def load_model_config() -> Dict[str, str]:
    """Load model configuration."""
    try:
        if MODEL_CONFIG_FILE.exists():
            with open(MODEL_CONFIG_FILE, 'r') as f:
                return json.load(f)
    except Exception as e:
        logger.error(f"Failed to load model config: {e}")
    return {"provider": "OpenAI", "model": "gpt-4o-mini"}

# =============================================================================
# ENHANCED UTILITIES
# =============================================================================

# safe_divide() now imported from core.utils

# safe_calculate_forecast_duration() now imported from core.utils

def safe_financial_metrics(ev: float, ac: float, pv: float = None) -> dict:
    """Safely calculate CPI, SPI, and related metrics, handling AC=0 cases.

    Args:
        ev: Earned Value
        ac: Actual Cost
        pv: Planned Value (optional, for SPI calculation)

    Returns:
        Dictionary with safely calculated metrics
    """
    metrics = {}

    # CPI calculation - return NaN for AC=0 to display as 'N/A'
    if ac == 0:
        metrics['cpi'] = float('nan')
    else:
        metrics['cpi'] = safe_divide(ev, ac, 0.0)

    # SPI calculation - only if PV is provided
    if pv is not None:
        metrics['spi'] = safe_divide(ev, pv, 0.0)

    return metrics

def format_financial_metric(value: float, decimals: int = 3, as_percentage: bool = False) -> str:
    """Format financial metrics, displaying NaN as 'N/A' and handling AC=0 cases.

    Args:
        value: The numeric value to format
        decimals: Number of decimal places
        as_percentage: Whether to format as percentage

    Returns:
        Formatted string or 'N/A' for invalid values
    """
    try:
        if pd.isna(value) or math.isnan(value):
            return "N/A"
        if math.isinf(value):
            return "∞"
        if as_percentage:
            return f"{value * 100:.{decimals}f}%"
        return f"{value:.{decimals}f}"
    except (ValueError, TypeError):
        return "N/A"

# is_valid_finite_number() now imported from core.utils

# parse_date_any() now imported from core.utils

def format_currency(amount: float, symbol: str, postfix: str = "", decimals: int = 2) -> str:
    """Enhanced currency formatting with comma separators and postfix options.
    
    The postfix indicates the unit the user's figures are already in.
    For example, if user enters 1200 with postfix 'Million', we display '1,200 M'
    because the user's figure is already in millions.
    
    Optimized for displaying up to 9-digit numbers with proper formatting.
    """
    if not is_valid_finite_number(amount):
        return "—"
    
    try:
        # Handle postfix labels (no scaling - user's figures are already in specified units)
        if postfix.lower() == "thousand":
            postfix_label = "K"
        elif postfix.lower() == "million":
            postfix_label = "M"
        elif postfix.lower() == "billion":
            postfix_label = "B"
        else:
            postfix_label = ""
        
        # Format with commas for better readability of large numbers (up to 9 digits)
        formatted_amount = f"{amount:,.{decimals}f}"
        
        # Ensure we don't exceed screen width - compact display for very long numbers
        if len(formatted_amount) > 12 and decimals > 0:
            # Try with 0 decimals if too long
            formatted_amount = f"{amount:,.0f}"
            
        result = f"{symbol} {formatted_amount}"
        if postfix_label:
            result += f" {postfix_label}"
            
        return result
    except (ValueError, OverflowError):
        return "—"

def format_percentage(value: float) -> str:
    """Format percentage values consistently with 2 decimal places."""
    if not is_valid_finite_number(value):
        return "—"
    return f"{value:.2f}%"

def format_performance_index(value: float) -> str:
    """Format performance indices (CPI, SPI, SPIe) consistently with 2 decimal places."""
    if not is_valid_finite_number(value):
        return "N/A"
    return f"{value:.2f}"

def format_duration(value: float, unit: str = "months") -> str:
    """Format duration values as rounded integers."""
    if not is_valid_finite_number(value):
        return "—"
    return f"{int(round(value))} {unit}"

def format_date_dmy(date_str: str) -> str:
    """Format date string to dd-mm-yyyy format."""
    try:
        if date_str == 'N/A' or not date_str:
            return "N/A"
        # Parse the date and format as dd-mm-yyyy
        parsed_date = parse_date_any(date_str)
        return parsed_date.strftime('%d-%m-%Y')
    except:
        return date_str  # Return original if parsing fails

def maybe(val, default="—"):
    """Return default if value is None or invalid."""
    if val is None:
        return default
    if isinstance(val, (int, float)) and not is_valid_finite_number(val):
        return default
    return val

def create_gauge_chart(value: float, title: str, min_val: float = 0.4, max_val: float = 1.1) -> None:
    """Create a simple gauge chart for performance indices with scale 0.4 to 1.1."""
    try:
        if not MATPLOTLIB_AVAILABLE:
            st.error("📊 Charts require matplotlib. Please install: pip install matplotlib")
            return
        
        fig, ax = plt.subplots(figsize=(4, 3))
        
        # Create semicircular gauge
        theta1, theta2 = 0, 180  # Semicircle from 0 to 180 degrees
        
        # Background arc
        arc = patches.Arc((0.5, 0), 1, 1, angle=0, theta1=theta1, theta2=theta2, linewidth=20, color='lightgray')
        ax.add_patch(arc)
        
        # Color zones based on new scale (0.4 to 1.1)
        range_val = max_val - min_val
        
        # Red zone (0.4-0.9)
        red_start = 0
        red_end = ((0.9 - min_val) / range_val) * 180
        red_arc = patches.Arc((0.5, 0), 1, 1, angle=0, theta1=red_start, theta2=red_end, linewidth=20, color='#ff4444')
        ax.add_patch(red_arc)
        
        # Yellow zone (0.9-1.0)
        yellow_start = red_end
        yellow_end = ((1.0 - min_val) / range_val) * 180
        yellow_arc = patches.Arc((0.5, 0), 1, 1, angle=0, theta1=yellow_start, theta2=yellow_end, linewidth=20, color='#ffaa00')
        ax.add_patch(yellow_arc)
        
        # Green zone (1.0-1.1)
        green_start = yellow_end
        green_end = 180
        green_arc = patches.Arc((0.5, 0), 1, 1, angle=0, theta1=green_start, theta2=green_end, linewidth=20, color='#44aa44')
        ax.add_patch(green_arc)
        
        # Value indicator
        if is_valid_finite_number(value):
            # Clamp value to our scale range
            value_clamped = max(min_val, min(value, max_val))
            # Calculate angle based on position within our range
            value_position = (value_clamped - min_val) / range_val
            value_angle = value_position * 180
            value_rad = np.radians(value_angle)
            
            # Pointer
            ax.arrow(0.5, 0, 0.4 * np.cos(value_rad), 0.4 * np.sin(value_rad),
                    head_width=0.03, head_length=0.02, fc='black', ec='black', linewidth=2)
        
        # Labels
        ax.text(0.5, -0.2, title, ha='center', va='center', fontsize=12, fontweight='bold')
        ax.text(0.5, -0.35, f'{format_performance_index(value)}' if is_valid_finite_number(value) else 'N/A', 
               ha='center', va='center', fontsize=14, fontweight='bold', color='blue')
        
        # Scale labels based on new range
        ax.text(0, -0.05, f'{min_val}', ha='center', va='center', fontsize=10)
        ax.text(0.25, 0.25, f'{(min_val + range_val*0.25):.1f}', ha='center', va='center', fontsize=8)
        ax.text(0.5, 0.45, '1.0', ha='center', va='center', fontsize=10, fontweight='bold')
        ax.text(0.75, 0.25, f'{(min_val + range_val*0.75):.1f}', ha='center', va='center', fontsize=8)
        ax.text(1, -0.05, f'{max_val}', ha='center', va='center', fontsize=10)
        
        ax.set_xlim(0, 1)
        ax.set_ylim(-0.4, 0.6)
        ax.set_aspect('equal')
        ax.axis('off')
        
        return fig
        
    except Exception as e:
        logger.error(f"Gauge chart creation failed: {e}")
        return None

# =============================================================================
# PORTFOLIO ANALYTICS FUNCTIONS
# =============================================================================

def calculate_portfolio_summary(batch_results_df: pd.DataFrame) -> Dict[str, Any]:
    """Calculate comprehensive portfolio analytics from batch results."""
    if batch_results_df.empty:
        return {}
    
    # Filter out error rows - safe boolean indexing
    if 'error' in batch_results_df.columns:
        valid_results = batch_results_df[batch_results_df['error'].isna()].copy()
    else:
        valid_results = batch_results_df.copy()
    
    if valid_results.empty:
        return {"error": "No valid project results found"}
    
    try:
        # Portfolio totals
        total_bac = valid_results['bac'].sum() if 'bac' in valid_results else 0
        total_ac = valid_results['ac'].sum() if 'ac' in valid_results else 0
        total_pv = valid_results['pv'].sum() if 'pv' in valid_results else 0
        total_ev = valid_results['ev'].sum() if 'ev' in valid_results else 0

        # Present value calculations - sum the new financial metrics directly
        # Removed Present Value Progress - duplicate of Present Value
        total_planned_value_project = valid_results['planned_value_project'].sum() if 'planned_value_project' in valid_results else 0
        total_likely_value_project = valid_results['likely_value_project'].sum() if 'likely_value_project' in valid_results else 0

        # Calculate portfolio percentages
        total_percent_present_value_project = safe_divide(total_planned_value_project, total_bac) * 100 if total_bac > 0 else 0.0
        total_percent_likely_value_project = safe_divide(total_likely_value_project, total_bac) * 100 if total_bac > 0 else 0.0

        # Calculate Total Planned Present Value (PMT=BAC/OD)
        total_planned_present_value = 0
        # Calculate Total Likely Present Value (PMT=BAC/Likely Duration)
        total_likely_present_value = 0

        # Get the inflation rate from the first row (assuming consistent across portfolio)
        inflation_rate = valid_results.iloc[0].get('inflation_rate', 5.0) / 100.0 if len(valid_results) > 0 else 0.05
        monthly_rate = (1 + inflation_rate) ** (1/12) - 1 if inflation_rate > 0 else 0.0  # Correct compound rate conversion

        # Calculate additional present values for each project
        for _, row in valid_results.iterrows():
            project_bac = row.get('bac', 0)
            project_od = row.get('original_duration_months', 0)
            project_forecast_duration = row.get('forecast_duration', 0)

            # Total Planned Present Value (PMT = BAC/OD)
            if (is_valid_finite_number(project_bac) and is_valid_finite_number(project_od) and
                project_bac > 0 and project_od > 0):
                pmt_planned = project_bac / project_od
                if monthly_rate > 0:
                    try:
                        factor_planned = (1 - (1 + monthly_rate) ** (-project_od)) / monthly_rate
                        planned_pv = pmt_planned * factor_planned
                    except (OverflowError, ValueError):
                        planned_pv = project_bac
                else:
                    planned_pv = project_bac
                total_planned_present_value += planned_pv

            # Total Likely Present Value (PMT = BAC/Likely Duration)
            if (is_valid_finite_number(project_bac) and is_valid_finite_number(project_forecast_duration) and
                project_bac > 0 and project_forecast_duration > 0):
                pmt_likely = project_bac / project_forecast_duration
                if monthly_rate > 0:
                    try:
                        factor_likely = (1 - (1 + monthly_rate) ** (-project_forecast_duration)) / monthly_rate
                        likely_pv = pmt_likely * factor_likely
                    except (OverflowError, ValueError):
                        likely_pv = project_bac
                else:
                    likely_pv = project_bac
                total_likely_present_value += likely_pv
        
        # Portfolio performance indices - use portfolio-level sums to avoid unrealistic individual values
        # CPI = SUM(EV)/SUM(AC), SPI = SUM(EV)/SUM(PV)
        portfolio_cpi = safe_divide(total_ev, total_ac, 0.0)
        portfolio_spi = safe_divide(total_ev, total_pv, 0.0)

        # Portfolio TCPI = (SUM(BAC) - SUM(EV)) / (SUM(BAC) - SUM(AC))
        work_remaining_portfolio = total_bac - total_ev
        budget_remaining_portfolio = total_bac - total_ac
        portfolio_tcpi = safe_divide(work_remaining_portfolio, budget_remaining_portfolio, 0.0)
        
        # Calculate weighted averages (CPI*BAC / Sum BAC, SPI*BAC / Sum BAC)
        total_cpi_weighted = 0
        total_spi_weighted = 0

        # Calculate % Time Used using correct formula: Sum(AD * BAC) / Sum(OD * BAC)
        total_ad_bac = 0  # Sum of Actual Duration * BAC
        total_od_bac = 0  # Sum of Original Duration * BAC

        for _, row in valid_results.iterrows():
            project_bac = row.get('bac', 0)
            project_cpi = row.get('cpi', 0)
            project_spi = row.get('spi', 0)
            project_ad = row.get('actual_duration_months', 0)
            project_od = row.get('original_duration_months', 0)

            if is_valid_finite_number(project_cpi) and is_valid_finite_number(project_bac) and project_bac > 0:
                total_cpi_weighted += project_cpi * project_bac
            if is_valid_finite_number(project_spi) and is_valid_finite_number(project_bac) and project_bac > 0:
                total_spi_weighted += project_spi * project_bac

            # Calculate time used components
            if (is_valid_finite_number(project_ad) and is_valid_finite_number(project_od) and
                is_valid_finite_number(project_bac) and project_bac > 0 and project_od > 0):
                total_ad_bac += project_ad * project_bac
                total_od_bac += project_od * project_bac

        weighted_avg_cpi = safe_divide(total_cpi_weighted, total_bac, 0.0)
        weighted_avg_spi = safe_divide(total_spi_weighted, total_bac, 0.0)
        weighted_avg_time_used = safe_divide(total_ad_bac, total_od_bac, 0.0) * 100  # Convert to percentage

        # Note: Portfolio CPI/SPI already calculated above using portfolio-level sums
        # This avoids issues with unrealistically high individual CPI/SPI values
        
        # Performance quadrant analysis
        quadrants = {
            'on_budget_on_schedule': 0,
            'on_budget_behind_schedule': 0,
            'over_budget_on_schedule': 0,
            'over_budget_behind_schedule': 0
        }
        
        quadrant_budgets = {
            'on_budget_on_schedule': 0,
            'on_budget_behind_schedule': 0,
            'over_budget_on_schedule': 0,
            'over_budget_behind_schedule': 0
        }
        
        for _, row in valid_results.iterrows():
            cpi = row.get('cpi', 0)
            spi = row.get('spi', 0)
            project_bac = row.get('bac', 0)
            
            if cpi >= 1.0 and spi >= 1.0:
                quadrants['on_budget_on_schedule'] += 1
                quadrant_budgets['on_budget_on_schedule'] += project_bac
            elif cpi >= 1.0 and spi < 1.0:
                quadrants['on_budget_behind_schedule'] += 1
                quadrant_budgets['on_budget_behind_schedule'] += project_bac
            elif cpi < 1.0 and spi >= 1.0:
                quadrants['over_budget_on_schedule'] += 1
                quadrant_budgets['over_budget_on_schedule'] += project_bac
            else:
                quadrants['over_budget_behind_schedule'] += 1
                quadrant_budgets['over_budget_behind_schedule'] += project_bac
        
        total_projects = len(valid_results)
        
        # Calculate percentages
        quadrant_percentages = {
            key: safe_divide(count, total_projects) * 100 
            for key, count in quadrants.items()
        }
        
        budget_percentages = {
            key: safe_divide(budget, total_bac) * 100 
            for key, budget in quadrant_budgets.items()
        }
        
        return {
            'total_projects': total_projects,
            'total_bac': total_bac,
            'total_ac': total_ac,
            'total_pv': total_pv,
            'total_ev': total_ev,
            # 'total_present_value_progress': removed duplicate
            'total_planned_value_project': total_planned_value_project,
            'total_likely_value_project': total_likely_value_project,
            'total_percent_present_value_project': total_percent_present_value_project,
            'total_percent_likely_value_project': total_percent_likely_value_project,
            'total_planned_present_value': total_planned_present_value,
            'total_likely_present_value': total_likely_present_value,
            'portfolio_cpi': portfolio_cpi,
            'portfolio_spi': portfolio_spi,
            'portfolio_tcpi': portfolio_tcpi,
            'weighted_avg_cpi': weighted_avg_cpi,
            'weighted_avg_spi': weighted_avg_spi,
            'weighted_avg_time_used': weighted_avg_time_used,
            'quadrants': quadrants,
            'quadrant_percentages': quadrant_percentages,
            'quadrant_budgets': quadrant_budgets,
            'budget_percentages': budget_percentages,
            'average_progress': safe_divide(total_ev, total_bac) * 100
        }
    
    except Exception as e:
        logger.error(f"Portfolio summary calculation failed: {e}")
        return {"error": str(e)}

def is_project_table(table_name: str) -> bool:
    """Check if a table contains project data by examining required columns."""
    try:
        df = load_table(table_name)
        required_columns = ['Project ID', 'BAC', 'AC', 'Plan Start', 'Plan Finish']
        
        # Check if table has project-like columns (case insensitive)
        df_columns_lower = [col.lower() for col in df.columns]
        
        project_indicators = 0
        for req_col in required_columns:
            for col in df_columns_lower:
                if req_col.lower().replace(' ', '').replace('_', '') in col.replace(' ', '').replace('_', ''):
                    project_indicators += 1
                    break
        
        # If it has at least 3 out of 5 required indicators, consider it a project table
        return project_indicators >= 3
        
    except Exception:
        return False

def save_column_mapping(table_name: str, mapping: Dict[str, str]):
    """Save column mapping for a specific table persistently."""
    try:
        ensure_config_dir()
        mapping_file = CONFIG_DIR / f"mapping_{table_name}.json"
        with open(mapping_file, 'w') as f:
            json.dump(mapping, f)
    except Exception as e:
        logger.error(f"Failed to save column mapping for {table_name}: {e}")

def load_column_mapping(table_name: str) -> Dict[str, str]:
    """Load column mapping for a specific table."""
    try:
        mapping_file = CONFIG_DIR / f"mapping_{table_name}.json"
        if mapping_file.exists():
            with open(mapping_file, 'r') as f:
                return json.load(f)
    except Exception as e:
        logger.error(f"Failed to load column mapping for {table_name}: {e}")
    return {}

# =============================================================================
# SESSION STATE DATA FUNCTIONS (formerly database functions)
# =============================================================================

def ensure_db():
    """Ensure session state is initialized (replaces database initialization)."""
    try:
        # Initialize session state data structures if not already done
        if "data_df" not in st.session_state:
            st.session_state.data_df = None
        if "config_dict" not in st.session_state:
            st.session_state.config_dict = {}
        if "data_loaded" not in st.session_state:
            st.session_state.data_loaded = False
        if "original_filename" not in st.session_state:
            st.session_state.original_filename = None
        if "file_type" not in st.session_state:
            st.session_state.file_type = None
        if "raw_csv_df" not in st.session_state:
            st.session_state.raw_csv_df = None
        if "csv_filename" not in st.session_state:
            st.session_state.csv_filename = None
    except Exception as e:
        logger.error(f"Failed to initialize session state: {e}")
        raise

@st.cache_data(show_spinner=False)
def list_session_tables() -> List[str]:
    """List available tables from session state."""
    try:
        tables = []
        
        # Add main data table if exists and has data
        if (st.session_state.data_df is not None and 
            not st.session_state.data_df.empty):
            tables.append(DEFAULT_DATASET_TABLE)
        
        # Add tables from config dict
        if ("tables" in st.session_state.config_dict and 
            st.session_state.config_dict["tables"]):
            config_tables = [name for name in st.session_state.config_dict["tables"].keys() 
                           if validate_table_name(name)]
            tables.extend(config_tables)
        
        # Remove duplicates and sort
        return sorted(list(set(tables)))
        
    except Exception as e:
        logger.error(f"Failed to list session tables: {e}")
        return []

@st.cache_data(show_spinner=False)
def load_table(table_name: str) -> pd.DataFrame:
    """Load table from session state (replaces SQLite loading)."""
    if not validate_table_name(table_name):
        raise ValueError(f"Invalid table name: {table_name}")
    
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
        logger.warning(f"Table {table_name} not found in session state")
        return pd.DataFrame()
        
    except Exception as e:
        logger.error(f"Failed to load session table {table_name}: {e}")
        raise

def save_table_replace(df: pd.DataFrame, table_name: str):
    """Save table to session state (replaces SQLite saving)."""
    if not validate_table_name(table_name):
        raise ValueError(f"Invalid table name: {table_name}")
    
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
        logger.error(f"Failed to save session table {table_name}: {e}")
        raise

def delete_table(table_name: str):
    """Delete a table from session state (replaces SQLite table deletion)."""
    if not validate_table_name(table_name):
        raise ValueError(f"Invalid table name: {table_name}")
    
    try:
        table_deleted = False
        
        # Handle main dataset table
        if table_name == DEFAULT_DATASET_TABLE:
            if st.session_state.data_df is not None:
                st.session_state.data_df = None
                st.session_state.data_loaded = False
                table_deleted = True
        else:
            # Handle tables in config_dict
            if ("tables" in st.session_state.config_dict and 
                table_name in st.session_state.config_dict["tables"]):
                del st.session_state.config_dict["tables"][table_name]
                table_deleted = True
        
        if not table_deleted:
            raise ValueError(f"Table '{table_name}' not found")
    except Exception as e:
        logger.error(f"Failed to delete session table {table_name}: {e}")
        raise

def insert_project_record(project_data: Dict[str, Any], table_name: str = DEFAULT_DATASET_TABLE):
    """Insert a new project record into session state."""
    if not validate_table_name(table_name):
        raise ValueError(f"Invalid table name: {table_name}")
    
    try:
        # Initialize session data if needed
        if st.session_state.data_df is None:
            st.session_state.data_df = pd.DataFrame({
                "Project ID": [], "Project": [], "Organization": [], "Project Manager": [],
                "BAC": [], "AC": [], "Plan Start": [], "Plan Finish": [],
                "Use_Manual_PV": [], "Manual_PV": [], "Use_Manual_EV": [], "Manual_EV": []
            })
        
        # Get the target DataFrame
        if table_name == DEFAULT_DATASET_TABLE:
            existing_df = st.session_state.data_df
            # Ensure all expected columns exist
            expected_cols = ["Project ID", "Project", "Organization", "Project Manager",
                           "BAC", "AC", "Plan Start", "Plan Finish", "Use_Manual_PV", "Manual_PV", "Use_Manual_EV", "Manual_EV"]
            for col in expected_cols:
                if col not in existing_df.columns:
                    existing_df[col] = None
        else:
            # Handle other tables from config_dict
            if "tables" not in st.session_state.config_dict:
                st.session_state.config_dict["tables"] = {}
            
            if table_name in st.session_state.config_dict["tables"]:
                existing_df = pd.DataFrame(st.session_state.config_dict["tables"][table_name])
            else:
                existing_df = pd.DataFrame({
                    "Project ID": [], "Project": [], "Organization": [], "Project Manager": [],
                    "BAC": [], "AC": [], "Plan Start": [], "Plan Finish": [],
                    "Use_Manual_PV": [], "Manual_PV": [], "Use_Manual_EV": [], "Manual_EV": []
                })
        
        # Check for duplicate Project ID
        if not existing_df.empty:
            if str(project_data['Project ID']) in existing_df['Project ID'].astype(str).values:
                raise ValueError("Project ID already exists")
        
        # Create new record DataFrame
        new_record_df = pd.DataFrame([project_data])
        
        # Append new record using concat
        updated_df = pd.concat([existing_df, new_record_df], ignore_index=True)
        
        # Save back to session state
        if table_name == DEFAULT_DATASET_TABLE:
            st.session_state.data_df = updated_df
            st.session_state.data_loaded = True
        else:
            st.session_state.config_dict["tables"][table_name] = updated_df.to_dict('records')
    except Exception as e:
        logger.error(f"Failed to insert project into session: {e}")
        raise

def update_project_record(project_id: str, project_data: Dict[str, Any], table_name: str = DEFAULT_DATASET_TABLE):
    """Update an existing project record in session state."""
    if not validate_table_name(table_name):
        raise ValueError(f"Invalid table name: {table_name}")
    
    try:
        # Load DataFrame from session state
        if table_name == DEFAULT_DATASET_TABLE:
            if st.session_state.data_df is None or st.session_state.data_df.empty:
                raise ValueError("No data available to update")
            df = st.session_state.data_df.copy()
            # Ensure all expected columns exist
            expected_cols = ["Project ID", "Project", "Organization", "Project Manager",
                           "BAC", "AC", "Plan Start", "Plan Finish", "Use_Manual_PV", "Manual_PV", "Use_Manual_EV", "Manual_EV"]
            for col in expected_cols:
                if col not in df.columns:
                    df[col] = None
        else:
            # Handle other tables from config
            if ("tables" not in st.session_state.config_dict or 
                table_name not in st.session_state.config_dict["tables"]):
                raise ValueError(f"Table {table_name} not found")
            
            table_records = st.session_state.config_dict["tables"][table_name]
            df = pd.DataFrame(table_records)
        
        # Create mask for WHERE clause equivalent
        mask = df['Project ID'].astype(str) == str(project_id)
        
        # Check if project exists
        if not mask.any():
            raise ValueError(f"Project ID '{project_id}' not found")
        
        # Update records using .loc
        for key, value in project_data.items():
            if key in df.columns:
                df.loc[mask, key] = value
            else:
                logger.warning(f"Column '{key}' not found in table, skipping update")
        
        # Save updated DataFrame back to session state
        if table_name == DEFAULT_DATASET_TABLE:
            st.session_state.data_df = df
        else:
            st.session_state.config_dict["tables"][table_name] = df.to_dict('records')
    except Exception as e:
        logger.error(f"Failed to update project in session: {e}")
        raise

def delete_project_record(project_id: str, table_name: str = DEFAULT_DATASET_TABLE):
    """Delete a project record from session state."""
    if not validate_table_name(table_name):
        raise ValueError(f"Invalid table name: {table_name}")
    
    try:
        # Load DataFrame from session state
        if table_name == DEFAULT_DATASET_TABLE:
            if st.session_state.data_df is None or st.session_state.data_df.empty:
                raise ValueError("No data available")
            df = st.session_state.data_df.copy()
            # Ensure Project ID column exists for deletion
            if "Project ID" not in df.columns:
                raise ValueError("Project ID column not found in data")
        else:
            # Handle other tables from config
            if ("tables" not in st.session_state.config_dict or 
                table_name not in st.session_state.config_dict["tables"]):
                raise ValueError(f"Table {table_name} not found")
            
            table_records = st.session_state.config_dict["tables"][table_name]
            df = pd.DataFrame(table_records)
        
        # Find rows to delete
        mask = df['Project ID'].astype(str) == str(project_id)
        matching_indices = df.index[mask]
        
        if len(matching_indices) == 0:
            raise ValueError(f"Project ID '{project_id}' not found")
        
        # Perform deletion using .drop()
        df_after_delete = df.drop(matching_indices, axis=0).reset_index(drop=True)
        
        # Save updated DataFrame back to session state
        if table_name == DEFAULT_DATASET_TABLE:
            st.session_state.data_df = df_after_delete
        else:
            st.session_state.config_dict["tables"][table_name] = df_after_delete.to_dict('records')
    except Exception as e:
        logger.error(f"Failed to delete project from session: {e}")
        raise

def get_project_record(project_id: str, table_name: str = DEFAULT_DATASET_TABLE) -> Optional[Dict]:
    """Get a specific project record from session state."""
    try:
        # Load DataFrame from session state
        if table_name == DEFAULT_DATASET_TABLE:
            if st.session_state.data_df is None or st.session_state.data_df.empty:
                return None
            df = st.session_state.data_df
        else:
            # Handle other tables from config
            if ("tables" not in st.session_state.config_dict or 
                table_name not in st.session_state.config_dict["tables"]):
                return None
            
            table_records = st.session_state.config_dict["tables"][table_name]
            df = pd.DataFrame(table_records)
        
        if df.empty:
            return None
        
        # Filter using .loc for WHERE clause equivalent
        mask = df['Project ID'].astype(str) == str(project_id)
        matching_rows = df.loc[mask]
        
        # Check if any matches found
        if matching_rows.empty:
            return None
        
        # Return first match as dictionary
        return matching_rows.iloc[0].to_dict()
        
    except Exception as e:
        logger.error(f"Failed to get project {project_id} from session: {e}")
        return None

def create_demo_data():
    """Create empty demo data with proper schema at startup."""
    try:
        ensure_db()
        
        # Check if demo data already exists and has content
        try:
            existing_df = load_table(DEFAULT_DATASET_TABLE)
            if len(existing_df) > 0:
                return  # Don't overwrite existing data
        except:
            pass  # Table doesn't exist yet
        
        # Create empty demo data with proper schema
        demo_data = {
            "Project ID": [],
            "Project": [],
            "Organization": [],
            "Project Manager": [],
            "BAC": [],
            "AC": [],
            "Plan Start": [],
            "Plan Finish": [],
            "Use_Manual_PV": [],
            "Manual_PV": [],
            "Use_Manual_EV": [],
            "Manual_EV": []
        }
        
        demo_df = pd.DataFrame(demo_data)
        save_table_replace(demo_df, DEFAULT_DATASET_TABLE)
    except Exception as e:
        logger.error(f"Failed to create demo data: {e}")
        # Don't raise - application should continue even if demo creation fails

# =============================================================================
# EVM CALCULATION ENGINE (UNCHANGED)
# =============================================================================

# scurve_cdf() now imported from core.utils

# calculate_durations() now imported from core.utils

def calculate_present_value(ac, duration_months, annual_inflation_rate) -> float:
    """Calculate present value with improved validation."""
    try:
        ac = validate_numeric_input(ac, "AC", min_val=0.0)
        duration_months = validate_numeric_input(duration_months, "Duration", min_val=0.0)
        annual_inflation_rate = validate_numeric_input(annual_inflation_rate, "Inflation Rate", min_val=0.0, max_val=1.0)
        
        if annual_inflation_rate == 0 or duration_months == 0:
            return round(ac, 2)

        monthly_rate = (1 + annual_inflation_rate) ** (1/12) - 1  # Correct compound rate conversion
        pmt = ac / max(duration_months, 1e-9)
        
        try:
            factor = (1 - (1 + monthly_rate) ** (-duration_months)) / monthly_rate
            return round(max(pmt * factor, 0.0), 2)
        except (OverflowError, ValueError):
            logger.warning("PV calculation overflow, returning AC")
            return round(ac, 2)
            
    except ValueError as e:
        logger.error(f"Present value calculation failed: {e}")
        return 0.0

def calculate_present_value_of_progress(ac, ad, annual_inflation_rate) -> float:
    """Calculate Present Value of Progress: (AC / AD) * (1-(1+r)^(-AD))/r where r = (1+inflation_rate)^(1/12)"""
    try:
        ac = validate_numeric_input(ac, "AC", min_val=0.0)
        ad = validate_numeric_input(ad, "AD", min_val=0.0)
        annual_inflation_rate = validate_numeric_input(annual_inflation_rate, "Inflation Rate", min_val=0.0, max_val=1.0)

        if ad == 0:
            return 0.0
        if annual_inflation_rate == 0:
            return round(ac, 2)

        r = (1 + annual_inflation_rate) ** (1/12) - 1  # Monthly rate
        pmt = ac / ad

        try:
            factor = (1 - (1 + r) ** (-ad)) / r
            return round(max(pmt * factor, 0.0), 2)
        except (OverflowError, ValueError):
            logger.warning("Present Value of Progress calculation overflow, returning AC")
            return round(ac, 2)

    except ValueError as e:
        logger.error(f"Present Value of Progress calculation failed: {e}")
        return 0.0

def calculate_planned_value_of_project(bac, od, annual_inflation_rate) -> float:
    """Calculate Planned Value of Project: (BAC/OD) * (1-(1+r)^(-OD))/r where r = (1+inflation_rate)^(1/12)"""
    try:
        bac = validate_numeric_input(bac, "BAC", min_val=0.0)
        od = validate_numeric_input(od, "OD", min_val=0.0)
        annual_inflation_rate = validate_numeric_input(annual_inflation_rate, "Inflation Rate", min_val=0.0, max_val=1.0)

        if od == 0:
            return 0.0
        if annual_inflation_rate == 0:
            return round(bac, 2)

        r = (1 + annual_inflation_rate) ** (1/12) - 1  # Monthly rate
        pmt = bac / od

        try:
            factor = (1 - (1 + r) ** (-od)) / r
            return round(max(pmt * factor, 0.0), 2)
        except (OverflowError, ValueError):
            logger.warning("Planned Value of Project calculation overflow, returning BAC")
            return round(bac, 2)

    except ValueError as e:
        logger.error(f"Planned Value of Project calculation failed: {e}")
        return 0.0

def calculate_likely_value_of_project(bac, ld, annual_inflation_rate) -> float:
    """Calculate Likely Value of Project: (BAC/LD) * (1-(1+r)^(-LD))/r where r = (1+inflation_rate)^(1/12)"""
    try:
        bac = validate_numeric_input(bac, "BAC", min_val=0.0)
        ld = validate_numeric_input(ld, "LD", min_val=0.0)
        annual_inflation_rate = validate_numeric_input(annual_inflation_rate, "Inflation Rate", min_val=0.0, max_val=1.0)

        if ld == 0:
            return 0.0
        if annual_inflation_rate == 0:
            return round(bac, 2)

        r = (1 + annual_inflation_rate) ** (1/12) - 1  # Monthly rate
        pmt = bac / ld

        try:
            factor = (1 - (1 + r) ** (-ld)) / r
            return round(max(pmt * factor, 0.0), 2)
        except (OverflowError, ValueError):
            logger.warning("Likely Value of Project calculation overflow, returning BAC")
            return round(bac, 2)

    except ValueError as e:
        logger.error(f"Likely Value of Project calculation failed: {e}")
        return 0.0

# calculate_pv_linear() now imported from core.utils

# calculate_pv_scurve() now imported from core.utils

def calculate_evm_metrics(bac, ac, present_value, planned_value, manual_ev=None, use_manual_ev=False) -> Dict[str, float]:
    """Calculate EVM metrics with improved error handling."""
    try:
        bac = validate_numeric_input(bac, "BAC", min_val=0.0)
        ac = validate_numeric_input(ac, "AC", min_val=0.0)
        present_value = validate_numeric_input(present_value, "Present Value", min_val=0.0)
        planned_value = validate_numeric_input(planned_value, "Planned Value", min_val=0.0)

        # Use manual EV if specified, otherwise calculate automatically
        if use_manual_ev and manual_ev is not None and is_valid_finite_number(manual_ev):
            earned_value = float(manual_ev)
            percent_complete = safe_divide(earned_value, bac)
        else:
            percent_complete = safe_divide(present_value, bac, 0.0)
            earned_value = bac * percent_complete
        cost_variance = earned_value - ac
        schedule_variance = earned_value - planned_value

        # Use safe financial metrics to handle AC=0 cases
        financial_metrics = safe_financial_metrics(earned_value, ac, planned_value)
        cpi = financial_metrics['cpi']
        spi = financial_metrics['spi']

        # Handle EAC calculation for AC=0 cases
        if pd.isna(cpi) or math.isnan(cpi):  # AC=0 case
            eac = float("inf")
        else:
            eac = safe_divide(bac, cpi, float("inf")) if cpi > 1e-10 else float("inf")
        etc = eac - ac if is_valid_finite_number(eac) else float("inf")
        vac = bac - eac if is_valid_finite_number(eac) else float("-inf")
        
        return {
            'percent_complete': round(percent_complete * 100, 2),
            'ev': round(earned_value, 2),
            'cv': round(cost_variance, 2),
            'sv': round(schedule_variance, 2),
            'cpi': round(cpi, 3) if is_valid_finite_number(cpi) and not (pd.isna(cpi) or math.isnan(cpi)) else cpi,
            'spi': round(spi, 3) if is_valid_finite_number(spi) and not (pd.isna(spi) or math.isnan(spi)) else spi,
            'eac': round(eac, 2) if is_valid_finite_number(eac) else float("inf"),
            'etc': round(etc, 2) if is_valid_finite_number(etc) else float("inf"),
            'vac': round(vac, 2) if is_valid_finite_number(vac) else float("-inf"),
        }
        
    except ValueError as e:
        logger.error(f"EVM metrics calculation failed: {e}")
        return {
            'percent_complete': 0.0, 'ev': 0.0, 'cv': 0.0,
            'sv': 0.0, 'cpi': 0.0,
            'spi': 0.0, 'eac': float("inf"),
            'etc': float("inf"), 'vac': float("-inf")
        }

def find_earned_schedule_linear(earned_value, bac, total_duration) -> float:
    """Find earned schedule (linear) with validation."""
    try:
        earned_value = validate_numeric_input(earned_value, "Earned Value", min_val=0.0)
        bac = validate_numeric_input(bac, "BAC", min_val=0.01)
        total_duration = validate_numeric_input(total_duration, "Total Duration", min_val=0.0)
        
        es = safe_divide(earned_value, bac) * total_duration
        return round(max(min(es, total_duration), 0.0), 2)
        
    except ValueError as e:
        logger.error(f"Linear earned schedule calculation failed: {e}")
        return 0.0

def find_earned_schedule_scurve(earned_value, bac, total_duration, alpha=2.0, beta=2.0) -> float:
    """Find earned schedule (S-curve) with improved algorithm."""
    try:
        earned_value = validate_numeric_input(earned_value, "Earned Value", min_val=0.0)
        bac = validate_numeric_input(bac, "BAC", min_val=0.01)
        total_duration = validate_numeric_input(total_duration, "Total Duration", min_val=0.0)
        
        if total_duration <= 0:
            return 0.0
            
        target = max(min(earned_value / bac, 1.0), 0.0)
        
        if target == 0.0:
            return 0.0
        if target == 1.0:
            return total_duration
        
        # Binary search for more accurate results
        low, high = 0.0, total_duration
        tolerance = 0.01
        max_iterations = 100
        
        for _ in range(max_iterations):
            mid = (low + high) / 2
            ratio = mid / total_duration
            cdf = scurve_cdf(ratio, alpha, beta)
            
            if abs(cdf - target) < tolerance / total_duration:
                return round(mid, 2)
            elif cdf < target:
                low = mid
            else:
                high = mid
        
        return round((low + high) / 2, 2)
        
    except ValueError as e:
        logger.error(f"S-curve earned schedule calculation failed: {e}")
        return 0.0

def add_months_approx(start_dt: datetime, months: int) -> datetime:
    """Add months with improved calendar handling."""
    try:
        months = int(months)
        y = start_dt.year + (start_dt.month - 1 + months) // 12
        m = (start_dt.month - 1 + months) % 12 + 1
        
        # Handle leap years properly
        days_in_month = [31, 28, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31]
        if m == 2 and ((y % 4 == 0 and y % 100 != 0) or (y % 400 == 0)):
            days_in_month[1] = 29
            
        d = min(start_dt.day, days_in_month[m-1])
        return datetime(y, m, d)
    except (ValueError, OverflowError) as e:
        logger.error(f"Error adding {months} months to {start_dt}: {e}")
        return start_dt

def calculate_earned_schedule_metrics(earned_schedule, actual_duration, total_duration, plan_start, original_duration=None) -> Dict[str, Union[float, int, str]]:
    """Calculate earned schedule metrics with validation."""
    try:
        earned_schedule = validate_numeric_input(earned_schedule, "Earned Schedule", min_val=0.0)
        actual_duration = max(validate_numeric_input(actual_duration, "Actual Duration", min_val=0.0), 1e-9)
        total_duration = validate_numeric_input(total_duration, "Total Duration", min_val=0.0)
        ps = parse_date_any(plan_start)
        
        spie = safe_divide(earned_schedule, actual_duration, 1.0)
        forecast_duration = safe_calculate_forecast_duration(total_duration, spie, original_duration)
        
        if is_valid_finite_number(forecast_duration):
            forecast_duration_rounded = max(1, math.ceil(forecast_duration))
            fcomp = add_months_approx(ps, forecast_duration_rounded)
            fcomp_str = fcomp.strftime('%Y-%m-%d')
        else:
            forecast_duration_rounded = None
            fcomp_str = "N/A"
        
        return {
            'es': round(earned_schedule, 2),
            'spie': round(spie, 3),
            'forecast_duration': int(forecast_duration_rounded) if forecast_duration_rounded is not None else None,
            'forecast_completion': fcomp_str
        }
        
    except (ValueError, OverflowError) as e:
        logger.error(f"Earned schedule metrics calculation failed: {e}")
        return {
            'es': 0.0,
            'spie': 1.0,
            'forecast_duration': None,
            'forecast_completion': "N/A"
        }

# NOTE: The local perform_complete_evm_analysis function has been REMOVED
# We now use the centralized version from core.evm_engine (imported at line 43-44)
# This wrapper normalizes field names for backward compatibility

# IMPORTANT: Cache key includes a version to force cache refresh when needed
@st.cache_data(show_spinner=False)
def perform_complete_evm_analysis_wrapper(bac, ac, plan_start, plan_finish, data_date,
                                          annual_inflation_rate, curve_type='linear', alpha=2.0, beta=2.0,
                                          manual_pv=None, use_manual_pv=False, manual_ev=None, use_manual_ev=False,
                                          _cache_version='v2') -> Dict[str, Any]:
    """Wrapper for centralized EVM analysis with field name normalization.

    Args:
        _cache_version: Internal parameter to force cache invalidation when needed
    """
    # Import the actual function from core.evm_engine
    from core.evm_engine import perform_complete_evm_analysis as evm_analysis

    # Call the centralized EVM engine
    results = evm_analysis(
        bac, ac, plan_start, plan_finish, data_date,
        annual_inflation_rate, curve_type, alpha, beta,
        manual_pv, use_manual_pv, manual_ev, use_manual_ev
    )

    # Normalize field names for backward compatibility
    # evm_engine returns 'likely_completion', but Project Analysis expects 'forecast_completion'
    if 'likely_completion' in results and 'forecast_completion' not in results:
        results['forecast_completion'] = results['likely_completion']

    # Also add 'forecast_duration' if it's using 'ld' instead
    if 'ld' in results and 'forecast_duration' not in results:
        results['forecast_duration'] = results['ld']

    return results

# =============================================================================
# CSV HANDLING
# =============================================================================

def try_read_csv(file, header_option: str) -> pd.DataFrame:
    """Read CSV with improved error handling."""
    encodings = ['utf-8-sig', 'utf-8', 'latin1', 'cp1252']
    
    for encoding in encodings:
        try:
            file.seek(0)
            if header_option == "First row has headers":
                return pd.read_csv(file, encoding=encoding)
            elif header_option == "No header (assign generic)":
                return pd.read_csv(file, header=None, encoding=encoding)
            else:  # Auto-detect
                return pd.read_csv(file, encoding=encoding)
        except UnicodeDecodeError:
            continue
        except Exception as e:
            logger.warning(f"CSV read failed with encoding {encoding}: {e}")
            continue
    
    raise ValueError("Unable to read CSV file with any supported encoding")

# =============================================================================
# BATCH RESULTS FORMATTING FUNCTIONS
# =============================================================================

def format_batch_results_for_download(batch_df: pd.DataFrame) -> pd.DataFrame:
    """Reorganize batch results DataFrame with requested column order and formatting."""
    if batch_df.empty:
        return batch_df

    # Create a new DataFrame with the required columns in the specified order
    formatted_df = pd.DataFrame()

    # Required columns in the specified order
    column_mapping = {
        'Project ID': 'project_id',
        'Project Name': 'project_name',
        'Organization': 'organization',
        'Project Manager': 'project_manager',
        'Budget': 'bac',
        'Plan Start': 'plan_start',
        'Plan Finish': 'plan_finish',
        'Likely Finish': 'forecast_completion',
        '% Budget Used': 'percent_budget_used',
        '% Time Used': 'percent_time_used',
        'Original Dur': 'original_duration_months',
        'Actual Dur': 'actual_duration_months',
        'Likely Dur': 'forecast_duration',
        'Actual Cost': 'ac',
        'Present Value': 'present_value',
        'Plan Value': 'pv',
        'Earned Value': 'ev',
        # 'Present Value Progress': 'present_value_progress', # Removed duplicate
        'Planned Value Project': 'planned_value_project',
        'Likely Value Project': 'likely_value_project',
        '% Present Value Project': 'percent_present_value_project',
        '% Likely Value Project': 'percent_likely_value_project',
        'CPI': 'cpi',
        'SPI': 'spi',
        'SPIe': 'spie',
        'ETC': 'etc',
        'EAC': 'eac'
    }

    # Build the formatted DataFrame
    for display_name, source_col in column_mapping.items():
        if source_col in batch_df.columns:
            if display_name in ['% Budget Used', '% Time Used', '% Present Value Project', '% Likely Value Project']:
                # Divide by 100 for proper Excel percentage display
                formatted_df[display_name] = batch_df[source_col] / 100
            else:
                # Copy values as-is for all other columns
                formatted_df[display_name] = batch_df[source_col]
        else:
            # Handle missing columns with appropriate defaults
            if display_name in ['Project ID', 'Project Name', 'Organization', 'Project Manager']:
                formatted_df[display_name] = 'N/A'
            elif display_name in ['Budget', 'Actual Cost', 'Plan Value', 'Earned Value', 'ETC', 'EAC', 'Present Value', 'Planned Value Project', 'Likely Value Project']:
                formatted_df[display_name] = 0.0
            elif display_name in ['CPI', 'SPI', 'SPIe']:
                formatted_df[display_name] = 1.0
            elif display_name in ['% Budget Used', '% Time Used', '% Present Value Project', '% Likely Value Project']:
                formatted_df[display_name] = 0.0
            elif display_name in ['Original Dur', 'Actual Dur', 'Likely Dur']:
                formatted_df[display_name] = 0
            else:
                formatted_df[display_name] = 'N/A'

    return formatted_df

def format_batch_results_for_display(batch_df: pd.DataFrame, currency_symbol: str = "$", currency_postfix: str = "") -> pd.DataFrame:
    """Format batch results DataFrame for display with proper formatting (percentages, currencies, dates, durations)."""
    if batch_df.empty:
        return batch_df

    # Create a copy for display formatting
    display_df = batch_df.copy()

    # Apply formatting to specific columns (moved to percentage_cols section below)

    # Format currency columns including new financial analysis fields
    currency_cols = ['bac', 'ac', 'pv', 'ev', 'present_value',
                     'planned_value_project', 'likely_value_project', 'etc', 'eac']
    for col in currency_cols:
        if col in display_df.columns:
            new_col_name = {
                'bac': 'Budget',
                'ac': 'Actual Cost',
                'pv': 'Plan Value',
                'ev': 'Earned Value',
                'present_value': 'Present Value',
                # 'present_value_progress': 'Present Value Progress', # Removed duplicate
                'planned_value_project': 'Planned Value Project',
                'likely_value_project': 'Likely Value Project',
                'etc': 'ETC',
                'eac': 'EAC'
            }.get(col, col)
            display_df[new_col_name] = display_df[col].apply(lambda x: format_currency(x, currency_symbol, currency_postfix) if pd.notna(x) else "—")

    # Format percentage columns including new financial percentages
    percentage_cols = ['percent_budget_used', 'percent_time_used', 'percent_present_value_project', 'percent_likely_value_project']
    for col in percentage_cols:
        if col in display_df.columns:
            new_col_name = {
                'percent_budget_used': '% Budget Used',
                'percent_time_used': '% Time Used',
                'percent_present_value_project': '% Present Value Project',
                'percent_likely_value_project': '% Likely Value Project'
            }.get(col, col)
            display_df[new_col_name] = display_df[col].apply(lambda x: f"{x:.2f}%" if pd.notna(x) else "—")

    # Format performance indices
    perf_cols = ['cpi', 'spi', 'spie']
    for col in perf_cols:
        if col in display_df.columns:
            new_col_name = {'cpi': 'CPI', 'spi': 'SPI', 'spie': 'SPIe'}.get(col, col)
            display_df[new_col_name] = display_df[col].apply(lambda x: format_performance_index(x) if pd.notna(x) else "N/A")

    # Format duration columns
    duration_cols = ['original_duration_months', 'actual_duration_months', 'forecast_duration']
    for col in duration_cols:
        if col in display_df.columns:
            new_col_name = {'original_duration_months': 'Original Dur', 'actual_duration_months': 'Actual Dur', 'forecast_duration': 'Likely Dur'}.get(col, col)
            display_df[new_col_name] = display_df[col].apply(lambda x: format_duration(x) if pd.notna(x) else "—")

    # Format date columns
    date_cols = ['plan_start', 'plan_finish', 'forecast_completion']
    for col in date_cols:
        if col in display_df.columns:
            new_col_name = {'plan_start': 'Plan Start', 'plan_finish': 'Plan Finish', 'forecast_completion': 'Likely Finish'}.get(col, col)
            display_df[new_col_name] = display_df[col].apply(lambda x: format_date_dmy(x) if pd.notna(x) and str(x) != 'nan' else "N/A")

    # Rename remaining columns
    rename_map = {
        'project_id': 'Project ID',
        'project_name': 'Project Name'
    }

    for old_name, new_name in rename_map.items():
        if old_name in display_df.columns:
            display_df[new_name] = display_df[old_name]

    # Select and reorder columns for display
    display_columns = [
        'Project ID', 'Project Name', 'Budget', 'Plan Start', 'Plan Finish', 'Likely Finish',
        '% Budget Used', '% Time Used', 'Original Dur', 'Actual Dur', 'Likely Dur',
        'Actual Cost', 'Present Value', 'Plan Value', 'Earned Value',
        'Planned Value Project', 'Likely Value Project',
        '% Present Value Project', '% Likely Value Project',
        'CPI', 'SPI', 'SPIe', 'ETC', 'EAC'
    ]

    # Only include columns that exist
    available_columns = [col for col in display_columns if col in display_df.columns]
    return display_df[available_columns]

# =============================================================================
# BATCH CALCULATION FUNCTIONS
# =============================================================================

def perform_batch_calculation(df: pd.DataFrame, column_mapping: Dict[str, str], 
                            curve_type: str, alpha: float, beta: float, 
                            data_date: date, inflation_rate: float) -> pd.DataFrame:
    """Perform EVM calculations on entire dataset."""
    results_list = []
    
    # Ensure we have a clean copy of the dataframe
    df_clean = df.copy().reset_index(drop=True)
    
    for idx, row in df_clean.iterrows():
        try:
            # Safely extract values using column mapping
            pid_col = column_mapping.get('pid_col')
            bac_col = column_mapping.get('bac_col')
            ac_col = column_mapping.get('ac_col')
            st_col = column_mapping.get('st_col')
            fn_col = column_mapping.get('fn_col')
            
            if not all([pid_col, bac_col, ac_col, st_col, fn_col]):
                raise ValueError("Missing required column mappings")
            
            # Extract values with proper error handling
            project_id = str(row[pid_col]) if pid_col in row.index else f"Row_{idx}"
            
            # Stop processing if we encounter blank/empty Project ID
            if pd.isna(row[pid_col]) or str(row[pid_col]).strip() == "":
                break
                
            bac = float(row[bac_col]) if bac_col in row.index else 0.0
            ac = float(row[ac_col]) if ac_col in row.index else 0.0
            plan_start = row[st_col] if st_col in row.index else None
            plan_finish = row[fn_col] if fn_col in row.index else None
            
            if plan_start is None or plan_finish is None:
                raise ValueError("Missing plan start or finish date")
            
            # Optional fields - safe extraction
            pname_col = column_mapping.get('pname_col')
            org_col = column_mapping.get('org_col')
            pm_col = column_mapping.get('pm_col')
            pv_col = column_mapping.get('pv_col')
            ev_col = column_mapping.get('ev_col')
            
            project_name = str(row[pname_col]) if pname_col and pname_col in row.index else ""
            organization = str(row[org_col]) if org_col and org_col in row.index else ""
            project_manager = str(row[pm_col]) if pm_col and pm_col in row.index else ""
            
            # Clean up nan/null values
            if project_name == 'nan' or pd.isna(project_name):
                project_name = ""
            if organization == 'nan' or pd.isna(organization):
                organization = ""
            if project_manager == 'nan' or pd.isna(project_manager):
                project_manager = ""
            
            # Check for manual PV settings
            use_manual_pv = False
            manual_pv_val = None
            
            # First check if we have a mapped PV column
            if pv_col and pv_col in row.index and not pd.isna(row[pv_col]):
                try:
                    manual_pv_val = float(row[pv_col])
                    use_manual_pv = True
                    # Validate: if manual_pv > bac, clamp to bac
                    if manual_pv_val > bac:
                        manual_pv_val = bac
                except (ValueError, TypeError):
                    manual_pv_val = None
                    use_manual_pv = False
            
            # Then check for Use_Manual_PV and Manual_PV columns (takes precedence)
            if 'Use_Manual_PV' in row.index:
                use_manual_pv = bool(row.get('Use_Manual_PV', False))
            if 'Manual_PV' in row.index and use_manual_pv:
                try:
                    manual_pv_val = float(row.get('Manual_PV'))
                    # Validate: if manual_pv > bac, clamp to bac
                    if manual_pv_val > bac:
                        manual_pv_val = bac
                except (ValueError, TypeError):
                    manual_pv_val = None
                    use_manual_pv = False
            
            # Check for manual EV settings
            use_manual_ev = False
            manual_ev_val = None
            
            # First check if we have a mapped EV column
            if ev_col and ev_col in row.index and not pd.isna(row[ev_col]):
                try:
                    manual_ev_val = float(row[ev_col])
                    use_manual_ev = True
                    # Validate: if manual_ev > bac, clamp to bac
                    if manual_ev_val > bac:
                        manual_ev_val = bac
                except (ValueError, TypeError):
                    manual_ev_val = None
                    use_manual_ev = False
            
            # Then check for Use_Manual_EV and Manual_EV columns (takes precedence)
            if 'Use_Manual_EV' in row.index:
                use_manual_ev = bool(row.get('Use_Manual_EV', False))
            if 'Manual_EV' in row.index and use_manual_ev:
                try:
                    manual_ev_val = float(row.get('Manual_EV'))
                    # Validate: if manual_ev > bac, clamp to bac
                    if manual_ev_val > bac:
                        manual_ev_val = bac
                except (ValueError, TypeError):
                    manual_ev_val = None
                    use_manual_ev = False
            
            # Perform EVM analysis using wrapper for field name compatibility
            results = perform_complete_evm_analysis_wrapper(
                bac=bac, ac=ac, plan_start=plan_start, plan_finish=plan_finish,
                data_date=data_date, annual_inflation_rate=inflation_rate/100.0,
                curve_type=curve_type, alpha=alpha, beta=beta,
                manual_pv=manual_pv_val, use_manual_pv=use_manual_pv,
                manual_ev=manual_ev_val, use_manual_ev=use_manual_ev
            )
            
            # Add project info to results
            results['project_id'] = project_id
            results['project_name'] = project_name
            results['organization'] = organization
            results['project_manager'] = project_manager
            results['calculation_date'] = datetime.now().isoformat()
            results['row_index'] = idx
            
            results_list.append(results)
            
        except Exception as e:
            logger.error(f"Batch calculation failed for row {idx}: {e}")
            # Add error record with safe project ID extraction
            try:
                error_pid = str(row[column_mapping.get('pid_col', 'Project ID')]) if column_mapping.get('pid_col') else f"Row_{idx}"
            except:
                error_pid = f"Row_{idx}"
                
            error_result = {
                'project_id': error_pid,
                'error': str(e),
                'calculation_date': datetime.now().isoformat(),
                'row_index': idx
            }
            results_list.append(error_result)
    
    # Convert to DataFrame with better error handling
    try:
        return pd.DataFrame(results_list)
    except Exception as e:
        logger.error(f"Failed to create results DataFrame: {e}")
        # Return minimal error DataFrame
        return pd.DataFrame([{
            'error': 'Failed to process batch calculation results',
            'calculation_date': datetime.now().isoformat()
        }])

# =============================================================================
# LLM INTEGRATION
# =============================================================================


def create_portfolio_executive_summary(portfolio_summary: Dict[str, Any], controls: Dict[str, Any]) -> str:
    """Create executive summary prompt for portfolio health assessment."""
    try:
        currency = controls['currency_symbol']
        postfix = controls['currency_postfix']
        
        # Format key financial metrics
        total_budget = format_currency(portfolio_summary['total_bac'], currency, postfix)
        total_ac = format_currency(portfolio_summary['total_ac'], currency, postfix)
        total_ev = format_currency(portfolio_summary['total_ev'], currency, postfix)
        total_pv = format_currency(portfolio_summary['total_pv'], currency, postfix)
        
        # Calculate key ratios
        portfolio_cpi = portfolio_summary['portfolio_cpi']
        portfolio_spi = portfolio_summary['portfolio_spi']
        portfolio_tcpi = portfolio_summary['portfolio_tcpi']
        weighted_avg_cpi = portfolio_summary['weighted_avg_cpi']
        weighted_avg_spi = portfolio_summary['weighted_avg_spi']
        avg_progress = portfolio_summary['average_progress']
        
        # Budget utilization
        budget_used_pct = safe_divide(portfolio_summary['total_ac'], portfolio_summary['total_bac']) * 100
        value_earned_pct = safe_divide(portfolio_summary['total_ev'], portfolio_summary['total_bac']) * 100
        
        # Performance quadrants
        quadrants = portfolio_summary['quadrants']
        total_projects = portfolio_summary['total_projects']
        
        prompt = f"""
As a senior project portfolio management consultant, analyze this portfolio performance data and provide a comprehensive executive briefing.

PORTFOLIO OVERVIEW:
- Total Projects: {total_projects}
- Total Portfolio Budget: {total_budget}
- Total Actual Cost: {total_ac}
- Total Planned Value: {total_pv}  
- Total Earned Value: {total_ev}
- Average Project Progress: {avg_progress:.1f}%

PERFORMANCE INDICES:
- Portfolio Cost Performance Index (CPI): {portfolio_cpi:.2f}
- Portfolio Schedule Performance Index (SPI): {portfolio_spi:.2f}
- Portfolio To Complete Performance Index (TCPI): {portfolio_tcpi:.2f}
- Weighted Average CPI: {weighted_avg_cpi:.2f}
- Weighted Average SPI: {weighted_avg_spi:.2f}

FINANCIAL UTILIZATION:
- Budget Utilized: {budget_used_pct:.1f}%
- Value Earned: {value_earned_pct:.1f}%

PROJECT PERFORMANCE DISTRIBUTION:
- On Budget & On Schedule: {quadrants['on_budget_on_schedule']} projects ({(quadrants['on_budget_on_schedule']/total_projects)*100:.1f}%)
- On Budget & Behind Schedule: {quadrants['on_budget_behind_schedule']} projects ({(quadrants['on_budget_behind_schedule']/total_projects)*100:.1f}%)
- Over Budget & On Schedule: {quadrants['over_budget_on_schedule']} projects ({(quadrants['over_budget_on_schedule']/total_projects)*100:.1f}%)
- Over Budget & Behind Schedule: {quadrants['over_budget_behind_schedule']} projects ({(quadrants['over_budget_behind_schedule']/total_projects)*100:.1f}%)

Please provide:

1. **EXECUTIVE SUMMARY** (2-3 key bullets on overall portfolio health)

2. **FINANCIAL PERFORMANCE ASSESSMENT** - Cost performance analysis
   - Budget utilization efficiency
   - Value realization status

3. **SCHEDULE PERFORMANCE ASSESSMENT**
   - Schedule adherence analysis
   - Timeline risk assessment

4. **PORTFOLIO HEALTH RATING** (Excellent/Good/Fair/Poor with rationale)

5. **KEY RISKS & CONCERNS** (Top 3 portfolio risks)

6. **STRATEGIC RECOMMENDATIONS** (Top 5 actionable recommendations for portfolio improvement)

7. **IMMEDIATE ACTIONS REQUIRED** (Next 30-60 days priority actions)

Focus on strategic insights and actionable recommendations for senior leadership. Use professional language appropriate for C-suite executives and portfolio sponsors.
"""
        return prompt.strip()
        
    except Exception as e:
        logger.error(f"Portfolio executive summary creation failed: {e}")
        return "Error creating portfolio summary prompt"

def safe_llm_request(provider: str, model: str, api_key: str, 
                    temperature: float, timeout_sec: int, prompt: str) -> str:
    """Make LLM request with comprehensive error handling."""
    try:
        if not prompt.strip():
            return "Error: Empty prompt provided"
        
        if not api_key.strip():
            return "No API key available"
        
        timeout_sec = max(MIN_TIMEOUT_SECONDS, min(timeout_sec, MAX_TIMEOUT_SECONDS))
        temperature = max(0.0, min(temperature, 2.0))
        
        if provider == "OpenAI":
            url = "https://api.openai.com/v1/chat/completions"
            headers = {
                "Authorization": f"Bearer {api_key.strip()}",
                "Content-Type": "application/json"
            }
            payload = {
                "model": model.strip(),
                "messages": [
                    {"role": "system", "content": "You are a project controls expert specializing in earned value management."},
                    {"role": "user", "content": prompt.strip()}
                ],
                "temperature": float(temperature),
                "max_tokens": 2000
            }
            
            response = requests.post(url, headers=headers, json=payload, timeout=timeout_sec)
            response.raise_for_status()
            data = response.json()
            
            if 'choices' not in data or not data['choices']:
                return "Error: Invalid response from OpenAI API"
                
            return data["choices"][0]["message"]["content"].strip()
            
        elif provider == "Gemini":
            # Ensure model has the correct format (without 'models/' prefix in the name itself)
            model_name = model.strip()
            if not model_name.startswith('models/'):
                model_name = f"models/{model_name}"
            url = f"https://generativelanguage.googleapis.com/v1beta/{model_name}:generateContent"
            headers = {
                "Content-Type": "application/json"
            }
            params = {
                "key": api_key.strip()
            }
            
            # System message is included in the user prompt for Gemini
            full_prompt = f"You are a project controls expert specializing in earned value management.\n\n{prompt.strip()}"
            
            payload = {
                "contents": [
                    {
                        "parts": [
                            {"text": full_prompt}
                        ]
                    }
                ],
                "generationConfig": {
                    "temperature": float(temperature),
                    "maxOutputTokens": 8000  # Increased from 2000 to allow longer responses
                }
            }

            response = requests.post(url, headers=headers, params=params, json=payload, timeout=timeout_sec)
            response.raise_for_status()
            data = response.json()

            # Debug: Log the response structure
            if 'candidates' not in data or not data['candidates']:
                return f"Error: Invalid response from Gemini API. Response keys: {list(data.keys())}"

            candidate = data['candidates'][0]

            # Check for finish reason (if response was truncated)
            finish_reason = candidate.get('finishReason', 'UNKNOWN')

            if 'content' not in candidate:
                return f"Error: No 'content' in candidate. Candidate keys: {list(candidate.keys())}"

            if 'parts' not in candidate['content']:
                return f"Error: No 'parts' in content. Content keys: {list(candidate['content'].keys())}"

            if not candidate['content']['parts']:
                return f"Error: Empty 'parts' list in response"

            if 'text' not in candidate['content']['parts'][0]:
                return f"Error: No 'text' in parts[0]. Parts[0] keys: {list(candidate['content']['parts'][0].keys())}"

            response_text = candidate['content']['parts'][0]['text'].strip()

            # Add warning if response was truncated
            if finish_reason == 'MAX_TOKENS':
                response_text += "\n\n⚠️ **Note**: Response was truncated due to length limits. Consider increasing maxOutputTokens."
            elif finish_reason not in ['STOP', 'UNKNOWN']:
                response_text += f"\n\n⚠️ **Note**: Response ended with reason: {finish_reason}"

            return response_text

        elif provider == "Claude":
            url = "https://api.anthropic.com/v1/messages"
            headers = {
                "x-api-key": api_key.strip(),
                "anthropic-version": "2023-06-01",
                "Content-Type": "application/json"
            }

            payload = {
                "model": model.strip(),
                "max_tokens": 4096,
                "temperature": float(temperature),
                "system": "You are a project controls expert specializing in earned value management.",
                "messages": [
                    {
                        "role": "user",
                        "content": prompt.strip()
                    }
                ]
            }

            response = requests.post(url, headers=headers, json=payload, timeout=timeout_sec)
            response.raise_for_status()
            data = response.json()

            if 'content' not in data or not data['content']:
                return "Error: Invalid response from Claude API"

            # Extract text from content blocks
            response_text = ""
            for block in data['content']:
                if block.get('type') == 'text':
                    response_text += block.get('text', '')

            if not response_text.strip():
                return "Error: No text content in Claude response"

            # Check for stop reason
            stop_reason = data.get('stop_reason', 'UNKNOWN')
            if stop_reason == 'max_tokens':
                response_text += "\n\n⚠️ **Note**: Response was truncated due to length limits."
            elif stop_reason not in ['end_turn', 'UNKNOWN']:
                response_text += f"\n\n⚠️ **Note**: Response ended with reason: {stop_reason}"

            return response_text.strip()

        elif provider == "Kimi":
            url = "https://api.moonshot.ai/v1/chat/completions"
            headers = {
                "Authorization": f"Bearer {api_key.strip()}",
                "Content-Type": "application/json"
            }
            payload = {
                "model": model.strip(),
                "messages": [
                    {"role": "system", "content": "You are a project controls expert specializing in earned value management."},
                    {"role": "user", "content": prompt.strip()}
                ],
                "temperature": float(temperature),
                "max_tokens": 4000
            }

            response = requests.post(url, headers=headers, json=payload, timeout=timeout_sec)
            response.raise_for_status()
            data = response.json()

            if 'choices' not in data or not data['choices']:
                return "Error: Invalid response from Kimi API"

            return data["choices"][0]["message"]["content"].strip()

        else:
            return f"Error: Unsupported provider '{provider}'"

    except requests.exceptions.Timeout:
        return f"Error: Request timed out after {timeout_sec} seconds"
    except requests.exceptions.ConnectionError:
        return "Error: Unable to connect to the API endpoint"
    except requests.exceptions.HTTPError as e:
        return f"Error: HTTP {e.response.status_code} - {e.response.text[:200]}"
    except json.JSONDecodeError:
        return "Error: Invalid JSON response from API"
    except Exception as e:
        logger.error(f"LLM request failed: {e}")
        return f"Error: {str(e)[:200]}"

# =============================================================================
# ENHANCED SIDEBAR COMPONENTS
# =============================================================================

def normalize_dataframe_columns(df: pd.DataFrame, column_mapping: Dict[str, str]) -> pd.DataFrame:
    """Normalize DataFrame columns to standard names for consistent manual entry operations."""
    if df.empty or not column_mapping:
        return df
    
    # Standard column mapping for manual entry compatibility
    standard_mapping = {
        'pid_col': 'Project ID',
        'pname_col': 'Project', 
        'org_col': 'Organization',
        'pm_col': 'Project Manager',
        'bac_col': 'BAC',
        'ac_col': 'AC', 
        'st_col': 'Plan Start',
        'fn_col': 'Plan Finish'
    }
    
    # Create a copy to avoid modifying original
    normalized_df = df.copy()

    # Define mandatory fields that must have values
    # Based on user requirements: Project ID, Start Date, Finish Date, BAC, ACBAC, AC, Plan Start and Plan Finish
    # Note: ACBAC might be same as BAC, Start/Finish Date might be same as Plan Start/Finish
    mandatory_fields = ['pid_col', 'bac_col', 'ac_col', 'st_col', 'fn_col']

    # Get the actual column names for mandatory fields
    mandatory_columns = []
    for field in mandatory_fields:
        if field in column_mapping and column_mapping[field] in normalized_df.columns:
            mandatory_columns.append(column_mapping[field])

    # Track skipped rows for logging
    skipped_rows = []
    valid_indices = []

    # Check each row for mandatory field completeness
    for idx, row in normalized_df.iterrows():
        has_missing_mandatory = False
        missing_fields = []

        for col in mandatory_columns:
            value = row[col]
            # Check if value is missing, null, or empty string
            if pd.isna(value) or str(value).strip() == '':
                has_missing_mandatory = True
                missing_fields.append(col)

        if has_missing_mandatory:
            skipped_rows.append({
                'index': idx,
                'missing_fields': missing_fields,
                'project_id': str(row[column_mapping.get('pid_col', '')]) if column_mapping.get('pid_col', '') in row.index else f"Row_{idx}"
            })
        else:
            valid_indices.append(idx)

    # Filter to keep only valid rows
    if skipped_rows:
        normalized_df = normalized_df.loc[valid_indices].copy()
        logger.warning(f"Skipped {len(skipped_rows)} rows with missing mandatory fields:")
        for skip in skipped_rows:
            logger.warning(f"  Row {skip['index']} (Project ID: {skip['project_id']}): Missing {', '.join(skip['missing_fields'])}")

    if normalized_df.empty:
        logger.warning("All rows were skipped due to missing mandatory fields")
        return normalized_df

    # Rename columns based on mapping
    rename_dict = {}
    for map_key, original_col in column_mapping.items():
        if map_key in standard_mapping and original_col in normalized_df.columns:
            standard_name = standard_mapping[map_key]
            if original_col != standard_name:  # Only rename if different
                rename_dict[original_col] = standard_name
    
    if rename_dict:
        normalized_df = normalized_df.rename(columns=rename_dict)
    
    # Add manual flag columns for PV and EV if mapped columns contain non-zero data
    pv_col = column_mapping.get('pv_col')
    ev_col = column_mapping.get('ev_col')
    
    # Initialize manual flag columns if they don't exist
    if 'Use_Manual_PV' not in normalized_df.columns:
        normalized_df['Use_Manual_PV'] = False
    if 'Manual_PV' not in normalized_df.columns:
        normalized_df['Manual_PV'] = None
    if 'Use_Manual_EV' not in normalized_df.columns:
        normalized_df['Use_Manual_EV'] = False
    if 'Manual_EV' not in normalized_df.columns:
        normalized_df['Manual_EV'] = None
    
    # Process Manual PV from mapped column
    if pv_col and pv_col in normalized_df.columns:
        for idx, row in normalized_df.iterrows():
            pv_value = row[pv_col]
            if pd.notna(pv_value) and pv_value != 0:
                try:
                    pv_float = float(pv_value)
                    if pv_float > 0:
                        normalized_df.at[idx, 'Use_Manual_PV'] = True
                        normalized_df.at[idx, 'Manual_PV'] = pv_float
                except (ValueError, TypeError):
                    pass  # Skip invalid values
    
    # Process Manual EV from mapped column
    if ev_col and ev_col in normalized_df.columns:
        for idx, row in normalized_df.iterrows():
            ev_value = row[ev_col]
            if pd.notna(ev_value) and ev_value != 0:
                try:
                    ev_float = float(ev_value)
                    if ev_float > 0:
                        normalized_df.at[idx, 'Use_Manual_EV'] = True
                        normalized_df.at[idx, 'Manual_EV'] = ev_float
                except (ValueError, TypeError):
                    pass  # Skip invalid values
    
    return normalized_df

def render_data_source_section():
    """Render A. Data Source section."""
    st.markdown('<div class="sidebar-section">', unsafe_allow_html=True)
    st.markdown('<div class="section-header">A. Data Source</div>', unsafe_allow_html=True)
    
    # Simple data source selection
    data_source = st.radio(
        "Select Data Source",
        options=["Load JSON", "Load CSV", "Manual Entry"],
        index=0,
        key="data_source_radio",
        help="Choose JSON for comprehensive data with configuration, CSV for data-only imports, or Manual Entry to create a blank table"
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
    else:  # Manual Entry
        st.info("💡 Create a blank table that you can fill in manually with your project data")
        file_types = None
        help_text = "Start with an empty table and add your projects manually"
    
    # File uploader for JSON and CSV options only
    uploaded_file = None
    if data_source != "Manual Entry":
        uploaded_file = st.file_uploader(
            "Choose file",
            type=file_types,
            key="unified_file_uploader",
            help=help_text,
            label_visibility="visible"
        )
    
    # Process uploaded file based on selection
    if uploaded_file is not None:
        # Create a unique identifier for the current file to avoid reprocessing
        current_file_info = (uploaded_file.name, uploaded_file.size)

        # Process only if it's a new file
        if st.session_state.get('processed_file_info') != current_file_info:
            try:
                if data_source == "Load JSON":
                    # JSON processing - direct read
                    df, config_data, filename = load_json_file(uploaded_file)
                    st.success(f"✅ JSON loaded: {filename}")
                    
                    if not df.empty:
                        st.session_state.data_df = df
                        st.session_state.data_loaded = True
                        st.session_state.config_dict.update(config_data)
                        st.session_state.original_filename = filename
                        st.session_state.file_type = 'json'

                        # Simple approach - just confirm the JSON has controls
                        if 'controls' in config_data:
                            controls = config_data['controls']
                            st.success(f"✅ JSON settings loaded: {controls.get('curve_type', 'N/A')}, {controls.get('currency_symbol', 'N/A')} {controls.get('currency_postfix', 'N/A')}")
                            st.info("💡 Controls should now show the correct values below")
                    else:
                        st.warning("⚠️ No project data found in JSON file")
                        
                else:  # CSV reading (headers only)
                    df, filename = load_csv_file(uploaded_file)
                    st.success(f"✅ CSV file read: {filename}")
                    st.info("📋 Please configure column mapping below, then click 'Load CSV Data' to import")
                    
                    # Store raw CSV data for mapping, but don't process yet
                    st.session_state.raw_csv_df = df
                    st.session_state.csv_filename = filename
                    st.session_state.file_type = 'csv'
                
                # After successful processing, store the file's info
                st.session_state.processed_file_info = current_file_info
                # Clear cache to ensure UI updates with new data
                st.cache_data.clear()
                st.rerun()

            except Exception as e:
                st.error(f"Failed to load file: {e}")
                # Clear the info if processing fails
                if 'processed_file_info' in st.session_state:
                    del st.session_state['processed_file_info']

    # Handle CSV column mapping workflow
    if hasattr(st.session_state, 'raw_csv_df') and st.session_state.raw_csv_df is not None:
        df = st.session_state.raw_csv_df
        
        with st.expander("📊 Data Preview", expanded=False):
            st.dataframe(df.head(10), width="stretch")
            st.caption(f"{len(df)} rows available for mapping.")
            
        # Get stored column mappings from session state
        stored_mappings = st.session_state.config_dict.get('column_mappings', {})
        column_mapping = render_column_mapping_section(df.columns.tolist(), stored_mappings)
        
        # Store column mappings in session state
        if 'column_mappings' not in st.session_state.config_dict:
            st.session_state.config_dict['column_mappings'] = {}
        st.session_state.config_dict['column_mappings'] = column_mapping
        
        # Load CSV Data button
        if st.button("📥 Load CSV Data", type="primary", help="Apply column mapping and import CSV data"):
            try:
                # Validate required mappings
                required_keys = ['pid_col', 'bac_col', 'ac_col', 'st_col', 'fn_col']
                missing_mappings = [key.replace('_col', '').upper() for key in required_keys if not column_mapping.get(key)]
                
                if missing_mappings:
                    st.error(f"Please map required columns: {', '.join(missing_mappings)}")
                else:
                    # Apply normalization and import
                    normalized_df = normalize_dataframe_columns(df, column_mapping)

                    # Update column mapping to use normalized column names
                    normalized_column_mapping = {
                        'pid_col': 'Project ID',
                        'pname_col': 'Project',
                        'org_col': 'Organization',
                        'pm_col': 'Project Manager',
                        'bac_col': 'BAC',
                        'ac_col': 'AC',
                        'st_col': 'Plan Start',
                        'fn_col': 'Plan Finish'
                    }

                    st.session_state.data_df = normalized_df
                    st.session_state.data_loaded = True
                    st.session_state.original_filename = st.session_state.csv_filename
                    st.session_state.file_type = "csv"
                    st.session_state.config_dict.update({
                        "import_metadata": {
                            "import_date": datetime.now().isoformat(),
                            "source": "csv_import",
                            "original_filename": st.session_state.csv_filename
                        },
                        "column_mappings": normalized_column_mapping
                    })
                    
                    # Clear raw CSV data
                    st.session_state.raw_csv_df = None
                    st.session_state.csv_filename = None
                    
                    st.success(f"✅ CSV data imported successfully!")
                    st.cache_data.clear()
                    st.rerun()
                    
            except Exception as e:
                st.error(f"Failed to import CSV: {e}")

    # Handle Manual Entry option
    elif data_source == "Manual Entry":
        if st.button("📝 Create Demo Project", type="primary", help="Create a demo project to get started with manual entry"):
            # Create dataframe with demo record for manual entry
            blank_data = {
                "Project ID": ["Demo"],
                "Project": ["Demo"],
                "Organization": ["Demo Org"],
                "Project Manager": ["Demo Manager"],
                "BAC": [1000.0],
                "AC": [500.0],
                "Plan Start": ["01/01/2025"],
                "Plan Finish": ["31/10/2025"],
                "Use_Manual_PV": [False],
                "Manual_PV": [0.0],
                "Use_Manual_EV": [True],
                "Manual_EV": [500.0]
            }

            blank_df = pd.DataFrame(blank_data)
            st.session_state.data_df = blank_df
            st.session_state.data_loaded = True
            st.session_state.original_filename = "Manual Entry"
            st.session_state.file_type = 'manual'
            st.session_state.config_dict.update({
                "import_metadata": {
                    "import_date": datetime.now().isoformat(),
                    "source": "manual_entry",
                    "original_filename": "Manual Entry"
                },
                "column_mappings": {
                    'pid_col': 'Project ID',
                    'pname_col': 'Project',
                    'org_col': 'Organization',
                    'pm_col': 'Project Manager',
                    'bac_col': 'BAC',
                    'ac_col': 'AC',
                    'st_col': 'Plan Start',
                    'fn_col': 'Plan Finish'
                },
                "curve_type": "S-curve",
                "data_date": "30/05/2025",
                "currency_symbol": "PKR",
                "currency_postfix": "Million"
            })

            st.success("✅ Demo project created! You can now edit this demo record or add more projects using the Manual Entry section in the sidebar.")
            st.cache_data.clear()
            st.rerun()

    # Use processed data from session state if available
    elif st.session_state.data_df is not None and not st.session_state.data_df.empty:
        df = st.session_state.data_df
        selected_table = DEFAULT_DATASET_TABLE

        with st.expander("📊 Data Preview", expanded=False):
            st.dataframe(df.head(10), width="stretch")
            st.caption(f"{len(df)} rows loaded.")
            
        # Get stored column mappings from session state (for display only)
        stored_mappings = st.session_state.config_dict.get('column_mappings', {})
        column_mapping = stored_mappings  # Don't re-render mapping for processed data

    # Final check: if we have data in session state but haven't set return variables yet
    if (df is None and st.session_state.data_df is not None and
        not st.session_state.data_df.empty):
        df = st.session_state.data_df
        selected_table = DEFAULT_DATASET_TABLE
        stored_mappings = st.session_state.config_dict.get('column_mappings', {})
        column_mapping = stored_mappings

    st.markdown('</div>', unsafe_allow_html=True)
    return df, selected_table, column_mapping

def render_column_mapping_section(columns: List[str], stored_mapping: Dict = None) -> Dict[str, str]:
    """Render column mapping interface."""
    st.markdown("**Column Mapping**")
    
    stored_mapping = stored_mapping or {}
    
    def smart_guess(target: str, keywords: List[str]) -> str:
        # Exact match first
        for col in columns:
            if col.lower().strip() == target.lower():
                return col
        # Contains match
        for keyword in keywords:
            for col in columns:
                if keyword.lower() in col.lower():
                    return col
        return columns[0] if columns else ""

    # Required mappings
    mapping = {}
    mapping['pid_col'] = st.selectbox(
        "🔹 Project ID *",
        columns,
        index=columns.index(stored_mapping.get('pid_col', smart_guess("project id", ["project", "id", "proj"]))) 
        if stored_mapping.get('pid_col') in columns else columns.index(smart_guess("project id", ["project", "id", "proj"])) 
        if smart_guess("project id", ["project", "id", "proj"]) in columns else 0,
        key="map_pid"
    )
    
    mapping['bac_col'] = st.selectbox(
        "🔹 BAC (Budget) *",
        columns,
        index=columns.index(stored_mapping.get('bac_col', smart_guess("bac", ["bac", "budget", "total"])))
        if stored_mapping.get('bac_col') in columns else columns.index(smart_guess("bac", ["bac", "budget", "total"]))
        if smart_guess("bac", ["bac", "budget", "total"]) in columns else 0,
        key="map_bac"
    )
    
    mapping['ac_col'] = st.selectbox(
        "🔹 AC (Actual Cost) *",
        columns,
        index=columns.index(stored_mapping.get('ac_col', smart_guess("ac", ["ac", "actual", "cost", "spent"])))
        if stored_mapping.get('ac_col') in columns else columns.index(smart_guess("ac", ["ac", "actual", "cost", "spent"]))
        if smart_guess("ac", ["ac", "actual", "cost", "spent"]) in columns else 0,
        key="map_ac"
    )
    
    mapping['st_col'] = st.selectbox(
        "🔹 Plan Start *",
        columns,
        index=columns.index(stored_mapping.get('st_col', smart_guess("plan start", ["start", "begin", "plan start"])))
        if stored_mapping.get('st_col') in columns else columns.index(smart_guess("plan start", ["start", "begin", "plan start"]))
        if smart_guess("plan start", ["start", "begin", "plan start"]) in columns else 0,
        key="map_start"
    )
    
    mapping['fn_col'] = st.selectbox(
        "🔹 Plan Finish *",
        columns,
        index=columns.index(stored_mapping.get('fn_col', smart_guess("plan finish", ["finish", "end", "complete", "plan finish"])))
        if stored_mapping.get('fn_col') in columns else columns.index(smart_guess("plan finish", ["finish", "end", "complete", "plan finish"]))
        if smart_guess("plan finish", ["finish", "end", "complete", "plan finish"]) in columns else 0,
        key="map_finish"
    )
    
    # Optional mappings
    st.markdown("_Optional Fields:_")
    none_option = "— None —"
    
    optional_cols = [none_option] + columns
    
    mapping['pname_col'] = st.selectbox(
        "Project",
        optional_cols,
        index=optional_cols.index(stored_mapping.get('pname_col', none_option)) 
        if stored_mapping.get('pname_col') in optional_cols else 0,
        key="map_pname"
    )
    
    mapping['org_col'] = st.selectbox(
        "Organization",
        optional_cols,
        index=optional_cols.index(stored_mapping.get('org_col', none_option))
        if stored_mapping.get('org_col') in optional_cols else 0,
        key="map_org"
    )
    
    mapping['pm_col'] = st.selectbox(
        "Project Manager",
        optional_cols,
        index=optional_cols.index(stored_mapping.get('pm_col', none_option))
        if stored_mapping.get('pm_col') in optional_cols else 0,
        key="map_pm"
    )
    
    mapping['pv_col'] = st.selectbox(
        "Manual PV",
        optional_cols,
        index=optional_cols.index(stored_mapping.get('pv_col', none_option))
        if stored_mapping.get('pv_col') in optional_cols else 0,
        key="map_pv"
    )
    
    mapping['ev_col'] = st.selectbox(
        "Manual EV",
        optional_cols,
        index=optional_cols.index(stored_mapping.get('ev_col', none_option))
        if stored_mapping.get('ev_col') in optional_cols else 0,
        key="map_ev"
    )
    
    # Clean up None values
    for key in ['pname_col', 'org_col', 'pm_col', 'pv_col', 'ev_col']:
        if mapping[key] == none_option:
            mapping[key] = None
    
    return mapping

def render_manual_entry_link():
    """Render B. Manual Data Entry navigation link."""
    st.markdown('<div class="sidebar-section">', unsafe_allow_html=True)
    st.markdown('<div class="section-header">B. Manual Entry</div>', unsafe_allow_html=True)

    has_data = (st.session_state.data_df is not None and not st.session_state.data_df.empty)
    if has_data:
        num_projects = len(st.session_state.data_df)
        st.info(f"📊 {num_projects} project(s) loaded")
    else:
        st.warning("⚠️ No project data loaded")

    st.markdown("### 🚀 Professional Data Management")
    st.markdown("Use our enhanced manual data entry interface with:")
    st.markdown("• **Table view** with sorting and selection")
    st.markdown("• **Advanced editing** with validation")
    st.markdown("• **Bulk operations** and data export")

    if st.button("🎯 **Open Manual Data Entry Page**",
                type="primary",
                width='stretch',
                key="open_manual_entry"):
        st.switch_page("pages/2_Manual_Data_Entry.py")

    st.markdown('</div>', unsafe_allow_html=True)

    
def render_controls_section():
    """Render B. Controls section."""
    st.markdown('<div class="sidebar-section">', unsafe_allow_html=True)
    st.markdown('<div class="section-header">B. Controls</div>', unsafe_allow_html=True)

    # Load saved controls from JSON (if any)
    saved_controls = st.session_state.config_dict.get('controls', {})


    # Curve settings
    curve_value = saved_controls.get('curve_type', 'linear').lower()
    curve_index = 0 if curve_value == 'linear' else 1

    curve_type = st.selectbox(
        "Curve Type (PV)",
        ["Linear", "S-Curve"],
        index=curve_index
    )
    
    if curve_type == "S-Curve":
        col1, col2 = st.columns(2)
        with col1:
            try:
                alpha = st.number_input("S-Curve α", min_value=0.1, max_value=10.0,
                                      value=float(saved_controls.get('alpha', 2.0)), step=0.1, key="s_alpha")
            except Exception as e:
                st.error(f"Error in alpha number_input: {e}")
                alpha = 2.0
        with col2:
            try:
                beta = st.number_input("S-Curve β", min_value=0.1, max_value=10.0,
                                     value=float(saved_controls.get('beta', 2.0)), step=0.1, key="s_beta")
            except Exception as e:
                st.error(f"Error in beta number_input: {e}")
                beta = 2.0
    else:
        alpha, beta = 2.0, 2.0
    
    # Date and financial settings with robust error handling
    saved_date = saved_controls.get('data_date')
    try:
        if saved_date and isinstance(saved_date, str):
            # Try different date formats
            try:
                saved_date = datetime.fromisoformat(saved_date).date()
            except ValueError:
                try:
                    saved_date = datetime.strptime(saved_date, '%Y-%m-%d').date()
                except ValueError:
                    saved_date = date.today()
        elif not isinstance(saved_date, date):
            saved_date = date.today()
    except Exception as e:
        st.error(f"Date parsing error: {e}")
        saved_date = date.today()

    data_date = st.date_input(
        "Data Date",
        value=saved_date,
        min_value=date(2000, 1, 1),
        max_value=date(2035, 12, 31)
    )
    
    
    
    try:
        inflation_rate = st.number_input(
            "Inflation Rate (% APR)",
            min_value=0.0, max_value=100.0,
            value=float(saved_controls.get('inflation_rate', 12.0)), step=0.1,
            key="controls_inflation"
        )
    except Exception as e:
        st.error(f"Error in inflation_rate number_input: {e}")
        inflation_rate = 12.0
    
    # Currency settings
    st.markdown("**Currency Settings**")
    col1, col2 = st.columns(2)
    with col1:
        currency_symbol = st.text_input("Currency Symbol",
                                       value=saved_controls.get('currency_symbol', "PKR"),
                                       max_chars=10)
    with col2:
        # Currency postfix
        postfix_options = ["", "Thousand", "Million", "Billion"]
        saved_postfix = saved_controls.get('currency_postfix', "")

        try:
            postfix_index = postfix_options.index(saved_postfix)
        except ValueError:
            postfix_index = 0

        currency_postfix = st.selectbox(
            "Currency Postfix",
            postfix_options,
            index=postfix_index
        )
    
    st.markdown('</div>', unsafe_allow_html=True)
    
    return {
        'curve_type': curve_type.lower(),
        'alpha': alpha,
        'beta': beta,
        'data_date': data_date,
        'inflation_rate': inflation_rate,
        'currency_symbol': currency_symbol,
        'currency_postfix': currency_postfix
    }

def render_llm_provider_section():
    """Render E. LLM Provider section."""
    st.markdown('<div class="sidebar-section">', unsafe_allow_html=True)
    st.markdown('<div class="section-header">E. LLM Provider (Executive Brief)</div>', unsafe_allow_html=True)
    
    # Load saved configuration
    model_config = load_model_config()
    
    # Determine provider index
    provider_options = ["OpenAI", "Gemini", "Claude", "Kimi"]
    saved_provider = model_config.get("provider", "OpenAI")
    provider_index = provider_options.index(saved_provider) if saved_provider in provider_options else 0

    provider = st.radio(
        "Choose Provider",
        provider_options,
        index=provider_index,
        key="llm_provider"
    )
    
    # Read API key from portfolio settings
    api_key = st.session_state.config_dict.get('llm_config', {}).get('api_key', '') if 'config_dict' in st.session_state else ''
    if api_key:
        st.success("✅ API key configured in Portfolio Settings")
    else:
        st.info("ℹ️ No API key configured. Set it in Portfolio Management > LLM Provider Configuration.")
    
    if provider == "OpenAI":
        # Model selection for OpenAI
        openai_models = ["gpt-4o", "gpt-4o-mini", "gpt-4-turbo", "gpt-3.5-turbo"]
        default_model = model_config.get("model", "gpt-4o-mini") if model_config.get("provider") == "OpenAI" else "gpt-4o-mini"
        if default_model not in openai_models:
            openai_models.append(default_model)

        selected_model = st.selectbox(
            "OpenAI Model",
            openai_models,
            index=openai_models.index(default_model) if default_model in openai_models else 0,
            key="openai_model"
        )

    elif provider == "Gemini":
        # Model selection for Gemini
        gemini_models = ["gemini-1.5-pro", "gemini-1.5-flash", "gemini-pro"]
        default_model = model_config.get("model", "gemini-1.5-flash") if model_config.get("provider") == "Gemini" else "gemini-1.5-flash"
        if default_model not in gemini_models:
            gemini_models.append(default_model)

        selected_model = st.selectbox(
            "Gemini Model",
            gemini_models,
            index=gemini_models.index(default_model) if default_model in gemini_models else 0,
            key="gemini_model"
        )

    elif provider == "Claude":
        # Model selection for Claude
        claude_models = ["claude-3-5-sonnet-20241022", "claude-3-5-haiku-20241022", "claude-3-opus-20240229"]
        default_model = model_config.get("model", "claude-3-5-sonnet-20241022") if model_config.get("provider") == "Claude" else "claude-3-5-sonnet-20241022"
        if default_model not in claude_models:
            claude_models.append(default_model)

        selected_model = st.selectbox(
            "Claude Model",
            claude_models,
            index=claude_models.index(default_model) if default_model in claude_models else 0,
            key="claude_model"
        )

    else:  # Kimi
        # Model selection for Kimi (Moonshot AI)
        kimi_models = ["kimi-k2-0711-preview", "moonshot-v1-128k", "moonshot-v1-8k"]
        default_model = model_config.get("model", "kimi-k2-0711-preview") if model_config.get("provider") == "Kimi" else "kimi-k2-0711-preview"
        if default_model not in kimi_models:
            kimi_models.append(default_model)

        selected_model = st.selectbox(
            "Kimi Model",
            kimi_models,
            index=kimi_models.index(default_model) if default_model in kimi_models else 0,
            key="kimi_model"
        )
    
    # Save model configuration
    save_model_config(provider, selected_model)
    
    # LLM settings
    temperature = st.slider(
        "Temperature",
        min_value=0.0, max_value=1.0, value=0.2, step=0.01,
        key="llm_temperature"
    )
    
    timeout = st.slider(
        "Timeout (seconds)",
        min_value=MIN_TIMEOUT_SECONDS, max_value=MAX_TIMEOUT_SECONDS, value=60, step=5,
        key="llm_timeout"
    )
    
    st.markdown('</div>', unsafe_allow_html=True)
    
    return {
        'provider': provider,
        'model': selected_model,
        'api_key': api_key,
        'temperature': temperature,
        'timeout': timeout
    }

def render_batch_calculation_section():
    """Render C. Batch Calculations section."""
    st.markdown('<div class="sidebar-section">', unsafe_allow_html=True)
    st.markdown('<div class="section-header">C. Batch Calculations</div>', unsafe_allow_html=True)
    
    enable_batch = st.checkbox(
        "Enable Batch Calculation",
        value=False,
        help="Calculate EVM for all projects in the selected dataset",
        key="enable_batch"
    )
    
    if enable_batch:
        st.info("🔄 Batch mode enabled. EVM calculations will be performed for the entire dataset.")
        
        # Batch results management
        if st.button("🔍 Check Previous Results", key="check_batch_results"):
            try:
                # Check if batch results exist
                results_df = load_table(f"{RESULTS_TABLE}_batch")
                st.success(f"Found {len(results_df)} previous batch results")
                if st.button("📥 Download Previous Results"):
                    # Format the results for better display and download
                    formatted_results_df = format_batch_results_for_download(results_df)
                    csv = formatted_results_df.to_csv(index=False)
                    st.download_button(
                        "Download CSV",
                        csv,
                        file_name=f"batch_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                        mime="text/csv"
                    )
            except:
                st.info("No previous batch results found")
    else:
        st.info("💡 Enable batch calculation to process multiple projects simultaneously")
    
    # Application exit button
    st.markdown("---")
    if st.button("🚪 Exit Application", type="secondary", help="Save settings and close application properly"):
        st.info("👋 Goodbye! Your settings have been saved.")
        st.stop()
    
    st.markdown('</div>', unsafe_allow_html=True)
    return enable_batch

def render_help_section():
    """Render G. Help & Information section."""
    st.markdown('<div class="sidebar-section">', unsafe_allow_html=True)
    st.markdown('<div class="section-header">G. Help & Information</div>', unsafe_allow_html=True)
    
    # Collapsible help sections
    with st.expander("📚 Data Source Options"):
        st.markdown("""
        ### 🗃️ Data Source
        - **Session Data**: Load from current session or uploaded files
        - **Load CSV**: Upload new CSV files and map columns to EVM fields
        
        ### ✏️ Manual Entry
        - Add new projects directly to session
        - Edit existing projects
        - Delete projects
        
        ### 📊 Analysis Options
        - Single project EVM analysis
        - Batch processing for multiple projects
        - AI-powered executive reporting
        """)
    
    with st.expander("🎯 Quick Start Guide"):
        st.markdown("""
        **For New Users:**
        1. Session data is automatically initialized for you
        2. Use "Manual Entry" → "Add New Project" to start
        3. Or upload JSON/CSV data using the sidebar
        
        **Key Fields Required:**
        - Project ID (unique identifier)
        - BAC (Budget at Completion)
        - AC (Actual Cost to date)
        - Plan Start & Finish dates
        """)
    
    st.markdown('</div>', unsafe_allow_html=True)

def render_save_download_section():
    """Render F. Save & Download section."""
    st.markdown('<div class="sidebar-section">', unsafe_allow_html=True)
    st.markdown('<div class="section-header">F. Save & Download</div>', unsafe_allow_html=True)
    
    # Check if there's data to save
    has_data = False
    data_sources = []
    
    # Check session state data
    if st.session_state.data_df is not None and not st.session_state.data_df.empty:
        has_data = True
        data_sources.append(f"Session Data ({len(st.session_state.data_df)} rows)")
    
    # Check batch results
    if st.session_state.batch_results is not None and not st.session_state.batch_results.empty:
        has_data = True
        data_sources.append(f"Batch Results ({len(st.session_state.batch_results)} projects)")
    
    # Check config data
    config_items = len(st.session_state.config_dict) if st.session_state.config_dict else 0
    if config_items > 0:
        data_sources.append(f"Configuration ({config_items} settings)")
    
    if has_data or config_items > 0:
        # Show what will be saved
        st.markdown("**[Package] Available for Export:**")
        for source in data_sources:
            st.markdown(f"• {source}")
        
        # Custom filename input for JSON export
        if hasattr(st.session_state, 'original_filename') and st.session_state.original_filename:
            original_name = st.session_state.original_filename
            base_name = original_name.rsplit('.', 1)[0] if '.' in original_name else original_name
            default_filename = f"{base_name}_updated"
        else:
            default_filename = f"portfolio_export_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        
        custom_filename = st.text_input(
            "[File] JSON Filename (without .json extension):",
            value=default_filename,
            help="Enter a custom filename for your JSON export",
            key="json_filename_input"
        )
        
        # Main save button
        if st.button("Save JSON", 
                    type="primary", 
                    help="Save data and configuration as JSON file",
                    width="stretch"):
            if not custom_filename.strip():
                st.error("X Please enter a filename")
            else:
                try:
                    with st.spinner("Creating JSON file..."):
                        # Determine what data to include
                        if st.session_state.data_df is not None and not st.session_state.data_df.empty:
                            export_df = st.session_state.data_df
                        elif st.session_state.batch_results is not None:
                            export_df = st.session_state.batch_results
                        else:
                            # Create empty DataFrame with schema
                            export_df = pd.DataFrame({
                                "Project ID": [],
                                "Project": [],
                                "Organization": [],
                                "Project Manager": [],
                                "BAC": [],
                                "AC": [],
                                "Plan Start": [],
                                "Plan Finish": [],
                                "Use_Manual_PV": [],
                                "Manual_PV": []
                            })
                        
                        # Define required and optional fields for export
                        # Required fields (must be present)
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

                        # Optional fields (include if present, add with None if missing)
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

                        # Combine all allowed fields
                        all_allowed_fields = {**required_fields, **optional_fields}

                        # Convert DataFrame to list of dictionaries with whitelist filter
                        if not export_df.empty:
                            data_records = []
                            for record in export_df.to_dict('records'):
                                # Create mapping of lowercase column names to actual values
                                record_map = {k.lower(): (k, v) for k, v in record.items()}

                                # Build clean record with standardized names
                                clean_record = {}

                                # Add required fields
                                for field_lower, standard_name in required_fields.items():
                                    if field_lower in record_map:
                                        clean_record[standard_name] = record_map[field_lower][1]
                                    # If required field missing, still add with None
                                    else:
                                        clean_record[standard_name] = None

                                # Add optional fields (only if present)
                                for field_lower, standard_name in optional_fields.items():
                                    if field_lower in record_map:
                                        clean_record[standard_name] = record_map[field_lower][1]
                                    else:
                                        # Add optional field with appropriate default
                                        if 'use_manual' in field_lower:
                                            clean_record[standard_name] = False
                                        elif 'manual_' in field_lower:
                                            clean_record[standard_name] = 0.0
                                        else:
                                            clean_record[standard_name] = None

                                data_records.append(clean_record)
                        else:
                            data_records = []

                        # Clean config - use whitelist approach to only export essential config
                        import copy

                        # Whitelist: Only export these config sections
                        export_config = {}

                        # Export controls (EVM settings)
                        if 'controls' in st.session_state.config_dict:
                            export_config['controls'] = copy.deepcopy(st.session_state.config_dict['controls'])

                        # Export LLM config (but remove API keys)
                        if 'llm_config' in st.session_state.config_dict:
                            export_config['llm_config'] = copy.deepcopy(st.session_state.config_dict['llm_config'])
                            # Remove API keys
                            if 'api_key' in export_config['llm_config']:
                                export_config['llm_config']['api_key'] = ""
                                export_config['llm_config']['has_api_key'] = False

                        # Also check for LLM config nested in controls
                        if 'controls' in export_config and 'llm_config' in export_config['controls']:
                            if 'api_key' in export_config['controls']['llm_config']:
                                export_config['controls']['llm_config']['api_key'] = ""
                                export_config['controls']['llm_config']['has_api_key'] = False

                        # NOTE: Explicitly NOT exporting:
                        # - 'tables' (calculated batch results)
                        # - 'export_metadata' (transient)
                        # - 'batch_setting_from_json' (transient)
                        # - 'column_mappings' (import-specific)
                        # - 'import_metadata' (import-specific)
                        # - any other transient state

                        # Add export metadata and data
                        export_config.update({
                            "export_metadata": {
                                "export_date": datetime.now().isoformat(),
                                "app_version": "Project Portfolio Intelligence Suite",
                                "data_rows": len(data_records),
                                "exported_fields": len(all_allowed_fields),
                                "export_note": "Standardized export with all required and optional input fields; EVM calculations regenerated on import"
                            },
                            "data": data_records
                        })
                        
                        # Create JSON content
                        json_content = json.dumps(export_config, indent=2, default=str)
                        
                        # Clean filename and add .json extension
                        clean_filename = custom_filename.strip().replace('.json', '') + '.json'
                        
                        # Provide download button
                        st.download_button(
                            label="Download JSON File",
                            data=json_content,
                            file_name=clean_filename,
                            mime="application/json",
                            width="stretch",
                            help=f"Save as: {clean_filename}"
                        )
                        
                        st.success(f"✓ JSON file ready for download: {clean_filename}")
                        st.info("[Info] You can choose where to save this file using your browser's download settings")
                    
                except Exception as e:
                    st.error(f"X Export failed: {e}")
                    logger.error(f"Export error: {e}")
        
        # Exit button
        if st.button("Exit", 
                    type="secondary", 
                    help="Clear session and restart application",
                    width="stretch"):
            # Clear all session state
            for key in list(st.session_state.keys()):
                del st.session_state[key]
            
            st.success("[Wave] Session cleared! Restarting application...")
            st.rerun()
        
    else:
        st.info("[Info] **No data to export**\n\nAdd projects via Manual Entry or upload JSON/CSV files to enable downloads.")
        
        # Quick action buttons when no data
        st.markdown("**[Action] Quick Actions:**")
        if st.button("[File] Load Demo Data", 
                    width="stretch",
                    help="Load sample data for testing"):
            try:
                # Create demo data
                demo_data = {
                    "Project ID": ["DEMO001", "DEMO002", "DEMO003"],
                    "Project": ["Website Redesign", "Mobile App", "Data Migration"],
                    "Organization": ["IT", "Product", "Operations"],
                    "Project Manager": ["Alice Smith", "Bob Jones", "Carol Davis"],
                    "BAC": [150000, 250000, 100000],
                    "AC": [120000, 180000, 85000],
                    "Plan Start": ["2024-01-01", "2024-02-15", "2024-03-01"],
                    "Plan Finish": ["2024-06-30", "2024-08-15", "2024-05-30"],
                    "Use_Manual_PV": [False, False, True],
                    "Manual_PV": [None, None, 75000]
                }
                st.session_state.data_df = pd.DataFrame(demo_data)
                st.session_state.data_loaded = True
                st.session_state.original_filename = "demo_data"
                st.session_state.file_type = "demo"
                st.success("✓ Demo data loaded!")
                st.rerun()
            except Exception as e:
                st.error(f"Failed to load demo data: {e}")
    
    st.markdown('</div>', unsafe_allow_html=True)
    
# =============================================================================
# ENHANCED RESULTS DISPLAY
# =============================================================================

def render_enhanced_inputs_tab(project_data: Dict, results: Dict, controls: Dict):
    """Render enhanced inputs tab with better organization."""
    st.markdown("### 📋 Project Information")
    
    # Project Details - Organized in multiple rows
    st.markdown("#### Project Details")
    col1, col2 = st.columns(2)
    with col1:
        st.text_input("Project ID", value=project_data.get('project_id', ''), disabled=True)
        st.text_input("Project", value=project_data.get('project_name', ''), disabled=True)
    with col2:
        st.text_input("Organization", value=project_data.get('organization', ''), disabled=True)
        st.text_input("Project Manager", value=project_data.get('project_manager', ''), disabled=True)
    
    # Financial Summary - Optimized for large numbers
    st.markdown("#### Financial Summary")
    
    # Enhanced styling for Financial Summary with larger fonts and better spacing
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
    
    # Row 1: Budget (BAC) and Actual Cost (AC)
    col1, col2 = st.columns(2)
    with col1:
        bac_formatted = format_currency(results['bac'], controls['currency_symbol'], controls['currency_postfix'])
        st.markdown(f'<div class="financial-metric">💰 **Budget (BAC)**<br><span class="financial-value">{bac_formatted}</span></div>', unsafe_allow_html=True)
    with col2:
        ac_formatted = format_currency(results['ac'], controls['currency_symbol'], controls['currency_postfix'])
        st.markdown(f'<div class="financial-metric">💸 **Actual Cost (AC)**<br><span class="financial-value">{ac_formatted}</span></div>', unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)  # Add spacing

    # Row 2: Planned Value (PV) and Earned Value (EV)
    col3, col4 = st.columns(2)
    with col3:
        pv_formatted = format_currency(results['pv'], controls['currency_symbol'], controls['currency_postfix'])
        use_manual_pv = results.get('use_manual_pv', False)
        pv_label = f"📊 **Planned Value (PV{'**' if not use_manual_pv else ''})**"
        st.markdown(f'<div class="financial-metric">{pv_label}<br><span class="financial-value">{pv_formatted}</span></div>', unsafe_allow_html=True)
    with col4:
        ev_formatted = format_currency(results['ev'], controls['currency_symbol'], controls['currency_postfix'])
        use_manual_ev = results.get('use_manual_ev', False)
        ev_label = f"💎 **Earned Value (EV{'**' if not use_manual_ev else ''})**"
        st.markdown(f'<div class="financial-metric">{ev_label}<br><span class="financial-value">{ev_formatted}</span></div>', unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)  # Add spacing

    # Row 3: Present Value of Project (PrV) and % Present Value of Project
    col5, col6 = st.columns(2)
    with col5:
        prv_formatted = format_currency(results['planned_value_project'], controls['currency_symbol'], controls['currency_postfix'])
        st.markdown(f'<div class="financial-metric">🏗️ **Present Value of Project (PrV)**<br><span class="financial-value">{prv_formatted}</span></div>', unsafe_allow_html=True)
    with col6:
        percent_prv = results.get('percent_present_value_project', 0.0)
        st.markdown(f'<div class="financial-metric">📈 **% Present Value of Project**<br><span class="financial-value">{percent_prv:.2f}%</span></div>', unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)  # Add spacing

    # Row 4: Likely Value of Project (LkV) and % Likely Value of Project
    col7, col8 = st.columns(2)
    with col7:
        lkv_formatted = format_currency(results['likely_value_project'], controls['currency_symbol'], controls['currency_postfix'])
        st.markdown(f'<div class="financial-metric">🔮 **Likely Value of Project (LkV)**<br><span class="financial-value">{lkv_formatted}</span></div>', unsafe_allow_html=True)
    with col8:
        percent_lkv = results.get('percent_likely_value_project', 0.0)
        st.markdown(f'<div class="financial-metric">🎯 **% Likely Value of Project**<br><span class="financial-value">{percent_lkv:.2f}%</span></div>', unsafe_allow_html=True)
    
    # Performance Indicators - Multi-row layout
    st.markdown("#### Performance Indicators")
    
    # Row 1: CPI and SPI
    col1, col2 = st.columns(2)
    with col1:
        cpi = results['cpi']
        delta_cpi = f"{cpi - 1:.2f}" if is_valid_finite_number(cpi) else "N/A"
        status_cpi = "normal" if cpi >= 1.0 else "inverse"
        st.metric("Cost Performance Index (CPI)", format_performance_index(cpi), 
                 delta=delta_cpi, delta_color=status_cpi)
    
    with col2:
        spi = results['spi']
        delta_spi = f"{spi - 1:.2f}" if is_valid_finite_number(spi) else "N/A"
        status_spi = "normal" if spi >= 1.0 else "inverse"
        st.metric("Schedule Performance Index (SPI)", format_performance_index(spi),
                 delta=delta_spi, delta_color=status_spi)
    
    # Row 2: Progress metrics
    col1, col2, col3 = st.columns(3)
    with col1:
        budget_used = safe_divide(results['ac'], results['bac']) * 100
        st.metric("% Budget Used", f"{budget_used:.1f}%")
    
    with col2:
        st.metric("% Physical Progress", format_percentage(results['percent_complete']))
    
    with col3:
        time_used = safe_divide(results['actual_duration_months'], results['original_duration_months']) * 100
        st.metric("% Time Used", f"{time_used:.1f}%")
    
    # Project Timeline
    st.markdown("#### Project Timeline")
    col1, col2, col3 = st.columns(3)
    with col1:
        st.text_input("Plan Start", value=results['plan_start'], disabled=True)
    with col2:
        st.text_input("Plan Finish", value=results['plan_finish'], disabled=True)
    with col3:
        st.text_input("Data Date", value=results['data_date'], disabled=True)

def build_enhanced_results_table(results: dict, controls: dict, project_data: dict) -> pd.DataFrame:
    """Build enhanced results table with better formatting."""
    currency = controls['currency_symbol']
    postfix = controls['currency_postfix']
    
    # Check if manual EV and PV are used
    use_manual_ev = results.get('use_manual_ev', False)
    use_manual_pv = results.get('use_manual_pv', False)
    
    # Helper function for currency formatting
    def fmt_curr(amount):
        if amount is None or not is_valid_finite_number(amount):
            return "—"
        return format_currency(amount, currency, postfix)
    
    # Calculate derived metrics
    bac = results.get('bac', 0)
    ac = results.get('ac', 0)
    pv = results.get('pv', 0)
    ev = results.get('ev', 0)
    cpi = results.get('cpi', 0)
    spi = results.get('spi', 0)
    spie = results.get('spie', 0)
    tcpi = results.get('tcpi', 0)
    cv = results.get('cv', 0)
    sv = results.get('sv', 0)
    
    # Status indicators
    def get_status_color(value, good_threshold=1.0, inverse=False):
        """Get status color indicator.

        Args:
            value: The metric value
            good_threshold: Threshold for good performance (default 1.0)
            inverse: If True, lower values are better (for TCPI). Default False.
        """
        if not is_valid_finite_number(value):
            return "⚪"

        if inverse:
            # For TCPI: lower is better
            return "🟢" if value <= 1.0 else "🟡" if value <= 1.1 else "🔴"
        else:
            # For CPI, SPI, SPIe: higher is better
            return "🟢" if value >= good_threshold else "🟡" if value >= 0.9 else "🔴"
    
    # Build comprehensive results table
    rows = [
        # Project Information
        ("📋 PROJECT INFORMATION", "", "", ""),
        ("Project ID", "", project_data.get('project_id', ''), ""),
        ("Project", "", project_data.get('project_name', ''), ""),
        ("Organization", "", project_data.get('organization', ''), ""),
        ("Project Manager", "", project_data.get('project_manager', ''), ""),
        ("Plan Start", "", format_date_dmy(results.get('plan_start', 'N/A')), ""),
        ("Plan Finish", "", format_date_dmy(results.get('plan_finish', 'N/A')), ""),
        ("% Budget Used", "AC ÷ BAC × 100", format_percentage(results.get('percent_budget_used', 0)), ""),
        ("% Time Used", "AT ÷ OD × 100", format_percentage(results.get('percent_time_used', 0)), ""),
        ("Present Value", "Discounted actual cost", fmt_curr(results.get('present_value', 0)), ""),
        ("", "", "", ""),  # Spacer
        
        # Financial Overview
        ("💰 FINANCIAL OVERVIEW", "", "", ""),
        ("Budget at Completion (BAC)", "Total planned budget", fmt_curr(bac), ""),
        ("Actual Cost (AC)", "Total spent to date", fmt_curr(ac), ""),
        (f"Planned Value (PV{'**' if not use_manual_pv else ''})", "Value of work planned", fmt_curr(pv), ""),
        (f"Earned Value (EV{'**' if not use_manual_ev else ''})", "Value of work completed", fmt_curr(ev), ""),
        ("", "", "", ""),  # Spacer

        # Advanced Financial Analysis
        ("🏗️ ADVANCED FINANCIAL ANALYSIS", "", "", ""),
        # ("Present Value of Progress", "(AC/AD) × PV Factor", fmt_curr(results.get('present_value_progress', 0)), "Discounted value of work progress"), # Removed duplicate
        ("Planned Value of Project", "(BAC/OD) × PV Factor", fmt_curr(results.get('planned_value_project', 0)), "Total project value at planned pace"),
        ("Likely Value of Project", "(BAC/LD) × PV Factor", fmt_curr(results.get('likely_value_project', 0)), "Total project value at forecast pace"),
        ("% Present Value of Project", "PrV ÷ BAC × 100", f"{results.get('percent_present_value_project', 0):.2f}%", "Planned value efficiency"),
        ("% Likely Value of Project", "LkV ÷ BAC × 100", f"{results.get('percent_likely_value_project', 0):.2f}%", "Forecast value efficiency"),
        ("", "", "", ""),  # Spacer
        
        # Performance Metrics
        ("📊 PERFORMANCE METRICS", "", "", ""),
        ("Cost Performance Index (CPI)", "EV ÷ AC", f"{get_status_color(cpi, 1.0)} {format_performance_index(cpi)}" if is_valid_finite_number(cpi) else "N/A", 
         "Excellent" if cpi > 1.1 else "Good" if cpi > 1.0 else "Poor" if cpi < 0.9 else "Fair"),
        ("Schedule Performance Index (SPI)", "EV ÷ PV", f"{get_status_color(spi, 1.0)} {format_performance_index(spi)}" if is_valid_finite_number(spi) else "N/A",
         "Excellent" if spi > 1.1 else "Good" if spi > 1.0 else "Poor" if spi < 0.9 else "Fair"),
        ("Schedule Performance Index (SPIe)", "ES ÷ AT", f"{get_status_color(spie, 1.0)} {format_performance_index(spie)}" if is_valid_finite_number(spie) else "N/A",
         "Excellent" if spie > 1.1 else "Good" if spie > 1.0 else "Poor" if spie < 0.9 else "Fair"),
        ("To Complete Performance Index (TCPI)", "(BAC - EV) ÷ (BAC - AC)", f"{get_status_color(tcpi, 1.0, inverse=True)} {format_performance_index(tcpi)}" if is_valid_finite_number(tcpi) else "N/A",
         "Excellent" if tcpi < 0.9 else "Good" if tcpi <= 1.0 else "Challenging" if tcpi <= 1.1 else "Critical"),
        ("", "", "", ""),  # Spacer
        
        # Variances
        ("📈 VARIANCES", "", "", ""),
        ("Cost Variance (CV)", "EV - AC", fmt_curr(cv), "Under budget" if cv > 0 else "Over budget" if cv < 0 else "On budget"),
        ("Schedule Variance (SV)", "EV - PV", fmt_curr(sv), "Ahead of schedule" if sv > 0 else "Behind schedule" if sv < 0 else "On schedule"),
        ("", "", "", ""),  # Spacer
        
        # Forecasts
        ("🔮 FORECASTS", "", "", ""),
        ("Estimate at Completion (EAC)", "BAC ÷ CPI", fmt_curr(results.get('eac')) if is_valid_finite_number(results.get('eac')) else "Cannot determine", ""),
        ("Variance at Completion (VAC)", "BAC - EAC", fmt_curr(results.get('vac')) if is_valid_finite_number(results.get('vac')) else "Cannot determine", ""),
        ("Estimate to Complete (ETC)", "EAC - AC", fmt_curr(results.get('etc')) if is_valid_finite_number(results.get('etc')) else "Cannot determine", ""),
        ("", "", "", ""),  # Spacer
        
        # Progress Analysis
        ("⏱️ PROGRESS ANALYSIS", "", "", ""),
        ("Physical Progress", "EV ÷ BAC × 100", format_percentage(results['percent_complete']), ""),
        ("Budget Utilization", "AC ÷ BAC × 100", format_percentage(safe_divide(ac, bac) * 100), ""),
        ("Time Utilization", "AT ÷ OD × 100", format_percentage(safe_divide(results['actual_duration_months'], results['original_duration_months']) * 100), ""),
        ("", "", "", ""),  # Spacer
        
        # Schedule Analysis
        ("📅 SCHEDULE ANALYSIS", "", "", ""),
        ("Original Duration", "Planned months", format_duration(results['original_duration_months']), ""),
        ("Elapsed Duration", "Months to date", format_duration(results['actual_duration_months']), ""),
        ("Earned Schedule", "Time where PV = EV", format_duration(results['es']), ""),
        ("Expected Duration", "OD ÷ SPIe", format_duration(results.get('forecast_duration', 0)) if results.get('forecast_duration') else "Cannot determine", ""),
        ("Expected Finish Date", "Based on current performance", format_date_dmy(results.get('forecast_completion', 'N/A')), ""),
    ]
    
    # Add currency note if postfix is used
    if postfix:
        note = f"Note: User figures entered in {postfix.lower()}s (as indicated by {postfix.upper()[0]} postfix)"
        rows.insert(-1, ("", "", note, ""))
    
    return pd.DataFrame(rows, columns=["Category", "Description", "Value", "Status"])

# =============================================================================
# MAIN APPLICATION
# =============================================================================

def main():
    """Main application with enhanced UX."""
    st.markdown('<h1 class="main-header">🎯 Single Project Analysis</h1>', unsafe_allow_html=True)
    st.markdown("Detailed analysis of individual projects")

    # Portfolio & Period Selection
    st.markdown("---")
    portfolio_id, status_date = render_portfolio_context(show_period_selector=True, show_progress_filter=True)
    st.markdown("---")

    if not portfolio_id:
        st.warning("⚠️ Please select a portfolio to continue")
        st.info("Go to **Portfolio Management** to create or select a portfolio")
        st.stop()

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

    try:
        # Load data from database in database mode
        if USE_DATABASE and portfolio_id and status_date:
            adapter = data_manager.get_data_adapter()
            projects_df = adapter.get_projects_for_analysis(portfolio_id=portfolio_id, status_date=status_date)

            # Store in session state for compatibility with existing code
            if not projects_df.empty:
                st.session_state.data_df = projects_df
                st.session_state.data_loaded = True
                st.session_state.batch_results_ready = True
                st.session_state.batch_results = projects_df  # Use same data as batch results
            else:
                st.session_state.data_df = None
                st.session_state.data_loaded = False

        # Initialize session state for first run
        if "session_initialized" not in st.session_state:
            create_demo_data()
            st.session_state.batch_results = None
            st.session_state.data_df = None
            st.session_state.config_dict = {}
            st.session_state.data_loaded = False
            st.session_state.original_filename = None
            st.session_state.file_type = None
            st.session_state.processed_file_info = None
            st.session_state.session_initialized = True

        # Load configuration from portfolio settings in database mode
        if USE_DATABASE and portfolio_id:
            from utils.portfolio_settings import load_portfolio_settings
            portfolio_settings = load_portfolio_settings(portfolio_id)
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
        else:
            # Load configuration from session state (set in File Management)
            controls = st.session_state.config_dict.get('controls', {
                'curve_type': 'linear',
                'alpha': 2.0,
                'beta': 2.0,
                'currency_symbol': '$',
                'currency_postfix': '',
                'date_format': 'YYYY-MM-DD'
            })

        # Default column mapping for single project analysis
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

        # Check if batch results are available
        if st.session_state.get('batch_results_ready'):
            st.success("✅ Batch calculations completed")
            st.info("💡 Visit Portfolio Analysis for portfolio-level insights")

        # Only render Help section
        render_help_section()
        
        
        # Main content area - prioritize session state data
        session_has_data = (st.session_state.data_df is not None and 
                           not st.session_state.data_df.empty)
        
        if not session_has_data:
            st.info("🚀 **Ready to start!** Use **File Management** to import data and configure settings, then return here for single project analysis.")

            col1, col2, col3 = st.columns(3)
            with col1:
                st.markdown("""
                ### 🎯 Quick Start Workflow
                1. **File Management**: Import data & configure settings
                2. **Project Analysis**: Individual project EVM analysis
                3. **Portfolio Analysis**: Multi-project strategic view
                """)

                if st.button("📁 **Go to File Management**", key="go_to_file_mgmt", type="primary"):
                    st.switch_page("pages/1_File_Management.py")

            with col2:
                st.markdown("""
                ### ✏️ Manual Entry
                - Add new projects directly
                - Edit existing projects
                - Delete projects
                - No files required!
                """)

            with col3:
                st.markdown("""
                ### 🔧 Controls & Help
                - Configure curve types & inflation
                - Customize currency display
                - Check Help section for guidance
                """)
            
            return

        # Use adapter to get data for display (routes to database OR session state)
        adapter = data_manager.get_data_adapter()

        if USE_DATABASE and st.session_state.get('current_portfolio_id'):
            # Database mode: Get projects from database
            portfolio_id = st.session_state.get('current_portfolio_id')
            status_date = st.session_state.get('current_status_date')
            display_df = adapter.get_projects_for_analysis(
                portfolio_id=portfolio_id,
                status_date=status_date
            )
            # Also ensure session state has the data for compatibility
            if not display_df.empty:
                st.session_state.data_df = display_df
        else:
            # Session state mode: Use existing data
            display_df = st.session_state.data_df

        # Apply progress filter if enabled (toggle is in portfolio context)
        if not display_df.empty:
            from utils.portfolio_context import apply_progress_filter
            ac_col = column_mapping.get('ac_col', 'AC')
            display_df, filter_message = apply_progress_filter(display_df, ac_col)

            if filter_message:
                st.caption(filter_message)

            if display_df.empty:
                st.warning("⚠️ No projects with Actual Cost > 0 found for this period.")
                st.info("💡 Turn off the filter to see all projects, or upload data with actual costs.")
                return

        # Single project analysis mode

        # For single project analysis, ensure column mapping is available
        # Use existing column_mapping from sidebar, or provide fallback for demo/csv data
        if st.session_state.get('file_type') in ['demo', 'csv']:
            # Override with standard mapping for demo/csv since data is already normalized
            column_mapping = {
                'pid_col': 'Project ID',
                'pname_col': 'Project',
                'org_col': 'Organization',
                'pm_col': 'Project Manager',
                'bac_col': 'BAC',
                'ac_col': 'AC',
                'st_col': 'Plan Start',
                'fn_col': 'Plan Finish'
            }
        else:
            # Validate that column mapping exists and has required fields
            if not column_mapping or not all(column_mapping.get(key) for key in ['pid_col', 'bac_col', 'ac_col', 'st_col', 'fn_col']):
                st.warning("⚠️ Please complete the column mapping in the sidebar first.")
                return

        try:
            pid_col = column_mapping['pid_col']
            pname_col = column_mapping.get('pname_col')

            # Debug information
            if pid_col not in display_df.columns:
                st.error(f"Column '{pid_col}' not found in data. Available columns: {list(display_df.columns)}")
                return

            project_ids = display_df[pid_col].astype(str).tolist()
            project_names = display_df[pname_col].astype(str).fillna("").tolist() if pname_col and pname_col in display_df.columns else [""] * len(project_ids)
            display_options = [f"{pid} — {pname}" if pname and pname != "nan" else pid for pid, pname in zip(project_ids, project_names)]

            selected_idx = st.selectbox("Select Project for Analysis", range(len(display_options)), format_func=lambda i: display_options[i])
            selected_project_id = project_ids[selected_idx]

        except Exception as e:
            st.error(f"Error processing project list: {e}")
            st.error(f"Available columns: {list(display_df.columns) if 'display_df' in locals() else 'DataFrame not available'}")
            st.error(f"Column mapping: {column_mapping}")
            return
        
        project_row = display_df[display_df[pid_col].astype(str) == selected_project_id].iloc[0]
        
        try:
            project_data = {
                'project_id': selected_project_id,
                'project_name': str(project_row.get(column_mapping.get('pname_col'), "")),
                'organization': str(project_row.get(column_mapping.get('org_col'), "")),
                'project_manager': str(project_row.get(column_mapping.get('pm_col'), "")),
                'bac': float(project_row[column_mapping['bac_col']]),
                'ac': float(project_row[column_mapping['ac_col']]),
                'plan_start': project_row[column_mapping['st_col']],
                'plan_finish': project_row[column_mapping['fn_col']]
            }
        except Exception as e:
            st.error(f"Error extracting project data: {e}")
            return
        
        try:
            # Check if we have batch results for this project
            batch_results_available = False

            # Retrieve stored EVM results
            if USE_DATABASE and portfolio_id and status_date:
                # Database mode: Get EVM results from database
                try:
                    batch_results_df = db_manager.get_evm_results_for_period(portfolio_id, status_date)
                except Exception as e:
                    logging.error(f"Error retrieving EVM results: {e}")
                    batch_results_df = pd.DataFrame()
            else:
                # Session state mode: Use adapter
                batch_results_df = adapter.get_batch_results()

            if (batch_results_df is not None and
                not batch_results_df.empty and
                'project_id' in batch_results_df.columns):

                # Try to find this project in batch results
                batch_project = batch_results_df[
                    batch_results_df['project_id'].astype(str) == selected_project_id
                ]

                if not batch_project.empty:
                    batch_results_available = True
                    results = batch_project.iloc[0].to_dict()

                    # Normalize field names for backward compatibility
                    # Batch results from evm_engine have 'likely_completion', but we need 'forecast_completion'
                    if 'likely_completion' in results and 'forecast_completion' not in results:
                        results['forecast_completion'] = results['likely_completion']
                    if 'ld' in results and 'forecast_duration' not in results:
                        results['forecast_duration'] = results['ld']

            # If no batch results, calculate fresh
            if not batch_results_available:
                with st.spinner("Calculating EVM metrics..."):
                    use_manual_pv = bool(project_row.get('Use_Manual_PV', False))
                    manual_pv_val = project_row.get('Manual_PV')
                    use_manual_ev = bool(project_row.get('Use_Manual_EV', False))
                    manual_ev_val = project_row.get('Manual_EV')

                    # Use project-specific EVM settings if available, otherwise use global controls
                    project_curve_type = project_row.get('Curve Type')
                    project_alpha = project_row.get('Alpha')
                    project_beta = project_row.get('Beta')
                    project_inflation = project_row.get('Inflation Rate')

                    curve_type_for_calc = project_curve_type if (project_curve_type is not None and project_curve_type != "") else controls['curve_type']
                    alpha_for_calc = project_alpha if (project_alpha is not None and project_alpha != 0.0) else controls['alpha']
                    beta_for_calc = project_beta if (project_beta is not None and project_beta != 0.0) else controls['beta']
                    inflation_for_calc = (project_inflation / 100.0) if (project_inflation is not None and project_inflation != 0.0) else (controls['inflation_rate'] / 100.0)

                    results = perform_complete_evm_analysis_wrapper(
                        bac=project_data['bac'], ac=project_data['ac'],
                        plan_start=project_data['plan_start'], plan_finish=project_data['plan_finish'],
                        data_date=controls['data_date'], annual_inflation_rate=inflation_for_calc,
                        curve_type=curve_type_for_calc, alpha=alpha_for_calc, beta=beta_for_calc,
                        manual_pv=manual_pv_val, use_manual_pv=use_manual_pv,
                        manual_ev=manual_ev_val, use_manual_ev=use_manual_ev
                    )

            # Display key metrics: SPI, SPIe, CPI, TCPI
            col1, col2, col3, col4 = st.columns(4)
            with col1:
                spi = results['spi']
                spi_status = "🟢" if spi >= 1.0 else "🟡" if spi >= 0.9 else "🔴"
                st.metric("SPI (Schedule)", f"{spi_status} {format_performance_index(spi)}", f"{spi - 1:.2f}")
            with col2:
                spie = results.get('spie', 0.0)
                spie_status = "🟢" if spie >= 1.0 else "🟡" if spie >= 0.9 else "🔴"
                st.metric("SPIe (Earned Schedule)", f"{spie_status} {format_performance_index(spie)}", f"{spie - 1:.2f}")
            with col3:
                cpi = results['cpi']
                cpi_status = "🟢" if cpi >= 1.0 else "🟡" if cpi >= 0.9 else "🔴"
                st.metric("CPI (Cost)", f"{cpi_status} {format_performance_index(cpi)}", f"{cpi - 1:.2f}")
            with col4:
                tcpi = results.get('tcpi', 0.0)
                tcpi_status = "🟢" if tcpi <= 1.0 else "🟡" if tcpi <= 1.1 else "🔴"
                st.metric("TCPI (To Complete)", f"{tcpi_status} {format_performance_index(tcpi)}", f"{tcpi - 1:.2f}")
            
            tab1, tab2, tab3, tab4, tab5 = st.tabs(["📋 Inputs", "📊 Results", "🤖 Executive Brief", "📈 Charts", "📉 Trend"])
            
            with tab1:
                render_enhanced_inputs_tab(project_data, results, controls)
            
            with tab2:
                st.markdown("### 📊 Detailed EVM Analysis")
                results_table = build_enhanced_results_table(results, controls, project_data)
                st.dataframe(results_table, width="stretch", height=600)
                
                csv_buffer = io.StringIO()
                results_table.to_csv(csv_buffer, index=False)
                st.download_button("📥 Download Results (CSV)", csv_buffer.getvalue(), file_name=f"evm_analysis_{selected_project_id}.csv")
            
            with tab3:
                st.markdown("### 🤖 Executive Brief")

                # Add toggle for single vs. all projects status display
                col_toggle, col_spacer = st.columns([1, 3])
                with col_toggle:
                    brief_mode = st.radio(
                        "Report Scope",
                        options=["Single", "All"],
                        index=0,  # Default to "Single"
                        horizontal=True,
                        help="Single: Current project only | All: All projects with status history"
                    )

                project_name = project_data['project_name'] or selected_project_id
                combined_text = f"{project_name} {project_data['organization']}".lower()
                
                project_context = ""
                if any(w in combined_text for w in ['bridge', 'construction', 'road']): project_context = "This is likely a construction project."
                elif any(w in combined_text for w in ['software', 'it', 'system', 'app']): project_context = "This is likely a technology project."

                # Build prompt based on mode
                if brief_mode == "Single":
                    prompt = textwrap.dedent(f"""
                Create an executive-level project status report for the following Earned Value Management analysis.
                Structure for C-suite and senior stakeholders who need quick decision-making information.

                **IMPORTANT CONTEXT:**
                - This project is currently in its execution stage (actively being worked on)
                - AD (Actual Duration) represents the time already spent on this project, NOT the total project duration
                - The project is ongoing and not yet completed

                PROJECT: {project_name}
                ORGANIZATION: {project_data.get('organization', '')}
                PROJECT MANAGER: {project_data.get('project_manager', '')}
                ANALYSIS DATE: {results['data_date']}
                CURRENCY: {controls['currency_symbol']} {controls.get('currency_postfix', '')}

                === PROJECT DATA ===
                FINANCIALS:
                - Budget (BAC): {format_currency(results['bac'], controls['currency_symbol'], controls['currency_postfix'])}
                - Actual Cost (AC): {format_currency(results['ac'], controls['currency_symbol'], controls['currency_postfix'])}
                - Earned Value (EV{'**' if not results.get('use_manual_ev', False) else ''}): {format_currency(results['ev'], controls['currency_symbol'], controls['currency_postfix'])}
                - Planned Value (PV{'**' if not results.get('use_manual_pv', False) else ''}): {format_currency(results['pv'], controls['currency_symbol'], controls['currency_postfix'])}

                SCHEDULE:
                - Plan Start: {results['plan_start']}
                - Plan Finish: {results['plan_finish']}
                - Expected Finish: {results['forecast_completion']}
                - Physical Completion: {format_percentage(results['percent_complete'])}

                PERFORMANCE INDICES:
                - Cost Performance (CPI): {format_performance_index(results['cpi'])}
                - Schedule Performance (SPI): {format_performance_index(results['spi'])}
                - Budget Utilization: {format_percentage(safe_divide(results['ac'], results['bac']) * 100)}
                - Time Utilization: {format_percentage(safe_divide(results['actual_duration_months'], results['original_duration_months']) * 100)}

                FORECASTS:
                - Estimate at Completion (EAC): {format_currency(results['eac'], controls['currency_symbol'], controls['currency_postfix']) if is_valid_finite_number(results['eac']) else 'Cannot determine'}
                - Variance at Completion (VAC): {format_currency(results['vac'], controls['currency_symbol'], controls['currency_postfix']) if is_valid_finite_number(results['vac']) else 'Cannot determine'}

                STRUCTURE YOUR RESPONSE AS FOLLOWS:

                # PROJECT STATUS REPORT

                **Project:** {project_name}  
                **Organization:** {project_data.get('organization', 'N/A')}  
                **Project Manager:** {project_data.get('project_manager', 'N/A')}  
                **Report Date:** {results['data_date']}  
                **Currency:** {controls['currency_symbol']} {controls.get('currency_postfix', '')}

                ---

                ### PROJECT DASHBOARD
                Create a formatted table with these key metrics:
                | Metric | Value | Status |
                |--------|--------|---------|
                | Data Date | [date] | |
                | Original Budget | [amount] | |
                | Budget Used % | [percentage] | [Good/Warning/Critical] |
                | Time Elapsed % | [percentage] | [Good/Warning/Critical] |
                | Physical Progress % | [percentage] | |
                | Cost Performance (CPI) | [value] | [Good/Warning/Critical] |
                | Schedule Performance (SPI) | [value] | [Good/Warning/Critical] |
                | Target Finish | [original date] | |
                | Expected Finish | [forecast date] | [On Time/Delayed] |
                | Budget Variance | [amount over/under] | [Good/Warning/Critical] |

                ### EXECUTIVE SUMMARY
                - Provide 3-4 bullet points with the most critical information
                - Include overall project health rating (GREEN/YELLOW/RED)
                - Highlight immediate concerns requiring executive attention
                - Quantify financial and schedule impacts

                ### PERFORMANCE ANALYSIS
                **Cost Performance:**
                - Analysis of spending efficiency and budget variance
                - Root cause assessment if CPI < 0.95

                **Schedule Performance:** 
                - Timeline analysis and delay impact
                - Critical path concerns if SPI < 0.90

                ### RISK ASSESSMENT & IMPACT
                - Rate risk level as LOW/MEDIUM/HIGH for each category:
                  - Financial risk (cost overrun probability)
                  - Schedule risk (delivery delay probability) 
                  - Stakeholder risk (reputation/contract penalties)
                - Quantify potential impacts where possible

                ### CORRECTIVE ACTIONS (Priority Ranked)
                Provide specific, measurable actions with:
                1. **Immediate Actions (0-30 days)** - What must be done now
                2. **Short-term Actions (30-90 days)** - Recovery measures
                3. **Long-term Monitoring** - Ongoing controls

                For each action, specify:
                - Specific deliverable/outcome
                - Success metric
                - Responsibility/ownership
                - Timeline

                ### DECISION POINTS
                List any decisions requiring executive/sponsor approval:
                - Budget increase authorizations
                - Scope change considerations  
                - Resource reallocation needs
                - Contract modification requirements

                **CRITICAL**: Use ONLY the data provided above. DO NOT add assumed information, make up details, or infer data that is not explicitly given.

                Focus on being actionable, specific, and quantified. Avoid generic project management advice.
                Use professional tone appropriate for executive leadership.
                """)
                else:
                    # All mode - include all status history for the selected project
                    prompt_parts = []
                    prompt_parts.append(f"""Create an executive-level project status report with full historical trend analysis.
                Structure for C-suite and senior stakeholders who need quick decision-making information.

                **IMPORTANT CONTEXT:**
                - This project is currently in its execution stage (actively being worked on)
                - Complete status history is provided to identify trends and patterns
                - The project is ongoing and not yet completed

                PROJECT: {project_name}
                ORGANIZATION: {project_data.get('organization', '')}
                PROJECT MANAGER: {project_data.get('project_manager', '')}
                CURRENT ANALYSIS DATE: {results['data_date']}
                CURRENCY: {controls['currency_symbol']} {controls.get('currency_postfix', '')}

                === CURRENT STATUS (LATEST) ===
                FINANCIALS:
                - Budget (BAC): {format_currency(results['bac'], controls['currency_symbol'], controls['currency_postfix'])}
                - Actual Cost (AC): {format_currency(results['ac'], controls['currency_symbol'], controls['currency_postfix'])}
                - Earned Value (EV): {format_currency(results['ev'], controls['currency_symbol'], controls['currency_postfix'])}
                - Planned Value (PV): {format_currency(results['pv'], controls['currency_symbol'], controls['currency_postfix'])}

                SCHEDULE:
                - Plan Start: {results['plan_start']}
                - Plan Finish: {results['plan_finish']}
                - Expected Finish: {results['forecast_completion']}
                - Physical Completion: {format_percentage(results['percent_complete'])}

                PERFORMANCE INDICES:
                - Cost Performance (CPI): {format_performance_index(results['cpi'])}
                - Schedule Performance (SPI): {format_performance_index(results['spi'])}
                - Schedule Performance Efficiency (SPIe): {format_performance_index(results.get('spie', 0))}
                - Budget Utilization: {format_percentage(safe_divide(results['ac'], results['bac']) * 100)}
                - Time Utilization: {format_percentage(safe_divide(results['actual_duration_months'], results['original_duration_months']) * 100)}

                FORECASTS:
                - Estimate at Completion (EAC): {format_currency(results['eac'], controls['currency_symbol'], controls['currency_postfix']) if is_valid_finite_number(results['eac']) else 'Cannot determine'}
                - Variance at Completion (VAC): {format_currency(results['vac'], controls['currency_symbol'], controls['currency_postfix']) if is_valid_finite_number(results['vac']) else 'Cannot determine'}
                """)

                    # Get historical status data if in database mode
                    if USE_DATABASE and 'project_id' in project_data:
                        try:
                            adapter = data_manager.get_data_adapter()
                            status_history_df = adapter.db_manager.get_status_reports(project_data['project_id'])

                            if not status_history_df.empty:
                                status_history_df = status_history_df.sort_values('status_date')

                                prompt_parts.append(f"\n=== HISTORICAL STATUS TREND ({len(status_history_df)} data points) ===")
                                prompt_parts.append("Complete project history showing performance evolution over time:\n")

                                # Get project baseline data for calculations
                                bac = results.get('bac', 0)
                                plan_start = pd.to_datetime(project_data.get('plan_start'))
                                plan_finish = pd.to_datetime(project_data.get('plan_finish'))
                                total_duration = ((plan_finish - plan_start).days / 30.44) if plan_start and plan_finish else 0

                                # Get curve parameters for ES calculation
                                curve_type = controls.get('curve_type', 'linear')
                                alpha = controls.get('alpha', 2.0)
                                beta = controls.get('beta', 2.0)

                                for idx, status_row in status_history_df.iterrows():
                                    # Extract raw values from database
                                    # Try calculated columns first (from migration), then fall back to original columns
                                    ac = status_row.get('actual_cost', 0)
                                    pv = status_row.get('calculated_pv') if pd.notna(status_row.get('calculated_pv')) else status_row.get('planned_value', 0)
                                    ev = status_row.get('calculated_ev') if pd.notna(status_row.get('calculated_ev')) else status_row.get('earned_value', 0)
                                    status_date_val = pd.to_datetime(status_row.get('status_date'))

                                    # Use pre-calculated metrics if available, otherwise calculate
                                    if pd.notna(status_row.get('cpi')) and pd.notna(status_row.get('spi')):
                                        # Use stored calculated values
                                        cpi = status_row.get('cpi', 0)
                                        spi = status_row.get('spi', 0)
                                        spie = status_row.get('espi', 0)  # Column is named 'espi' in database
                                        percent_complete = status_row.get('percent_complete', 0)
                                        forecast_completion_str = status_row.get('likely_completion', 'N/A')
                                        if forecast_completion_str and forecast_completion_str != 'N/A':
                                            forecast_completion_str = pd.to_datetime(forecast_completion_str).strftime('%Y-%m-%d')
                                    else:
                                        # Calculate performance metrics
                                        cpi = safe_divide(ev, ac, 1.0) if ac > 0 else 0.0
                                        spi = safe_divide(ev, pv, 1.0) if pv > 0 else 0.0
                                        percent_complete = safe_divide(ev, bac, 0.0) * 100 if bac > 0 else 0.0

                                        # Calculate Earned Schedule (ES) and SPIe
                                        actual_duration = ((status_date_val - plan_start).days / 30.44) if plan_start else 0
                                        if curve_type == 'linear':
                                            es = find_earned_schedule_linear(ev, bac, total_duration)
                                        else:  # s-curve
                                            es = find_earned_schedule_scurve(ev, bac, total_duration, alpha, beta)

                                        spie = safe_divide(es, actual_duration, 1.0) if actual_duration > 0 else 1.0

                                        # Calculate Expected Finish
                                        forecast_duration = safe_divide(total_duration, spie, total_duration) if spie > 0 else total_duration
                                        if is_valid_finite_number(forecast_duration):
                                            forecast_duration_rounded = max(1, math.ceil(forecast_duration))
                                            forecast_completion = add_months_approx(plan_start, forecast_duration_rounded)
                                            forecast_completion_str = forecast_completion.strftime('%Y-%m-%d')
                                        else:
                                            forecast_completion_str = "N/A"

                                    # Add to prompt
                                    prompt_parts.append(f"\n--- Status Date: {status_row.get('status_date')} ---")
                                    prompt_parts.append(f"Data Date: {status_row.get('status_date')}")
                                    prompt_parts.append(f"Budget (BAC): {format_currency(bac, controls['currency_symbol'], controls['currency_postfix'])}")
                                    prompt_parts.append(f"Actual Cost (AC): {format_currency(ac, controls['currency_symbol'], controls['currency_postfix'])}")
                                    prompt_parts.append(f"Planned Value (PV): {format_currency(pv, controls['currency_symbol'], controls['currency_postfix'])}")
                                    prompt_parts.append(f"Earned Value (EV): {format_currency(ev, controls['currency_symbol'], controls['currency_postfix'])}")
                                    prompt_parts.append(f"Cost Performance (CPI): {format_performance_index(cpi)}")
                                    prompt_parts.append(f"Schedule Performance (SPI): {format_performance_index(spi)}")
                                    prompt_parts.append(f"Schedule Performance Efficiency (SPIe): {format_performance_index(spie)}")
                                    prompt_parts.append(f"Expected Finish: {forecast_completion_str}")
                                    prompt_parts.append(f"Physical Completion: {format_percentage(percent_complete)}")

                        except Exception as e:
                            prompt_parts.append(f"\n\nNote: Could not load historical data: {str(e)}")
                            prompt_parts.append("Analysis will be based on current status only.")

                    prompt_parts.append(f"""

                STRUCTURE YOUR RESPONSE AS FOLLOWS:

                # PROJECT STATUS REPORT WITH TREND ANALYSIS

                **Project:** {project_name}
                **Organization:** {project_data.get('organization', 'N/A')}
                **Project Manager:** {project_data.get('project_manager', 'N/A')}
                **Report Date:** {results['data_date']}
                **Currency:** {controls['currency_symbol']} {controls.get('currency_postfix', '')}

                ---

                ### PROJECT DASHBOARD
                Create a formatted table with these key metrics:
                | Metric | Value | Status |
                |--------|--------|---------|
                | Data Date | [date] | |
                | Original Budget | [amount] | |
                | Budget Used % | [percentage] | [Good/Warning/Critical] |
                | Time Elapsed % | [percentage] | [Good/Warning/Critical] |
                | Physical Progress % | [percentage] | |
                | Cost Performance (CPI) | [value] | [Good/Warning/Critical] |
                | Schedule Performance (SPI) | [value] | [Good/Warning/Critical] |
                | Target Finish | [original date] | |
                | Expected Finish | [forecast date] | [On Time/Delayed] |
                | Budget Variance | [amount over/under] | [Good/Warning/Critical] |

                ### EXECUTIVE SUMMARY
                - Provide 3-4 bullet points with the most critical information
                - Include overall project health rating (GREEN/YELLOW/RED)
                - Highlight immediate concerns requiring executive attention
                - Quantify financial and schedule impacts

                ### TREND ANALYSIS (Based on Historical Data)
                **Performance Trends:**
                - Analyze how CPI and SPI have changed over time
                - Identify if performance is improving, stable, or degrading
                - Note any significant inflection points or changes in trajectory
                - Compare current performance against historical average

                **Cost Trend:**
                - Spending pattern analysis over time
                - Rate of cost variance change
                - Acceleration or deceleration of spend

                **Schedule Trend:**
                - Schedule performance evolution
                - Delay accumulation or recovery patterns
                - Progress velocity changes

                ### PERFORMANCE ANALYSIS
                **Cost Performance:**
                - Analysis of spending efficiency and budget variance
                - Root cause assessment if CPI < 0.95
                - Trend: improving or degrading?

                **Schedule Performance:**
                - Timeline analysis and delay impact
                - Critical path concerns if SPI < 0.90
                - Trend: recovering or deteriorating?

                ### RISK ASSESSMENT & IMPACT
                - Rate risk level as LOW/MEDIUM/HIGH for each category:
                  - Financial risk (cost overrun probability)
                  - Schedule risk (delivery delay probability)
                  - Stakeholder risk (reputation/contract penalties)
                - Quantify potential impacts where possible
                - Consider trend direction in risk assessment

                ### CORRECTIVE ACTIONS (Priority Ranked)
                Provide specific, measurable actions with:
                1. **Immediate Actions (0-30 days)** - What must be done now
                2. **Short-term Actions (30-90 days)** - Recovery measures
                3. **Long-term Monitoring** - Ongoing controls

                For each action, specify:
                - Specific deliverable/outcome
                - Success metric
                - Responsibility/ownership
                - Timeline

                ### DECISION POINTS
                List any decisions requiring executive/sponsor approval:
                - Budget increase authorizations
                - Scope change considerations
                - Resource reallocation needs
                - Contract modification requirements

                **CRITICAL**: Use ONLY the data provided above. DO NOT add assumed information, make up details, or infer data that is not explicitly given.

                Focus on being actionable, specific, and quantified. Pay special attention to trends and trajectory.
                Use professional tone appropriate for executive leadership.
                """)

                    prompt = "\n".join(prompt_parts)

                # Get LLM config from session state (set in Portfolio Management)
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
                    generate_clicked = st.button("🚀 Generate Executive Brief", type="primary", key="gen_project_brief")

                get_prompt_clicked = st.button("📋 Get Prompt", key="get_prompt_project_brief", help="Get the prompt to paste into ChatGPT, Claude, or any LLM")

                # Handle Generate button
                if show_generate_button and generate_clicked:
                    with st.spinner("Generating executive brief..."):
                        brief_response = safe_llm_request(
                            llm_config.get('provider', ''),
                            llm_config.get('model', ''),
                            llm_config.get('api_key', ''),
                            llm_config.get('temperature', 0.2),
                            llm_config.get('timeout', 60),
                            prompt
                        )
                        st.session_state.executive_brief = brief_response
                        # Clear chat history and infographic when new brief is generated
                        from components.brief_chat import clear_chat_history
                        clear_chat_history('project_brief_chat_history')
                        if 'project_infographic' in st.session_state:
                            del st.session_state.project_infographic

                # Handle Get Prompt button
                if get_prompt_clicked:
                    st.success("✅ Prompt generated! Copy the text below and paste it into ChatGPT, Claude, or any LLM.")

                    st.text_area(
                        "Prompt for LLM",
                        value=prompt,
                        height=400,
                        key="project_prompt_text_area",
                        help="Copy this entire prompt and paste it into your preferred AI chat interface"
                    )

                    # Download button for the prompt
                    st.download_button(
                        "📥 Download Prompt as Text File",
                        prompt,
                        file_name=f"project_brief_prompt_{selected_project_id}.txt",
                        mime="text/plain",
                        key="download_project_prompt"
                    )
                
                if "executive_brief" in st.session_state:
                    brief = st.session_state.executive_brief
                    if brief.startswith("Error:") or brief == "No API key available":
                        if brief == "No API key available":
                            st.warning("⚠️ No API key available. Please upload an API key file in the LLM Provider section.")
                        else:
                            st.error(brief)
                    else:
                        st.markdown("#### 📄 Executive Summary Report")
                        # Clean up LaTeX/math formatting issues from LLM response
                        # Replace inline math delimiters that shouldn't be interpreted as LaTeX
                        cleaned_brief = brief.replace('$', r'\$')  # Escape dollar signs to prevent LaTeX rendering
                        st.markdown(cleaned_brief)
                        st.download_button("📥 Download Brief", brief, file_name=f"executive_brief_{selected_project_id}.md", mime="text/markdown")

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
                            if st.button("🖼️ Generate Infographic", key="gen_project_infographic", help="Generate an AI-powered visual summary of project metrics"):
                                # Calculate metrics for infographic
                                budget_pct = safe_divide(results['ac'], results['bac']) * 100
                                time_pct = safe_divide(results['actual_duration_months'], results['original_duration_months']) * 100

                                # Determine health status
                                cpi_val = results.get('cpi', 1.0)
                                spi_val = results.get('spi', 1.0)
                                if cpi_val >= 1.0 and spi_val >= 1.0:
                                    health_status = "Healthy"
                                elif cpi_val >= 0.9 or spi_val >= 0.9:
                                    health_status = "At Risk"
                                else:
                                    health_status = "Critical"

                                infographic_metrics = {
                                    'project_name': project_data.get('project_name', selected_project_id),
                                    'cpi': cpi_val,
                                    'spi': spi_val,
                                    'budget_pct': budget_pct,
                                    'time_pct': time_pct,
                                    'health_status': health_status,
                                    'completion_pct': results.get('percent_complete', 0),
                                }

                                with st.spinner("Generating infographic with AI..."):
                                    try:
                                        from services.infographic_service import InfographicService
                                        service = InfographicService(replicate_key, model=infographic_model)
                                        image_bytes, error = service.generate_project_infographic(infographic_metrics)

                                        if image_bytes:
                                            st.session_state.project_infographic = image_bytes
                                            st.rerun()
                                        else:
                                            st.error(f"Failed to generate infographic: {error}")
                                    except Exception as e:
                                        st.error(f"Infographic generation error: {e}")

                            # Display generated infographic
                            if 'project_infographic' in st.session_state:
                                st.image(st.session_state.project_infographic, caption=f"Project Summary Infographic")
                                st.download_button(
                                    "📥 Download Infographic",
                                    st.session_state.project_infographic,
                                    file_name=f"project_infographic_{selected_project_id}.png",
                                    mime="image/png",
                                    key="download_project_infographic"
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
                            history_key='project_brief_chat_history',
                            llm_config=llm_config,
                            voice_config=voice_config,
                            expander_title="Chat with your Project Brief"
                        )

            with tab4:
                st.markdown("### 📈 Performance Visualization")

                # Add toggle for single vs. all status display
                col_toggle, col_spacer = st.columns([1, 3])
                with col_toggle:
                    chart_mode_viz = st.radio(
                        "Chart Display Mode",
                        options=["Single", "All"],
                        index=0,  # Default to "Single" for this tab
                        horizontal=True,
                        help="Single: Show only the current status date | All: Show all historical status points",
                        key="chart_mode_viz"
                    )

                try:
                    import plotly.graph_objects as go
                    from plotly.subplots import make_subplots

                    if not MATPLOTLIB_AVAILABLE:
                        st.error("📊 Charts require matplotlib. Please install: pip install matplotlib")
                        st.info("Charts are disabled until matplotlib is installed.")
                        return

                    # Load historical data if in "All" mode
                    historical_data = None
                    if chart_mode_viz == "All" and USE_DATABASE and 'project_id' in project_data:
                        adapter = data_manager.get_data_adapter()
                        historical_data = adapter.db_manager.get_status_reports(project_data['project_id'])
                        if not historical_data.empty:
                            historical_data['status_date'] = pd.to_datetime(historical_data['status_date'])
                            historical_data = historical_data.sort_values('status_date')

                    # Chart 1: Performance Matrix (CPI vs SPI) - Using Plotly for hover info
                    st.markdown("#### 📊 Performance Matrix")

                    cpi, spi = results['cpi'], results['spi']

                    fig_matrix = go.Figure()

                    # Add quadrant backgrounds
                    # Q1: SPI>=1, CPI>=1 (top right) - Green
                    fig_matrix.add_shape(type="rect", x0=1.0, y0=1.0, x1=1.5, y1=1.5,
                                        fillcolor="green", opacity=0.2, line_width=0, layer="below")
                    # Q2: SPI<1, CPI>=1 (top left) - Yellow
                    fig_matrix.add_shape(type="rect", x0=0.5, y0=1.0, x1=1.0, y1=1.5,
                                        fillcolor="yellow", opacity=0.2, line_width=0, layer="below")
                    # Q3: SPI<1, CPI<1 (bottom left) - Red
                    fig_matrix.add_shape(type="rect", x0=0.5, y0=0.5, x1=1.0, y1=1.0,
                                        fillcolor="red", opacity=0.2, line_width=0, layer="below")
                    # Q4: SPI>=1, CPI<1 (bottom right) - Yellow
                    fig_matrix.add_shape(type="rect", x0=1.0, y0=0.5, x1=1.5, y1=1.0,
                                        fillcolor="yellow", opacity=0.2, line_width=0, layer="below")

                    # Add reference lines
                    fig_matrix.add_hline(y=1.0, line_dash="dash", line_color="gray", opacity=0.7)
                    fig_matrix.add_vline(x=1.0, line_dash="dash", line_color="gray", opacity=0.7)

                    # Plot project points
                    if chart_mode_viz == "All" and historical_data is not None and not historical_data.empty:
                        # Plot all historical points
                        hist_spi = historical_data['spi'].fillna(0)
                        hist_cpi = historical_data['cpi'].fillna(0)
                        hist_dates = historical_data['status_date'].dt.strftime('%Y-%m-%d')

                        # Determine colors for each point
                        colors = ['green' if (c >= 1.0 and s >= 1.0) else 'orange' if (c >= 0.9 or s >= 0.9) else 'red'
                                 for c, s in zip(hist_cpi, hist_spi)]

                        fig_matrix.add_trace(go.Scatter(
                            x=hist_spi,
                            y=hist_cpi,
                            mode='markers',
                            marker=dict(size=10, color=colors, opacity=0.6, line=dict(width=1, color='black')),
                            customdata=hist_dates,
                            hovertemplate='<b>Status Date:</b> %{customdata}<br><b>SPI:</b> %{x:.3f}<br><b>CPI:</b> %{y:.3f}<extra></extra>',
                            name='Historical',
                            showlegend=True
                        ))

                        # Highlight current point
                        color = 'green' if (cpi >= 1.0 and spi >= 1.0) else 'orange' if (cpi >= 0.9 or spi >= 0.9) else 'red'
                        fig_matrix.add_trace(go.Scatter(
                            x=[spi],
                            y=[cpi],
                            mode='markers',
                            marker=dict(size=15, color=color, symbol='star', line=dict(width=2, color='black')),
                            customdata=[pd.to_datetime(status_date).strftime('%Y-%m-%d')],
                            hovertemplate='<b>Current Status Date:</b> %{customdata}<br><b>SPI:</b> %{x:.3f}<br><b>CPI:</b> %{y:.3f}<extra></extra>',
                            name='Current',
                            showlegend=True
                        ))
                    else:
                        # Plot single current point
                        color = 'green' if (cpi >= 1.0 and spi >= 1.0) else 'orange' if (cpi >= 0.9 or spi >= 0.9) else 'red'
                        fig_matrix.add_trace(go.Scatter(
                            x=[spi],
                            y=[cpi],
                            mode='markers',
                            marker=dict(size=15, color=color, line=dict(width=2, color='black')),
                            customdata=[pd.to_datetime(status_date).strftime('%Y-%m-%d')],
                            hovertemplate='<b>Status Date:</b> %{customdata}<br><b>SPI:</b> %{x:.3f}<br><b>CPI:</b> %{y:.3f}<extra></extra>',
                            name='Current',
                            showlegend=False
                        ))

                    title_suffix = ' (All Status Dates)' if chart_mode_viz == "All" else ''
                    fig_matrix.update_layout(
                        title=f'Performance Matrix{title_suffix}',
                        xaxis_title='Schedule Performance Index (SPI)',
                        yaxis_title='Cost Performance Index (CPI)',
                        xaxis=dict(range=[0.5, 1.5], gridcolor='lightgray'),
                        yaxis=dict(range=[0.5, 1.5], gridcolor='lightgray'),
                        height=400,
                        hovermode='closest'
                    )

                    st.plotly_chart(fig_matrix, width='stretch')

                    # Create comprehensive charts for remaining matplotlib charts
                    st.markdown("#### 📊 Additional Performance Charts")
                    fig = plt.figure(figsize=(20, 6))

                    # Professional styling
                    plt.style.use('seaborn-v0_8' if 'seaborn-v0_8' in plt.style.available else 'default')

                    # Chart 2: Progress Comparison
                    ax2 = plt.subplot(1, 2, 1)
                    metrics = ['Budget Used', 'Time Used', 'Work Complete']
                    values = [
                        safe_divide(results['ac'], results['bac']) * 100,
                        safe_divide(results['actual_duration_months'], results['original_duration_months']) * 100,
                        results['percent_complete']
                    ]
                    colors = ['#ff6b6b', '#4ecdc4', '#45b7d1']
                    
                    bars = ax2.barh(metrics, values, color=colors, alpha=0.8)
                    ax2.axvline(100, color='gray', linestyle='--', alpha=0.7)
                    ax2.set_xlim(0, max(120, max(values) + 10))
                    ax2.set_xlabel('Percentage (%)')
                    ax2.set_title('Progress Comparison', fontweight='bold')
                    
                    # Add value labels
                    for bar, value in zip(bars, values):
                        ax2.text(value + 2, bar.get_y() + bar.get_height()/2, 
                                f'{value:.1f}%', va='center', fontweight='bold')
                    
                    # Chart 3: Time/Budget Performance Curve - Using Plotly for hover info
                    st.markdown("#### 📈 Time/Budget Performance Curve")

                    # Calculate normalized time and earned value
                    normalized_time = safe_divide(results['actual_duration_months'], results['original_duration_months'])
                    normalized_ev = safe_divide(results['ev'], results['bac'])
                    normalized_data_date = normalized_time

                    # Create normalized time array from 0 to 1
                    T = np.linspace(0, 1, 101)

                    # Define the performance curves
                    blue_curve = -0.794*T**3 + 0.632*T**2 + 1.162*T
                    red_curve = -0.387*T**3 + 1.442*T**2 - 0.055*T

                    fig_curve = go.Figure()

                    # Plot the reference curves
                    fig_curve.add_trace(go.Scatter(
                        x=T,
                        y=blue_curve,
                        mode='lines',
                        name='Blue Curve',
                        line=dict(color='blue', width=2),
                        hovertemplate='<b>Blue Curve</b><br>Time: %{x:.2f}<br>PV: %{y:.2f}<extra></extra>'
                    ))

                    fig_curve.add_trace(go.Scatter(
                        x=T,
                        y=red_curve,
                        mode='lines',
                        name='Red Curve',
                        line=dict(color='red', width=2),
                        hovertemplate='<b>Red Curve</b><br>Time: %{x:.2f}<br>PV: %{y:.2f}<extra></extra>'
                    ))

                    # Plot historical points if in "All" mode
                    if chart_mode_viz == "All" and historical_data is not None and not historical_data.empty:
                        # Calculate normalized values for all historical points
                        plan_start = pd.to_datetime(project_data.get('plan_start'))
                        plan_finish = pd.to_datetime(project_data.get('plan_finish'))

                        if pd.notna(plan_start) and pd.notna(plan_finish):
                            original_duration_days = (plan_finish - plan_start).days
                            bac = results.get('bac', 0)

                            hist_normalized_time = []
                            hist_normalized_ev = []
                            hist_dates = []

                            for _, row in historical_data.iterrows():
                                status_date_dt = row['status_date']
                                if pd.notna(status_date_dt) and pd.notna(plan_start):
                                    actual_days = (status_date_dt - plan_start).days
                                    norm_time = actual_days / original_duration_days if original_duration_days > 0 else 0
                                    ev_val = row.get('calculated_ev', row.get('earned_value', 0))
                                    norm_ev = ev_val / bac if bac > 0 else 0
                                    hist_normalized_time.append(norm_time)
                                    hist_normalized_ev.append(norm_ev)
                                    hist_dates.append(status_date_dt.strftime('%Y-%m-%d'))

                            # Plot historical points
                            fig_curve.add_trace(go.Scatter(
                                x=hist_normalized_time,
                                y=hist_normalized_ev,
                                mode='markers',
                                marker=dict(size=8, color='orange', opacity=0.6, line=dict(width=1, color='black')),
                                customdata=hist_dates,
                                hovertemplate='<b>Status Date:</b> %{customdata}<br><b>Normalized Time:</b> %{x:.2f}<br><b>Normalized EV:</b> %{y:.2f}<extra></extra>',
                                name='Historical',
                                showlegend=True
                            ))

                        # Highlight current point
                        fig_curve.add_trace(go.Scatter(
                            x=[normalized_data_date],
                            y=[normalized_ev],
                            mode='markers',
                            marker=dict(size=12, color='red', symbol='star', line=dict(width=2, color='black')),
                            customdata=[pd.to_datetime(status_date).strftime('%Y-%m-%d')],
                            hovertemplate='<b>Current Status Date:</b> %{customdata}<br><b>Normalized Time:</b> %{x:.2f}<br><b>Normalized EV:</b> %{y:.2f}<extra></extra>',
                            name='Current EV',
                            showlegend=True
                        ))
                    else:
                        # Add vertical line at normalized Data Date
                        fig_curve.add_vline(x=normalized_data_date, line_dash="dash", line_color="yellow",
                                           opacity=0.8, annotation_text="Data Date")

                        # Add current point
                        fig_curve.add_trace(go.Scatter(
                            x=[normalized_data_date],
                            y=[normalized_ev],
                            mode='markers',
                            marker=dict(size=12, color='red', line=dict(width=2, color='black')),
                            customdata=[pd.to_datetime(status_date).strftime('%Y-%m-%d')],
                            hovertemplate='<b>Status Date:</b> %{customdata}<br><b>Normalized Time:</b> %{x:.2f}<br><b>Normalized EV:</b> %{y:.2f}<extra></extra>',
                            name='Current EV',
                            showlegend=True
                        ))

                    title_suffix = ' (All Status Dates)' if chart_mode_viz == "All" else ''
                    fig_curve.update_layout(
                        title=f'Time/Budget Performance Curve{title_suffix}',
                        xaxis_title='Time (Normalized)',
                        yaxis_title='PV (Normalized)',
                        xaxis=dict(range=[0, 1], gridcolor='lightgray'),
                        yaxis=dict(range=[0, 1.2], gridcolor='lightgray'),
                        height=400,
                        hovermode='closest'
                    )

                    st.plotly_chart(fig_curve, width='stretch')

                    # Chart 4: EVM Curves
                    ax4 = plt.subplot(1, 2, 2)
                    
                    # Create timeline
                    total_months = results['original_duration_months']
                    actual_months = results['actual_duration_months']
                    
                    timeline = np.linspace(0, total_months, int(total_months * 2) + 1)
                    actual_timeline = np.linspace(0, actual_months, int(actual_months * 2) + 1) if actual_months > 0 else [0]
                    
                    # Calculate curves
                    # Use project-specific settings if available, otherwise use global controls
                    # Project-specific settings are stored in columns: "Curve Type", "Alpha", "Beta"
                    project_curve_type = project_row.get('Curve Type')
                    project_alpha = project_row.get('Alpha')
                    project_beta = project_row.get('Beta')

                    # Use project-specific values if they exist and are not None/empty, otherwise fall back to global controls
                    curve_type_to_use = project_curve_type if (project_curve_type is not None and project_curve_type != "") else controls['curve_type']
                    alpha_to_use = project_alpha if (project_alpha is not None and project_alpha != 0.0) else controls['alpha']
                    beta_to_use = project_beta if (project_beta is not None and project_beta != 0.0) else controls['beta']
                    
                    if curve_type_to_use == 's-curve':
                        pv_values = [calculate_pv_scurve(results['bac'], t, total_months, alpha_to_use, beta_to_use) for t in timeline]
                        if actual_months > 0:
                            ac_values = [calculate_pv_scurve(results['ac'], t, actual_months, alpha_to_use, beta_to_use) for t in actual_timeline]
                            ev_values = [calculate_pv_scurve(results['ev'], t, actual_months, alpha_to_use, beta_to_use) for t in actual_timeline]
                        else:
                            ac_values, ev_values = [0], [0]
                    else:
                        pv_values = [calculate_pv_linear(results['bac'], t, total_months) for t in timeline]
                        if actual_months > 0:
                            ac_values = [calculate_pv_linear(results['ac'], t, actual_months) for t in actual_timeline]
                            ev_values = [calculate_pv_linear(results['ev'], t, actual_months) for t in actual_timeline]
                        else:
                            ac_values, ev_values = [0], [0]
                    
                    # Plot curves
                    ax4.plot(timeline, pv_values, 'b-', linewidth=3, label='Planned Value (PV)', alpha=0.8)
                    ax4.plot(actual_timeline, ac_values, 'r-', linewidth=3, label='Actual Cost (AC)', alpha=0.8)
                    ax4.plot(actual_timeline, ev_values, 'g-', linewidth=3, label='Earned Value (EV)', alpha=0.8)
                    
                    # Data date line
                    ax4.axvline(actual_months, color='orange', linestyle='--', linewidth=2, alpha=0.8, label='Data Date')
                    
                    ax4.set_xlabel('Time (Months)')
                    ax4.set_ylabel(f'Value ({controls["currency_symbol"]})')
                    ax4.set_title('EVM Performance Curves', fontweight='bold', fontsize=14)
                    ax4.legend(loc='upper left')
                    ax4.grid(True, alpha=0.3)
                    ax4.set_xlim(0, total_months * 1.1)
                    
                    plt.tight_layout()
                    st.pyplot(fig, width='stretch')
                    
                except Exception as e:
                    st.error(f"Chart generation failed: {e}")
                    st.info("Charts require matplotlib. Ensure it's properly installed.")

            with tab5:
                st.markdown("### 📉 Trend Analysis")

                # Add toggle for single vs. all status display
                col_toggle, col_spacer = st.columns([1, 3])
                with col_toggle:
                    chart_mode = st.radio(
                        "Chart Display Mode",
                        options=["Single", "All"],
                        index=1,  # Default to "All"
                        horizontal=True,
                        help="Single: Show only the current status date | All: Show all historical status points"
                    )

                try:
                    import plotly.graph_objects as go
                    from plotly.subplots import make_subplots

                    # Get historical data for this project
                    if USE_DATABASE and 'project_id' in project_data:
                        adapter = data_manager.get_data_adapter()

                        # Get all status reports for this project
                        trend_df = adapter.db_manager.get_status_reports(project_data['project_id'])

                        if not trend_df.empty and len(trend_df) > 0:
                            # Convert status_date to datetime
                            trend_df['status_date'] = pd.to_datetime(trend_df['status_date'])

                            # Sort by date
                            trend_df = trend_df.sort_values('status_date')

                            # Filter data based on chart mode
                            if chart_mode == "Single":
                                # Show only the currently selected status date
                                selected_status_date = pd.to_datetime(status_date)
                                trend_df_display = trend_df[trend_df['status_date'] == selected_status_date]

                                # If no data for exact date, find the closest date
                                if trend_df_display.empty:
                                    # Find the closest date to the selected status date
                                    trend_df['date_diff'] = abs(trend_df['status_date'] - selected_status_date)
                                    closest_idx = trend_df['date_diff'].idxmin()
                                    trend_df_display = trend_df.loc[[closest_idx]]
                                    trend_df = trend_df.drop('date_diff', axis=1)
                                    st.info(f"Showing data for closest available date: {trend_df_display['status_date'].iloc[0].strftime('%Y-%m-%d')}")
                            else:
                                # Show all historical data
                                trend_df_display = trend_df.copy()

                            # Get BAC from project data
                            bac = results.get('bac', 0)
                            plan_start = pd.to_datetime(project_data.get('plan_start'))
                            plan_finish = pd.to_datetime(project_data.get('plan_finish'))

                            if pd.notna(plan_start) and pd.notna(plan_finish):
                                total_duration_days = (plan_finish - plan_start).days
                            else:
                                total_duration_days = 365  # Default to 1 year if dates not available

                            # Chart 1: Time/Budget Performance Curve
                            st.markdown("#### 📊 Time/Budget Performance Curve")

                            fig1 = go.Figure()

                            # Determine mode for display
                            display_mode = 'markers' if chart_mode == "Single" else 'lines+markers'

                            # Add PV, EV, AC lines with custom hover text
                            fig1.add_trace(go.Scatter(
                                x=trend_df_display['status_date'],
                                y=trend_df_display['calculated_pv'].fillna(trend_df_display['planned_value']).fillna(0),
                                mode=display_mode,
                                name='Planned Value (PV)',
                                line=dict(color='blue', width=2),
                                marker=dict(size=8 if chart_mode == "Single" else 8),
                                customdata=trend_df_display['status_date'].dt.strftime('%Y-%m-%d'),
                                hovertemplate='<b>Status Date:</b> %{customdata}<br><b>PV:</b> %{y:,.2f}<extra></extra>'
                            ))

                            fig1.add_trace(go.Scatter(
                                x=trend_df_display['status_date'],
                                y=trend_df_display['calculated_ev'].fillna(trend_df_display['earned_value']).fillna(0),
                                mode=display_mode,
                                name='Earned Value (EV)',
                                line=dict(color='green', width=2),
                                marker=dict(size=8 if chart_mode == "Single" else 8),
                                customdata=trend_df_display['status_date'].dt.strftime('%Y-%m-%d'),
                                hovertemplate='<b>Status Date:</b> %{customdata}<br><b>EV:</b> %{y:,.2f}<extra></extra>'
                            ))

                            fig1.add_trace(go.Scatter(
                                x=trend_df_display['status_date'],
                                y=trend_df_display['actual_cost'].fillna(0),
                                mode=display_mode,
                                name='Actual Cost (AC)',
                                line=dict(color='red', width=2),
                                marker=dict(size=8 if chart_mode == "Single" else 8),
                                customdata=trend_df_display['status_date'].dt.strftime('%Y-%m-%d'),
                                hovertemplate='<b>Status Date:</b> %{customdata}<br><b>AC:</b> %{y:,.2f}<extra></extra>'
                            ))

                            # Add BAC reference line
                            fig1.add_hline(
                                y=bac,
                                line_dash="dash",
                                line_color="gray",
                                annotation_text=f"BAC: {format_currency(bac, controls['currency_symbol'], controls['currency_postfix'])}",
                                annotation_position="right"
                            )

                            fig1.update_layout(
                                title="EVM Performance Over Time",
                                xaxis_title="Status Date",
                                yaxis_title=f"Value ({controls['currency_symbol']})",
                                hovermode='x unified',
                                height=400
                            )

                            st.plotly_chart(fig1, width='stretch')

                            # Chart 2: SPI, CPI, SPIe Trend
                            st.markdown("#### 📈 Performance Indices Trend")

                            fig2 = go.Figure()

                            fig2.add_trace(go.Scatter(
                                x=trend_df_display['status_date'],
                                y=trend_df_display['spi'].fillna(0),
                                mode=display_mode,
                                name='SPI',
                                line=dict(color='blue', width=2),
                                marker=dict(size=8 if chart_mode == "Single" else 8),
                                customdata=trend_df_display['status_date'].dt.strftime('%Y-%m-%d'),
                                hovertemplate='<b>Status Date:</b> %{customdata}<br><b>SPI:</b> %{y:.3f}<extra></extra>'
                            ))

                            fig2.add_trace(go.Scatter(
                                x=trend_df_display['status_date'],
                                y=trend_df_display['cpi'].fillna(0),
                                mode=display_mode,
                                name='CPI',
                                line=dict(color='green', width=2),
                                marker=dict(size=8 if chart_mode == "Single" else 8),
                                customdata=trend_df_display['status_date'].dt.strftime('%Y-%m-%d'),
                                hovertemplate='<b>Status Date:</b> %{customdata}<br><b>CPI:</b> %{y:.3f}<extra></extra>'
                            ))

                            fig2.add_trace(go.Scatter(
                                x=trend_df_display['status_date'],
                                y=trend_df_display['espi'].fillna(0),
                                mode=display_mode,
                                name='SPIe',
                                line=dict(color='orange', width=2),
                                marker=dict(size=8 if chart_mode == "Single" else 8),
                                customdata=trend_df_display['status_date'].dt.strftime('%Y-%m-%d'),
                                hovertemplate='<b>Status Date:</b> %{customdata}<br><b>SPIe:</b> %{y:.3f}<extra></extra>'
                            ))

                            # Add 1.0 reference line
                            fig2.add_hline(
                                y=1.0,
                                line_dash="dash",
                                line_color="gray",
                                annotation_text="Target: 1.0",
                                annotation_position="right"
                            )

                            fig2.update_layout(
                                title="Performance Indices Over Time",
                                xaxis_title="Status Date",
                                yaxis_title="Index Value",
                                hovermode='x unified',
                                height=400
                            )

                            st.plotly_chart(fig2, width='stretch')

                            # Chart 3: % Budget Used and % Time Used Trend
                            st.markdown("#### ⏱️ Budget and Time Utilization Trend")

                            fig3 = go.Figure()

                            fig3.add_trace(go.Scatter(
                                x=trend_df_display['status_date'],
                                y=trend_df_display['percent_budget_used'].fillna(0),
                                mode=display_mode,
                                name='% Budget Used',
                                line=dict(color='red', width=2),
                                marker=dict(size=8 if chart_mode == "Single" else 8),
                                yaxis='y',
                                customdata=trend_df_display['status_date'].dt.strftime('%Y-%m-%d'),
                                hovertemplate='<b>Status Date:</b> %{customdata}<br><b>Budget Used:</b> %{y:.2f}%<extra></extra>'
                            ))

                            fig3.add_trace(go.Scatter(
                                x=trend_df_display['status_date'],
                                y=trend_df_display['percent_time_used'].fillna(0),
                                mode=display_mode,
                                name='% Time Used',
                                line=dict(color='blue', width=2),
                                marker=dict(size=8 if chart_mode == "Single" else 8),
                                yaxis='y',
                                customdata=trend_df_display['status_date'].dt.strftime('%Y-%m-%d'),
                                hovertemplate='<b>Status Date:</b> %{customdata}<br><b>Time Used:</b> %{y:.2f}%<extra></extra>'
                            ))

                            # Add % Complete for reference
                            fig3.add_trace(go.Scatter(
                                x=trend_df_display['status_date'],
                                y=trend_df_display['percent_complete'].fillna(0),
                                mode=display_mode,
                                name='% Complete',
                                line=dict(color='green', width=2, dash='dot'),
                                marker=dict(size=8 if chart_mode == "Single" else 8),
                                yaxis='y',
                                customdata=trend_df_display['status_date'].dt.strftime('%Y-%m-%d'),
                                hovertemplate='<b>Status Date:</b> %{customdata}<br><b>Complete:</b> %{y:.2f}%<extra></extra>'
                            ))

                            fig3.update_layout(
                                title="Budget and Time Utilization Over Time",
                                xaxis_title="Status Date",
                                yaxis_title="Percentage (%)",
                                hovermode='x unified',
                                height=400
                            )

                            st.plotly_chart(fig3, width='stretch')

                            # Table: Detailed Trend Data
                            st.markdown("#### 📋 Detailed Trend Data")

                            # Calculate delay for each row
                            trend_display = trend_df_display.copy()

                            # Format the data for display
                            display_columns = []
                            column_config = {}

                            if 'status_date' in trend_display.columns:
                                trend_display['Data Date'] = pd.to_datetime(trend_display['status_date']).dt.strftime('%Y-%m-%d')
                                display_columns.append('Data Date')
                                column_config['Data Date'] = st.column_config.TextColumn('Data Date', width='medium')

                            if 'actual_cost' in trend_display.columns:
                                trend_display['AC'] = trend_display['actual_cost'].fillna(0)
                                display_columns.append('AC')
                                column_config['AC'] = st.column_config.NumberColumn('AC', format=f"{controls['currency_symbol']} %.2f")

                            if 'calculated_pv' in trend_display.columns or 'planned_value' in trend_display.columns:
                                trend_display['PV'] = trend_display['calculated_pv'].fillna(trend_display.get('planned_value', 0)).fillna(0)
                                display_columns.append('PV')
                                column_config['PV'] = st.column_config.NumberColumn('PV', format=f"{controls['currency_symbol']} %.2f")

                            if 'calculated_ev' in trend_display.columns or 'earned_value' in trend_display.columns:
                                trend_display['EV'] = trend_display['calculated_ev'].fillna(trend_display.get('earned_value', 0)).fillna(0)
                                display_columns.append('EV')
                                column_config['EV'] = st.column_config.NumberColumn('EV', format=f"{controls['currency_symbol']} %.2f")

                            if 'spi' in trend_display.columns:
                                trend_display['SPI'] = trend_display['spi'].fillna(0)
                                display_columns.append('SPI')
                                column_config['SPI'] = st.column_config.NumberColumn('SPI', format="%.3f")

                            if 'espi' in trend_display.columns:
                                trend_display['SPIe'] = trend_display['espi'].fillna(0)
                                display_columns.append('SPIe')
                                column_config['SPIe'] = st.column_config.NumberColumn('SPIe', format="%.3f")

                            if 'cpi' in trend_display.columns:
                                trend_display['CPI'] = trend_display['cpi'].fillna(0)
                                display_columns.append('CPI')
                                column_config['CPI'] = st.column_config.NumberColumn('CPI', format="%.3f")

                            if 'eac' in trend_display.columns:
                                trend_display['EAC'] = trend_display['eac'].fillna(0)
                                display_columns.append('EAC')
                                column_config['EAC'] = st.column_config.NumberColumn('EAC', format=f"{controls['currency_symbol']} %.2f")

                            if 'vac' in trend_display.columns:
                                trend_display['VAC'] = trend_display['vac'].fillna(0)
                                display_columns.append('VAC')
                                column_config['VAC'] = st.column_config.NumberColumn('VAC', format=f"{controls['currency_symbol']} %.2f")

                            if 'likely_completion' in trend_display.columns:
                                trend_display['Likely Finish Date'] = pd.to_datetime(trend_display['likely_completion'], errors='coerce').dt.strftime('%Y-%m-%d')
                                display_columns.append('Likely Finish Date')
                                column_config['Likely Finish Date'] = st.column_config.TextColumn('Likely Finish Date', width='medium')

                            # Calculate delay in days
                            if 'likely_completion' in trend_display.columns and pd.notna(plan_finish):
                                trend_display['Delay (Days)'] = (pd.to_datetime(trend_display['likely_completion'], errors='coerce') - plan_finish).dt.days
                                display_columns.append('Delay (Days)')
                                column_config['Delay (Days)'] = st.column_config.NumberColumn('Delay (Days)', format="%d")

                            # Display the table
                            if display_columns:
                                st.dataframe(
                                    trend_display[display_columns],
                                    width='stretch',
                                    height=400,
                                    column_config=column_config,
                                    hide_index=True
                                )

                                # Download button
                                csv_buffer = io.StringIO()
                                trend_display[display_columns].to_csv(csv_buffer, index=False)
                                st.download_button(
                                    "📥 Download Trend Data (CSV)",
                                    csv_buffer.getvalue(),
                                    file_name=f"trend_analysis_{selected_project_id}.csv",
                                    mime="text/csv"
                                )
                            else:
                                st.warning("No trend data columns available to display")
                        else:
                            st.info("📊 No historical data available for trend analysis. Multiple status dates are required.")
                    else:
                        st.warning("⚠️ Trend analysis is only available in database mode with historical status data.")

                except Exception as e:
                    st.error(f"Trend analysis failed: {e}")
                    logger.error(f"Trend analysis error: {e}")

        except Exception as e:
            st.error(f"EVM calculation failed: {e}")
            logger.error(f"EVM calculation error: {e}")

    except Exception as e:
        st.error(f"An unexpected application error occurred: {e}")
        logger.error(f"Application error: {e}", exc_info=True)

    # Show user info in sidebar
    from utils.auth import show_user_info_sidebar
    show_user_info_sidebar()


# =============================================================================
# FILE HANDLING FUNCTIONS
# =============================================================================
def load_json_file(uploaded_file) -> tuple[pd.DataFrame, Dict[str, Any], str]:
    """Load JSON file and return DataFrame, config, and filename."""
    try:
        json_content = uploaded_file.getvalue()
        filename = uploaded_file.name
        config_data = json.loads(json_content.decode('utf-8'))
        
        project_data = []
        if 'data' in config_data and isinstance(config_data['data'], list):
            project_data = config_data['data']
        elif isinstance(config_data, list):
            project_data = config_data
        
        df = pd.DataFrame(project_data) if project_data else pd.DataFrame()
        return df, config_data, filename
        
    except json.JSONDecodeError as e:
        raise ValueError(f"Invalid JSON format: {e}")
    except Exception as e:
        logger.error(f"Failed to load JSON file: {e}")
        raise


def load_csv_file(uploaded_file) -> tuple[pd.DataFrame, str]:
    """Load CSV file."""
    try:
        filename = uploaded_file.name
        encodings = ['utf-8-sig', 'utf-8', 'latin-1', 'cp1252']
        
        for encoding in encodings:
            try:
                uploaded_file.seek(0)
                df = pd.read_csv(uploaded_file, encoding=encoding)

                # Clean up unnamed columns (like index columns from exported CSVs)
                unnamed_cols = [col for col in df.columns if col.startswith('Unnamed:')]
                if unnamed_cols:
                    df = df.drop(columns=unnamed_cols)

                if not df.empty:
                    return df, filename
            except Exception:
                continue
        
        raise ValueError("Could not load CSV file with any supported encoding.")
        
    except Exception as e:
        logger.error(f"Failed to load CSV file: {e}")
        raise

if __name__ == "__main__":
    main()
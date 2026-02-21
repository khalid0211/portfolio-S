"""
Core modules for the Portfolio Management Suite
"""

# Import ALL functions from utils module for easy access
from .utils import *  # noqa: F401, F403
# Explicit imports for IDE support and documentation
from .utils import (
    # Validation & Safety
    validate_numeric_input,
    safe_divide,
    is_valid_finite_number,
    safe_calculate_forecast_duration,

    # Date Functions
    parse_date_any,
    add_months_approx,

    # Mathematical Functions
    scurve_cdf,
    calculate_durations,

    # Constants
    DAYS_PER_MONTH,
    INTEGRATION_STEPS,
    EXCEL_ORDINAL_BASE,
)

# Import from evm_engine module
from .evm_engine import (
    # Main APIs
    perform_batch_calculation,
    perform_complete_evm_analysis,
    calculate_evm_metrics,

    # EVM calculation functions
    calculate_pv_linear,
    calculate_pv_scurve,
    find_earned_schedule_linear,
    find_earned_schedule_scurve,
    calculate_earned_schedule_metrics,

    # Financial calculations
    calculate_present_value,

    # Additional utilities from evm_engine
    safe_financial_metrics,
)

__all__ = [
    # Main APIs
    'perform_batch_calculation',
    'perform_complete_evm_analysis',
    'calculate_evm_metrics',

    # EVM calculation functions
    'calculate_pv_linear',
    'calculate_pv_scurve',
    'find_earned_schedule_linear',
    'find_earned_schedule_scurve',
    'calculate_earned_schedule_metrics',

    # Financial calculations
    'calculate_present_value',
    'safe_financial_metrics',

    # Utility functions (from core.utils)
    'validate_numeric_input',
    'safe_divide',
    'is_valid_finite_number',
    'safe_calculate_forecast_duration',
    'parse_date_any',
    'add_months_approx',
    'scurve_cdf',
    'calculate_durations',

    # Constants
    'DAYS_PER_MONTH',
    'INTEGRATION_STEPS',
    'EXCEL_ORDINAL_BASE',
]
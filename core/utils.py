"""
Core Utility Functions for EVM Portfolio Management System

This module contains shared utility functions that are used across the EVM calculation
engine and other modules. These functions provide:
- Numeric validation and safe mathematical operations
- Date parsing and manipulation
- S-curve calculations for project scheduling
- Duration calculations

All functions are designed to be resilient with comprehensive error handling.
"""

from __future__ import annotations
import math
import logging
import calendar
from datetime import datetime, date, timedelta
from typing import Any, Tuple, Optional

import pandas as pd
import numpy as np
from dateutil import parser as date_parser

# Set up logging
logger = logging.getLogger(__name__)

# Constants
DAYS_PER_MONTH = 30.44  # Average days per month for duration calculations
INTEGRATION_STEPS = 200  # Steps for numerical integration in S-curve calculations
EXCEL_ORDINAL_BASE = datetime(1899, 12, 30)  # Excel date ordinal base


# ============================================================================
# VALIDATION & SAFETY FUNCTIONS
# ============================================================================

def validate_numeric_input(
    value: Any,
    field_name: str,
    min_val: Optional[float] = None,
    max_val: Optional[float] = None
) -> float:
    """
    Validate and convert a value to a valid finite float.

    Args:
        value: The value to validate (can be any type)
        field_name: Name of the field (for error messages)
        min_val: Minimum allowed value (inclusive), or None for no minimum
        max_val: Maximum allowed value (inclusive), or None for no maximum

    Returns:
        float: The validated numeric value

    Raises:
        ValueError: If the value is invalid, NaN, infinite, or out of range

    Examples:
        >>> validate_numeric_input(100.5, "BAC", min_val=0.0)
        100.5
        >>> validate_numeric_input(-10, "Budget", min_val=0.0)
        ValueError: Budget must be >= 0.0
    """
    try:
        num_val = float(value)
        if math.isnan(num_val) or math.isinf(num_val):
            raise ValueError(f"{field_name} cannot be NaN or infinite")
        if min_val is not None and num_val < min_val:
            raise ValueError(f"{field_name} must be >= {min_val}")
        if max_val is not None and num_val > max_val:
            raise ValueError(f"{field_name} must be <= {max_val}")
        return num_val
    except (ValueError, TypeError) as e:
        raise ValueError(f"Invalid {field_name}: {e}")


def safe_divide(
    numerator: float,
    denominator: float,
    default: float = 0.0,
    return_na_for_zero_ac: bool = False
) -> float:
    """
    Safely divide two numbers, avoiding division by zero.

    Args:
        numerator: The numerator value
        denominator: The denominator value
        default: Default value to return on division errors
        return_na_for_zero_ac: If True, returns NaN for AC=0 cases (displayed as 'N/A')

    Returns:
        float: The division result, or default/NaN on error

    Examples:
        >>> safe_divide(100, 50)
        2.0
        >>> safe_divide(100, 0)
        0.0
        >>> safe_divide(100, 0, return_na_for_zero_ac=True)
        nan
    """
    try:
        if abs(denominator) < 1e-10:
            if return_na_for_zero_ac and denominator == 0:
                return float('nan')  # Will be displayed as 'N/A'
            return default
        result = numerator / denominator
        if math.isinf(result) or math.isnan(result):
            return default
        return result
    except (ZeroDivisionError, TypeError, ValueError):
        return default


def is_valid_finite_number(value: Any) -> bool:
    """
    Check if a value is a valid finite number (not None, NaN, or Inf).

    Args:
        value: The value to check

    Returns:
        bool: True if the value is a valid finite number, False otherwise

    Examples:
        >>> is_valid_finite_number(42.5)
        True
        >>> is_valid_finite_number(float('nan'))
        False
        >>> is_valid_finite_number(None)
        False
        >>> is_valid_finite_number(float('inf'))
        False
    """
    try:
        if value is None:
            return False
        num_val = float(value)
        return math.isfinite(num_val)
    except (ValueError, TypeError, OverflowError):
        return False


def safe_calculate_forecast_duration(
    total_duration: float,
    spie: float,
    original_duration: Optional[float] = None
) -> float:
    """
    Safely calculate forecast duration (Likely Duration) with constraint that LD <= 2.5 × OD.

    This function implements the constraint that prevents unrealistic schedule forecasts
    by capping the likely duration at 2.5 times the original duration.

    Args:
        total_duration: The original planned project duration (months)
        spie: The Schedule Performance Index (Earned Schedule)
        original_duration: Alternative parameter name for total_duration (for compatibility)

    Returns:
        float: The likely duration in months, capped at 2.5 × original duration

    Examples:
        >>> safe_calculate_forecast_duration(12, 0.8)  # Behind schedule
        15.0  # 12/0.8 = 15 months
        >>> safe_calculate_forecast_duration(12, 0.3)  # Way behind
        30.0  # Capped at 12 * 2.5 = 30 months (would be 40 without cap)
        >>> safe_calculate_forecast_duration(12, 1.2)  # Ahead of schedule
        10.0  # 12/1.2 = 10 months
    """
    try:
        duration = original_duration if original_duration is not None else total_duration

        if duration <= 0 or not is_valid_finite_number(duration):
            return duration

        if spie <= 0 or not is_valid_finite_number(spie):
            return min(duration * 2.5, duration * 2.5)  # Default to max constraint

        likely_duration = duration / spie
        max_duration = duration * 2.5

        return min(likely_duration, max_duration)

    except (ValueError, TypeError, ZeroDivisionError):
        return total_duration if total_duration > 0 else 12.0  # Fallback to input or 1 year


# ============================================================================
# DATE PARSING & MANIPULATION
# ============================================================================

def parse_date_any(x: Any) -> Optional[datetime]:
    """
    Parse various date formats and types into a datetime object.

    Supports:
    - datetime objects (passthrough)
    - date objects (converted to datetime)
    - pandas Timestamp
    - Excel ordinal dates (numeric)
    - String dates in multiple formats (DD/MM/YYYY, MM/DD/YYYY, YYYY-MM-DD, etc.)
    - Unix timestamps

    Args:
        x: The date value to parse (any type)

    Returns:
        datetime: Parsed datetime object, or None if parsing fails

    Examples:
        >>> parse_date_any("01/01/2025")
        datetime(2025, 1, 1, 0, 0)
        >>> parse_date_any(44927)  # Excel ordinal
        datetime(2023, 1, 1, 0, 0)
        >>> parse_date_any(None)
        None
    """
    if x is None:
        return None

    # Already a datetime
    if isinstance(x, datetime):
        return x

    # Python date object
    elif isinstance(x, date):
        return datetime.combine(x, datetime.min.time())

    # Pandas Timestamp
    elif isinstance(x, pd.Timestamp):
        return x.to_pydatetime()

    # Excel ordinal or numeric timestamp
    elif isinstance(x, (int, float)):
        try:
            if x > 1:  # Reasonable check for Excel dates (Excel epoch starts at 1900)
                return EXCEL_ORDINAL_BASE + timedelta(days=x)
        except (OverflowError, ValueError):
            pass
        return None

    # String date
    elif isinstance(x, str):
        if x.strip() == "":
            return None
        try:
            # Try multiple common date formats
            formats = [
                '%d/%m/%Y', '%m/%d/%Y', '%Y-%m-%d', '%d-%m-%Y',
                '%d/%m/%y', '%m/%d/%y', '%y-%m-%d', '%d-%m-%y',
                '%Y/%m/%d', '%d.%m.%Y', '%Y.%m.%d'
            ]

            for fmt in formats:
                try:
                    return datetime.strptime(x.strip(), fmt)
                except ValueError:
                    continue

            # Fallback to dateutil parser (handles many formats automatically)
            return date_parser.parse(x, dayfirst=True)
        except (ValueError, TypeError):
            try:
                # Last resort: try as Unix timestamp
                return datetime.fromtimestamp(float(x))
            except (ValueError, TypeError, OSError):
                return None

    return None


def add_months_approx(start_dt: datetime, months: int) -> datetime:
    """
    Add months to a datetime, approximating for month-end edge cases.

    Handles edge cases like adding 1 month to January 31st (result: Feb 28/29).
    Uses calendar-aware logic to ensure valid dates.

    Args:
        start_dt: Starting datetime
        months: Number of months to add (can be negative for subtraction)

    Returns:
        datetime: New datetime with months added

    Examples:
        >>> add_months_approx(datetime(2025, 1, 15), 2)
        datetime(2025, 3, 15, 0, 0)
        >>> add_months_approx(datetime(2025, 1, 31), 1)  # Jan 31 + 1 month
        datetime(2025, 2, 28, 0, 0)  # Feb 28 (Feb 31 doesn't exist)
        >>> add_months_approx(datetime(2025, 3, 15), -1)
        datetime(2025, 2, 15, 0, 0)
    """
    try:
        if not isinstance(start_dt, datetime):
            start_dt = parse_date_any(start_dt)
            if start_dt is None:
                return datetime.now()

        # Calculate new year and month
        new_month = start_dt.month + months
        new_year = start_dt.year + (new_month - 1) // 12
        new_month = ((new_month - 1) % 12) + 1

        # Handle day overflow (e.g., Jan 31 + 1 month should be Feb 28/29)
        try:
            new_dt = start_dt.replace(year=new_year, month=new_month)
        except ValueError:
            # Day doesn't exist in new month (e.g., Feb 30), so use last day of month
            max_day = calendar.monthrange(new_year, new_month)[1]
            new_dt = start_dt.replace(year=new_year, month=new_month, day=max_day)

        return new_dt

    except (ValueError, TypeError, AttributeError):
        # Fallback: approximate using average days per month
        days_to_add = months * DAYS_PER_MONTH
        return start_dt + timedelta(days=days_to_add)


# ============================================================================
# MATHEMATICAL FUNCTIONS
# ============================================================================

def scurve_cdf(x: float, alpha: float = 2.0, beta: float = 2.0) -> float:
    """
    Calculate S-curve cumulative distribution function using Beta distribution.

    The S-curve is used to model realistic project progress curves, where:
    - Projects start slowly (ramp-up)
    - Progress accelerates in the middle
    - Progress slows near completion

    This uses the Beta distribution CDF with parameters alpha and beta.
    Default Beta(2,2) gives a symmetric S-curve.

    Args:
        x: Progress ratio (0.0 to 1.0, representing 0% to 100% of duration)
        alpha: Beta distribution alpha parameter (controls early curve shape)
        beta: Beta distribution beta parameter (controls late curve shape)

    Returns:
        float: S-curve value (0.0 to 1.0, representing expected % complete)

    Examples:
        >>> scurve_cdf(0.5)  # At 50% of duration
        0.5  # Expect 50% complete (Beta(2,2) is symmetric)
        >>> scurve_cdf(0.25)  # At 25% of duration
        0.15625  # Expect ~16% complete (slow start)
        >>> scurve_cdf(0.75)  # At 75% of duration
        0.84375  # Expect ~84% complete

    Notes:
        - Alpha > Beta: Front-loaded curve (more work early)
        - Alpha < Beta: Back-loaded curve (more work late)
        - Alpha = Beta = 2: Symmetric S-curve (industry standard)
    """
    try:
        # Clamp x to valid range [0, 1]
        x = max(0.0, min(1.0, float(x)))
        alpha = max(0.1, float(alpha))
        beta = max(0.1, float(beta))

        # Closed-form solution for Beta(2,2) - most common case
        if abs(alpha - 2.0) < 1e-9 and abs(beta - 2.0) < 1e-9:
            return 3 * x * x - 2 * x * x * x

        # Edge cases
        if x == 0.0:
            return 0.0
        if x == 1.0:
            return 1.0

        # Numerical integration for general Beta distribution
        n = max(INTEGRATION_STEPS, 100)
        xs = np.linspace(0, x, n + 1)
        pdf_vals = (xs**(alpha-1)) * ((1 - xs)**(beta-1))

        # Calculate Beta function normalization constant using gamma functions
        try:
            B = math.gamma(alpha) * math.gamma(beta) / math.gamma(alpha + beta)
        except (ValueError, OverflowError):
            logger.warning(f"Gamma function overflow for alpha={alpha}, beta={beta}")
            return 3 * x * x - 2 * x * x * x  # Fallback to Beta(2,2)

        # Integrate using trapezoidal rule
        result = float(np.trapz(pdf_vals, xs) / B)
        return max(0.0, min(1.0, result))

    except Exception as e:
        logger.error(f"S-curve CDF calculation failed: {e}")
        return 3 * x * x - 2 * x * x * x  # Safe fallback to Beta(2,2)


def calculate_durations(
    plan_start: Any,
    plan_finish: Any,
    data_date: Any
) -> Tuple[float, float]:
    """
    Calculate actual duration (elapsed) and original duration (planned).

    Durations are calculated in months using an average of 30.44 days/month.

    Args:
        plan_start: Project planned start date (any format parseable by parse_date_any)
        plan_finish: Project planned finish date (any format parseable by parse_date_any)
        data_date: Current/as-of date for analysis (any format parseable by parse_date_any)

    Returns:
        Tuple[float, float]: (actual_duration, original_duration) in months
            - actual_duration: Months from plan_start to data_date (AD)
            - original_duration: Months from plan_start to plan_finish (OD)

    Examples:
        >>> calculate_durations("01/01/2025", "31/12/2025", "01/07/2025")
        (6.0, 12.0)  # 6 months elapsed, 12 months planned
        >>> calculate_durations("01/01/2025", "01/01/2026", "01/01/2025")
        (0.0, 12.0)  # Just started, 12 months planned

    Notes:
        - Returns (0.0, 0.0) if date parsing fails
        - Negative durations are clamped to 0.0
        - Results are rounded to 2 decimal places
    """
    try:
        # Parse dates (handle both raw and pre-parsed datetime objects)
        ps = plan_start if isinstance(plan_start, datetime) else parse_date_any(plan_start)
        pf = plan_finish if isinstance(plan_finish, datetime) else parse_date_any(plan_finish)
        dd = data_date if isinstance(data_date, datetime) else parse_date_any(data_date)

        # Calculate durations in months
        duration_to_date = max(((dd - ps).days / DAYS_PER_MONTH), 0.0)
        original_duration = max(((pf - ps).days / DAYS_PER_MONTH), 0.0)

        return round(duration_to_date, 2), round(original_duration, 2)

    except Exception as e:
        logger.error(f"Duration calculation failed: {e}")
        return 0.0, 0.0


# ============================================================================
# MODULE EXPORTS
# ============================================================================

__all__ = [
    # Validation & Safety
    'validate_numeric_input',
    'safe_divide',
    'is_valid_finite_number',
    'safe_calculate_forecast_duration',

    # Date Functions
    'parse_date_any',
    'add_months_approx',

    # Mathematical Functions
    'scurve_cdf',
    'calculate_durations',

    # Constants
    'DAYS_PER_MONTH',
    'INTEGRATION_STEPS',
    'EXCEL_ORDINAL_BASE',
]

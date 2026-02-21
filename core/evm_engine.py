"""
EVM Calculation Engine - Centralized Earned Value Management calculations

This module provides all the core EVM calculation functions extracted from
the Project Analysis module to enable reuse across the application.
"""

from __future__ import annotations
import io
import math
import logging
from datetime import datetime, date, timedelta
from typing import Optional, Dict, Any, List, Union, Tuple

import pandas as pd
import numpy as np
from dateutil import parser as date_parser

# Set up logging
logger = logging.getLogger(__name__)

# Constants
DAYS_PER_MONTH = 30.44
INTEGRATION_STEPS = 200
EXCEL_ORDINAL_BASE = datetime(1899, 12, 30)

# ============================================================================
# SECTION 1: UTILITY FUNCTIONS (Level 4 - Dependencies)
# ============================================================================

def validate_numeric_input(value: Any, field_name: str, min_val: float = None, max_val: float = None) -> float:
    """Validate numeric input with proper error handling."""
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


def safe_divide(numerator: float, denominator: float, default: float = 0.0, return_na_for_zero_ac: bool = False) -> float:
    """Safely divide two numbers, avoiding division by zero.

    Args:
        numerator: The numerator value
        denominator: The denominator value
        default: Default value to return on division errors
        return_na_for_zero_ac: If True, returns 'N/A' for AC=0 cases in financial metrics
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


def safe_calculate_forecast_duration(total_duration: float, spie: float, original_duration: float = None) -> float:
    """Safely calculate forecast duration (LD) with constraint that LD <= 2.5 * OD.

    Args:
        total_duration: The original planned project duration
        spie: The Schedule Performance Index (Earned Schedule)
        original_duration: Alternative name for total_duration (for compatibility)

    Returns:
        The likely duration, capped at 2.5 times the original duration
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
        return total_duration if total_duration > 0 else 12.0  # Fallback


def safe_financial_metrics(ev: float, ac: float, pv: float = None) -> dict:
    """Calculate financial metrics safely with enhanced AC=0 handling.

    Args:
        ev: Earned Value
        ac: Actual Cost
        pv: Planned Value (optional)

    Returns:
        Dictionary with cpi, spi, and related metrics
    """
    metrics = {}

    # Cost Performance Index (CPI)
    if ac == 0:
        # Special case: AC=0 means no money spent yet
        if ev == 0:
            metrics['cpi'] = float('nan')  # 0/0 case - will show as 'N/A'
        else:
            metrics['cpi'] = float('inf')  # EV>0, AC=0 - perfect efficiency
    else:
        metrics['cpi'] = safe_divide(ev, ac, default=0.0)

    # Schedule Performance Index (SPI)
    if pv is not None:
        metrics['spi'] = safe_divide(ev, pv, default=0.0)
    else:
        metrics['spi'] = 0.0

    return metrics


def is_valid_finite_number(value: Any) -> bool:
    """Check if a value is a valid finite number."""
    try:
        if value is None:
            return False
        num_val = float(value)
        return math.isfinite(num_val)
    except (ValueError, TypeError, OverflowError):
        return False


def parse_date_any(x):
    """Parse various date formats and types into a datetime object."""
    if x is None:
        return None

    if isinstance(x, datetime):
        return x
    elif isinstance(x, date):
        return datetime.combine(x, datetime.min.time())
    elif isinstance(x, pd.Timestamp):
        return x.to_pydatetime()
    elif isinstance(x, (int, float)):
        # Handle Excel ordinal dates
        try:
            if x > 1:  # Reasonable check for Excel dates
                return EXCEL_ORDINAL_BASE + timedelta(days=x)
        except (OverflowError, ValueError):
            pass
        return None
    elif isinstance(x, str):
        if x.strip() == "":
            return None
        try:
            # Try multiple date formats
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

            # Fallback to dateutil parser
            return date_parser.parse(x, dayfirst=True)
        except (ValueError, TypeError):
            try:
                # Last resort: try as timestamp
                return datetime.fromtimestamp(float(x))
            except (ValueError, TypeError, OSError):
                return None

    return None


def add_months_approx(start_dt: datetime, months: int) -> datetime:
    """Add months to a datetime, approximating for month-end edge cases.

    Args:
        start_dt: Starting datetime
        months: Number of months to add (can be negative)

    Returns:
        New datetime with months added
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
            import calendar
            max_day = calendar.monthrange(new_year, new_month)[1]
            new_dt = start_dt.replace(year=new_year, month=new_month, day=max_day)

        return new_dt

    except (ValueError, TypeError, AttributeError):
        # Fallback: approximate using days
        days_to_add = months * DAYS_PER_MONTH
        return start_dt + timedelta(days=days_to_add)


# ============================================================================
# SECTION 2: MATHEMATICAL FUNCTIONS (Level 3 - Math utilities)
# ============================================================================

def scurve_cdf(x: float, alpha: float = 2.0, beta: float = 2.0) -> float:
    """S-curve CDF with improved error handling and validation."""
    try:
        x = max(0.0, min(1.0, float(x)))
        alpha = max(0.1, float(alpha))
        beta = max(0.1, float(beta))

        # Closed-form for Beta(2,2)
        if abs(alpha - 2.0) < 1e-9 and abs(beta - 2.0) < 1e-9:
            return 3 * x * x - 2 * x * x * x

        # Improved numeric integration for other parameters
        if x == 0.0:
            return 0.0
        if x == 1.0:
            return 1.0

        n = max(INTEGRATION_STEPS, 100)
        xs = np.linspace(0, x, n + 1)
        pdf_vals = (xs**(alpha-1)) * ((1 - xs)**(beta-1))

        # Use gamma function with error handling
        try:
            B = math.gamma(alpha) * math.gamma(beta) / math.gamma(alpha + beta)
        except (ValueError, OverflowError):
            logger.warning(f"Gamma function overflow for alpha={alpha}, beta={beta}")
            return 3 * x * x - 2 * x * x * x  # Fallback to Beta(2,2)

        result = float(np.trapz(pdf_vals, xs) / B)
        return max(0.0, min(1.0, result))

    except Exception as e:
        logger.error(f"S-curve CDF calculation failed: {e}")
        return 3 * x * x - 2 * x * x * x  # Safe fallback


def calculate_durations(plan_start, plan_finish, data_date) -> Tuple[float, float]:
    """Calculate durations with improved error handling."""
    try:
        # Handle both raw dates and already-parsed datetime objects
        ps = plan_start if isinstance(plan_start, datetime) else parse_date_any(plan_start)
        pf = plan_finish if isinstance(plan_finish, datetime) else parse_date_any(plan_finish)
        dd = data_date if isinstance(data_date, datetime) else parse_date_any(data_date)

        duration_to_date = max(((dd - ps).days / DAYS_PER_MONTH), 0.0)
        original_duration = max(((pf - ps).days / DAYS_PER_MONTH), 0.0)

        return round(duration_to_date, 2), round(original_duration, 2)

    except Exception as e:
        logger.error(f"Duration calculation failed: {e}")
        return 0.0, 0.0


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

        monthly_rate = (1 + annual_inflation_rate) ** (1/12) - 1
        pmt = ac / max(ad, 1e-9)

        try:
            factor = (1 - (1 + monthly_rate) ** (-ad)) / monthly_rate
            return round(max(pmt * factor, 0.0), 2)
        except (OverflowError, ValueError):
            logger.warning("PV of Progress calculation overflow, returning AC")
            return round(ac, 2)

    except ValueError as e:
        logger.error(f"Present value of progress calculation failed: {e}")
        return 0.0


def calculate_planned_value_of_project(bac, od, annual_inflation_rate) -> float:
    """Calculate Planned Value of Project: (BAC / OD) * (1-(1+r)^(-OD))/r"""
    try:
        bac = validate_numeric_input(bac, "BAC", min_val=0.0)
        od = validate_numeric_input(od, "OD", min_val=0.0)
        annual_inflation_rate = validate_numeric_input(annual_inflation_rate, "Inflation Rate", min_val=0.0, max_val=1.0)

        if od == 0:
            return 0.0
        if annual_inflation_rate == 0:
            return round(bac, 2)

        monthly_rate = (1 + annual_inflation_rate) ** (1/12) - 1
        pmt = bac / max(od, 1e-9)

        try:
            factor = (1 - (1 + monthly_rate) ** (-od)) / monthly_rate
            return round(max(pmt * factor, 0.0), 2)
        except (OverflowError, ValueError):
            logger.warning("PV of Project calculation overflow, returning BAC")
            return round(bac, 2)

    except ValueError as e:
        logger.error(f"Planned value of project calculation failed: {e}")
        return 0.0


def calculate_likely_value_of_project(bac, ld, annual_inflation_rate) -> float:
    """Calculate Likely Value of Project: (BAC / LD) * (1-(1+r)^(-LD))/r"""
    try:
        bac = validate_numeric_input(bac, "BAC", min_val=0.0)
        ld = validate_numeric_input(ld, "LD", min_val=0.0)
        annual_inflation_rate = validate_numeric_input(annual_inflation_rate, "Inflation Rate", min_val=0.0, max_val=1.0)

        if ld == 0:
            return 0.0
        if annual_inflation_rate == 0:
            return round(bac, 2)

        monthly_rate = (1 + annual_inflation_rate) ** (1/12) - 1
        pmt = bac / max(ld, 1e-9)

        try:
            factor = (1 - (1 + monthly_rate) ** (-ld)) / monthly_rate
            return round(max(pmt * factor, 0.0), 2)
        except (OverflowError, ValueError):
            logger.warning("Likely Value calculation overflow, returning BAC")
            return round(bac, 2)

    except ValueError as e:
        logger.error(f"Likely value calculation failed: {e}")
        return 0.0


# ============================================================================
# SECTION 3: EVM CALCULATION LAYER (Level 2 - EVM specific)
# ============================================================================

def calculate_pv_linear(bac, current_duration, total_duration) -> float:
    """Calculate planned value (linear) with validation."""
    try:
        bac = validate_numeric_input(bac, "BAC", min_val=0.0)
        current_duration = validate_numeric_input(current_duration, "Current Duration", min_val=0.0)
        total_duration = validate_numeric_input(total_duration, "Total Duration", min_val=0.0)

        if total_duration <= 0:
            return round(bac, 2) if current_duration > 0 else 0.0
        if current_duration >= total_duration:
            return round(bac, 2)

        progress_ratio = max(min(current_duration / total_duration, 1.0), 0.0)
        return round(bac * progress_ratio, 2)

    except ValueError as e:
        logger.error(f"Linear PV calculation failed: {e}")
        return 0.0


def calculate_pv_scurve(bac, current_duration, total_duration, alpha=2.0, beta=2.0) -> float:
    """Calculate planned value (S-curve) with validation."""
    try:
        bac = validate_numeric_input(bac, "BAC", min_val=0.0)
        current_duration = validate_numeric_input(current_duration, "Current Duration", min_val=0.0)
        total_duration = validate_numeric_input(total_duration, "Total Duration", min_val=0.0)

        if total_duration <= 0:
            return round(bac, 2) if current_duration > 0 else 0.0
        if current_duration >= total_duration:
            return round(bac, 2)

        progress_ratio = max(min(current_duration / total_duration, 1.0), 0.0)
        return round(bac * scurve_cdf(progress_ratio, alpha, beta), 2)

    except ValueError as e:
        logger.error(f"S-curve PV calculation failed: {e}")
        return 0.0


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

        # Calculate TCPI (To Complete Performance Index)
        # TCPI = (BAC - EV) / (BAC - AC)
        # Represents the cost performance index needed for remaining work to meet BAC
        work_remaining = bac - earned_value
        budget_remaining = bac - ac

        if budget_remaining == 0:
            # No budget remaining
            if work_remaining == 0:
                tcpi = float('nan')  # 0/0 case - project complete
            else:
                tcpi = float('inf')  # Work remaining but no budget
        else:
            tcpi = safe_divide(work_remaining, budget_remaining, default=0.0)

        return {
            'percent_complete': round(percent_complete * 100, 2),
            'ev': round(earned_value, 2),
            'cv': round(cost_variance, 2),
            'sv': round(schedule_variance, 2),
            'cpi': round(cpi, 3) if is_valid_finite_number(cpi) and not (pd.isna(cpi) or math.isnan(cpi)) else cpi,
            'spi': round(spi, 3) if is_valid_finite_number(spi) and not (pd.isna(spi) or math.isnan(spi)) else spi,
            'tcpi': round(tcpi, 3) if is_valid_finite_number(tcpi) and not (pd.isna(tcpi) or math.isnan(tcpi)) else tcpi,
            'eac': round(eac, 2) if is_valid_finite_number(eac) else float("inf"),
            'etc': round(etc, 2) if is_valid_finite_number(etc) else float("inf"),
            'vac': round(vac, 2) if is_valid_finite_number(vac) else float("-inf"),
        }

    except ValueError as e:
        logger.error(f"EVM metrics calculation failed: {e}")
        return {
            'percent_complete': 0.0, 'ev': 0.0, 'cv': 0.0,
            'sv': 0.0, 'cpi': 0.0, 'spi': 0.0, 'tcpi': 0.0,
            'eac': float("inf"), 'etc': float("inf"), 'vac': float("-inf")
        }


def calculate_earned_schedule_days(
    plan_start, plan_finish, data_date,
    ev, bac,
    curve_type="linear", alpha=2.0, beta=2.0
) -> Dict[str, Union[float, None]]:
    """Calculate Earned Schedule using days-based formula.

    Args:
        plan_start: Planned start date
        plan_finish: Planned finish date
        data_date: Current status date
        ev: Earned Value
        bac: Budget at Completion
        curve_type: "linear" or "s-curve"
        alpha, beta: Beta distribution parameters for s-curve

    Returns:
        Dictionary with es (days), at (days), spie, sv_t (days)
    """
    try:
        from scipy.stats import beta as beta_dist

        # Parse dates
        ps = parse_date_any(plan_start)
        pf = parse_date_any(plan_finish)
        dd = parse_date_any(data_date)

        if not all([ps, pf, dd]):
            logger.error(f"Invalid dates for ES calculation - ps: {ps}, pf: {pf}, dd: {dd}")
            return {'es': 0.0, 'at': 0.0, 'spie': None, 'sv_t': 0.0}

        # Step 1: Time values in DAYS
        pd = (pf - ps).days  # Planned Duration (days)
        at = (dd - ps).days  # Actual Time (days)

        logger.debug(f"Date calculation: PD={pd} days, AT={at} days (ps={ps}, pf={pf}, dd={dd})")

        if pd <= 0:
            logger.error(f"Invalid planned duration: {pd} days (plan_start={ps}, plan_finish={pf})")
            return {'es': 0.0, 'at': at, 'spie': None, 'sv_t': -at}

        if at <= 0:
            logger.warning(f"Data date on or before plan start: AT = {at} days (setting AT=1 to avoid division by zero)")
            at = 1  # Changed from 0 to 1 to allow SPIe calculation

        # Step 2: Earned Schedule (ES) in DAYS
        ev_ratio = max(0.0, min(ev / bac if bac > 0 else 0.0, 1.0))  # Clamp to [0,1]

        logger.debug(f"EV ratio: {ev_ratio} (EV={ev}, BAC={bac})")

        if curve_type.lower() == "linear":
            es = ev_ratio * pd
            logger.debug(f"Linear ES: {es} days")
        else:  # s-curve
            try:
                # Use scipy beta distribution's percent point function (inverse CDF)
                # ppf returns the value x such that CDF(x) = ev_ratio
                if ev_ratio <= 0.0:
                    es = 0.0
                elif ev_ratio >= 1.0:
                    es = pd
                else:
                    # beta_dist.ppf returns value in [0,1], multiply by pd to get days
                    es = beta_dist.ppf(ev_ratio, alpha, beta) * pd
                logger.debug(f"S-curve ES: {es} days (alpha={alpha}, beta={beta})")
            except Exception as e:
                logger.warning(f"S-curve ES calculation failed, using linear: {e}")
                es = ev_ratio * pd

        # Step 3: SPIe and SV(t)
        spie = es / at if at > 0 else 0.0  # Changed: return 0.0 instead of None when AT=0
        sv_t = es - at  # Schedule Variance (time) in days

        logger.debug(f"Final ES metrics: ES={es}, AT={at}, SPIe={spie}, SV(t)={sv_t}")

        return {
            'es': round(es, 2),
            'at': round(at, 2),
            'spie': round(spie, 4) if spie is not None else 0.0,
            'sv_t': round(sv_t, 2)
        }

    except Exception as e:
        logger.error(f"Earned schedule calculation failed: {e}")
        import traceback
        logger.error(f"Traceback: {traceback.format_exc()}")
        return {'es': 0.0, 'at': 0.0, 'spie': 0.0, 'sv_t': 0.0}


def find_earned_schedule_linear(earned_value, bac, total_duration) -> float:
    """DEPRECATED: Use calculate_earned_schedule_days instead.

    Kept for backward compatibility. Returns ES in months.
    """
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
    """DEPRECATED: Use calculate_earned_schedule_days instead.

    Kept for backward compatibility. Returns ES in months.
    """
    try:
        from scipy.stats import beta as beta_dist

        earned_value = validate_numeric_input(earned_value, "Earned Value", min_val=0.0)
        bac = validate_numeric_input(bac, "BAC", min_val=0.01)
        total_duration = validate_numeric_input(total_duration, "Total Duration", min_val=0.0)

        if total_duration <= 0:
            return 0.0

        ev_ratio = max(min(earned_value / bac, 1.0), 0.0)

        if ev_ratio <= 0.0:
            return 0.0
        if ev_ratio >= 1.0:
            return total_duration

        try:
            # Use scipy beta distribution's ppf (inverse CDF)
            es = beta_dist.ppf(ev_ratio, alpha, beta) * total_duration
            return round(max(min(es, total_duration), 0.0), 2)
        except Exception as e:
            logger.warning(f"S-curve calculation failed, using linear: {e}")
            return round(ev_ratio * total_duration, 2)

    except ValueError as e:
        logger.error(f"S-curve earned schedule calculation failed: {e}")
        return 0.0


def calculate_earned_schedule_metrics(earned_schedule, actual_duration, total_duration, plan_start, original_duration=None) -> Dict[str, Union[float, int, str]]:
    """Calculate earned schedule metrics with validation."""
    try:
        earned_schedule = validate_numeric_input(earned_schedule, "Earned Schedule", min_val=0.0)
        actual_duration = max(validate_numeric_input(actual_duration, "Actual Duration", min_val=0.0), 1e-9)
        total_duration = validate_numeric_input(total_duration, "Total Duration", min_val=0.0)
        ps = parse_date_any(plan_start)

        duration_to_use = original_duration if original_duration is not None else total_duration

        # Schedule Performance Index (Earned Schedule)
        spie = safe_divide(earned_schedule, actual_duration, 0.0)

        # Time Variance (Earned Schedule) in months
        tve = earned_schedule - actual_duration

        # Likely Duration calculation with constraint
        likely_duration = safe_calculate_forecast_duration(total_duration, spie, duration_to_use)

        # Calculate likely completion date
        if ps and likely_duration > 0:
            likely_completion = add_months_approx(ps, int(likely_duration))
            likely_completion_str = likely_completion.strftime('%d/%m/%Y')
        else:
            likely_completion_str = "Unknown"

        return {
            'es': round(earned_schedule, 2),
            'spie': round(spie, 3) if is_valid_finite_number(spie) else 0.0,
            'tve': round(tve, 2),
            'ld': round(likely_duration, 2) if is_valid_finite_number(likely_duration) else total_duration,
            'likely_completion': likely_completion_str
        }

    except ValueError as e:
        logger.error(f"Earned schedule metrics calculation failed: {e}")
        return {
            'es': 0.0, 'spie': 0.0, 'tve': 0.0,
            'ld': total_duration, 'likely_completion': 'Unknown'
        }


# ============================================================================
# SECTION 4: ORCHESTRATION LAYER (Level 1 - Main APIs)
# ============================================================================

def perform_complete_evm_analysis(bac, ac, plan_start, plan_finish, data_date,
                                  annual_inflation_rate, curve_type='linear', alpha=2.0, beta=2.0,
                                  manual_pv=None, use_manual_pv=False, manual_ev=None, use_manual_ev=False) -> Dict[str, Any]:
    """Perform complete EVM analysis with comprehensive error handling."""
    try:
        # Sanitize AC: negative values replaced with 0
        try:
            ac = float(ac) if ac is not None else 0.0
            if ac < 0:
                logger.info(f"Negative AC ({ac}) replaced with 0")
                ac = 0.0
        except (ValueError, TypeError):
            logger.warning(f"Invalid AC value '{ac}', using 0")
            ac = 0.0

        # Parse dates once and store the parsed versions
        parsed_plan_start = parse_date_any(plan_start)
        parsed_plan_finish = parse_date_any(plan_finish)
        parsed_data_date = parse_date_any(data_date)

        # Check if project has not started yet:
        # 1. Data date on or before plan start, OR
        # 2. AC = 0 (no actual cost incurred means project hasn't started)
        not_started_by_date = parsed_plan_start and parsed_data_date and parsed_data_date <= parsed_plan_start
        not_started_by_ac = ac == 0.0

        if not_started_by_date or not_started_by_ac:
            reason = []
            if not_started_by_date:
                reason.append(f"data_date ({parsed_data_date}) <= plan_start ({parsed_plan_start})")
            if not_started_by_ac:
                reason.append("AC = 0")
            logger.info(f"Project not started: {'; '.join(reason)}. Skipping EVM calculations.")

            original_duration = calculate_durations(parsed_plan_start, parsed_plan_finish, parsed_data_date)[1] if parsed_plan_start and parsed_plan_finish else 0.0
            return {
                'bac': float(bac),
                'ac': 0.0,  # No actual cost before project starts
                'plan_start': parsed_plan_start.strftime('%Y-%m-%d') if parsed_plan_start else 'Unknown',
                'plan_finish': parsed_plan_finish.strftime('%Y-%m-%d') if parsed_plan_finish else 'Unknown',
                'data_date': parsed_data_date.strftime('%Y-%m-%d') if parsed_data_date else 'Unknown',
                'inflation_rate': round(annual_inflation_rate * 100.0, 3),
                'curve_type': curve_type,
                'alpha': alpha,
                'beta': beta,
                'actual_duration_months': 0.0,
                'original_duration_months': original_duration,
                'present_value': 0.0,
                'pv': 0.0,  # No planned value before start
                'ev': 0.0,  # No earned value before start
                'cv': 0.0,
                'sv': 0.0,
                'cpi': 0.0,
                'spi': 0.0,
                'eac': float(bac),  # Estimate at completion = BAC (no performance data yet)
                'etc': float(bac),  # Estimate to complete = BAC
                'vac': 0.0,
                'tcpi': 0.0,
                'percent_budget_used': 0.0,
                'percent_time_used': 0.0,
                'use_manual_pv': use_manual_pv,
                'manual_pv': manual_pv if use_manual_pv else None,
                'use_manual_ev': use_manual_ev,
                'manual_ev': manual_ev if use_manual_ev else None,
                'planned_value_project': float(bac),
                'likely_value_project': float(bac),
                'percent_present_value_project': 100.0,
                'percent_likely_value_project': 100.0,
                'es': 0.0,
                'es_days': 0.0,
                'at_days': 0.0,
                'spie': 0.0,
                'tve': 0.0,
                'sv_t': 0.0,
                'ld': original_duration,
                'likely_completion': parsed_plan_finish.strftime('%d/%m/%Y') if parsed_plan_finish else 'Unknown',
                'project_status': 'Not Started'
            }

        actual_duration, original_duration = calculate_durations(parsed_plan_start, parsed_plan_finish, parsed_data_date)
        present_value = calculate_present_value(ac, actual_duration, annual_inflation_rate)

        # Use manual PV if specified, otherwise calculate automatically
        if use_manual_pv and manual_pv is not None and is_valid_finite_number(manual_pv):
            planned_value = float(manual_pv)
        else:
            if curve_type.lower() == 's-curve':
                planned_value = calculate_pv_scurve(bac, actual_duration, original_duration, alpha, beta)
            else:
                planned_value = calculate_pv_linear(bac, actual_duration, original_duration)

        evm_metrics = calculate_evm_metrics(bac, ac, present_value, planned_value, manual_ev, use_manual_ev)

        # Calculate Earned Schedule using new days-based formula
        es_days_result = calculate_earned_schedule_days(
            plan_start=parsed_plan_start,
            plan_finish=parsed_plan_finish,
            data_date=parsed_data_date,
            ev=evm_metrics['ev'],
            bac=bac,
            curve_type=curve_type,
            alpha=alpha,
            beta=beta
        )

        # Debug logging
        logger.debug(f"ES days result: {es_days_result}")
        logger.debug(f"EV: {evm_metrics['ev']}, BAC: {bac}, AT: {es_days_result.get('at')}, ES: {es_days_result.get('es')}")

        # Convert days to months for duration calculations (approximate: 30.44 days/month)
        es_months = es_days_result['es'] / 30.44 if es_days_result['es'] else 0.0
        at_months = es_days_result['at'] / 30.44 if es_days_result['at'] else actual_duration

        # Calculate likely duration using SPIe from days-based calculation
        spie = es_days_result.get('spie')
        if spie is None or spie == 0:
            logger.warning(f"SPIe is {spie} - AT: {es_days_result.get('at')} days, ES: {es_days_result.get('es')} days")
        if spie and spie > 0:
            likely_duration = safe_calculate_forecast_duration(original_duration, spie, original_duration)
        else:
            likely_duration = original_duration

        # Calculate likely completion date
        if parsed_plan_start and likely_duration > 0:
            likely_completion = add_months_approx(parsed_plan_start, int(likely_duration))
            likely_completion_str = likely_completion.strftime('%d/%m/%Y')
        else:
            likely_completion_str = "Unknown"

        # Create es_metrics dict with both days and months values
        es_metrics = {
            'es': round(es_months, 2),  # ES in months for compatibility
            'es_days': es_days_result['es'],  # ES in days (new)
            'at_days': es_days_result['at'],  # AT in days (new)
            'spie': round(spie, 4) if spie is not None else 0.0,  # SPIe from days calculation
            'tve': round(es_days_result['sv_t'] / 30.44, 2),  # SV(t) in months for compatibility
            'sv_t': es_days_result['sv_t'],  # SV(t) in days (new)
            'ld': round(likely_duration, 2) if is_valid_finite_number(likely_duration) else original_duration,
            'likely_completion': likely_completion_str
        }

        # Calculate percentage metrics
        percent_budget_used = safe_divide(ac, bac) * 100 if bac > 0 else 0.0
        percent_time_used = safe_divide(actual_duration, original_duration) * 100 if original_duration > 0 else 0.0

        # Calculate new financial metrics
        planned_value_project = calculate_planned_value_of_project(bac, original_duration, annual_inflation_rate)

        # Get likely duration from es_metrics for likely value calculation
        likely_duration = es_metrics.get('ld', original_duration)
        if likely_duration is None or not is_valid_finite_number(likely_duration):
            likely_duration = original_duration
        likely_value_project = calculate_likely_value_of_project(bac, likely_duration, annual_inflation_rate)

        # Calculate percentages
        percent_present_value_project = safe_divide(planned_value_project, bac) * 100 if bac > 0 else 0.0
        percent_likely_value_project = safe_divide(likely_value_project, bac) * 100 if bac > 0 else 0.0

        return {
            'bac': float(bac),
            'ac': float(ac),
            'plan_start': parsed_plan_start.strftime('%Y-%m-%d'),
            'plan_finish': parsed_plan_finish.strftime('%Y-%m-%d'),
            'data_date': parsed_data_date.strftime('%Y-%m-%d'),
            'inflation_rate': round(annual_inflation_rate * 100.0, 3),
            'curve_type': curve_type,
            'alpha': alpha,
            'beta': beta,
            'actual_duration_months': actual_duration,
            'original_duration_months': original_duration,
            'present_value': present_value,
            'pv': planned_value,
            'percent_budget_used': percent_budget_used,
            'percent_time_used': percent_time_used,
            'use_manual_pv': use_manual_pv,
            'manual_pv': manual_pv if use_manual_pv else None,
            'use_manual_ev': use_manual_ev,
            'manual_ev': manual_ev if use_manual_ev else None,
            'planned_value_project': planned_value_project,
            'likely_value_project': likely_value_project,
            'percent_present_value_project': percent_present_value_project,
            'percent_likely_value_project': percent_likely_value_project,
            **evm_metrics,
            **es_metrics
        }

    except Exception as e:
        logger.error(f"Complete EVM analysis failed: {e}")
        raise


def perform_batch_calculation(df: pd.DataFrame, column_mapping: Dict[str, str],
                            curve_type: str, alpha: float, beta: float,
                            data_date: date, inflation_rate: float, mode: str = 'global') -> pd.DataFrame:
    """Perform EVM calculations on entire dataset.

    Args:
        df: DataFrame with project data
        column_mapping: Column name mappings
        curve_type: Global curve type ('linear' or 's-curve')
        alpha: Global alpha parameter
        beta: Global beta parameter
        data_date: Data date for calculations
        inflation_rate: Global inflation rate (percentage)
        mode: Calculation mode - 'global' or 'project'
            - 'global': Use global settings for all, preserve project settings (fill blanks)
            - 'project': Use project settings with fallback to global

    Returns:
        DataFrame with calculation results
    """
    results_list = []

    # Ensure we have a clean copy of the dataframe
    df_clean = df.copy().reset_index(drop=True)

    # Default settings for when global settings don't exist
    DEFAULT_CURVE = 'linear'
    DEFAULT_ALPHA = 2.0
    DEFAULT_BETA = 2.0
    DEFAULT_INFLATION = 0.0

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
                logger.info(f"Stopping CSV import at row {idx} due to blank Project ID")
                break

            bac = float(row[bac_col]) if bac_col in row.index else 0.0
            ac = float(row[ac_col]) if ac_col in row.index else 0.0
            plan_start = row[st_col] if st_col in row.index else None
            plan_finish = row[fn_col] if fn_col in row.index else None

            # Additional project info
            project_name = str(row.get(column_mapping.get('pname_col', ''), project_id))
            organization = str(row.get(column_mapping.get('org_col', ''), ''))
            project_manager = str(row.get(column_mapping.get('pm_col', ''), ''))

            # Handle manual PV/EV values
            manual_pv_val = None
            use_manual_pv = False
            manual_ev_val = None
            use_manual_ev = False

            # Check for manual PV
            if 'pv_col' in column_mapping and column_mapping['pv_col'] in row.index:
                try:
                    manual_pv_val = float(row[column_mapping['pv_col']])
                    use_manual_pv = manual_pv_val > 0
                except (ValueError, TypeError):
                    manual_pv_val = None
                    use_manual_pv = False

            # Check for manual EV
            if 'ev_col' in column_mapping and column_mapping['ev_col'] in row.index:
                try:
                    manual_ev_val = float(row[column_mapping['ev_col']])
                    use_manual_ev = manual_ev_val > 0
                except (ValueError, TypeError):
                    manual_ev_val = None
                    use_manual_ev = False

            # Extract per-project EVM settings (optional columns)
            project_curve = None
            project_alpha = None
            project_beta = None
            project_inflation = None

            if 'curve_type_col' in column_mapping and column_mapping['curve_type_col'] in row.index:
                val = row[column_mapping['curve_type_col']]
                if pd.notna(val) and str(val).strip():
                    project_curve = str(val).strip().lower()

            if 'alpha_col' in column_mapping and column_mapping['alpha_col'] in row.index:
                try:
                    val = float(row[column_mapping['alpha_col']])
                    if is_valid_finite_number(val):
                        project_alpha = val
                except (ValueError, TypeError):
                    pass

            if 'beta_col' in column_mapping and column_mapping['beta_col'] in row.index:
                try:
                    val = float(row[column_mapping['beta_col']])
                    if is_valid_finite_number(val):
                        project_beta = val
                except (ValueError, TypeError):
                    pass

            if 'inflation_rate_col' in column_mapping and column_mapping['inflation_rate_col'] in row.index:
                try:
                    val = float(row[column_mapping['inflation_rate_col']])
                    if is_valid_finite_number(val):
                        project_inflation = val
                except (ValueError, TypeError):
                    pass

            # Determine which settings to use for calculation based on mode
            if mode == 'global':
                # Global Mode: Use global settings for calculation
                calc_curve = curve_type or DEFAULT_CURVE
                calc_alpha = alpha if alpha is not None else DEFAULT_ALPHA
                calc_beta = beta if beta is not None else DEFAULT_BETA
                calc_inflation = inflation_rate if inflation_rate is not None else DEFAULT_INFLATION

                # Fill blank project settings with global (preserve non-blank)
                if project_curve is None:
                    project_curve = calc_curve
                if project_alpha is None:
                    project_alpha = calc_alpha
                if project_beta is None:
                    project_beta = calc_beta
                if project_inflation is None:
                    project_inflation = calc_inflation

            else:  # mode == 'project'
                # Project Mode: Use project settings with fallback to global
                calc_curve = project_curve if project_curve is not None else (curve_type or DEFAULT_CURVE)
                calc_alpha = project_alpha if project_alpha is not None else (alpha if alpha is not None else DEFAULT_ALPHA)
                calc_beta = project_beta if project_beta is not None else (beta if beta is not None else DEFAULT_BETA)
                calc_inflation = project_inflation if project_inflation is not None else (inflation_rate if inflation_rate is not None else DEFAULT_INFLATION)

                # Update project settings with used values (fills blanks)
                project_curve = calc_curve
                project_alpha = calc_alpha
                project_beta = calc_beta
                project_inflation = calc_inflation

            # Clear alpha/beta to 0 for linear curves
            if calc_curve == 'linear':
                calc_alpha = 0
                calc_beta = 0
                project_alpha = 0
                project_beta = 0

            # Perform EVM analysis with determined settings
            results = perform_complete_evm_analysis(
                bac=bac, ac=ac, plan_start=plan_start, plan_finish=plan_finish,
                data_date=data_date, annual_inflation_rate=calc_inflation/100.0,
                curve_type=calc_curve, alpha=calc_alpha, beta=calc_beta,
                manual_pv=manual_pv_val, use_manual_pv=use_manual_pv,
                manual_ev=manual_ev_val, use_manual_ev=use_manual_ev
            )

            # Store the project-specific settings in results (for audit/export)
            results['curve_type'] = project_curve
            results['alpha'] = project_alpha
            results['beta'] = project_beta
            results['inflation_rate'] = project_inflation

            # Add project info to results
            results['project_id'] = project_id
            results['project_name'] = project_name
            results['organization'] = organization
            results['project_manager'] = project_manager
            results['calculation_date'] = datetime.now().isoformat()

            results_list.append(results)

        except Exception as e:
            logger.error(f"Error processing row {idx}: {e}")
            # Add error record
            error_record = {
                'project_id': f"Row_{idx}",
                'project_name': 'Error',
                'error': str(e),
                'calculation_date': datetime.now().isoformat()
            }
            results_list.append(error_record)

    return pd.DataFrame(results_list)


# ============================================================================
# SECTION 5: PUBLIC API
# ============================================================================

__all__ = [
    # Main APIs - These are the primary functions other modules should use
    'perform_batch_calculation',
    'perform_complete_evm_analysis',
    'calculate_evm_metrics',

    # EVM calculation functions
    'calculate_pv_linear',
    'calculate_pv_scurve',
    'calculate_earned_schedule_days',  # New: Days-based ES calculation with scipy
    'find_earned_schedule_linear',  # Deprecated: kept for compatibility
    'find_earned_schedule_scurve',  # Deprecated: kept for compatibility
    'calculate_earned_schedule_metrics',

    # Mathematical functions
    'scurve_cdf',
    'calculate_durations',
    'calculate_present_value',

    # Utility functions
    'safe_divide',
    'safe_financial_metrics',
    'validate_numeric_input',
    'is_valid_finite_number',
    'parse_date_any',
    'add_months_approx',
    'safe_calculate_forecast_duration',
]
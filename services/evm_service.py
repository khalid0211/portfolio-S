"""EVM calculation service module."""

from __future__ import annotations
import logging
import math
from datetime import datetime, timedelta
from typing import Dict, Any, Union

from models.project import ProjectData, AnalysisConfig, ProjectValidator


logger = logging.getLogger(__name__)

# Constants
DAYS_PER_MONTH = 30.44
INTEGRATION_STEPS = 200


class EVMCalculator:
    """Service class for EVM calculations."""

    def __init__(self, config: AnalysisConfig):
        self.config = config
        self.validator = ProjectValidator()

    def calculate_durations(self, plan_start: datetime, plan_finish: datetime, data_date: datetime) -> tuple[float, float]:
        """Calculate durations with improved error handling."""
        try:
            duration_to_date = max(((data_date - plan_start).days / DAYS_PER_MONTH), 0.0)
            original_duration = max(((plan_finish - plan_start).days / DAYS_PER_MONTH), 0.0)

            return round(duration_to_date, 2), round(original_duration, 2)
        except Exception as e:
            logger.error(f"Duration calculation failed: {e}")
            return 0.0, 0.0

    def calculate_present_value(self, ac: float, duration_months: float) -> float:
        """Calculate present value with improved validation."""
        try:
            ac = self.validator.validate_numeric_input(ac, "AC", min_val=0.0)
            duration_months = self.validator.validate_numeric_input(duration_months, "Duration", min_val=0.0)
            annual_rate = self.validator.validate_numeric_input(self.config.annual_inflation_rate, "Inflation Rate", min_val=0.0, max_val=1.0)

            if annual_rate == 0 or duration_months == 0:
                return round(ac, 2)

            monthly_rate = (1 + annual_rate) ** (1/12) - 1
            present_value = ac / ((1 + monthly_rate) ** duration_months)
            return round(present_value, 2)

        except Exception as e:
            logger.error(f"Present value calculation failed: {e}")
            return 0.0

    def calculate_present_value_of_progress(self, ac: float, ad: float) -> float:
        """Calculate Present Value of Progress."""
        try:
            ac = self.validator.validate_numeric_input(ac, "AC", min_val=0.0)
            ad = self.validator.validate_numeric_input(ad, "AD", min_val=0.0)
            annual_rate = self.config.annual_inflation_rate

            if ad == 0:
                return 0.0
            if annual_rate == 0:
                return round(ac, 2)

            monthly_rate = (1 + annual_rate) ** (1/12) - 1
            if monthly_rate == 0:
                return round(ac, 2)

            pv_progress = (ac / ad) * (1 - (1 + monthly_rate) ** (-ad)) / monthly_rate
            return round(pv_progress, 2)

        except Exception as e:
            logger.error(f"Present value of progress calculation failed: {e}")
            return 0.0

    def calculate_planned_value_of_project(self, bac: float, od: float) -> float:
        """Calculate Planned Value of Project."""
        try:
            bac = self.validator.validate_numeric_input(bac, "BAC", min_val=0.0)
            od = self.validator.validate_numeric_input(od, "OD", min_val=0.0)
            annual_rate = self.config.annual_inflation_rate

            if od == 0:
                return 0.0
            if annual_rate == 0:
                return round(bac, 2)

            monthly_rate = (1 + annual_rate) ** (1/12) - 1
            if monthly_rate == 0:
                return round(bac, 2)

            pv_project = (bac / od) * (1 - (1 + monthly_rate) ** (-od)) / monthly_rate
            return round(pv_project, 2)

        except Exception as e:
            logger.error(f"Planned value of project calculation failed: {e}")
            return 0.0

    def calculate_likely_value_of_project(self, bac: float, ld: float) -> float:
        """Calculate Likely Value of Project."""
        try:
            bac = self.validator.validate_numeric_input(bac, "BAC", min_val=0.0)
            ld = self.validator.validate_numeric_input(ld, "LD", min_val=0.0)
            annual_rate = self.config.annual_inflation_rate

            if ld == 0:
                return 0.0
            if annual_rate == 0:
                return round(bac, 2)

            monthly_rate = (1 + annual_rate) ** (1/12) - 1
            if monthly_rate == 0:
                return round(bac, 2)

            lv_project = (bac / ld) * (1 - (1 + monthly_rate) ** (-ld)) / monthly_rate
            return round(lv_project, 2)

        except Exception as e:
            logger.error(f"Likely value of project calculation failed: {e}")
            return 0.0

    def calculate_pv_linear(self, bac: float, current_duration: float, total_duration: float) -> float:
        """Calculate planned value using linear progression."""
        try:
            bac = self.validator.validate_numeric_input(bac, "BAC", min_val=0.0)
            current_duration = self.validator.validate_numeric_input(current_duration, "Current Duration", min_val=0.0)
            total_duration = self.validator.validate_numeric_input(total_duration, "Total Duration", min_val=0.0)

            if total_duration <= 0:
                return round(bac, 2) if current_duration > 0 else 0.0
            if current_duration >= total_duration:
                return round(bac, 2)

            pv = bac * (current_duration / total_duration)
            return round(max(pv, 0.0), 2)

        except Exception as e:
            logger.error(f"Linear PV calculation failed: {e}")
            return 0.0

    def calculate_pv_scurve(self, bac: float, current_duration: float, total_duration: float) -> float:
        """Calculate planned value using S-curve progression."""
        try:
            bac = self.validator.validate_numeric_input(bac, "BAC", min_val=0.0)
            current_duration = self.validator.validate_numeric_input(current_duration, "Current Duration", min_val=0.0)
            total_duration = self.validator.validate_numeric_input(total_duration, "Total Duration", min_val=0.0)

            if total_duration <= 0:
                return round(bac, 2) if current_duration > 0 else 0.0
            if current_duration >= total_duration:
                return round(bac, 2)

            x = current_duration / total_duration
            cdf_value = self._scurve_cdf(x, self.config.alpha, self.config.beta)
            pv = bac * cdf_value
            return round(max(pv, 0.0), 2)

        except Exception as e:
            logger.error(f"S-curve PV calculation failed: {e}")
            return 0.0

    def _scurve_cdf(self, x: float, alpha: float = 2.0, beta: float = 2.0) -> float:
        """Beta distribution CDF using closed-form solution."""
        if x <= 0:
            return 0.0
        if x >= 1:
            return 1.0

        # For Beta(2,2), use closed-form CDF: 3x² - 2x³
        if alpha == 2.0 and beta == 2.0:
            return 3 * x**2 - 2 * x**3

        # Fallback to numerical integration for other parameters
        return self._beta_cdf_numerical(x, alpha, beta)

    def _beta_cdf_numerical(self, x: float, alpha: float, beta: float) -> float:
        """Numerical integration for Beta CDF."""
        step = x / INTEGRATION_STEPS
        integral = 0.0

        for i in range(INTEGRATION_STEPS):
            t = (i + 0.5) * step
            if 0 < t < 1:
                integral += (t**(alpha-1)) * ((1-t)**(beta-1)) * step

        # Approximate normalization constant for Beta(α,β)
        norm_constant = (alpha + beta - 1) / (math.gamma(alpha) * math.gamma(beta) / math.gamma(alpha + beta))
        return min(integral * norm_constant, 1.0)

    def calculate_evm_metrics(self, bac: float, ac: float, present_value: float, planned_value: float,
                            manual_ev: float = None, use_manual_ev: bool = False) -> Dict[str, float]:
        """Calculate core EVM metrics."""
        try:
            bac = self.validator.validate_numeric_input(bac, "BAC", min_val=0.0)
            ac = self.validator.validate_numeric_input(ac, "AC", min_val=0.0)
            present_value = self.validator.validate_numeric_input(present_value, "Present Value", min_val=0.0)
            planned_value = self.validator.validate_numeric_input(planned_value, "Planned Value", min_val=0.0)

            # Use manual EV if specified
            if use_manual_ev and manual_ev is not None and self.validator.is_valid_finite_number(manual_ev):
                earned_value = float(manual_ev)
            else:
                earned_value = present_value

            # Core metrics
            cv = earned_value - ac
            sv = earned_value - planned_value

            # Performance indices
            cpi = self._safe_divide(earned_value, ac, 1.0)
            spi = self._safe_divide(earned_value, planned_value, 1.0)

            # Forecasts
            eac = self._safe_divide(bac, cpi) if cpi > 0 else float('inf')
            etc = max(eac - ac, 0.0)
            vac = bac - eac

            return {
                'earned_value': round(earned_value, 2),
                'cost_variance': round(cv, 2),
                'schedule_variance': round(sv, 2),
                'cost_performance_index': round(cpi, 4),
                'schedule_performance_index': round(spi, 4),
                'estimate_at_completion': round(eac, 2),
                'estimate_to_complete': round(etc, 2),
                'variance_at_completion': round(vac, 2)
            }

        except Exception as e:
            logger.error(f"EVM metrics calculation failed: {e}")
            return {}

    def find_earned_schedule_linear(self, earned_value: float, bac: float, total_duration: float) -> float:
        """Find earned schedule using linear progression."""
        try:
            earned_value = self.validator.validate_numeric_input(earned_value, "Earned Value", min_val=0.0)
            bac = self.validator.validate_numeric_input(bac, "BAC", min_val=0.01)
            total_duration = self.validator.validate_numeric_input(total_duration, "Total Duration", min_val=0.0)

            es = self._safe_divide(earned_value, bac) * total_duration
            return round(max(min(es, total_duration), 0.0), 2)

        except Exception as e:
            logger.error(f"Linear earned schedule calculation failed: {e}")
            return 0.0

    def find_earned_schedule_scurve(self, earned_value: float, bac: float, total_duration: float) -> float:
        """Find earned schedule using S-curve progression."""
        try:
            earned_value = self.validator.validate_numeric_input(earned_value, "Earned Value", min_val=0.0)
            bac = self.validator.validate_numeric_input(bac, "BAC", min_val=0.01)
            total_duration = self.validator.validate_numeric_input(total_duration, "Total Duration", min_val=0.0)

            if total_duration <= 0:
                return 0.0

            target = max(min(earned_value / bac, 1.0), 0.0)

            # Binary search for S-curve inverse
            low, high = 0.0, 1.0
            tolerance = 1e-6
            max_iterations = 100

            for _ in range(max_iterations):
                mid = (low + high) / 2
                cdf_val = self._scurve_cdf(mid, self.config.alpha, self.config.beta)

                if abs(cdf_val - target) < tolerance:
                    return round(mid * total_duration, 2)
                elif cdf_val < target:
                    low = mid
                else:
                    high = mid

            return round(((low + high) / 2) * total_duration, 2)

        except Exception as e:
            logger.error(f"S-curve earned schedule calculation failed: {e}")
            return 0.0

    def calculate_earned_schedule_metrics(self, earned_schedule: float, actual_duration: float,
                                        total_duration: float, plan_start: datetime,
                                        original_duration: float = None) -> Dict[str, Union[float, str]]:
        """Calculate earned schedule metrics."""
        try:
            earned_schedule = self.validator.validate_numeric_input(earned_schedule, "Earned Schedule", min_val=0.0)
            actual_duration = max(self.validator.validate_numeric_input(actual_duration, "Actual Duration", min_val=0.0), 1e-9)
            total_duration = self.validator.validate_numeric_input(total_duration, "Total Duration", min_val=0.0)

            spie = self._safe_divide(earned_schedule, actual_duration, 1.0)
            forecast_duration = self._safe_calculate_forecast_duration(total_duration, spie, original_duration)

            # Time metrics
            sv_time = earned_schedule - actual_duration

            # Forecast dates
            forecast_months = forecast_duration if self.validator.is_valid_finite_number(forecast_duration) else total_duration
            forecast_completion = self._add_months_approx(plan_start, int(forecast_months))

            return {
                'earned_schedule': round(earned_schedule, 2),
                'schedule_performance_index_time': round(spie, 4),
                'schedule_variance_time': round(sv_time, 2),
                'forecast_duration': round(forecast_duration, 2),
                'forecast_completion': forecast_completion.strftime('%Y-%m-%d')
            }

        except Exception as e:
            logger.error(f"Earned schedule metrics calculation failed: {e}")
            return {}

    def perform_complete_analysis(self, project_data: ProjectData) -> Dict[str, Any]:
        """Perform complete EVM analysis."""
        try:
            # Calculate durations
            actual_duration, original_duration = self.calculate_durations(
                project_data.plan_start, project_data.plan_finish, project_data.data_date
            )

            # Calculate present value
            present_value = self.calculate_present_value(project_data.ac, actual_duration)

            # Calculate planned value
            if project_data.use_manual_pv and project_data.manual_pv is not None:
                planned_value = float(project_data.manual_pv)
            else:
                if self.config.curve_type.lower() == 's-curve':
                    planned_value = self.calculate_pv_scurve(project_data.bac, actual_duration, original_duration)
                else:
                    planned_value = self.calculate_pv_linear(project_data.bac, actual_duration, original_duration)

            # Calculate EVM metrics
            evm_metrics = self.calculate_evm_metrics(
                project_data.bac, project_data.ac, present_value, planned_value,
                project_data.manual_ev, project_data.use_manual_ev
            )

            # Calculate earned schedule
            if self.config.curve_type.lower() == 's-curve':
                earned_schedule = self.find_earned_schedule_scurve(
                    evm_metrics.get('earned_value', 0), project_data.bac, original_duration
                )
            else:
                earned_schedule = self.find_earned_schedule_linear(
                    evm_metrics.get('earned_value', 0), project_data.bac, original_duration
                )

            # Calculate earned schedule metrics
            es_metrics = self.calculate_earned_schedule_metrics(
                earned_schedule, actual_duration, original_duration, project_data.plan_start, original_duration
            )

            # Calculate additional financial metrics
            planned_value_project = self.calculate_planned_value_of_project(project_data.bac, original_duration)
            likely_duration = es_metrics.get('forecast_duration', original_duration)
            likely_value_project = self.calculate_likely_value_of_project(project_data.bac, likely_duration)

            # Calculate percentages
            percent_budget_used = self._safe_divide(project_data.ac, project_data.bac) * 100
            percent_time_used = self._safe_divide(actual_duration, original_duration) * 100

            return {
                **project_data.to_dict(),
                'actual_duration_months': actual_duration,
                'original_duration_months': original_duration,
                'present_value': present_value,
                'planned_value': planned_value,
                'planned_value_project': planned_value_project,
                'likely_value_project': likely_value_project,
                'percent_budget_used': percent_budget_used,
                'percent_time_used': percent_time_used,
                'inflation_rate': round(self.config.annual_inflation_rate * 100.0, 3),
                'curve_type': self.config.curve_type,
                **evm_metrics,
                **es_metrics
            }

        except Exception as e:
            logger.error(f"Complete EVM analysis failed: {e}")
            return {'error': str(e)}

    def _safe_divide(self, numerator: float, denominator: float, default: float = 0.0) -> float:
        """Safe division with default value."""
        try:
            if denominator == 0 or not self.validator.is_valid_finite_number(denominator):
                return default
            result = numerator / denominator
            return result if self.validator.is_valid_finite_number(result) else default
        except:
            return default

    def _safe_calculate_forecast_duration(self, total_duration: float, spie: float, original_duration: float = None) -> float:
        """Safe forecast duration calculation."""
        try:
            if spie <= 0 or not self.validator.is_valid_finite_number(spie):
                return original_duration if original_duration else total_duration

            forecast = total_duration / spie

            if not self.validator.is_valid_finite_number(forecast):
                return original_duration if original_duration else total_duration

            return max(forecast, 0.0)

        except:
            return original_duration if original_duration else total_duration

    def _add_months_approx(self, start_dt: datetime, months: int) -> datetime:
        """Add months to datetime with approximation."""
        try:
            days_to_add = int(months * DAYS_PER_MONTH)
            return start_dt + timedelta(days=days_to_add)
        except:
            return start_dt
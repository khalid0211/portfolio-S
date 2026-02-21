"""Formatting service for EVM data display."""

from __future__ import annotations
import math
import pandas as pd
from typing import Any, Union
from datetime import datetime

from models.project import ProjectValidator


class FormattingService:
    """Service for formatting EVM data for display."""

    def __init__(self, currency_symbol: str = "$", currency_postfix: str = "", date_format: str = "YYYY-MM-DD"):
        self.currency_symbol = currency_symbol
        self.currency_postfix = currency_postfix
        self.date_format = date_format
        self.validator = ProjectValidator()

    def format_currency(self, amount: float, decimals: int = 2) -> str:
        """Enhanced currency formatting with comma separators and postfix options."""
        if not self.validator.is_valid_finite_number(amount):
            return "—"

        try:
            # Handle postfix labels (no scaling - user's figures are already in specified units)
            postfix_label = ""
            if self.currency_postfix.lower() == "thousand":
                postfix_label = "K"
            elif self.currency_postfix.lower() == "million":
                postfix_label = "M"
            elif self.currency_postfix.lower() == "billion":
                postfix_label = "B"

            # Format with commas and specified decimals
            if decimals == 0:
                formatted_amount = f"{amount:,.0f}"
            else:
                formatted_amount = f"{amount:,.{decimals}f}"

            # Construct final string
            result = f"{self.currency_symbol}{formatted_amount}"
            if postfix_label:
                result += f" {postfix_label}"

            return result

        except Exception:
            return "—"

    def format_percentage(self, value: float, decimals: int = 2) -> str:
        """Format percentage values consistently."""
        if not self.validator.is_valid_finite_number(value):
            return "—"
        return f"{value:.{decimals}f}%"

    def format_performance_index(self, value: float, decimals: int = 2) -> str:
        """Format performance indices (CPI, SPI, SPIe) consistently."""
        if not self.validator.is_valid_finite_number(value):
            return "N/A"
        return f"{value:.{decimals}f}"

    def format_duration(self, value: float, unit: str = "months") -> str:
        """Format duration values as rounded integers."""
        if not self.validator.is_valid_finite_number(value):
            return "—"
        return f"{int(round(value))} {unit}"

    def format_date(self, date_input: Union[str, datetime], output_format: str = None) -> str:
        """Format date to specified format."""
        if output_format is None:
            output_format = '%d-%m-%Y' if self.date_format == 'DD-MM-YYYY' else '%Y-%m-%d'

        try:
            if isinstance(date_input, str):
                if date_input in ['N/A', ''] or not date_input:
                    return "N/A"
                # Parse the date string
                from core.evm_engine import parse_date_any
                parsed_date = parse_date_any(date_input)
            elif isinstance(date_input, datetime):
                parsed_date = date_input
            else:
                return "N/A"

            return parsed_date.strftime(output_format)

        except Exception:
            return str(date_input) if date_input else "N/A"

    def format_financial_metric(self, value: float, decimals: int = 3, as_percentage: bool = False) -> str:
        """Format financial metrics, displaying NaN as 'N/A' and handling special cases."""
        try:
            if pd.isna(value) or math.isnan(value):
                return "N/A"
            if math.isinf(value):
                return "∞"

            if as_percentage:
                return self.format_percentage(value * 100, decimals)
            else:
                return f"{value:.{decimals}f}"

        except Exception:
            return "N/A"

    def maybe(self, val: Any, default: str = "—") -> str:
        """Return default if value is None or invalid."""
        if val is None:
            return default
        if isinstance(val, (int, float)) and not self.validator.is_valid_finite_number(val):
            return default
        return str(val)

    def format_batch_results_for_display(self, batch_df: pd.DataFrame) -> pd.DataFrame:
        """Format batch results DataFrame for display with proper formatting."""
        if batch_df.empty:
            return batch_df

        # Create a copy for display formatting
        display_df = batch_df.copy()

        # Format currency columns
        currency_cols = [
            'bac', 'ac', 'planned_value', 'earned_value', 'present_value',
            'planned_value_project', 'likely_value_project', 'estimate_to_complete', 'estimate_at_completion'
        ]
        for col in currency_cols:
            if col in display_df.columns:
                display_df[col] = display_df[col].apply(lambda x: self.format_currency(x) if pd.notna(x) else "—")

        # Format percentage columns
        percentage_cols = [
            'percent_budget_used', 'percent_time_used', 'percent_present_value_project',
            'percent_likely_value_project', 'inflation_rate'
        ]
        for col in percentage_cols:
            if col in display_df.columns:
                display_df[col] = display_df[col].apply(lambda x: self.format_percentage(x) if pd.notna(x) else "—")

        # Format variance columns (can be negative)
        variance_cols = ['cost_variance', 'schedule_variance', 'variance_at_completion', 'schedule_variance_time']
        for col in variance_cols:
            if col in display_df.columns:
                display_df[col] = display_df[col].apply(lambda x: self.format_currency(x) if pd.notna(x) else "—")

        # Format performance index columns
        performance_cols = ['cost_performance_index', 'schedule_performance_index', 'schedule_performance_index_time']
        for col in performance_cols:
            if col in display_df.columns:
                display_df[col] = display_df[col].apply(lambda x: self.format_performance_index(x) if pd.notna(x) else "N/A")

        # Format duration columns
        duration_cols = [
            'actual_duration_months', 'original_duration_months', 'earned_schedule', 'forecast_duration'
        ]
        for col in duration_cols:
            if col in display_df.columns:
                display_df[col] = display_df[col].apply(lambda x: self.format_duration(x) if pd.notna(x) else "—")

        # Format date columns
        date_cols = ['plan_start', 'plan_finish', 'data_date', 'forecast_completion']
        for col in date_cols:
            if col in display_df.columns:
                display_df[col] = display_df[col].apply(lambda x: self.format_date(x) if pd.notna(x) else "N/A")

        return display_df

    def format_batch_results_for_download(self, batch_df: pd.DataFrame) -> pd.DataFrame:
        """Reorganize batch results DataFrame with requested column order and minimal formatting."""
        if batch_df.empty:
            return batch_df

        # Create a new DataFrame with required columns in specified order
        formatted_df = pd.DataFrame()

        # Column mapping for download format
        column_mapping = {
            'Project ID': 'project_id',
            'Project Name': 'project_name',
            'Organization': 'organization',
            'Project Manager': 'project_manager',
            'Budget': 'bac',
            'Plan Start': 'plan_start',
            'Plan Finish': 'plan_finish',
            'Data Date': 'data_date',
            'Actual Cost': 'ac',
            'Earned Value': 'earned_value',
            'Planned Value': 'planned_value',
            'Present Value': 'present_value',
            'Cost Variance': 'cost_variance',
            'Schedule Variance': 'schedule_variance',
            'CPI': 'cost_performance_index',
            'SPI': 'schedule_performance_index',
            'EAC': 'estimate_at_completion',
            'ETC': 'estimate_to_complete',
            'VAC': 'variance_at_completion',
            'Earned Schedule': 'earned_schedule',
            'SPIe': 'schedule_performance_index_time',
            'Schedule Variance (Time)': 'schedule_variance_time',
            'Forecast Duration': 'forecast_duration',
            'Forecast Completion': 'forecast_completion',
            'Actual Duration': 'actual_duration_months',
            'Original Duration': 'original_duration_months',
            'Budget Used %': 'percent_budget_used',
            'Time Used %': 'percent_time_used',
            'Planned Value of Project': 'planned_value_project',
            'Likely Value of Project': 'likely_value_project',
            'Present Value Project %': 'percent_present_value_project',
            'Likely Value Project %': 'percent_likely_value_project',
            'Inflation Rate %': 'inflation_rate',
            'Curve Type': 'curve_type'
        }

        # Add columns in the specified order
        for display_name, source_col in column_mapping.items():
            if source_col in batch_df.columns:
                formatted_df[display_name] = batch_df[source_col]
            else:
                formatted_df[display_name] = "N/A"

        return formatted_df

    def build_enhanced_results_table(self, results: dict, controls: dict, project_data: dict) -> pd.DataFrame:
        """Build enhanced results table for single project display."""
        try:
            # Core project information
            data = [
                ["Project ID", self.maybe(project_data.get('project_id', 'N/A'))],
                ["Project Name", self.maybe(project_data.get('project_name', 'N/A'))],
                ["Organization", self.maybe(project_data.get('organization', 'N/A'))],
                ["Project Manager", self.maybe(project_data.get('project_manager', 'N/A'))],
                ["", ""],  # Separator

                # Financial metrics
                ["📊 FINANCIAL METRICS", ""],
                ["Budget at Completion (BAC)", self.format_currency(results.get('bac', 0))],
                ["Actual Cost (AC)", self.format_currency(results.get('ac', 0))],
                ["Earned Value (EV)", self.format_currency(results.get('earned_value', 0))],
                ["Planned Value (PV)", self.format_currency(results.get('planned_value', 0))],
                ["Present Value", self.format_currency(results.get('present_value', 0))],
                ["", ""],

                # Performance indices
                ["📈 PERFORMANCE INDICES", ""],
                ["Cost Performance Index (CPI)", self.format_performance_index(results.get('cost_performance_index', 0))],
                ["Schedule Performance Index (SPI)", self.format_performance_index(results.get('schedule_performance_index', 0))],
                ["Schedule Performance Index (SPIe)", self.format_performance_index(results.get('schedule_performance_index_time', 0))],
                ["", ""],

                # Variances
                ["📊 VARIANCES", ""],
                ["Cost Variance (CV)", self.format_currency(results.get('cost_variance', 0))],
                ["Schedule Variance (SV)", self.format_currency(results.get('schedule_variance', 0))],
                ["Schedule Variance Time (SVt)", self.format_duration(results.get('schedule_variance_time', 0))],
                ["", ""],

                # Forecasts
                ["🔮 FORECASTS", ""],
                ["Estimate at Completion (EAC)", self.format_currency(results.get('estimate_at_completion', 0))],
                ["Estimate to Complete (ETC)", self.format_currency(results.get('estimate_to_complete', 0))],
                ["Variance at Completion (VAC)", self.format_currency(results.get('variance_at_completion', 0))],
                ["Forecast Duration", self.format_duration(results.get('forecast_duration', 0))],
                ["Forecast Completion", self.format_date(results.get('forecast_completion', 'N/A'))],
                ["", ""],

                # Time metrics
                ["⏱️ TIME METRICS", ""],
                ["Earned Schedule", self.format_duration(results.get('earned_schedule', 0))],
                ["Actual Duration", self.format_duration(results.get('actual_duration_months', 0))],
                ["Original Duration", self.format_duration(results.get('original_duration_months', 0))],
                ["", ""],

                # Advanced financial metrics
                ["💰 ADVANCED FINANCIAL", ""],
                ["Planned Value of Project", self.format_currency(results.get('planned_value_project', 0))],
                ["Likely Value of Project", self.format_currency(results.get('likely_value_project', 0))],
                ["Present Value Project %", self.format_percentage(results.get('percent_present_value_project', 0))],
                ["Likely Value Project %", self.format_percentage(results.get('percent_likely_value_project', 0))],
                ["", ""],

                # Progress indicators
                ["📈 PROGRESS INDICATORS", ""],
                ["Budget Used %", self.format_percentage(results.get('percent_budget_used', 0))],
                ["Time Used %", self.format_percentage(results.get('percent_time_used', 0))],
                ["", ""],

                # Configuration
                ["⚙️ ANALYSIS CONFIG", ""],
                ["Curve Type", results.get('curve_type', 'linear').title()],
                ["Inflation Rate", f"{results.get('inflation_rate', 0):.1f}%"],
                ["Data Date", self.format_date(results.get('data_date', 'N/A'))],
            ]

            return pd.DataFrame(data, columns=["Metric", "Value"])

        except Exception as e:
            # Return minimal table on error
            return pd.DataFrame([["Error", str(e)]], columns=["Metric", "Value"])
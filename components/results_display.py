"""Results display UI component."""

import streamlit as st
import pandas as pd
from typing import Dict, Any
import logging

from services.formatting_service import FormattingService
from models.project import AnalysisConfig


logger = logging.getLogger(__name__)


class ResultsDisplayComponent:
    """Component for displaying EVM analysis results."""

    def __init__(self):
        self.formatter = None

    def render(self, results: Dict[str, Any], config: AnalysisConfig, project_data: Dict[str, Any]):
        """Render the results display."""
        if not results or 'error' in results:
            if results and 'error' in results:
                st.error(f"Analysis Error: {results['error']}")
            else:
                st.info("No analysis results to display. Please run analysis first.")
            return

        # Initialize formatter with current config
        self.formatter = FormattingService(
            currency_symbol=config.currency_symbol,
            currency_postfix=config.currency_postfix,
            date_format=config.date_format
        )

        # Create tabs for different views
        tab1, tab2, tab3 = st.tabs(["📊 Summary", "📋 Detailed Results", "📈 Performance Indicators"])

        with tab1:
            self._render_summary_tab(results, config, project_data)

        with tab2:
            self._render_detailed_tab(results, config, project_data)

        with tab3:
            self._render_performance_tab(results, config)

    def _render_summary_tab(self, results: Dict, config: AnalysisConfig, project_data: Dict):
        """Render summary dashboard."""
        st.markdown("## 📊 Project Performance Summary")

        # Key metrics in columns
        col1, col2, col3, col4 = st.columns(4)

        with col1:
            cpi = results.get('cost_performance_index', 0)
            cpi_color = "normal" if 0.9 <= cpi <= 1.1 else "inverse" if cpi < 0.9 else "off"
            st.metric(
                label="Cost Performance Index",
                value=self.formatter.format_performance_index(cpi),
                delta=f"{((cpi - 1) * 100):+.1f}%" if cpi else None
            )

        with col2:
            spi = results.get('schedule_performance_index', 0)
            spi_color = "normal" if 0.9 <= spi <= 1.1 else "inverse" if spi < 0.9 else "off"
            st.metric(
                label="Schedule Performance Index",
                value=self.formatter.format_performance_index(spi),
                delta=f"{((spi - 1) * 100):+.1f}%" if spi else None
            )

        with col3:
            budget_used = results.get('percent_budget_used', 0)
            st.metric(
                label="Budget Used",
                value=self.formatter.format_percentage(budget_used),
                delta=f"{budget_used - 50:+.1f}% vs 50%" if budget_used else None
            )

        with col4:
            time_used = results.get('percent_time_used', 0)
            st.metric(
                label="Time Used",
                value=self.formatter.format_percentage(time_used),
                delta=f"{time_used - 50:+.1f}% vs 50%" if time_used else None
            )

        # Progress bars
        st.markdown("### Progress Overview")
        col1, col2 = st.columns(2)

        with col1:
            budget_progress = min(budget_used / 100, 1.0) if budget_used else 0
            st.markdown("**Budget Progress**")
            st.progress(budget_progress)
            st.caption(f"{self.formatter.format_currency(results.get('ac', 0))} of {self.formatter.format_currency(results.get('bac', 0))}")

        with col2:
            time_progress = min(time_used / 100, 1.0) if time_used else 0
            st.markdown("**Time Progress**")
            st.progress(time_progress)
            st.caption(f"{self.formatter.format_duration(results.get('actual_duration_months', 0))} of {self.formatter.format_duration(results.get('original_duration_months', 0))}")

        # Health indicators
        st.markdown("### Project Health")
        col1, col2, col3 = st.columns(3)

        with col1:
            cv = results.get('cost_variance', 0)
            cv_status = "🟢 Under Budget" if cv > 0 else "🔴 Over Budget" if cv < 0 else "🟡 On Budget"
            st.metric("Cost Variance", self.formatter.format_currency(cv), cv_status)

        with col2:
            sv = results.get('schedule_variance', 0)
            sv_status = "🟢 Ahead" if sv > 0 else "🔴 Behind" if sv < 0 else "🟡 On Schedule"
            st.metric("Schedule Variance", self.formatter.format_currency(sv), sv_status)

        with col3:
            eac = results.get('estimate_at_completion', 0)
            bac = results.get('bac', 0)
            eac_status = "🟢 Under BAC" if eac < bac else "🔴 Over BAC" if eac > bac else "🟡 At BAC"
            st.metric("Estimate at Completion", self.formatter.format_currency(eac), eac_status)

    def _render_detailed_tab(self, results: Dict, config: AnalysisConfig, project_data: Dict):
        """Render detailed results table."""
        st.markdown("## 📋 Detailed Analysis Results")

        # Build comprehensive results table
        results_df = self.formatter.build_enhanced_results_table(results, config.__dict__, project_data)

        # Display table
        st.dataframe(
            results_df,
            width="stretch",
            hide_index=True,
            column_config={
                "Metric": st.column_config.TextColumn("Metric", width="medium"),
                "Value": st.column_config.TextColumn("Value", width="medium")
            }
        )

        # Export options
        st.markdown("### Export Options")
        col1, col2 = st.columns(2)

        with col1:
            # CSV export
            csv_data = results_df.to_csv(index=False)
            st.download_button(
                label="📄 Download as CSV",
                data=csv_data,
                file_name=f"evm_analysis_{project_data.get('project_id', 'project')}.csv",
                mime="text/csv"
            )

        with col2:
            # JSON export
            json_data = pd.DataFrame([results]).to_json(orient='records', indent=2)
            st.download_button(
                label="📄 Download as JSON",
                data=json_data,
                file_name=f"evm_analysis_{project_data.get('project_id', 'project')}.json",
                mime="application/json"
            )

    def _render_performance_tab(self, results: Dict, config: AnalysisConfig):
        """Render performance indicators and forecasts."""
        st.markdown("## 📈 Performance Indicators & Forecasts")

        # Performance indices comparison
        st.markdown("### Performance Indices")
        performance_data = {
            'Index': ['Cost Performance Index (CPI)', 'Schedule Performance Index (SPI)', 'Schedule Performance Index Time (SPIe)'],
            'Value': [
                results.get('cost_performance_index', 0),
                results.get('schedule_performance_index', 0),
                results.get('schedule_performance_index_time', 0)
            ],
            'Status': []
        }

        for value in performance_data['Value']:
            if value >= 1.1:
                status = "🟢 Excellent"
            elif value >= 0.9:
                status = "🟡 Acceptable"
            else:
                status = "🔴 Poor"
            performance_data['Status'].append(status)

        performance_df = pd.DataFrame(performance_data)
        performance_df['Value'] = performance_df['Value'].apply(self.formatter.format_performance_index)

        st.dataframe(
            performance_df,
            width="stretch",
            hide_index=True
        )

        # Forecasts
        st.markdown("### Forecasts")
        col1, col2 = st.columns(2)

        with col1:
            st.markdown("**Cost Forecasts**")
            eac = results.get('estimate_at_completion', 0)
            etc = results.get('estimate_to_complete', 0)
            vac = results.get('variance_at_completion', 0)

            st.metric("Estimate at Completion (EAC)", self.formatter.format_currency(eac))
            st.metric("Estimate to Complete (ETC)", self.formatter.format_currency(etc))
            st.metric("Variance at Completion (VAC)", self.formatter.format_currency(vac))

        with col2:
            st.markdown("**Schedule Forecasts**")
            forecast_duration = results.get('forecast_duration', 0)
            forecast_completion = results.get('forecast_completion', 'N/A')
            original_duration = results.get('original_duration_months', 0)

            st.metric("Forecast Duration", self.formatter.format_duration(forecast_duration))
            st.metric("Forecast Completion", self.formatter.format_date(forecast_completion))

            if forecast_duration and original_duration:
                schedule_variance_months = forecast_duration - original_duration
                st.metric(
                    "Schedule Impact",
                    self.formatter.format_duration(abs(schedule_variance_months)),
                    f"{'Late' if schedule_variance_months > 0 else 'Early' if schedule_variance_months < 0 else 'On Time'}"
                )

        # Financial analysis
        st.markdown("### Advanced Financial Analysis")
        col1, col2 = st.columns(2)

        with col1:
            st.markdown("**Present Value Analysis**")
            present_value = results.get('present_value', 0)
            planned_value_project = results.get('planned_value_project', 0)

            st.metric("Present Value", self.formatter.format_currency(present_value))
            st.metric("Planned Value of Project", self.formatter.format_currency(planned_value_project))

        with col2:
            st.markdown("**Value Percentages**")
            pv_percent = results.get('percent_present_value_project', 0)
            lv_percent = results.get('percent_likely_value_project', 0)

            st.metric("Present Value %", self.formatter.format_percentage(pv_percent))
            st.metric("Likely Value %", self.formatter.format_percentage(lv_percent))


# Global instance
results_display_component = ResultsDisplayComponent()
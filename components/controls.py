"""Controls UI component for EVM analysis configuration."""

import streamlit as st
from typing import Dict, Any

from models.project import AnalysisConfig


class ControlsComponent:
    """Component for EVM analysis configuration controls."""

    def render(self, current_config: Dict[str, Any] = None) -> AnalysisConfig:
        """Render the controls section and return configuration."""
        st.markdown('<div class="sidebar-section">', unsafe_allow_html=True)
        st.markdown('<div class="section-header">C. Analysis Controls</div>', unsafe_allow_html=True)

        # Get current config or use defaults
        if current_config is None:
            current_config = {}

        # Curve type selection
        curve_type = st.selectbox(
            "Progress Curve",
            options=['linear', 's-curve'],
            index=0 if current_config.get('curve_type', 'linear') == 'linear' else 1,
            key="curve_type_single",
            help="Linear: Steady progress. S-curve: Slow start, rapid middle, slow finish"
        )

        # S-curve parameters (only show if S-curve selected)
        alpha = 2.0
        beta = 2.0
        if curve_type == 's-curve':
            st.markdown("**S-curve Parameters:**")
            alpha = st.slider(
                "Alpha (α)",
                min_value=0.5,
                max_value=5.0,
                value=current_config.get('alpha', 2.0),
                step=0.1,
                key="alpha_single",
                help="Controls early project acceleration"
            )
            beta = st.slider(
                "Beta (β)",
                min_value=0.5,
                max_value=5.0,
                value=current_config.get('beta', 2.0),
                step=0.1,
                key="beta_single",
                help="Controls late project deceleration"
            )

        # Currency settings
        st.markdown("**Currency Settings:**")
        col1, col2 = st.columns(2)
        with col1:
            currency_symbol = st.text_input(
                "Symbol",
                value=current_config.get('currency_symbol', '$'),
                key="currency_symbol_single",
                help="Currency symbol to display"
            )
        with col2:
            currency_postfix = st.selectbox(
                "Scale",
                options=['', 'Thousand', 'Million', 'Billion'],
                index=0,
                key="currency_postfix_single",
                help="Unit scale for amounts"
            )

        # Date format
        date_format = st.selectbox(
            "Date Format",
            options=['YYYY-MM-DD', 'DD-MM-YYYY'],
            index=0 if current_config.get('date_format', 'YYYY-MM-DD') == 'YYYY-MM-DD' else 1,
            key="date_format_single"
        )

        # Inflation rate
        current_rate = current_config.get('annual_inflation_rate', 0.03)
        # Handle both decimal (0.03) and percentage (3.0) formats
        display_rate = current_rate * 100 if current_rate <= 1.0 else current_rate

        inflation_rate = st.number_input(
            "Annual Inflation Rate (%)",
            min_value=0.0,
            max_value=100.0,
            value=min(display_rate, 100.0),  # Ensure it doesn't exceed 100
            step=0.1,
            key="inflation_rate_single",
            help="Used for present value calculations"
        ) / 100.0

        st.markdown('</div>', unsafe_allow_html=True)

        return AnalysisConfig(
            curve_type=curve_type,
            alpha=alpha,
            beta=beta,
            currency_symbol=currency_symbol,
            currency_postfix=currency_postfix,
            date_format=date_format,
            annual_inflation_rate=inflation_rate
        )


# Global instance
controls_component = ControlsComponent()
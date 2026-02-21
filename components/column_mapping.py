"""Column mapping UI component."""

import streamlit as st
from typing import Dict, List

from models.project import ColumnMapping
from services.data_service import data_manager


class ColumnMappingComponent:
    """Component for mapping DataFrame columns to expected project fields."""

    def __init__(self):
        self.data_manager = data_manager

    def render(self, columns: List[str], stored_mapping: Dict = None) -> Dict[str, str]:
        """Render column mapping interface."""
        st.markdown('<div class="sidebar-section">', unsafe_allow_html=True)
        st.markdown('<div class="section-header">B. Column Mapping</div>', unsafe_allow_html=True)

        if not columns:
            st.info("No data loaded. Please select a data source first.")
            st.markdown('</div>', unsafe_allow_html=True)
            return {}

        # Load stored mapping or use defaults
        if stored_mapping is None:
            stored_mapping = ColumnMapping.DEFAULT_MAPPING.copy()

        mapping = {}

        st.markdown("**Map your columns to expected fields:**")

        # Core required fields
        st.markdown("*Required Fields:*")
        for key, default_name in [
            ('pid_col', 'Project ID'),
            ('pname_col', 'Project'),
            ('bac_col', 'BAC'),
            ('ac_col', 'AC'),
            ('st_col', 'Plan Start'),
            ('fn_col', 'Plan Finish'),
            ('dd_col', 'Data Date')
        ]:
            current_value = stored_mapping.get(key, default_name)

            # Add "-- Not Available --" option for missing columns
            options = ['-- Not Available --'] + columns

            if current_value in columns:
                default_index = columns.index(current_value) + 1  # +1 for "Not Available" option
            else:
                default_index = 0  # "Not Available"

            selected = st.selectbox(
                f"{default_name}:",
                options=options,
                index=default_index,
                key=f"mapping_{key}_single"
            )

            # Only add to mapping if not "Not Available"
            if selected != '-- Not Available --':
                mapping[key] = selected

        # Optional fields
        with st.expander("Optional Fields"):
            for key, default_name in [
                ('org_col', 'Organization'),
                ('pm_col', 'Project Manager'),
                ('ev_col', 'EV'),
                ('pv_col', 'PV')
            ]:
                current_value = stored_mapping.get(key, default_name)
                options = ['-- Not Available --'] + columns

                if current_value in columns:
                    default_index = columns.index(current_value) + 1
                else:
                    default_index = 0

                selected = st.selectbox(
                    f"{default_name}:",
                    options=options,
                    index=default_index,
                    key=f"mapping_{key}_single"
                )

                if selected != '-- Not Available --':
                    mapping[key] = selected

        # Validation
        missing_columns = ColumnMapping.validate_mapping(mapping, columns)
        if missing_columns:
            st.error(f"⚠️ Missing required columns: {', '.join(missing_columns)}")
        else:
            st.success("✅ All required columns mapped")

        # Save mapping button
        if st.button("Save Column Mapping", key="save_mapping_single"):
            if hasattr(st.session_state, 'selected_table_single') and st.session_state.selected_table_single:
                self.data_manager.save_column_mapping(st.session_state.selected_table_single, mapping)
                st.success("Column mapping saved!")

        st.markdown('</div>', unsafe_allow_html=True)
        return mapping


# Global instance
column_mapping_component = ColumnMappingComponent()
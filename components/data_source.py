"""Data source UI component."""

import streamlit as st
import pandas as pd
from typing import Dict, Any
import logging

from services.data_service import data_manager
from models.project import ColumnMapping


logger = logging.getLogger(__name__)


class DataSourceComponent:
    """Component for handling data source selection and file uploads."""

    def __init__(self):
        self.data_manager = data_manager

    def render(self) -> Dict[str, Any]:
        """Render the data source section."""
        st.markdown('<div class="sidebar-section">', unsafe_allow_html=True)
        st.markdown('<div class="section-header">A. Data Source</div>', unsafe_allow_html=True)

        # Available data tables
        available_tables = self.data_manager.list_session_tables()

        if available_tables:
            selected_table = st.selectbox(
                "Available Data:",
                options=available_tables,
                key="selected_table_single"
            )
        else:
            st.info("No data loaded. Upload a file or create demo data.")
            selected_table = None

        # File upload section
        st.markdown("**Upload New Data:**")
        uploaded_file = st.file_uploader(
            "Choose file",
            type=['csv', 'json'],
            key="single_analysis_upload"
        )

        # Demo data option
        if st.button("Create Demo Data", key="create_demo_single"):
            self.data_manager.create_demo_data()
            st.rerun()

        result = {
            'selected_table': selected_table,
            'uploaded_file': uploaded_file,
            'available_tables': available_tables
        }

        # Handle file upload
        if uploaded_file:
            result.update(self._handle_file_upload(uploaded_file))

        st.markdown('</div>', unsafe_allow_html=True)
        return result

    def _handle_file_upload(self, uploaded_file) -> Dict[str, Any]:
        """Handle file upload and processing."""
        try:
            file_type = uploaded_file.name.split('.')[-1].lower()

            if file_type == 'json':
                df, config, filename = self.data_manager.load_json_file(uploaded_file)
                # Store configuration in session state
                st.session_state.config_dict = config
            elif file_type == 'csv':
                df, filename = self.data_manager.load_csv_file(uploaded_file)
                config = {}
            else:
                st.error("Unsupported file type")
                return {}

            if not df.empty:
                # Save to session state
                self.data_manager.save_table_replace(df, "dataset")
                st.session_state.original_filename = filename
                st.session_state.file_type = file_type

                st.success(f"✅ Loaded {filename}: {len(df)} rows, {len(df.columns)} columns")

                # Validate required columns
                missing_cols = ColumnMapping.validate_mapping(
                    ColumnMapping.DEFAULT_MAPPING,
                    df.columns.tolist()
                )

                if missing_cols:
                    st.warning("⚠️ Some required columns are missing. Please map them in section B.")

                return {
                    'data_loaded': True,
                    'filename': filename,
                    'file_type': file_type,
                    'missing_columns': missing_cols
                }

        except Exception as e:
            st.error(f"❌ Error loading file: {e}")
            logger.error(f"File upload error: {e}")

        return {'data_loaded': False}


# Global instance
data_source_component = DataSourceComponent()
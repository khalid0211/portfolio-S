"""Data management service for EVM application."""

from __future__ import annotations
import json
import logging
import io
from pathlib import Path
from typing import Dict, Any, List, Optional, Tuple
import pandas as pd
import streamlit as st

from models.project import ProjectValidator, ProjectData, ColumnMapping
from config.constants import USE_DATABASE


logger = logging.getLogger(__name__)

# Configuration paths
CONFIG_DIR = Path.home() / ".portfolio_suite"
DEFAULT_DATASET_TABLE = "dataset"


class DataManager:
    """Manages data operations for the EVM application."""

    def __init__(self):
        self.validator = ProjectValidator()
        self._ensure_config_dir()

    def _ensure_config_dir(self):
        """Ensure configuration directory exists."""
        try:
            CONFIG_DIR.mkdir(parents=True, exist_ok=True)
        except Exception as e:
            logger.warning(f"Could not create config directory: {e}")

    # Session State Management
    def initialize_session_state(self):
        """Initialize session state variables."""
        if "session_initialized" not in st.session_state:
            st.session_state.batch_results = None
            st.session_state.data_df = None
            st.session_state.config_dict = {}
            st.session_state.data_loaded = False
            st.session_state.original_filename = None
            st.session_state.file_type = None
            st.session_state.processed_file_info = None
            st.session_state.session_initialized = True

    def list_session_tables(self) -> List[str]:
        """List available tables from session state."""
        try:
            tables = []

            # Add main data table if exists and has data
            if (hasattr(st.session_state, 'data_df') and
                st.session_state.data_df is not None and
                not st.session_state.data_df.empty):
                tables.append(DEFAULT_DATASET_TABLE)

            # Add batch results table if exists
            if (hasattr(st.session_state, 'batch_results') and
                st.session_state.batch_results is not None and
                not st.session_state.batch_results.empty):
                tables.append("batch_results")

            return tables

        except Exception as e:
            logger.error(f"Error listing session tables: {e}")
            return []

    def load_table(self, table_name: str) -> pd.DataFrame:
        """Load table from session state."""
        if not self.validator.validate_table_name(table_name):
            raise ValueError(f"Invalid table name: {table_name}")

        try:
            if table_name == DEFAULT_DATASET_TABLE:
                if hasattr(st.session_state, 'data_df') and st.session_state.data_df is not None:
                    return st.session_state.data_df.copy()
            elif table_name == "batch_results":
                if hasattr(st.session_state, 'batch_results') and st.session_state.batch_results is not None:
                    return st.session_state.batch_results.copy()

            # Return empty DataFrame if table doesn't exist
            return pd.DataFrame()

        except Exception as e:
            logger.error(f"Error loading table {table_name}: {e}")
            return pd.DataFrame()

    def save_table_replace(self, df: pd.DataFrame, table_name: str):
        """Save table to session state."""
        if not self.validator.validate_table_name(table_name):
            raise ValueError(f"Invalid table name: {table_name}")

        try:
            if table_name == DEFAULT_DATASET_TABLE:
                st.session_state.data_df = df.copy()
                st.session_state.data_loaded = True
            elif table_name == "batch_results":
                st.session_state.batch_results = df.copy()
            else:
                logger.warning(f"Unknown table name for session state: {table_name}")

        except Exception as e:
            logger.error(f"Error saving table {table_name}: {e}")
            raise

    def delete_table(self, table_name: str):
        """Delete a table from session state."""
        if not self.validator.validate_table_name(table_name):
            raise ValueError(f"Invalid table name: {table_name}")

        try:
            if table_name == DEFAULT_DATASET_TABLE:
                st.session_state.data_df = None
                st.session_state.data_loaded = False
                st.session_state.original_filename = None
                st.session_state.file_type = None
            elif table_name == "batch_results":
                st.session_state.batch_results = None
            else:
                logger.warning(f"Unknown table name for deletion: {table_name}")

        except Exception as e:
            logger.error(f"Error deleting table {table_name}: {e}")

    # Project CRUD Operations
    def insert_project_record(self, project_data: Dict[str, Any], table_name: str = DEFAULT_DATASET_TABLE):
        """Insert a new project record into session state."""
        if not self.validator.validate_table_name(table_name):
            raise ValueError(f"Invalid table name: {table_name}")

        try:
            # Load existing data or create new DataFrame
            df = self.load_table(table_name)
            if df.empty:
                df = pd.DataFrame()

            # Convert project data to DataFrame row
            new_row = pd.DataFrame([project_data])

            # Concatenate with existing data
            if not df.empty:
                df = pd.concat([df, new_row], ignore_index=True)
            else:
                df = new_row

            # Save back to session state
            self.save_table_replace(df, table_name)

        except Exception as e:
            logger.error(f"Error inserting project record: {e}")
            raise

    def update_project_record(self, project_id: str, project_data: Dict[str, Any], table_name: str = DEFAULT_DATASET_TABLE):
        """Update an existing project record in session state."""
        if not self.validator.validate_table_name(table_name):
            raise ValueError(f"Invalid table name: {table_name}")

        try:
            df = self.load_table(table_name)
            if df.empty:
                raise ValueError(f"No data found in table {table_name}")

            # Find the record to update
            project_id_col = 'Project ID'
            if project_id_col not in df.columns:
                raise ValueError(f"Project ID column not found in {table_name}")

            mask = df[project_id_col] == project_id
            if not mask.any():
                raise ValueError(f"Project ID {project_id} not found in {table_name}")

            # Update the record
            for key, value in project_data.items():
                if key in df.columns:
                    df.loc[mask, key] = value

            # Save back to session state
            self.save_table_replace(df, table_name)

        except Exception as e:
            logger.error(f"Error updating project record: {e}")
            raise

    def delete_project_record(self, project_id: str, table_name: str = DEFAULT_DATASET_TABLE):
        """Delete a project record from session state."""
        if not self.validator.validate_table_name(table_name):
            raise ValueError(f"Invalid table name: {table_name}")

        try:
            df = self.load_table(table_name)
            if df.empty:
                raise ValueError(f"No data found in table {table_name}")

            project_id_col = 'Project ID'
            if project_id_col not in df.columns:
                raise ValueError(f"Project ID column not found in {table_name}")

            # Remove the record
            mask = df[project_id_col] != project_id
            df = df[mask]

            # Save back to session state
            self.save_table_replace(df, table_name)

        except Exception as e:
            logger.error(f"Error deleting project record: {e}")
            raise

    def get_project_record(self, project_id: str, table_name: str = DEFAULT_DATASET_TABLE) -> Optional[Dict]:
        """Get a specific project record from session state."""
        try:
            df = self.load_table(table_name)
            if df.empty:
                return None

            project_id_col = 'Project ID'
            if project_id_col not in df.columns:
                return None

            mask = df[project_id_col] == project_id
            if not mask.any():
                return None

            record = df[mask].iloc[0].to_dict()
            return record

        except Exception as e:
            logger.error(f"Error getting project record: {e}")
            return None

    # Configuration Management
    def save_model_config(self, provider: str, model: str):
        """Save model configuration."""
        try:
            self._ensure_config_dir()
            config = {"provider": provider, "model": model}
            model_config_file = CONFIG_DIR / "model_config.json"
            with open(model_config_file, 'w') as f:
                json.dump(config, f, indent=2)
        except Exception as e:
            logger.error(f"Error saving model config: {e}")

    def load_model_config(self) -> Dict[str, str]:
        """Load model configuration."""
        try:
            model_config_file = CONFIG_DIR / "model_config.json"
            if model_config_file.exists():
                with open(model_config_file, 'r') as f:
                    return json.load(f)
        except Exception as e:
            logger.error(f"Error loading model config: {e}")
        return {"provider": "openai", "model": "gpt-3.5-turbo"}

    def save_column_mapping(self, table_name: str, mapping: Dict[str, str]):
        """Save column mapping for a specific table persistently."""
        try:
            self._ensure_config_dir()
            mapping_file = CONFIG_DIR / f"mapping_{table_name}.json"
            with open(mapping_file, 'w') as f:
                json.dump(mapping, f, indent=2)
        except Exception as e:
            logger.error(f"Error saving column mapping: {e}")

    def load_column_mapping(self, table_name: str) -> Dict[str, str]:
        """Load column mapping for a specific table."""
        try:
            mapping_file = CONFIG_DIR / f"mapping_{table_name}.json"
            if mapping_file.exists():
                with open(mapping_file, 'r') as f:
                    return json.load(f)
        except Exception as e:
            logger.error(f"Error loading column mapping: {e}")
        return ColumnMapping.DEFAULT_MAPPING.copy()

    # File Operations
    def load_json_file(self, uploaded_file) -> Tuple[pd.DataFrame, Dict[str, Any], str]:
        """Load JSON file and return DataFrame, config, and filename."""
        try:
            json_content = uploaded_file.getvalue()
            filename = uploaded_file.name
            config_data = json.loads(json_content.decode('utf-8'))

            # Extract DataFrame if present
            df = pd.DataFrame()
            if 'data' in config_data:
                df = pd.DataFrame(config_data['data'])

            # Extract configuration
            config = config_data.get('config', {})

            return df, config, filename

        except Exception as e:
            logger.error(f"Error loading JSON file: {e}")
            raise ValueError(f"Failed to load JSON file: {e}")

    def load_csv_file(self, uploaded_file) -> Tuple[pd.DataFrame, str]:
        """Load CSV file."""
        try:
            filename = uploaded_file.name
            encodings = ['utf-8-sig', 'utf-8', 'latin-1', 'cp1252']

            df = None
            for encoding in encodings:
                try:
                    content = uploaded_file.getvalue().decode(encoding)
                    df = pd.read_csv(io.StringIO(content))
                    break
                except UnicodeDecodeError:
                    continue
                except Exception as e:
                    logger.warning(f"Failed to parse CSV with encoding {encoding}: {e}")
                    continue

            if df is None:
                raise ValueError("Could not decode file with any supported encoding")

            return df, filename

        except Exception as e:
            logger.error(f"Error loading CSV file: {e}")
            raise ValueError(f"Failed to load CSV file: {e}")

    def is_project_table(self, table_name: str) -> bool:
        """Check if a table contains project data by examining required columns."""
        try:
            df = self.load_table(table_name)
            required_columns = ColumnMapping.get_required_columns()

            # Check if at least the core required columns exist
            core_required = ['Project ID', 'BAC', 'AC', 'Plan Start', 'Plan Finish']
            return all(col in df.columns for col in core_required)

        except Exception:
            return False

    def normalize_dataframe_columns(self, df: pd.DataFrame, column_mapping: Dict[str, str]) -> pd.DataFrame:
        """Normalize DataFrame columns based on mapping."""
        try:
            normalized_df = df.copy()

            # Create reverse mapping (from expected column names to actual column names)
            reverse_mapping = {}
            for key, expected_col in ColumnMapping.DEFAULT_MAPPING.items():
                mapped_col = column_mapping.get(key, expected_col)
                if mapped_col in df.columns:
                    reverse_mapping[mapped_col] = expected_col

            # Rename columns
            normalized_df = normalized_df.rename(columns=reverse_mapping)

            return normalized_df

        except Exception as e:
            logger.error(f"Error normalizing DataFrame columns: {e}")
            return df

    def create_demo_data(self):
        """Create demo data for testing."""
        try:
            demo_data = {
                'Project ID': ['PROJ-001', 'PROJ-002', 'PROJ-003'],
                'Project': ['Website Redesign', 'Mobile App', 'Database Migration'],
                'Organization': ['IT Department', 'Product Team', 'Infrastructure'],
                'Project Manager': ['Alice Johnson', 'Bob Smith', 'Carol Davis'],
                'BAC': [150000, 200000, 300000],
                'AC': [120000, 180000, 250000],
                'Plan Start': ['2024-01-01', '2024-02-01', '2024-03-01'],
                'Plan Finish': ['2024-06-30', '2024-08-31', '2024-12-31'],
                'Data Date': ['2024-04-15', '2024-06-15', '2024-09-15']
            }

            demo_df = pd.DataFrame(demo_data)
            self.save_table_replace(demo_df, DEFAULT_DATASET_TABLE)
            st.session_state.data_loaded = True

        except Exception as e:
            logger.error(f"Error creating demo data: {e}")

    # ========================================================================
    # PHASE B: Data Adapter Integration
    # ========================================================================

    def get_data_adapter(self):
        """Get unified data adapter based on config

        Returns adapter that routes to session state OR database based on
        USE_DATABASE flag.

        Returns:
            DatabaseAdapter if USE_DATABASE=True, otherwise self (session manager)

        Example:
            >>> adapter = data_manager.get_data_adapter()
            >>> projects = adapter.get_projects_for_analysis()
        """
        if USE_DATABASE:
            return DatabaseAdapter()
        else:
            return self

    # ========================================================================
    # Adapter Compatibility Methods
    # ========================================================================

    def get_projects_for_analysis(
        self,
        portfolio_id: Optional[int] = None,
        status_date: Optional[Any] = None
    ) -> pd.DataFrame:
        """Get projects for EVM analysis (compatibility method)

        Args:
            portfolio_id: Ignored in session state mode
            status_date: Ignored in session state mode

        Returns:
            DataFrame with project data
        """
        return self.load_table(DEFAULT_DATASET_TABLE)

    def get_batch_results(
        self,
        portfolio_id: Optional[int] = None,
        status_date: Optional[Any] = None
    ) -> pd.DataFrame:
        """Get batch calculation results (compatibility method)

        Args:
            portfolio_id: Ignored in session state mode
            status_date: Ignored in session state mode

        Returns:
            DataFrame with batch results
        """
        if hasattr(st.session_state, 'batch_results') and st.session_state.batch_results is not None:
            return st.session_state.batch_results.copy()
        return pd.DataFrame()

    def save_batch_results(
        self,
        results_df: pd.DataFrame,
        portfolio_id: Optional[int] = None
    ):
        """Save batch calculation results (compatibility method)

        Args:
            results_df: DataFrame with results
            portfolio_id: Ignored in session state mode
        """
        self.save_table_replace(results_df, "batch_results")


class DatabaseAdapter:
    """Adapter that routes data operations to database service.

    Provides the same interface as DataSessionManager's adapter methods
    but fetches data from the database via db_data_service.
    """

    def __init__(self):
        # Import here to avoid circular imports
        from services.db_data_service import db_data_manager
        self._db = db_data_manager

    @property
    def db_manager(self):
        """Expose db_data_manager for direct access to database methods."""
        return self._db

    def get_projects_for_analysis(
        self,
        portfolio_id: Optional[int] = None,
        status_date: Optional[Any] = None
    ) -> pd.DataFrame:
        """Get projects for EVM analysis from database.

        Args:
            portfolio_id: Portfolio ID to fetch projects for
            status_date: Status date filter (optional)

        Returns:
            DataFrame with project data ready for analysis
        """
        if portfolio_id is None:
            logger.warning("No portfolio_id provided to get_projects_for_analysis")
            return pd.DataFrame()

        try:
            return self._db.get_projects_for_batch_calculation(portfolio_id, status_date)
        except Exception as e:
            logger.error(f"Error getting projects for analysis: {e}")
            return pd.DataFrame()

    def get_batch_results(
        self,
        portfolio_id: Optional[int] = None,
        status_date: Optional[Any] = None
    ) -> pd.DataFrame:
        """Get batch calculation results from database.

        Args:
            portfolio_id: Portfolio ID
            status_date: Status date filter

        Returns:
            DataFrame with batch results
        """
        if portfolio_id is None:
            # Fall back to session state if no portfolio specified
            if hasattr(st.session_state, 'batch_results') and st.session_state.batch_results is not None:
                return st.session_state.batch_results.copy()
            return pd.DataFrame()

        try:
            return self._db.get_projects_for_batch_calculation(portfolio_id, status_date)
        except Exception as e:
            logger.error(f"Error getting batch results: {e}")
            return pd.DataFrame()

    def save_batch_results(
        self,
        results_df: pd.DataFrame,
        portfolio_id: Optional[int] = None
    ):
        """Save batch calculation results.

        Args:
            results_df: DataFrame with results
            portfolio_id: Portfolio ID
        """
        # Store in session state for compatibility
        st.session_state.batch_results = results_df.copy()

        # Also save to database if portfolio_id provided
        if portfolio_id is not None and not results_df.empty:
            try:
                from datetime import date
                status_date = date.today()
                if 'Data Date' in results_df.columns:
                    first_date = results_df['Data Date'].iloc[0]
                    if pd.notna(first_date):
                        status_date = pd.to_datetime(first_date).date()

                self._db.save_batch_evm_results(portfolio_id, status_date, results_df)
            except Exception as e:
                logger.error(f"Error saving batch results to database: {e}")

    def load_csv_data(
        self,
        df: pd.DataFrame,
        portfolio_id: int,
        status_date,
        column_mapping: Dict[str, str]
    ) -> Dict[str, int]:
        """Load CSV data into database tables.

        Delegates to DatabaseDataManager.load_csv_to_db().

        Args:
            df: DataFrame with project data (mapped columns)
            portfolio_id: Portfolio ID to assign projects to
            status_date: Status date for status reports
            column_mapping: Column name mappings

        Returns:
            Dictionary with counts:
            {
                'projects_created': int,
                'projects_updated': int,
                'status_reports_created': int,
                'errors': int
            }
        """
        return self._db.load_csv_to_db(
            df=df,
            portfolio_id=portfolio_id,
            status_date=status_date,
            column_mapping=column_mapping
        )


# Global instance for easy access
data_manager = DataManager()

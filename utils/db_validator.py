"""
Database Connection Validator
Tests and validates DuckDB connections before switching
"""

import duckdb
import logging
from typing import Tuple, Optional
from pathlib import Path

logger = logging.getLogger(__name__)


def _is_path_in_duckdb_directory(db_path: str) -> bool:
    """
    Validate that the database path is within the application's DuckDB directory.
    Security check to prevent directory traversal attacks.

    Args:
        db_path: Path to validate

    Returns:
        bool: True if path is safe (within DuckDB/ directory), False otherwise
    """
    try:
        # Get the application's DuckDB directory
        app_root = Path(__file__).parent.parent
        duckdb_dir = app_root / "DuckDB"

        # Resolve both paths to absolute paths
        resolved_path = Path(db_path).resolve()
        resolved_duckdb_dir = duckdb_dir.resolve()

        # Check if the resolved path is within the DuckDB directory
        try:
            # This will raise ValueError if resolved_path is not relative to resolved_duckdb_dir
            resolved_path.relative_to(resolved_duckdb_dir)
            return True
        except ValueError:
            return False

    except Exception as e:
        logger.error(f"Error validating path: {e}")
        return False


def validate_motherduck_connection(database_name: str, token: Optional[str] = None) -> Tuple[bool, str]:
    """
    Validate MotherDuck connection before switching

    Args:
        database_name: Name of MotherDuck database (without md: prefix)
        token: Optional MotherDuck token (if None, uses environment/secrets)

    Returns:
        Tuple of (success: bool, message: str)
    """
    try:
        # Build connection string
        if token:
            conn_str = f"md:{database_name}?motherduck_token={token}"
        else:
            # Try to get org token from secrets
            try:
                import streamlit as st
                if hasattr(st, 'secrets') and 'motherduck' in st.secrets:
                    org_token = st.secrets['motherduck'].get('token', '')
                    if org_token:
                        conn_str = f"md:{database_name}?motherduck_token={org_token}"
                    else:
                        return False, "MotherDuck token not found in secrets"
                else:
                    return False, "MotherDuck configuration not found in secrets"
            except Exception as e:
                return False, f"Could not load MotherDuck configuration: {str(e)}"

        # Test connection
        conn = duckdb.connect(conn_str)

        # Query information_schema to verify connectivity
        result = conn.execute("SELECT COUNT(*) FROM information_schema.tables").fetchone()
        table_count = result[0] if result else 0

        conn.close()

        return True, f"Successfully connected to MotherDuck database '{database_name}' ({table_count} tables found)"

    except Exception as e:
        error_msg = str(e)
        if "Authentication failed" in error_msg or "token" in error_msg.lower():
            return False, f"Authentication failed: Invalid MotherDuck token"
        elif "database" in error_msg.lower() and "not found" in error_msg.lower():
            return False, f"Database '{database_name}' not found in MotherDuck"
        else:
            return False, f"Connection failed: {error_msg}"


def validate_local_connection(db_path: str, create_if_missing: bool = False) -> Tuple[bool, str]:
    """
    Validate local DuckDB file connection

    Args:
        db_path: Path to local .duckdb file
        create_if_missing: If True, create new database if it doesn't exist

    Returns:
        Tuple of (success: bool, message: str)
    """
    try:
        # Security check: Ensure path is within DuckDB directory
        if not _is_path_in_duckdb_directory(db_path):
            return False, f"Security: Database path must be within the application's DuckDB/ directory"

        # Validate file extension
        if not db_path.endswith('.duckdb'):
            return False, "Invalid file extension: Database file must have .duckdb extension"

        path_obj = Path(db_path)

        # Check if file exists
        file_exists = path_obj.exists()

        if not file_exists and not create_if_missing:
            return False, f"Database file not found: {db_path}"

        # Ensure parent directory exists
        path_obj.parent.mkdir(parents=True, exist_ok=True)

        # Test connection
        conn = duckdb.connect(str(path_obj))

        # Query to verify connectivity
        result = conn.execute("SELECT COUNT(*) FROM information_schema.tables").fetchone()
        table_count = result[0] if result else 0

        conn.close()

        if not file_exists:
            return True, f"New database created at {db_path} (needs initialization)"
        else:
            return True, f"Successfully connected to local database ({table_count} tables found)"

    except Exception as e:
        return False, f"Connection failed: {str(e)}"


def test_connection(config: dict) -> Tuple[bool, str]:
    """
    Test database connection based on configuration

    Args:
        config: Database configuration dict with keys:
            - connection_type: 'local' or 'motherduck'
            - local_path: Path for local DuckDB
            - motherduck_database: MotherDuck database name
            - motherduck_token: Optional MotherDuck token

    Returns:
        Tuple of (success: bool, message: str)
    """
    connection_type = config.get('connection_type')

    if connection_type == 'motherduck':
        database_name = config.get('motherduck_database')
        token = config.get('motherduck_token')

        if not database_name:
            return False, "MotherDuck database name is required"

        return validate_motherduck_connection(database_name, token)

    elif connection_type == 'local':
        db_path = config.get('local_path')

        if not db_path:
            return False, "Local database path is required"

        return validate_local_connection(db_path, create_if_missing=False)

    else:
        return False, f"Invalid connection type: {connection_type}"

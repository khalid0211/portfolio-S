"""DuckDB connection management for Portfolio Analysis Suite

This module provides a singleton connection manager for DuckDB database operations.
Supports connection pooling, transaction management, and query execution.
"""

import duckdb
from pathlib import Path
from typing import Optional, Any, List, Tuple
import logging
from contextlib import contextmanager

logger = logging.getLogger(__name__)


def _sanitize_connection_string(conn_str: str) -> str:
    """Sanitize connection string for safe logging by masking tokens

    Args:
        conn_str: Database connection string that may contain sensitive tokens

    Returns:
        Sanitized string with tokens masked
    """
    if not conn_str:
        return conn_str

    # Mask motherduck_token parameter
    if 'motherduck_token=' in conn_str:
        # Find the token and mask it
        parts = conn_str.split('motherduck_token=')
        if len(parts) > 1:
            # Get base path and mask the token
            base = parts[0]
            # Token ends at & or end of string
            token_part = parts[1]
            if '&' in token_part:
                remaining = '&' + token_part.split('&', 1)[1]
            else:
                remaining = ''
            return f"{base}motherduck_token=***MASKED***{remaining}"

    return conn_str


class DatabaseConnection:
    """Manages DuckDB connection lifecycle with singleton pattern"""

    def __init__(self, db_path: Optional[str] = None):
        """Initialize database connection manager

        Args:
            db_path: Path to DuckDB database file. If None, defaults to MotherDuck portfolio_cloud.
                    Only SuperAdmins can use local database paths.
        """
        if db_path is None:
            # Default to MotherDuck portfolio_cloud for all users
            # Only SuperAdmins can configure custom database connections via get_user_database_config()
            try:
                import streamlit as st
                if hasattr(st, 'secrets') and 'motherduck' in st.secrets:
                    token = st.secrets['motherduck'].get('token', '')
                    if token:
                        db_path = f"md:portfolio_cloud?motherduck_token={token}"
                        logger.info("No db_path provided, defaulting to MotherDuck portfolio_cloud (token from secrets)")
                    else:
                        db_path = "md:portfolio_cloud"
                        logger.info("No db_path provided, defaulting to MotherDuck portfolio_cloud")
                else:
                    db_path = "md:portfolio_cloud"
                    logger.info("No db_path provided, defaulting to MotherDuck portfolio_cloud")
            except Exception:
                db_path = "md:portfolio_cloud"
                logger.info("No db_path provided, defaulting to MotherDuck portfolio_cloud")

        self.db_path = db_path
        self._conn: Optional[duckdb.DuckDBPyConnection] = None
        logger.info(f"DatabaseConnection initialized with path: {_sanitize_connection_string(db_path)}")

    def get_connection(self) -> duckdb.DuckDBPyConnection:
        """Get or create database connection

        Returns:
            Active DuckDB connection
        """
        if self._conn is None:
            try:
                self._conn = duckdb.connect(self.db_path)
                logger.info(f"DuckDB connected: {_sanitize_connection_string(self.db_path)}")
            except Exception as e:
                logger.error(f"Failed to connect to DuckDB: {e}")
                raise
        return self._conn

    def close(self):
        """Close database connection"""
        if self._conn:
            try:
                self._conn.close()
                logger.info("DuckDB connection closed")
            except Exception as e:
                logger.error(f"Error closing connection: {e}")
            finally:
                self._conn = None

    def execute(self, query: str, params: Optional[Tuple] = None, retry: bool = True) -> duckdb.DuckDBPyConnection:
        """Execute SQL query with optional parameters

        Args:
            query: SQL query string
            params: Optional tuple of parameter values for parameterized queries
            retry: If True, retry once on connection error

        Returns:
            DuckDB connection with executed query cursor

        Example:
            >>> db.execute("SELECT * FROM project WHERE project_id = ?", (123,))
        """
        conn = self.get_connection()
        try:
            if params:
                return conn.execute(query, params)
            return conn.execute(query)
        except Exception as e:
            error_str = str(e).lower()
            if retry and ('no open result set' in error_str or 'null' in error_str or 'connection' in error_str):
                logger.warning(f"Connection issue detected during execute, attempting reconnection: {e}")
                self._reconnect()
                conn = self.get_connection()
                if params:
                    return conn.execute(query, params)
                return conn.execute(query)
            logger.error(f"Query execution failed: {query[:100]}... Error: {e}")
            raise

    def query(self, sql: str) -> duckdb.DuckDBPyRelation:
        """Execute query and return DuckDB relation object

        Args:
            sql: SQL query string

        Returns:
            DuckDB relation object (similar to DataFrame)

        Example:
            >>> relation = db.query("SELECT * FROM project")
            >>> df = relation.df()  # Convert to pandas DataFrame
        """
        return self.get_connection().query(sql)

    def fetch_one(self, query: str, params: Optional[Tuple] = None, retry: bool = True) -> Optional[Tuple]:
        """Execute query and fetch one result

        Args:
            query: SQL query string
            params: Optional query parameters
            retry: If True, retry once on connection error

        Returns:
            Single row as tuple, or None if no results
        """
        try:
            result = self.execute(query, params)
            return result.fetchone()
        except Exception as e:
            error_str = str(e).lower()
            if retry and ('no open result set' in error_str or 'null' in error_str):
                logger.warning(f"Connection issue detected, attempting reconnection: {e}")
                self._reconnect()
                result = self.execute(query, params)
                return result.fetchone()
            raise

    def _reconnect(self):
        """Force reconnection to the database"""
        logger.info("Forcing database reconnection...")
        if self._conn:
            try:
                self._conn.close()
            except Exception:
                pass
            self._conn = None
        # get_connection will create a new connection
        self.get_connection()

    def fetch_all(self, query: str, params: Optional[Tuple] = None, retry: bool = True) -> List[Tuple]:
        """Execute query and fetch all results

        Args:
            query: SQL query string
            params: Optional query parameters
            retry: If True, retry once on connection error

        Returns:
            List of rows as tuples
        """
        try:
            result = self.execute(query, params)
            return result.fetchall()
        except Exception as e:
            error_str = str(e).lower()
            if retry and ('no open result set' in error_str or 'null' in error_str):
                logger.warning(f"Connection issue detected in fetch_all, attempting reconnection: {e}")
                self._reconnect()
                result = self.execute(query, params)
                return result.fetchall()
            raise

    def table_exists(self, table_name: str) -> bool:
        """Check if a table exists in the database

        Args:
            table_name: Name of table to check

        Returns:
            True if table exists, False otherwise
        """
        try:
            query = """
                SELECT COUNT(*)
                FROM information_schema.tables
                WHERE table_name = ?
            """
            result = self.fetch_one(query, (table_name,))
            return result[0] > 0 if result else False
        except Exception as e:
            logger.error(f"Error checking table existence: {e}")
            return False

    def get_table_columns(self, table_name: str) -> List[str]:
        """Get list of column names for a table

        Args:
            table_name: Name of table

        Returns:
            List of column names
        """
        try:
            query = f"PRAGMA table_info('{table_name}')"
            result = self.fetch_all(query, retry=True)
            # table_info returns: (cid, name, type, notnull, dflt_value, pk)
            return [row[1] for row in result]
        except Exception as e:
            logger.error(f"Error getting table columns: {e}")
            # Try reconnecting and retry once more
            try:
                self._reconnect()
                result = self.fetch_all(query, retry=False)
                return [row[1] for row in result]
            except Exception:
                return []

    @contextmanager
    def transaction(self):
        """Context manager for database transactions

        Example:
            >>> with db.transaction():
            ...     db.execute("INSERT INTO project ...")
            ...     db.execute("INSERT INTO project_baseline ...")
            # Commits on success, rolls back on exception
        """
        conn = self.get_connection()
        try:
            conn.execute("BEGIN TRANSACTION")
            yield conn
            conn.execute("COMMIT")
            logger.debug("Transaction committed")
        except Exception as e:
            conn.execute("ROLLBACK")
            logger.error(f"Transaction rolled back due to error: {e}")
            raise

    def execute_script(self, script: str):
        """Execute multi-statement SQL script

        Args:
            script: SQL script with multiple statements separated by semicolons
        """
        statements = [stmt.strip() for stmt in script.split(';') if stmt.strip()]
        conn = self.get_connection()

        for stmt in statements:
            try:
                conn.execute(stmt)
                logger.debug(f"Executed: {stmt[:50]}...")
            except Exception as e:
                logger.error(f"Script execution failed at statement: {stmt[:100]}... Error: {e}")
                raise

    def vacuum(self):
        """Optimize database by reclaiming unused space"""
        try:
            self.execute("VACUUM")
            logger.info("Database vacuumed successfully")
        except Exception as e:
            logger.error(f"Vacuum failed: {e}")

    def __enter__(self):
        """Context manager entry - get connection"""
        return self.get_connection()

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit - close connection"""
        self.close()


# Global singleton instance
_db_instance: Optional[DatabaseConnection] = None


def get_db(db_path: Optional[str] = None, user_email: Optional[str] = None) -> DatabaseConnection:
    """Get global database instance (singleton pattern)

    Args:
        db_path: Optional database path. Only used on first call.
        user_email: Optional user email for user-specific database config.
                   If provided, loads user's database preference from Firestore.

    Returns:
        DatabaseConnection singleton instance

    Example:
        >>> db = get_db()
        >>> result = db.execute("SELECT * FROM project")
        >>> # Or with user-specific config:
        >>> db = get_db(user_email="user@example.com")
    """
    import os
    global _db_instance

    # Check for MOTHERDUCK_TOKEN environment variable first (for Docker/cloud deployment)
    motherduck_token = os.environ.get('MOTHERDUCK_TOKEN')
    if _db_instance is None and db_path is None and motherduck_token:
        # Use MotherDuck with environment token
        logger.info("MOTHERDUCK_TOKEN environment variable detected, using MotherDuck (portfolio_cloud)")
        db_path = "md:portfolio_cloud"

    # If user_email is provided and we don't have an instance yet,
    # load user-specific database configuration
    if _db_instance is None and user_email is not None and db_path is None:
        try:
            from utils.database_config import get_user_database_config

            user_config = get_user_database_config(user_email)

            if user_config.get('connection_type') == 'motherduck':
                # Build MotherDuck connection string
                db_name = user_config.get('motherduck_database')
                token = user_config.get('motherduck_token', '')

                if token:
                    db_path = f"md:{db_name}?motherduck_token={token}"
                else:
                    # Use org token from secrets
                    try:
                        import streamlit as st
                        if hasattr(st, 'secrets') and 'motherduck' in st.secrets:
                            org_token = st.secrets['motherduck'].get('token', '')
                            if org_token:
                                db_path = f"md:{db_name}?motherduck_token={org_token}"
                            else:
                                logger.warning("MotherDuck token not found in secrets, using default connection")
                        else:
                            logger.warning("MotherDuck secrets not configured, using default connection")
                    except Exception as e:
                        logger.warning(f"Could not load MotherDuck token from secrets: {e}, using default connection")
            else:
                # Use local path from user config
                db_path = user_config.get('local_path')
                if db_path:
                    logger.info(f"Using user-specific local database: {db_path}")

        except Exception as e:
            logger.warning(f"Could not load user database config: {e}, using default")

    if _db_instance is None:
        _db_instance = DatabaseConnection(db_path)
    return _db_instance


def reset_db_instance():
    """Reset the global database instance (mainly for testing)"""
    global _db_instance
    if _db_instance:
        _db_instance.close()
    _db_instance = None

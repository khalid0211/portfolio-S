"""User synchronization between authentication and database

Ensures that authenticated users are available in the app_user table
"""

import streamlit as st
from typing import Optional, Dict, Any
from database.db_connection import get_db
import logging

logger = logging.getLogger(__name__)


def sync_authenticated_user_to_db() -> Optional[int]:
    """Sync the currently authenticated user to the app_user table

    Returns:
        user_id if successful, None if not authenticated
    """
    if not st.session_state.get('authenticated'):
        return None

    user_info = st.session_state.get('user_info')
    if not user_info:
        return None

    email = user_info.get('email')
    name = user_info.get('name', 'Unknown User')

    if not email:
        return None

    db = get_db()

    try:
        # Check if user already exists
        existing = db.fetch_one(
            "SELECT user_id FROM app_user WHERE email = ?",
            (email,)
        )

        if existing:
            return existing[0]

        # Create new user
        max_id = db.fetch_one("SELECT COALESCE(MAX(user_id), 0) + 1 FROM app_user")
        if max_id is None:
            logger.error("Database connection error: could not generate user_id")
            return None
        user_id = max_id[0]

        db.execute("""
            INSERT INTO app_user (
                user_id, display_name, email, role,
                is_active, created_at, last_login
            ) VALUES (?, ?, ?, ?, TRUE, now(), now())
        """, (user_id, name, email, 'User'))

        logger.info(f"Created database user for {name} ({email}) with ID {user_id}")
        return user_id

    except Exception as e:
        logger.error(f"Error syncing user to database: {e}")
        return None


def get_current_user_id() -> Optional[int]:
    """Get the database user_id for the currently logged-in user

    Returns:
        user_id if user is authenticated and synced, None otherwise
    """
    return sync_authenticated_user_to_db()


def get_current_user_info() -> Optional[Dict[str, Any]]:
    """Get the database user record for the currently logged-in user

    Returns:
        Dictionary with user info, or None if not found
    """
    user_id = get_current_user_id()
    if not user_id:
        return None

    db = get_db()

    try:
        result = db.fetch_one(
            "SELECT * FROM app_user WHERE user_id = ?",
            (user_id,)
        )

        if result:
            columns = db.get_table_columns('app_user')
            return dict(zip(columns, result))

        return None

    except Exception as e:
        logger.error(f"Error getting user info: {e}")
        return None

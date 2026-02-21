"""Portfolio Access Control Module

Manages portfolio-level user access: who can view and who can edit each portfolio.
Uses the portfolio_access table for shared access alongside portfolio_ownership for owners.

Usage:
    from utils.portfolio_access import can_write_portfolio, get_portfolio_access_level

    level = get_portfolio_access_level(portfolio_id, user_email)  # 'owner', 'write', 'read', or None
    if can_write_portfolio(portfolio_id, user_email):
        # allow save/delete operations
"""

import logging
import pandas as pd
from typing import Optional, List

from database.db_connection import get_db

logger = logging.getLogger(__name__)


def _is_super_admin(user_email: str) -> bool:
    """Check if user is super admin via Firestore.

    Returns False if Firestore is unavailable (graceful degradation).
    """
    try:
        from utils.firestore_client import get_firestore_client
        from utils.user_manager import is_super_admin
        fs = get_firestore_client()
        if fs:
            return is_super_admin(fs, user_email)
    except Exception as e:
        logger.debug(f"Could not check super admin status: {e}")
    return False


def get_portfolio_access_level(portfolio_id: int, user_email: Optional[str] = None) -> Optional[str]:
    """Get a user's access level for a portfolio.

    Args:
        portfolio_id: Portfolio ID
        user_email: User email (if None, gets from session state)

    Returns:
        'owner'  - user owns the portfolio (full control)
        'write'  - user has read/write shared access
        'read'   - user has read-only shared access
        None     - user has no access
    """
    if user_email is None:
        from utils.portfolio_context import get_current_user_email
        user_email = get_current_user_email()

    if not user_email:
        return None

    # Super admins get owner-level access to everything
    if _is_super_admin(user_email):
        return 'owner'

    try:
        db = get_db()

        # Check ownership first
        owner_check = db.fetch_one("""
            SELECT 1
            FROM portfolio_ownership po
            JOIN app_user u ON po.owner_user_id = u.user_id
            WHERE po.portfolio_id = ?
              AND u.email = ?
              AND po.is_active = TRUE
        """, (portfolio_id, user_email))

        if owner_check:
            return 'owner'

        # Check shared access
        access_check = db.fetch_one("""
            SELECT pa.access_level
            FROM portfolio_access pa
            JOIN app_user u ON pa.user_id = u.user_id
            WHERE pa.portfolio_id = ?
              AND u.email = ?
              AND pa.is_active = TRUE
        """, (portfolio_id, user_email))

        if access_check:
            return access_check[0]  # 'read' or 'write'

        return None

    except Exception as e:
        logger.error(f"Error checking portfolio access level: {e}")
        return None


def can_write_portfolio(portfolio_id: int, user_email: Optional[str] = None) -> bool:
    """Check if user can modify data in this portfolio.

    Returns True if user is owner, has write access, or is super admin.
    """
    level = get_portfolio_access_level(portfolio_id, user_email)
    return level in ('owner', 'write')


def get_accessible_portfolio_ids(user_email: Optional[str] = None) -> List[int]:
    """Get all portfolio IDs the user can access (owner OR shared).

    Super admins get all active portfolios.

    Args:
        user_email: User email (if None, gets from session state)

    Returns:
        List of portfolio_id values
    """
    if user_email is None:
        from utils.portfolio_context import get_current_user_email
        user_email = get_current_user_email()

    if not user_email:
        return []

    # Super admins see everything
    if _is_super_admin(user_email):
        try:
            db = get_db()
            result = db.execute(
                "SELECT portfolio_id FROM portfolio WHERE is_active = TRUE"
            )
            return [row[0] for row in result.fetchall()]
        except Exception as e:
            logger.error(f"Error fetching all portfolios for super admin: {e}")
            return []

    try:
        db = get_db()

        # Union of owned + shared portfolios
        result = db.execute("""
            SELECT DISTINCT p.portfolio_id
            FROM portfolio p
            LEFT JOIN portfolio_ownership po ON p.portfolio_id = po.portfolio_id AND po.is_active = TRUE
            LEFT JOIN app_user u_owner ON po.owner_user_id = u_owner.user_id
            LEFT JOIN portfolio_access pa ON p.portfolio_id = pa.portfolio_id AND pa.is_active = TRUE
            LEFT JOIN app_user u_access ON pa.user_id = u_access.user_id
            WHERE p.is_active = TRUE
              AND (u_owner.email = ? OR u_access.email = ?)
        """, (user_email, user_email))

        return [row[0] for row in result.fetchall()]

    except Exception as e:
        logger.error(f"Error fetching accessible portfolio IDs: {e}")
        return []


def list_portfolio_users(portfolio_id: int) -> pd.DataFrame:
    """Get all users with access to a portfolio.

    Returns DataFrame with columns: user_id, display_name, email, access_level
    Owner is listed with access_level='owner'.
    """
    try:
        db = get_db()

        query = """
            SELECT
                u.user_id,
                u.display_name,
                u.email,
                'owner' as access_level,
                po.assigned_at as granted_at
            FROM portfolio_ownership po
            JOIN app_user u ON po.owner_user_id = u.user_id
            WHERE po.portfolio_id = ? AND po.is_active = TRUE

            UNION ALL

            SELECT
                u.user_id,
                u.display_name,
                u.email,
                pa.access_level,
                pa.granted_at
            FROM portfolio_access pa
            JOIN app_user u ON pa.user_id = u.user_id
            WHERE pa.portfolio_id = ? AND pa.is_active = TRUE

            ORDER BY access_level, display_name
        """

        return db.execute(query, (portfolio_id, portfolio_id)).df()

    except Exception as e:
        logger.error(f"Error listing portfolio users: {e}")
        return pd.DataFrame(columns=['user_id', 'display_name', 'email', 'access_level', 'granted_at'])


def add_portfolio_access(
    portfolio_id: int,
    user_id: int,
    access_level: str = 'read',
    granted_by_user_id: Optional[int] = None
) -> bool:
    """Grant a user access to a portfolio.

    If the user already has an active access record, updates the level.

    Args:
        portfolio_id: Portfolio ID
        user_id: User ID to grant access to
        access_level: 'read' or 'write'
        granted_by_user_id: User ID of the granter

    Returns:
        True if successful
    """
    if access_level not in ('read', 'write'):
        raise ValueError(f"Invalid access_level: {access_level}. Must be 'read' or 'write'.")

    try:
        db = get_db()

        # Check if user is already the owner
        owner_check = db.fetch_one("""
            SELECT 1 FROM portfolio_ownership
            WHERE portfolio_id = ? AND owner_user_id = ? AND is_active = TRUE
        """, (portfolio_id, user_id))

        if owner_check:
            logger.warning(f"User {user_id} is already the owner of portfolio {portfolio_id}")
            return False

        # Upsert: insert or update existing record
        # Check if record exists (active or inactive)
        existing = db.fetch_one("""
            SELECT is_active, access_level FROM portfolio_access
            WHERE portfolio_id = ? AND user_id = ?
        """, (portfolio_id, user_id))

        if existing:
            # Update existing record
            db.execute("""
                UPDATE portfolio_access
                SET access_level = ?, is_active = TRUE, granted_at = now(), granted_by_user_id = ?
                WHERE portfolio_id = ? AND user_id = ?
            """, (access_level, granted_by_user_id, portfolio_id, user_id))
        else:
            # Insert new record
            db.execute("""
                INSERT INTO portfolio_access (portfolio_id, user_id, access_level, granted_at, granted_by_user_id, is_active)
                VALUES (?, ?, ?, now(), ?, TRUE)
            """, (portfolio_id, user_id, access_level, granted_by_user_id))

        logger.info(f"Granted {access_level} access to user {user_id} for portfolio {portfolio_id}")
        return True

    except Exception as e:
        logger.error(f"Error adding portfolio access: {e}")
        return False


def remove_portfolio_access(portfolio_id: int, user_id: int) -> bool:
    """Remove a user's access to a portfolio (soft-delete).

    Args:
        portfolio_id: Portfolio ID
        user_id: User ID to remove access from

    Returns:
        True if successful
    """
    try:
        db = get_db()
        db.execute("""
            UPDATE portfolio_access
            SET is_active = FALSE
            WHERE portfolio_id = ? AND user_id = ?
        """, (portfolio_id, user_id))

        logger.info(f"Removed access for user {user_id} from portfolio {portfolio_id}")
        return True

    except Exception as e:
        logger.error(f"Error removing portfolio access: {e}")
        return False


def update_portfolio_access_level(portfolio_id: int, user_id: int, new_level: str) -> bool:
    """Change access level for an existing user.

    Args:
        portfolio_id: Portfolio ID
        user_id: User ID
        new_level: 'read' or 'write'

    Returns:
        True if successful
    """
    if new_level not in ('read', 'write'):
        raise ValueError(f"Invalid access level: {new_level}. Must be 'read' or 'write'.")

    try:
        db = get_db()
        db.execute("""
            UPDATE portfolio_access
            SET access_level = ?
            WHERE portfolio_id = ? AND user_id = ? AND is_active = TRUE
        """, (new_level, portfolio_id, user_id))

        logger.info(f"Updated access level to {new_level} for user {user_id} on portfolio {portfolio_id}")
        return True

    except Exception as e:
        logger.error(f"Error updating portfolio access level: {e}")
        return False

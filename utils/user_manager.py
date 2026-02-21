"""
User Management Module for Firestore
Handles user profiles, permissions, OAuth tokens, and database preferences
Replaces the legacy auth_utils.py module
"""

import logging
from datetime import datetime, timedelta
from typing import Dict, List, Optional
from google.cloud import firestore
import requests
import os

logger = logging.getLogger(__name__)

# Bootstrap admin usage only.
# This list is ONLY used when creating a NEW user to set their initial role.
# It is NOT used for runtime checks.
BOOTSTRAP_ADMINS = [
    "khalid0211@gmail.com",
    "dev@localhost"
]

# Default permissions for new users
# Options: 'write', 'read', 'none'
# New users only have access to the main screen - admin must grant page permissions
DEFAULT_PERMISSIONS = {
    'portfolio_management': 'none',
    'file_management': 'none',
    'manual_data_entry': 'none',
    'project_analysis': 'none',
    'portfolio_analysis': 'none',
    'portfolio_charts': 'none',
    'baseline_management': 'none',
    'strategic_factors': 'none',
    'sdg_management': 'none',
    'cash_flow_simulator': 'none',
    'evm_simulator': 'none',
    'ai_assistant': 'none',
    'database_diagnostics': 'none',
    'user_management': 'none'
}


def is_admin(db: firestore.Client, user_email: str) -> bool:
    """
    Check if user is an administrator by checking Firestore role.
    Also returns True for BOOTSTRAP_ADMINS to prevent lockout.

    Args:
        db: Firestore client instance
        user_email: User's email address

    Returns:
        bool: True if user has 'admin' role, False otherwise
    """
    if not db or not user_email:
        return False

    # Hardcoded override for bootstrap admins
    if user_email in BOOTSTRAP_ADMINS:
        return True

    try:
        # We can optimize this by caching or passing the user dict if available
        # For now, we fetch validation to be secure
        user_doc = db.collection('users').document(user_email).get()
        if user_doc.exists:
            data = user_doc.to_dict()
            return data.get('role') == 'admin' and data.get('status') == 'active'

        return False
    except Exception as e:
        logger.error(f"Error checking admin status: {e}")
        return False


def is_super_admin(db: firestore.Client, user_email: str) -> bool:
    """
    Check if user is a super administrator with database configuration privileges.
    Super admins can configure database connections and perform system-level operations.

    Args:
        db: Firestore client instance
        user_email: User's email address

    Returns:
        bool: True if user has 'super_admin' role, False otherwise
    """
    if not db or not user_email:
        return False

    # Bootstrap admins automatically become super admins
    if user_email in BOOTSTRAP_ADMINS:
        return True

    try:
        user_doc = db.collection('users').document(user_email).get()
        if user_doc.exists:
            data = user_doc.to_dict()
            return data.get('role') == 'super_admin' and data.get('status') == 'active'

        return False
    except Exception as e:
        logger.error(f"Error checking super admin status: {e}")
        return False


def ensure_user_exists(db: firestore.Client, user_email: str, user_name: str = None) -> bool:
    """
    Create or update user in Firestore on login.
    Handles migration to new schema concepts (role, status).

    Args:
        db: Firestore client instance
        user_email: User's email address
        user_name: User's display name

    Returns:
        bool: True if operation successful, False otherwise
    """
    if not db or not user_email:
        # If DB calls fail due to 403, we might pass None.
        # Check if user is bootstrap admin before giving up.
        if user_email in BOOTSTRAP_ADMINS:
            return True
        logger.error("Invalid parameters for ensure_user_exists")
        return False

    try:
        user_ref = db.collection('users').document(user_email)
        user_doc = user_ref.get()

        target_role = 'user'
        if user_email in BOOTSTRAP_ADMINS:
            target_role = 'super_admin'  # Bootstrap admins get super_admin role

        if user_doc.exists:
            # Update existing user
            # We don't overwrite role or permissions here to avoid resetting existing config
            user_data = user_doc.to_dict()

            # Get current access count (check both locations for migration)
            profile_data = user_data.get('profile', {})
            profile_access_count = profile_data.get('access_count', 0) or 0
            root_access_count = user_data.get('access_count', 0) or 0

            # If there's a root-level access_count that needs migration, add it to profile count
            if root_access_count > 0 and profile_access_count == 0:
                # Migrate: set profile.access_count to root value + 1 (for this login)
                new_count = root_access_count + 1
                updates = {
                    'profile.last_access_date': firestore.SERVER_TIMESTAMP,
                    'profile.access_count': new_count,
                    'access_count': firestore.DELETE_FIELD  # Remove old root-level field
                }
                logger.info(f"Migrating access_count for {user_email}: {root_access_count} -> {new_count}")
            else:
                # Normal increment
                updates = {
                    'profile.last_access_date': firestore.SERVER_TIMESTAMP,
                    'profile.access_count': firestore.Increment(1)
                }

            # Migration check: Ensure role and status exist
            if 'role' not in user_data:
                updates['role'] = target_role # Default logic for migration
            if 'status' not in user_data:
                updates['status'] = 'active'

            # MIGRATION: Upgrade existing 'admin' users to 'super_admin'
            existing_role = user_data.get('role')
            if existing_role == 'admin':
                updates['role'] = 'super_admin'
                logger.info(f"Migrating user {user_email} from 'admin' to 'super_admin'")

            # Use update() instead of set(merge=True) for Increment to work correctly
            user_ref.update(updates)
            logger.info(f"Updated existing user: {user_email}")
        else:
            # Create new user
            user_ref.set({
                'profile': {
                    'email': user_email,
                    'name': user_name or user_email,
                    'created_at': firestore.SERVER_TIMESTAMP,
                    'last_access_date': firestore.SERVER_TIMESTAMP,
                    'access_count': 1
                },
                'role': target_role,
                'status': 'active',
                'permissions': DEFAULT_PERMISSIONS.copy()
            })
            logger.info(f"Created new user: {user_email}")

        return True
    except Exception as e:
        logger.error(f"Error ensuring user exists: {e}")
        # If API is disabled (403) or database missing (404), swallow error
        error_str = str(e)
        if any(x in error_str for x in ["403", "404", "SERVICE_DISABLED", "does not exist", "NotFound"]):
            return user_email in BOOTSTRAP_ADMINS
        return False


def get_user_permissions(db: firestore.Client, user_email: str) -> Optional[Dict]:
    """
    Get user's permissions from Firestore
    """
    if not db or not user_email:
        return None

    try:
        user_doc = db.collection('users').document(user_email).get()
        if user_doc.exists:
            user_data = user_doc.to_dict()
            # If status is not active, return empty permissions (lockout)
            if user_data.get('status') != 'active':
                return {}
            
            # If admin, return full write access for everything
            if user_data.get('role') == 'admin':
                return {k: 'write' for k in DEFAULT_PERMISSIONS.keys()}
                
            return user_data.get('permissions', {})
        return None
    except Exception as e:
        logger.error(f"Error getting user permissions: {e}")
        return None


def get_page_permission_level(db: firestore.Client, user_email: str, page_name: str) -> str:
    """
    Get the specific permission level for a page.
    Returns: 'write', 'read', or 'none'
    """
    if not db or not user_email:
        return 'none'

    # Hardcoded override for bootstrap admins
    if user_email in BOOTSTRAP_ADMINS:
        return 'write'
        
    try:
        user_doc = db.collection('users').document(user_email).get()
        if not user_doc.exists:
            return 'none'
            
        data = user_doc.to_dict()
        
        # Check Account Status
        if data.get('status') != 'active':
            return 'none'
            
        # Admin Override
        if data.get('role') == 'admin':
            return 'write'
            
        # Check granular permission
        perms = data.get('permissions', {})
        
        # Handle boolean migration (backward compatibility)
        raw_perm = perms.get(page_name)
        
        if raw_perm is True:
            return 'write'
        if raw_perm is False or raw_perm is None:
            # Check for string value if it was already migrated
            if isinstance(raw_perm, str):
                return raw_perm
            return 'none'
            
        return str(raw_perm)
        
    except Exception as e:
        logger.error(f"Error checking page permission: {e}")
        return 'none'


def check_page_access(db: firestore.Client, user_email: str, page_name: str) -> bool:
    """
    Check if user has AT LEAST read access to a specific page.
    Used for sidebar visibility and basic guarding.
    """
    level = get_page_permission_level(db, user_email, page_name)
    return level in ['read', 'write']


def can_write(db: firestore.Client, user_email: str, page_name: str) -> bool:
    """
    Check if user has WRITE access to a specific page.
    Used for enabling 'Save' buttons.
    """
    level = get_page_permission_level(db, user_email, page_name)
    return level == 'write'


def update_user_permissions(db: firestore.Client, operator_email: str, target_email: str, permissions: Dict) -> bool:
    """
    Update user's permissions in Firestore.
    Requires operator to be an admin.
    """
    if not db or not operator_email or not target_email:
        return False

    if not is_admin(db, operator_email):
        logger.warning(f"Unauthorized permission update attempt by {operator_email}")
        return False

    try:
        db.collection('users').document(target_email).set({
            'permissions': permissions
        }, merge=True)
        logger.info(f"Updated permissions for {target_email} by {operator_email}")
        return True
    except Exception as e:
        logger.error(f"Error updating permissions: {e}")
        return False


def set_user_role(db: firestore.Client, operator_email: str, target_email: str, role: str) -> bool:
    """
    Set a user's role (admin/user).
    """
    if not is_admin(db, operator_email):
        return False
        
    if role not in ['admin', 'user']:
        return False
        
    try:
        db.collection('users').document(target_email).set({
            'role': role
        }, merge=True)
        return True
    except Exception as e:
        logger.error(f"Error setting role: {e}")
        return False


def set_user_status(db: firestore.Client, operator_email: str, target_email: str, status: str) -> bool:
    """
    Set a user's status (active/suspended).
    """
    if not is_admin(db, operator_email):
        return False
        
    if status not in ['active', 'suspended']:
        return False
        
    try:
        db.collection('users').document(target_email).set({
            'status': status
        }, merge=True)
        return True
    except Exception as e:
        logger.error(f"Error setting status: {e}")
        return False


def get_all_users(db: firestore.Client) -> List[Dict]:
    """
    Get list of all users from Firestore
    """
    if not db:
        return []

    try:
        users_ref = db.collection('users')
        users = []
        for doc in users_ref.stream():
            user_data = doc.to_dict()
            user_data['email'] = doc.id  # Add document ID as email
            
            # Normalize for UI
            if 'role' not in user_data:
                user_data['role'] = 'user'
            if 'status' not in user_data:
                user_data['status'] = 'active'
                
            users.append(user_data)
        return users
    except Exception as e:
        logger.error(f"Error getting all users: {e}")
        return []


def delete_user(db: firestore.Client, operator_email: str, target_email: str) -> bool:
    """
    Delete user from Firestore.
    """
    if not db or not target_email or not operator_email:
        return False

    if not is_admin(db, operator_email):
        return False

    if target_email == operator_email:
        logger.warning("Admin cannot delete themselves")
        return False

    try:
        db.collection('users').document(target_email).delete()
        logger.info(f"Deleted user: {target_email}")
        return True
    except Exception as e:
        logger.error(f"Error deleting user: {e}")
        return False


# ==================== OAuth & Preference ====================

def store_oauth_tokens(
    db: firestore.Client,
    user_email: str,
    access_token: str,
    refresh_token: str,
    token_expiry: datetime,
    token_scope: str
) -> bool:
    """
    Store OAuth tokens in Firestore
    """
    if not db or not user_email:
        return False

    try:
        db.collection('users').document(user_email).set({
            'oauth_tokens': {
                'access_token': access_token,
                'refresh_token': refresh_token,
                'token_expiry': token_expiry,
                'token_scope': token_scope,
                'last_updated': firestore.SERVER_TIMESTAMP
            }
        }, merge=True)
        logger.info(f"Stored OAuth tokens for {user_email}")
        return True
    except Exception as e:
        logger.error(f"Error storing OAuth tokens: {e}")
        return False


def get_oauth_tokens(db: firestore.Client, user_email: str) -> Optional[Dict]:
    """
    Get OAuth tokens from Firestore
    """
    if not db or not user_email:
        return None

    try:
        user_doc = db.collection('users').document(user_email).get()
        if user_doc.exists:
            user_data = user_doc.to_dict()
            return user_data.get('oauth_tokens', {})
        return None
    except Exception as e:
        logger.error(f"Error getting OAuth tokens: {e}")
        return None


def refresh_oauth_tokens(db: firestore.Client, user_email: str) -> bool:
    """
    Refresh expired OAuth tokens using the refresh token
    """
    if not db or not user_email:
        return False

    try:
        # Get current tokens
        oauth_tokens = get_oauth_tokens(db, user_email)
        if not oauth_tokens or 'refresh_token' not in oauth_tokens:
            logger.warning(f"No refresh token available for {user_email}")
            return False

        refresh_token = oauth_tokens['refresh_token']

        # Get OAuth credentials from environment/secrets
        try:
            import streamlit as st
            if hasattr(st, 'secrets') and 'google_oauth' in st.secrets:
                client_id = st.secrets['google_oauth']['client_id']
                client_secret = st.secrets['google_oauth']['client_secret']
            else:
                client_id = os.environ.get('GOOGLE_CLIENT_ID')
                client_secret = os.environ.get('GOOGLE_CLIENT_SECRET')
        except:
            client_id = os.environ.get('GOOGLE_CLIENT_ID')
            client_secret = os.environ.get('GOOGLE_CLIENT_SECRET')

        if not client_id or not client_secret:
            logger.error("OAuth credentials not configured")
            return False

        # Call Google token endpoint
        token_url = 'https://oauth2.googleapis.com/token'
        token_data = {
            'refresh_token': refresh_token,
            'client_id': client_id,
            'client_secret': client_secret,
            'grant_type': 'refresh_token'
        }

        response = requests.post(token_url, data=token_data)

        if response.status_code == 200:
            token_json = response.json()
            new_access_token = token_json['access_token']
            expires_in = token_json.get('expires_in', 3600)
            new_expiry = datetime.utcnow() + timedelta(seconds=expires_in)

            # Update Firestore with new access token
            db.collection('users').document(user_email).set({
                'oauth_tokens': {
                    'access_token': new_access_token,
                    'token_expiry': new_expiry,
                    'last_refreshed': firestore.SERVER_TIMESTAMP
                }
            }, merge=True)

            logger.info(f"Refreshed OAuth tokens for {user_email}")
            return True
        else:
            logger.error(f"Token refresh failed: {response.status_code} {response.text}")
            return False

    except Exception as e:
        logger.error(f"Error refreshing OAuth tokens: {e}")
        return False


def revoke_oauth_tokens(db: firestore.Client, user_email: str) -> bool:
    """
    Revoke OAuth tokens for a user (e.g., on logout)
    """
    if not db or not user_email:
        return False

    try:
        # Get access token to revoke
        oauth_tokens = get_oauth_tokens(db, user_email)
        if oauth_tokens and 'access_token' in oauth_tokens:
            access_token = oauth_tokens['access_token']

            # Revoke via Google
            revoke_url = 'https://oauth2.googleapis.com/revoke'
            try:
                requests.post(revoke_url, data={'token': access_token})
            except Exception as e:
                logger.warning(f"Could not revoke token with Google: {e}")

        # Clear from Firestore
        db.collection('users').document(user_email).set({
            'oauth_tokens': firestore.DELETE_FIELD
        }, merge=True)

        logger.info(f"Revoked OAuth tokens for {user_email}")
        return True
    except Exception as e:
        logger.error(f"Error revoking OAuth tokens: {e}")
        return False


def get_user_database_preference(db: firestore.Client, user_email: str) -> Optional[Dict]:
    """
    Get user's database preference from Firestore
    """
    if not db or not user_email:
        return None

    try:
        user_doc = db.collection('users').document(user_email).get()
        if user_doc.exists:
            user_data = user_doc.to_dict()
            return user_data.get('database_preference', None)
        return None
    except Exception as e:
        logger.error(f"Error getting database preference: {e}")
        # If API is disabled (403) or database missing (404), swallow error
        error_str = str(e)
        if any(x in error_str for x in ["403", "404", "SERVICE_DISABLED", "does not exist", "NotFound"]):
             return None
        return None


def set_user_database_preference(db: firestore.Client, user_email: str, config: Dict) -> bool:
    """
    Set user's database preference in Firestore
    """
    if not db or not user_email or not config:
        return False

    try:
        # Add timestamp
        config['last_updated'] = firestore.SERVER_TIMESTAMP

        db.collection('users').document(user_email).set({
            'database_preference': config
        }, merge=True)

        logger.info(f"Set database preference for {user_email}: {config.get('connection_type')}")
        return True
    except Exception as e:
        logger.error(f"Error setting database preference: {e}")
        return False

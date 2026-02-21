"""
Database Configuration Manager
Handles loading, saving, and validating database connection settings
"""

import json
import os
from pathlib import Path
from typing import Dict, Optional
from datetime import datetime
import logging

logger = logging.getLogger(__name__)

# Path to the configuration file
CONFIG_FILE = Path(__file__).parent.parent / "config" / "database_config.json"

# Default configuration
# NOTE: For non-SuperAdmin users, MotherDuck portfolio_cloud is enforced in get_user_database_config()
# This default is only used for SuperAdmins without a preference or for application-level fallback
DEFAULT_CONFIG = {
    "connection_type": "motherduck",
    "local_path": str(Path(__file__).parent.parent / "DuckDB" / "portfolio.duckdb"),
    "motherduck_database": "portfolio_cloud",
    "last_updated": datetime.now().isoformat()
}


def load_database_config() -> Dict:
    """
    Load database configuration from JSON file.

    Returns:
        dict: Configuration dictionary with connection settings

    If file doesn't exist or is invalid, returns default configuration
    and creates the file.
    """
    try:
        if CONFIG_FILE.exists():
            with open(CONFIG_FILE, 'r') as f:
                config = json.load(f)

            # Validate the loaded config
            if validate_config(config):
                return config
            else:
                logger.debug("Invalid config file, using defaults")
                return _create_default_config()
        else:
            logger.debug("Config file not found, creating with defaults")
            return _create_default_config()

    except Exception as e:
        logger.error(f"Error loading config: {e}, using defaults")
        return _create_default_config()


def save_database_config(config: Dict) -> bool:
    """
    Save database configuration to JSON file.

    Args:
        config: Configuration dictionary to save

    Returns:
        bool: True if save was successful, False otherwise
    """
    try:
        # Validate before saving
        if not validate_config(config):
            logger.error("Invalid configuration, not saving")
            return False

        # Update timestamp
        config["last_updated"] = datetime.now().isoformat()

        # Ensure directory exists
        CONFIG_FILE.parent.mkdir(parents=True, exist_ok=True)

        # Write to file
        with open(CONFIG_FILE, 'w') as f:
            json.dump(config, f, indent=2)

        logger.debug(f"Configuration saved successfully: {config['connection_type']}")
        return True

    except Exception as e:
        logger.error(f"Error saving config: {e}")
        return False


def validate_config(config: Dict) -> bool:
    """
    Validate configuration structure.

    Args:
        config: Configuration dictionary to validate

    Returns:
        bool: True if configuration is valid, False otherwise
    """
    required_fields = ["connection_type", "local_path", "motherduck_database"]

    # Check all required fields exist
    for field in required_fields:
        if field not in config:
            logger.error(f"Missing required field: {field}")
            return False

    # Validate connection_type
    if config["connection_type"] not in ["local", "motherduck"]:
        logger.error(f"Invalid connection_type: {config['connection_type']}")
        return False

    # Validate local_path format (should be a non-empty string)
    if not isinstance(config["local_path"], str) or not config["local_path"]:
        logger.error("Invalid local_path")
        return False

    # Validate motherduck_database format
    if not isinstance(config["motherduck_database"], str) or not config["motherduck_database"]:
        logger.error("Invalid motherduck_database")
        return False

    return True


def get_current_connection_string() -> str:
    """
    Get the current connection string based on configuration.

    Returns:
        str: Connection string for DuckDB (either file path or md: string)
    """
    config = load_database_config()

    if config["connection_type"] == "motherduck":
        # Return MotherDuck connection string
        return f"md:{config['motherduck_database']}"
    else:
        # Return local file path
        return config["local_path"]


def get_connection_type() -> str:
    """
    Get the current connection type.

    Returns:
        str: Either 'local' or 'motherduck'
    """
    config = load_database_config()
    return config["connection_type"]


def _create_default_config() -> Dict:
    """
    Create default configuration file.

    Returns:
        dict: Default configuration
    """
    try:
        # Ensure directory exists
        CONFIG_FILE.parent.mkdir(parents=True, exist_ok=True)

        # Write default config
        with open(CONFIG_FILE, 'w') as f:
            json.dump(DEFAULT_CONFIG, f, indent=2)

        logger.debug("Created default configuration file")
        return DEFAULT_CONFIG.copy()

    except Exception as e:
        logger.error(f"Error creating default config file: {e}")
        return DEFAULT_CONFIG.copy()


def update_connection_type(connection_type: str) -> bool:
    """
    Update just the connection type in the configuration.

    Args:
        connection_type: Either 'local' or 'motherduck'

    Returns:
        bool: True if update was successful
    """
    if connection_type not in ["local", "motherduck"]:
        logger.error(f"Invalid connection type: {connection_type}")
        return False

    config = load_database_config()
    config["connection_type"] = connection_type
    return save_database_config(config)


def update_local_path(local_path: str) -> bool:
    """
    Update the local database path in the configuration.

    Args:
        local_path: Path to local DuckDB file

    Returns:
        bool: True if update was successful
    """
    config = load_database_config()
    config["local_path"] = local_path
    return save_database_config(config)


def update_motherduck_database(database_name: str) -> bool:
    """
    Update the MotherDuck database name in the configuration.

    Args:
        database_name: Name of MotherDuck database (without md: prefix)

    Returns:
        bool: True if update was successful
    """
    config = load_database_config()
    config["motherduck_database"] = database_name
    return save_database_config(config)


# ==================== NEW: User-Specific Database Preferences ====================


def get_user_database_config(user_email: str) -> Dict:
    """
    Load user-specific database configuration from Firestore.

    IMPORTANT: Non-SuperAdmin users are restricted to MotherDuck (portfolio_cloud) only.
    Only SuperAdmins can configure custom database connections or use local DuckDB.

    Args:
        user_email: User's email address

    Returns:
        dict: User's database configuration
              - SuperAdmins: Their custom preference or application default
              - All other users: MotherDuck portfolio_cloud (enforced)
    """
    # Default MotherDuck config for non-SuperAdmin users
    MOTHERDUCK_DEFAULT = {
        'connection_type': 'motherduck',
        'local_path': '',
        'motherduck_database': 'portfolio_cloud',
        'motherduck_token': '',
        'last_updated': datetime.now().isoformat()
    }

    if not user_email:
        logger.warning("No user email provided, using MotherDuck default")
        return MOTHERDUCK_DEFAULT

    try:
        from utils.firestore_client import get_firestore_client
        from utils.user_manager import get_user_database_preference, is_super_admin

        db = get_firestore_client()
        if not db:
            logger.warning("Firestore not configured, using MotherDuck default")
            return MOTHERDUCK_DEFAULT

        # Check if user is SuperAdmin - only they can use custom database configs
        if not is_super_admin(db, user_email):
            # Non-SuperAdmin users always use MotherDuck portfolio_cloud
            logger.debug(f"User {user_email} is not SuperAdmin, enforcing MotherDuck portfolio_cloud")
            return MOTHERDUCK_DEFAULT

        # SuperAdmin: Get their custom preference from Firestore
        user_pref = get_user_database_preference(db, user_email)

        if user_pref and user_pref.get('connection_type'):
            logger.debug(f"SuperAdmin {user_email} using custom config: {user_pref.get('connection_type')}")
            # Return SuperAdmin's preference (already has all fields from Firestore)
            return {
                'connection_type': user_pref['connection_type'],
                'local_path': user_pref.get('local_path', ''),
                'motherduck_database': user_pref.get('motherduck_database', 'portfolio_cloud'),
                'motherduck_token': user_pref.get('motherduck_token', ''),  # User-specific token
                'last_updated': user_pref.get('last_updated', datetime.now().isoformat())
            }

        # SuperAdmin with no preference yet, return application default
        logger.debug(f"No preference found for SuperAdmin {user_email}, using application default")
        return load_database_config()

    except Exception as e:
        logger.error(f"Error loading user database config: {e}, using MotherDuck default")
        return MOTHERDUCK_DEFAULT


def set_user_database_config(user_email: str, config: dict) -> bool:
    """
    Save user-specific database configuration to Firestore.

    Args:
        user_email: User's email address
        config: Database configuration dict with keys:
            - connection_type: 'local' or 'motherduck'
            - local_path: (optional) Path for local DuckDB
            - motherduck_database: (optional) MotherDuck database name
            - motherduck_token: (optional) MotherDuck access token

    Returns:
        bool: True if successful, False otherwise
    """
    if not user_email or not config:
        logger.error("Invalid parameters for set_user_database_config")
        return False

    try:
        from utils.firestore_client import get_firestore_client
        from utils.user_manager import set_user_database_preference

        db = get_firestore_client()
        if not db:
            logger.error("Firestore not configured, cannot save user preference")
            return False

        # Save to Firestore
        success = set_user_database_preference(db, user_email, config)

        if success:
            logger.info(f"Saved database preference for {user_email}: {config.get('connection_type')}")

        return success

    except Exception as e:
        logger.error(f"Error saving user database config: {e}")
        return False

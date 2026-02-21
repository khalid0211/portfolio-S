"""
Firestore Database Client for Google Cloud Run
Supports ADC (Application Default Credentials) and local development
"""

import os
import json
import logging
from google.cloud import firestore
from google.oauth2 import service_account

logger = logging.getLogger(__name__)

# Global client instance (singleton pattern)
_firestore_client = None


def get_firestore_client():
    """
    Get Firestore client with cloud-deployment priority

    Priority:
    1. ADC (Google Cloud Run) - Automatic, no configuration needed
    2. Streamlit secrets (local development) - From .streamlit/secrets.toml
    3. GOOGLE_APPLICATION_CREDENTIALS env var - Service account JSON path
    4. FIRESTORE_PROJECT_ID env var - Explicit project ID

    Returns:
        firestore.Client: Firestore client instance or None if not configured
    """
    global _firestore_client

    # Return existing client if already initialized
    if _firestore_client is not None:
        return _firestore_client

    # Try ADC first (works automatically in Cloud Run)
    try:
        _firestore_client = firestore.Client()
        logger.info("Firestore client initialized using ADC")
        return _firestore_client
    except Exception as e:
        logger.debug(f"ADC not available: {e}")

    # Try Streamlit secrets (local development)
    try:
        import streamlit as st
        if hasattr(st, 'secrets') and 'firestore' in st.secrets:
            creds_dict = dict(st.secrets['firestore'])
            credentials = service_account.Credentials.from_service_account_info(creds_dict)
            project_id = creds_dict['project_id']
            _firestore_client = firestore.Client(project=project_id, credentials=credentials)
            logger.info("Firestore client initialized from Streamlit secrets")
            return _firestore_client
    except Exception as e:
        logger.debug(f"Could not load from Streamlit secrets: {e}")

    # Try service account from GOOGLE_APPLICATION_CREDENTIALS environment variable
    creds_path = os.environ.get("GOOGLE_APPLICATION_CREDENTIALS")
    if creds_path and os.path.exists(creds_path):
        try:
            _firestore_client = firestore.Client.from_service_account_json(creds_path)
            logger.info(f"Firestore client initialized from service account file: {creds_path}")
            return _firestore_client
        except Exception as e:
            logger.error(f"Failed to load service account from {creds_path}: {e}")

    # Try explicit project ID (last resort)
    project_id = os.environ.get("FIRESTORE_PROJECT_ID")
    if project_id:
        try:
            _firestore_client = firestore.Client(project=project_id)
            logger.info(f"Firestore client initialized for project: {project_id}")
            return _firestore_client
        except Exception as e:
            logger.error(f"Failed to initialize Firestore with project ID {project_id}: {e}")

    # All methods failed
    logger.error("Could not initialize Firestore client - no credentials available")
    logger.error("For Cloud Run: Ensure service account has 'roles/datastore.user' IAM role")
    logger.error("For local dev: Add Firestore credentials to .streamlit/secrets.toml")
    return None


def is_firestore_configured():
    """
    Check if Firestore is configured and accessible

    Returns:
        bool: True if Firestore client can be initialized, False otherwise
    """
    return get_firestore_client() is not None


def reset_firestore_client():
    """
    Reset the global Firestore client instance
    Useful for testing or forcing reconnection
    """
    global _firestore_client
    _firestore_client = None
    logger.info("Firestore client reset")

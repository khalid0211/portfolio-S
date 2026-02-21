"""
Authentication module for Portfolio Management Suite
Handles Google OAuth authentication and session management
"""

import os
import sys
import warnings
import logging

# Suppress Google Cloud ALTS warnings (harmless warnings when not running on GCP)
# These must be set BEFORE importing any Google/gRPC libraries
os.environ['GRPC_VERBOSITY'] = 'ERROR'
os.environ['GRPC_TRACE'] = ''
os.environ['GRPC_LOG_SEVERITY_LEVEL'] = 'ERROR'
os.environ['GLOG_minloglevel'] = '2'
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

# Suppress Python warnings
warnings.filterwarnings('ignore', category=UserWarning, module='google')
warnings.filterwarnings('ignore', category=FutureWarning, module='google')

# Suppress absl logging
try:
    import absl.logging
    absl.logging.set_verbosity(absl.logging.ERROR)
    absl.logging.set_stderrthreshold(absl.logging.ERROR)
except ImportError:
    pass

# Suppress gRPC logging
logging.getLogger('grpc').setLevel(logging.ERROR)
logging.getLogger('google').setLevel(logging.ERROR)
logging.getLogger('google.auth').setLevel(logging.ERROR)

import streamlit as st
import json
from urllib.parse import urlencode
import requests

# OAuth Configuration - Try environment variables first (Render), then Streamlit secrets
GOOGLE_CLIENT_ID = os.environ.get("GOOGLE_CLIENT_ID", "")
GOOGLE_CLIENT_SECRET = os.environ.get("GOOGLE_CLIENT_SECRET", "")

# If not in environment, try Streamlit secrets (local/Streamlit Cloud)
if not GOOGLE_CLIENT_ID or not GOOGLE_CLIENT_SECRET:
    try:
        GOOGLE_CLIENT_ID = st.secrets.get("google_oauth", {}).get("client_id", "")
        GOOGLE_CLIENT_SECRET = st.secrets.get("google_oauth", {}).get("client_secret", "")
    except:
        pass  # Will be handled by validation later

# This will be set dynamically in check_authentication()
REDIRECT_URI = None

# Google OAuth endpoints
AUTHORIZATION_BASE_URL = "https://accounts.google.com/o/oauth2/v2/auth"
TOKEN_URL = "https://oauth2.googleapis.com/token"
USERINFO_URL = "https://www.googleapis.com/oauth2/v2/userinfo"


def check_authentication():
    """
    Check if user is authenticated using Google OAuth
    Returns True if authenticated, False otherwise
    """
    global REDIRECT_URI

    # Dynamically detect redirect URI based on current URL
    if REDIRECT_URI is None:
        # Try to get the current URL from Streamlit's context
        try:
            # Check environment variable first
            env_uri = os.environ.get("REDIRECT_URI", "")
            if env_uri:
                REDIRECT_URI = env_uri
            # Check if hostname indicates Streamlit Cloud
            elif os.environ.get("HOSTNAME") == "streamlit":
                REDIRECT_URI = "https://portfolio-suite.streamlit.app/"
            else:
                # Default to localhost
                REDIRECT_URI = "http://localhost:8501"
        except:
            REDIRECT_URI = "http://localhost:8501"

    # Initialize session state for authentication
    if 'authenticated' not in st.session_state:
        st.session_state.authenticated = False

    if 'user_info' not in st.session_state:
        st.session_state.user_info = None

    if 'user_email' not in st.session_state:
        st.session_state.user_email = None

    # If already authenticated, return True
    if st.session_state.authenticated:
        return True

    # Show login page
    st.markdown("# 🔐 Loading Portfolio Management Suite")
    st.markdown("Authentication in progress")

    # Check if OAuth credentials are configured
    if not GOOGLE_CLIENT_ID or not GOOGLE_CLIENT_SECRET:
        st.error("❌ OAuth credentials not configured")
        st.info("""
        **Setup Instructions:**
        1. Create a Google Cloud Project
        2. Enable Google OAuth 2.0
        3. Add credentials to `.streamlit/secrets.toml`
        """)
        return False

    # Check if we have an authorization code in the URL
    query_params = st.query_params

    if 'code' in query_params:
        # Exchange authorization code for access token
        auth_code = query_params['code']

        try:
            # Exchange code for token
            token_data = {
                'code': auth_code,
                'client_id': GOOGLE_CLIENT_ID,
                'client_secret': GOOGLE_CLIENT_SECRET,
                'redirect_uri': REDIRECT_URI,
                'grant_type': 'authorization_code'
            }

            token_response = requests.post(TOKEN_URL, data=token_data)

            if token_response.status_code == 200:
                token_json = token_response.json()
                access_token = token_json.get('access_token')

                # Get user info using access token
                headers = {'Authorization': f'Bearer {access_token}'}
                userinfo_response = requests.get(USERINFO_URL, headers=headers)

                if userinfo_response.status_code == 200:
                    user_info = userinfo_response.json()

                    # Store user information in session state
                    st.session_state.authenticated = True
                    st.session_state.user_email = user_info.get('email')
                    st.session_state.user_info = user_info

                    # Create/update user record in Firestore
                    # Suppress stderr during Firestore import/initialization
                    import io
                    import sys
                    old_stderr = sys.stderr
                    sys.stderr = io.StringIO()

                    from utils.firestore_client import get_firestore_client
                    from utils.user_manager import ensure_user_exists

                    # Restore stderr
                    sys.stderr = old_stderr

                    db = get_firestore_client()
                    if db is not None:
                        success = ensure_user_exists(db, user_info.get('email'), user_info.get('name', 'User'))
                        if not success:
                            st.warning("Could not create user profile, but authentication succeeded")

                    # Clear the code from URL
                    st.query_params.clear()
                    st.rerun()
                else:
                    st.error(f"Failed to get user info: {userinfo_response.status_code}")
            else:
                st.error(f"Failed to exchange code for token: {token_response.status_code}")
                st.code(token_response.text)

        except Exception as e:
            st.error(f"Authentication error: {str(e)}")
            st.info("Please try signing in again or contact support if the issue persists.")

        return False

    # Show Google Sign-In button
    st.markdown("### Sign in with Google")

    # Generate OAuth URL
    oauth_params = {
        'client_id': GOOGLE_CLIENT_ID,
        'redirect_uri': REDIRECT_URI,
        'response_type': 'code',
        'scope': 'openid email profile',
        'access_type': 'online',
        'prompt': 'select_account'
    }

    auth_url = f"{AUTHORIZATION_BASE_URL}?{urlencode(oauth_params)}"

    # Display login button - simple link approach
    st.markdown(f"""
    <div style="text-align: left; margin: 20px 0;">
        <a href="{auth_url}" style="
            display: inline-block;
            background-color: #4285f4;
            color: white;
            padding: 12px 24px;
            text-decoration: none;
            border-radius: 4px;
            font-weight: 500;
            font-size: 16px;
            transition: background-color 0.3s;
        " onmouseover="this.style.backgroundColor='#357ae8'"
           onmouseout="this.style.backgroundColor='#4285f4'">
            🔐 Sign in with Google
        </a>
    </div>
    """, unsafe_allow_html=True)

    return False


def require_page_access(page_name, page_title):
    """
    Check if user has access to a specific page
    Stops execution if access is denied

    Args:
        page_name: Internal page identifier
        page_title: Display name of the page
    """
    from utils.user_manager import check_page_access
    from utils.firestore_client import get_firestore_client

    user_email = st.session_state.get('user_email')
    db = get_firestore_client()

    if not check_page_access(db, user_email, page_name):
        st.error(f"❌ Access Denied to {page_title}")
        st.info("You do not have permission to access this page. Please contact your administrator.")
        st.stop()


def show_user_info_sidebar():
    """
    Display user information and logout button in sidebar
    """
    if st.session_state.get('authenticated') and st.session_state.get('user_info'):
        user_info = st.session_state.user_info

        with st.sidebar:
            st.markdown("---")
            st.markdown("### 👤 User Info")

            # Show user picture if available
            if user_info.get('picture'):
                st.image(user_info['picture'], width=60)

            st.markdown(f"**{user_info.get('name', 'User')}**")
            st.markdown(f"_{user_info.get('email', 'N/A')}_")

            # Logout button
            if st.button("🚪 Logout", width="stretch"):
                # Clear session state
                st.session_state.authenticated = False
                st.session_state.user_info = None
                st.session_state.user_email = None
                st.rerun()

"""
User Management - Portfolio Management Suite
Admin interface for managing user access permissions
"""

import streamlit as st
import pandas as pd
from datetime import datetime
from utils.auth import check_authentication, require_page_access
from utils.user_manager import get_all_users, update_user_permissions, is_admin, delete_user
from utils.firestore_client import get_firestore_client, is_firestore_configured

# Initialize Firestore client
db = get_firestore_client()

# Page configuration
st.set_page_config(
    page_title="User Management - Portfolio Suite",
    page_icon="👥",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# Check authentication and page access
if not check_authentication():
    st.stop()

require_page_access('user_management', 'User Management')

# Custom CSS
st.markdown("""
<style>
    .main-header {
        text-align: center;
        color: #2c3e50;
        font-size: 2.2rem;
        font-weight: 600;
        margin-bottom: 1rem;
        border-bottom: 3px solid #3498db;
        padding-bottom: 0.5rem;
    }
    .user-card {
        background: rgba(255, 255, 255, 0.1);
        padding: 1.5rem;
        border-radius: 15px;
        margin: 1rem 0;
        border: 1px solid rgba(255, 255, 255, 0.2);
    }
    .permission-toggle {
        margin: 0.5rem 0;
    }
</style>
""", unsafe_allow_html=True)

# Main header
st.markdown('<h1 class="main-header">👥 User Management</h1>', unsafe_allow_html=True)
st.markdown("Manage user access permissions for the Portfolio Management Suite")

# Check if Firestore is configured
if not is_firestore_configured():
    st.error("❌ Firestore database is not configured")
    st.info("""
    **Firebase Configuration Required:**
    User management requires Firebase Firestore to be configured.
    Please add Firebase credentials to `.streamlit/secrets.toml`
    """)
    st.stop()

# Get current user
current_user_email = st.session_state.get('user_email')

if not is_admin(db, current_user_email):
    st.error("❌ Access Denied")
    st.info("Only administrators can access the User Management page.")
    st.stop()

# Page permissions configuration
PAGE_PERMISSIONS = {
    'portfolio_management': '📁 Portfolio Management',
    'file_management': '📥 Load Progress Data',
    'manual_data_entry': '📝 Enter Progress Data',
    'baseline_management': '📊 Baseline Management',
    'project_analysis': '🔍 Project Analysis',
    'portfolio_analysis': '📈 Portfolio Analysis',
    'portfolio_charts': '📉 Portfolio Charts',
    'strategic_factors': '🎯 Strategic Factors',
    'sdg_management': '🌍 SDG Management',
    'ai_assistant': '🤖 AI Assistant',
    'user_management': '👥 User Management',
    'database_diagnostics': '🔧 Database Diagnostics'
}

# Fetch all users
users = get_all_users(db)

if not users:
    st.warning("⚠️ No users found in the database")
    st.info("Users will be automatically added when they sign in with Google OAuth")
else:
    st.success(f"✅ Managing {len(users)} user(s)")

    # Create summary table with color coding
    st.markdown("## 📋 Users Overview")
    st.caption("💡 Red background = No access permissions | Green background = Administrator | No background = Has all modules except User Management | Yellow = Has access but missing Portfolio Charts & User Management | Light Gray = Other access combinations")

    def get_user_field(user, field_name, default=None):
        """Helper to get user field from either modern (profile) or legacy structure"""
        # Try modern structure first (nested in profile)
        if 'profile' in user:
            value = user.get('profile', {}).get(field_name)
            if value is not None:
                return value
        # Fall back to legacy structure (top-level)
        value = user.get(field_name)
        if value is not None:
            return value
        return default

    def format_date_short(date_obj):
        if date_obj:
            try:
                if hasattr(date_obj, 'strftime'):
                    return date_obj.strftime('%Y-%m-%d %H:%M')
                else:
                    return str(date_obj)[:16]
            except:
                return "Unknown"
        return "Never"

    def check_has_access(user):
        """Check if user has any access permissions"""
        user_email = get_user_field(user, 'email')
        if is_admin(db, user_email):
            return True
        permissions = user.get('permissions', {})
        return any(permissions.values())

    def check_has_other_modules(user):
        """Check if user has access to modules other than admin and portfolio_charts, but is missing those two"""
        user_email = get_user_field(user, 'email')
        if is_admin(db, user_email):
            return False  # Admins are handled separately
        permissions = user.get('permissions', {})

        # Check if user is missing both portfolio_charts and user_management
        has_portfolio_charts = permissions.get('portfolio_charts', False)
        has_user_management = permissions.get('user_management', False)

        # If they have either of these, they don't qualify for yellow
        if has_portfolio_charts or has_user_management:
            return False

        # Check if user has any other modules enabled
        for module, enabled in permissions.items():
            if enabled and module not in ['user_management', 'portfolio_charts']:
                return True
        return False

    def check_has_all_except_user_management(user):
        """Check if user has access to all modules except user_management"""
        user_email = get_user_field(user, 'email')
        if is_admin(db, user_email):
            return False  # Admins are handled separately
        permissions = user.get('permissions', {})

        # Must NOT have user_management
        if permissions.get('user_management', False):
            return False

        # Check if user has all other modules (excluding user_management)
        required_modules = ['portfolio_management', 'file_management', 'manual_data_entry', 'baseline_management',
                          'project_analysis', 'portfolio_analysis', 'portfolio_charts', 'strategic_factors',
                          'sdg_management', 'database_diagnostics']

        for module in required_modules:
            if not permissions.get(module, False):
                return False

        return True

    summary_data = []
    for idx, user in enumerate(users):
        has_access = check_has_access(user)
        has_other_modules = check_has_other_modules(user)
        has_all_except_user_mgmt = check_has_all_except_user_management(user)

        summary_data.append({
            'Select': f"user_{idx}",
            'Name': get_user_field(user, 'name', 'User'),
            'Email': get_user_field(user, 'email', 'Unknown'),
            'Created': format_date_short(get_user_field(user, 'created_at')),
            'Last Access': format_date_short(get_user_field(user, 'last_access_date')),
            'Access Count': get_user_field(user, 'access_count', 0),
            'Role': '🔑 Admin' if is_admin(db, get_user_field(user, 'email')) else '👤 User',
            'Status': '✅ Access' if has_access else '🚫 No Access',
            'has_access': has_access,
            'has_other_modules': has_other_modules,
            'has_all_except_user_mgmt': has_all_except_user_mgmt
        })

    summary_df = pd.DataFrame(summary_data)

    # Apply color styling based on Status column
    # First create a copy of the dataframe for styling that includes the helper columns
    def highlight_no_access(row):
        if row['Status'] == '🚫 No Access':
            style = 'background-color: #ff6b6b; color: white'
        elif row['Role'] == '🔑 Admin':
            style = 'background-color: #51cf66; color: white'
        elif row['has_all_except_user_mgmt']:
            style = ''
        elif row['has_other_modules']:
            style = 'background-color: #ffd43b; color: black'
        else:
            style = 'background-color: #e6e6e6; color: black'

        # Return style for all columns
        return [style] * len(row)

    # Apply styling to full dataframe with helper columns
    styled_full_df = summary_df.style.apply(highlight_no_access, axis=1)

    # Convert styled dataframe to HTML and then parse to drop columns
    # Or simply drop columns first and reapply styling
    display_df = summary_df.drop(columns=['has_access', 'has_other_modules', 'has_all_except_user_mgmt', 'Select'])

    # Create a new styling function that references the original dataframe
    def highlight_no_access_display(row):
        # Get the original row data by matching the displayed row
        orig_row = summary_df.iloc[row.name]

        if orig_row['Status'] == '🚫 No Access':
            style = 'background-color: #ff6b6b; color: white'
        elif orig_row['Role'] == '🔑 Admin':
            style = 'background-color: #51cf66; color: white'
        elif orig_row['has_all_except_user_mgmt']:
            style = ''
        elif orig_row['has_other_modules']:
            style = 'background-color: #ffd43b; color: black'
        else:
            style = 'background-color: #e6e6e6; color: black'

        return [style] * len(row)

    styled_df = display_df.style.apply(highlight_no_access_display, axis=1)

    st.markdown("**Click on a row in the table below to select a user to manage:**")

    # Use on_select to capture row selection
    selection = st.dataframe(
        styled_df,
        width="stretch",
        hide_index=True,
        on_select="rerun",
        selection_mode="single-row",
        key="user_table_selection"
    )

    # Get selected user index from table selection
    if selection and selection.selection and selection.selection.rows:
        user_idx = selection.selection.rows[0]
        st.session_state['selected_user_idx'] = user_idx
    elif 'selected_user_idx' not in st.session_state:
        st.session_state['selected_user_idx'] = 0
        user_idx = 0
    else:
        user_idx = st.session_state['selected_user_idx']

    st.markdown("---")
    st.markdown("## 👥 Selected User Details")

    user = users[user_idx]

    # Helper function for this section too
    def get_field(field_name, default=None):
        """Helper to get user field from either modern (profile) or legacy structure"""
        if 'profile' in user:
            return user.get('profile', {}).get(field_name, default)
        return user.get(field_name, default)

    user_email = get_field('email', 'Unknown')
    user_name = get_field('name', 'User')
    permissions = user.get('permissions', {})
    created_at = get_field('created_at')
    last_access_date = get_field('last_access_date')
    access_count = get_field('access_count', 0)

    # Format dates
    def format_date(date_obj):
        if date_obj:
            try:
                if hasattr(date_obj, 'strftime'):
                    return date_obj.strftime('%Y-%m-%d %H:%M:%S')
                else:
                    return str(date_obj)
            except:
                return "Unknown"
        return "Never"

    created_at_str = format_date(created_at)
    last_access_str = format_date(last_access_date)

    # User details section
    st.markdown("---")
    st.markdown(f"### 👤 {user_name}")

    col1, col2, col3 = st.columns(3)

    with col1:
        st.markdown(f"**Email:** {user_email}")
        st.markdown(f"**Created Date:** {created_at_str}")

    with col2:
        st.markdown(f"**Last Access:** {last_access_str}")
        st.markdown(f"**Access Count:** {access_count} time(s)")

    with col3:
        # Show admin badge or current user indicator
        if is_admin(db, user_email):
            st.info("🔑 **Administrator** (Full Access)")
        if user_email == current_user_email:
            st.info("📌 **This is you**")

    st.markdown("---")
    st.markdown("### 🔐 Access Permissions")

    # Use a form to prevent rerun on every checkbox click
    with st.form(key=f"permissions_form_{user_idx}"):
        # Create permission toggles
        updated_permissions = {}
        cols = st.columns(2)

        for idx, (perm_key, perm_label) in enumerate(PAGE_PERMISSIONS.items()):
            col = cols[idx % 2]

            with col:
                # Disable toggle for admins (they always have full access)
                disabled = is_admin(db, user_email)

                current_value = permissions.get(perm_key, False)

                # If admin, always show as enabled
                if disabled:
                    st.checkbox(
                        perm_label,
                        value=True,
                        key=f"perm_{user_idx}_{perm_key}",
                        disabled=True,
                        help="Administrators always have full access"
                    )
                    updated_permissions[perm_key] = True
                else:
                    new_value = st.checkbox(
                        perm_label,
                        value=current_value,
                        key=f"perm_{user_idx}_{perm_key}"
                    )
                    updated_permissions[perm_key] = new_value

        # Save button inside form
        st.markdown("---")
        col1_form, col2_form = st.columns([1, 3])
        with col1_form:
            save_submitted = st.form_submit_button("💾 Save Permissions", type="primary", width='stretch')
        with col2_form:
            if is_admin(db, user_email):
                st.caption("ℹ️ Administrators always have full access")
            else:
                enabled_count = sum(1 for v in updated_permissions.values() if v)
                st.caption(f"📊 {enabled_count}/{len(PAGE_PERMISSIONS)} pages enabled")

    # Handle form submission
    if save_submitted:
        if not is_admin(db, user_email):  # Only update if not admin
            success = update_user_permissions(db, current_user_email, user_email, updated_permissions)

            if success:
                st.success("✅ Permissions updated successfully!")
                st.rerun()
            else:
                st.error("❌ Failed to update permissions")
        else:
            st.info("ℹ️ Cannot modify administrator permissions")

    # Action buttons (outside form)
    st.markdown("---")
    col1, col2, col3 = st.columns([1, 1, 2])

    with col1:
        if st.button("🔄 Reset to Default", key=f"reset_{user_idx}", width='stretch'):
            if not is_admin(db, user_email):
                default_permissions = {key: False for key in PAGE_PERMISSIONS.keys()}
                success = update_user_permissions(db, current_user_email, user_email, default_permissions)

                if success:
                    st.success("✅ Permissions reset to default (all disabled)")
                    st.rerun()
                else:
                    st.error("❌ Failed to reset permissions")
            else:
                st.info("ℹ️ Cannot reset administrator permissions")

    with col2:
        # Delete button with confirmation
        delete_confirm_key = f"delete_confirm_{user_idx}"

        # Initialize confirmation state if not exists
        if delete_confirm_key not in st.session_state:
            st.session_state[delete_confirm_key] = False

        if not st.session_state[delete_confirm_key]:
            # First click: Ask for confirmation
            if st.button("🗑️ Delete User", key=f"delete_{user_idx}", disabled=is_admin(db, user_email), width='stretch'):
                if user_email == current_user_email:
                    st.error("❌ You cannot delete yourself!")
                elif is_admin(db, user_email):
                    st.error("❌ Cannot delete administrators")
                else:
                    st.session_state[delete_confirm_key] = True
                    st.rerun()
        else:
            # Show confirmation buttons
            conf_col1, conf_col2 = st.columns(2)
            with conf_col1:
                if st.button("⚠️ Confirm", key=f"delete_confirm_btn_{user_idx}", type="secondary", width='stretch'):
                    success = delete_user(db, current_user_email, user_email)
                    if success:
                        st.success(f"✅ User {user_email} deleted successfully!")
                        # Reset confirmation state
                        st.session_state[delete_confirm_key] = False
                        st.session_state['selected_user_idx'] = 0
                        st.rerun()
                    else:
                        st.error("❌ Failed to delete user")
                        st.session_state[delete_confirm_key] = False
            with conf_col2:
                if st.button("❌ Cancel", key=f"delete_cancel_{user_idx}", width='stretch'):
                    st.session_state[delete_confirm_key] = False
                    st.rerun()

    with col3:
        # Delete warning
        if st.session_state.get(f"delete_confirm_{user_idx}", False):
            st.warning("⚠️ Are you sure you want to delete this user?")

# Summary section
st.markdown("---")
st.markdown("## 📊 Summary")

col1, col2, col3 = st.columns(3)

with col1:
    st.metric("Total Users", len(users))

with col2:
    def get_email(u):
        if 'profile' in u:
            return u.get('profile', {}).get('email')
        return u.get('email')
    admin_count = sum(1 for u in users if is_admin(db, get_email(u)))
    st.metric("Administrators", admin_count)

with col3:
    def get_last_access(u):
        if 'profile' in u:
            return u.get('profile', {}).get('last_access_date')
        return u.get('last_access_date')
    active_users = sum(1 for u in users if get_last_access(u))
    st.metric("Users with Login", active_users)

# Export email addresses section
st.markdown("---")
st.markdown("## 📧 Export Email Addresses")

col1, col2 = st.columns([1, 3])

with col1:
    if st.button("📋 Generate Email List", type="primary"):
        # Extract all email addresses
        def get_user_email(u):
            if 'profile' in u:
                return u.get('profile', {}).get('email', '')
            return u.get('email', '')
        email_list = [get_user_email(u) for u in users if get_user_email(u)]
        email_string = "; ".join(email_list)

        # Store in session state to display
        st.session_state['email_export'] = email_string

with col2:
    # Display email string if generated
    if 'email_export' in st.session_state and st.session_state['email_export']:
        st.text_area(
            "Email addresses (semicolon-separated):",
            value=st.session_state['email_export'],
            height=100,
            help="Copy this string and paste it into your email program's To/Cc/Bcc field"
        )
        st.caption(f"✅ {len(users)} email address(es) ready to copy")

# Help section
with st.expander("ℹ️ User Management Help"):
    st.markdown("""
    **How User Management Works:**

    1. **Automatic User Creation**: When a user signs in with Google OAuth, they are automatically added to the database

    2. **Default Permissions**:
       - khalid0211@gmail.com gets full access automatically (Administrator)
       - Other users' default permissions are configured in the system settings

    3. **Managing Permissions**:
       - Toggle permissions on/off for each page
       - Click "Save Permissions" to apply changes
       - Use "Reset to Default" to disable all permissions

    4. **Deleting Users**:
       - Click "Delete User" button to remove a user from the database
       - Confirmation required to prevent accidental deletion
       - Cannot delete administrators or yourself
       - Deleted users can re-register by signing in again

    5. **Exporting Email Addresses**:
       - Click "Generate Email List" to create a semicolon-separated list of all user emails
       - Copy the generated string directly into your email program's To/Cc/Bcc field
       - Perfect for sending announcements or updates to all users

    6. **Administrator Role**:
       - Administrators always have full access (cannot be changed)
       - Only administrators can access this User Management page
       - Primary admin: khalid0211@gmail.com

    7. **Page Access Control**:
       - Users can only access pages they have permission for
       - If denied, they'll see an "Access Denied" message
       - User Management is only accessible to administrators

    **Note:** Changes take effect immediately. Users may need to refresh their page to see updated access.
    """)

# Show user info in sidebar
from utils.auth import show_user_info_sidebar
show_user_info_sidebar()

# MotherDuck Service Token Migration

## ✅ Migration Complete

Successfully migrated from personal MotherDuck account to Service Account authentication using service tokens.

---

## 📋 Changes Implemented

### 1. Connection String Format Update

**Before (Personal Account):**
```python
# Set token as environment variable
os.environ['motherduck_token'] = token
# Connect with just database name
conn = duckdb.connect("md:portfolio_cloud")
```

**After (Service Account):**
```python
# Include token directly in connection string
conn = duckdb.connect(f"md:portfolio_cloud?motherduck_token={token}")
```

### 2. Files Modified

#### `database/db_connection.py`
- ✅ Updated `_get_motherduck_token()` to prioritize `MOTHERDUCK_TOKEN` environment variable
- ✅ Modified `get_connection()` to build connection string with token parameter
- ✅ Added clear error messages if token is missing
- ✅ Removed browser-based authentication logic

**Key Changes:**
```python
# Extract database name and build connection string
db_name = self.db_path.replace('md:', '').replace('motherduck:', '')
connection_string = f"md:{db_name}?motherduck_token={token}"

# Connect with service token
self._conn = duckdb.connect(connection_string)
```

#### `pages/99_Database_Diagnostics.py`
- ✅ Added `import os` for environment variable access
- ✅ Created helper functions:
  - `get_motherduck_token()` - Get token from environment or secrets
  - `build_motherduck_connection_string()` - Build proper connection string
- ✅ Updated all `duckdb.connect()` calls for MotherDuck to use service token
- ✅ Added token validation with clear error messages

#### `requirements.txt`
- ✅ Added `google-cloud-secret-manager>=2.16.0` for future Secret Manager integration

### 3. Error Handling

**Clear error messages when token is missing:**

```
❌ MotherDuck service token not found!

For production (Google Cloud Run):
  Set environment variable: MOTHERDUCK_TOKEN=service_token_***
  Or use Secret Manager: gcloud run services update --set-secrets MOTHERDUCK_TOKEN=motherduck-token:latest

For local development:
  Add to .streamlit/secrets.toml:
  [motherduck]
  token = "service_token_***"

Get your service token from: https://motherduck.com/settings/access-tokens
```

---

## 🚀 Setup Instructions

### Local Development

**Option 1: Environment Variable**
```bash
export MOTHERDUCK_TOKEN="service_token_***"
```

**Option 2: Streamlit Secrets (Recommended)**

Create or update `.streamlit/secrets.toml`:
```toml
[motherduck]
token = "service_token_***"
```

### Google Cloud Run Deployment

**Option 1: Environment Variable**
```bash
gcloud run services update portfolio-app \
  --set-env-vars MOTHERDUCK_TOKEN="service_token_***"
```

**Option 2: Secret Manager (Recommended for Production)**

1. **Create secret:**
```bash
# Create the secret
echo -n "service_token_***" | gcloud secrets create motherduck-token --data-file=-

# Grant access to Cloud Run service account
PROJECT_ID=$(gcloud config get-value project)
SERVICE_ACCOUNT="${PROJECT_ID}@${PROJECT_ID}.iam.gserviceaccount.com"

gcloud secrets add-iam-policy-binding motherduck-token \
  --member="serviceAccount:${SERVICE_ACCOUNT}" \
  --role="roles/secretmanager.secretAccessor"
```

2. **Update Cloud Run deployment:**
```bash
gcloud run services update portfolio-app \
  --set-secrets MOTHERDUCK_TOKEN=motherduck-token:latest
```

---

## 🔑 Getting Your Service Token

1. Go to [MotherDuck Settings](https://motherduck.com/settings/access-tokens)
2. Click "Create New Token"
3. Select **Service Token** type
4. Name it (e.g., "Production Cloud Run")
5. Set appropriate permissions
6. Copy the token (starts with `service_token_`)
7. Store it securely in environment variable or Secret Manager

**Important:** Service tokens are long-lived and should be treated as highly sensitive credentials.

---

## 🔄 Migration Checklist

### Pre-Migration
- [x] Obtained MotherDuck service token
- [x] Identified all `duckdb.connect()` instances
- [x] Understood current authentication flow

### Implementation
- [x] Updated `db_connection.py` with service token logic
- [x] Updated `database_diagnostics.py` with helper functions
- [x] Added proper error handling for missing tokens
- [x] Added `google-cloud-secret-manager` to requirements.txt
- [x] Verified no hardcoded tokens remain
- [x] Confirmed no browser login popup logic

### Testing
- [ ] Test local connection with service token
- [ ] Test Cloud Run deployment with environment variable
- [ ] Test Cloud Run deployment with Secret Manager
- [ ] Verify error messages display correctly when token is missing
- [ ] Test database creation with service token
- [ ] Test schema initialization with service token

### Deployment
- [ ] Store service token in Secret Manager
- [ ] Update Cloud Run service with secret reference
- [ ] Verify production connection works
- [ ] Test user-specific MotherDuck connections
- [ ] Monitor logs for any authentication issues

---

## 📊 Token Priority Order

1. **`MOTHERDUCK_TOKEN` Environment Variable** (Production - Cloud Run)
   - Set via `--set-env-vars` or `--set-secrets`
   - Highest priority for production deployments

2. **Streamlit Secrets** (Local Development)
   - `.streamlit/secrets.toml` under `[motherduck]` section
   - Used for local testing and development

3. **User-Specific Token** (Per-User Override)
   - Stored in Firestore `database_preference.motherduck_token`
   - Allows individual users to use their own MotherDuck accounts

---

## 🔐 Security Best Practices

### ✅ Do's
- ✅ Store service token in Google Secret Manager for production
- ✅ Use environment variables for service tokens
- ✅ Rotate tokens periodically
- ✅ Use different tokens for development and production
- ✅ Monitor token usage in MotherDuck dashboard
- ✅ Revoke old tokens after migration

### ❌ Don'ts
- ❌ Never commit tokens to Git
- ❌ Don't include tokens in Docker images
- ❌ Don't share service tokens between environments
- ❌ Don't log tokens in application logs
- ❌ Don't use personal account tokens for production

---

## 🐛 Troubleshooting

### Issue: "MotherDuck service token not found"

**Solution:**
1. Check if `MOTHERDUCK_TOKEN` environment variable is set:
   ```bash
   echo $MOTHERDUCK_TOKEN
   ```
2. For Cloud Run, verify secret is mounted:
   ```bash
   gcloud run services describe portfolio-app --format="value(spec.template.spec.containers[0].env)"
   ```
3. Check Streamlit secrets file exists and has correct format

### Issue: "MotherDuck authentication failed"

**Solution:**
1. Verify token is a **service token** (starts with `service_token_`)
2. Check token hasn't expired or been revoked
3. Verify token has correct permissions for database
4. Test token manually:
   ```python
   import duckdb
   import os

   token = os.environ.get('MOTHERDUCK_TOKEN')
   conn = duckdb.connect(f"md:portfolio_cloud?motherduck_token={token}")
   print(conn.execute("SELECT current_database()").fetchone())
   ```

### Issue: "Connection string format error"

**Solution:**
The connection string must include the token parameter:
```python
# CORRECT
connection_string = f"md:portfolio_cloud?motherduck_token={token}"

# INCORRECT (old format)
connection_string = "md:portfolio_cloud"
os.environ['motherduck_token'] = token
```

### Issue: Browser login popup appears

**Solution:**
This should not happen with service tokens. If you see a browser popup:
1. Check if old personal account credentials are cached
2. Clear DuckDB cache: `rm -rf ~/.duckdb/`
3. Verify service token is being used in connection string
4. Restart application to clear any cached connections

---

## 📈 Benefits of Service Token Migration

1. **No Browser Interaction** - Eliminates browser-based SAML/OAuth popups
2. **Production Ready** - Designed for server environments and containers
3. **Better Security** - Fine-grained permissions, easier to rotate
4. **Team Management** - Service tokens can be shared across team securely
5. **Audit Trail** - Track which applications use which tokens
6. **Automated Deployments** - Works seamlessly in CI/CD pipelines

---

## 🔮 Future Enhancements

### Secret Manager Integration (Optional)

For enhanced security, you can fetch the token from Secret Manager at runtime:

```python
from google.cloud import secretmanager

def get_motherduck_token_from_secret_manager(project_id: str, secret_id: str) -> str:
    """Fetch MotherDuck token from Google Secret Manager"""
    client = secretmanager.SecretManagerServiceClient()
    secret_name = f"projects/{project_id}/secrets/{secret_id}/versions/latest"

    response = client.access_secret_version(request={"name": secret_name})
    return response.payload.data.decode('UTF-8')
```

This provides:
- Centralized secret management
- Automatic rotation support
- Fine-grained IAM permissions
- Audit logging

---

## 📚 Additional Resources

- [MotherDuck Service Tokens Documentation](https://motherduck.com/docs/authenticating-to-motherduck/#service-tokens)
- [DuckDB MotherDuck Extension](https://duckdb.org/docs/extensions/motherduck)
- [Google Secret Manager Documentation](https://cloud.google.com/secret-manager/docs)
- [Cloud Run Secrets](https://cloud.google.com/run/docs/configuring/secrets)

---

## ✅ Migration Summary

**Status:** ✅ Complete

**Files Modified:** 3
- `database/db_connection.py`
- `pages/99_Database_Diagnostics.py`
- `requirements.txt`

**Breaking Changes:** None (backward compatible)

**Testing Required:** Local and Cloud Run deployment verification

**Rollback Plan:** Service token can be removed, will fall back to Streamlit secrets for local development

---

**Migration completed on:** 2026-01-13

**Next Steps:**
1. Store service token in Google Secret Manager
2. Update Cloud Run deployment with secret reference
3. Test all MotherDuck connections
4. Monitor for authentication issues
5. Rotate token after successful migration

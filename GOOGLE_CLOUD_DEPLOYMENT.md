# Google Cloud Deployment Guide

## Overview
This application is fully configured for Google Cloud Run deployment. All sensitive credentials can be injected via environment variables, eliminating the need to include `secrets.toml` in Docker containers.

---

## 🔐 Required Environment Variables

### 1. MotherDuck Database
```bash
MOTHERDUCK_TOKEN="your_motherduck_token_here"
```

### 2. Google OAuth
```bash
GOOGLE_CLIENT_ID="your_google_oauth_client_id"
GOOGLE_CLIENT_SECRET="your_google_oauth_client_secret"
```

### 3. Firebase Authentication (Optional - if using Firebase)

**Option A: Base64-Encoded Credentials (RECOMMENDED)**
```bash
# Encode your Firebase service account JSON file
FIREBASE_CREDENTIALS_BASE64=$(base64 -w 0 path/to/firebase-credentials.json)

# Or on macOS:
FIREBASE_CREDENTIALS_BASE64=$(base64 -i path/to/firebase-credentials.json)

# Use this single environment variable
FIREBASE_CREDENTIALS_BASE64="your_base64_encoded_credentials_here"
```

**Option B: Individual Environment Variables (Alternative)**
```bash
FIREBASE_TYPE="service_account"
FIREBASE_PROJECT_ID="your_project_id"
FIREBASE_PRIVATE_KEY_ID="your_private_key_id"
FIREBASE_PRIVATE_KEY="-----BEGIN PRIVATE KEY-----\n...\n-----END PRIVATE KEY-----\n"
FIREBASE_CLIENT_EMAIL="your_service_account@your_project.iam.gserviceaccount.com"
FIREBASE_CLIENT_ID="your_client_id"
FIREBASE_AUTH_URI="https://accounts.google.com/o/oauth2/auth"
FIREBASE_TOKEN_URI="https://oauth2.googleapis.com/token"
FIREBASE_AUTH_PROVIDER_CERT_URL="https://www.googleapis.com/oauth2/v1/certs"
FIREBASE_CLIENT_CERT_URL="https://www.googleapis.com/robot/v1/metadata/x509/your_service_account"
FIREBASE_UNIVERSE_DOMAIN="googleapis.com"
```

---

## 🚀 Deployment Methods

### Method 1: Direct Environment Variables (Quick Start)

#### With Firebase (Base64-encoded):
```bash
# First, encode your Firebase credentials
FIREBASE_B64=$(base64 -w 0 path/to/firebase-credentials.json)

# Deploy with all environment variables
gcloud run deploy portfolio-app \
  --image gcr.io/YOUR_PROJECT_ID/portfolio-app \
  --platform managed \
  --region us-central1 \
  --allow-unauthenticated \
  --set-env-vars \
MOTHERDUCK_TOKEN="your_token",\
GOOGLE_CLIENT_ID="your_client_id",\
GOOGLE_CLIENT_SECRET="your_client_secret",\
FIREBASE_CREDENTIALS_BASE64="$FIREBASE_B64"
```

#### Without Firebase:
```bash
gcloud run deploy portfolio-app \
  --image gcr.io/YOUR_PROJECT_ID/portfolio-app \
  --platform managed \
  --region us-central1 \
  --allow-unauthenticated \
  --set-env-vars \
MOTHERDUCK_TOKEN="your_token",\
GOOGLE_CLIENT_ID="your_client_id",\
GOOGLE_CLIENT_SECRET="your_client_secret"
```

### Method 2: Google Secret Manager (Most Secure - Recommended)

#### Step 1: Create secrets in Secret Manager
```bash
# MotherDuck token
echo -n "your_motherduck_token" | gcloud secrets create motherduck-token --data-file=-

# Google OAuth Client ID
echo -n "your_client_id" | gcloud secrets create google-oauth-client-id --data-file=-

# Google OAuth Client Secret
echo -n "your_client_secret" | gcloud secrets create google-oauth-client-secret --data-file=-

# Firebase credentials (Base64-encoded)
base64 -w 0 path/to/firebase-credentials.json | gcloud secrets create firebase-credentials-base64 --data-file=-

# Or on macOS:
base64 -i path/to/firebase-credentials.json | gcloud secrets create firebase-credentials-base64 --data-file=-
```

#### Step 2: Grant Cloud Run service account access
```bash
# Get your Cloud Run service account
PROJECT_ID=$(gcloud config get-value project)
SERVICE_ACCOUNT="${PROJECT_ID}@${PROJECT_ID}.iam.gserviceaccount.com"

# Grant access to all secrets
for SECRET in motherduck-token google-oauth-client-id google-oauth-client-secret firebase-credentials-base64
do
  gcloud secrets add-iam-policy-binding $SECRET \
    --member="serviceAccount:${SERVICE_ACCOUNT}" \
    --role="roles/secretmanager.secretAccessor"
done
```

#### Step 3: Deploy with secret references
```bash
# With Firebase
gcloud run deploy portfolio-app \
  --image gcr.io/YOUR_PROJECT_ID/portfolio-app \
  --platform managed \
  --region us-central1 \
  --allow-unauthenticated \
  --set-secrets \
MOTHERDUCK_TOKEN=motherduck-token:latest,\
GOOGLE_CLIENT_ID=google-oauth-client-id:latest,\
GOOGLE_CLIENT_SECRET=google-oauth-client-secret:latest,\
FIREBASE_CREDENTIALS_BASE64=firebase-credentials-base64:latest

# Without Firebase
gcloud run deploy portfolio-app \
  --image gcr.io/YOUR_PROJECT_ID/portfolio-app \
  --platform managed \
  --region us-central1 \
  --allow-unauthenticated \
  --set-secrets \
MOTHERDUCK_TOKEN=motherduck-token:latest,\
GOOGLE_CLIENT_ID=google-oauth-client-id:latest,\
GOOGLE_CLIENT_SECRET=google-oauth-client-secret:latest
```

### Method 3: Using .env file (Local Testing)

Create a `.env` file (DO NOT commit to Git):
```bash
MOTHERDUCK_TOKEN=your_motherduck_token
GOOGLE_CLIENT_ID=your_google_oauth_client_id
GOOGLE_CLIENT_SECRET=your_google_oauth_client_secret
# Optional: Add Firebase if needed
FIREBASE_CREDENTIALS_BASE64=your_base64_encoded_firebase_credentials
```

Load and run locally:
```bash
# Load environment variables
export $(cat .env | xargs)

# Run Streamlit
streamlit run main.py
```

---

## 🔥 Firebase Credentials Encoding Guide

### Why Base64 Encoding?
Firebase service account credentials are JSON files with multi-line private keys that are difficult to pass as environment variables. Base64 encoding solves this by converting the entire JSON into a single-line string.

### How to Encode Firebase Credentials

#### Step 1: Download Your Firebase Service Account Key
1. Go to Firebase Console: https://console.firebase.google.com/
2. Select your project
3. Go to **Project Settings** (gear icon) → **Service Accounts**
4. Click **Generate New Private Key**
5. Save the JSON file (e.g., `firebase-credentials.json`)

#### Step 2: Encode to Base64

**On Linux/WSL:**
```bash
# Encode and copy to clipboard
base64 -w 0 firebase-credentials.json

# Or save to a variable
FIREBASE_B64=$(base64 -w 0 firebase-credentials.json)
echo $FIREBASE_B64
```

**On macOS:**
```bash
# Encode and copy to clipboard
base64 -i firebase-credentials.json | pbcopy

# Or save to a variable
FIREBASE_B64=$(base64 -i firebase-credentials.json)
echo $FIREBASE_B64
```

**On Windows (PowerShell):**
```powershell
# Encode Firebase credentials
$bytes = [System.IO.File]::ReadAllBytes("firebase-credentials.json")
$base64 = [System.Convert]::ToBase64String($bytes)
Write-Output $base64

# Or copy to clipboard
$base64 | Set-Clipboard
```

**Using Python (Cross-platform):**
```python
import base64
import json

# Read and encode
with open('firebase-credentials.json', 'r') as f:
    data = json.load(f)
    json_str = json.dumps(data)
    encoded = base64.b64encode(json_str.encode('utf-8')).decode('utf-8')
    print(encoded)
```

#### Step 3: Use the Encoded String

**For Local Testing:**
```bash
export FIREBASE_CREDENTIALS_BASE64="your_encoded_string_here"
streamlit run main.py
```

**For Google Cloud Run:**
```bash
gcloud run deploy portfolio-app \
  --set-env-vars FIREBASE_CREDENTIALS_BASE64="your_encoded_string_here"
```

**For Google Secret Manager:**
```bash
# Create secret from encoded credentials
echo -n "your_encoded_string_here" | gcloud secrets create firebase-credentials-base64 --data-file=-

# Or directly from file
base64 -w 0 firebase-credentials.json | gcloud secrets create firebase-credentials-base64 --data-file=-
```

#### Step 4: Verify Decoding (Optional)

To verify your encoding works correctly:

**Linux/macOS:**
```bash
echo "your_encoded_string_here" | base64 -d | jq .
```

**Python:**
```python
import base64
import json

encoded = "your_encoded_string_here"
decoded = base64.b64decode(encoded).decode('utf-8')
creds = json.loads(decoded)
print(json.dumps(creds, indent=2))
```

### Security Note
- **NEVER** commit the encoded string to Git
- **NEVER** share the encoded string publicly
- Use Secret Manager for production
- The encoded string is just as sensitive as the original JSON file

---

## 📦 Docker Setup

### Dockerfile (No Secrets Included)
```dockerfile
FROM python:3.11-slim

WORKDIR /app

# Install dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code (NO secrets!)
COPY . .

# Expose Streamlit port
EXPOSE 8501

# Health check
HEALTHCHECK CMD curl --fail http://localhost:8501/_stcore/health || exit 1

# Run Streamlit
CMD ["streamlit", "run", "main.py", "--server.port=8501", "--server.address=0.0.0.0"]
```

### .dockerignore
```
# Secrets (NEVER include in Docker images)
.streamlit/secrets.toml
.env
.env.*
*.key
*.pem
firebase-credentials.json

# Local files
.git
.gitignore
__pycache__
*.pyc
*.pyo
*.pyd
.Python
venv/
.venv/
*.db
*.duckdb
.DS_Store
```

### Build and Push to Google Container Registry
```bash
# Build Docker image
docker build -t gcr.io/YOUR_PROJECT_ID/portfolio-app .

# Test locally with environment variables
docker run -p 8501:8501 \
  -e MOTHERDUCK_TOKEN="your_token" \
  -e GOOGLE_CLIENT_ID="your_client_id" \
  -e GOOGLE_CLIENT_SECRET="your_client_secret" \
  gcr.io/YOUR_PROJECT_ID/portfolio-app

# Push to Google Container Registry
docker push gcr.io/YOUR_PROJECT_ID/portfolio-app
```

---

## 🔒 Security Best Practices

### ✅ DO:
- ✅ Use environment variables for all secrets in production
- ✅ Use Google Secret Manager for sensitive data
- ✅ Add `.streamlit/secrets.toml` to `.gitignore` and `.dockerignore`
- ✅ Use least-privilege service accounts
- ✅ Enable Cloud Run service authentication if possible
- ✅ Rotate credentials regularly
- ✅ Use different credentials for dev/staging/production

### ❌ DON'T:
- ❌ Include `secrets.toml` in Docker images
- ❌ Commit secrets to Git repositories
- ❌ Hardcode credentials in source code
- ❌ Share production credentials via Slack/Email
- ❌ Use the same credentials across environments
- ❌ Log credential values (even in debug mode)

---

## 🔄 Credential Priority Order

The application checks for credentials in this order:

### 1. MotherDuck Token
1. `MOTHERDUCK_TOKEN` environment variable ← **Production**
2. `st.secrets["motherduck"]["token"]` ← **Local development**

### 2. Google OAuth
1. `GOOGLE_CLIENT_ID` & `GOOGLE_CLIENT_SECRET` environment variables ← **Production**
2. `st.secrets["google_oauth"]` ← **Local development**

### 3. Firebase
1. `FIREBASE_CREDENTIALS_BASE64` environment variable (Base64-encoded JSON) ← **Production (Recommended)**
2. Individual `FIREBASE_*` environment variables ← **Production (Alternative)**
3. `st.secrets["firebase"]` ← **Local development**
4. `FIREBASE_CREDENTIALS_PATH` file path ← **Fallback**

---

## 🧪 Testing Deployment

### 1. Test Environment Variables Locally
```bash
# Set environment variables
export MOTHERDUCK_TOKEN="test_token"
export GOOGLE_CLIENT_ID="test_client_id"
export GOOGLE_CLIENT_SECRET="test_secret"

# Optional: Add Firebase Base64 credentials
export FIREBASE_CREDENTIALS_BASE64=$(base64 -w 0 path/to/firebase-credentials.json)

# Run application
streamlit run main.py

# Verify in logs: should show "loaded from environment variables"
# For Firebase: should show "loaded from FIREBASE_CREDENTIALS_BASE64"
```

### 2. Test Docker Container Locally
```bash
# Encode Firebase credentials first
FIREBASE_B64=$(base64 -w 0 path/to/firebase-credentials.json)

# Run container with all credentials
docker run -p 8501:8501 \
  -e MOTHERDUCK_TOKEN="test_token" \
  -e GOOGLE_CLIENT_ID="test_client_id" \
  -e GOOGLE_CLIENT_SECRET="test_secret" \
  -e FIREBASE_CREDENTIALS_BASE64="$FIREBASE_B64" \
  gcr.io/YOUR_PROJECT_ID/portfolio-app
```

### 3. Verify Cloud Run Deployment
```bash
# Get Cloud Run service URL
gcloud run services describe portfolio-app --region us-central1 --format 'value(status.url)'

# Test the URL
curl -I https://your-app-url.run.app
```

---

## 📊 Monitoring and Logs

### View Logs
```bash
# Real-time logs
gcloud run services logs tail portfolio-app --region us-central1

# Search for credential loading logs
gcloud run services logs read portfolio-app --region us-central1 | grep "credentials loaded"
```

### Check Secret Usage
```bash
# List secrets
gcloud secrets list

# View secret metadata (NOT the actual secret)
gcloud secrets describe motherduck-token

# Audit secret access
gcloud logging read "resource.type=secret_manager_secret" --limit 50
```

---

## 🆘 Troubleshooting

### Issue: "No MotherDuck token found"
**Solution**: Verify environment variable is set:
```bash
gcloud run services describe portfolio-app --region us-central1 --format 'json' | jq '.spec.template.spec.containers[0].env'
```

### Issue: OAuth not working in Cloud Run
**Solution**: Update OAuth redirect URI in Google Cloud Console:
- Go to: https://console.cloud.google.com/apis/credentials
- Add authorized redirect URI: `https://your-app-url.run.app`

### Issue: Firebase connection fails
**Solution 1**: Verify Base64 encoding is correct:
```bash
# Decode and validate JSON
echo "$FIREBASE_CREDENTIALS_BASE64" | base64 -d | jq .

# Should output valid JSON without errors
```

**Solution 2**: Check if environment variable is set:
```bash
gcloud run services describe portfolio-app --region us-central1 --format 'json' | \
  jq '.spec.template.spec.containers[0].env[] | select(.name=="FIREBASE_CREDENTIALS_BASE64")'
```

**Solution 3**: Check service account permissions:
```bash
gcloud projects get-iam-policy YOUR_PROJECT_ID \
  --flatten="bindings[].members" \
  --filter="bindings.members:YOUR_SERVICE_ACCOUNT"
```

### Issue: Firebase Base64 decoding error
**Solution**: Ensure proper encoding:
```bash
# Linux/WSL - use -w 0 to disable line wrapping
base64 -w 0 firebase-credentials.json

# macOS - use -i flag
base64 -i firebase-credentials.json

# Avoid line breaks in the encoded string
```

---

## 🔄 CI/CD with Cloud Build

### cloudbuild.yaml
```yaml
steps:
  # Build Docker image
  - name: 'gcr.io/cloud-builders/docker'
    args: ['build', '-t', 'gcr.io/$PROJECT_ID/portfolio-app', '.']

  # Push to Container Registry
  - name: 'gcr.io/cloud-builders/docker'
    args: ['push', 'gcr.io/$PROJECT_ID/portfolio-app']

  # Deploy to Cloud Run
  - name: 'gcr.io/cloud-builders/gcloud'
    args:
      - 'run'
      - 'deploy'
      - 'portfolio-app'
      - '--image=gcr.io/$PROJECT_ID/portfolio-app'
      - '--region=us-central1'
      - '--platform=managed'
      - '--allow-unauthenticated'
      - '--set-secrets=MOTHERDUCK_TOKEN=motherduck-token:latest,GOOGLE_CLIENT_ID=google-oauth-client-id:latest,GOOGLE_CLIENT_SECRET=google-oauth-client-secret:latest,FIREBASE_CREDENTIALS_BASE64=firebase-credentials-base64:latest'

images:
  - 'gcr.io/$PROJECT_ID/portfolio-app'
```

### Trigger deployment
```bash
gcloud builds submit --config cloudbuild.yaml .
```

---

## 📚 Additional Resources

- [Google Cloud Run Documentation](https://cloud.google.com/run/docs)
- [Google Secret Manager Documentation](https://cloud.google.com/secret-manager/docs)
- [Streamlit Deployment Guide](https://docs.streamlit.io/deploy)
- [MotherDuck Documentation](https://motherduck.com/docs)

---

## ✅ Pre-Deployment Checklist

- [ ] All secrets added to Google Secret Manager
- [ ] Service account has necessary permissions
- [ ] Docker image builds successfully without errors
- [ ] `.dockerignore` includes `secrets.toml`
- [ ] `.gitignore` includes `secrets.toml` and `.env`
- [ ] OAuth redirect URIs updated in Google Console
- [ ] Environment variables tested locally
- [ ] Docker container tested locally
- [ ] Cloud Run service deployed successfully
- [ ] Application accessible via Cloud Run URL
- [ ] Database connection working (MotherDuck)
- [ ] Authentication working (Google OAuth)
- [ ] Logs show credentials loaded from environment variables

---

**Your application is now fully cloud-ready! 🎉**

For questions or issues, refer to the troubleshooting section or check application logs.

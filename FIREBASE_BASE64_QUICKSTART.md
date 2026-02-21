# Firebase Base64 Encoding - Quick Start Guide

## Why Base64 Encoding?
Firebase service account credentials contain multi-line private keys that are difficult to pass as environment variables. Base64 encoding converts the entire JSON file into a single-line string that can be easily passed to Cloud Run.

---

## Quick Steps

### 1. Encode Your Firebase Credentials

**Linux/WSL:**
```bash
base64 -w 0 firebase-credentials.json
```

**macOS:**
```bash
base64 -i firebase-credentials.json
```

**Windows PowerShell:**
```powershell
$bytes = [System.IO.File]::ReadAllBytes("firebase-credentials.json")
[System.Convert]::ToBase64String($bytes)
```

**Python (Cross-platform):**
```python
import base64
import json

with open('firebase-credentials.json', 'r') as f:
    data = json.load(f)
    json_str = json.dumps(data)
    encoded = base64.b64encode(json_str.encode('utf-8')).decode('utf-8')
    print(encoded)
```

---

## 2. Use the Encoded String

### Local Testing
```bash
# Set environment variable
export FIREBASE_CREDENTIALS_BASE64="your_encoded_string_here"

# Run app
streamlit run main.py
```

### Google Cloud Run (Direct)
```bash
# Encode and store in variable
FIREBASE_B64=$(base64 -w 0 firebase-credentials.json)

# Deploy
gcloud run deploy portfolio-app \
  --image gcr.io/YOUR_PROJECT_ID/portfolio-app \
  --region us-central1 \
  --set-env-vars \
MOTHERDUCK_TOKEN="your_token",\
GOOGLE_CLIENT_ID="your_client_id",\
GOOGLE_CLIENT_SECRET="your_client_secret",\
FIREBASE_CREDENTIALS_BASE64="$FIREBASE_B64"
```

### Google Secret Manager (Recommended)
```bash
# Create secret from Base64 encoded credentials
base64 -w 0 firebase-credentials.json | \
  gcloud secrets create firebase-credentials-base64 --data-file=-

# Grant access to Cloud Run service account
PROJECT_ID=$(gcloud config get-value project)
SERVICE_ACCOUNT="${PROJECT_ID}@${PROJECT_ID}.iam.gserviceaccount.com"

gcloud secrets add-iam-policy-binding firebase-credentials-base64 \
  --member="serviceAccount:${SERVICE_ACCOUNT}" \
  --role="roles/secretmanager.secretAccessor"

# Deploy with secret reference
gcloud run deploy portfolio-app \
  --image gcr.io/YOUR_PROJECT_ID/portfolio-app \
  --region us-central1 \
  --set-secrets \
MOTHERDUCK_TOKEN=motherduck-token:latest,\
GOOGLE_CLIENT_ID=google-oauth-client-id:latest,\
GOOGLE_CLIENT_SECRET=google-oauth-client-secret:latest,\
FIREBASE_CREDENTIALS_BASE64=firebase-credentials-base64:latest
```

---

## 3. Verify Encoding

### Check if encoding is valid:
```bash
# Linux/macOS
echo "your_encoded_string" | base64 -d | jq .

# Should output valid JSON
```

### Check environment variable in Cloud Run:
```bash
gcloud run services describe portfolio-app --region us-central1 --format 'json' | \
  jq '.spec.template.spec.containers[0].env[] | select(.name=="FIREBASE_CREDENTIALS_BASE64")'
```

---

## 4. Troubleshooting

### Error: "Failed to decode FIREBASE_CREDENTIALS_BASE64"

**Cause**: Line breaks in encoded string

**Solution**: Use `-w 0` flag (Linux) or `-i` flag (macOS) to avoid line breaks:
```bash
# Linux/WSL
base64 -w 0 firebase-credentials.json

# macOS
base64 -i firebase-credentials.json
```

### Error: "Invalid JSON"

**Cause**: Encoding issues or corrupted file

**Solution**: Verify original JSON is valid:
```bash
jq . firebase-credentials.json
```

### Environment Variable Not Set

**Solution**: Check Cloud Run configuration:
```bash
gcloud run services describe portfolio-app --region us-central1 --format 'value(spec.template.spec.containers[0].env)'
```

---

## 5. Security Checklist

- [ ] Original JSON file added to `.gitignore`
- [ ] Encoded string added to `.gitignore` (if stored in file)
- [ ] Secrets stored in Google Secret Manager (production)
- [ ] Service account has minimal required permissions
- [ ] Encoded string NOT committed to Git repository
- [ ] Different credentials for dev/staging/production

---

## Complete Example

```bash
#!/bin/bash
# Complete deployment script with Firebase Base64 encoding

# Configuration
PROJECT_ID="your-gcp-project-id"
REGION="us-central1"
APP_NAME="portfolio-app"

# Step 1: Encode Firebase credentials
echo "📝 Encoding Firebase credentials..."
FIREBASE_B64=$(base64 -w 0 firebase-credentials.json)

# Step 2: Store in Secret Manager
echo "🔐 Storing credentials in Secret Manager..."
echo -n "$FIREBASE_B64" | gcloud secrets create firebase-credentials-base64 --data-file=- --project=$PROJECT_ID

# Step 3: Grant access to Cloud Run service account
echo "🔓 Granting access to service account..."
SERVICE_ACCOUNT="${PROJECT_ID}@${PROJECT_ID}.iam.gserviceaccount.com"
gcloud secrets add-iam-policy-binding firebase-credentials-base64 \
  --member="serviceAccount:${SERVICE_ACCOUNT}" \
  --role="roles/secretmanager.secretAccessor" \
  --project=$PROJECT_ID

# Step 4: Deploy to Cloud Run
echo "🚀 Deploying to Cloud Run..."
gcloud run deploy $APP_NAME \
  --image gcr.io/$PROJECT_ID/$APP_NAME \
  --region $REGION \
  --platform managed \
  --allow-unauthenticated \
  --set-secrets \
MOTHERDUCK_TOKEN=motherduck-token:latest,\
GOOGLE_CLIENT_ID=google-oauth-client-id:latest,\
GOOGLE_CLIENT_SECRET=google-oauth-client-secret:latest,\
FIREBASE_CREDENTIALS_BASE64=firebase-credentials-base64:latest \
  --project=$PROJECT_ID

echo "✅ Deployment complete!"
```

---

## How It Works Internally

The application's `utils/firebase_db.py` automatically:

1. Checks for `FIREBASE_CREDENTIALS_BASE64` environment variable
2. Decodes from Base64 to JSON string
3. Parses JSON to dictionary
4. Passes dictionary to `firebase_admin.credentials.Certificate()`
5. Falls back to other methods if Base64 not found

**Priority Order:**
1. `FIREBASE_CREDENTIALS_BASE64` ← **You are here**
2. Individual `FIREBASE_*` environment variables
3. `st.secrets["firebase"]` (local development)
4. `FIREBASE_CREDENTIALS_PATH` (file path)

---

## Additional Resources

- [Firebase Admin SDK Setup](https://firebase.google.com/docs/admin/setup)
- [Google Cloud Secret Manager](https://cloud.google.com/secret-manager/docs)
- [Base64 Encoding Reference](https://developer.mozilla.org/en-US/docs/Glossary/Base64)

---

**Your Firebase credentials are now cloud-ready! 🚀**

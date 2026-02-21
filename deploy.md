# Google Cloud Run Deployment Guide

Step-by-step instructions for redeploying the Portfolio Analysis Suite to Google Cloud Run.

## Prerequisites

- [Google Cloud SDK](https://cloud.google.com/sdk/docs/install) installed and configured
- Docker Desktop installed and running
- Authenticated with Google Cloud: `gcloud auth login`
- Project set: `gcloud config set project portfoliosuite`

## Deployment Details

| Setting | Value |
|---------|-------|
| Project ID | `portfoliosuite` |
| Region | `us-central1` |
| Service Name | `portfoliosuite-app` |
| Service URL | `https://portfoliosuite-app-avyl2luyta-uc.a.run.app` |

---

## Quick Deploy (After Code Changes)

Run these commands from the project directory (`F:\Coding\Portfolio-C`):

```bash
# 1. Build the Docker image
docker build -t gcr.io/portfoliosuite/portfoliosuite-app .

# 2. Push to Google Container Registry
docker push gcr.io/portfoliosuite/portfoliosuite-app

# 3. Deploy to Cloud Run
gcloud run deploy portfoliosuite-app --image gcr.io/portfoliosuite/portfoliosuite-app --region us-central1 --platform managed --allow-unauthenticated
```

---

## Detailed Steps

### Step 1: Build the Docker Image

From the project root directory (where `Dockerfile` is located):

```bash
docker build -t gcr.io/portfoliosuite/portfoliosuite-app .
```

**Troubleshooting:**
- If Docker isn't running, start Docker Desktop first
- Build can take several minutes on first run (downloads Python dependencies)

### Step 2: Configure Docker for GCR (First Time Only)

If this is your first time pushing to Google Container Registry:

```bash
gcloud auth configure-docker
```

### Step 3: Push the Image to Google Container Registry

```bash
docker push gcr.io/portfoliosuite/portfoliosuite-app
```

### Step 4: Deploy to Cloud Run

```bash
gcloud run deploy portfoliosuite-app --image gcr.io/portfoliosuite/portfoliosuite-app --region us-central1 --platform managed --allow-unauthenticated
```

---

## Environment Variables

These environment variables are configured in Cloud Run:

| Variable | Value |
|----------|-------|
| `REDIRECT_URI` | `https://portfoliosuite-app-avyl2luyta-uc.a.run.app` |
| `GOOGLE_CLIENT_ID` | *(configured in Cloud Run)* |
| `GOOGLE_CLIENT_SECRET` | *(configured in Cloud Run)* |
| `FIRESTORE_PROJECT_ID` | *(configured in Cloud Run)* |

### Updating Environment Variables

```bash
gcloud run services update portfoliosuite-app --region us-central1 --set-env-vars "VAR_NAME=value"
```

---

## Viewing Logs

```bash
# View recent logs
gcloud run logs read portfoliosuite-app --region us-central1

# Stream logs in real-time
gcloud run logs tail portfoliosuite-app --region us-central1
```

---

## Common Issues

### OAuth Redirect Error (localhost refused to connect)

The `REDIRECT_URI` environment variable is missing or incorrect.

```bash
gcloud run services update portfoliosuite-app --region us-central1 --set-env-vars "REDIRECT_URI=https://portfoliosuite-app-avyl2luyta-uc.a.run.app"
```

Also ensure the redirect URI is registered in [Google Cloud Console Credentials](https://console.cloud.google.com/apis/credentials).

### OAuth 400: redirect_uri_mismatch

The redirect URI isn't registered in Google OAuth settings.

1. Go to [Google Cloud Console Credentials](https://console.cloud.google.com/apis/credentials)
2. Click your OAuth 2.0 Client ID
3. Add `https://portfoliosuite-app-avyl2luyta-uc.a.run.app` to Authorized redirect URIs
4. Save and wait 1-2 minutes

### Container Fails to Start

Check logs:
```bash
gcloud run logs read portfoliosuite-app --region us-central1 --limit 50
```

---

## Useful Commands

```bash
# Check current service status
gcloud run services describe portfoliosuite-app --region us-central1

# List all revisions
gcloud run revisions list --service portfoliosuite-app --region us-central1

# Get service URL
gcloud run services describe portfoliosuite-app --region us-central1 --format="value(status.url)"
```

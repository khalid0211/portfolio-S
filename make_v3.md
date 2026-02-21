# Migration Plan: MotherDuck to DuckDB on GCP Compute Engine

## Overview

Migrate from MotherDuck (managed cloud DuckDB) to a self-hosted DuckDB on Google Cloud Compute Engine VM (e2-micro, free tier).

### Key Decisions

| Decision | Choice |
|----------|--------|
| Database | DuckDB on Compute Engine e2-micro |
| Connection method | HTTP API (Flask server) |
| Expected concurrent users | 1-5 users |
| Backup frequency | Every 6 hours |
| Backup retention | 7 days |
| Local development | Environment-based switching (local file vs remote API) |
| Data migration | None - copy existing local DuckDB to VM |
| Estimated monthly cost | ~$0.18 |

---

## Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                    Google Cloud                              │
│  ┌─────────────────┐      ┌─────────────────────────────┐   │
│  │   Cloud Run     │      │  Compute Engine             │   │
│  │   (Streamlit)   │─────►│  e2-micro (free tier)       │   │
│  │                 │ HTTP │  ┌───────────────────────┐  │   │
│  └─────────────────┘  API │  │ portfolio.duckdb      │  │   │
│                           │  └───────────────────────┘  │   │
│                           │  Flask server on port 5432  │   │
│                           └─────────────────────────────┘   │
│                                      │                       │
│                           ┌──────────▼──────────┐           │
│                           │  Cloud Storage      │           │
│                           │  (backup bucket)    │           │
│                           │  Every 6 hours      │           │
│                           └─────────────────────┘           │
└─────────────────────────────────────────────────────────────┘
```

---

## Implementation Tasks

### Phase 1: Create Server Components

#### 1.1 Create `server/duckdb_server.py`
Flask-based HTTP API for DuckDB operations. Must implement these endpoints:

| Endpoint | Method | Purpose |
|----------|--------|---------|
| `/health` | GET | Health check |
| `/query` | POST | SELECT queries, returns JSON |
| `/execute` | POST | INSERT/UPDATE/DELETE |
| `/fetch_one` | POST | Single row result |
| `/fetch_all` | POST | All rows as list |
| `/dataframe` | POST | DataFrame-optimized response |
| `/table_exists` | GET | Check if table exists |
| `/table_columns` | GET | Get column names |

Features:
- API key authentication via `X-API-Key` header
- Proper error handling and logging
- JSON serialization of dates, decimals, None values
- Connection management (open/close per request)

#### 1.2 Create `server/requirements.txt`
```
flask>=2.0.0
duckdb>=1.0.0
pandas
gunicorn
```

#### 1.3 Create `server/duckdb.service`
Systemd service file for auto-start on VM boot:
```ini
[Unit]
Description=DuckDB HTTP Server
After=network.target

[Service]
Type=simple
User=USER
WorkingDirectory=/home/USER
Environment="DUCKDB_PATH=/home/USER/data/portfolio.duckdb"
Environment="API_KEY=your-secret-key"
Environment="PORT=5432"
ExecStart=/usr/bin/python3 /home/USER/duckdb_server.py
Restart=always
RestartSec=10

[Install]
WantedBy=multi-user.target
```

---

### Phase 2: Create Remote Connection Client

#### 2.1 Create `database/remote_connection.py`
New class `RemoteDuckDBConnection` that mirrors `DatabaseConnection` API:

```python
class RemoteDuckDBConnection:
    def __init__(self, endpoint: str, api_key: str):
        self.endpoint = endpoint
        self.api_key = api_key

    def execute(self, query: str, params: tuple = None) -> 'RemoteDuckDBConnection':
        # POST to /execute endpoint
        # Return self for chaining

    def df(self) -> pd.DataFrame:
        # POST to /dataframe endpoint
        # Return pandas DataFrame

    def fetch_one(self, query: str, params: tuple = None) -> Optional[tuple]:
        # POST to /fetch_one endpoint

    def fetch_all(self, query: str, params: tuple = None) -> List[tuple]:
        # POST to /fetch_all endpoint

    def table_exists(self, table_name: str) -> bool:
        # GET /table_exists?table_name=X

    def get_table_columns(self, table_name: str) -> List[str]:
        # GET /table_columns?table_name=X
```

Must handle:
- Request timeout (30 seconds)
- Retry on connection errors (3 attempts)
- Proper error messages from server responses
- API key in headers

#### 2.2 Modify `database/db_connection.py`
Update `get_db()` function to support remote mode:

```python
def get_db(db_path: Optional[str] = None, user_email: Optional[str] = None) -> DatabaseConnection:
    # Check for remote mode first
    connection_type = os.environ.get('DATABASE_CONNECTION_TYPE', 'local')

    if connection_type == 'remote':
        endpoint = os.environ.get('DATABASE_REMOTE_ENDPOINT')
        api_key = os.environ.get('DATABASE_API_KEY')
        return RemoteDuckDBConnection(endpoint, api_key)

    # Existing local/MotherDuck logic...
```

---

### Phase 3: Update Configuration

#### 3.1 Modify `utils/database_config.py`
Add support for remote connection type:

```python
# Add to DEFAULT_CONFIG
DEFAULT_CONFIG = {
    "connection_type": "local",  # local, motherduck, or remote
    "local_path": "...",
    "motherduck_database": "portfolio_cloud",
    "remote_endpoint": "",
    "remote_api_key": "",
    "last_updated": "..."
}

# Add validation for 'remote' type
if config["connection_type"] == "remote":
    if not config.get("remote_endpoint"):
        return False
```

#### 3.2 Update `.streamlit/secrets.toml`
Add database configuration section:

```toml
[database]
# Options: local, motherduck, remote
connection_type = "local"

# Local settings
local_path = "DuckDB/portfolio.duckdb"

# Remote settings (for production)
remote_endpoint = ""
remote_api_key = ""

# MotherDuck settings (legacy)
motherduck_database = "portfolio_cloud"
```

#### 3.3 Update `requirements.txt`
Add `requests` library for HTTP client:
```
requests>=2.28.0
```

---

### Phase 4: Create Backup Scripts

#### 4.1 Create `scripts/backup_to_gcs.sh`
Backup script for VM cron job:

```bash
#!/bin/bash
# Run every 6 hours: 0 */6 * * * /home/USER/scripts/backup_to_gcs.sh

BUCKET="gs://your-project-backups"
DB_PATH="/home/USER/data/portfolio.duckdb"
TIMESTAMP=$(date +%Y%m%d_%H%M)

# Copy current database to GCS
gsutil cp "$DB_PATH" "$BUCKET/portfolio-$TIMESTAMP.duckdb"

# Also maintain a "latest" copy
gsutil cp "$DB_PATH" "$BUCKET/portfolio-latest.duckdb"

# Delete backups older than 7 days (keep ~28 backups at 6-hour intervals)
gsutil ls "$BUCKET/portfolio-*.duckdb" | while read file; do
    file_date=$(echo "$file" | grep -oP '\d{8}')
    if [[ -n "$file_date" ]]; then
        file_epoch=$(date -d "$file_date" +%s 2>/dev/null)
        cutoff_epoch=$(date -d '7 days ago' +%s)
        if [[ "$file_epoch" -lt "$cutoff_epoch" ]]; then
            gsutil rm "$file"
        fi
    fi
done
```

#### 4.2 Create `scripts/download_backup.py`
Helper for local development:

```python
#!/usr/bin/env python3
"""Download latest backup from GCS for local development"""

import subprocess
import sys
from pathlib import Path

BUCKET = "gs://your-project-backups"
LOCAL_PATH = Path(__file__).parent.parent / "DuckDB" / "portfolio.duckdb"

def download_latest():
    print(f"Downloading latest backup to {LOCAL_PATH}...")
    result = subprocess.run([
        "gsutil", "cp",
        f"{BUCKET}/portfolio-latest.duckdb",
        str(LOCAL_PATH)
    ], capture_output=True, text=True)

    if result.returncode == 0:
        print("Download complete!")
    else:
        print(f"Error: {result.stderr}")
        sys.exit(1)

if __name__ == "__main__":
    download_latest()
```

---

### Phase 5: VM Setup (Manual Steps)

These commands are run manually in GCP Console or via gcloud CLI:

#### 5.1 Create VM
```bash
gcloud compute instances create duckdb-server \
  --zone=us-central1-a \
  --machine-type=e2-micro \
  --image-family=debian-12 \
  --image-project=debian-cloud \
  --boot-disk-size=30GB \
  --tags=duckdb-server
```

#### 5.2 Configure Firewall
```bash
# Allow internal traffic from Cloud Run
gcloud compute firewall-rules create allow-duckdb-internal \
  --allow=tcp:5432 \
  --source-ranges=10.0.0.0/8 \
  --target-tags=duckdb-server
```

#### 5.3 SSH and Install Dependencies
```bash
gcloud compute ssh duckdb-server --zone=us-central1-a

# On VM:
sudo apt update && sudo apt install -y python3 python3-pip
pip3 install flask duckdb pandas gunicorn

mkdir -p /home/$USER/data
mkdir -p /home/$USER/scripts
```

#### 5.4 Copy Local Database to VM
```bash
# From local machine:
gcloud compute scp DuckDB/portfolio.duckdb duckdb-server:/home/USER/data/ --zone=us-central1-a
```

#### 5.5 Copy Server Script to VM
```bash
gcloud compute scp server/duckdb_server.py duckdb-server:/home/USER/ --zone=us-central1-a
gcloud compute scp scripts/backup_to_gcs.sh duckdb-server:/home/USER/scripts/ --zone=us-central1-a
```

#### 5.6 Setup Systemd Service
```bash
# On VM:
sudo cp duckdb.service /etc/systemd/system/
sudo systemctl daemon-reload
sudo systemctl enable duckdb
sudo systemctl start duckdb
```

#### 5.7 Setup Backup Cron
```bash
# On VM:
chmod +x /home/USER/scripts/backup_to_gcs.sh
crontab -e
# Add: 0 */6 * * * /home/USER/scripts/backup_to_gcs.sh
```

#### 5.8 Get VM Internal IP
```bash
gcloud compute instances describe duckdb-server \
  --zone=us-central1-a \
  --format='get(networkInterfaces[0].networkIP)'
```

---

### Phase 6: Update Cloud Run Deployment

Update environment variables in Cloud Run:

```bash
gcloud run services update portfolio-suite \
  --set-env-vars="DATABASE_CONNECTION_TYPE=remote" \
  --set-env-vars="DATABASE_REMOTE_ENDPOINT=http://10.x.x.x:5432" \
  --set-env-vars="DATABASE_API_KEY=your-secret-key"
```

---

## Files to Create/Modify Summary

| File | Action | Description |
|------|--------|-------------|
| `server/duckdb_server.py` | CREATE | Flask HTTP API |
| `server/requirements.txt` | CREATE | VM Python dependencies |
| `server/duckdb.service` | CREATE | Systemd service config |
| `database/remote_connection.py` | CREATE | Remote connection client |
| `database/db_connection.py` | MODIFY | Add remote mode support |
| `utils/database_config.py` | MODIFY | Add remote config validation |
| `scripts/backup_to_gcs.sh` | CREATE | Backup script for VM |
| `scripts/download_backup.py` | CREATE | Local dev helper |
| `.streamlit/secrets.toml` | MODIFY | Add database config section |
| `requirements.txt` | MODIFY | Add `requests` |

---

## Verification Steps

After implementation, verify:

1. **Local mode works**: Run `streamlit run main.py` locally, confirm it uses local DuckDB
2. **Server starts**: On VM, `curl http://localhost:5432/health` returns healthy
3. **Remote mode works**: Set env vars, confirm app connects to VM
4. **Backup works**: Run backup script, check GCS bucket
5. **Auto-restart works**: Reboot VM, confirm server starts automatically
6. **All pages work**: Test portfolio, project, status report CRUD operations

---

## Rollback Plan

If migration fails:

1. Change `DATABASE_CONNECTION_TYPE` back to `motherduck`
2. Redeploy Cloud Run
3. App immediately uses MotherDuck again
4. No data loss (MotherDuck data still exists)

---

## Cost Breakdown

| Component | Monthly Cost |
|-----------|-------------|
| Compute Engine e2-micro | $0 (free tier) |
| 30 GB persistent disk | $0 (free tier) |
| Cloud Storage (~1 GB backups) | ~$0.02 |
| Network egress (internal) | $0 |
| **Total** | **~$0.02-0.20/month** |

Free tier requires:
- Region: us-central1, us-east1, or us-west1
- Machine type: e2-micro only
- Disk: Standard persistent disk, max 30 GB

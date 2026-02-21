# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

Portfolio Analysis Suite is a Streamlit-based web application for project portfolio management and EVM (Earned Value Management) analysis. It supports multi-tenant portfolios with Google OAuth authentication and role-based access control via Firestore. Data is stored in DuckDB (local) or MotherDuck (cloud).

## Development Commands

```bash
# Install dependencies
pip install -r requirements.txt

# Run the application locally (port 8501)
streamlit run main.py

# Clear Streamlit cache
streamlit cache clear
```

**Note**: No automated test suite exists. Manual testing via the running application.

```bash
# Docker build and run (for deployment)
docker build -t portfolio-suite .
docker run -p 8080:8080 -e MOTHERDUCK_TOKEN=<token> portfolio-suite
```

## Architecture

### Technology Stack
- **UI**: Streamlit 1.42+ with native OAuth
- **Database**: DuckDB (local OLAP) at `DuckDB/portfolio.duckdb` or MotherDuck (cloud) via `MOTHERDUCK_TOKEN` env var
- **Auth/Users**: Google OAuth + Firestore for user profiles/permissions, DuckDB for portfolio data
- **Visualization**: Plotly, Altair, Matplotlib

### Layer Structure
```
pages/           → Streamlit multi-page apps (numbered for nav order)
services/        → Business logic (db_data_service, evm_service)
database/        → DuckDB connection manager & queries
core/            → EVM calculation engine (evm_engine.py)
models/          → Data models (ProjectData, ColumnMapping)
utils/           → Auth, Firestore client, portfolio context
components/      → Reusable UI components
config/          → Constants and feature flags
```

### Key Patterns

**Database Access**: Use `get_db()` from `database/db_connection.py` - returns singleton `DatabaseConnection`. Services use `DatabaseDataManager` from `services/db_data_service.py` for CRUD operations.

**DuckDB Connection Rules** (Critical):
- Use `db.fetch_one(query, params)` instead of `db.execute(query, params).fetchone()` - the chained pattern can cause "No open result set" errors
- Use `db.execute(query, params).df()` to get pandas DataFrames (DuckDB native method)
- **Never** make database calls inside loops - fetch all data in one query, then process in memory
- Connection corruption during page navigation requires Streamlit restart

**EVM Data Flow**:
1. `project_baseline` stores planned values (BAC, planned dates) with effectivity windows
2. `project_status_report` stores actuals (AC, EV, data_date) per reporting period
3. `database/queries.py:get_effective_baseline()` retrieves the correct baseline for a status date
4. `core/evm_engine.py` computes metrics (CPI, SPI, EAC, etc.) from baseline + status data

**Session State**: Streamlit reruns on every interaction. Use `st.session_state` for persistence across reruns. Auth info stored in `st.session_state.user_info`.

### Database Schema

Schema defined in `database/db_schema.py`. Key constraints:
- One active baseline per project (`baseline_end_date IS NULL`)
- Baseline effectivity: `baseline_start_date <= status_date < baseline_end_date`
- Planned start date locked after `actual_cost > 0` exists
- Soft deletes via `is_active` column on most tables

**Migrations**: `database/migrations/` contains schema migration scripts. Run manually as needed.

#### Table Definitions (MotherDuck: `md:portfolio_cloud`)

```sql
-- User management (synced from Firestore)
CREATE TABLE app_user(
  user_id BIGINT,
  display_name VARCHAR,
  email VARCHAR,
  "role" VARCHAR,
  department VARCHAR,
  is_active BOOLEAN,
  created_at TIMESTAMP,
  last_login TIMESTAMP
);

-- Portfolio with default S-curve settings and up to 10 strategic factors
CREATE TABLE portfolio(
  portfolio_id BIGINT,
  portfolio_name VARCHAR,
  managing_organization VARCHAR,
  portfolio_manager VARCHAR,
  is_active BOOLEAN,
  created_at TIMESTAMP,
  description VARCHAR,
  default_curve_type VARCHAR,           -- linear, beta, front-loaded, back-loaded
  default_alpha DECIMAL(5,2),
  default_beta DECIMAL(5,2),
  default_inflation_rate DECIMAL(5,3),
  factor_1_name VARCHAR, factor_1_weight DECIMAL(5,2),
  factor_2_name VARCHAR, factor_2_weight DECIMAL(5,2),
  factor_3_name VARCHAR, factor_3_weight DECIMAL(5,2),
  factor_4_name VARCHAR, factor_4_weight DECIMAL(5,2),
  factor_5_name VARCHAR, factor_5_weight DECIMAL(5,2),
  factor_6_name VARCHAR, factor_6_weight DECIMAL(5,2),
  factor_7_name VARCHAR, factor_7_weight DECIMAL(5,2),
  factor_8_name VARCHAR, factor_8_weight DECIMAL(5,2),
  factor_9_name VARCHAR, factor_9_weight DECIMAL(5,2),
  factor_10_name VARCHAR, factor_10_weight DECIMAL(5,2),
  settings_json VARCHAR
);

-- Strategic factors for portfolio scoring
CREATE TABLE portfolio_factor(
  factor_id BIGINT,
  portfolio_id BIGINT,
  factor_name VARCHAR,
  factor_weight_percent DECIMAL(5,2),
  likert_min INTEGER,
  likert_max INTEGER,
  is_active BOOLEAN,
  created_at TIMESTAMP
);

-- Many-to-many: users who own/manage portfolios
CREATE TABLE portfolio_ownership(
  portfolio_id BIGINT,
  owner_user_id BIGINT,
  assigned_at TIMESTAMP,
  assigned_by VARCHAR,
  is_active BOOLEAN
);

-- Project master data with current baseline snapshot
CREATE TABLE project(
  project_id BIGINT,
  portfolio_id BIGINT,
  project_code VARCHAR,
  project_name VARCHAR,
  responsible_organization VARCHAR,
  project_manager VARCHAR,
  project_status VARCHAR,
  initial_budget DECIMAL(18,2),
  current_budget DECIMAL(18,2),
  is_active BOOLEAN,
  created_at TIMESTAMP,
  budget_at_completion DECIMAL(18,2),
  planned_start_date DATE,
  planned_finish_date DATE,
  baseline_version INTEGER,
  baseline_approved_by VARCHAR,
  baseline_approved_date DATE,
  baseline_reason VARCHAR,
  curve_type VARCHAR,
  alpha DECIMAL(5,2),
  beta DECIMAL(5,2),
  inflation_rate DECIMAL(5,3),
  updated_at TIMESTAMP
);

-- Baseline history with effectivity windows
CREATE TABLE project_baseline(
  baseline_id BIGINT,
  project_id BIGINT,
  portfolio_id BIGINT,
  baseline_version INTEGER,
  baseline_start_date DATE,             -- effectivity start
  baseline_end_date DATE,               -- NULL = current active baseline
  planned_start_date DATE,
  planned_finish_date DATE,
  budget_at_completion DECIMAL(18,2),
  project_status VARCHAR,
  baseline_reason VARCHAR,
  approved_by VARCHAR,
  approved_date DATE,
  created_at TIMESTAMP,
  is_active BOOLEAN
);

-- Project strategic factor scores (Likert scale)
CREATE TABLE project_factor_score(
  portfolio_id BIGINT NOT NULL,
  project_id BIGINT,
  factor_id BIGINT,
  score INTEGER NOT NULL,
  scored_at TIMESTAMP DEFAULT(now()) NOT NULL,
  scored_by_user_id BIGINT,
  is_active BOOLEAN DEFAULT(true),
  PRIMARY KEY(project_id, factor_id)
);

-- Project SDG (Sustainable Development Goals) alignment
CREATE TABLE project_sdg(
  portfolio_id BIGINT,
  project_id BIGINT,
  sdg_id INTEGER,
  weight_percent DECIMAL(5,2),
  created_at TIMESTAMP,
  is_active BOOLEAN
);

-- EVM status reports with all calculated metrics
CREATE TABLE project_status_report(
  status_report_id BIGINT,
  portfolio_id BIGINT,
  project_id BIGINT,
  status_date DATE,
  -- Input values
  actual_cost DECIMAL(18,2),
  planned_value DECIMAL(18,2),
  earned_value DECIMAL(18,2),
  notes VARCHAR,
  created_at TIMESTAMP,
  -- Calculation metadata
  calculation_performed BOOLEAN,
  calculation_timestamp TIMESTAMP,
  curve_type VARCHAR,
  alpha DECIMAL(5,2),
  beta DECIMAL(5,2),
  inflation_rate DECIMAL(5,3),
  -- Duration metrics
  actual_duration_months DECIMAL(10,2),
  original_duration_months DECIMAL(10,2),
  -- Calculated EVM metrics
  calculated_pv DECIMAL(18,2),
  calculated_ev DECIMAL(18,2),
  present_value DECIMAL(18,2),
  percent_complete DECIMAL(5,2),
  percent_budget_used DECIMAL(5,2),
  percent_time_used DECIMAL(5,2),
  percent_present_value_project DECIMAL(5,2),
  percent_likely_value_project DECIMAL(5,2),
  -- Cost metrics
  cv DECIMAL(18,2),                     -- Cost Variance = EV - AC
  cpi DECIMAL(10,3),                    -- Cost Performance Index = EV / AC
  eac DECIMAL(18,2),                    -- Estimate at Completion
  etc DECIMAL(18,2),                    -- Estimate to Complete
  vac DECIMAL(18,2),                    -- Variance at Completion
  tcpi DECIMAL(10,3),                   -- To-Complete Performance Index
  -- Schedule metrics
  sv DECIMAL(18,2),                     -- Schedule Variance = EV - PV
  spi DECIMAL(10,3),                    -- Schedule Performance Index = EV / PV
  es DECIMAL(10,2),                     -- Earned Schedule
  espi DECIMAL(10,3),                   -- Earned Schedule Performance Index
  svt DECIMAL(10,2),                    -- Schedule Variance (time)
  likely_duration DECIMAL(10,2),
  likely_completion DATE,
  likely_completion_date DATE,
  -- Snapshot of baseline values at report time
  planned_value_project DECIMAL(18,2),
  likely_value_project DECIMAL(18,2),
  ac DECIMAL(18,2),
  pv DECIMAL(18,2),
  ev DECIMAL(18,2),
  bac DECIMAL(18,2),
  plan_start_date DATE,
  plan_finish_date DATE,
  baseline_version INTEGER,
  is_active BOOLEAN
);

-- SDG reference table (17 UN Sustainable Development Goals)
CREATE TABLE sdg(
  sdg_id INTEGER,
  sdg_name VARCHAR
);
```

### Feature Flags
`USE_DATABASE = True` in `config/constants.py` enables DuckDB storage (vs session state fallback).

## EVM Domain Context

EVM (Earned Value Management) metrics calculated in `core/evm_engine.py`:
- **BAC**: Budget at Completion (total planned budget)
- **PV**: Planned Value (budgeted cost of work scheduled)
- **EV**: Earned Value (budgeted cost of work performed)
- **AC**: Actual Cost (actual cost of work performed)
- **CPI**: Cost Performance Index = EV/AC (>1 = under budget)
- **SPI**: Schedule Performance Index = EV/PV (>1 = ahead of schedule)
- **EAC**: Estimate at Completion (forecasted total cost)

S-curves (linear, beta, front-loaded, back-loaded) model planned value distribution over time.

## Page Numbering Convention
Pages in `pages/` are numbered for navigation order:
- 1-4: Data management (Portfolio, Load Data, Enter Data, Baselines)
- 5-7: Analysis & visualization
- 8-9: Strategic factors & SDGs
- 10: User management
- 99: Database diagnostics

## Environment Variables
Required for deployment:
- `GOOGLE_CLIENT_ID`, `GOOGLE_CLIENT_SECRET` - OAuth credentials
- `FIRESTORE_PROJECT_ID` - Firebase project
- `GOOGLE_APPLICATION_CREDENTIALS` - Service account path

Secrets stored in `.streamlit/secrets.toml` for local development.

**Dual Storage Architecture**: Firestore stores user profiles, permissions, and page access. DuckDB/MotherDuck stores all portfolio and project data. Users are synced to DuckDB's `app_user` table via `utils/user_manager.py:ensure_user_exists()` on login.

**Database Access Policy**:
- All users (except SuperAdmin) connect to MotherDuck `md:portfolio_cloud` only
- Non-SuperAdmin users cannot configure database connections or use local DuckDB
- Only SuperAdmins can use custom database connections or local DuckDB files
- This is enforced in `utils/database_config.py:get_user_database_config()`

## Authentication & Authorization

**OAuth Flow**:
1. `utils/auth.py:check_authentication()` checks authentication status
2. Streamlit native OAuth exchanges code for token and retrieves user info
3. `utils/user_manager.py:ensure_user_exists()` creates/updates Firestore user profile
4. `utils/user_sync.py:sync_authenticated_user_to_db()` syncs user to DuckDB `app_user` table

**Permission Levels**:
- **Roles**: `super_admin`, `admin`, `user`
- **Status**: `active`, `suspended`
- **Page Permissions**: `write`, `read`, `none` (per page, per user in Firestore)
- `require_page_access()` checks for read OR write access (page visibility)
- `can_write()` checks for write access only (save button visibility)

**Bootstrap Admins**: `khalid0211@gmail.com`, `dev@localhost` automatically receive `super_admin` role to prevent lockout.

## Baseline Effectivity System

**Critical Concept**: Multiple baseline versions per project track changes over time. Each baseline has an effectivity window `[baseline_start_date, baseline_end_date)`. Only ONE active baseline exists per project (where `baseline_end_date IS NULL`).

**Effectivity Query Pattern** (`database/queries.py:get_effective_baseline()`):
```python
# Retrieves the baseline that was active on a given status_date
SELECT * FROM project_baseline
WHERE project_id = ?
  AND baseline_start_date <= ?              # Status date must be after baseline started
  AND (baseline_end_date IS NULL            # Active baseline OR
       OR ? < baseline_end_date)            # Status date before baseline ended
  AND is_active = TRUE
ORDER BY baseline_version DESC
LIMIT 1
```

**Baseline Constraints**:
- Baseline version 0 (initial baseline) cannot be edited or deleted
- Planned start date is **locked** after any `actual_cost > 0` exists for the project
- Creating a new active baseline automatically closes the previous one by setting its `baseline_end_date`
- Historical status reports always reference the baseline version that was active at their `status_date`

## EVM Calculation Details

**Data Flow**:
1. User inputs actuals (AC) via status reports
2. System retrieves effective baseline for status_date
3. `core/evm_engine.py:perform_complete_evm_analysis()` calculates all metrics
4. Results stored back in `project_status_report` with `calculation_performed = TRUE`

**S-Curve Implementation** (uses scipy.stats.beta):
- **Linear**: `PV = BAC × (time_elapsed / total_duration)`
- **S-Curve (Beta)**: Uses beta distribution CDF for smooth acceleration/deceleration
- Default parameters: `alpha=2.0, beta=2.0` (symmetric S-curve)
- Earned Schedule uses inverse CDF: `scipy.stats.beta.ppf()` for days-based calculation

**Key Constraints**:
- **"Not Started" Detection**: Project considered not started if `data_date <= plan_start_date` OR `AC == 0`
- **Likely Duration Cap**: `LD = min(OD / SPIe, 2.5 × OD)` prevents unrealistic forecasts
- **Present Value**: Uses compound monthly rate from annual inflation: `monthly_rate = (1 + annual_rate)^(1/12) - 1`

**Critical Metrics**:
- **CPI** = EV / AC (cost efficiency)
- **SPI** = EV / PV (schedule efficiency based on value)
- **SPIe** = ES / AT (earned schedule efficiency based on time)
- **TCPI** = (BAC - EV) / (BAC - AC) (efficiency needed to meet budget)
- **EAC** = BAC / CPI (forecasted total cost)

## Session State Management

**Streamlit Behavior**: Every user interaction triggers a complete page rerun. Use `st.session_state` for data persistence.

**Common Session State Keys**:
- `st.session_state.authenticated` - OAuth authentication status
- `st.session_state.user_info` - User profile (name, email, photo)
- `st.session_state.current_portfolio_id` - Selected portfolio
- `st.session_state.current_status_date` - Selected reporting period
- `st.session_state.config_dict` - Portfolio configuration (tier configs, currency)

**Portfolio Context Pattern**:
```python
from utils.portfolio_context import render_portfolio_context

# Renders portfolio/date selector and stores in session state
portfolio_id, status_date = render_portfolio_context()
```

## Common Pitfalls & Anti-Patterns

**Database Connection Issues**:
```python
# ❌ WRONG - Database calls in loops (causes connection corruption)
for _, row in df.iterrows():
    db.execute("INSERT INTO ...", (row['col1'], row['col2']))

# ✅ CORRECT - Single batch query
values = [(row['col1'], row['col2']) for _, row in df.iterrows()]
db.executemany("INSERT INTO ...", values)

# ❌ WRONG - Chained fetch pattern (causes "No open result set" errors)
result = db.execute(query, params).fetchone()

# ✅ CORRECT - Use direct fetch method
result = db.fetch_one(query, params)
```

**Date Handling from CSV**:
```python
# CSV dates come as strings, need parsing with dayfirst=True for DD/MM/YYYY
if isinstance(date_val, str):
    date_val = pd.to_datetime(date_val, dayfirst=True, errors='coerce')
if isinstance(date_val, pd.Timestamp):
    date_val = date_val.date()  # Convert to Python date object
```

**Soft Delete Cascade**:
When soft-deleting portfolios, cascade to all related entities:
1. Set `portfolio.is_active = FALSE`
2. Set `project.is_active = FALSE` for all projects in portfolio
3. Set `project_baseline.is_active = FALSE` for all baselines
4. Set `project_status_report.is_active = FALSE` for all status reports
5. Set `project_sdg.is_active = FALSE`, `project_factor_score.is_active = FALSE`

**Connection Corruption Recovery**: If database connection becomes corrupted (symptoms: hangs, "No open result set" errors), restart Streamlit. Connection is singleton and persists across page navigation.

## Multi-Tenant Architecture

**Portfolio Ownership Model**:
- `portfolio_ownership` table (many-to-many): maps portfolios to owner users
- All queries filtered by: `portfolio_id IN (SELECT portfolio_id FROM portfolio_ownership WHERE owner_user_id = ?)`
- SuperAdmins see all portfolios; regular users see only owned portfolios

**Data Isolation**:
- `portfolio_id` denormalized into all child tables (`project`, `project_baseline`, `project_status_report`, etc.)
- Improves query performance (avoids joins)
- All queries include `WHERE portfolio_id = ? AND is_active = TRUE`

## Performance Patterns

**Database Indexes** (in `database/db_schema.py`):
- `idx_project_portfolio` on `project(portfolio_id)` - portfolio filtering
- `idx_baseline_dates` on `project_baseline(baseline_start_date, baseline_end_date)` - effectivity queries
- `idx_status_date` on `project_status_report(status_date)` - time-series queries

**Query Optimization**:
- Use `.df()` for bulk DataFrame operations: `df = db.execute(query).df()`
- Pre-calculate EVM metrics and store in database (avoid recalculation)
- Denormalize frequently-joined columns (e.g., `portfolio_id` in all tables)

## Schema Migrations

**Migration Pattern**:
- Stored in `database/migrations/*.py`
- Run manually (no automatic migration system)
- Must be idempotent (safe to rerun)
- Pattern: Check if column/table exists → ALTER TABLE if needed → Backfill data

**Recent Migrations**:
- `add_evm_columns.py` - Added `calculation_performed` and all EVM metric columns
- `add_is_active_to_all_tables.py` - Added soft delete support
- `add_portfolio_id_to_baseline.py` - Denormalized for performance
- `add_portfolio_ownership.py` - Separated ownership from portfolio table

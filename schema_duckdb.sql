-- DuckDB schema with Ownership, Baselines, SDGs, and Factors
-- Enforce some rules in application logic (DuckDB triggers/partial indexes differ from Postgres).

CREATE TABLE app_user (
  user_id BIGINT PRIMARY KEY,
  display_name VARCHAR NOT NULL,
  email VARCHAR NOT NULL UNIQUE,
  is_active BOOLEAN NOT NULL DEFAULT TRUE,
  created_at TIMESTAMP NOT NULL DEFAULT now()
);

CREATE TABLE portfolio (
  portfolio_id BIGINT PRIMARY KEY,
  portfolio_name VARCHAR NOT NULL UNIQUE,
  managing_organization VARCHAR,
  portfolio_manager VARCHAR,
  owner_user_id BIGINT NOT NULL REFERENCES app_user(user_id),
  is_active BOOLEAN NOT NULL DEFAULT TRUE,
  created_at TIMESTAMP NOT NULL DEFAULT now()
);

CREATE TABLE project (
  project_id BIGINT PRIMARY KEY,
  portfolio_id BIGINT NOT NULL REFERENCES portfolio(portfolio_id),
  project_code VARCHAR,
  project_name VARCHAR NOT NULL,
  responsible_organization VARCHAR,
  project_manager VARCHAR,
  project_status VARCHAR NOT NULL DEFAULT 'Ongoing',
  planned_start_date DATE,
  planned_finish_date DATE,
  initial_budget DECIMAL(18,2),
  current_budget DECIMAL(18,2),
  is_active BOOLEAN NOT NULL DEFAULT TRUE,
  created_at TIMESTAMP NOT NULL DEFAULT now(),
  updated_at TIMESTAMP NOT NULL DEFAULT now(),
  UNIQUE(portfolio_id, project_name)
);

CREATE TABLE project_baseline (
  baseline_id BIGINT PRIMARY KEY,
  project_id BIGINT NOT NULL REFERENCES project(project_id),
  portfolio_id BIGINT NOT NULL REFERENCES portfolio(portfolio_id),
  baseline_version INTEGER NOT NULL,
  baseline_start_date DATE NOT NULL,
  baseline_end_date DATE,
  planned_start_date DATE,
  planned_finish_date DATE,
  budget_at_completion DECIMAL(18,2),
  project_status VARCHAR,
  baseline_reason VARCHAR,
  approved_by VARCHAR,
  approved_date DATE,
  is_active BOOLEAN NOT NULL DEFAULT TRUE,
  created_at TIMESTAMP NOT NULL DEFAULT now(),
  UNIQUE(project_id, baseline_version)
);

CREATE TABLE project_status_report (
  status_report_id BIGINT PRIMARY KEY,
  portfolio_id BIGINT NOT NULL REFERENCES portfolio(portfolio_id),
  project_id BIGINT NOT NULL REFERENCES project(project_id),
  status_date DATE NOT NULL,
  actual_cost DECIMAL(18,2) NOT NULL DEFAULT 0,
  planned_value DECIMAL(18,2),
  earned_value DECIMAL(18,2),
  notes VARCHAR,
  is_active BOOLEAN NOT NULL DEFAULT TRUE,
  created_at TIMESTAMP NOT NULL DEFAULT now(),
  UNIQUE(project_id, status_date)
);

CREATE TABLE sdg (
  sdg_id INTEGER PRIMARY KEY,
  sdg_name VARCHAR NOT NULL
);

CREATE TABLE project_sdg (
  portfolio_id BIGINT NOT NULL REFERENCES portfolio(portfolio_id),
  project_id BIGINT NOT NULL REFERENCES project(project_id),
  sdg_id INTEGER NOT NULL REFERENCES sdg(sdg_id),
  weight_percent DECIMAL(5,2) NOT NULL,
  is_active BOOLEAN NOT NULL DEFAULT TRUE,
  created_at TIMESTAMP NOT NULL DEFAULT now(),
  PRIMARY KEY(project_id, sdg_id)
);

CREATE TABLE portfolio_factor (
  factor_id BIGINT PRIMARY KEY,
  portfolio_id BIGINT NOT NULL REFERENCES portfolio(portfolio_id),
  factor_name VARCHAR NOT NULL,
  factor_weight_percent DECIMAL(5,2) NOT NULL,
  likert_min INTEGER NOT NULL DEFAULT 1,
  likert_max INTEGER NOT NULL DEFAULT 5,
  is_active BOOLEAN NOT NULL DEFAULT TRUE,
  created_at TIMESTAMP NOT NULL DEFAULT now(),
  UNIQUE(portfolio_id, factor_name)
);

CREATE TABLE project_factor_score (
  portfolio_id BIGINT NOT NULL,
  project_id BIGINT NOT NULL REFERENCES project(project_id),
  factor_id BIGINT NOT NULL,
  score INTEGER NOT NULL,
  is_active BOOLEAN NOT NULL DEFAULT TRUE,
  scored_at TIMESTAMP NOT NULL DEFAULT now(),
  scored_by_user_id BIGINT REFERENCES app_user(user_id),
  PRIMARY KEY(project_id, factor_id)
);

-- DuckDB enforcement notes:
-- 1) Planned start lock after AC exists: reject baseline inserts that change planned_start_date if any status row has actual_cost > 0.
-- 2) One active baseline per project: ensure only one baseline_end_date IS NULL per project.
-- 3) SDG weights sum to 100 and factor weights sum to 100: validate at save-time.

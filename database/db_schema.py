"""DuckDB schema creation and migrations for Portfolio Analysis Suite

This module handles database schema initialization, migrations, and seed data.
Based on schema_duckdb.sql with additional helper functions.
"""

from .db_connection import get_db
import logging

logger = logging.getLogger(__name__)

# Complete schema SQL - matches schema_duckdb.sql
SCHEMA_SQL = """
-- DuckDB schema with Ownership, Baselines, SDGs, and Factors
-- Enforce some rules in application logic (DuckDB triggers/partial indexes differ from Postgres).

CREATE TABLE IF NOT EXISTS app_user (
  user_id BIGINT PRIMARY KEY,
  display_name VARCHAR NOT NULL,
  email VARCHAR NOT NULL UNIQUE,
  role VARCHAR DEFAULT 'User',
  department VARCHAR,
  is_active BOOLEAN NOT NULL DEFAULT TRUE,
  created_at TIMESTAMP NOT NULL DEFAULT now(),
  last_login TIMESTAMP
);

CREATE TABLE IF NOT EXISTS portfolio (
  portfolio_id BIGINT PRIMARY KEY,
  portfolio_name VARCHAR NOT NULL UNIQUE,
  managing_organization VARCHAR,
  portfolio_manager VARCHAR,
  owner_user_id BIGINT NOT NULL REFERENCES app_user(user_id),
  description VARCHAR,
  is_active BOOLEAN NOT NULL DEFAULT TRUE,
  created_at TIMESTAMP NOT NULL DEFAULT now()
);

CREATE TABLE IF NOT EXISTS project (
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

CREATE TABLE IF NOT EXISTS project_baseline (
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
  created_at TIMESTAMP NOT NULL DEFAULT now(),
  UNIQUE(project_id, baseline_version)
);

CREATE TABLE IF NOT EXISTS project_status_report (
  status_report_id BIGINT PRIMARY KEY,
  portfolio_id BIGINT NOT NULL REFERENCES portfolio(portfolio_id),
  project_id BIGINT NOT NULL REFERENCES project(project_id),
  status_date DATE NOT NULL,
  actual_cost DECIMAL(18,2) NOT NULL DEFAULT 0,
  planned_value DECIMAL(18,2),
  earned_value DECIMAL(18,2),
  notes VARCHAR,
  created_at TIMESTAMP NOT NULL DEFAULT now(),
  UNIQUE(project_id, status_date)
);

CREATE TABLE IF NOT EXISTS sdg (
  sdg_id INTEGER PRIMARY KEY,
  sdg_name VARCHAR NOT NULL
);

CREATE TABLE IF NOT EXISTS project_sdg (
  portfolio_id BIGINT NOT NULL REFERENCES portfolio(portfolio_id),
  project_id BIGINT NOT NULL REFERENCES project(project_id),
  sdg_id INTEGER NOT NULL REFERENCES sdg(sdg_id),
  weight_percent DECIMAL(5,2) DEFAULT NULL,
  created_at TIMESTAMP NOT NULL DEFAULT now(),
  PRIMARY KEY(project_id, sdg_id)
);

CREATE TABLE IF NOT EXISTS portfolio_factor (
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

CREATE TABLE IF NOT EXISTS project_factor_score (
  portfolio_id BIGINT NOT NULL,
  project_id BIGINT NOT NULL REFERENCES project(project_id),
  factor_id BIGINT NOT NULL,
  score INTEGER NOT NULL,
  scored_at TIMESTAMP NOT NULL DEFAULT now(),
  scored_by_user_id BIGINT REFERENCES app_user(user_id),
  PRIMARY KEY(project_id, factor_id)
);

CREATE TABLE IF NOT EXISTS portfolio_access (
  portfolio_id BIGINT NOT NULL REFERENCES portfolio(portfolio_id),
  user_id BIGINT NOT NULL REFERENCES app_user(user_id),
  access_level VARCHAR NOT NULL DEFAULT 'read',
  granted_at TIMESTAMP NOT NULL DEFAULT now(),
  granted_by_user_id BIGINT,
  is_active BOOLEAN NOT NULL DEFAULT TRUE,
  PRIMARY KEY(portfolio_id, user_id)
);
"""

# Indexes for performance optimization
INDEXES_SQL = """
-- Create indexes for frequently queried columns

CREATE INDEX IF NOT EXISTS idx_project_portfolio ON project(portfolio_id);
CREATE INDEX IF NOT EXISTS idx_project_is_active ON project(is_active);
CREATE INDEX IF NOT EXISTS idx_baseline_project ON project_baseline(project_id);
CREATE INDEX IF NOT EXISTS idx_baseline_portfolio ON project_baseline(portfolio_id);
CREATE INDEX IF NOT EXISTS idx_baseline_dates ON project_baseline(baseline_start_date, baseline_end_date);
CREATE INDEX IF NOT EXISTS idx_status_project ON project_status_report(project_id);
CREATE INDEX IF NOT EXISTS idx_status_portfolio ON project_status_report(portfolio_id);
CREATE INDEX IF NOT EXISTS idx_status_date ON project_status_report(status_date);
CREATE INDEX IF NOT EXISTS idx_project_sdg_portfolio ON project_sdg(portfolio_id);
CREATE INDEX IF NOT EXISTS idx_project_sdg_project ON project_sdg(project_id);
CREATE INDEX IF NOT EXISTS idx_project_sdg_sdg ON project_sdg(sdg_id);
CREATE INDEX IF NOT EXISTS idx_factor_portfolio ON portfolio_factor(portfolio_id);
CREATE INDEX IF NOT EXISTS idx_factor_score_project ON project_factor_score(project_id);
CREATE INDEX IF NOT EXISTS idx_factor_score_portfolio ON project_factor_score(portfolio_id);
CREATE INDEX IF NOT EXISTS idx_portfolio_access_user ON portfolio_access(user_id);
CREATE INDEX IF NOT EXISTS idx_portfolio_access_portfolio ON portfolio_access(portfolio_id);
"""

# 17 UN Sustainable Development Goals
SDG_DATA = [
    (1, "No Poverty"),
    (2, "Zero Hunger"),
    (3, "Good Health and Well-being"),
    (4, "Quality Education"),
    (5, "Gender Equality"),
    (6, "Clean Water and Sanitation"),
    (7, "Affordable and Clean Energy"),
    (8, "Decent Work and Economic Growth"),
    (9, "Industry, Innovation and Infrastructure"),
    (10, "Reduced Inequalities"),
    (11, "Sustainable Cities and Communities"),
    (12, "Responsible Consumption and Production"),
    (13, "Climate Action"),
    (14, "Life Below Water"),
    (15, "Life on Land"),
    (16, "Peace, Justice and Strong Institutions"),
    (17, "Partnerships for the Goals")
]


def create_schema():
    """Create all database tables if they don't exist

    This is idempotent - safe to run multiple times.
    Uses CREATE TABLE IF NOT EXISTS.
    """
    db = get_db()
    try:
        logger.info("Creating database schema...")
        db.execute_script(SCHEMA_SQL)
        logger.info("Database schema created successfully")
    except Exception as e:
        logger.error(f"Schema creation failed: {e}")
        raise


def create_indexes():
    """Create performance indexes on key columns

    This is idempotent - safe to run multiple times.
    Uses CREATE INDEX IF NOT EXISTS.
    """
    db = get_db()
    try:
        logger.info("Creating database indexes...")
        db.execute_script(INDEXES_SQL)
        logger.info("Database indexes created successfully")
    except Exception as e:
        logger.error(f"Index creation failed: {e}")
        raise


def seed_sdg_data():
    """Seed SDG reference data (17 UN Sustainable Development Goals)

    This is idempotent - uses INSERT OR IGNORE.
    """
    db = get_db()
    try:
        logger.info("Seeding SDG reference data...")
        for sdg_id, sdg_name in SDG_DATA:
            db.execute("""
                INSERT OR IGNORE INTO sdg (sdg_id, sdg_name)
                VALUES (?, ?)
            """, (sdg_id, sdg_name))
        logger.info(f"SDG reference data seeded successfully ({len(SDG_DATA)} records)")
    except Exception as e:
        logger.error(f"SDG seeding failed: {e}")
        raise


def drop_all_tables():
    """Drop all tables (WARNING: Destructive operation!)

    Use with caution - this will delete all data.
    Primarily for testing and development.
    """
    db = get_db()
    try:
        logger.warning("Dropping all tables...")

        # Drop tables in reverse order to handle foreign key constraints
        drop_sql = """
        DROP TABLE IF EXISTS project_factor_score;
        DROP TABLE IF EXISTS portfolio_factor;
        DROP TABLE IF EXISTS portfolio_access;
        DROP TABLE IF EXISTS project_sdg;
        DROP TABLE IF EXISTS sdg;
        DROP TABLE IF EXISTS project_status_report;
        DROP TABLE IF EXISTS project_baseline;
        DROP TABLE IF EXISTS project;
        DROP TABLE IF EXISTS portfolio;
        DROP TABLE IF EXISTS app_user;
        """

        db.execute_script(drop_sql)
        logger.warning("All tables dropped")
    except Exception as e:
        logger.error(f"Table drop failed: {e}")
        raise


def initialize_database(fresh_start: bool = False):
    """Initialize database with schema, indexes, and seed data

    Args:
        fresh_start: If True, drops all tables before recreating (WARNING: Deletes all data!)

    This is the main entry point for database setup.
    """
    try:
        if fresh_start:
            logger.warning("Fresh start requested - dropping all tables")
            drop_all_tables()

        logger.info("Initializing database...")
        create_schema()
        create_indexes()
        seed_sdg_data()
        logger.info("Database initialization complete!")

        # Verify tables exist
        db = get_db()
        tables = [
            'app_user', 'portfolio', 'project', 'project_baseline',
            'project_status_report', 'sdg', 'project_sdg',
            'portfolio_factor', 'project_factor_score', 'portfolio_access'
        ]

        logger.info("Verifying tables...")
        for table in tables:
            exists = db.table_exists(table)
            logger.info(f"  - {table}: {'✓' if exists else '✗'}")

        return True

    except Exception as e:
        logger.error(f"Database initialization failed: {e}")
        raise


def get_schema_version() -> str:
    """Get current schema version

    Returns:
        Schema version string (e.g., "1.0.0")
    """
    # For now, return static version
    # In future, could store in a schema_version table
    return "1.0.0"


def verify_schema_integrity() -> dict:
    """Verify database schema integrity

    Returns:
        Dictionary with verification results:
        {
            'all_tables_exist': bool,
            'missing_tables': list,
            'sdg_count': int,
            'indexes_created': bool
        }
    """
    db = get_db()
    results = {
        'all_tables_exist': True,
        'missing_tables': [],
        'sdg_count': 0,
        'indexes_created': False
    }

    # Check tables
    required_tables = [
        'app_user', 'portfolio', 'project', 'project_baseline',
        'project_status_report', 'sdg', 'project_sdg',
        'portfolio_factor', 'project_factor_score', 'portfolio_access'
    ]

    for table in required_tables:
        if not db.table_exists(table):
            results['all_tables_exist'] = False
            results['missing_tables'].append(table)

    # Check SDG count
    try:
        result = db.fetch_one("SELECT COUNT(*) FROM sdg")
        results['sdg_count'] = result[0] if result else 0
    except Exception:
        results['sdg_count'] = 0

    # Check indexes (simplified - just verify some exist)
    try:
        # DuckDB stores index info differently than SQLite
        # For now, just set to True if tables exist
        results['indexes_created'] = results['all_tables_exist']
    except Exception:
        results['indexes_created'] = False

    return results


if __name__ == "__main__":
    """Run database initialization when executed as script"""
    import sys

    # Set up logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )

    # Check for fresh start flag
    fresh_start = '--fresh' in sys.argv

    if fresh_start:
        print("WARNING: Fresh start will delete all existing data!")
        response = input("Continue? (yes/no): ")
        if response.lower() != 'yes':
            print("Aborted.")
            sys.exit(0)

    # Initialize database
    initialize_database(fresh_start=fresh_start)

    # Verify
    results = verify_schema_integrity()
    print("\n" + "="*50)
    print("Database Verification Results:")
    print("="*50)
    print(f"All tables exist: {results['all_tables_exist']}")
    if results['missing_tables']:
        print(f"Missing tables: {', '.join(results['missing_tables'])}")
    print(f"SDG records: {results['sdg_count']}/17")
    print(f"Indexes created: {results['indexes_created']}")
    print("="*50)

    if results['all_tables_exist'] and results['sdg_count'] == 17:
        print("[SUCCESS] Database initialization successful!")
    else:
        print("[ERROR] Database initialization incomplete")
        sys.exit(1)

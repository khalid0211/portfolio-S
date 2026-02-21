"""Application constants and configuration."""

from pathlib import Path
import re

# Application constants
APP_TITLE = "Project Analysis - EVM Intelligence Suite 📊"
DEFAULT_DATASET_TABLE = "dataset"
RESULTS_TABLE = "evm_results"
CONFIG_TABLE = "app_config"

# Calculation constants
DAYS_PER_MONTH = 30.44
INTEGRATION_STEPS = 200
MAX_TIMEOUT_SECONDS = 120
MIN_TIMEOUT_SECONDS = 10

# Config file paths for local storage
CONFIG_DIR = Path.home() / ".portfolio_suite"
MODEL_CONFIG_FILE = CONFIG_DIR / "model_config.json"

# Validation patterns
VALID_TABLE_NAME_PATTERN = re.compile(r'^[a-zA-Z0-9_-]+$')
VALID_COLUMN_NAME_PATTERN = re.compile(r'^[a-zA-Z0-9_\s-]+$')

# Default analysis configuration
DEFAULT_ANALYSIS_CONFIG = {
    'curve_type': 'linear',
    'alpha': 2.0,
    'beta': 2.0,
    'currency_symbol': '$',
    'currency_postfix': '',
    'date_format': 'YYYY-MM-DD',
    'annual_inflation_rate': 0.03
}

# UI Configuration
SIDEBAR_SECTIONS = [
    'A. Data Source',
    'B. Column Mapping',
    'C. Analysis Controls',
    'D. LLM Provider',
    'E. Batch Calculation',
    'F. Save & Download'
]

# Color schemes for performance indicators
PERFORMANCE_COLORS = {
    'excellent': '#28a745',  # Green
    'good': '#17a2b8',       # Blue
    'acceptable': '#ffc107', # Yellow
    'poor': '#dc3545',       # Red
    'critical': '#6f42c1'    # Purple
}

# ============================================================================
# DATABASE SETTINGS (Phase A: Database Foundation)
# ============================================================================

# Feature flag - set to True to use database instead of session state
# Default: False (session state mode - existing behavior)
USE_DATABASE = True

# Database file path
# None = default to ~/.portfolio_suite/portfolio.duckdb
# Using DuckDB subdirectory in application root
DATABASE_PATH = Path(__file__).parent.parent / "DuckDB" / "portfolio.duckdb"

# Database configuration
DATABASE_CONFIG = {
    'enable_query_logging': False,  # Set to True for debugging
    'connection_timeout': 30,        # Seconds
    'enable_cache': True,            # Enable query result caching
    'cache_ttl': 300                 # Cache time-to-live in seconds
}
"""Migration: Add EVM calculation result columns to project_status_report

This migration adds 30+ columns to store calculated EVM metrics,
eliminating the need to recalculate when viewing historical periods.

Run with: python -m database.migrations.add_evm_columns
"""

import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from database.db_connection import get_db
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def add_evm_columns():
    """Add EVM calculation result columns to project_status_report table"""

    db = get_db()
    conn = db.get_connection()

    columns_to_add = [
        # Calculation metadata
        ("calculation_performed", "BOOLEAN DEFAULT FALSE"),
        ("calculation_timestamp", "TIMESTAMP"),
        ("curve_type", "VARCHAR"),
        ("alpha", "DECIMAL(5,2)"),
        ("beta", "DECIMAL(5,2)"),
        ("inflation_rate", "DECIMAL(5,3)"),

        # Duration metrics
        ("actual_duration_months", "DECIMAL(10,2)"),
        ("original_duration_months", "DECIMAL(10,2)"),

        # Calculated values (not manual inputs)
        ("calculated_pv", "DECIMAL(18,2)"),
        ("calculated_ev", "DECIMAL(18,2)"),
        ("present_value", "DECIMAL(18,2)"),

        # Percentage metrics
        ("percent_complete", "DECIMAL(5,2)"),
        ("percent_budget_used", "DECIMAL(5,2)"),
        ("percent_time_used", "DECIMAL(5,2)"),
        ("percent_present_value_project", "DECIMAL(5,2)"),
        ("percent_likely_value_project", "DECIMAL(5,2)"),

        # Cost variance metrics
        ("cv", "DECIMAL(18,2)"),
        ("cpi", "DECIMAL(10,3)"),
        ("eac", "DECIMAL(18,2)"),
        ("etc", "DECIMAL(18,2)"),
        ("vac", "DECIMAL(18,2)"),
        ("tcpi", "DECIMAL(10,3)"),

        # Schedule variance metrics
        ("sv", "DECIMAL(18,2)"),
        ("spi", "DECIMAL(10,3)"),

        # Earned Schedule metrics
        ("es", "DECIMAL(10,2)"),
        ("espi", "DECIMAL(10,3)"),
        ("likely_duration", "DECIMAL(10,2)"),
        ("likely_completion", "DATE"),

        # Financial projections
        ("planned_value_project", "DECIMAL(18,2)"),
        ("likely_value_project", "DECIMAL(18,2)"),
    ]

    logger.info("Starting EVM columns migration...")
    added_count = 0
    skipped_count = 0

    for column_name, column_type in columns_to_add:
        try:
            alter_sql = f"ALTER TABLE project_status_report ADD COLUMN {column_name} {column_type}"
            conn.execute(alter_sql)
            logger.info(f"  [OK] Added column: {column_name}")
            added_count += 1
        except Exception as e:
            error_msg = str(e).lower()
            if "already exists" in error_msg or "duplicate" in error_msg:
                logger.info(f"  [SKIP] Column already exists: {column_name}")
                skipped_count += 1
            else:
                logger.error(f"  [ERROR] Failed to add {column_name}: {e}")
                raise

    logger.info("")
    logger.info("="*60)
    logger.info("Migration Summary:")
    logger.info(f"  Columns added: {added_count}")
    logger.info(f"  Columns skipped (already exist): {skipped_count}")
    logger.info(f"  Total columns: {len(columns_to_add)}")
    logger.info("="*60)

    # Verify the table structure
    logger.info("")
    logger.info("Verifying table structure...")
    cols = conn.execute("PRAGMA table_info(project_status_report)").fetchall()

    logger.info(f"project_status_report now has {len(cols)} columns:")
    for col in cols:
        logger.info(f"  - {col[1]:35s} {col[2]}")

    db.close()
    logger.info("")
    logger.info("[SUCCESS] Migration complete!")


if __name__ == "__main__":
    print("")
    print("="*60)
    print("EVM Columns Migration")
    print("="*60)
    print("")

    try:
        add_evm_columns()
        sys.exit(0)
    except Exception as e:
        logger.error(f"Migration failed: {e}")
        sys.exit(1)

"""Migration: Add is_active column to all tables

This migration adds is_active BOOLEAN DEFAULT TRUE column to:
1. project_baseline
2. project_status_report
3. project_sdg
4. project_factor_score
5. portfolio_ownership (if exists)

All existing records will be set to is_active = TRUE

Run with: python -m database.migrations.add_is_active_to_all_tables
"""

import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from database.db_connection import get_db
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def check_column_exists(conn, table_name: str, column_name: str) -> bool:
    """Check if a column exists in a table"""
    try:
        result = conn.execute(f"""
            SELECT COUNT(*)
            FROM information_schema.columns
            WHERE table_name = '{table_name}'
            AND column_name = '{column_name}'
        """).fetchone()
        return result[0] > 0
    except:
        return False


def check_table_exists(conn, table_name: str) -> bool:
    """Check if a table exists"""
    try:
        result = conn.execute(f"""
            SELECT COUNT(*)
            FROM information_schema.tables
            WHERE table_name = '{table_name}'
        """).fetchone()
        return result[0] > 0
    except:
        return False


def add_is_active_column(conn, table_name: str) -> bool:
    """Add is_active column to a table if it doesn't exist"""

    if check_column_exists(conn, table_name, 'is_active'):
        logger.info(f"  ⚠️  {table_name}.is_active already exists - skipping")
        return False

    logger.info(f"  Adding is_active to {table_name}...")

    # Add the column with DEFAULT TRUE (DuckDB doesn't support NOT NULL in ALTER TABLE ADD COLUMN)
    conn.execute(f"""
        ALTER TABLE {table_name}
        ADD COLUMN is_active BOOLEAN DEFAULT TRUE
    """)

    # Set all existing records to TRUE explicitly
    conn.execute(f"""
        UPDATE {table_name}
        SET is_active = TRUE
        WHERE is_active IS NULL OR is_active IS NOT TRUE
    """)

    # Verify all records are TRUE
    null_count = conn.execute(f"""
        SELECT COUNT(*) FROM {table_name} WHERE is_active IS NULL
    """).fetchone()[0]

    if null_count > 0:
        logger.warning(f"  ⚠️  WARNING: {null_count} NULL values found in {table_name}.is_active")
    else:
        logger.info(f"  ✅ Added is_active to {table_name} (all records set to TRUE)")
    return True


def migrate():
    """Add is_active column to all tables that need it"""

    db = get_db()
    conn = db.get_connection()

    logger.info("="*60)
    logger.info("Starting is_active column migration...")
    logger.info("="*60)
    logger.info("")

    tables_to_migrate = [
        'project_baseline',
        'project_status_report',
        'project_sdg',
        'project_factor_score',
        'portfolio_ownership'
    ]

    added_count = 0
    skipped_count = 0

    try:
        for table_name in tables_to_migrate:
            # Check if table exists first
            if not check_table_exists(conn, table_name):
                logger.info(f"  ⚠️  Table {table_name} does not exist - skipping")
                skipped_count += 1
                continue

            # Add is_active column
            if add_is_active_column(conn, table_name):
                added_count += 1
            else:
                skipped_count += 1

        logger.info("")
        logger.info("="*60)
        logger.info("Migration Completed Successfully!")
        logger.info("="*60)
        logger.info(f"  ✅ Added is_active column to {added_count} table(s)")
        logger.info(f"  ⚠️  Skipped {skipped_count} table(s) (already had column or didn't exist)")
        logger.info("="*60)
        logger.info("")
        logger.info("📋 Summary:")
        logger.info("  • All existing records are set to is_active = TRUE")
        logger.info("  • New inserts will default to is_active = TRUE")
        logger.info("  • Soft delete: Set is_active = FALSE")
        logger.info("  • Hard delete: Use purge function in Database Diagnostics")
        logger.info("="*60)

        db.close()

    except Exception as e:
        logger.error(f"Migration failed: {e}")
        import traceback
        traceback.print_exc()
        raise


if __name__ == "__main__":
    print("")
    print("="*60)
    print("Add is_active Column Migration")
    print("="*60)
    print("")
    print("This will add is_active BOOLEAN DEFAULT TRUE to:")
    print("  1. project_baseline")
    print("  2. project_status_report")
    print("  3. project_sdg")
    print("  4. project_factor_score")
    print("  5. portfolio_ownership (if exists)")
    print("")
    print("All existing records will be set to is_active = TRUE")
    print("")

    try:
        migrate()
        print("")
        print("[SUCCESS] Migration complete!")
        print("")
        sys.exit(0)
    except Exception as e:
        logger.error(f"Failed: {e}")
        sys.exit(1)

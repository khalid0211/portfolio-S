"""Migration: Add portfolio_id to project_sdg table

This migration adds portfolio_id as a NOT NULL foreign key to the project_sdg table
for denormalization and improved query performance.

Run this migration if you have an existing database without portfolio_id in project_sdg.
"""

import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from database.db_connection import get_db
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def check_column_exists():
    """Check if portfolio_id column already exists in project_sdg"""
    db = get_db()
    try:
        # Try to query the column
        db.execute("SELECT portfolio_id FROM project_sdg LIMIT 1")
        return True
    except Exception:
        return False


def migrate():
    """Add portfolio_id column to project_sdg table"""
    db = get_db()

    try:
        # Check if column already exists
        if check_column_exists():
            logger.info("portfolio_id column already exists in project_sdg table. No migration needed.")
            return True

        logger.info("Starting migration: Adding portfolio_id to project_sdg table...")

        # Step 1: Check if table is empty
        result = db.fetch_one("SELECT COUNT(*) FROM project_sdg")
        row_count = result[0] if result else 0

        if row_count > 0:
            logger.warning(f"Warning: project_sdg table has {row_count} rows. Migration will populate portfolio_id from project table.")

            # Step 2: Add column as nullable first
            logger.info("Adding portfolio_id column (nullable)...")
            db.execute("""
                ALTER TABLE project_sdg
                ADD COLUMN portfolio_id BIGINT
            """)

            # Step 3: Populate portfolio_id from project table
            logger.info("Populating portfolio_id from project table...")
            db.execute("""
                UPDATE project_sdg ps
                SET portfolio_id = (
                    SELECT p.portfolio_id
                    FROM project p
                    WHERE p.project_id = ps.project_id
                )
            """)

            # Step 4: Verify all rows were updated
            result = db.fetch_one("SELECT COUNT(*) FROM project_sdg WHERE portfolio_id IS NULL")
            null_count = result[0] if result else 0

            if null_count > 0:
                raise Exception(f"Migration failed: {null_count} rows still have NULL portfolio_id")

            # Step 5: Add NOT NULL constraint
            logger.info("Adding NOT NULL constraint...")
            db.execute("""
                ALTER TABLE project_sdg
                ALTER COLUMN portfolio_id SET NOT NULL
            """)

            # Step 6: Add foreign key constraint
            logger.info("Adding foreign key constraint...")
            db.execute("""
                ALTER TABLE project_sdg
                ADD CONSTRAINT fk_project_sdg_portfolio
                FOREIGN KEY (portfolio_id) REFERENCES portfolio(portfolio_id)
            """)
        else:
            # Table is empty, can add column directly with constraints
            logger.info("Table is empty. Adding portfolio_id column with constraints...")
            db.execute("""
                ALTER TABLE project_sdg
                ADD COLUMN portfolio_id BIGINT NOT NULL REFERENCES portfolio(portfolio_id)
            """)

        logger.info("Migration completed successfully!")

        # Verify the column exists
        if check_column_exists():
            logger.info("✓ Verification: portfolio_id column exists in project_sdg table")
            return True
        else:
            logger.error("✗ Verification failed: portfolio_id column not found")
            return False

    except Exception as e:
        logger.error(f"Migration failed: {e}")
        raise


def rollback():
    """Rollback migration - remove portfolio_id column"""
    db = get_db()

    try:
        logger.warning("Rolling back migration: Removing portfolio_id from project_sdg table...")

        # Drop foreign key constraint first (if it exists)
        try:
            db.execute("""
                ALTER TABLE project_sdg
                DROP CONSTRAINT IF EXISTS fk_project_sdg_portfolio
            """)
        except Exception as e:
            logger.warning(f"Could not drop constraint: {e}")

        # Drop column
        db.execute("""
            ALTER TABLE project_sdg
            DROP COLUMN IF EXISTS portfolio_id
        """)

        logger.warning("Rollback completed!")
        return True

    except Exception as e:
        logger.error(f"Rollback failed: {e}")
        raise


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description='Add portfolio_id to project_sdg table')
    parser.add_argument('--rollback', action='store_true', help='Rollback the migration')
    args = parser.parse_args()

    try:
        if args.rollback:
            rollback()
        else:
            migrate()
    except Exception as e:
        logger.error(f"Operation failed: {e}")
        sys.exit(1)

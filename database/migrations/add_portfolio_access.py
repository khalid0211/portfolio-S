"""Migration: Add portfolio_access table for shared portfolio access control

This migration:
1. Creates portfolio_access table
2. Creates indexes for efficient queries

Run with: python -m database.migrations.add_portfolio_access
"""

import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from database.db_connection import get_db
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def migrate():
    """Add portfolio_access table"""

    db = get_db()
    conn = db.get_connection()

    logger.info("Starting portfolio_access migration...")

    try:
        # Step 1: Create portfolio_access table
        logger.info("Step 1: Creating portfolio_access table...")
        conn.execute("""
            CREATE TABLE IF NOT EXISTS portfolio_access (
                portfolio_id BIGINT NOT NULL,
                user_id BIGINT NOT NULL,
                access_level VARCHAR NOT NULL DEFAULT 'read',
                granted_at TIMESTAMP NOT NULL DEFAULT now(),
                granted_by_user_id BIGINT,
                is_active BOOLEAN NOT NULL DEFAULT TRUE,
                PRIMARY KEY(portfolio_id, user_id)
            )
        """)
        logger.info("  portfolio_access table created")

        # Step 2: Create indexes
        logger.info("Step 2: Creating indexes...")
        conn.execute("CREATE INDEX IF NOT EXISTS idx_portfolio_access_user ON portfolio_access(user_id)")
        conn.execute("CREATE INDEX IF NOT EXISTS idx_portfolio_access_portfolio ON portfolio_access(portfolio_id)")
        logger.info("  Indexes created")

        logger.info("")
        logger.info("=" * 60)
        logger.info("Migration Completed Successfully!")
        logger.info("=" * 60)
        logger.info("  portfolio_access table created")
        logger.info("  Indexes created")
        logger.info("=" * 60)

        db.close()

    except Exception as e:
        logger.error(f"Migration failed: {e}")
        import traceback
        traceback.print_exc()
        raise


if __name__ == "__main__":
    print("")
    print("=" * 60)
    print("Portfolio Access Migration")
    print("=" * 60)
    print("")
    print("This will:")
    print("  1. Create portfolio_access table")
    print("  2. Create indexes for user and portfolio lookups")
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

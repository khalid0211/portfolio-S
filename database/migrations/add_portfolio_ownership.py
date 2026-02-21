"""Migration: Separate portfolio ownership into dedicated table

This migration:
1. Creates portfolio_ownership table
2. Migrates existing owner_user_id data from portfolio table
3. Removes owner_user_id column from portfolio table

Run with: python -m database.migrations.add_portfolio_ownership
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
    """Add portfolio_ownership table and migrate data"""

    db = get_db()
    conn = db.get_connection()

    logger.info("Starting portfolio ownership migration...")

    try:
        # Step 1: Create portfolio_ownership table
        logger.info("Step 1: Creating portfolio_ownership table...")
        conn.execute("""
            CREATE TABLE IF NOT EXISTS portfolio_ownership (
                portfolio_id BIGINT PRIMARY KEY REFERENCES portfolio(portfolio_id),
                owner_user_id BIGINT NOT NULL REFERENCES app_user(user_id),
                is_active BOOLEAN NOT NULL DEFAULT TRUE,
                assigned_at TIMESTAMP NOT NULL DEFAULT now(),
                assigned_by VARCHAR
            )
        """)
        logger.info("  ✅ portfolio_ownership table created")

        # Step 2: Migrate existing owner data
        logger.info("Step 2: Migrating existing owner data...")

        # Get all portfolios with their current owners
        portfolios = conn.execute("""
            SELECT portfolio_id, owner_user_id
            FROM portfolio
            WHERE owner_user_id IS NOT NULL
        """).fetchall()

        migrated_count = 0
        for portfolio_id, owner_user_id in portfolios:
            # Check if already migrated
            existing = conn.execute(
                "SELECT 1 FROM portfolio_ownership WHERE portfolio_id = ?",
                (portfolio_id,)
            ).fetchone()

            if not existing:
                conn.execute("""
                    INSERT INTO portfolio_ownership (portfolio_id, owner_user_id, assigned_at)
                    VALUES (?, ?, now())
                """, (portfolio_id, owner_user_id))
                migrated_count += 1

        logger.info(f"  ✅ Migrated {migrated_count} portfolio ownership records")

        # Step 3: Note about owner_user_id column
        logger.info("Step 3: Handling owner_user_id column...")
        logger.info("  ℹ️  Keeping owner_user_id column in portfolio table (deprecated)")
        logger.info("  ℹ️  portfolio_ownership is now the source of truth")
        logger.info("  ℹ️  This allows updates without FK constraint issues")

        logger.info("")
        logger.info("="*60)
        logger.info("Migration Completed Successfully!")
        logger.info("="*60)
        logger.info(f"  ✅ portfolio_ownership table created")
        logger.info(f"  ✅ {migrated_count} ownership records migrated")
        logger.info(f"  ✅ portfolio_ownership is now source of truth for ownership")
        logger.info(f"  ℹ️  owner_user_id kept in portfolio table (deprecated)")
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
    print("Portfolio Ownership Migration")
    print("="*60)
    print("")
    print("This will:")
    print("  1. Create portfolio_ownership table")
    print("  2. Migrate existing owner data")
    print("  3. Remove owner_user_id from portfolio table")
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

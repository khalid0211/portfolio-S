"""Migration: Fix portfolio_factor foreign key constraint

Adds ON UPDATE CASCADE to allow factor updates without FK violations

Run with: python -m database.migrations.fix_factor_fk_constraint
"""

import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from database.db_connection import get_db
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def fix_factor_fk():
    """Fix the foreign key constraint on project_factor_score"""

    db = get_db()
    conn = db.get_connection()

    logger.info("Starting foreign key constraint fix...")

    try:
        # Step 1: Backup data
        logger.info("Step 1: Backing up project_factor_score data...")
        scores = conn.execute("SELECT * FROM project_factor_score").fetchall()
        score_columns = [desc[0] for desc in conn.execute("SELECT * FROM project_factor_score LIMIT 0").description]
        logger.info(f"  Backed up {len(scores)} factor scores")

        # Step 2: Drop the table
        logger.info("Step 2: Dropping project_factor_score table...")
        conn.execute("DROP TABLE project_factor_score")
        logger.info("  Table dropped")

        # Step 3: Recreate with proper foreign key
        logger.info("Step 3: Recreating table with ON UPDATE CASCADE...")
        conn.execute("""
            CREATE TABLE project_factor_score (
                project_id BIGINT NOT NULL REFERENCES project(project_id),
                factor_id BIGINT NOT NULL REFERENCES portfolio_factor(factor_id) ON UPDATE CASCADE,
                score INTEGER NOT NULL,
                scored_at TIMESTAMP NOT NULL DEFAULT now(),
                scored_by_user_id BIGINT REFERENCES app_user(user_id),
                PRIMARY KEY(project_id, factor_id)
            )
        """)
        logger.info("  Table recreated with ON UPDATE CASCADE")

        # Step 4: Restore data
        logger.info("Step 4: Restoring factor scores...")
        if scores:
            placeholders = ','.join(['?' for _ in score_columns])
            for score in scores:
                conn.execute(f"INSERT INTO project_factor_score VALUES ({placeholders})", score)
        logger.info(f"  Restored {len(scores)} factor scores")

        # Step 5: Recreate index
        logger.info("Step 5: Recreating index...")
        conn.execute("CREATE INDEX IF NOT EXISTS idx_factor_score_project ON project_factor_score(project_id)")
        logger.info("  Index created")

        logger.info("")
        logger.info("="*60)
        logger.info("Migration Summary:")
        logger.info(f"  ✅ Foreign key constraint fixed")
        logger.info(f"  ✅ Restored {len(scores)} factor scores")
        logger.info(f"  ✅ Index recreated")
        logger.info(f"  ✅ Factor updates will now work without FK errors")
        logger.info("="*60)

        db.close()
        logger.info("")
        logger.info("[SUCCESS] Migration complete!")

    except Exception as e:
        logger.error(f"Migration failed: {e}")
        import traceback
        traceback.print_exc()
        raise


if __name__ == "__main__":
    print("")
    print("="*60)
    print("Fix Factor Foreign Key Constraint Migration")
    print("="*60)
    print("")
    print("This migration will:")
    print("  - Backup project_factor_score data")
    print("  - Drop and recreate the table with ON UPDATE CASCADE")
    print("  - Restore all data")
    print("  - Allow factor updates without foreign key errors")
    print("")

    response = input("Continue? (yes/no): ")
    if response.lower() != 'yes':
        print("Migration cancelled")
        sys.exit(0)

    try:
        fix_factor_fk()
        sys.exit(0)
    except Exception as e:
        logger.error(f"Migration failed: {e}")
        sys.exit(1)

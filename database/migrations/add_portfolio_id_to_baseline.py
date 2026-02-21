"""Migration: Add portfolio_id to project_baseline table

This migration:
1. Checks if portfolio_id column already exists
2. Adds portfolio_id column to project_baseline (if needed)
3. Populates it from the project table
4. Adds Foreign Key constraint to portfolio table
5. Recreates indexes

Run with: python -m database.migrations.add_portfolio_id_to_baseline

IMPORTANT: Close all applications using the database before running this migration.
"""

import sys
import os
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from database.db_connection import get_db, reset_db_instance
import logging

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def migrate():
    """Run the migration to add portfolio_id to project_baseline"""
    # Reset any existing connections to ensure clean state
    reset_db_instance()
    db = get_db()
    conn = db.get_connection()

    logger.info("="*70)
    logger.info("MIGRATION: Add portfolio_id to project_baseline")
    logger.info("="*70)

    try:
        # Check if column already exists
        logger.info("\n1. Checking current schema...")
        columns = conn.execute("PRAGMA table_info('project_baseline')").fetchall()
        col_names = [c[1] for c in columns]
        logger.info(f"   Current columns: {', '.join(col_names)}")

        if 'portfolio_id' in col_names:
            logger.info("✅ Column 'portfolio_id' already exists in project_baseline.")
            logger.info("   Migration not needed. Exiting.")
            return True

        # Get baseline count before migration
        baseline_count = conn.execute("SELECT COUNT(*) FROM project_baseline").fetchone()[0]
        logger.info(f"   Found {baseline_count} baseline records to migrate")

        # 1. Add column (nullable first)
        logger.info("\n2. Adding portfolio_id column...")
        conn.execute("ALTER TABLE project_baseline ADD COLUMN portfolio_id BIGINT")
            
        logger.info("   Column added successfully")

        # 2. Populate data
        logger.info("\n3. Populating portfolio_id from project table...")
        conn.execute("""
            UPDATE project_baseline
            SET portfolio_id = project.portfolio_id
            FROM project
            WHERE project_baseline.project_id = project.project_id
        """)

        updated_count = conn.execute(
            "SELECT COUNT(*) FROM project_baseline WHERE portfolio_id IS NOT NULL"
        ).fetchone()[0]
        logger.info(f"   Updated {updated_count} baseline records with portfolio_id")

        # 3. Handle orphaned baselines (if any, though strict FKs usually prevent this)
        null_count = conn.execute(
            "SELECT COUNT(*) FROM project_baseline WHERE portfolio_id IS NULL"
        ).fetchone()[0]

        if null_count > 0:
            logger.warning(f"\n⚠️  Found {null_count} orphaned baselines (no associated project)")
            logger.warning("   These will be deleted to ensure data integrity")
            conn.execute("DELETE FROM project_baseline WHERE portfolio_id IS NULL")
            logger.info(f"   Deleted {null_count} orphaned baseline records")

        # 4. Recreate table to enforce NOT NULL and FOREIGN KEY
        logger.info("\n4. Recreating table to enforce FK and NOT NULL constraints...")
        
        conn.execute("BEGIN TRANSACTION")
        
        # Rename old table
        conn.execute("ALTER TABLE project_baseline RENAME TO project_baseline_old")
        
        # Create new table with FK
        # Copying schema from previous definition but adding portfolio_id
        conn.execute("""
            CREATE TABLE project_baseline (
                baseline_id BIGINT PRIMARY KEY,
                project_id BIGINT NOT NULL REFERENCES project(project_id),
                portfolio_id BIGINT NOT NULL REFERENCES portfolio(portfolio_id), -- New Column
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
            )
        """)
        
        # Copy data back
        logger.info("Restoring data...")
        conn.execute("""
            INSERT INTO project_baseline (
                baseline_id, project_id, portfolio_id, baseline_version,
                baseline_start_date, baseline_end_date, planned_start_date,
                planned_finish_date, budget_at_completion, project_status,
                baseline_reason, approved_by, approved_date, created_at
            )
            SELECT 
                baseline_id, project_id, portfolio_id, baseline_version,
                baseline_start_date, baseline_end_date, planned_start_date,
                planned_finish_date, budget_at_completion, project_status,
                baseline_reason, approved_by, approved_date, created_at
            FROM project_baseline_old
        """)
        
        # Drop old table
        conn.execute("DROP TABLE project_baseline_old")
        
        # Recreate Indexes
        logger.info("\n5. Recreating indexes...")
        conn.execute("CREATE INDEX IF NOT EXISTS idx_baseline_project ON project_baseline(project_id)")
        conn.execute("CREATE INDEX IF NOT EXISTS idx_baseline_portfolio ON project_baseline(portfolio_id)")
        conn.execute("CREATE INDEX IF NOT EXISTS idx_baseline_dates ON project_baseline(baseline_start_date, baseline_end_date)")
        logger.info("   Indexes created successfully")

        conn.execute("COMMIT")

        # Verify final schema
        logger.info("\n6. Verifying final schema...")
        final_columns = conn.execute("PRAGMA table_info('project_baseline')").fetchall()
        final_col_names = [c[1] for c in final_columns]
        logger.info(f"   Final columns: {', '.join(final_col_names)}")

        if 'portfolio_id' in final_col_names:
            logger.info("\n" + "="*70)
            logger.info("✅ MIGRATION COMPLETED SUCCESSFULLY")
            logger.info("="*70)
            logger.info(f"   - {baseline_count} baseline records migrated")
            logger.info("   - portfolio_id column added with FK constraint")
            logger.info("   - All indexes recreated")
            logger.info("="*70)
            return True
        else:
            raise Exception("Migration completed but portfolio_id column not found in final schema")

    except Exception as e:
        try:
            conn.execute("ROLLBACK")
            logger.error("   Transaction rolled back")
        except:
            pass
        logger.error("\n" + "="*70)
        logger.error("❌ MIGRATION FAILED")
        logger.error("="*70)
        logger.error(f"Error: {e}")
        logger.error("="*70)
        import traceback
        logger.error(traceback.format_exc())
        return False

if __name__ == "__main__":
    success = migrate()

    if success:
        # Force checkpoint to ensure changes are written to main database file
        logger.info("\nCheckpointing database to ensure changes are persisted...")
        db = get_db()
        try:
            db.execute("CHECKPOINT")
            logger.info("✅ Database checkpointed successfully")
        except Exception as e:
            logger.warning(f"⚠️  Checkpoint warning: {e}")

        # Close connection to flush WAL
        db.close()
        logger.info("✅ Database connection closed, WAL flushed")

    exit(0 if success else 1)

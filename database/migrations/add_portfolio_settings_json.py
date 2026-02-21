"""
Migration: Add settings_json column to portfolio table

This migration adds a settings_json column to store portfolio-specific settings
including currency, tier configurations, and LLM provider settings.
"""

import logging
from database.db_connection import get_db

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def run_migration():
    """Add settings_json column to portfolio table"""
    db = get_db()
    conn = db.get_connection()

    try:
        logger.info("Starting migration: add_portfolio_settings_json")

        # Check if column already exists
        check_query = """
            SELECT column_name
            FROM information_schema.columns
            WHERE table_name = 'portfolio' AND column_name = 'settings_json'
        """
        existing = conn.execute(check_query).fetchone()

        if existing:
            logger.info("settings_json column already exists, skipping migration")
            return

        # Add settings_json column
        logger.info("Adding settings_json column to portfolio table...")
        conn.execute("""
            ALTER TABLE portfolio
            ADD COLUMN settings_json VARCHAR DEFAULT NULL
        """)

        logger.info("Migration completed successfully!")

    except Exception as e:
        logger.error(f"Migration failed: {e}")
        raise

if __name__ == "__main__":
    run_migration()

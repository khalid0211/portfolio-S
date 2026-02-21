"""
Reinitialize MotherDuck database with fresh schema
Connects directly to MotherDuck, bypassing user-specific config
"""

import sys
import duckdb
from database.db_schema import SCHEMA_SQL, INDEXES_SQL

def reinit_motherduck():
    print("=" * 60)
    print("MOTHERDUCK REINITIALIZATION")
    print("=" * 60)

    # Get MotherDuck token from secrets
    try:
        import streamlit as st
        if hasattr(st, 'secrets') and 'motherduck' in st.secrets:
            token = st.secrets['motherduck'].get('token', '')
            if not token:
                print("ERROR: MotherDuck token not found in secrets")
                return False
        else:
            print("ERROR: MotherDuck secrets not configured")
            return False
    except Exception as e:
        print(f"ERROR: Could not load MotherDuck token: {e}")
        return False

    # Ask for database name
    db_name = input("\nEnter MotherDuck database name (default: portfolio_db): ").strip()
    if not db_name:
        db_name = "portfolio_db"

    conn_str = f"md:{db_name}?motherduck_token={token}"

    print(f"\nConnecting to MotherDuck database: {db_name}")

    try:
        conn = duckdb.connect(conn_str)
        print("[OK] Connected to MotherDuck")
    except Exception as e:
        print(f"[ERROR] Connection failed: {e}")
        return False

    # Get current tables
    try:
        result = conn.execute("SELECT table_name FROM information_schema.tables WHERE table_schema = 'main'").fetchall()
        current_tables = [r[0] for r in result]
        print(f"\nFound {len(current_tables)} existing tables")
        if current_tables:
            print("Tables:", ", ".join(current_tables))
    except Exception as e:
        print(f"Error checking tables: {e}")
        current_tables = []

    # Confirm
    print("\nWARNING: This will DROP all existing tables and recreate them!")
    response = input("Continue? (yes/no): ").strip().lower()

    if response != 'yes':
        print("Aborted.")
        conn.close()
        return False

    # Drop existing tables in correct order
    print("\n" + "=" * 60)
    print("DROPPING EXISTING TABLES")
    print("=" * 60)

    drop_order = [
        'project_status_report',
        'project_sdg',
        'project_factor_score',
        'project_baseline',
        'project',
        'portfolio_ownership',
        'portfolio_factor',
        'portfolio',
        'sdg',
        'app_user'
    ]

    for table in drop_order:
        if table in current_tables:
            try:
                conn.execute(f"DROP TABLE IF EXISTS {table} CASCADE")
                print(f"[OK] Dropped table: {table}")
            except Exception as e:
                print(f"[WARN] Warning dropping {table}: {e}")

    # Create schema
    print("\n" + "=" * 60)
    print("CREATING SCHEMA")
    print("=" * 60)

    try:
        # Execute schema creation
        statements = [s.strip() for s in SCHEMA_SQL.split(';') if s.strip()]
        for i, stmt in enumerate(statements, 1):
            try:
                conn.execute(stmt)
                # Extract table name for better logging
                if 'CREATE TABLE' in stmt.upper():
                    table_name = stmt.split('CREATE TABLE')[1].split('(')[0].strip()
                    print(f"[OK] [{i}/{len(statements)}] Created table: {table_name}")
                else:
                    print(f"[OK] [{i}/{len(statements)}] Executed statement")
            except Exception as e:
                print(f"[ERROR] Error in statement {i}: {e}")
                print(f"Statement: {stmt[:100]}...")
                raise

        print("\n[OK] Schema created successfully")
    except Exception as e:
        print(f"\n[ERROR] Schema creation failed: {e}")
        conn.close()
        return False

    # Create indexes
    print("\n" + "=" * 60)
    print("CREATING INDEXES")
    print("=" * 60)

    try:
        statements = [s.strip() for s in INDEXES_SQL.split(';') if s.strip()]
        for i, stmt in enumerate(statements, 1):
            try:
                conn.execute(stmt)
                print(f"[OK] [{i}/{len(statements)}] Created index")
            except Exception as e:
                print(f"[WARN] Warning creating index {i}: {e}")

        print("\n[OK] Indexes created successfully")
    except Exception as e:
        print(f"\n[WARN] Some indexes failed: {e}")

    # Seed SDG data
    print("\n" + "=" * 60)
    print("SEEDING SDG DATA")
    print("=" * 60)

    try:
        sdg_data = [
            (1, 'No Poverty'),
            (2, 'Zero Hunger'),
            (3, 'Good Health and Well-being'),
            (4, 'Quality Education'),
            (5, 'Gender Equality'),
            (6, 'Clean Water and Sanitation'),
            (7, 'Affordable and Clean Energy'),
            (8, 'Decent Work and Economic Growth'),
            (9, 'Industry, Innovation and Infrastructure'),
            (10, 'Reduced Inequalities'),
            (11, 'Sustainable Cities and Communities'),
            (12, 'Responsible Consumption and Production'),
            (13, 'Climate Action'),
            (14, 'Life Below Water'),
            (15, 'Life on Land'),
            (16, 'Peace, Justice and Strong Institutions'),
            (17, 'Partnerships for the Goals')
        ]

        for sdg_id, sdg_name in sdg_data:
            conn.execute(
                "INSERT INTO sdg (sdg_id, sdg_name) VALUES (?, ?) ON CONFLICT DO NOTHING",
                (sdg_id, sdg_name)
            )

        print(f"[OK] Seeded {len(sdg_data)} SDG records")
    except Exception as e:
        print(f"[ERROR] SDG seeding failed: {e}")
        conn.close()
        return False

    # Verify
    print("\n" + "=" * 60)
    print("VERIFICATION")
    print("=" * 60)

    try:
        result = conn.execute("SELECT table_name FROM information_schema.tables WHERE table_schema = 'main' ORDER BY table_name").fetchall()
        tables = [r[0] for r in result]
        print(f"[OK] Total tables created: {len(tables)}")
        print("Tables:", ", ".join(tables))

        sdg_count = conn.execute("SELECT COUNT(*) FROM sdg").fetchone()[0]
        print(f"[OK] SDG records: {sdg_count}/17")
    except Exception as e:
        print(f"[WARN] Verification error: {e}")

    conn.close()

    print("\n" + "=" * 60)
    print("[SUCCESS] MOTHERDUCK REINITIALIZATION COMPLETE!")
    print("=" * 60)

    return True


if __name__ == "__main__":
    try:
        success = reinit_motherduck()
        sys.exit(0 if success else 1)
    except Exception as e:
        print(f"\n[ERROR] Fatal error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

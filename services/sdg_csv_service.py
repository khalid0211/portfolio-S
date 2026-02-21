"""SDG CSV Import/Export Service

This module provides functionality to export and import SDG weights in CSV format.
Export format: portfolio_id, project_id, sdg1, sdg2, sdg3, ..., sdg17
Import format: Same as export
"""

import pandas as pd
from typing import Optional, Dict, List, Tuple
from database.db_connection import get_db
import logging
from datetime import datetime

logger = logging.getLogger(__name__)


def export_sdg_to_csv(
    output_path: str,
    portfolio_id: Optional[int] = None
) -> Tuple[bool, str, int]:
    """Export SDG weights to CSV in wide format

    Args:
        output_path: Path to save the CSV file
        portfolio_id: Optional portfolio ID to filter by. If None, exports all portfolios.

    Returns:
        Tuple of (success: bool, message: str, row_count: int)

    CSV Format:
        portfolio_id, project_id, project_name, organization, project_manager, sdg1, sdg2, sdg3, ..., sdg17
        Each sdgN column contains the weight_percent for that SDG (or NULL if not assigned)
        Note: project_name, organization, project_manager are for reference only and ignored on import
    """
    try:
        db = get_db()

        # Build query - include project details for reference
        if portfolio_id:
            query = """
                SELECT
                    ps.portfolio_id,
                    ps.project_id,
                    ps.sdg_id,
                    ps.weight_percent
                FROM project_sdg ps
                WHERE ps.portfolio_id = ?
                ORDER BY ps.portfolio_id, ps.project_id, ps.sdg_id
            """
            params = (portfolio_id,)
        else:
            query = """
                SELECT
                    ps.portfolio_id,
                    ps.project_id,
                    ps.sdg_id,
                    ps.weight_percent
                FROM project_sdg ps
                ORDER BY ps.portfolio_id, ps.project_id, ps.sdg_id
            """
            params = ()

        # Fetch data
        result = db.execute(query, params)
        rows = result.fetchall()

        if not rows:
            logger.warning("No SDG data found to export")
            return False, "No SDG data found to export", 0

        # Convert to DataFrame
        df = pd.DataFrame(rows, columns=['portfolio_id', 'project_id', 'sdg_id', 'weight_percent'])

        # Pivot to wide format: one row per project, columns for each SDG
        pivot_df = df.pivot_table(
            index=['portfolio_id', 'project_id'],
            columns='sdg_id',
            values='weight_percent',
            aggfunc='first'
        ).reset_index()

        # Rename columns to sdg1, sdg2, ..., sdg17
        column_mapping = {i: f'sdg{i}' for i in range(1, 18)}
        pivot_df.rename(columns=column_mapping, inplace=True)

        # Ensure all SDG columns exist (1-17)
        for i in range(1, 18):
            col_name = f'sdg{i}'
            if col_name not in pivot_df.columns:
                pivot_df[col_name] = None

        # Get project details (name, organization, project_manager) for reference
        project_ids = pivot_df['project_id'].tolist()
        if project_ids:
            # Build query to get project details
            placeholders = ','.join(['?' for _ in project_ids])
            project_query = f"""
                SELECT project_id, project_name, responsible_organization, project_manager
                FROM project
                WHERE project_id IN ({placeholders})
            """
            project_result = db.execute(project_query, tuple(project_ids))
            project_rows = project_result.fetchall()
            project_details_df = pd.DataFrame(
                project_rows,
                columns=['project_id', 'project_name', 'organization', 'project_manager']
            )

            # Merge project details with pivot_df
            pivot_df = pivot_df.merge(project_details_df, on='project_id', how='left')

        # Reorder columns: portfolio_id, project_id, project_name, organization, project_manager, sdg1, sdg2, ..., sdg17
        column_order = ['portfolio_id', 'project_id', 'project_name', 'organization', 'project_manager'] + [f'sdg{i}' for i in range(1, 18)]
        # Only include columns that exist
        column_order = [col for col in column_order if col in pivot_df.columns]
        pivot_df = pivot_df[column_order]

        # Export to CSV
        pivot_df.to_csv(output_path, index=False)

        row_count = len(pivot_df)
        logger.info(f"Exported {row_count} projects to {output_path}")

        return True, f"Successfully exported {row_count} projects", row_count

    except Exception as e:
        logger.error(f"Export failed: {e}")
        return False, f"Export failed: {str(e)}", 0


def import_sdg_from_csv(
    input_path: str,
    portfolio_id: Optional[int] = None,
    validate_totals: bool = True,
    replace_existing: bool = True
) -> Tuple[bool, str, Dict[str, int]]:
    """Import SDG weights from CSV in wide format

    Args:
        input_path: Path to the CSV file
        portfolio_id: Optional portfolio ID to filter/validate. If provided, only imports this portfolio.
        validate_totals: If True, validates that weights sum to 100 for each project
        replace_existing: If True, deletes existing SDG weights before importing

    Returns:
        Tuple of (success: bool, message: str, stats: dict)
        stats dict contains: {'imported': int, 'updated': int, 'errors': int, 'skipped': int}

    CSV Format:
        portfolio_id, project_id, sdg1, sdg2, sdg3, ..., sdg17
        Each sdgN column contains the weight_percent for that SDG (or NULL/empty if not assigned)
    """
    stats = {
        'imported': 0,
        'updated': 0,
        'errors': 0,
        'skipped': 0
    }

    try:
        db = get_db()

        # Read CSV
        df = pd.read_csv(input_path)

        # Validate required columns
        required_cols = ['portfolio_id', 'project_id']
        sdg_cols = [f'sdg{i}' for i in range(1, 18)]

        if not all(col in df.columns for col in required_cols):
            return False, f"Missing required columns: {required_cols}", stats

        # Filter by portfolio if specified
        if portfolio_id:
            df = df[df['portfolio_id'] == portfolio_id]
            if df.empty:
                return False, f"No data found for portfolio_id {portfolio_id}", stats

        # Process each project
        for idx, row in df.iterrows():
            try:
                proj_portfolio_id = int(row['portfolio_id'])
                proj_project_id = int(row['project_id'])

                # Verify project exists
                project_check = db.fetch_one(
                    "SELECT 1 FROM project WHERE project_id = ? AND portfolio_id = ?",
                    (proj_project_id, proj_portfolio_id)
                )

                if not project_check:
                    logger.warning(
                        f"Project {proj_project_id} not found in portfolio {proj_portfolio_id}, skipping"
                    )
                    stats['skipped'] += 1
                    continue

                # Extract SDG weights
                sdg_weights = {}
                for i in range(1, 18):
                    col_name = f'sdg{i}'
                    if col_name in row and pd.notna(row[col_name]):
                        weight = float(row[col_name])
                        if weight > 0:  # Only include positive weights
                            sdg_weights[i] = weight

                # Skip if no weights
                if not sdg_weights:
                    stats['skipped'] += 1
                    continue

                # Validate totals if required
                if validate_totals:
                    total = sum(sdg_weights.values())
                    if abs(total - 100.0) > 0.01:  # Allow small floating point errors
                        logger.warning(
                            f"Project {proj_project_id}: SDG weights sum to {total}, expected 100. "
                            f"{'Importing anyway' if not validate_totals else 'Skipping'}"
                        )
                        if validate_totals:
                            stats['errors'] += 1
                            continue

                # Delete existing weights if replace_existing
                if replace_existing:
                    db.execute(
                        "DELETE FROM project_sdg WHERE project_id = ?",
                        (proj_project_id,)
                    )

                # Insert new weights
                for sdg_id, weight in sdg_weights.items():
                    db.execute("""
                        INSERT INTO project_sdg
                        (portfolio_id, project_id, sdg_id, weight_percent, is_active, created_at)
                        VALUES (?, ?, ?, ?, TRUE, ?)
                    """, (proj_portfolio_id, proj_project_id, sdg_id, weight, datetime.now()))

                if replace_existing:
                    stats['updated'] += 1
                else:
                    stats['imported'] += 1

            except Exception as e:
                logger.error(f"Error processing row {idx}: {e}")
                stats['errors'] += 1
                continue

        # Summary
        total_processed = stats['imported'] + stats['updated']
        message = (
            f"Import complete: {total_processed} projects processed "
            f"({stats['imported']} new, {stats['updated']} updated, "
            f"{stats['skipped']} skipped, {stats['errors']} errors)"
        )

        success = stats['errors'] == 0 or total_processed > 0

        logger.info(message)
        return success, message, stats

    except Exception as e:
        logger.error(f"Import failed: {e}")
        return False, f"Import failed: {str(e)}", stats


def get_sdg_template_csv(output_path: str, portfolio_id: Optional[int] = None) -> Tuple[bool, str]:
    """Generate a template CSV file for SDG import

    Args:
        output_path: Path to save the template CSV
        portfolio_id: Optional portfolio ID to include projects from

    Returns:
        Tuple of (success: bool, message: str)
    """
    try:
        db = get_db()

        # Get projects with details for reference
        if portfolio_id:
            query = """
                SELECT portfolio_id, project_id, project_name, responsible_organization, project_manager
                FROM project
                WHERE portfolio_id = ? AND is_active = TRUE
                ORDER BY project_id
            """
            params = (portfolio_id,)
        else:
            query = """
                SELECT portfolio_id, project_id, project_name, responsible_organization, project_manager
                FROM project
                WHERE is_active = TRUE
                ORDER BY portfolio_id, project_id
            """
            params = ()

        result = db.execute(query, params)
        projects = result.fetchall()

        if not projects:
            return False, "No active projects found"

        # Create DataFrame with project details
        df = pd.DataFrame(projects, columns=['portfolio_id', 'project_id', 'project_name', 'organization', 'project_manager'])

        # Add SDG columns (all empty)
        for i in range(1, 18):
            df[f'sdg{i}'] = None

        # Export to CSV
        df.to_csv(output_path, index=False)

        message = f"Template created with {len(df)} projects at {output_path}"
        logger.info(message)

        return True, message

    except Exception as e:
        logger.error(f"Template generation failed: {e}")
        return False, f"Template generation failed: {str(e)}"


def validate_sdg_csv(input_path: str) -> Tuple[bool, List[str]]:
    """Validate SDG CSV file format and data

    Args:
        input_path: Path to the CSV file

    Returns:
        Tuple of (is_valid: bool, errors: List[str])
    """
    errors = []

    try:
        # Read CSV
        df = pd.read_csv(input_path)

        # Check required columns
        required_cols = ['portfolio_id', 'project_id']
        sdg_cols = [f'sdg{i}' for i in range(1, 18)]

        missing_cols = [col for col in required_cols if col not in df.columns]
        if missing_cols:
            errors.append(f"Missing required columns: {', '.join(missing_cols)}")

        missing_sdg_cols = [col for col in sdg_cols if col not in df.columns]
        if missing_sdg_cols:
            errors.append(f"Missing SDG columns: {', '.join(missing_sdg_cols)}")

        if errors:
            return False, errors

        # Validate data
        for idx, row in df.iterrows():
            row_num = idx + 2  # +2 for header and 0-indexing

            # Check portfolio_id and project_id are valid integers
            if pd.isna(row['portfolio_id']) or not str(row['portfolio_id']).replace('.', '', 1).isdigit():
                errors.append(f"Row {row_num}: Invalid portfolio_id")

            if pd.isna(row['project_id']) or not str(row['project_id']).replace('.', '', 1).isdigit():
                errors.append(f"Row {row_num}: Invalid project_id")

            # Validate SDG weights
            weights = []
            for i in range(1, 18):
                col_name = f'sdg{i}'
                if col_name in row and pd.notna(row[col_name]):
                    try:
                        weight = float(row[col_name])
                        if weight < 0 or weight > 100:
                            errors.append(f"Row {row_num}, {col_name}: Weight must be between 0 and 100")
                        elif weight > 0:
                            weights.append(weight)
                    except (ValueError, TypeError):
                        errors.append(f"Row {row_num}, {col_name}: Invalid numeric value")

            # Check total weights
            if weights:
                total = sum(weights)
                if abs(total - 100.0) > 0.01:
                    errors.append(f"Row {row_num}: SDG weights sum to {total:.2f}, expected 100")

        is_valid = len(errors) == 0
        return is_valid, errors

    except Exception as e:
        errors.append(f"Failed to read CSV: {str(e)}")
        return False, errors


if __name__ == "__main__":
    """Example usage and testing"""
    import sys

    logging.basicConfig(level=logging.INFO)

    if len(sys.argv) < 2:
        print("Usage:")
        print("  Export: python sdg_csv_service.py export <output.csv> [portfolio_id]")
        print("  Import: python sdg_csv_service.py import <input.csv> [portfolio_id]")
        print("  Template: python sdg_csv_service.py template <output.csv> [portfolio_id]")
        print("  Validate: python sdg_csv_service.py validate <input.csv>")
        sys.exit(1)

    command = sys.argv[1].lower()

    if command == 'export':
        output_path = sys.argv[2]
        portfolio_id = int(sys.argv[3]) if len(sys.argv) > 3 else None
        success, message, count = export_sdg_to_csv(output_path, portfolio_id)
        print(message)
        sys.exit(0 if success else 1)

    elif command == 'import':
        input_path = sys.argv[2]
        portfolio_id = int(sys.argv[3]) if len(sys.argv) > 3 else None
        success, message, stats = import_sdg_from_csv(input_path, portfolio_id)
        print(message)
        print(f"Stats: {stats}")
        sys.exit(0 if success else 1)

    elif command == 'template':
        output_path = sys.argv[2]
        portfolio_id = int(sys.argv[3]) if len(sys.argv) > 3 else None
        success, message = get_sdg_template_csv(output_path, portfolio_id)
        print(message)
        sys.exit(0 if success else 1)

    elif command == 'validate':
        input_path = sys.argv[2]
        is_valid, errors = validate_sdg_csv(input_path)
        if is_valid:
            print("✓ CSV file is valid")
            sys.exit(0)
        else:
            print("✗ CSV file has errors:")
            for error in errors:
                print(f"  - {error}")
            sys.exit(1)

    else:
        print(f"Unknown command: {command}")
        sys.exit(1)

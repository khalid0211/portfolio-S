# SDG Import/Export Guide

## Overview

The SDG Import/Export feature allows you to manage Sustainable Development Goals (SDG) weights for projects using CSV files. This is useful for:
- Bulk updating SDG weights across multiple projects
- Backing up SDG data
- Sharing SDG configurations between portfolios
- Editing SDG weights in Excel or other spreadsheet software

## CSV Format

### Structure

The CSV file uses a "wide" format with one row per project:

```csv
portfolio_id,project_id,sdg1,sdg2,sdg3,sdg4,sdg5,sdg6,sdg7,sdg8,sdg9,sdg10,sdg11,sdg12,sdg13,sdg14,sdg15,sdg16,sdg17
1001,5001,20.0,30.0,50.0,,,,,,,,,,,,,
1001,5002,,,40.0,60.0,,,,,,,,,,,,,
1002,5003,100.0,,,,,,,,,,,,,,,,
```

### Column Descriptions

| Column | Type | Description |
|--------|------|-------------|
| portfolio_id | Integer | Portfolio ID (must match existing portfolio) |
| project_id | Integer | Project ID (must match existing project) |
| sdg1 to sdg17 | Decimal | Weight percentage for each SDG (0-100) |

### Rules

1. **Portfolio & Project IDs**: Must reference existing records in the database
2. **SDG Weights**:
   - Must be between 0 and 100
   - Should sum to 100 for each project (validation can be disabled)
   - Empty cells or NULL values mean that SDG is not assigned
3. **Decimal Precision**: Up to 2 decimal places (e.g., 33.33)

## Using the Web Interface

### 1. Export SDG Weights

1. Navigate to **SDG Import/Export** page
2. Go to the **Export** tab
3. Choose scope:
   - **Current portfolio only**: Export SDG weights for current portfolio
   - **All portfolios**: Export SDG weights for all portfolios
4. Click **Export to CSV**
5. Download the generated CSV file

### 2. Import SDG Weights

1. Navigate to **SDG Import/Export** page
2. Go to the **Import** tab
3. Upload your CSV file
4. Review the preview and validation results
5. Choose import options:
   - **Import scope**: Current portfolio or all portfolios in file
   - **Validate totals**: Whether to enforce that weights sum to 100
6. Click **Import from CSV**
7. Review the import statistics

**⚠️ Warning**: Importing will replace existing SDG weights for projects in the CSV file.

### 3. Download Template

1. Navigate to **SDG Import/Export** page
2. Go to the **Template** tab
3. Choose template scope (current portfolio or all)
4. Click **Generate Template**
5. Download the template CSV
6. Fill in the SDG weights in Excel or similar
7. Use the Import tab to upload your filled template

## Using the Command Line

The SDG service can also be used from the command line:

### Export

```bash
# Export current portfolio
python services/sdg_csv_service.py export output.csv 1001

# Export all portfolios
python services/sdg_csv_service.py export output.csv
```

### Import

```bash
# Import for specific portfolio
python services/sdg_csv_service.py import input.csv 1001

# Import all portfolios in file
python services/sdg_csv_service.py import input.csv
```

### Generate Template

```bash
# Template for specific portfolio
python services/sdg_csv_service.py template template.csv 1001

# Template for all portfolios
python services/sdg_csv_service.py template template.csv
```

### Validate

```bash
# Validate CSV file before importing
python services/sdg_csv_service.py validate input.csv
```

## Using the Python API

```python
from services.sdg_csv_service import (
    export_sdg_to_csv,
    import_sdg_from_csv,
    get_sdg_template_csv,
    validate_sdg_csv
)

# Export SDG weights
success, message, count = export_sdg_to_csv(
    output_path='sdg_export.csv',
    portfolio_id=1001  # Optional: None for all portfolios
)

# Import SDG weights
success, message, stats = import_sdg_from_csv(
    input_path='sdg_import.csv',
    portfolio_id=1001,  # Optional: None for all portfolios
    validate_totals=True,  # Enforce sum to 100
    replace_existing=True  # Replace existing weights
)

# Generate template
success, message = get_sdg_template_csv(
    output_path='template.csv',
    portfolio_id=1001  # Optional: None for all portfolios
)

# Validate CSV
is_valid, errors = validate_sdg_csv('input.csv')
if not is_valid:
    for error in errors:
        print(f"Error: {error}")
```

## Common Scenarios

### Scenario 1: Bulk Update SDG Weights

1. Export current SDG weights to CSV
2. Open in Excel and edit the weights
3. Save the file
4. Validate the file using the web interface or CLI
5. Import the updated file

### Scenario 2: Copy SDG Weights to Another Portfolio

1. Export SDG weights from source portfolio
2. Open the CSV and update the `portfolio_id` column to target portfolio
3. Update `project_id` values to match projects in target portfolio
4. Import into target portfolio

### Scenario 3: Backup SDG Configuration

1. Export all portfolios to CSV
2. Save the file with a timestamp (e.g., `sdg_backup_20231201.csv`)
3. Store in version control or backup location

### Scenario 4: Bulk Setup New Projects

1. Download template for current portfolio
2. Fill in SDG weights for all projects in Excel
3. Import the completed template

## Validation Rules

The validator checks:

1. ✓ Required columns exist (`portfolio_id`, `project_id`, `sdg1`-`sdg17`)
2. ✓ Portfolio IDs and Project IDs are valid integers
3. ✓ SDG weights are numeric values between 0 and 100
4. ✓ SDG weights sum to 100 for each project (if validation enabled)
5. ✓ Projects exist in the database (checked during import)

## Error Handling

### Common Errors

| Error | Cause | Solution |
|-------|-------|----------|
| "Missing required columns" | CSV doesn't have all required columns | Check column names match exactly |
| "Invalid portfolio_id" | Portfolio ID is not a number or doesn't exist | Verify portfolio exists in database |
| "SDG weights sum to X, expected 100" | Weights don't add up to 100 | Adjust weights or disable validation |
| "Project X not found" | Project doesn't exist in database | Create project first or remove from CSV |

### Import Statistics

After import, you'll see:
- **Imported**: New SDG weight assignments created
- **Updated**: Existing SDG weights replaced
- **Skipped**: Projects with no weights or missing from database
- **Errors**: Rows that failed to import

## Best Practices

1. **Always validate** before importing
2. **Backup first** - export current state before bulk import
3. **Start small** - test with a few projects before bulk operations
4. **Use templates** - ensure correct format and all columns present
5. **Check totals** - verify weights sum to 100 unless intentionally partial
6. **Version control** - keep copies of CSV files with timestamps

## UN Sustainable Development Goals Reference

| ID | Goal |
|----|------|
| 1 | No Poverty |
| 2 | Zero Hunger |
| 3 | Good Health and Well-being |
| 4 | Quality Education |
| 5 | Gender Equality |
| 6 | Clean Water and Sanitation |
| 7 | Affordable and Clean Energy |
| 8 | Decent Work and Economic Growth |
| 9 | Industry, Innovation and Infrastructure |
| 10 | Reduced Inequalities |
| 11 | Sustainable Cities and Communities |
| 12 | Responsible Consumption and Production |
| 13 | Climate Action |
| 14 | Life Below Water |
| 15 | Life on Land |
| 16 | Peace, Justice and Strong Institutions |
| 17 | Partnerships for the Goals |

## Troubleshooting

### CSV File Won't Import

1. Open in a text editor to check encoding (should be UTF-8)
2. Verify no extra blank rows at the end
3. Check for special characters in numeric fields
4. Ensure commas are properly escaped if used in text fields

### Weights Don't Sum to 100

This is usually due to:
- Rounding errors in Excel
- Missing SDG assignments
- Typos in weight values

**Solution**: Use Excel's SUM function to verify: `=SUM(C2:S2)` should equal 100

### Import Shows All Skipped

This means:
- Projects don't exist in the database
- Portfolio IDs don't match
- All weights are zero or blank

**Solution**: Download a template to get the correct project IDs

## Support

For issues or questions:
1. Check the validation errors for specific problems
2. Review this guide for common scenarios
3. Contact system administrator for database access issues

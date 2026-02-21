"""Date parsing and formatting utilities."""

from __future__ import annotations
import logging
from datetime import datetime, date, timedelta
import pandas as pd
from dateutil import parser as date_parser


logger = logging.getLogger(__name__)

EXCEL_ORDINAL_BASE = datetime(1899, 12, 30)


def parse_date_any(x):
    """Parse date from various formats with improved error handling."""
    if pd.isna(x) or x is None:
        return None

    if isinstance(x, datetime):
        return x
    elif isinstance(x, date):
        return datetime.combine(x, datetime.min.time())

    # Handle string inputs
    if isinstance(x, str):
        x = x.strip()
        if not x:
            return None

        # Common format shortcuts
        try:
            # Try ISO format first (fastest)
            if len(x) == 10 and x.count('-') == 2:
                return datetime.fromisoformat(x)

            # Try dateutil parser (handles most formats)
            return date_parser.parse(x, fuzzy=False)
        except:
            pass

    # Handle numeric inputs (Excel ordinal dates)
    if isinstance(x, (int, float)):
        try:
            if 1 <= x <= 2958465:  # Valid Excel date range
                return EXCEL_ORDINAL_BASE + timedelta(days=x)
        except:
            pass

    logger.warning(f"Could not parse date: {x}")
    return None


def format_date_safe(date_obj: datetime, format_str: str = '%Y-%m-%d') -> str:
    """Safely format datetime object to string."""
    try:
        if isinstance(date_obj, datetime):
            return date_obj.strftime(format_str)
        elif isinstance(date_obj, str):
            parsed = parse_date_any(date_obj)
            if parsed:
                return parsed.strftime(format_str)
        return str(date_obj) if date_obj else "N/A"
    except Exception:
        return "N/A"
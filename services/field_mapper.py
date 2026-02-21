"""Field Mapper - Backward Compatibility for Schema Migration

Handles mapping between old and new field names during the schema refactoring transition.
Ensures zero application breakage by supporting both naming conventions.
"""

from typing import Dict, Any, List, Optional
import pandas as pd


class FieldMapper:
    """Handles backward compatibility for renamed database fields"""

    # Field mapping: old_name -> new_name
    FIELD_MAPPING = {
        # Status Report Fields
        'actual_cost': 'ac',
        'planned_value': 'pv',
        'earned_value': 'ev',
        'likely_completion': 'likely_completion_date',

        # Baseline snapshot fields (being added, not renamed)
        # These don't have old equivalents but included for completeness
    }

    # Reverse mapping: new_name -> old_name
    REVERSE_MAPPING = {v: k for k, v in FIELD_MAPPING.items()}

    # New fields being added (no old equivalent)
    NEW_FIELDS = {
        'bac', 'plan_start_date', 'plan_finish_date', 'baseline_version'
    }

    @classmethod
    def map_to_new(cls, data: Dict[str, Any]) -> Dict[str, Any]:
        """Map old field names to new names

        Args:
            data: Dictionary with field names (can be old or new)

        Returns:
            Dictionary with new field names

        Example:
            >>> data = {'actual_cost': 1000, 'planned_value': 1200}
            >>> FieldMapper.map_to_new(data)
            {'ac': 1000, 'pv': 1200}
        """
        result = {}
        for key, value in data.items():
            new_key = cls.FIELD_MAPPING.get(key, key)
            result[new_key] = value
        return result

    @classmethod
    def map_to_old(cls, data: Dict[str, Any]) -> Dict[str, Any]:
        """Map new field names to old names (for backward compatibility)

        Args:
            data: Dictionary with field names (can be old or new)

        Returns:
            Dictionary with old field names

        Example:
            >>> data = {'ac': 1000, 'pv': 1200}
            >>> FieldMapper.map_to_old(data)
            {'actual_cost': 1000, 'planned_value': 1200}
        """
        result = {}
        for key, value in data.items():
            old_key = cls.REVERSE_MAPPING.get(key, key)
            result[old_key] = value
        return result

    @classmethod
    def normalize_dict(cls, data: Dict[str, Any], prefer_new: bool = True) -> Dict[str, Any]:
        """Normalize dictionary to have both old and new field names

        Args:
            data: Input dictionary
            prefer_new: If True, prioritize new field names when both exist

        Returns:
            Dictionary with both old and new field names
        """
        result = data.copy()

        # Add new names if only old names exist
        for old_name, new_name in cls.FIELD_MAPPING.items():
            if old_name in data and new_name not in data:
                result[new_name] = data[old_name]
            elif new_name in data and old_name not in data:
                result[old_name] = data[new_name]

        return result

    @classmethod
    def get_value(cls, data: Dict[str, Any], field_name: str, default: Any = None) -> Any:
        """Get value from dict using either old or new field name

        Args:
            data: Dictionary to search
            field_name: Field name (can be old or new)
            default: Default value if not found

        Returns:
            Field value or default

        Example:
            >>> data = {'ac': 1000}
            >>> FieldMapper.get_value(data, 'actual_cost')  # Try old name
            1000
            >>> FieldMapper.get_value(data, 'ac')  # Try new name
            1000
        """
        # Try new name first
        new_name = cls.FIELD_MAPPING.get(field_name, field_name)
        if new_name in data:
            return data[new_name]

        # Try old name
        old_name = cls.REVERSE_MAPPING.get(field_name, field_name)
        if old_name in data:
            return data[old_name]

        # Try original name
        if field_name in data:
            return data[field_name]

        return default

    @classmethod
    def normalize_dataframe(cls, df: pd.DataFrame, prefer_new: bool = True) -> pd.DataFrame:
        """Normalize DataFrame column names to include both old and new

        Args:
            df: Input DataFrame
            prefer_new: If True, prioritize new column names

        Returns:
            DataFrame with both old and new column names (aliased)

        Example:
            >>> df = pd.DataFrame({'ac': [1000], 'pv': [1200]})
            >>> df_norm = FieldMapper.normalize_dataframe(df)
            >>> 'actual_cost' in df_norm.columns
            True
            >>> 'ac' in df_norm.columns
            True
        """
        df_copy = df.copy()

        # Add new column names as aliases for old columns
        for old_name, new_name in cls.FIELD_MAPPING.items():
            if old_name in df_copy.columns and new_name not in df_copy.columns:
                df_copy[new_name] = df_copy[old_name]
            elif new_name in df_copy.columns and old_name not in df_copy.columns:
                df_copy[old_name] = df_copy[new_name]

        return df_copy

    @classmethod
    def create_select_aliases(cls) -> str:
        """Create SQL SELECT clause with aliases for backward compatibility

        Returns:
            SQL fragment with aliased columns

        Example:
            SELECT ac as actual_cost, pv as planned_value, ...
        """
        aliases = []
        for old_name, new_name in cls.FIELD_MAPPING.items():
            aliases.append(f"{new_name} as {old_name}")
            aliases.append(new_name)
        return ", ".join(aliases)

    @classmethod
    def create_insert_columns(cls, include_new_fields: bool = True) -> str:
        """Create column list for INSERT statement (dual-write)

        Args:
            include_new_fields: Include new baseline fields

        Returns:
            SQL column list

        Example:
            actual_cost, planned_value, earned_value, ac, pv, ev, bac, ...
        """
        columns = []

        # Old field names
        columns.extend(cls.FIELD_MAPPING.keys())

        # New field names
        columns.extend(cls.FIELD_MAPPING.values())

        # New baseline fields
        if include_new_fields:
            columns.extend(cls.NEW_FIELDS)

        return ", ".join(columns)

    @classmethod
    def prepare_dual_write_values(cls, data: Dict[str, Any]) -> List[Any]:
        """Prepare values for dual-write INSERT/UPDATE

        Args:
            data: Input data dictionary

        Returns:
            List of values in correct order for dual-write

        Example:
            Input: {'ac': 1000, 'pv': 1200, 'ev': 1100, 'bac': 5000000}
            Output: [1000, 1200, 1100, 1000, 1200, 1100, 5000000, ...]
        """
        values = []

        # Normalize data to have both old and new names
        normalized = cls.normalize_dict(data)

        # Old field values
        for old_name in cls.FIELD_MAPPING.keys():
            values.append(normalized.get(old_name))

        # New field values (same as old)
        for new_name in cls.FIELD_MAPPING.values():
            values.append(normalized.get(new_name))

        # New baseline field values
        for new_field in cls.NEW_FIELDS:
            values.append(normalized.get(new_field))

        return values

    @classmethod
    def validate_field_name(cls, field_name: str) -> Optional[str]:
        """Validate and normalize field name

        Args:
            field_name: Field name to validate

        Returns:
            Normalized field name (new name), or None if invalid
        """
        # If it's an old name, return the new name
        if field_name in cls.FIELD_MAPPING:
            return cls.FIELD_MAPPING[field_name]

        # If it's already a new name, return it
        if field_name in cls.REVERSE_MAPPING:
            return field_name

        # If it's a new baseline field, return it
        if field_name in cls.NEW_FIELDS:
            return field_name

        # Unknown field
        return None

    @classmethod
    def is_renamed_field(cls, field_name: str) -> bool:
        """Check if field has been renamed

        Args:
            field_name: Field name to check

        Returns:
            True if field was renamed
        """
        return field_name in cls.FIELD_MAPPING or field_name in cls.REVERSE_MAPPING

    @classmethod
    def get_field_info(cls) -> Dict[str, str]:
        """Get mapping information for documentation

        Returns:
            Dictionary with field mapping info
        """
        return {
            'mapping': dict(cls.FIELD_MAPPING),
            'reverse_mapping': dict(cls.REVERSE_MAPPING),
            'new_fields': list(cls.NEW_FIELDS)
        }


# Convenience functions for common use cases

def safe_get(data: Dict[str, Any], field: str, default: Any = None) -> Any:
    """Safely get field value using either old or new name

    Args:
        data: Data dictionary
        field: Field name (old or new)
        default: Default value

    Returns:
        Field value or default
    """
    return FieldMapper.get_value(data, field, default)


def normalize_row(row: Dict[str, Any]) -> Dict[str, Any]:
    """Normalize a data row to have both old and new field names

    Args:
        row: Data row

    Returns:
        Normalized row
    """
    return FieldMapper.normalize_dict(row)


def normalize_df(df: pd.DataFrame) -> pd.DataFrame:
    """Normalize DataFrame to have both old and new column names

    Args:
        df: DataFrame

    Returns:
        Normalized DataFrame
    """
    return FieldMapper.normalize_dataframe(df)

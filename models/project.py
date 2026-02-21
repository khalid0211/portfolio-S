"""Project data models and validation."""

from __future__ import annotations
import re
from dataclasses import dataclass
from datetime import datetime
from typing import Optional, Dict, Any, List
from pathlib import Path


# Validation patterns
VALID_TABLE_NAME_PATTERN = re.compile(r'^[a-zA-Z0-9_-]+$')
VALID_COLUMN_NAME_PATTERN = re.compile(r'^[a-zA-Z0-9_\s-]+$')


@dataclass
class ProjectData:
    """Project data structure for EVM analysis."""
    project_id: str
    project_name: str
    organization: str
    project_manager: str
    bac: float
    ac: float
    plan_start: datetime
    plan_finish: datetime
    data_date: datetime
    manual_ev: Optional[float] = None
    manual_pv: Optional[float] = None
    use_manual_ev: bool = False
    use_manual_pv: bool = False
    # Per-project EVM settings (optional - will use global if None)
    curve_type: Optional[str] = None
    alpha: Optional[float] = None
    beta: Optional[float] = None
    inflation_rate: Optional[float] = None

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for storage."""
        return {
            'project_id': self.project_id,
            'project_name': self.project_name,
            'organization': self.organization,
            'project_manager': self.project_manager,
            'bac': self.bac,
            'ac': self.ac,
            'plan_start': self.plan_start.isoformat() if isinstance(self.plan_start, datetime) else self.plan_start,
            'plan_finish': self.plan_finish.isoformat() if isinstance(self.plan_finish, datetime) else self.plan_finish,
            'data_date': self.data_date.isoformat() if isinstance(self.data_date, datetime) else self.data_date,
            'manual_ev': self.manual_ev,
            'manual_pv': self.manual_pv,
            'use_manual_ev': self.use_manual_ev,
            'use_manual_pv': self.use_manual_pv,
            'curve_type': self.curve_type,
            'alpha': self.alpha,
            'beta': self.beta,
            'inflation_rate': self.inflation_rate
        }


@dataclass
class AnalysisConfig:
    """Configuration for EVM analysis."""
    curve_type: str = 'linear'
    alpha: float = 2.0
    beta: float = 2.0
    currency_symbol: str = '$'
    currency_postfix: str = ''
    date_format: str = 'YYYY-MM-DD'
    annual_inflation_rate: float = 0.03


class ProjectValidator:
    """Validation utilities for project data."""

    @staticmethod
    def validate_table_name(table_name: str) -> bool:
        """Validate table name against security pattern."""
        return bool(VALID_TABLE_NAME_PATTERN.match(table_name))

    @staticmethod
    def validate_column_name(column_name: str) -> bool:
        """Validate column name against security pattern."""
        return bool(VALID_COLUMN_NAME_PATTERN.match(column_name))

    @staticmethod
    def sanitize_sql_identifier(identifier: str) -> str:
        """Sanitize SQL identifier by removing invalid characters."""
        return re.sub(r'[^a-zA-Z0-9_-]', '_', identifier)

    @staticmethod
    def validate_numeric_input(value: Any, field_name: str, min_val: float = None, max_val: float = None) -> float:
        """Validate numeric input with optional range checking."""
        try:
            num_value = float(value)
            if min_val is not None and num_value < min_val:
                raise ValueError(f"{field_name} must be >= {min_val}")
            if max_val is not None and num_value > max_val:
                raise ValueError(f"{field_name} must be <= {max_val}")
            return num_value
        except (ValueError, TypeError) as e:
            raise ValueError(f"Invalid {field_name}: {value}") from e

    @staticmethod
    def is_valid_finite_number(value: Any) -> bool:
        """Check if value is a valid finite number."""
        try:
            num_val = float(value)
            return not (float('inf') == abs(num_val) or num_val != num_val)  # Check for inf and NaN
        except (ValueError, TypeError):
            return False


class ColumnMapping:
    """Standard column mappings for project data."""

    DEFAULT_MAPPING = {
        'pid_col': 'Project ID',
        'pname_col': 'Project',
        'org_col': 'Organization',
        'pm_col': 'Project Manager',
        'bac_col': 'BAC',
        'ac_col': 'AC',
        'st_col': 'Plan Start',
        'fn_col': 'Plan Finish',
        'dd_col': 'Data Date',
        'ev_col': 'EV',
        'pv_col': 'PV',
        # Optional EVM settings columns
        'curve_type_col': 'Curve Type',
        'alpha_col': 'Alpha',
        'beta_col': 'Beta',
        'inflation_rate_col': 'Inflation Rate'
    }

    @classmethod
    def get_required_columns(cls) -> List[str]:
        """Get list of required columns for analysis."""
        return [
            cls.DEFAULT_MAPPING['pid_col'],
            cls.DEFAULT_MAPPING['pname_col'],
            cls.DEFAULT_MAPPING['bac_col'],
            cls.DEFAULT_MAPPING['ac_col'],
            cls.DEFAULT_MAPPING['st_col'],
            cls.DEFAULT_MAPPING['fn_col'],
            cls.DEFAULT_MAPPING['dd_col']
        ]

    @classmethod
    def validate_mapping(cls, mapping: Dict[str, str], available_columns: List[str]) -> List[str]:
        """Validate column mapping and return missing required columns."""
        missing = []
        for key, expected_col in cls.DEFAULT_MAPPING.items():
            mapped_col = mapping.get(key, expected_col)
            if mapped_col not in available_columns:
                missing.append(f"{key}: {mapped_col}")
        return missing
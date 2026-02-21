"""
Portfolio Optimization Service

Budget-constrained project selection using integer linear programming.
Maximizes strategic value under budget constraints with multiple duration prioritization modes.
"""

import pandas as pd
import numpy as np
from dataclasses import dataclass, field
from typing import List, Optional, Dict, Any, Tuple
from enum import IntEnum
from scipy.optimize import milp, LinearConstraint, Bounds
import logging

logger = logging.getLogger(__name__)


class DurationMode(IntEnum):
    """Duration prioritization modes for portfolio optimization"""
    NONE = 0          # Maximize value only
    SOFT = 1          # Soft preference - favor shorter projects via Dmax/Di multiplier
    LEXICOGRAPHIC = 2 # Maximize value first, then minimize duration within epsilon
    HARD = 3          # Minimum % of budget must go to "short" projects


@dataclass
class OptimizationConfig:
    """Configuration for portfolio optimization"""
    budget_limit: float
    duration_mode: DurationMode = DurationMode.NONE
    epsilon_pct: float = 5.0               # Mode 2: % tolerance for value reduction
    short_threshold_months: float = 12.0   # Mode 3: threshold for "short" projects
    min_short_budget_pct: float = 30.0     # Mode 3: min % budget to short projects
    precommit_budget_pct_threshold: Optional[float] = None  # Pre-commit projects with >= X% budget used


@dataclass
class OptimizationResult:
    """Result of portfolio optimization"""
    status: str  # 'optimal', 'infeasible', 'no_projects', 'error'
    message: str = ""
    selected_project_ids: List[int] = field(default_factory=list)
    not_selected_project_ids: List[int] = field(default_factory=list)
    total_budget_used: float = 0.0
    budget_utilization_pct: float = 0.0
    total_strategic_value: float = 0.0
    avg_strategic_score: float = 0.0
    total_duration_months: float = 0.0
    avg_duration_months: float = 0.0
    project_count: int = 0
    selected_df: pd.DataFrame = field(default_factory=pd.DataFrame)
    not_selected_df: pd.DataFrame = field(default_factory=pd.DataFrame)
    warnings: List[str] = field(default_factory=list)
    # Mode-specific info
    optimal_value_stage1: Optional[float] = None  # For lexicographic mode
    # Portfolio value metrics
    total_portfolio_value: float = 0.0  # Sum of ALL eligible projects' base_value
    value_captured_pct: float = 0.0     # Percentage of total value captured
    # Pre-committed project tracking
    precommitted_project_ids: List[int] = field(default_factory=list)
    precommitted_budget: float = 0.0


class PortfolioOptimizer:
    """
    Optimizer for selecting projects to maximize strategic value within budget constraints.
    Uses scipy.optimize.milp with HiGHS solver for binary integer linear programming.
    """

    def __init__(self, config: OptimizationConfig):
        self.config = config

    def prepare_data(
        self,
        projects_df: pd.DataFrame,
        factors_df: pd.DataFrame,
        scores_lookup: Dict[Tuple[int, int], int]
    ) -> Tuple[pd.DataFrame, List[str]]:
        """
        Prepare project data for optimization.

        Args:
            projects_df: DataFrame with project_id, project_name, responsible_organization,
                        current_budget, original_duration_months
            factors_df: DataFrame with factor_id, factor_name, factor_weight_percent, likert_max
            scores_lookup: Dict mapping (project_id, factor_id) -> score

        Returns:
            Tuple of (prepared DataFrame, list of warnings)
        """
        warnings = []

        if projects_df.empty:
            return pd.DataFrame(), ["No projects available"]

        if factors_df.empty:
            return pd.DataFrame(), ["No factors defined for this portfolio"]

        # Start with a copy
        df = projects_df.copy()

        # Ensure required columns exist
        required_cols = ['project_id', 'project_name', 'current_budget']
        for col in required_cols:
            if col not in df.columns:
                return pd.DataFrame(), [f"Missing required column: {col}"]

        # Handle budget column - use current_budget as the budget for optimization
        df['budget'] = df['current_budget'].fillna(0)

        # Handle duration - prefer original_duration_months from status report
        if 'original_duration_months' in df.columns:
            df['duration_months'] = df['original_duration_months']
        elif 'duration_months' in df.columns:
            pass  # Already exists
        else:
            # Calculate from planned dates if available
            if 'planned_start_date' in df.columns and 'planned_finish_date' in df.columns:
                df['duration_months'] = (
                    (pd.to_datetime(df['planned_finish_date']) - pd.to_datetime(df['planned_start_date']))
                    .dt.days / 30.44
                ).round(2)
            else:
                df['duration_months'] = np.nan

        # Track excluded projects
        excluded_budget = df[df['budget'] <= 0]
        if len(excluded_budget) > 0:
            warnings.append(f"{len(excluded_budget)} project(s) excluded due to missing/zero budget")

        excluded_duration = df[(df['duration_months'].isna()) | (df['duration_months'] <= 0)]
        if len(excluded_duration) > 0 and self.config.duration_mode != DurationMode.NONE:
            warnings.append(f"{len(excluded_duration)} project(s) excluded due to missing duration data")

        # Filter to valid projects
        if self.config.duration_mode != DurationMode.NONE:
            df = df[
                (df['budget'] > 0) &
                (df['duration_months'].notna()) &
                (df['duration_months'] > 0)
            ].copy()
        else:
            df = df[df['budget'] > 0].copy()

        if df.empty:
            return df, warnings + ["No valid projects after filtering"]

        # Calculate base strategic value for each project
        total_weight = factors_df['factor_weight_percent'].sum()
        if abs(total_weight - 100) > 0.01:
            warnings.append(f"Factor weights sum to {total_weight:.1f}% (not 100%)")

        def calc_base_value(project_id):
            """Calculate normalized strategic value (0-1 scale)"""
            value = 0.0
            factors_scored = 0

            for _, factor in factors_df.iterrows():
                factor_id = factor['factor_id']
                key = (project_id, factor_id)

                if key in scores_lookup:
                    score = scores_lookup[key]
                    if score is not None:
                        likert_max = factor['likert_max']
                        weight_pct = factor['factor_weight_percent']
                        # Normalized contribution: (score/max) * (weight/100)
                        value += (score / likert_max) * (weight_pct / 100)
                        factors_scored += 1

            return value, factors_scored

        # Calculate base value for all projects
        base_values = []
        factors_scored_list = []
        excluded_no_scores = 0

        for project_id in df['project_id']:
            value, factors_scored = calc_base_value(project_id)
            if factors_scored == 0:
                excluded_no_scores += 1
            base_values.append(value)
            factors_scored_list.append(factors_scored)

        df['base_value'] = base_values
        df['factors_scored'] = factors_scored_list

        # Exclude projects with no scores
        if excluded_no_scores > 0:
            warnings.append(f"{excluded_no_scores} project(s) excluded due to no factor scores")
            df = df[df['factors_scored'] > 0].copy()

        if df.empty:
            return df, warnings + ["No projects with factor scores"]

        # Reset index
        df = df.reset_index(drop=True)

        # Calculate strategic score as percentage (for display)
        df['strategic_score_pct'] = (df['base_value'] * 100).round(2)

        return df, warnings

    def optimize(self, prepared_df: pd.DataFrame) -> OptimizationResult:
        """
        Run optimization based on configured mode.

        Args:
            prepared_df: DataFrame from prepare_data()

        Returns:
            OptimizationResult
        """
        if prepared_df.empty:
            return OptimizationResult(
                status='no_projects',
                message="No valid projects to optimize"
            )

        # Check if any single project fits in budget
        min_budget = prepared_df['budget'].min()
        if min_budget > self.config.budget_limit:
            return OptimizationResult(
                status='infeasible',
                message=f"Budget limit ({self.config.budget_limit:,.0f}) is less than smallest project budget ({min_budget:,.0f})"
            )

        # Run appropriate optimization
        if self.config.duration_mode == DurationMode.NONE:
            return self._optimize_base(prepared_df)
        elif self.config.duration_mode == DurationMode.SOFT:
            return self._optimize_soft_duration(prepared_df)
        elif self.config.duration_mode == DurationMode.LEXICOGRAPHIC:
            return self._optimize_lexicographic(prepared_df)
        elif self.config.duration_mode == DurationMode.HARD:
            return self._optimize_hard_duration(prepared_df)
        else:
            return OptimizationResult(
                status='error',
                message=f"Unknown duration mode: {self.config.duration_mode}"
            )

    def _solve_milp(
        self,
        c: np.ndarray,
        A_ub: Optional[np.ndarray] = None,
        b_ub: Optional[np.ndarray] = None,
        A_eq: Optional[np.ndarray] = None,
        b_eq: Optional[np.ndarray] = None,
        maximize: bool = True
    ) -> Tuple[Optional[np.ndarray], str]:
        """
        Solve a binary integer linear program using scipy.optimize.milp.

        Args:
            c: Objective coefficients (n,)
            A_ub: Inequality constraint matrix (m, n) for A_ub @ x <= b_ub
            b_ub: Inequality constraint bounds (m,)
            A_eq: Equality constraint matrix (p, n) for A_eq @ x == b_eq
            b_eq: Equality constraint bounds (p,)
            maximize: If True, maximize c @ x; else minimize

        Returns:
            Tuple of (solution array or None, status string)
        """
        n = len(c)

        # Convert to minimization (milp minimizes)
        if maximize:
            c_min = -c
        else:
            c_min = c

        # Variable bounds: binary variables [0, 1]
        bounds = Bounds(lb=np.zeros(n), ub=np.ones(n))

        # Integrality: all variables are binary (1 = integer)
        integrality = np.ones(n, dtype=int)

        # Build constraints
        constraints = []

        if A_ub is not None and b_ub is not None:
            # Upper bound constraints: A_ub @ x <= b_ub
            constraints.append(LinearConstraint(A_ub, -np.inf, b_ub))

        if A_eq is not None and b_eq is not None:
            # Equality constraints: A_eq @ x == b_eq
            constraints.append(LinearConstraint(A_eq, b_eq, b_eq))

        try:
            result = milp(
                c=c_min,
                constraints=constraints if constraints else None,
                bounds=bounds,
                integrality=integrality
            )

            if result.success:
                # Round to binary (handle numerical precision)
                x = np.round(result.x).astype(int)
                return x, 'optimal'
            else:
                return None, 'infeasible'

        except Exception as e:
            logger.error(f"MILP solver error: {e}")
            return None, 'error'

    def _build_result(
        self,
        df: pd.DataFrame,
        x: np.ndarray,
        status: str,
        warnings: List[str] = None,
        optimal_value_stage1: Optional[float] = None
    ) -> OptimizationResult:
        """Build OptimizationResult from solution."""
        if warnings is None:
            warnings = []

        selected_mask = x == 1

        selected_df = df[selected_mask].copy()
        not_selected_df = df[~selected_mask].copy()

        # Sort selected by contribution (base_value * budget as proxy for importance)
        selected_df['contribution'] = selected_df['base_value'] * selected_df['budget']
        selected_df = selected_df.sort_values('contribution', ascending=False)
        selected_df['rank'] = range(1, len(selected_df) + 1)

        # Sort not_selected by strategic score (high value projects that weren't selected)
        not_selected_df = not_selected_df.sort_values('strategic_score_pct', ascending=False)

        total_budget_used = selected_df['budget'].sum()
        total_strategic_value = selected_df['base_value'].sum()
        total_duration = selected_df['duration_months'].sum() if 'duration_months' in selected_df.columns else 0

        # Calculate total portfolio value (sum of ALL eligible projects' base_value)
        total_portfolio_value = df['base_value'].sum()
        value_captured_pct = (total_strategic_value / total_portfolio_value * 100) if total_portfolio_value > 0 else 0

        return OptimizationResult(
            status=status,
            message="Optimization completed successfully",
            selected_project_ids=selected_df['project_id'].tolist(),
            not_selected_project_ids=not_selected_df['project_id'].tolist(),
            total_budget_used=total_budget_used,
            budget_utilization_pct=(total_budget_used / self.config.budget_limit * 100) if self.config.budget_limit > 0 else 0,
            total_strategic_value=total_strategic_value,
            avg_strategic_score=selected_df['strategic_score_pct'].mean() if len(selected_df) > 0 else 0,
            total_duration_months=total_duration,
            avg_duration_months=selected_df['duration_months'].mean() if len(selected_df) > 0 and 'duration_months' in selected_df.columns else 0,
            project_count=len(selected_df),
            selected_df=selected_df,
            not_selected_df=not_selected_df,
            warnings=warnings,
            optimal_value_stage1=optimal_value_stage1,
            total_portfolio_value=total_portfolio_value,
            value_captured_pct=value_captured_pct
        )

    def _optimize_base(self, df: pd.DataFrame) -> OptimizationResult:
        """
        Mode 0: Standard 0/1 knapsack - maximize value only.

        Maximize: sum(BaseValue_i * x_i)
        Subject to: sum(Budget_i * x_i) <= BudgetLimit
        """
        n = len(df)

        # Objective: maximize base value
        c = df['base_value'].values

        # Budget constraint
        A_ub = df['budget'].values.reshape(1, -1)
        b_ub = np.array([self.config.budget_limit])

        x, status = self._solve_milp(c, A_ub, b_ub, maximize=True)

        if x is None:
            return OptimizationResult(
                status=status,
                message="Could not find feasible solution"
            )

        return self._build_result(df, x, status)

    def _optimize_soft_duration(self, df: pd.DataFrame) -> OptimizationResult:
        """
        Mode 1: Soft duration preference - favor shorter projects.

        AdjustedValue_i = BaseValue_i * (Dmax / Duration_i)
        Maximize: sum(AdjustedValue_i * x_i)
        Subject to: sum(Budget_i * x_i) <= BudgetLimit
        """
        n = len(df)

        # Calculate Dmax (maximum duration)
        d_max = df['duration_months'].max()

        # Adjusted value favoring shorter duration
        df = df.copy()
        df['adjusted_value'] = df['base_value'] * (d_max / df['duration_months'])

        # Objective: maximize adjusted value
        c = df['adjusted_value'].values

        # Budget constraint
        A_ub = df['budget'].values.reshape(1, -1)
        b_ub = np.array([self.config.budget_limit])

        x, status = self._solve_milp(c, A_ub, b_ub, maximize=True)

        if x is None:
            return OptimizationResult(
                status=status,
                message="Could not find feasible solution"
            )

        return self._build_result(df, x, status)

    def _optimize_lexicographic(self, df: pd.DataFrame) -> OptimizationResult:
        """
        Mode 2: Lexicographic - maximize value first, then minimize duration.

        Stage 1: Solve Mode 0 -> get V*
        Stage 2: Minimize sum(Duration_i * x_i)
        Subject to: sum(BaseValue_i * x_i) >= V* * (1 - epsilon)
                    sum(Budget_i * x_i) <= BudgetLimit
        """
        n = len(df)

        # Stage 1: Maximize value (same as base)
        c_value = df['base_value'].values
        A_budget = df['budget'].values.reshape(1, -1)
        b_budget = np.array([self.config.budget_limit])

        x1, status1 = self._solve_milp(c_value, A_budget, b_budget, maximize=True)

        if x1 is None:
            return OptimizationResult(
                status=status1,
                message="Stage 1 (value maximization) found no feasible solution"
            )

        # Get optimal value from stage 1
        v_star = np.dot(c_value, x1)

        # Stage 2: Minimize duration while maintaining near-optimal value
        epsilon = self.config.epsilon_pct / 100.0
        min_value = v_star * (1 - epsilon)

        # Objective: minimize duration
        c_duration = df['duration_months'].values

        # Constraints:
        # 1. Budget: sum(budget_i * x_i) <= budget_limit
        # 2. Value: sum(value_i * x_i) >= min_value  ->  -sum(value_i * x_i) <= -min_value
        A_ub = np.vstack([
            A_budget,
            -c_value.reshape(1, -1)
        ])
        b_ub = np.array([self.config.budget_limit, -min_value])

        x2, status2 = self._solve_milp(c_duration, A_ub, b_ub, maximize=False)

        if x2 is None:
            # Fallback to stage 1 solution
            return self._build_result(
                df, x1, 'optimal',
                warnings=["Stage 2 optimization failed, using Stage 1 result"],
                optimal_value_stage1=v_star
            )

        return self._build_result(df, x2, 'optimal', optimal_value_stage1=v_star)

    def _optimize_hard_duration(self, df: pd.DataFrame) -> OptimizationResult:
        """
        Mode 3: Hard rule - minimum % of budget must go to "short" projects.

        Maximize: sum(BaseValue_i * x_i)
        Subject to: sum(Budget_i * x_i) <= BudgetLimit
                    sum(Budget_i * x_i * IsShort_i) >= MinShortPct * sum(Budget_i * x_i)

        Where: IsShort_i = 1 if Duration_i <= ShortThreshold else 0

        The second constraint can be rewritten as:
        sum(Budget_i * x_i * IsShort_i) - MinShortPct * sum(Budget_i * x_i) >= 0
        sum(Budget_i * x_i * (IsShort_i - MinShortPct)) >= 0
        -sum(Budget_i * x_i * (IsShort_i - MinShortPct)) <= 0
        """
        n = len(df)

        # Identify short projects
        is_short = (df['duration_months'] <= self.config.short_threshold_months).astype(float).values

        # Check if there are any short projects
        n_short = is_short.sum()
        if n_short == 0:
            return OptimizationResult(
                status='infeasible',
                message=f"No projects with duration <= {self.config.short_threshold_months} months. "
                        f"Consider increasing the short project threshold."
            )

        # Objective: maximize base value
        c = df['base_value'].values

        # Budget values
        budgets = df['budget'].values

        # Min short percentage as fraction
        min_short_pct = self.config.min_short_budget_pct / 100.0

        # Constraints:
        # 1. Budget: sum(budget_i * x_i) <= budget_limit
        # 2. Short budget: -sum(budget_i * x_i * (is_short_i - min_short_pct)) <= 0
        #    This ensures: sum(budget_i * x_i * is_short_i) >= min_short_pct * sum(budget_i * x_i)

        short_constraint_coeffs = -budgets * (is_short - min_short_pct)

        A_ub = np.vstack([
            budgets.reshape(1, -1),
            short_constraint_coeffs.reshape(1, -1)
        ])
        b_ub = np.array([self.config.budget_limit, 0])

        x, status = self._solve_milp(c, A_ub, b_ub, maximize=True)

        if x is None:
            # Check if it's because of the short project constraint
            # Try without the constraint to see if base problem is feasible
            x_base, status_base = self._solve_milp(c, budgets.reshape(1, -1), np.array([self.config.budget_limit]), maximize=True)

            if x_base is not None:
                return OptimizationResult(
                    status='infeasible',
                    message=f"Cannot meet the {self.config.min_short_budget_pct:.0f}% minimum budget for short projects "
                            f"(duration <= {self.config.short_threshold_months} months). "
                            f"Consider reducing the minimum percentage or increasing the short project threshold."
                )

            return OptimizationResult(
                status=status,
                message="Could not find feasible solution"
            )

        # Calculate actual short budget percentage
        selected_mask = x == 1
        selected_budgets = budgets[selected_mask]
        selected_is_short = is_short[selected_mask]
        total_selected_budget = selected_budgets.sum()
        short_budget = (selected_budgets * selected_is_short).sum()
        actual_short_pct = (short_budget / total_selected_budget * 100) if total_selected_budget > 0 else 0

        warnings = [f"Short project budget: {actual_short_pct:.1f}% (minimum required: {self.config.min_short_budget_pct:.0f}%)"]

        return self._build_result(df, x, 'optimal', warnings=warnings)


def generate_optimization_narrative(
    result: OptimizationResult,
    config: OptimizationConfig,
    total_portfolio_budget: float,
    total_projects: int
) -> str:
    """
    Generate a narrative summary of optimization results.

    Args:
        result: OptimizationResult from optimization
        config: OptimizationConfig used
        total_portfolio_budget: Total budget of all projects in portfolio
        total_projects: Total number of projects considered

    Returns:
        Formatted narrative string
    """
    if result.status != 'optimal':
        return f"**Optimization Status:** {result.status}\n\n{result.message}"

    lines = []

    # Overview
    lines.append("## Optimization Summary")
    lines.append("")
    lines.append(f"**Budget Constraint:** {config.budget_limit:,.0f} "
                f"({config.budget_limit / total_portfolio_budget * 100:.1f}% of total portfolio budget)")
    lines.append(f"**Projects Selected:** {result.project_count} of {total_projects} eligible projects")
    lines.append(f"**Budget Utilization:** {result.budget_utilization_pct:.1f}% "
                f"({result.total_budget_used:,.0f} of {config.budget_limit:,.0f})")
    lines.append("")

    # Strategic value
    lines.append("### Strategic Value")
    lines.append(f"- **Value Captured:** {result.value_captured_pct:.1f}% of total portfolio value")
    lines.append(f"- **Average Strategic Score:** {result.avg_strategic_score:.1f}%")
    lines.append("")

    # Duration info
    if result.total_duration_months > 0:
        lines.append("### Duration")
        lines.append(f"- **Total Duration:** {result.total_duration_months:.1f} months")
        lines.append(f"- **Average Duration:** {result.avg_duration_months:.1f} months")
        lines.append("")

    # Mode-specific info
    lines.append("### Optimization Mode")
    mode_descriptions = {
        DurationMode.NONE: "**Value Only** - Maximized strategic value without duration consideration",
        DurationMode.SOFT: "**Soft Duration Preference** - Favored shorter projects while maximizing value",
        DurationMode.LEXICOGRAPHIC: f"**Lexicographic** - Maximized value first, then minimized duration (within {config.epsilon_pct:.0f}% tolerance)",
        DurationMode.HARD: f"**Hard Duration Rule** - Required minimum {config.min_short_budget_pct:.0f}% budget for projects <= {config.short_threshold_months:.0f} months"
    }
    lines.append(mode_descriptions.get(config.duration_mode, "Unknown mode"))

    if config.duration_mode == DurationMode.LEXICOGRAPHIC and result.optimal_value_stage1 is not None:
        lines.append(f"- Stage 1 optimal value: {result.optimal_value_stage1:.2f}")
        lines.append(f"- Final value: {result.total_strategic_value:.2f} "
                    f"({result.total_strategic_value / result.optimal_value_stage1 * 100:.1f}% of maximum)")

    lines.append("")

    # Top selected projects
    if not result.selected_df.empty:
        lines.append("### Top Selected Projects")
        top_n = min(3, len(result.selected_df))
        for i, (_, row) in enumerate(result.selected_df.head(top_n).iterrows(), 1):
            lines.append(f"{i}. **{row['project_name']}** - Score: {row['strategic_score_pct']:.1f}%, "
                        f"Budget: {row['budget']:,.0f}")
        lines.append("")

    # Notable excluded projects
    if not result.not_selected_df.empty:
        high_value_excluded = result.not_selected_df[
            result.not_selected_df['strategic_score_pct'] >= result.avg_strategic_score
        ].head(3)

        if not high_value_excluded.empty:
            lines.append("### Notable Excluded High-Value Projects")
            for _, row in high_value_excluded.iterrows():
                lines.append(f"- **{row['project_name']}** - Score: {row['strategic_score_pct']:.1f}%, "
                            f"Budget: {row['budget']:,.0f}")
            lines.append("")

    # Warnings
    if result.warnings:
        lines.append("### Notes")
        for warning in result.warnings:
            lines.append(f"- {warning}")

    return "\n".join(lines)

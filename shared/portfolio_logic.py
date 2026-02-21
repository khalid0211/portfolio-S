"""
Standalone portfolio logic for external integrations (WhatsApp, API, etc.)

Extracted from:
- pages/11_AI_Assistant.py: Context building functions
- components/brief_chat.py: LLM provider implementations

This module has NO Streamlit dependencies and can be used in FastAPI/Flask apps.
"""

import os
import sys
from typing import Optional, Dict, List, Any
from datetime import date
from dataclasses import dataclass
import pandas as pd
import requests

# Add parent directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from database.db_connection import DatabaseConnection


@dataclass
class LLMConfig:
    """LLM configuration for API calls."""
    provider: str  # OpenAI, Gemini, Claude, Kimi
    model: str
    api_key: str
    temperature: float = 0.2
    timeout: int = 60


class PortfolioLogic:
    """
    Provides portfolio context building and LLM-powered Q&A without Streamlit.

    Usage:
        logic = PortfolioLogic("md:portfolio_cloud?motherduck_token=xxx")
        portfolios = logic.get_available_portfolios()
        context = logic.build_portfolio_context(portfolio_id, status_date)
        answer = logic.ask_question("What is the portfolio health?", context, [], llm_config)
    """

    def __init__(self, db_connection_string: str):
        """
        Initialize with explicit database connection string.

        Args:
            db_connection_string: DuckDB/MotherDuck connection string
                e.g., "md:portfolio_cloud?motherduck_token=..."
        """
        self._db = DatabaseConnection(db_connection_string)

    def close(self):
        """Close database connection."""
        if self._db:
            self._db.close()

    # =========================================================================
    # Portfolio & Project Queries
    # =========================================================================

    def get_available_portfolios(self) -> pd.DataFrame:
        """
        Get list of all active portfolios.

        Returns:
            DataFrame with portfolio_id, portfolio_name, portfolio_manager
        """
        query = """
            SELECT
                portfolio_id,
                portfolio_name,
                portfolio_manager,
                managing_organization
            FROM portfolio
            WHERE is_active = TRUE
            ORDER BY portfolio_name
        """
        return self._db.execute(query).df()

    def get_portfolio_by_name(self, name: str) -> Optional[Dict]:
        """
        Find portfolio by name (case-insensitive partial match).

        Returns:
            Dict with portfolio_id, portfolio_name or None
        """
        query = """
            SELECT portfolio_id, portfolio_name, portfolio_manager
            FROM portfolio
            WHERE LOWER(portfolio_name) LIKE LOWER(?)
            AND is_active = TRUE
            LIMIT 1
        """
        result = self._db.fetch_one(query, (f"%{name}%",))
        if result:
            return {
                "portfolio_id": result[0],
                "portfolio_name": result[1],
                "portfolio_manager": result[2]
            }
        return None

    def get_projects_in_portfolio(self, portfolio_id: int) -> pd.DataFrame:
        """
        Get list of projects in a portfolio.

        Returns:
            DataFrame with project_id, project_name, project_manager, project_status
        """
        query = """
            SELECT
                project_id,
                project_name,
                project_manager,
                responsible_organization,
                project_status
            FROM project
            WHERE portfolio_id = ? AND is_active = TRUE
            ORDER BY project_name
        """
        return self._db.execute(query, (portfolio_id,)).df()

    def get_project_by_name(self, portfolio_id: int, name: str) -> Optional[Dict]:
        """
        Find project by name within a portfolio.

        Returns:
            Dict with project_id, project_name or None
        """
        query = """
            SELECT project_id, project_name, project_manager
            FROM project
            WHERE portfolio_id = ?
            AND LOWER(project_name) LIKE LOWER(?)
            AND is_active = TRUE
            LIMIT 1
        """
        result = self._db.fetch_one(query, (portfolio_id, f"%{name}%"))
        if result:
            return {
                "project_id": result[0],
                "project_name": result[1],
                "project_manager": result[2]
            }
        return None

    def get_latest_status_date(self, portfolio_id: int) -> Optional[date]:
        """Get the most recent status date for a portfolio."""
        query = """
            SELECT MAX(status_date)
            FROM project_status_report
            WHERE portfolio_id = ? AND is_active = TRUE
        """
        result = self._db.fetch_one(query, (portfolio_id,))
        if result and result[0]:
            return result[0]
        return None

    def get_departments_in_portfolio(self, portfolio_id: int) -> pd.DataFrame:
        """
        Get departments (responsible_organization) with project counts and total budget.

        Returns:
            DataFrame with columns: department, project_count, total_budget
        """
        query = """
            SELECT
                responsible_organization as department,
                COUNT(*) as project_count,
                SUM(COALESCE(current_budget, 0)) as total_budget
            FROM project
            WHERE portfolio_id = ? AND is_active = TRUE
            GROUP BY responsible_organization
            ORDER BY responsible_organization
        """
        return self._db.execute(query, (portfolio_id,)).df()

    def get_projects_by_department(self, portfolio_id: int, department: str) -> pd.DataFrame:
        """
        Get projects filtered by responsible_organization (department).

        Returns:
            DataFrame with: project_id, project_name, current_budget, project_status
        """
        query = """
            SELECT
                project_id,
                project_name,
                current_budget,
                project_status,
                project_manager
            FROM project
            WHERE portfolio_id = ?
              AND responsible_organization = ?
              AND is_active = TRUE
            ORDER BY project_name
        """
        return self._db.execute(query, (portfolio_id, department)).df()

    def get_portfolio_summary(self, portfolio_id: int) -> Dict:
        """
        Get portfolio-level aggregates for display after selection.

        Returns:
            Dict with: portfolio_name, total_projects, total_budget, department_count
        """
        query = """
            SELECT
                pf.portfolio_name,
                COUNT(p.project_id) as total_projects,
                SUM(COALESCE(p.current_budget, 0)) as total_budget,
                COUNT(DISTINCT p.responsible_organization) as department_count
            FROM portfolio pf
            LEFT JOIN project p ON pf.portfolio_id = p.portfolio_id AND p.is_active = TRUE
            WHERE pf.portfolio_id = ? AND pf.is_active = TRUE
            GROUP BY pf.portfolio_id, pf.portfolio_name
        """
        result = self._db.fetch_one(query, (portfolio_id,))
        if result:
            return {
                "portfolio_name": result[0],
                "total_projects": result[1] or 0,
                "total_budget": result[2] or 0,
                "department_count": result[3] or 0
            }
        return {"portfolio_name": "", "total_projects": 0, "total_budget": 0, "department_count": 0}

    def get_department_summary(self, portfolio_id: int, department: str) -> Dict:
        """
        Get department-level aggregates.

        Returns:
            Dict with: department_name, project_count, total_budget
        """
        query = """
            SELECT
                responsible_organization as department,
                COUNT(*) as project_count,
                SUM(COALESCE(current_budget, 0)) as total_budget
            FROM project
            WHERE portfolio_id = ?
              AND responsible_organization = ?
              AND is_active = TRUE
            GROUP BY responsible_organization
        """
        result = self._db.fetch_one(query, (portfolio_id, department))
        if result:
            return {
                "department_name": result[0],
                "project_count": result[1] or 0,
                "total_budget": result[2] or 0
            }
        return {"department_name": department, "project_count": 0, "total_budget": 0}

    def get_project_evm_history(self, project_id: int, limit: int = 10) -> pd.DataFrame:
        """
        Get EVM status reports for a project ordered by date for trend analysis.

        Args:
            project_id: Project ID
            limit: Maximum number of recent periods to return (default 10)

        Returns:
            DataFrame with: status_date, pv, ac, ev, spi, cpi
        """
        query = """
            SELECT
                status_date,
                COALESCE(pv, planned_value, 0) as pv,
                COALESCE(ac, actual_cost, 0) as ac,
                COALESCE(ev, earned_value, 0) as ev,
                spi,
                cpi
            FROM project_status_report
            WHERE project_id = ? AND is_active = TRUE
            ORDER BY status_date DESC
            LIMIT ?
        """
        df = self._db.execute(query, (project_id, limit)).df()
        # Return in ascending order for display
        if not df.empty:
            df = df.sort_values('status_date', ascending=True).reset_index(drop=True)
        return df

    def get_portfolio_settings(self, portfolio_id: int) -> Dict:
        """
        Load portfolio settings including LLM config.

        Returns:
            Dict with settings including llm_config, currency_symbol, etc.
        """
        import json

        query = """
            SELECT settings_json,
                   COALESCE(default_curve_type, 'linear') as curve_type,
                   COALESCE(default_alpha, 2.0) as alpha,
                   COALESCE(default_beta, 2.0) as beta,
                   COALESCE(default_inflation_rate, 0.03) as inflation_rate
            FROM portfolio
            WHERE portfolio_id = ? AND is_active = TRUE
        """
        result = self._db.fetch_one(query, (portfolio_id,))

        if not result:
            return {"currency_symbol": "$", "currency_postfix": "", "llm_config": {}}

        settings_json = result[0]
        settings = {}

        if settings_json:
            try:
                settings = json.loads(settings_json)
            except json.JSONDecodeError:
                pass

        # Set defaults
        settings.setdefault("currency_symbol", "$")
        settings.setdefault("currency_postfix", "")
        settings.setdefault("llm_config", {})
        settings.setdefault("curve_type", result[1])
        settings.setdefault("alpha", float(result[2]) if result[2] else 2.0)
        settings.setdefault("beta", float(result[3]) if result[3] else 2.0)
        settings.setdefault("inflation_rate", float(result[4]) if result[4] else 0.03)

        return settings

    def get_evm_results_for_period(
        self,
        portfolio_id: int,
        status_date: date
    ) -> pd.DataFrame:
        """
        Get pre-calculated EVM results for a portfolio and status date.

        Returns:
            DataFrame with all EVM metrics per project
        """
        query = """
            SELECT
                p.project_id,
                p.project_name,
                p.project_manager,
                p.responsible_organization as organization,
                p.project_status as status,
                sr.status_date,
                COALESCE(sr.bac, b.budget_at_completion, 0) as bac,
                COALESCE(sr.ac, sr.actual_cost, 0) as ac,
                COALESCE(sr.ev, sr.calculated_ev, sr.earned_value, 0) as ev,
                COALESCE(sr.pv, sr.calculated_pv, sr.planned_value, 0) as pv,
                sr.cpi,
                sr.spi,
                sr.cv,
                sr.sv,
                sr.eac,
                sr.etc,
                sr.vac,
                sr.tcpi,
                sr.percent_complete,
                sr.percent_budget_used,
                sr.percent_time_used,
                sr.original_duration_months,
                sr.actual_duration_months,
                sr.likely_duration,
                sr.likely_completion_date,
                b.planned_start_date as plan_start,
                b.planned_finish_date as plan_finish
            FROM project p
            LEFT JOIN project_baseline b ON p.project_id = b.project_id
                AND b.is_active = TRUE
                AND b.baseline_end_date IS NULL
            LEFT JOIN project_status_report sr ON p.project_id = sr.project_id
                AND sr.status_date = ?
                AND sr.is_active = TRUE
            WHERE p.portfolio_id = ? AND p.is_active = TRUE
            ORDER BY p.project_name
        """
        return self._db.execute(query, (status_date, portfolio_id)).df()

    def get_project_details(
        self,
        project_id: int,
        status_date: date
    ) -> Dict:
        """
        Get detailed project data for AI context.

        Returns:
            Dict with project info and EVM metrics
        """
        query = """
            SELECT
                p.project_id,
                p.project_name,
                p.project_manager,
                p.responsible_organization as organization,
                p.project_status as status,
                b.budget_at_completion as bac,
                b.planned_start_date as plan_start,
                b.planned_finish_date as plan_finish,
                sr.status_date,
                COALESCE(sr.actual_cost, 0) as ac,
                COALESCE(sr.calculated_ev, sr.earned_value, 0) as ev,
                COALESCE(sr.calculated_pv, sr.planned_value, 0) as pv,
                sr.cpi,
                sr.spi,
                sr.cv,
                sr.sv,
                sr.eac,
                sr.etc,
                sr.vac,
                sr.tcpi,
                sr.percent_complete,
                sr.likely_completion_date as likely_completion
            FROM project p
            LEFT JOIN project_baseline b ON p.project_id = b.project_id
                AND b.is_active = TRUE
                AND (b.baseline_end_date IS NULL OR b.baseline_end_date > ?)
            LEFT JOIN project_status_report sr ON p.project_id = sr.project_id
                AND sr.status_date = ?
                AND sr.is_active = TRUE
            WHERE p.project_id = ? AND p.is_active = TRUE
        """
        result = self._db.fetch_one(query, (status_date, status_date, project_id))

        if result:
            columns = [
                'project_id', 'project_name', 'project_manager', 'organization',
                'status', 'bac', 'plan_start', 'plan_finish', 'status_date',
                'ac', 'ev', 'pv', 'cpi', 'spi', 'cv', 'sv', 'eac', 'etc',
                'vac', 'tcpi', 'percent_complete', 'likely_completion'
            ]
            return dict(zip(columns, result))
        return {}

    # =========================================================================
    # Context Building (extracted from pages/11_AI_Assistant.py)
    # =========================================================================

    @staticmethod
    def format_currency(value: float, symbol: str = "$", postfix: str = "") -> str:
        """Format currency value with appropriate suffix (K, M, B)."""
        if pd.isna(value) or value == 0:
            return f"{symbol}0{postfix}"

        abs_value = abs(value)
        if abs_value >= 1_000_000_000:
            formatted = f"{value/1_000_000_000:.2f}B"
        elif abs_value >= 1_000_000:
            formatted = f"{value/1_000_000:.2f}M"
        elif abs_value >= 1_000:
            formatted = f"{value/1_000:.1f}K"
        else:
            formatted = f"{value:.0f}"

        return f"{symbol}{formatted}{postfix}"

    def build_portfolio_context(
        self,
        portfolio_id: int,
        status_date: date,
        currency_symbol: str = "$",
        currency_postfix: str = ""
    ) -> str:
        """
        Build a structured data context for the AI from portfolio data.

        Returns formatted text snapshot with all EVM metrics.
        """
        df = self.get_evm_results_for_period(portfolio_id, status_date)

        if df.empty:
            return "No data available for this portfolio and period. Please ensure EVM calculations have been run."

        def safe_val(value, default=0):
            if pd.isna(value):
                return default
            try:
                return float(value)
            except:
                return default

        fmt = lambda v: self.format_currency(v, currency_symbol, currency_postfix)

        # Calculate portfolio totals
        total_projects = len(df)
        total_bac = df['bac'].sum() if 'bac' in df.columns else 0
        total_ac = df['ac'].sum() if 'ac' in df.columns else 0
        total_ev = df['ev'].sum() if 'ev' in df.columns else 0
        total_pv = df['pv'].sum() if 'pv' in df.columns else 0
        total_eac = df['eac'].sum() if 'eac' in df.columns else 0

        # Portfolio performance indices
        portfolio_cpi = total_ev / total_ac if total_ac > 0 else 0
        portfolio_spi = total_ev / total_pv if total_pv > 0 else 0

        # TCPI calculation
        work_remaining = total_bac - total_ev
        budget_remaining = total_bac - total_ac
        if budget_remaining <= 0:
            portfolio_tcpi = float('inf') if work_remaining > 0 else 0
        else:
            portfolio_tcpi = work_remaining / budget_remaining

        forecast_overrun = total_eac - total_bac
        percent_complete = (total_ev / total_bac * 100) if total_bac > 0 else 0
        percent_spent = (total_ac / total_bac * 100) if total_bac > 0 else 0

        # Build project details
        project_lines = []
        critical_count = 0
        at_risk_count = 0

        for _, row in df.iterrows():
            proj_cpi = safe_val(row.get('cpi', 0))
            proj_spi = safe_val(row.get('spi', 0))

            if proj_cpi < 0.9 or proj_spi < 0.9:
                health = "CRITICAL"
                critical_count += 1
            elif proj_cpi < 0.95 or proj_spi < 0.95:
                health = "AT RISK"
                at_risk_count += 1
            else:
                health = "Healthy"

            project_name = row.get('project_name', 'Unknown')
            bac = safe_val(row.get('bac', 0))
            ac = safe_val(row.get('ac', 0))
            ev = safe_val(row.get('ev', 0))
            pv = safe_val(row.get('pv', 0))
            proj_complete = safe_val(row.get('percent_complete', 0))
            proj_eac = safe_val(row.get('eac', 0))
            proj_orig_dur = safe_val(row.get('original_duration_months', 0))
            proj_actual_dur = safe_val(row.get('actual_duration_months', 0))
            proj_likely_dur = safe_val(row.get('likely_duration', 0))
            plan_start = row.get('plan_start', None)
            plan_finish = row.get('plan_finish', None)

            start_str = str(plan_start) if pd.notna(plan_start) else "N/A"
            finish_str = str(plan_finish) if pd.notna(plan_finish) else "N/A"

            project_lines.append(
                f"- {project_name}: [{health}]\n"
                f"    Budget (BAC): {fmt(bac)}, AC: {fmt(ac)}, EV: {fmt(ev)}, PV: {fmt(pv)}\n"
                f"    CPI={proj_cpi:.2f}, SPI={proj_spi:.2f}, {proj_complete:.1f}% complete, EAC: {fmt(proj_eac)}\n"
                f"    Plan Start: {start_str}, Plan Finish: {finish_str}\n"
                f"    Orig Duration: {proj_orig_dur:.1f}mo, Actual: {proj_actual_dur:.1f}mo, Likely: {proj_likely_dur:.1f}mo"
            )

        healthy_count = total_projects - critical_count - at_risk_count

        context = f"""PORTFOLIO DATA SNAPSHOT
=======================
Status Date: {status_date}
Total Projects: {total_projects}

FINANCIAL SUMMARY
-----------------
Total Budget (BAC): {fmt(total_bac)}
Actual Cost (AC): {fmt(total_ac)}
Earned Value (EV): {fmt(total_ev)}
Planned Value (PV): {fmt(total_pv)}

PERFORMANCE METRICS
-------------------
Cost Performance Index (CPI): {portfolio_cpi:.3f} {'(Over budget)' if portfolio_cpi < 1 else '(Under budget)' if portfolio_cpi > 1 else '(On budget)'}
Schedule Performance Index (SPI): {portfolio_spi:.3f} {'(Behind schedule)' if portfolio_spi < 1 else '(Ahead of schedule)' if portfolio_spi > 1 else '(On schedule)'}
To-Complete Performance Index (TCPI): {portfolio_tcpi:.3f}

FORECAST
--------
Estimate at Completion (EAC): {fmt(total_eac)}
Forecast Variance: {fmt(forecast_overrun)} {'OVERRUN' if forecast_overrun > 0 else 'UNDERRUN' if forecast_overrun < 0 else ''}

PROGRESS
--------
Work Complete: {percent_complete:.1f}%
Budget Spent: {percent_spent:.1f}%

PORTFOLIO HEALTH
----------------
Critical Projects: {critical_count}
At Risk Projects: {at_risk_count}
Healthy Projects: {healthy_count}

PROJECT DETAILS
---------------
{chr(10).join(project_lines)}
"""
        return context

    def build_project_context(
        self,
        project_id: int,
        status_date: date,
        currency_symbol: str = "$",
        currency_postfix: str = ""
    ) -> str:
        """
        Build a structured data context for a single project.
        """
        project_data = self.get_project_details(project_id, status_date)

        if not project_data:
            return "No data available for this project."

        fmt = lambda v: self.format_currency(v, currency_symbol, currency_postfix)

        # Extract values with defaults
        project_name = project_data.get('project_name', 'Unknown')
        pm = project_data.get('project_manager', 'Unknown')
        org = project_data.get('organization', 'Unknown')
        status = project_data.get('status', 'Unknown')

        bac = project_data.get('bac', 0) or 0
        ac = project_data.get('ac', 0) or 0
        ev = project_data.get('ev', 0) or 0
        pv = project_data.get('pv', 0) or 0

        cpi = project_data.get('cpi', 0) or 0
        spi = project_data.get('spi', 0) or 0
        cv = project_data.get('cv', 0) or 0
        sv = project_data.get('sv', 0) or 0
        eac = project_data.get('eac', 0) or 0
        etc = project_data.get('etc', 0) or 0
        vac = project_data.get('vac', 0) or 0
        tcpi = project_data.get('tcpi', 0) or 0

        percent_complete = project_data.get('percent_complete', 0) or 0
        plan_start = project_data.get('plan_start', 'N/A')
        plan_finish = project_data.get('plan_finish', 'N/A')
        likely_completion = project_data.get('likely_completion', 'N/A')
        status_date_val = project_data.get('status_date', 'N/A')

        # Health status
        if cpi < 0.9 or spi < 0.9:
            health = "CRITICAL - Immediate attention required"
        elif cpi < 0.95 or spi < 0.95:
            health = "AT RISK - Monitoring required"
        else:
            health = "HEALTHY - On track"

        context = f"""PROJECT DATA SNAPSHOT
=====================
Project: {project_name}
Project Manager: {pm}
Organization: {org}
Status: {status}
Status Date: {status_date_val}

SCHEDULE
--------
Planned Start: {plan_start}
Planned Finish: {plan_finish}
Likely Completion: {likely_completion}

FINANCIAL SUMMARY
-----------------
Budget at Completion (BAC): {fmt(bac)}
Actual Cost (AC): {fmt(ac)}
Earned Value (EV): {fmt(ev)}
Planned Value (PV): {fmt(pv)}

VARIANCES
---------
Cost Variance (CV): {fmt(cv)} {'(Over budget)' if cv < 0 else '(Under budget)' if cv > 0 else ''}
Schedule Variance (SV): {fmt(sv)} {'(Behind schedule)' if sv < 0 else '(Ahead)' if sv > 0 else ''}

PERFORMANCE INDICES
-------------------
Cost Performance Index (CPI): {cpi:.3f} {'(Inefficient)' if cpi < 1 else '(Efficient)' if cpi > 1 else ''}
Schedule Performance Index (SPI): {spi:.3f} {'(Behind)' if spi < 1 else '(Ahead)' if spi > 1 else ''}
To-Complete Performance Index (TCPI): {tcpi:.3f}

FORECAST
--------
Estimate at Completion (EAC): {fmt(eac)}
Estimate to Complete (ETC): {fmt(etc)}
Variance at Completion (VAC): {fmt(vac)}

PROGRESS
--------
Percent Complete: {percent_complete:.1f}%

HEALTH STATUS
-------------
{health}
"""
        return context

    # =========================================================================
    # LLM Calling (extracted from components/brief_chat.py)
    # =========================================================================

    def ask_question(
        self,
        question: str,
        context: str,
        conversation_history: List[Dict[str, str]],
        llm_config: LLMConfig
    ) -> str:
        """
        Generate answer using configured LLM provider.

        Args:
            question: User's question
            context: Portfolio/project data context
            conversation_history: List of {"question": str, "answer": str} dicts
            llm_config: LLM configuration (provider, model, api_key, etc.)

        Returns:
            LLM-generated answer string
        """
        if not llm_config.api_key.strip():
            raise ValueError("LLM API key is not configured")

        system_prompt = (
            "You are a project controls expert specializing in earned value management. "
            "Answer concisely in under 200 words based ONLY on the report context provided. "
            "Use specific numbers and project names from the report. "
            "IMPORTANT: Always use full names for EVM terms, not abbreviations. "
            "Examples: say 'Actual Cost' not 'AC', 'Budget at Completion' not 'BAC', "
            "'Planned Value' not 'PV', 'Earned Value' not 'EV', "
            "'Schedule Performance Index' not 'SPI', 'Cost Performance Index' not 'CPI'."
        )

        # Build conversation messages
        history_messages = []
        for qa in conversation_history:
            history_messages.append({"role": "user", "content": qa["question"]})
            history_messages.append({"role": "assistant", "content": qa["answer"]})

        user_content = f"REPORT DATA:\n{context}\n\nQUESTION: {question}"

        provider = llm_config.provider.lower()

        if provider == "openai":
            return self._call_openai(
                llm_config.api_key, llm_config.model, system_prompt,
                history_messages, user_content, llm_config.temperature, llm_config.timeout
            )
        elif provider == "gemini":
            return self._call_gemini(
                llm_config.api_key, llm_config.model, system_prompt,
                history_messages, user_content, llm_config.temperature, llm_config.timeout
            )
        elif provider == "claude":
            return self._call_claude(
                llm_config.api_key, llm_config.model, system_prompt,
                history_messages, user_content, llm_config.temperature, llm_config.timeout
            )
        elif provider == "kimi":
            return self._call_kimi(
                llm_config.api_key, llm_config.model, system_prompt,
                history_messages, user_content, llm_config.temperature, llm_config.timeout
            )
        else:
            raise ValueError(f"Unsupported LLM provider: {llm_config.provider}")

    def _call_openai(self, api_key, model, system_prompt, history, user_content, temperature, timeout):
        """Call OpenAI API."""
        messages = [{"role": "system", "content": system_prompt}]
        messages.extend(history)
        messages.append({"role": "user", "content": user_content})

        resp = requests.post(
            "https://api.openai.com/v1/chat/completions",
            headers={"Authorization": f"Bearer {api_key.strip()}", "Content-Type": "application/json"},
            json={"model": model.strip(), "messages": messages, "temperature": temperature, "max_tokens": 2000},
            timeout=timeout,
        )
        resp.raise_for_status()
        data = resp.json()
        return data["choices"][0]["message"]["content"].strip()

    def _call_gemini(self, api_key, model, system_prompt, history, user_content, temperature, timeout):
        """Call Google Gemini API."""
        model_name = model.strip()
        if not model_name.startswith("models/"):
            model_name = f"models/{model_name}"

        contents = []
        full_first = f"{system_prompt}\n\n{history[0]['content']}" if history else f"{system_prompt}\n\n{user_content}"

        if history:
            contents.append({"role": "user", "parts": [{"text": full_first}]})
            for msg in history[1:]:
                role = "user" if msg["role"] == "user" else "model"
                contents.append({"role": role, "parts": [{"text": msg["content"]}]})
            contents.append({"role": "user", "parts": [{"text": user_content}]})
        else:
            contents.append({"role": "user", "parts": [{"text": full_first}]})

        resp = requests.post(
            f"https://generativelanguage.googleapis.com/v1beta/{model_name}:generateContent",
            headers={"Content-Type": "application/json"},
            params={"key": api_key.strip()},
            json={"contents": contents, "generationConfig": {"temperature": temperature, "maxOutputTokens": 2000}},
            timeout=timeout,
        )
        resp.raise_for_status()
        data = resp.json()
        return data["candidates"][0]["content"]["parts"][0]["text"].strip()

    def _call_claude(self, api_key, model, system_prompt, history, user_content, temperature, timeout):
        """Call Anthropic Claude API."""
        messages = list(history)
        messages.append({"role": "user", "content": user_content})

        resp = requests.post(
            "https://api.anthropic.com/v1/messages",
            headers={
                "x-api-key": api_key.strip(),
                "anthropic-version": "2023-06-01",
                "Content-Type": "application/json"
            },
            json={
                "model": model.strip(),
                "max_tokens": 2000,
                "temperature": temperature,
                "system": system_prompt,
                "messages": messages
            },
            timeout=timeout,
        )
        resp.raise_for_status()
        data = resp.json()
        text = "".join(block.get("text", "") for block in data["content"] if block.get("type") == "text")
        return text.strip()

    def _call_kimi(self, api_key, model, system_prompt, history, user_content, temperature, timeout):
        """Call Kimi (Moonshot AI) API."""
        messages = [{"role": "system", "content": system_prompt}]
        messages.extend(history)
        messages.append({"role": "user", "content": user_content})

        resp = requests.post(
            "https://api.moonshot.ai/v1/chat/completions",
            headers={"Authorization": f"Bearer {api_key.strip()}", "Content-Type": "application/json"},
            json={"model": model.strip(), "messages": messages, "temperature": temperature, "max_tokens": 2000},
            timeout=timeout,
        )
        resp.raise_for_status()
        data = resp.json()
        return data["choices"][0]["message"]["content"].strip()

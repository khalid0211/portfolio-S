"""
Strategic Factors - Portfolio Management Suite
Define portfolio-level strategic factors and score projects for strategic alignment
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from typing import List, Optional, Dict, Tuple
from utils.auth import check_authentication, require_page_access
from config.constants import USE_DATABASE
from services.db_data_service import DatabaseDataManager
from services.portfolio_optimization_service import (
    PortfolioOptimizer, OptimizationConfig, OptimizationResult,
    DurationMode, generate_optimization_narrative
)
from database.db_connection import get_db

# Page configuration
st.set_page_config(
    page_title="Strategic Factors - Portfolio Suite",
    page_icon="🎯",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# Check authentication and page access
if not check_authentication():
    st.stop()

require_page_access('strategic_factors', 'Strategic Factors')

# Custom CSS
st.markdown("""
<style>
    .main-header {
        text-align: center;
        color: #2c3e50;
        font-size: 2.2rem;
        font-weight: 600;
        margin-bottom: 1rem;
        border-bottom: 3px solid #9b59b6;
        padding-bottom: 0.5rem;
    }
    .factor-card {
        background: rgba(255, 255, 255, 0.95);
        padding: 1.5rem;
        border-radius: 10px;
        margin: 1rem 0;
        border-left: 5px solid #9b59b6;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
    }
    .alignment-card {
        background: rgba(255, 255, 255, 0.95);
        padding: 1rem;
        border-radius: 10px;
        margin: 0.5rem 0;
        border-left: 5px solid #3498db;
        box-shadow: 0 2px 4px rgba(0, 0, 0, 0.1);
    }
    .alignment-card.high {
        border-left-color: #27ae60;
        background: rgba(39, 174, 96, 0.05);
    }
    .alignment-card.medium {
        border-left-color: #f39c12;
        background: rgba(243, 156, 18, 0.05);
    }
    .alignment-card.low {
        border-left-color: #e74c3c;
        background: rgba(231, 76, 60, 0.05);
    }
    .stat-box {
        background: linear-gradient(135deg, #9b59b6 0%, #8e44ad 100%);
        color: white;
        padding: 1.5rem;
        border-radius: 10px;
        text-align: center;
        margin: 0.5rem;
    }
    .stat-number {
        font-size: 2.5rem;
        font-weight: 700;
        margin: 0;
    }
    .stat-label {
        font-size: 0.9rem;
        opacity: 0.9;
        margin-top: 0.5rem;
    }
    .score-badge {
        display: inline-block;
        padding: 0.5rem 1rem;
        border-radius: 20px;
        font-size: 1.2rem;
        font-weight: 600;
        margin: 0.5rem;
    }
    .badge-high {
        background: #27ae60;
        color: white;
    }
    .badge-medium {
        background: #f39c12;
        color: white;
    }
    .badge-low {
        background: #e74c3c;
        color: white;
    }
</style>
""", unsafe_allow_html=True)

# Main header
st.markdown('<h1 class="main-header">🎯 Strategic Factors</h1>', unsafe_allow_html=True)
st.markdown("Define strategic factors and score projects for strategic alignment")

# Check database mode
if not USE_DATABASE:
    st.warning("⚠️ Database mode is not enabled")
    st.info("""
    **Database Mode Required:**
    Strategic Factors requires database mode to be enabled.
    Set `USE_DATABASE = True` in `config/constants.py`
    """)
    st.stop()

# Initialize database connection with user-specific config
user_email = st.session_state.get('user_email')
db = get_db(user_email=user_email)

# Initialize database manager (uses the singleton db instance)
db_manager = DatabaseDataManager()


def select_portfolio():
    """UI for selecting portfolio"""
    portfolios_df = db.execute("""
        SELECT portfolio_id, portfolio_name
        FROM portfolio
        WHERE is_active = TRUE
        ORDER BY portfolio_name
    """).df()

    if portfolios_df.empty:
        st.warning("⚠️ No portfolios found. Create a portfolio first in Portfolio Management.")
        return None

    portfolio_options = {
        row['portfolio_name']: row['portfolio_id']
        for _, row in portfolios_df.iterrows()
    }

    selected_portfolio_name = st.selectbox("Select Portfolio", options=list(portfolio_options.keys()))
    return portfolio_options[selected_portfolio_name]


def create_factor_ui(portfolio_id):
    """UI for creating strategic factors"""
    st.markdown("### ➕ Create Strategic Factor")

    st.info("""
    **Strategic Factors** are portfolio-specific criteria used to evaluate project alignment with strategic goals.
    Examples: Innovation Potential, Strategic Fit, Market Impact, Risk Level, etc.
    """)

    with st.form("create_factor_form"):
        col1, col2 = st.columns(2)

        with col1:
            factor_name = st.text_input(
                "Factor Name *",
                placeholder="e.g., Strategic Alignment, Innovation Potential"
            )
            weight = st.number_input(
                "Weight (%) *",
                min_value=0.0,
                max_value=100.0,
                value=20.0,
                step=5.0,
                help="Percentage weight of this factor. All factors should sum to 100%."
            )

        with col2:
            likert_min = st.number_input("Minimum Score", min_value=0, max_value=10, value=1)
            likert_max = st.number_input("Maximum Score", min_value=1, max_value=10, value=5)

        submitted = st.form_submit_button("🚀 Create Factor", type="primary")

        if submitted and not st.session_state.get('portfolio_access_level') in ('owner', 'write'):
            st.error("You have read-only access to this portfolio. Cannot create factors.")
        elif submitted:
            if not factor_name:
                st.error("❌ Please enter a factor name")
            elif likert_min >= likert_max:
                st.error("❌ Maximum score must be greater than minimum score")
            else:
                try:
                    # Check current total weight
                    factors_df = db_manager.list_factors(portfolio_id)
                    current_total_weight = factors_df['factor_weight_percent'].sum() if not factors_df.empty else 0
                    new_total = current_total_weight + weight

                    if new_total > 100:
                        st.warning(f"⚠️ Total weight would be {new_total:.1f}% (exceeds 100%). Current total: {current_total_weight:.1f}%")
                    else:
                        factor_id = db_manager.create_factor(
                            portfolio_id=portfolio_id,
                            factor_name=factor_name,
                            factor_weight_percent=weight,
                            likert_min=likert_min,
                            likert_max=likert_max
                        )
                        st.success(f"✅ Factor '{factor_name}' created successfully (ID: {factor_id})")
                        st.info(f"Total weight: {new_total:.1f}% of 100%")
                        st.rerun()

                except Exception as e:
                    st.error(f"❌ Error creating factor: {e}")


def list_factors_ui(portfolio_id):
    """UI for listing strategic factors"""
    st.markdown("### 📋 Portfolio Strategic Factors")

    factors_df = db_manager.list_factors(portfolio_id)

    if factors_df.empty:
        st.info("📭 No strategic factors defined yet. Create factors above to get started.")
        return

    # Display total weight
    total_weight = factors_df['factor_weight_percent'].sum()
    weight_status = "✅" if abs(total_weight - 100) < 0.01 else "⚠️"

    st.markdown(f"**Total Weight:** {weight_status} {total_weight:.1f}% of 100%")

    if abs(total_weight - 100) > 0.01:
        st.warning(f"⚠️ Factor weights should sum to 100%. Current total: {total_weight:.1f}%")

    st.markdown("---")

    # Display each factor as a card
    for _, factor in factors_df.iterrows():
        with st.container():
            st.markdown(f'<div class="factor-card">', unsafe_allow_html=True)

            col_info, col_details, col_actions = st.columns([3, 2, 1])

            with col_info:
                st.markdown(f"### {factor['factor_name']}")
                st.markdown(f"**Weight:** {factor['factor_weight_percent']:.1f}%")

            with col_details:
                st.markdown(f"**Score Range:** {factor['likert_min']} - {factor['likert_max']}")
                st.caption(f"Created: {factor['created_at']}")

            with col_actions:
                st.markdown("##")  # Spacing
                _can_edit = st.session_state.get('portfolio_access_level') in ('owner', 'write')
                if _can_edit:
                    col_edit, col_delete = st.columns(2)

                    with col_edit:
                        if st.button("✏️", key=f"edit_{factor['factor_id']}", help="Edit factor"):
                            st.session_state[f'editing_factor_{factor["factor_id"]}'] = True
                            st.rerun()

                    with col_delete:
                        if st.button("🗑️", key=f"delete_{factor['factor_id']}", help="Delete factor"):
                            st.session_state[f'confirm_delete_{factor["factor_id"]}'] = True
                            st.rerun()

            # Display weight as progress bar
            st.progress(factor['factor_weight_percent'] / 100.0)

            # Edit dialog
            if st.session_state.get(f'editing_factor_{factor["factor_id"]}', False):
                st.markdown("---")
                with st.form(f"edit_form_{factor['factor_id']}"):
                    st.markdown(f"#### Edit Factor: {factor['factor_name']}")

                    col1, col2 = st.columns(2)

                    with col1:
                        new_name = st.text_input("Factor Name", value=factor['factor_name'])
                        new_weight = st.number_input(
                            "Weight (%)",
                            min_value=0.0,
                            max_value=100.0,
                            value=float(factor['factor_weight_percent']),
                            step=5.0
                        )

                    with col2:
                        new_min = st.number_input("Minimum Score", min_value=0, max_value=10, value=int(factor['likert_min']))
                        new_max = st.number_input("Maximum Score", min_value=1, max_value=10, value=int(factor['likert_max']))

                    col_save, col_cancel = st.columns(2)

                    with col_save:
                        submitted = st.form_submit_button("💾 Save Changes", type="primary", width='stretch')

                    with col_cancel:
                        cancelled = st.form_submit_button("❌ Cancel", width='stretch')

                    if submitted:
                        if not new_name:
                            st.error("Factor name cannot be empty")
                        elif new_min >= new_max:
                            st.error("Maximum score must be greater than minimum score")
                        else:
                            try:
                                # Check total weight
                                other_factors_weight = factors_df[factors_df['factor_id'] != factor['factor_id']]['factor_weight_percent'].sum()
                                new_total = other_factors_weight + new_weight

                                if new_total > 100:
                                    st.warning(f"⚠️ Total weight would be {new_total:.1f}% (exceeds 100%)")
                                else:
                                    db_manager.update_factor(
                                        factor_id=factor['factor_id'],
                                        factor_name=new_name,
                                        factor_weight_percent=new_weight,
                                        likert_min=new_min,
                                        likert_max=new_max
                                    )
                                    st.success(f"✅ Factor updated successfully")
                                    del st.session_state[f'editing_factor_{factor["factor_id"]}']
                                    st.rerun()
                            except Exception as e:
                                st.error(f"Error updating factor: {e}")

                    if cancelled:
                        del st.session_state[f'editing_factor_{factor["factor_id"]}']
                        st.rerun()

            # Delete confirmation
            if st.session_state.get(f'confirm_delete_{factor["factor_id"]}', False):
                st.markdown("---")
                st.warning(f"⚠️ **Confirm Deletion**")
                st.markdown(f"Are you sure you want to delete factor **'{factor['factor_name']}'**?")
                st.info("Note: This is a soft delete. Project scores will be preserved but hidden.")

                col_confirm, col_cancel = st.columns(2)

                with col_confirm:
                    if st.button("✅ Yes, Delete", key=f"confirm_yes_{factor['factor_id']}", type="primary", width='stretch'):
                        try:
                            db_manager.delete_factor(factor['factor_id'])
                            st.success(f"✅ Factor '{factor['factor_name']}' deleted successfully")
                            del st.session_state[f'confirm_delete_{factor["factor_id"]}']
                            st.rerun()
                        except Exception as e:
                            st.error(f"Error deleting factor: {e}")

                with col_cancel:
                    if st.button("❌ Cancel", key=f"confirm_no_{factor['factor_id']}", width='stretch'):
                        del st.session_state[f'confirm_delete_{factor["factor_id"]}']
                        st.rerun()

            st.markdown('</div>', unsafe_allow_html=True)


def export_factor_scores_to_csv(portfolio_id):
    """Export factor scores in wide format: portfolio_id, project_id, project_name, organization, project_manager, factor_1, factor_2, ..."""

    # Get factors
    factors_df = db_manager.list_factors(portfolio_id)

    if factors_df.empty:
        st.warning("No factors defined to export")
        return None

    # Get all projects with organization and project manager - single query
    projects_df = db.execute("""
        SELECT project_id, project_name, responsible_organization, project_manager
        FROM project
        WHERE portfolio_id = ? AND is_active = TRUE
        ORDER BY project_name
    """, (portfolio_id,)).df()

    if projects_df.empty:
        st.warning("No projects found to export")
        return None

    # Get ALL factor scores for this portfolio in a single query
    all_scores_df = db.execute("""
        SELECT pfs.project_id, pfs.factor_id, pfs.score
        FROM project_factor_score pfs
        JOIN project p ON pfs.project_id = p.project_id
        WHERE p.portfolio_id = ? AND pfs.is_active = TRUE AND p.is_active = TRUE
    """, (portfolio_id,)).df()

    # Create base dataframe with portfolio and project info
    export_df = pd.DataFrame({
        'portfolio_id': [portfolio_id] * len(projects_df),
        'project_id': projects_df['project_id'],
        'project_name': projects_df['project_name'],
        'organization': projects_df['responsible_organization'],
        'project_manager': projects_df['project_manager']
    })

    # Create a lookup dict for scores: (project_id, factor_id) -> score
    scores_lookup = {}
    if not all_scores_df.empty:
        for _, row in all_scores_df.iterrows():
            key = (row['project_id'], row['factor_id'])
            score_val = row['score']
            if score_val is not None and not pd.isna(score_val):
                scores_lookup[key] = int(score_val)

    # Add columns for each factor using the lookup (no additional DB calls)
    for _, factor in factors_df.iterrows():
        factor_id = factor['factor_id']
        factor_name = factor['factor_name']

        # Get scores for this factor across all projects from lookup
        scores = []
        for project_id in projects_df['project_id']:
            score = scores_lookup.get((project_id, factor_id), None)
            scores.append(score)

        export_df[factor_name] = scores

    return export_df


def export_portfolio_scores_to_csv(portfolio_id):
    """Export comprehensive portfolio scores including budget, dates, factor scores, and status metrics"""

    db = get_db()

    # Get factors with weights
    factors_df = db_manager.list_factors(portfolio_id)

    if factors_df.empty:
        st.warning("No factors defined for this portfolio")
        return None

    # Get projects with latest status report data
    projects_df = db.execute("""
        SELECT p.project_id, p.project_name, p.responsible_organization, p.project_manager,
               p.current_budget, p.planned_start_date,
               sr.original_duration_months, sr.percent_budget_used, sr.percent_time_used
        FROM project p
        LEFT JOIN (
            SELECT project_id, MAX(status_date) as latest_date
            FROM project_status_report WHERE portfolio_id = ? AND is_active = TRUE GROUP BY project_id
        ) latest ON p.project_id = latest.project_id
        LEFT JOIN project_status_report sr ON p.project_id = sr.project_id
            AND sr.status_date = latest.latest_date AND sr.portfolio_id = ? AND sr.is_active = TRUE
        WHERE p.portfolio_id = ? AND p.is_active = TRUE
        ORDER BY p.project_name
    """, (portfolio_id, portfolio_id, portfolio_id)).df()

    if projects_df.empty:
        st.warning("No projects found to export")
        return None

    # Get ALL factor scores for this portfolio in a single query
    all_scores_df = db.execute("""
        SELECT pfs.project_id, pfs.factor_id, pfs.score
        FROM project_factor_score pfs
        JOIN project p ON pfs.project_id = p.project_id
        WHERE p.portfolio_id = ? AND pfs.is_active = TRUE AND p.is_active = TRUE
    """, (portfolio_id,)).df()

    # Create scores lookup: (project_id, factor_id) -> score
    scores_lookup = {}
    if not all_scores_df.empty:
        for _, row in all_scores_df.iterrows():
            key = (row['project_id'], row['factor_id'])
            score_val = row['score']
            if score_val is not None and not pd.isna(score_val):
                scores_lookup[key] = int(score_val)

    # Build export dataframe - use original_duration_months from latest status report
    export_df = pd.DataFrame({
        'portfolio_id': [portfolio_id] * len(projects_df),
        'project_id': projects_df['project_id'],
        'project_name': projects_df['project_name'],
        'organization': projects_df['responsible_organization'],
        'project_manager': projects_df['project_manager'],
        'budget': projects_df['current_budget'],
        'start_date': projects_df['planned_start_date'],
        'duration_months': projects_df['original_duration_months']
    })

    # Add factor score columns and calculate portfolio score
    factor_weights = {}
    for _, factor in factors_df.iterrows():
        factor_id = factor['factor_id']
        factor_name = factor['factor_name']
        factor_weights[factor_id] = float(factor['factor_weight_percent']) if pd.notna(factor['factor_weight_percent']) else 0.0

        scores = []
        for project_id in projects_df['project_id']:
            score = scores_lookup.get((project_id, factor_id), None)
            scores.append(score)

        export_df[factor_name] = scores

    # Calculate portfolio_score (weighted sum)
    portfolio_scores = []
    for project_id in projects_df['project_id']:
        weighted_sum = 0.0
        all_scored = True

        for factor_id, weight in factor_weights.items():
            score = scores_lookup.get((project_id, factor_id), None)
            if score is None:
                all_scored = False
                break
            weighted_sum += score * (weight / 100.0)

        if all_scored and len(factor_weights) > 0:
            portfolio_scores.append(round(weighted_sum, 2))
        else:
            portfolio_scores.append(None)

    export_df['portfolio_score'] = portfolio_scores

    # Add status metrics
    export_df['percent_budget_used'] = projects_df['percent_budget_used']
    export_df['percent_time_used'] = projects_df['percent_time_used']

    return export_df


def import_factor_scores_from_csv(portfolio_id, uploaded_file):
    """Import factor scores from CSV in wide format"""

    try:
        # Read CSV
        import_df = pd.read_csv(uploaded_file)

        # Validate required columns
        required_cols = ['portfolio_id', 'project_id', 'project_name']
        missing_cols = [col for col in required_cols if col not in import_df.columns]

        if missing_cols:
            st.error(f"Missing required columns: {', '.join(missing_cols)}")
            return 0

        # Get current factors
        factors_df = db_manager.list_factors(portfolio_id)

        if factors_df.empty:
            st.error("No factors defined. Create factors first before importing scores.")
            return 0

        # Create factor name to ID mapping
        factor_map = {row['factor_name']: row['factor_id'] for _, row in factors_df.iterrows()}

        # Get factor columns (any column that matches a factor name)
        factor_columns = [col for col in import_df.columns if col in factor_map]

        if not factor_columns:
            st.warning("No factor columns found in CSV that match existing factors")
            return 0

        # Import scores
        scores_imported = 0
        errors = []

        for idx, row in import_df.iterrows():
            project_id = row['project_id']

            # Verify project exists and belongs to this portfolio
            project_check = db.execute("""
                SELECT project_id FROM project
                WHERE project_id = ? AND portfolio_id = ? AND is_active = TRUE
            """, (project_id, portfolio_id)).fetchone()

            if not project_check:
                errors.append(f"Row {idx+1}: Project ID {project_id} not found or not in this portfolio")
                continue

            # Import scores for each factor
            for factor_name in factor_columns:
                score_value = row[factor_name]

                # Skip if empty/null
                if pd.isna(score_value):
                    continue

                try:
                    score = int(score_value)
                    factor_id = factor_map[factor_name]

                    # Get factor's likert range for validation
                    factor_info = factors_df[factors_df['factor_id'] == factor_id].iloc[0]
                    likert_min = int(factor_info['likert_min'])
                    likert_max = int(factor_info['likert_max'])

                    # Validate score range
                    if score < likert_min or score > likert_max:
                        errors.append(f"Row {idx+1}, {factor_name}: Score {score} out of range ({likert_min}-{likert_max})")
                        continue

                    # Save score
                    if db_manager.score_project_factor(project_id, factor_id, score):
                        scores_imported += 1
                    else:
                        errors.append(f"Row {idx+1}, {factor_name}: Failed to save score")

                except (ValueError, TypeError) as e:
                    errors.append(f"Row {idx+1}, {factor_name}: Invalid score value '{score_value}'")
                    continue

        # Display results
        if scores_imported > 0:
            st.success(f"✅ Imported {scores_imported} scores successfully")

        if errors:
            with st.expander(f"⚠️ {len(errors)} errors occurred", expanded=False):
                for error in errors[:20]:  # Show first 20 errors
                    st.warning(error)
                if len(errors) > 20:
                    st.info(f"... and {len(errors) - 20} more errors")

        return scores_imported

    except Exception as e:
        st.error(f"Failed to import CSV: {str(e)}")
        return 0


def score_projects_ui(portfolio_id):
    """UI for scoring projects against factors"""
    st.markdown("### 📊 Score Projects")

    # Get factors
    factors_df = db_manager.list_factors(portfolio_id)

    if factors_df.empty:
        st.info("📭 No strategic factors defined. Create factors first in the 'Manage Factors' tab.")
        return

    # Get projects
    projects_df = db.execute("""
        SELECT project_id, project_name
        FROM project
        WHERE portfolio_id = ? AND is_active = TRUE
        ORDER BY project_name
    """, (portfolio_id,)).df()

    if projects_df.empty:
        st.info("📭 No projects found in this portfolio")
        return

    # Import/Export section
    st.markdown("#### 📤 Import/Export Scores")

    col_factor_export, col_portfolio_export, col_import = st.columns(3)

    with col_factor_export:
        if st.button("📥 Export Factor Scores", width='stretch'):
            export_df = export_factor_scores_to_csv(portfolio_id)

            if export_df is not None:
                csv = export_df.to_csv(index=False)
                st.download_button(
                    label="💾 Download Factor CSV",
                    data=csv,
                    file_name=f"factor_scores_portfolio_{portfolio_id}.csv",
                    mime="text/csv",
                    width='stretch'
                )
                st.success(f"✅ Ready! ({len(export_df)} projects)")

    with col_portfolio_export:
        if st.button("📊 Export Portfolio Scores", width='stretch'):
            export_df = export_portfolio_scores_to_csv(portfolio_id)

            if export_df is not None:
                csv = export_df.to_csv(index=False)
                st.download_button(
                    label="💾 Download Portfolio CSV",
                    data=csv,
                    file_name=f"portfolio_scores_{portfolio_id}.csv",
                    mime="text/csv",
                    width='stretch'
                )
                st.success(f"✅ Ready! ({len(export_df)} projects)")

    with col_import:
        uploaded_file = st.file_uploader(
            "📤 Import from CSV",
            type=['csv'],
            help="Upload CSV with columns: portfolio_id, project_id, and factor names. Columns like project_name, organization, project_manager are ignored."
        )

        if uploaded_file is not None:
            if st.button("⬆️ Import Scores", width='stretch'):
                scores_imported = import_factor_scores_from_csv(portfolio_id, uploaded_file)
                if scores_imported > 0:
                    st.rerun()

    st.markdown("---")

    # Project selector
    project_options = {
        row['project_name']: row['project_id']
        for _, row in projects_df.iterrows()
    }

    selected_project_name = st.selectbox("Select Project", options=list(project_options.keys()))
    project_id = project_options[selected_project_name]

    st.markdown("---")

    # Get current scores
    scores_df = db_manager.get_project_factor_scores(project_id)

    st.markdown(f"#### Score '{selected_project_name}'")

    # Display scoring interface
    changes_made = False

    for _, factor in factors_df.iterrows():
        factor_id = factor['factor_id']
        factor_name = factor['factor_name']
        likert_min = int(factor['likert_min'])
        likert_max = int(factor['likert_max'])

        # Get current score if exists
        current_score = None
        if not scores_df.empty:
            factor_scores = scores_df[scores_df['factor_id'] == factor_id]
            if not factor_scores.empty:
                score_value = factor_scores.iloc[0]['score']
                if score_value is not None and not pd.isna(score_value):
                    current_score = int(score_value)

        st.markdown(f'<div class="factor-card">', unsafe_allow_html=True)

        col_name, col_weight, col_score = st.columns([3, 1, 2])

        with col_name:
            st.markdown(f"**{factor_name}**")

        with col_weight:
            st.caption(f"Weight: {factor['factor_weight_percent']:.1f}%")

        with col_score:
            new_score = st.slider(
                f"Score",
                min_value=likert_min,
                max_value=likert_max,
                value=current_score if current_score is not None else likert_min,
                key=f"score_{factor_id}",
                label_visibility="collapsed"
            )

        st.markdown('</div>', unsafe_allow_html=True)

        # Save score if changed
        if new_score != current_score:
            if db_manager.score_project_factor(project_id, factor_id, new_score):
                changes_made = True

    if changes_made:
        st.success("✅ Scores saved")
        st.rerun()


def strategic_alignment_ui(portfolio_id):
    """UI for viewing strategic alignment"""
    st.markdown("### 🎯 Strategic Alignment Dashboard")

    # Get alignment scores
    alignment_df = db_manager.get_portfolio_strategic_alignment(portfolio_id)

    if alignment_df.empty:
        st.info("📭 No alignment data available. Score projects first.")
        return

    # Calculate statistics
    avg_alignment = alignment_df['alignment_score'].mean()
    max_alignment = alignment_df['alignment_score'].max()
    min_alignment = alignment_df['alignment_score'].min()

    # Get budget information for budget-weighted average
    budget_query = """
        SELECT project_id, current_budget
        FROM project
        WHERE portfolio_id = ? AND is_active = TRUE
    """
    budget_df = db.execute(budget_query, (portfolio_id,)).df()

    # Merge budget with alignment scores
    alignment_with_budget = alignment_df.merge(budget_df, on='project_id', how='left')

    # Calculate budget weighted average (only for projects with budget and alignment)
    budget_weighted_df = alignment_with_budget[
        (alignment_with_budget['current_budget'].notna()) &
        (alignment_with_budget['current_budget'] > 0) &
        (alignment_with_budget['alignment_score'].notna())
    ]

    if not budget_weighted_df.empty:
        budget_weighted_avg = (
            (budget_weighted_df['alignment_score'] * budget_weighted_df['current_budget']).sum() /
            budget_weighted_df['current_budget'].sum()
        )
    else:
        budget_weighted_avg = None

    # Display summary statistics
    col1, col2, col3, col4 = st.columns(4)

    with col1:
        st.markdown(f"""
        <div class="stat-box">
            <div class="stat-number">{avg_alignment:.1f}%</div>
            <div class="stat-label">Simple Average Alignment</div>
        </div>
        """, unsafe_allow_html=True)

    with col2:
        if budget_weighted_avg is not None:
            st.markdown(f"""
            <div class="stat-box">
                <div class="stat-number">{budget_weighted_avg:.1f}%</div>
                <div class="stat-label">Budget Weighted Average</div>
            </div>
            """, unsafe_allow_html=True)
        else:
            st.markdown(f"""
            <div class="stat-box">
                <div class="stat-number">N/A</div>
                <div class="stat-label">Budget Weighted Average</div>
            </div>
            """, unsafe_allow_html=True)

    with col3:
        st.markdown(f"""
        <div class="stat-box">
            <div class="stat-number">{max_alignment:.1f}%</div>
            <div class="stat-label">Highest Alignment</div>
        </div>
        """, unsafe_allow_html=True)

    with col4:
        st.markdown(f"""
        <div class="stat-box">
            <div class="stat-number">{min_alignment:.1f}%</div>
            <div class="stat-label">Lowest Alignment</div>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("---")

    # Get department information for filtering
    dept_info_query = """
        SELECT project_id, responsible_organization
        FROM project
        WHERE portfolio_id = ? AND is_active = TRUE
    """
    dept_info_df = db.execute(dept_info_query, (portfolio_id,)).df()
    dept_info_df['responsible_organization'] = dept_info_df['responsible_organization'].fillna('Unassigned')

    # Filter to projects with scores
    scored_projects = alignment_df[alignment_df['alignment_score'].notna()].copy()

    # Merge department info with scored projects
    if not scored_projects.empty and not dept_info_df.empty:
        scored_projects = scored_projects.merge(dept_info_df, on='project_id', how='left')
        scored_projects['responsible_organization'] = scored_projects['responsible_organization'].fillna('Unassigned')

    # Project Alignment Scores in expander
    with st.expander("📊 Project Alignment Scores", expanded=False):
        # View Mode selector
        view_mode = st.radio(
            "View Mode:",
            options=["Project", "Department"],
            horizontal=True,
            key="alignment_scores_view_mode"
        )

        # Department filter (only for Project view)
        if view_mode == "Project" and not scored_projects.empty and 'responsible_organization' in scored_projects.columns:
            departments_proj = sorted(scored_projects['responsible_organization'].unique())
            dept_options_proj = ['All Departments'] + departments_proj

            selected_dept_proj = st.selectbox(
                "Filter by department:",
                options=dept_options_proj,
                key="dept_filter_proj_alignment"
            )

            # Filter by department
            if selected_dept_proj != 'All Departments':
                scored_projects_filtered = scored_projects[scored_projects['responsible_organization'] == selected_dept_proj].copy()
                st.info(f"📍 Showing data for: **{selected_dept_proj}**")
            else:
                scored_projects_filtered = scored_projects.copy()
        else:
            scored_projects_filtered = scored_projects.copy()

        # Handle Department view mode
        if view_mode == "Department":
            if not scored_projects.empty and 'responsible_organization' in scored_projects.columns:
                # Need budget information for weighting
                dept_alignment_query = """
                    SELECT
                        p.project_id,
                        p.current_budget
                    FROM project p
                    WHERE p.portfolio_id = ? AND p.is_active = TRUE
                """
                budget_df = db.execute(dept_alignment_query, (portfolio_id,)).df()

                if not budget_df.empty:
                    # Merge with scored projects
                    dept_data = scored_projects.merge(budget_df, on='project_id', how='left')

                    # Filter to projects with both budget and alignment scores
                    dept_data = dept_data[
                        (dept_data['current_budget'].notna()) &
                        (dept_data['current_budget'] > 0) &
                        (dept_data['alignment_score'].notna())
                    ]

                    if not dept_data.empty:
                        # Calculate budget-weighted alignment by department
                        dept_summary = dept_data.groupby('responsible_organization').apply(
                            lambda x: pd.Series({
                                'total_budget': x['current_budget'].sum(),
                                'weighted_alignment': (x['alignment_score'] * x['current_budget']).sum() / x['current_budget'].sum(),
                                'project_count': len(x),
                                'avg_factors_scored': x['factors_scored'].mean(),
                                'total_factors': x['total_factors'].iloc[0] if len(x) > 0 else 0
                            }),
                            include_groups=False
                        ).reset_index()

                        # Rename column for display
                        dept_summary.rename(columns={'responsible_organization': 'department'}, inplace=True)

                        # Sort by weighted alignment
                        dept_summary = dept_summary.sort_values('weighted_alignment', ascending=False)

                        # Create horizontal bar chart
                        fig = go.Figure()

                        # Color based on alignment level
                        colors = []
                        for score in dept_summary['weighted_alignment']:
                            if score >= 70:
                                colors.append('#27ae60')  # High - green
                            elif score >= 40:
                                colors.append('#f39c12')  # Medium - orange
                            else:
                                colors.append('#e74c3c')  # Low - red

                        fig.add_trace(go.Bar(
                            y=dept_summary['department'],
                            x=dept_summary['weighted_alignment'],
                            orientation='h',
                            marker=dict(
                                color=colors,
                                line=dict(color='white', width=2)
                            ),
                            text=[f"{s:.1f}%" for s in dept_summary['weighted_alignment']],
                            textposition='outside',
                            hovertemplate='<b>%{y}</b><br>' +
                                         'Budget-Weighted Alignment: %{x:.1f}%<br>' +
                                         'Projects: %{customdata[0]}<br>' +
                                         'Total Budget: $%{customdata[1]:,.0f}<extra></extra>',
                            customdata=dept_summary[['project_count', 'total_budget']].values
                        ))

                        fig.update_layout(
                            xaxis_title="Budget-Weighted Strategic Alignment Score (%)",
                            yaxis_title="",
                            height=max(400, len(dept_summary) * 50),
                            showlegend=False,
                            yaxis=dict(autorange="reversed"),
                            xaxis=dict(range=[0, 100])
                        )

                        st.plotly_chart(fig, width='stretch')

                        # Display department summary cards
                        st.markdown("---")
                        st.markdown("### 📋 Department Details")

                        for _, dept in dept_summary.iterrows():
                            score = dept['weighted_alignment']

                            # Determine alignment level
                            if score >= 70:
                                card_class = "high"
                                badge_class = "badge-high"
                                status = "🟢 High Alignment"
                            elif score >= 40:
                                card_class = "medium"
                                badge_class = "badge-medium"
                                status = "🟡 Medium Alignment"
                            else:
                                card_class = "low"
                                badge_class = "badge-low"
                                status = "🔴 Low Alignment"

                            st.markdown(f'<div class="alignment-card {card_class}">', unsafe_allow_html=True)

                            col_name, col_score = st.columns([3, 1])

                            with col_name:
                                st.markdown(f"**{dept['department']}**")
                                st.caption(f"Projects: {int(dept['project_count'])} | Budget: ${dept['total_budget']:,.0f}")

                            with col_score:
                                st.markdown(f'<div class="score-badge {badge_class}">{score:.1f}%</div>', unsafe_allow_html=True)

                            st.markdown('</div>', unsafe_allow_html=True)

                        st.info("💡 Department scores are budget-weighted: Σ(Alignment Score × Budget) / Σ(Budget)")

                    else:
                        st.warning("No departments have projects with both budget and alignment scores.")
                else:
                    st.warning("Budget data not available for departments.")
            else:
                st.warning("Organization data not available.")

        elif not scored_projects_filtered.empty:
            # Create horizontal bar chart
            fig = go.Figure()

            # Color based on alignment level
            colors = []
            for score in scored_projects_filtered['alignment_score']:
                if score >= 70:
                    colors.append('#27ae60')  # High - green
                elif score >= 40:
                    colors.append('#f39c12')  # Medium - orange
                else:
                    colors.append('#e74c3c')  # Low - red

            fig.add_trace(go.Bar(
                y=scored_projects_filtered['project_name'],
                x=scored_projects_filtered['alignment_score'],
                orientation='h',
                marker=dict(
                    color=colors,
                    line=dict(color='white', width=2)
                ),
                text=[f"{s:.1f}%" for s in scored_projects_filtered['alignment_score']],
                textposition='outside',
                hovertemplate='<b>%{y}</b><br>' +
                             'Alignment: %{x:.1f}%<br>' +
                             'Factors Scored: %{customdata[0]}/%{customdata[1]}<extra></extra>',
                customdata=scored_projects_filtered[['factors_scored', 'total_factors']].values
            ))

            fig.update_layout(
                xaxis_title="Strategic Alignment Score (%)",
                yaxis_title="",
                height=max(400, len(scored_projects_filtered) * 40),
                showlegend=False,
                yaxis=dict(autorange="reversed"),
                xaxis=dict(range=[0, 100])
            )

            st.plotly_chart(fig, width='stretch')

            # Display project cards
            st.markdown("---")
            st.markdown("### 📋 Project Details")

            for _, project in scored_projects_filtered.iterrows():
                score = project['alignment_score']

                # Determine alignment level
                if score >= 70:
                    card_class = "high"
                    badge_class = "badge-high"
                    status = "🟢 High Alignment"
                elif score >= 40:
                    card_class = "medium"
                    badge_class = "badge-medium"
                    status = "🟡 Medium Alignment"
                else:
                    card_class = "low"
                    badge_class = "badge-low"
                    status = "🔴 Low Alignment"

                st.markdown(f'<div class="alignment-card {card_class}">', unsafe_allow_html=True)

                col_name, col_score = st.columns([3, 1])

                with col_name:
                    st.markdown(f"**{project['project_name']}**")
                    st.caption(f"Scored: {int(project['factors_scored'])} of {int(project['total_factors'])} factors")

                with col_score:
                    st.markdown(f'<div class="score-badge {badge_class}">{score:.1f}%</div>', unsafe_allow_html=True)

                st.markdown('</div>', unsafe_allow_html=True)

        else:
            st.info("No projects have been scored yet.")

    # Budget Weighted Alignment by Department
    with st.expander("💰 Budget Weighted Alignment by Department", expanded=False):
        # Get alignment data with budget, department, and duration data
        dept_alignment_query = """
            SELECT
                p.responsible_organization as department,
                p.current_budget,
                p.project_name,
                p.project_id,
                psr.actual_duration_months,
                psr.original_duration_months
            FROM project p
            LEFT JOIN project_status_report psr ON p.project_id = psr.project_id
                AND psr.portfolio_id = ?
                AND psr.status_date = (
                    SELECT MAX(status_date)
                    FROM project_status_report
                    WHERE project_id = p.project_id AND portfolio_id = ?
                )
            WHERE p.portfolio_id = ? AND p.is_active = TRUE
        """

        dept_df = db.execute(dept_alignment_query, (portfolio_id, portfolio_id, portfolio_id)).df()

        if not dept_df.empty and not scored_projects.empty:
            # Merge with alignment scores
            dept_df = dept_df.merge(
                scored_projects[['project_id', 'alignment_score']],
                on='project_id',
                how='left'
            )

            # Filter to projects with budget, alignment scores, and duration data
            dept_df = dept_df[
                (dept_df['current_budget'].notna()) &
                (dept_df['current_budget'] > 0) &
                (dept_df['alignment_score'].notna()) &
                (dept_df['actual_duration_months'].notna()) &
                (dept_df['original_duration_months'].notna()) &
                (dept_df['original_duration_months'] > 0)
            ]

            if not dept_df.empty:
                # Fill null departments
                dept_df['department'] = dept_df['department'].fillna('Unassigned')

                # Calculate budget-weighted metrics by department
                dept_summary = dept_df.groupby('department').apply(
                    lambda x: pd.Series({
                        'total_budget': x['current_budget'].sum(),
                        'weighted_alignment': (x['alignment_score'] * x['current_budget']).sum() / x['current_budget'].sum(),
                        'weighted_time_used': (x['actual_duration_months'] * x['current_budget']).sum() / (x['original_duration_months'] * x['current_budget']).sum(),
                        'project_count': len(x)
                    }),
                    include_groups=False
                ).reset_index()

                # Convert weighted_time_used to percentage
                dept_summary['weighted_time_used_pct'] = dept_summary['weighted_time_used'] * 100

                # Create scatter plot with bubbles
                fig_dept = go.Figure()

                # Calculate bubble sizes based on budget
                total_max_budget = dept_summary['total_budget'].max()
                min_size = 15
                max_size = 60
                sizes = [min_size + (max_size - min_size) * (budget / total_max_budget) for budget in dept_summary['total_budget']]

                # Generate distinct colors for departments
                import plotly.colors as pc
                dept_colors = pc.qualitative.Plotly[:len(dept_summary)]
                if len(dept_summary) > len(dept_colors):
                    dept_colors = dept_colors * (len(dept_summary) // len(dept_colors) + 1)

                # Add each department as a separate trace
                for i, (idx, dept) in enumerate(dept_summary.iterrows()):
                    dept_name = dept['department']
                    color = dept_colors[i % len(dept_colors)]

                    # Create hover text
                    hover_text = (
                        f"<b>{dept_name}</b><br>" +
                        f"Projects: {int(dept['project_count'])}<br>" +
                        f"Total Budget: ${dept['total_budget']:,.0f}<br>" +
                        f"Weighted Alignment: {dept['weighted_alignment']:.1f}%<br>" +
                        f"Weighted % Time Used: {dept['weighted_time_used_pct']:.1f}%"
                    )

                    fig_dept.add_trace(go.Scatter(
                        x=[dept['weighted_time_used_pct']],
                        y=[dept['weighted_alignment']],
                        mode='markers',
                        name=dept_name,
                        marker=dict(
                            color=color,
                            size=sizes[i],
                            opacity=0.7,
                            line=dict(color='black', width=2)
                        ),
                        hovertemplate='%{hovertext}<extra></extra>',
                        hovertext=[hover_text]
                    ))

                # Add reference lines
                fig_dept.add_hline(y=70, line_dash="dash", line_color="green", opacity=0.5,
                                  annotation_text="High Alignment Threshold", annotation_position="right")
                fig_dept.add_hline(y=40, line_dash="dash", line_color="orange", opacity=0.5,
                                  annotation_text="Medium Alignment Threshold", annotation_position="right")
                fig_dept.add_vline(x=100, line_dash="dash", line_color="gray", opacity=0.5,
                                  annotation_text="Schedule Baseline", annotation_position="top")

                fig_dept.update_layout(
                    title="Department Performance: Strategic Alignment vs Time Performance",
                    xaxis_title="Weighted % Time Used (AD/OD)",
                    yaxis_title="Budget-Weighted Strategic Alignment Score (%)",
                    height=600,
                    showlegend=True,
                    legend=dict(
                        orientation="v",
                        yanchor="top",
                        y=1,
                        xanchor="left",
                        x=1.02
                    ),
                    xaxis=dict(range=[0, max(130, dept_summary['weighted_time_used_pct'].max() * 1.1)]),
                    yaxis=dict(range=[0, 100])
                )

                st.plotly_chart(fig_dept, width='stretch')

                # Add interpretation guide
                st.markdown("""
                **📊 Chart Interpretation:**
                - **X-axis**: % Time Used (AD/OD) - 100% = on schedule, <100% = ahead, >100% = delayed
                - **Y-axis**: Strategic Alignment Score - Higher is better aligned with strategic goals
                - **Bubble Size**: Proportional to department's total budget
                - **Target Zone**: Upper left area (high alignment, on/ahead of schedule)
                - **Green Line**: High alignment threshold (≥70%)
                - **Orange Line**: Medium alignment threshold (≥40%)
                """)

                st.info("💡 All metrics are budget-weighted: Σ(Metric × Budget) / Σ(Budget)")

                # Display department details
                st.markdown("---")
                st.markdown("### 📋 Department Details")

                for _, dept in dept_summary.iterrows():
                    col1, col2, col3, col4 = st.columns([3, 2, 2, 2])

                    with col1:
                        st.markdown(f"**{dept['department']}**")
                        st.caption(f"Projects: {int(dept['project_count'])}")

                    with col2:
                        st.metric("Budget", f"${dept['total_budget']:,.0f}")

                    with col3:
                        score = dept['weighted_alignment']
                        if score >= 70:
                            badge_class = "badge-high"
                        elif score >= 40:
                            badge_class = "badge-medium"
                        else:
                            badge_class = "badge-low"
                        st.markdown(f'<div class="score-badge {badge_class}">{score:.1f}%</div>', unsafe_allow_html=True)

                    with col4:
                        time_pct = dept['weighted_time_used_pct']
                        st.metric("% Time Used", f"{time_pct:.1f}%")
            else:
                st.info("No departments have projects with budget, alignment scores, and duration data.")
        else:
            st.info("No department/budget data available or no projects scored yet.")

    # Budget Weighted Alignment by Tier
    with st.expander("📊 Budget Weighted Alignment by Tier", expanded=False):
        # Tier type selector
        tier_type = st.radio(
            "Select Tier Type:",
            options=["Budget Tier", "Duration Tier"],
            horizontal=True,
            key="alignment_tier_type_selector"
        )

        # Get alignment data with budget, tier info, and duration data
        tier_alignment_query = """
            SELECT
                p.project_id,
                p.current_budget,
                p.project_name,
                psr.actual_duration_months,
                psr.original_duration_months
            FROM project p
            LEFT JOIN project_status_report psr ON p.project_id = psr.project_id
                AND psr.portfolio_id = ?
                AND psr.status_date = (
                    SELECT MAX(status_date)
                    FROM project_status_report
                    WHERE project_id = p.project_id AND portfolio_id = ?
                )
            WHERE p.portfolio_id = ? AND p.is_active = TRUE
        """

        tier_df = db.execute(tier_alignment_query, (portfolio_id, portfolio_id, portfolio_id)).df()

        if not tier_df.empty and not scored_projects.empty:
            # Merge with alignment scores
            tier_df = tier_df.merge(
                scored_projects[['project_id', 'alignment_score']],
                on='project_id',
                how='left'
            )

            # Filter to projects with budget, alignment scores, and duration data
            tier_df = tier_df[
                (tier_df['current_budget'].notna()) &
                (tier_df['current_budget'] > 0) &
                (tier_df['alignment_score'].notna()) &
                (tier_df['actual_duration_months'].notna()) &
                (tier_df['original_duration_months'].notna()) &
                (tier_df['original_duration_months'] > 0)
            ]

            if not tier_df.empty:
                # Get tier configuration based on selected mode
                if tier_type == "Budget Tier":
                    tier_config = st.session_state.get('config_dict', {}).get('controls', {}).get('tier_config', {})
                    default_tier_config = {
                        'cutoff_points': [4000, 8000, 15000],
                        'tier_names': ['Tier 1', 'Tier 2', 'Tier 3', 'Tier 4'],
                        'colors': ['#3498db', '#27ae60', '#f39c12', '#e74c3c']
                    }
                    cutoff_points = tier_config.get('cutoff_points', default_tier_config['cutoff_points'])
                    tier_names = tier_config.get('tier_names', default_tier_config['tier_names'])
                    tier_colors = tier_config.get('colors', default_tier_config['colors'])

                    # Function to determine tier based on budget
                    def get_tier(row):
                        budget = row['current_budget']
                        if budget <= cutoff_points[0]:
                            return tier_names[0]
                        elif budget <= cutoff_points[1]:
                            return tier_names[1]
                        elif budget <= cutoff_points[2]:
                            return tier_names[2]
                        else:
                            return tier_names[3]

                    tier_df['tier'] = tier_df.apply(get_tier, axis=1)

                else:  # Duration Tier
                    tier_config = st.session_state.get('config_dict', {}).get('controls', {}).get('duration_tier_config', {})
                    default_tier_config = {
                        'cutoff_points': [6, 12, 24],
                        'tier_names': ['Short', 'Medium', 'Long', 'Extra Long'],
                        'colors': ['#3498db', '#27ae60', '#f39c12', '#e74c3c']
                    }
                    cutoff_points = tier_config.get('cutoff_points', default_tier_config['cutoff_points'])
                    tier_names = tier_config.get('tier_names', default_tier_config['tier_names'])
                    tier_colors = tier_config.get('colors', default_tier_config['colors'])

                    # Function to determine tier based on duration
                    def get_tier(row):
                        duration = row['original_duration_months']
                        if pd.isna(duration):
                            return "Unknown"
                        duration = int(duration)
                        if duration >= cutoff_points[2]:
                            return tier_names[3]
                        elif duration >= cutoff_points[1]:
                            return tier_names[2]
                        elif duration >= cutoff_points[0]:
                            return tier_names[1]
                        else:
                            return tier_names[0]

                    tier_df['tier'] = tier_df.apply(get_tier, axis=1)

                # Calculate budget-weighted metrics by tier
                tier_summary = tier_df.groupby('tier').apply(
                    lambda x: pd.Series({
                        'total_budget': x['current_budget'].sum(),
                        'weighted_alignment': (x['alignment_score'] * x['current_budget']).sum() / x['current_budget'].sum(),
                        'weighted_time_used': (x['actual_duration_months'] * x['current_budget']).sum() / (x['original_duration_months'] * x['current_budget']).sum(),
                        'project_count': len(x)
                    }),
                    include_groups=False
                ).reset_index()

                # Convert weighted_time_used to percentage
                tier_summary['weighted_time_used_pct'] = tier_summary['weighted_time_used'] * 100

                # Filter out any "Unknown" tiers
                tier_summary = tier_summary[tier_summary['tier'] != 'Unknown']

                if not tier_summary.empty:
                    # Create scatter plot with bubbles
                    fig_tier = go.Figure()

                    # Calculate bubble sizes based on budget
                    total_max_budget = tier_summary['total_budget'].max()
                    min_size = 15
                    max_size = 60
                    sizes = [min_size + (max_size - min_size) * (budget / total_max_budget) for budget in tier_summary['total_budget']]

                    # Create color map for tiers
                    tier_color_map = {tier_names[i]: tier_colors[i] for i in range(len(tier_names))}

                    # Add each tier as a separate trace
                    for i, (idx, tier) in enumerate(tier_summary.iterrows()):
                        tier_name = tier['tier']
                        color = tier_color_map.get(tier_name, '#888888')

                        # Create hover text
                        hover_text = (
                            f"<b>{tier_name}</b><br>" +
                            f"Projects: {int(tier['project_count'])}<br>" +
                            f"Total Budget: ${tier['total_budget']:,.0f}<br>" +
                            f"Weighted Alignment: {tier['weighted_alignment']:.1f}%<br>" +
                            f"Weighted % Time Used: {tier['weighted_time_used_pct']:.1f}%"
                        )

                        fig_tier.add_trace(go.Scatter(
                            x=[tier['weighted_time_used_pct']],
                            y=[tier['weighted_alignment']],
                            mode='markers',
                            name=tier_name,
                            marker=dict(
                                color=color,
                                size=sizes[i],
                                opacity=0.7,
                                line=dict(color='black', width=2)
                            ),
                            hovertemplate='%{hovertext}<extra></extra>',
                            hovertext=[hover_text]
                        ))

                    # Add reference lines
                    fig_tier.add_hline(y=70, line_dash="dash", line_color="green", opacity=0.5,
                                      annotation_text="High Alignment Threshold", annotation_position="right")
                    fig_tier.add_hline(y=40, line_dash="dash", line_color="orange", opacity=0.5,
                                      annotation_text="Medium Alignment Threshold", annotation_position="right")
                    fig_tier.add_vline(x=100, line_dash="dash", line_color="gray", opacity=0.5,
                                      annotation_text="Schedule Baseline", annotation_position="top")

                    # Set title based on tier type
                    title_text = f"{'Budget' if tier_type == 'Budget Tier' else 'Duration'} Tier Performance: Strategic Alignment vs Time Performance"

                    fig_tier.update_layout(
                        title=title_text,
                        xaxis_title="Weighted % Time Used (AD/OD)",
                        yaxis_title="Budget-Weighted Strategic Alignment Score (%)",
                        height=600,
                        showlegend=True,
                        legend=dict(
                            orientation="v",
                            yanchor="top",
                            y=1,
                            xanchor="left",
                            x=1.02
                        ),
                        xaxis=dict(range=[0, max(130, tier_summary['weighted_time_used_pct'].max() * 1.1)]),
                        yaxis=dict(range=[0, 100])
                    )

                    st.plotly_chart(fig_tier, width='stretch')

                    # Add interpretation guide
                    st.markdown("""
                    **📊 Chart Interpretation:**
                    - **X-axis**: % Time Used (AD/OD) - 100% = on schedule, <100% = ahead, >100% = delayed
                    - **Y-axis**: Strategic Alignment Score - Higher is better aligned with strategic goals
                    - **Bubble Size**: Proportional to tier's total budget
                    - **Target Zone**: Upper left area (high alignment, on/ahead of schedule)
                    - **Green Line**: High alignment threshold (≥70%)
                    - **Orange Line**: Medium alignment threshold (≥40%)
                    """)

                    st.info("💡 All metrics are budget-weighted: Σ(Metric × Budget) / Σ(Budget)")

                    # Display tier details
                    st.markdown("---")
                    st.markdown("### 📋 Tier Details")

                    for _, tier_row in tier_summary.iterrows():
                        col1, col2, col3, col4 = st.columns([3, 2, 2, 2])

                        with col1:
                            st.markdown(f"**{tier_row['tier']}**")
                            st.caption(f"Projects: {int(tier_row['project_count'])}")

                        with col2:
                            st.metric("Budget", f"${tier_row['total_budget']:,.0f}")

                        with col3:
                            score = tier_row['weighted_alignment']
                            if score >= 70:
                                badge_class = "badge-high"
                            elif score >= 40:
                                badge_class = "badge-medium"
                            else:
                                badge_class = "badge-low"
                            st.markdown(f'<div class="score-badge {badge_class}">{score:.1f}%</div>', unsafe_allow_html=True)

                        with col4:
                            time_pct = tier_row['weighted_time_used_pct']
                            st.metric("% Time Used", f"{time_pct:.1f}%")

                    # Add tier legend
                    st.markdown("---")
                    if tier_type == "Budget Tier":
                        # Get currency symbol
                        currency_symbol = st.session_state.get('config_dict', {}).get('controls', {}).get('currency_symbol', '$')

                        st.markdown("**🎨 Tier Ranges:**")
                        legend_cols = st.columns(len(tier_names))
                        for i, (tier_name, tier_color) in enumerate(zip(tier_names, tier_colors)):
                            with legend_cols[i]:
                                if i == 0:
                                    range_text = f"< {currency_symbol}{cutoff_points[0]:,.0f}"
                                elif i == len(tier_names) - 1:
                                    range_text = f"≥ {currency_symbol}{cutoff_points[i-1]:,.0f}"
                                else:
                                    range_text = f"{currency_symbol}{cutoff_points[i-1]:,.0f}-{currency_symbol}{cutoff_points[i]:,.0f}"

                                st.markdown(
                                    f'<div style="background-color: {tier_color}; padding: 10px; border-radius: 5px; text-align: center; color: white; font-weight: bold;">'
                                    f'{tier_name}<br><span style="font-size: 0.8em;">{range_text}</span>'
                                    f'</div>',
                                    unsafe_allow_html=True
                                )
                    else:  # Duration Tier
                        st.markdown("**🎨 Tier Ranges:**")
                        legend_cols = st.columns(len(tier_names))
                        for i, (tier_name, tier_color) in enumerate(zip(tier_names, tier_colors)):
                            with legend_cols[i]:
                                if i == 0:
                                    range_text = f"< {cutoff_points[0]} mo"
                                elif i == len(tier_names) - 1:
                                    range_text = f"≥ {cutoff_points[i-1]} mo"
                                else:
                                    range_text = f"{cutoff_points[i-1]}-{cutoff_points[i]} mo"

                                st.markdown(
                                    f'<div style="background-color: {tier_color}; padding: 10px; border-radius: 5px; text-align: center; color: white; font-weight: bold;">'
                                    f'{tier_name}<br><span style="font-size: 0.8em;">{range_text}</span>'
                                    f'</div>',
                                    unsafe_allow_html=True
                                )
                else:
                    st.warning("No tiers have sufficient data for analysis.")
            else:
                st.info("No projects have budget, alignment scores, and duration data.")
        else:
            st.info("No data available or no projects scored yet.")

    # Alignment Score Histogram
    with st.expander("📊 Alignment Score Distribution", expanded=False):
        if not scored_projects.empty:
            # Department filter
            if 'responsible_organization' in scored_projects.columns:
                col_metric, col_dept_hist = st.columns(2)

                with col_metric:
                    # Toggle between frequency and budget
                    show_metric = st.radio(
                        "Show histogram by:",
                        ["Frequency (Number of Projects)", "Budget (Sum of Project Budgets)"],
                        horizontal=True,
                        key="hist_metric_toggle"
                    )

                with col_dept_hist:
                    departments_hist = sorted(scored_projects['responsible_organization'].unique())
                    dept_options_hist = ['All Departments'] + departments_hist

                    selected_dept_hist = st.selectbox(
                        "Filter by department:",
                        options=dept_options_hist,
                        key="dept_filter_histogram"
                    )

                # Filter scored_projects by department
                if selected_dept_hist != 'All Departments':
                    scored_projects_hist = scored_projects[scored_projects['responsible_organization'] == selected_dept_hist].copy()
                    st.info(f"📍 Showing data for: **{selected_dept_hist}**")
                else:
                    scored_projects_hist = scored_projects.copy()
            else:
                scored_projects_hist = scored_projects.copy()
                # Toggle between frequency and budget (fallback if no department column)
                show_metric = st.radio(
                    "Show histogram by:",
                    ["Frequency (Number of Projects)", "Budget (Sum of Project Budgets)"],
                    horizontal=True,
                    key="hist_metric_toggle_nodept"
                )

            # Get budget data for scored projects
            budget_query = """
                SELECT project_id, current_budget
                FROM project
                WHERE portfolio_id = ? AND is_active = TRUE
            """
            budget_df = db.execute(budget_query, (portfolio_id,)).df()

            # Merge budget with scored projects
            hist_df = scored_projects_hist.merge(budget_df, on='project_id', how='left')
            hist_df['current_budget'] = hist_df['current_budget'].fillna(0)

            # Create bins (0-100 in 10 bins)
            bins = list(range(0, 101, 10))
            bin_labels = [f"{bins[i]}-{bins[i+1]}" for i in range(len(bins)-1)]

            # Categorize scores into bins
            hist_df['score_bin'] = pd.cut(
                hist_df['alignment_score'],
                bins=bins,
                labels=bin_labels,
                include_lowest=True
            )

            if show_metric == "Frequency (Number of Projects)":
                # Count projects in each bin
                bin_counts = hist_df.groupby('score_bin', observed=True).size().reset_index(name='count')

                # Ensure all bins are present
                bin_counts_full = pd.DataFrame({'score_bin': bin_labels})
                bin_counts_full = bin_counts_full.merge(bin_counts, on='score_bin', how='left')
                bin_counts_full['count'] = bin_counts_full['count'].fillna(0)

                fig_hist = go.Figure()
                fig_hist.add_trace(go.Bar(
                    x=bin_counts_full['score_bin'],
                    y=bin_counts_full['count'],
                    marker=dict(
                        color='#9b59b6',
                        line=dict(color='white', width=2)
                    ),
                    text=bin_counts_full['count'].astype(int),
                    textposition='outside',
                    hovertemplate='<b>%{x}%</b><br>Projects: %{y}<extra></extra>'
                ))

                fig_hist.update_layout(
                    xaxis_title="Alignment Score Range (%)",
                    yaxis_title="Number of Projects",
                    height=400,
                    showlegend=False
                )

            else:  # Budget
                # Sum budget in each bin
                bin_budgets = hist_df.groupby('score_bin', observed=True)['current_budget'].sum().reset_index(name='total_budget')

                # Ensure all bins are present
                bin_budgets_full = pd.DataFrame({'score_bin': bin_labels})
                bin_budgets_full = bin_budgets_full.merge(bin_budgets, on='score_bin', how='left')
                bin_budgets_full['total_budget'] = bin_budgets_full['total_budget'].fillna(0)

                fig_hist = go.Figure()
                fig_hist.add_trace(go.Bar(
                    x=bin_budgets_full['score_bin'],
                    y=bin_budgets_full['total_budget'],
                    marker=dict(
                        color='#3498db',
                        line=dict(color='white', width=2)
                    ),
                    text=[f"${b:,.0f}" for b in bin_budgets_full['total_budget']],
                    textposition='outside',
                    hovertemplate='<b>%{x}%</b><br>Total Budget: $%{y:,.0f}<extra></extra>'
                ))

                fig_hist.update_layout(
                    xaxis_title="Alignment Score Range (%)",
                    yaxis_title="Total Budget ($)",
                    height=400,
                    showlegend=False
                )

            st.plotly_chart(fig_hist, width='stretch')
        else:
            st.info("No alignment scores available for histogram.")

    # Average Score by Factor
    with st.expander("📈 Average Score by Factor", expanded=False):
        # Get all factors for this portfolio
        factors_df = db_manager.list_factors(portfolio_id)

        if not factors_df.empty:
            # Get all factor scores with project and budget info
            factor_scores_query = """
                SELECT
                    pfs.factor_id,
                    pfs.project_id,
                    pfs.score,
                    p.project_name,
                    p.current_budget,
                    p.responsible_organization
                FROM project_factor_score pfs
                JOIN project p ON pfs.project_id = p.project_id
                WHERE pfs.portfolio_id = ? AND p.is_active = TRUE
            """

            scores_data_raw = db.execute(factor_scores_query, (portfolio_id,)).df()

            # Initialize selected_dept
            selected_dept = 'All Departments'

            col_avg_type, col_dept = st.columns(2)

            with col_avg_type:
                # Toggle between simple and budget-weighted average
                avg_type = st.radio(
                    "Calculate average by:",
                    ["Simple Average", "Budget Weighted Average"],
                    horizontal=True
                )

            with col_dept:
                if not scores_data_raw.empty:
                    # Get unique departments (fill NA for display purposes only)
                    temp_df = scores_data_raw.copy()
                    temp_df['responsible_organization'] = temp_df['responsible_organization'].fillna('Unassigned')
                    departments = sorted(temp_df['responsible_organization'].unique())
                    dept_options = ['All Departments'] + departments

                    selected_dept = st.selectbox(
                        "Filter by department:",
                        options=dept_options
                    )

            # Process the data based on department selection
            scores_data = scores_data_raw.copy()
            scores_data['responsible_organization'] = scores_data['responsible_organization'].fillna('Unassigned')

            # Filter by department if not "All Departments"
            if selected_dept != 'All Departments' and not scores_data.empty:
                scores_data = scores_data[scores_data['responsible_organization'] == selected_dept]
                st.info(f"📍 Showing data for: **{selected_dept}**")

            if not scores_data.empty:
                # Merge with factor details to get likert ranges
                scores_data = scores_data.merge(
                    factors_df[['factor_id', 'factor_name', 'factor_weight_percent', 'likert_min', 'likert_max']],
                    on='factor_id',
                    how='inner'
                )

                if avg_type == "Simple Average":
                    # Calculate simple average for each factor
                    factor_avg = scores_data.groupby(['factor_id', 'factor_name', 'likert_min', 'likert_max', 'factor_weight_percent']).agg({
                        'score': 'mean',
                        'project_id': 'count'
                    }).reset_index()
                    factor_avg.rename(columns={'score': 'avg_score', 'project_id': 'project_count'}, inplace=True)

                else:  # Budget Weighted Average
                    # Filter to projects with budget
                    scores_with_budget = scores_data[
                        (scores_data['current_budget'].notna()) &
                        (scores_data['current_budget'] > 0)
                    ].copy()

                    if not scores_with_budget.empty:
                        # Calculate budget-weighted average for each factor
                        factor_avg = scores_with_budget.groupby(['factor_id', 'factor_name', 'likert_min', 'likert_max', 'factor_weight_percent']).apply(
                            lambda x: pd.Series({
                                'avg_score': (x['score'] * x['current_budget']).sum() / x['current_budget'].sum(),
                                'project_count': len(x),
                                'total_budget': x['current_budget'].sum()
                            }),
                            include_groups=False
                        ).reset_index()
                    else:
                        st.warning("No projects with budget information found. Showing simple average instead.")
                        factor_avg = scores_data.groupby(['factor_id', 'factor_name', 'likert_min', 'likert_max', 'factor_weight_percent']).agg({
                            'score': 'mean',
                            'project_id': 'count'
                        }).reset_index()
                        factor_avg.rename(columns={'score': 'avg_score', 'project_id': 'project_count'}, inplace=True)

                if not factor_avg.empty:
                    # Calculate normalized percentage (0-100%)
                    factor_avg['normalized_pct'] = (
                        (factor_avg['avg_score'] - factor_avg['likert_min']) /
                        (factor_avg['likert_max'] - factor_avg['likert_min'])
                    ) * 100

                    # Sort by average score (descending)
                    factor_avg = factor_avg.sort_values('normalized_pct', ascending=False)

                    # Create bar chart
                    fig_factor = go.Figure()

                    # Color based on normalized percentage
                    colors_factor = []
                    for pct in factor_avg['normalized_pct']:
                        if pct >= 70:
                            colors_factor.append('#27ae60')
                        elif pct >= 40:
                            colors_factor.append('#f39c12')
                        else:
                            colors_factor.append('#e74c3c')

                    # Create hover text
                    if avg_type == "Simple Average":
                        hover_template = (
                            '<b>%{y}</b><br>' +
                            'Average Score: %{customdata[0]:.2f} / %{customdata[1]}<br>' +
                            'Normalized: %{x:.1f}%<br>' +
                            'Weight: %{customdata[2]:.1f}%<br>' +
                            'Projects Scored: %{customdata[3]}<extra></extra>'
                        )
                        customdata = factor_avg[['avg_score', 'likert_max', 'factor_weight_percent', 'project_count']].values
                    else:
                        hover_template = (
                            '<b>%{y}</b><br>' +
                            'Weighted Avg Score: %{customdata[0]:.2f} / %{customdata[1]}<br>' +
                            'Normalized: %{x:.1f}%<br>' +
                            'Weight: %{customdata[2]:.1f}%<br>' +
                            'Projects: %{customdata[3]}<br>' +
                            'Total Budget: $%{customdata[4]:,.0f}<extra></extra>'
                        )
                        if 'total_budget' in factor_avg.columns:
                            customdata = factor_avg[['avg_score', 'likert_max', 'factor_weight_percent', 'project_count', 'total_budget']].values
                        else:
                            customdata = factor_avg[['avg_score', 'likert_max', 'factor_weight_percent', 'project_count']].values
                            hover_template = (
                                '<b>%{y}</b><br>' +
                                'Average Score: %{customdata[0]:.2f} / %{customdata[1]}<br>' +
                                'Normalized: %{x:.1f}%<br>' +
                                'Weight: %{customdata[2]:.1f}%<br>' +
                                'Projects Scored: %{customdata[3]}<extra></extra>'
                            )

                    fig_factor.add_trace(go.Bar(
                        y=factor_avg['factor_name'],
                        x=factor_avg['normalized_pct'],
                        orientation='h',
                        marker=dict(
                            color=colors_factor,
                            line=dict(color='white', width=2)
                        ),
                        text=[f"{s:.1f}%" for s in factor_avg['normalized_pct']],
                        textposition='outside',
                        hovertemplate=hover_template,
                        customdata=customdata
                    ))

                    fig_factor.update_layout(
                        xaxis_title="Normalized Average Score (%)",
                        yaxis_title="",
                        height=max(300, len(factor_avg) * 60),
                        showlegend=False,
                        yaxis=dict(autorange="reversed"),
                        xaxis=dict(range=[0, 100])
                    )

                    st.plotly_chart(fig_factor, width='stretch')

                    # Display factor details
                    st.markdown("---")
                    st.markdown("### 📋 Factor Details")

                    for _, factor in factor_avg.iterrows():
                        col1, col2, col3, col4 = st.columns([3, 2, 2, 2])

                        with col1:
                            st.markdown(f"**{factor['factor_name']}**")
                            st.caption(f"Weight: {factor['factor_weight_percent']:.1f}%")

                        with col2:
                            st.metric(
                                "Avg Score",
                                f"{factor['avg_score']:.2f} / {int(factor['likert_max'])}"
                            )

                        with col3:
                            st.metric(
                                "Projects",
                                int(factor['project_count'])
                            )

                        with col4:
                            pct = factor['normalized_pct']
                            if pct >= 70:
                                badge_class = "badge-high"
                            elif pct >= 40:
                                badge_class = "badge-medium"
                            else:
                                badge_class = "badge-low"
                            st.markdown(f'<div class="score-badge {badge_class}">{pct:.1f}%</div>', unsafe_allow_html=True)
                else:
                    st.info("No factor score data available.")
            else:
                st.info("No projects have been scored yet.")
        else:
            st.info("No factors defined for this portfolio.")

    # Performance vs Factor Score Analysis
    with st.expander("📊 Performance vs Factor Score Analysis", expanded=False):
        st.markdown("""
        This analysis shows how project performance metrics correlate with factor scores.
        All metrics are budget-weighted within each score range.
        """)

        # Get all factors for this portfolio
        factors_df = db_manager.list_factors(portfolio_id)

        if not factors_df.empty:
            # Get available status dates for the portfolio
            status_dates_query = """
                SELECT DISTINCT psr.status_date
                FROM project_status_report psr
                JOIN project p ON psr.project_id = p.project_id
                WHERE p.portfolio_id = ? AND p.is_active = TRUE
                ORDER BY psr.status_date DESC
            """
            status_dates_df = db.execute(status_dates_query, (portfolio_id,)).df()

            # Factor selector and controls
            factor_options = {row['factor_name']: row['factor_id'] for _, row in factors_df.iterrows()}

            col_factor, col_status, col_dept_perf = st.columns(3)

            with col_factor:
                selected_factor_name = st.selectbox(
                    "Select factor to analyze:",
                    options=list(factor_options.keys()),
                    key="perf_factor_selector"
                )
                selected_factor_id = factor_options[selected_factor_name]

            with col_status:
                if not status_dates_df.empty:
                    # Convert dates to strings for display
                    status_date_options = status_dates_df['status_date'].dt.strftime('%Y-%m-%d').tolist()

                    selected_status_date_str = st.selectbox(
                        "Select reporting period:",
                        options=status_date_options,
                        index=0,  # Default to latest (first in descending order)
                        key="perf_status_date_selector"
                    )

                    # Convert back to date for query
                    import datetime
                    selected_status_date = datetime.datetime.strptime(selected_status_date_str, '%Y-%m-%d').date()
                else:
                    selected_status_date = None
                    st.warning("No status reports found")

            # Get comprehensive project data with performance metrics for selected date
            if selected_status_date:
                perf_query = """
                    SELECT
                        p.project_id,
                        p.project_name,
                        p.responsible_organization,
                        p.current_budget,
                        p.planned_start_date,
                        p.planned_finish_date,
                        psr.actual_cost,
                        psr.earned_value,
                        psr.planned_value,
                        psr.status_date,
                        psr.spi,
                        psr.cpi,
                        psr.percent_time_used,
                        psr.percent_budget_used
                    FROM project p
                    LEFT JOIN project_status_report psr
                        ON p.project_id = psr.project_id
                        AND psr.status_date = ?
                    WHERE p.portfolio_id = ? AND p.is_active = TRUE
                """

                perf_df = db.execute(perf_query, (selected_status_date, portfolio_id)).df()
            else:
                perf_df = pd.DataFrame()

            with col_dept_perf:
                if not perf_df.empty:
                    # Get unique departments
                    perf_df['responsible_organization'] = perf_df['responsible_organization'].fillna('Unassigned')
                    departments_perf = sorted(perf_df['responsible_organization'].unique())
                    dept_options_perf = ['All Departments'] + departments_perf

                    selected_dept_perf = st.selectbox(
                        "Filter by department:",
                        options=dept_options_perf,
                        key="dept_filter_perf"
                    )

                    # Filter by department
                    if selected_dept_perf != 'All Departments':
                        perf_df = perf_df[perf_df['responsible_organization'] == selected_dept_perf].copy()
                        st.info(f"📍 Showing data for: **{selected_dept_perf}**")

            if not perf_df.empty:
                # Get factor scores for the selected factor
                factor_scores_perf_query = """
                    SELECT project_id, score
                    FROM project_factor_score
                    WHERE portfolio_id = ? AND factor_id = ?
                """

                factor_scores_perf = db.execute(factor_scores_perf_query, (portfolio_id, selected_factor_id)).df()

                if not factor_scores_perf.empty:
                    # Merge factor scores with performance data
                    analysis_df = perf_df.merge(factor_scores_perf, on='project_id', how='inner')

                    if not analysis_df.empty:
                        # Use pre-calculated performance metrics from project_status_report
                        # spi, cpi, percent_time_used, percent_budget_used are already in the dataframe

                        # Rename columns for consistency
                        analysis_df['pct_duration_used'] = analysis_df['percent_time_used']
                        analysis_df['pct_budget_used'] = analysis_df['percent_budget_used']

                        # Create score bins (0-10 with 10 bins: 0-1, 1-2, ..., 9-10)
                        # Get the selected factor's likert range
                        selected_factor_info = factors_df[factors_df['factor_id'] == selected_factor_id].iloc[0]
                        likert_min = int(selected_factor_info['likert_min'])
                        likert_max = int(selected_factor_info['likert_max'])

                        # Create bins based on the actual likert range
                        bin_size = (likert_max - likert_min) / 10
                        bins = [likert_min + i * bin_size for i in range(11)]
                        bin_labels = [f"{bins[i]:.1f}-{bins[i+1]:.1f}" for i in range(10)]

                        analysis_df['score_bin'] = pd.cut(
                            analysis_df['score'],
                            bins=bins,
                            labels=bin_labels,
                            include_lowest=True
                        )

                        # Calculate budget-weighted averages for each bin
                        summary_rows = []

                        for bin_label in bin_labels:
                            bin_data = analysis_df[analysis_df['score_bin'] == bin_label].copy()

                            if not bin_data.empty:
                                # Filter to projects with budget for weighting
                                bin_data_with_budget = bin_data[
                                    (bin_data['current_budget'].notna()) &
                                    (bin_data['current_budget'] > 0)
                                ]

                                if not bin_data_with_budget.empty:
                                    total_budget = bin_data_with_budget['current_budget'].sum()

                                    # Budget-weighted SPI (filter out NaN and infinity)
                                    spi_data = bin_data_with_budget[
                                        bin_data_with_budget['spi'].notna() &
                                        np.isfinite(bin_data_with_budget['spi'])
                                    ]
                                    if not spi_data.empty:
                                        weighted_spi = (spi_data['spi'] * spi_data['current_budget']).sum() / spi_data['current_budget'].sum()
                                    else:
                                        weighted_spi = None

                                    # Budget-weighted CPI (filter out NaN and infinity)
                                    cpi_data = bin_data_with_budget[
                                        bin_data_with_budget['cpi'].notna() &
                                        np.isfinite(bin_data_with_budget['cpi'])
                                    ]
                                    if not cpi_data.empty:
                                        weighted_cpi = (cpi_data['cpi'] * cpi_data['current_budget']).sum() / cpi_data['current_budget'].sum()
                                    else:
                                        weighted_cpi = None

                                    # Budget-weighted % Duration Used
                                    duration_data = bin_data_with_budget[bin_data_with_budget['pct_duration_used'].notna()]
                                    if not duration_data.empty:
                                        weighted_duration = (duration_data['pct_duration_used'] * duration_data['current_budget']).sum() / duration_data['current_budget'].sum()
                                    else:
                                        weighted_duration = None

                                    # Budget-weighted % Budget Used
                                    budget_used_data = bin_data_with_budget[bin_data_with_budget['pct_budget_used'].notna()]
                                    if not budget_used_data.empty:
                                        weighted_budget_used = (budget_used_data['pct_budget_used'] * budget_used_data['current_budget']).sum() / budget_used_data['current_budget'].sum()
                                    else:
                                        weighted_budget_used = None

                                    summary_rows.append({
                                        'Score Range': bin_label,
                                        'Project Count': len(bin_data),
                                        'Total Budget': total_budget,
                                        'Avg SPI': weighted_spi,
                                        'Avg CPI': weighted_cpi,
                                        'Avg % Duration Used': weighted_duration,
                                        'Avg % Budget Used': weighted_budget_used
                                    })

                        if summary_rows:
                            summary_df = pd.DataFrame(summary_rows)

                            # Format the dataframe for display
                            display_df = summary_df.copy()
                            display_df['Total Budget'] = display_df['Total Budget'].apply(lambda x: f"${x:,.0f}" if pd.notna(x) else "N/A")
                            display_df['Avg SPI'] = display_df['Avg SPI'].apply(lambda x: f"{x:.2f}" if pd.notna(x) else "N/A")
                            display_df['Avg CPI'] = display_df['Avg CPI'].apply(lambda x: f"{x:.2f}" if pd.notna(x) else "N/A")
                            display_df['Avg % Duration Used'] = display_df['Avg % Duration Used'].apply(lambda x: f"{x:.1f}%" if pd.notna(x) else "N/A")
                            display_df['Avg % Budget Used'] = display_df['Avg % Budget Used'].apply(lambda x: f"{x:.1f}%" if pd.notna(x) else "N/A")

                            st.markdown(f"### Performance Analysis for: **{selected_factor_name}**")
                            st.markdown(f"*As of: {selected_status_date_str} | All averages are budget-weighted*")

                            # Display as table
                            st.dataframe(
                                display_df,
                                width='stretch',
                                hide_index=True
                            )

                            # Add explanatory notes
                            st.markdown("---")
                            st.markdown(f"""
                            **Metric Definitions (as of {selected_status_date_str}):**
                            - **SPI (Schedule Performance Index)**: Earned Value / Planned Value. >1.0 is ahead of schedule.
                            - **CPI (Cost Performance Index)**: Earned Value / Actual Cost. >1.0 is under budget.
                            - **% Duration Used**: Percentage of project timeline elapsed as of the selected reporting date.
                            - **% Budget Used**: Percentage of budget consumed as of the selected reporting date.

                            **Budget Weighting**: Larger projects have proportionally more influence on the averages.
                            """)
                        else:
                            st.info("No data available for the selected factor and department combination.")
                    else:
                        st.info("No projects have both performance data and factor scores.")
                else:
                    st.info(f"No projects have been scored for factor: {selected_factor_name}")
            else:
                st.info("No project data available.")
        else:
            st.info("No factors defined for this portfolio.")

    # Performance vs Factor Score Trend
    with st.expander("📈 Performance vs Factor Score Trend", expanded=False):
        st.markdown("""
        This analysis shows how performance metrics trend over time across different factor score ranges.
        Select a metric, factor, and department to visualize performance trends.
        """)

        # Get all factors for this portfolio
        factors_df = db_manager.list_factors(portfolio_id)

        if not factors_df.empty:
            # Controls
            factor_options = {row['factor_name']: row['factor_id'] for _, row in factors_df.iterrows()}

            col_factor_trend, col_metric_trend, col_dept_trend = st.columns(3)

            with col_factor_trend:
                selected_factor_name_trend = st.selectbox(
                    "Select factor to analyze:",
                    options=list(factor_options.keys()),
                    key="trend_factor_selector"
                )
                selected_factor_id_trend = factor_options[selected_factor_name_trend]

            with col_metric_trend:
                metric_options = {
                    "Average SPI": "spi",
                    "Average CPI": "cpi",
                    "Average % Duration Used": "pct_duration_used",
                    "Average % Budget Used": "pct_budget_used"
                }
                selected_metric_display = st.selectbox(
                    "Select metric:",
                    options=list(metric_options.keys()),
                    key="trend_metric_selector"
                )
                selected_metric_trend = metric_options[selected_metric_display]

            # Get available status dates for the portfolio
            status_dates_query_trend = """
                SELECT DISTINCT psr.status_date
                FROM project_status_report psr
                JOIN project p ON psr.project_id = p.project_id
                WHERE p.portfolio_id = ? AND p.is_active = TRUE
                ORDER BY psr.status_date ASC
            """
            status_dates_trend_df = db.execute(status_dates_query_trend, (portfolio_id,)).df()

            if not status_dates_trend_df.empty:
                # Get project data
                projects_query = """
                    SELECT project_id, responsible_organization
                    FROM project
                    WHERE portfolio_id = ? AND is_active = TRUE
                """
                projects_trend_df = db.execute(projects_query, (portfolio_id,)).df()
                projects_trend_df['responsible_organization'] = projects_trend_df['responsible_organization'].fillna('Unassigned')

                with col_dept_trend:
                    departments_trend = sorted(projects_trend_df['responsible_organization'].unique())
                    dept_options_trend = ['All Departments'] + departments_trend

                    selected_dept_trend = st.selectbox(
                        "Filter by department:",
                        options=dept_options_trend,
                        key="dept_filter_trend"
                    )

                # Get the selected factor's likert range for binning
                selected_factor_info_trend = factors_df[factors_df['factor_id'] == selected_factor_id_trend].iloc[0]
                likert_min_trend = int(selected_factor_info_trend['likert_min'])
                likert_max_trend = int(selected_factor_info_trend['likert_max'])

                # Create bins
                bin_size_trend = (likert_max_trend - likert_min_trend) / 10
                bins_trend = [likert_min_trend + i * bin_size_trend for i in range(11)]
                bin_labels_trend = [f"{bins_trend[i]:.1f}-{bins_trend[i+1]:.1f}" for i in range(10)]

                # Collect data for all status dates
                trend_data = []

                for status_date_row in status_dates_trend_df.itertuples():
                    status_date_trend = status_date_row.status_date

                    # Get performance data for this date
                    perf_query_trend = """
                        SELECT
                            p.project_id,
                            p.responsible_organization,
                            p.current_budget,
                            psr.spi,
                            psr.cpi,
                            psr.percent_time_used,
                            psr.percent_budget_used
                        FROM project p
                        INNER JOIN project_status_report psr
                            ON p.project_id = psr.project_id
                            AND psr.status_date = ?
                        WHERE p.portfolio_id = ? AND p.is_active = TRUE
                    """

                    perf_trend_df = db.execute(perf_query_trend, (status_date_trend, portfolio_id)).df()

                    if not perf_trend_df.empty:
                        # Fill missing departments
                        perf_trend_df['responsible_organization'] = perf_trend_df['responsible_organization'].fillna('Unassigned')

                        # Filter by department if needed
                        if selected_dept_trend != 'All Departments':
                            perf_trend_df = perf_trend_df[perf_trend_df['responsible_organization'] == selected_dept_trend]

                        if not perf_trend_df.empty:
                            # Get factor scores
                            factor_scores_trend_query = """
                                SELECT project_id, score
                                FROM project_factor_score
                                WHERE portfolio_id = ? AND factor_id = ?
                            """
                            factor_scores_trend = db.execute(
                                factor_scores_trend_query,
                                (portfolio_id, selected_factor_id_trend)
                            ).df()

                            if not factor_scores_trend.empty:
                                # Merge scores with performance data
                                merged_trend = perf_trend_df.merge(factor_scores_trend, on='project_id', how='inner')

                                if not merged_trend.empty:
                                    # Add renamed columns
                                    merged_trend['pct_duration_used'] = merged_trend['percent_time_used']
                                    merged_trend['pct_budget_used'] = merged_trend['percent_budget_used']

                                    # Categorize into bins
                                    merged_trend['score_bin'] = pd.cut(
                                        merged_trend['score'],
                                        bins=bins_trend,
                                        labels=bin_labels_trend,
                                        include_lowest=True
                                    )

                                    # Calculate budget-weighted average for each bin
                                    for bin_label in bin_labels_trend:
                                        bin_data = merged_trend[merged_trend['score_bin'] == bin_label]

                                        if not bin_data.empty:
                                            # Filter to projects with budget
                                            bin_data_with_budget = bin_data[
                                                (bin_data['current_budget'].notna()) &
                                                (bin_data['current_budget'] > 0)
                                            ]

                                            if not bin_data_with_budget.empty:
                                                # Calculate budget-weighted average for selected metric
                                                # Filter out NaN and infinity values (especially for SPI/CPI)
                                                metric_data = bin_data_with_budget[
                                                    bin_data_with_budget[selected_metric_trend].notna() &
                                                    np.isfinite(bin_data_with_budget[selected_metric_trend])
                                                ]

                                                if not metric_data.empty:
                                                    weighted_avg = (
                                                        (metric_data[selected_metric_trend] * metric_data['current_budget']).sum() /
                                                        metric_data['current_budget'].sum()
                                                    )

                                                    trend_data.append({
                                                        'status_date': status_date_trend,
                                                        'score_range': bin_label,
                                                        'metric_value': weighted_avg
                                                    })

                if trend_data:
                    trend_df = pd.DataFrame(trend_data)

                    # Create line chart
                    fig_trend = go.Figure()

                    # Define colors for different score ranges
                    colors = px.colors.qualitative.Set3[:10]  # Use first 10 colors from Set3 palette

                    # Add a line for each score range
                    for idx, bin_label in enumerate(bin_labels_trend):
                        bin_trend_data = trend_df[trend_df['score_range'] == bin_label]

                        if not bin_trend_data.empty:
                            fig_trend.add_trace(go.Scatter(
                                x=bin_trend_data['status_date'],
                                y=bin_trend_data['metric_value'],
                                mode='lines+markers',
                                name=bin_label,
                                line=dict(color=colors[idx % len(colors)], width=2),
                                marker=dict(size=8),
                                hovertemplate='<b>%{fullData.name}</b><br>' +
                                             'Date: %{x|%Y-%m-%d}<br>' +
                                             f'{selected_metric_display}: ' + '%{y:.2f}<extra></extra>'
                            ))

                    # Update layout
                    y_axis_title = selected_metric_display
                    if selected_metric_trend in ['pct_duration_used', 'pct_budget_used']:
                        y_axis_title += " (%)"

                    fig_trend.update_layout(
                        title=f"{selected_metric_display} Trend by {selected_factor_name_trend} Score Range",
                        xaxis_title="Status Date",
                        yaxis_title=y_axis_title,
                        height=500,
                        hovermode='x unified',
                        legend=dict(
                            title="Score Range",
                            orientation="v",
                            yanchor="top",
                            y=1,
                            xanchor="left",
                            x=1.02
                        )
                    )

                    # Add reference line at 1.0 for SPI and CPI
                    if selected_metric_trend in ['spi', 'cpi']:
                        fig_trend.add_hline(
                            y=1.0,
                            line_dash="dash",
                            line_color="gray",
                            annotation_text="Target (1.0)",
                            annotation_position="right"
                        )

                    st.plotly_chart(fig_trend, width='stretch')

                    # Display info message if department filter is active
                    if selected_dept_trend != 'All Departments':
                        st.info(f"📍 Showing trends for: **{selected_dept_trend}**")

                    # Add explanatory notes
                    st.markdown("---")
                    st.markdown(f"""
                    **Chart Interpretation:**
                    - Each line represents a different {selected_factor_name_trend} score range
                    - Values are budget-weighted averages within each score range
                    - Trends show how performance varies over time for different score levels

                    **Metric: {selected_metric_display}**
                    - {'Higher values indicate better schedule performance (>1.0 is ahead of schedule)' if selected_metric_trend == 'spi' else ''}
                    - {'Higher values indicate better cost performance (>1.0 is under budget)' if selected_metric_trend == 'cpi' else ''}
                    - {'Shows percentage of project timeline elapsed' if selected_metric_trend == 'pct_duration_used' else ''}
                    - {'Shows percentage of budget consumed' if selected_metric_trend == 'pct_budget_used' else ''}
                    """)
                else:
                    st.info("No trend data available for the selected factor, metric, and department combination.")
            else:
                st.warning("No status reports found for this portfolio.")
        else:
            st.info("No factors defined for this portfolio.")


def portfolio_optimization_ui(portfolio_id):
    """UI for portfolio optimization - budget-constrained project selection"""
    st.markdown("### Portfolio Optimization")
    st.info("""
    **Portfolio Optimization** selects projects to maximize strategic value within a budget constraint.
    Projects are selected based on their strategic factor scores, with optional duration prioritization.
    """)

    # Get factors
    factors_df = db_manager.list_factors(portfolio_id)

    if factors_df.empty:
        st.warning("No strategic factors defined. Create factors first in the 'Manage Factors' tab.")
        return

    # Check total weight
    total_weight = factors_df['factor_weight_percent'].sum()
    if abs(total_weight - 100) > 0.01:
        st.warning(f"Factor weights sum to {total_weight:.1f}% (should be 100%). Results may be affected.")

    # Get projects with budget and duration info
    projects_query = """
        SELECT
            p.project_id,
            p.project_name,
            p.responsible_organization,
            p.current_budget,
            p.planned_start_date,
            p.planned_finish_date,
            sr.original_duration_months,
            sr.percent_budget_used,
            sr.percent_time_used
        FROM project p
        LEFT JOIN (
            SELECT project_id, MAX(status_date) as latest_date
            FROM project_status_report
            WHERE portfolio_id = ? AND is_active = TRUE
            GROUP BY project_id
        ) latest ON p.project_id = latest.project_id
        LEFT JOIN project_status_report sr ON p.project_id = sr.project_id
            AND sr.status_date = latest.latest_date
            AND sr.portfolio_id = ?
        WHERE p.portfolio_id = ? AND p.is_active = TRUE
    """
    projects_df = db.execute(projects_query, (portfolio_id, portfolio_id, portfolio_id)).df()

    if projects_df.empty:
        st.warning("No projects found in this portfolio.")
        return

    # Get all factor scores for the portfolio
    scores_query = """
        SELECT project_id, factor_id, score
        FROM project_factor_score
        WHERE portfolio_id = ? AND is_active = TRUE
    """
    scores_df = db.execute(scores_query, (portfolio_id,)).df()

    # Create scores lookup dict
    scores_lookup: Dict[Tuple[int, int], int] = {}
    if not scores_df.empty:
        for _, row in scores_df.iterrows():
            scores_lookup[(row['project_id'], row['factor_id'])] = row['score']

    # Calculate total portfolio budget
    total_portfolio_budget = projects_df['current_budget'].sum()

    if total_portfolio_budget <= 0:
        st.error("No budget data available for projects in this portfolio.")
        return

    st.markdown("---")

    # Configuration Section
    st.markdown("#### Configuration")

    col_budget, col_mode = st.columns(2)

    with col_budget:
        # Budget limit input
        default_budget = total_portfolio_budget * 0.8  # Default to 80%
        budget_limit = st.number_input(
            "Budget Limit",
            min_value=0.0,
            max_value=float(total_portfolio_budget * 2),
            value=float(default_budget),
            step=float(total_portfolio_budget * 0.05),
            format="%.0f",
            help=f"Maximum budget for selected projects. Total portfolio budget: {total_portfolio_budget:,.0f}"
        )

        # Show budget as percentage
        budget_pct = (budget_limit / total_portfolio_budget * 100) if total_portfolio_budget > 0 else 0
        st.caption(f"Budget limit: {budget_pct:.1f}% of total portfolio budget ({total_portfolio_budget:,.0f})")

    with col_mode:
        # Duration mode selection
        mode_options = {
            "Value Only (No duration preference)": DurationMode.NONE,
            "Soft Preference (Favor shorter projects)": DurationMode.SOFT,
            "Lexicographic (Value first, then duration)": DurationMode.LEXICOGRAPHIC,
            "Hard Rule (Minimum % for short projects)": DurationMode.HARD
        }

        selected_mode_name = st.radio(
            "Duration Prioritization Mode",
            options=list(mode_options.keys()),
            index=0,
            help="How to handle project duration in optimization"
        )
        duration_mode = mode_options[selected_mode_name]

        # Explanation popovers for each mode
        pop1, pop2, pop3, pop4 = st.columns(4)
        with pop1:
            with st.popover("Value Only"):
                st.markdown(
                    "**Value Only** maximizes strategic value with no duration "
                    "consideration. Projects are selected purely by their strategic "
                    "factor scores within the budget constraint.\n\n"
                    "**Formula:** Maximize sum(Value_i x_i) subject to "
                    "sum(Budget_i x_i) <= BudgetLimit"
                )
        with pop2:
            with st.popover("Soft Pref."):
                st.markdown(
                    "**Soft Preference** multiplies each project's strategic value "
                    "by a duration factor (Dmax / Duration_i). Shorter projects "
                    "receive a boost, but longer high-value projects can still be "
                    "selected.\n\n"
                    "**Effect:** A 6-month project gets 2x the multiplier of a "
                    "12-month project (all else equal)."
                )
        with pop3:
            with st.popover("Lexicographic"):
                st.markdown(
                    "**Lexicographic** runs a two-stage optimization.\n\n"
                    "**Stage 1:** Find the maximum strategic value (V*).\n\n"
                    "**Stage 2:** Minimize total duration while keeping value "
                    "within an epsilon tolerance of V*.\n\n"
                    "The *Value Tolerance* slider controls how much value "
                    "reduction is acceptable to shorten duration."
                )
        with pop4:
            with st.popover("Hard Rule"):
                st.markdown(
                    "**Hard Rule** enforces a minimum percentage of the selected "
                    "budget going to \"short\" projects (below a configurable "
                    "duration threshold).\n\n"
                    "**Example:** At least 30% of selected budget must go to "
                    "projects finishing within 12 months. Guarantees quick-delivering "
                    "projects in the portfolio."
                )

    # Mode-specific parameters
    epsilon_pct = 5.0
    short_threshold = 12.0
    min_short_pct = 30.0

    if duration_mode == DurationMode.LEXICOGRAPHIC:
        st.markdown("##### Lexicographic Mode Settings")
        epsilon_pct = st.slider(
            "Value Tolerance (%)",
            min_value=0.0,
            max_value=20.0,
            value=5.0,
            step=1.0,
            help="Allow this much reduction in strategic value to minimize duration"
        )

    elif duration_mode == DurationMode.HARD:
        st.markdown("##### Hard Duration Rule Settings")
        col_thresh, col_min = st.columns(2)

        with col_thresh:
            short_threshold = st.number_input(
                "Short Project Threshold (months)",
                min_value=1.0,
                max_value=60.0,
                value=12.0,
                step=1.0,
                help="Projects with duration <= this are considered 'short'"
            )

        with col_min:
            min_short_pct = st.slider(
                "Minimum Budget for Short Projects (%)",
                min_value=0.0,
                max_value=100.0,
                value=30.0,
                step=5.0,
                help="At least this % of selected budget must go to short projects"
            )

    # Pre-commit near-completion projects
    st.markdown("#### Pre-commit Near-Completion Projects")
    precommit_option = st.radio(
        "Auto-include projects that have used a high percentage of their budget",
        options=["Skip", "Include near-completion projects"],
        index=0,
        horizontal=True,
        help="Projects above the threshold are pre-committed and their budget is reserved before optimization"
    )

    precommitted_df = pd.DataFrame()
    precommit_threshold = None
    projects_for_opt = projects_df.copy()
    effective_budget = budget_limit

    if precommit_option == "Include near-completion projects":
        precommit_threshold = st.number_input(
            "% Budget Used Threshold",
            min_value=1,
            max_value=100,
            value=80,
            step=5,
            help="Projects that have used >= this % of their budget will be automatically included"
        )

        # Identify qualifying projects
        mask = (
            projects_df['percent_budget_used'].notna() &
            (projects_df['percent_budget_used'] >= precommit_threshold)
        )
        precommitted_df = projects_df[mask].copy()

        if not precommitted_df.empty:
            reserved_budget = precommitted_df['current_budget'].sum()
            effective_budget = max(0, budget_limit - reserved_budget)

            st.info(
                f"**{len(precommitted_df)}** project(s) pre-committed "
                f"(>= {precommit_threshold}% budget used). "
                f"Reserved budget: **{reserved_budget:,.0f}**. "
                f"Remaining for optimization: **{effective_budget:,.0f}**"
            )

            # Show pre-committed projects
            precommit_display = precommitted_df[
                ['project_name', 'current_budget', 'percent_budget_used']
            ].copy()
            precommit_display.columns = ['Project', 'Budget', '% Budget Used']
            precommit_display['% Budget Used'] = precommit_display['% Budget Used'].round(1)
            st.dataframe(precommit_display, hide_index=True, width='stretch')

            if effective_budget <= 0:
                st.warning(
                    "Pre-committed projects consume the entire budget limit. "
                    "Increase the budget limit or raise the threshold."
                )

            # Remove pre-committed from optimization pool
            projects_for_opt = projects_df[~mask].copy()
        else:
            st.info(f"No projects found with >= {precommit_threshold}% budget used.")

    st.markdown("---")

    # Run Optimization Button
    if st.button("Optimize Portfolio", type="primary", width='stretch'):
        with st.spinner("Running optimization..."):
            # Create configuration
            config = OptimizationConfig(
                budget_limit=effective_budget,
                duration_mode=duration_mode,
                epsilon_pct=epsilon_pct,
                short_threshold_months=short_threshold,
                min_short_budget_pct=min_short_pct,
                precommit_budget_pct_threshold=precommit_threshold
            )

            # Create optimizer
            optimizer = PortfolioOptimizer(config)

            # Prepare data (using filtered projects, excluding pre-committed)
            prepared_df, prep_warnings = optimizer.prepare_data(
                projects_for_opt, factors_df, scores_lookup
            )

            # Show preparation warnings
            if prep_warnings:
                for warning in prep_warnings:
                    st.warning(warning)

            if prepared_df.empty and precommitted_df.empty:
                st.error("No valid projects available for optimization after data preparation.")
                return

            if prepared_df.empty and not precommitted_df.empty:
                # Only pre-committed projects, no optimization needed
                result = OptimizationResult(
                    status='optimal',
                    message='All eligible projects are pre-committed.',
                    project_count=0,
                    precommitted_project_ids=precommitted_df['project_id'].tolist(),
                    precommitted_budget=precommitted_df['current_budget'].sum()
                )
            else:
                # Run optimization on remaining projects
                result = optimizer.optimize(prepared_df)
                # Attach pre-committed info
                if not precommitted_df.empty:
                    result.precommitted_project_ids = precommitted_df['project_id'].tolist()
                    result.precommitted_budget = precommitted_df['current_budget'].sum()

            # Store result in session state for display
            st.session_state['optimization_result'] = result
            st.session_state['optimization_config'] = config
            st.session_state['optimization_total_budget'] = total_portfolio_budget
            st.session_state['optimization_total_projects'] = len(prepared_df)
            st.session_state['optimization_precommitted_df'] = precommitted_df

    # Display Results (if available)
    if 'optimization_result' in st.session_state:
        result = st.session_state['optimization_result']
        config = st.session_state['optimization_config']
        total_budget = st.session_state['optimization_total_budget']
        total_projects = st.session_state['optimization_total_projects']
        precommitted_display_df = st.session_state.get('optimization_precommitted_df', pd.DataFrame())

        st.markdown("---")
        st.markdown("### Optimization Results")

        # Show pre-committed projects section if any
        if not precommitted_display_df.empty:
            st.markdown("#### Pre-committed Projects (Near Completion)")
            pc_display = precommitted_display_df[
                ['project_name', 'current_budget', 'percent_budget_used']
            ].copy()
            pc_display.columns = ['Project', 'Budget', '% Budget Used']
            pc_display['% Budget Used'] = pc_display['% Budget Used'].round(1)
            st.dataframe(pc_display, hide_index=True, width='stretch')
            st.caption(
                f"**{len(precommitted_display_df)}** pre-committed project(s) | "
                f"Reserved budget: **{result.precommitted_budget:,.0f}**"
            )
            st.markdown("")

        if result.status != 'optimal':
            st.error(f"**Status:** {result.status}")
            st.markdown(result.message)
            return

        # Success - show results

        # Show warnings from result
        if result.warnings:
            for warning in result.warnings:
                st.info(warning)

        # Summary Metrics
        st.markdown("#### Summary")
        col1, col2, col3, col4 = st.columns(4)

        with col1:
            st.metric(
                "Budget Used",
                f"{result.total_budget_used:,.0f}",
                f"{result.budget_utilization_pct:.1f}% of limit"
            )

        with col2:
            st.metric(
                "Value Captured",
                f"{result.value_captured_pct:.1f}%",
                f"of total portfolio value"
            )

        with col3:
            st.metric(
                "Projects Selected",
                f"{result.project_count}",
                f"of {total_projects} eligible"
            )

        with col4:
            if result.total_duration_months > 0:
                st.metric(
                    "Total Duration",
                    f"{result.total_duration_months:.0f} mo",
                    f"Avg: {result.avg_duration_months:.1f} mo"
                )
            else:
                st.metric("Total Duration", "N/A")

        # Selected Projects Table
        st.markdown("#### Selected Projects")

        if not result.selected_df.empty:
            display_cols = ['rank', 'project_name', 'responsible_organization', 'budget',
                          'strategic_score_pct']
            if 'duration_months' in result.selected_df.columns:
                display_cols.insert(4, 'duration_months')

            display_df = result.selected_df[
                [c for c in display_cols if c in result.selected_df.columns]
            ].copy()

            # Rename columns for display
            display_df.columns = [
                'Rank', 'Project Name', 'Organization', 'Budget',
                *(['Duration (mo)'] if 'duration_months' in display_cols else []),
                'Strategic Score (%)'
            ]

            st.dataframe(
                display_df,
                width='stretch',
                hide_index=True
            )

            # Download button for selected projects
            csv = result.selected_df.to_csv(index=False)
            st.download_button(
                "Download Selected Projects CSV",
                data=csv,
                file_name=f"optimized_projects_portfolio_{portfolio_id}.csv",
                mime="text/csv"
            )
        else:
            st.info("No projects selected.")

        # Not Selected Projects (Top high-value)
        st.markdown("#### Top Excluded High-Value Projects")

        if not result.not_selected_df.empty:
            # Show top 10 by strategic score
            top_excluded = result.not_selected_df.head(10)

            display_cols_excl = ['project_name', 'responsible_organization', 'budget',
                                'strategic_score_pct']
            if 'duration_months' in top_excluded.columns:
                display_cols_excl.insert(3, 'duration_months')

            display_excl_df = top_excluded[
                [c for c in display_cols_excl if c in top_excluded.columns]
            ].copy()

            display_excl_df.columns = [
                'Project Name', 'Organization', 'Budget',
                *(['Duration (mo)'] if 'duration_months' in display_cols_excl else []),
                'Strategic Score (%)'
            ]

            st.dataframe(
                display_excl_df,
                width='stretch',
                hide_index=True
            )
        else:
            st.info("All eligible projects were selected.")

        # Visualizations
        st.markdown("#### Visualizations")

        viz_col1, viz_col2 = st.columns(2)

        with viz_col1:
            # Scatter: Duration vs Score
            if not result.selected_df.empty and 'duration_months' in result.selected_df.columns:
                # Combine selected and not selected for full picture
                all_projects = pd.concat([
                    result.selected_df.assign(status='Selected'),
                    result.not_selected_df.assign(status='Not Selected')
                ])

                if 'duration_months' in all_projects.columns and not all_projects['duration_months'].isna().all():
                    fig_scatter = px.scatter(
                        all_projects,
                        x='duration_months',
                        y='strategic_score_pct',
                        size='budget',
                        color='status',
                        hover_name='project_name',
                        labels={
                            'duration_months': 'Duration (months)',
                            'strategic_score_pct': 'Strategic Score (%)',
                            'budget': 'Budget',
                            'status': 'Selection'
                        },
                        title='Projects: Duration vs Strategic Score',
                        color_discrete_map={'Selected': '#27ae60', 'Not Selected': '#95a5a6'}
                    )
                    st.plotly_chart(fig_scatter, width='stretch')
                else:
                    st.info("Duration data not available for scatter plot.")
            else:
                st.info("No data available for scatter plot.")

        with viz_col2:
            # Bar: Selected projects by strategic value
            if not result.selected_df.empty:
                top_10_selected = result.selected_df.head(10)
                fig_bar = px.bar(
                    top_10_selected,
                    x='project_name',
                    y='strategic_score_pct',
                    color='budget',
                    labels={
                        'project_name': 'Project',
                        'strategic_score_pct': 'Strategic Score (%)',
                        'budget': 'Budget'
                    },
                    title='Top Selected Projects by Strategic Score'
                )
                fig_bar.update_xaxes(tickangle=45)
                st.plotly_chart(fig_bar, width='stretch')
            else:
                st.info("No selected projects to display.")

        # Pie chart: Budget by organization
        if not result.selected_df.empty and 'responsible_organization' in result.selected_df.columns:
            org_budget = result.selected_df.groupby('responsible_organization')['budget'].sum().reset_index()
            if len(org_budget) > 1:
                fig_pie = px.pie(
                    org_budget,
                    values='budget',
                    names='responsible_organization',
                    title='Selected Budget by Organization'
                )
                st.plotly_chart(fig_pie, width='stretch')

        # Consolidated Table View
        st.markdown("#### Consolidated Project List")
        show_table_view = st.toggle(
            "Show Table View",
            value=False,
            help="Display all projects in a single table with selection status"
        )

        if show_table_view:
            # Build consolidated dataframe with all projects
            consolidated_rows = []

            # Add pre-committed projects
            if not precommitted_display_df.empty:
                for _, row in precommitted_display_df.iterrows():
                    consolidated_rows.append({
                        'Project Name': row.get('project_name', ''),
                        'Organization': row.get('responsible_organization', ''),
                        'Prioritization Score (%)': row.get('strategic_score_pct', row.get('weighted_score', 0)),
                        'Strategic Value': row.get('strategic_score', row.get('weighted_score', 0)),
                        'Budget': row.get('current_budget', row.get('budget', 0)),
                        'Duration (mo)': row.get('duration_months', row.get('original_duration_months', '')),
                        '% Time Used': row.get('percent_time_used', row.get('pct_time_used', '')),
                        '% Budget Used': row.get('percent_budget_used', row.get('pct_budget_used', '')),
                        'Status': 'Pre-selected'
                    })

            # Add selected projects
            if not result.selected_df.empty:
                for _, row in result.selected_df.iterrows():
                    consolidated_rows.append({
                        'Project Name': row.get('project_name', ''),
                        'Organization': row.get('responsible_organization', ''),
                        'Prioritization Score (%)': row.get('strategic_score_pct', 0),
                        'Strategic Value': row.get('strategic_score', row.get('strategic_score_pct', 0)),
                        'Budget': row.get('budget', 0),
                        'Duration (mo)': row.get('duration_months', ''),
                        '% Time Used': row.get('percent_time_used', row.get('pct_time_used', '')),
                        '% Budget Used': row.get('percent_budget_used', row.get('pct_budget_used', '')),
                        'Status': 'Selected'
                    })

            # Add dropped (not selected) projects
            if not result.not_selected_df.empty:
                for _, row in result.not_selected_df.iterrows():
                    consolidated_rows.append({
                        'Project Name': row.get('project_name', ''),
                        'Organization': row.get('responsible_organization', ''),
                        'Prioritization Score (%)': row.get('strategic_score_pct', 0),
                        'Strategic Value': row.get('strategic_score', row.get('strategic_score_pct', 0)),
                        'Budget': row.get('budget', 0),
                        'Duration (mo)': row.get('duration_months', ''),
                        '% Time Used': row.get('percent_time_used', row.get('pct_time_used', '')),
                        '% Budget Used': row.get('percent_budget_used', row.get('pct_budget_used', '')),
                        'Status': 'Dropped'
                    })

            if consolidated_rows:
                consolidated_df = pd.DataFrame(consolidated_rows)

                # Sort by Prioritization Score descending
                consolidated_df = consolidated_df.sort_values(
                    'Prioritization Score (%)',
                    ascending=False
                ).reset_index(drop=True)

                # Add rank column
                consolidated_df.insert(0, 'Rank', range(1, len(consolidated_df) + 1))

                # Format numeric columns
                consolidated_df['Prioritization Score (%)'] = consolidated_df['Prioritization Score (%)'].round(1)
                consolidated_df['Budget'] = consolidated_df['Budget'].apply(lambda x: f"{x:,.0f}" if pd.notna(x) and x != '' else "")

                # Format % Time Used and % Budget Used
                consolidated_df['% Time Used'] = consolidated_df['% Time Used'].apply(
                    lambda x: f"{float(x):.1f}" if pd.notna(x) and x != '' else ""
                )
                consolidated_df['% Budget Used'] = consolidated_df['% Budget Used'].apply(
                    lambda x: f"{float(x):.1f}" if pd.notna(x) and x != '' else ""
                )

                # Display with color coding for status
                st.dataframe(
                    consolidated_df,
                    width='stretch',
                    hide_index=True,
                    column_config={
                        'Rank': st.column_config.NumberColumn('Rank', width='small'),
                        'Project Name': st.column_config.TextColumn('Project Name', width='large'),
                        'Organization': st.column_config.TextColumn('Organization', width='medium'),
                        'Prioritization Score (%)': st.column_config.NumberColumn('Score (%)', width='small'),
                        'Strategic Value': st.column_config.NumberColumn('Strategic Value', width='small'),
                        'Budget': st.column_config.TextColumn('Budget', width='medium'),
                        'Duration (mo)': st.column_config.TextColumn('Duration', width='small'),
                        '% Time Used': st.column_config.TextColumn('% Time', width='small'),
                        '% Budget Used': st.column_config.TextColumn('% Budget', width='small'),
                        'Status': st.column_config.TextColumn('Status', width='medium')
                    }
                )

                # Summary by status
                status_summary = consolidated_df.groupby('Status').size().to_dict()
                summary_parts = []
                if 'Pre-selected' in status_summary:
                    summary_parts.append(f"**{status_summary['Pre-selected']}** pre-selected")
                if 'Selected' in status_summary:
                    summary_parts.append(f"**{status_summary['Selected']}** selected")
                if 'Dropped' in status_summary:
                    summary_parts.append(f"**{status_summary['Dropped']}** dropped")

                st.caption(f"Total: {len(consolidated_df)} projects | " + " | ".join(summary_parts))

                # Download button for consolidated view
                csv_consolidated = consolidated_df.to_csv(index=False)
                st.download_button(
                    "📥 Download Consolidated Table (CSV)",
                    data=csv_consolidated,
                    file_name=f"portfolio_optimization_consolidated_{portfolio_id}.csv",
                    mime="text/csv"
                )
            else:
                st.info("No project data available for consolidated view.")

        # Narrative
        st.markdown("#### Summary Narrative")
        narrative = generate_optimization_narrative(
            result, config, total_budget, total_projects
        )
        st.markdown(narrative)


# Main page layout
portfolio_id = select_portfolio()

# Compute write access
_has_write = False
if portfolio_id:
    try:
        from utils.portfolio_access import get_portfolio_access_level
        from utils.portfolio_context import get_current_user_email
        _user_email = get_current_user_email()
        _access_level = get_portfolio_access_level(portfolio_id, _user_email)
        _has_write = _access_level in ('owner', 'write')
        st.session_state['portfolio_access_level'] = _access_level
    except Exception:
        _has_write = True

    if not _has_write:
        st.info("You have **read-only** access to this portfolio. Factor editing is disabled.")

if portfolio_id:
    st.markdown("---")

    tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs([
        "🎯 Alignment Dashboard",
        "📋 Manage Factors",
        "📊 Score Projects",
        "📈 Analytics",
        "📦 Portfolio Optimization",
        "ℹ️ Help"
    ])

    with tab1:
        strategic_alignment_ui(portfolio_id)

    with tab2:
        create_factor_ui(portfolio_id)
        st.markdown("---")
        list_factors_ui(portfolio_id)

    with tab3:
        score_projects_ui(portfolio_id)

    with tab4:
        st.markdown("### 📈 Advanced Analytics")
        st.info("🚧 Coming soon: Heatmaps, correlation analysis, and trend charts")

    with tab5:
        portfolio_optimization_ui(portfolio_id)

    with tab6:
        st.markdown("""
        ## Strategic Factors Help

        ### 🎯 What are Strategic Factors?

        **Strategic Factors** are portfolio-specific criteria used to evaluate how well projects align with your strategic goals and priorities. Unlike SDGs (which are universal), strategic factors are customized for each portfolio.

        ### 📋 Common Strategic Factors

        **Business Value:**
        - Strategic Alignment
        - Revenue Potential
        - Market Impact
        - Customer Value

        **Technical:**
        - Innovation Potential
        - Technical Feasibility
        - Scalability
        - Platform Reusability

        **Risk & Complexity:**
        - Risk Level
        - Implementation Complexity
        - Resource Availability
        - Dependencies

        **Organizational:**
        - Stakeholder Support
        - Team Capability
        - Change Impact
        - Regulatory Compliance

        ### 🔢 Factor Weights

        **Weight Percentage:**
        - Represents the relative importance of each factor
        - Must sum to 100% across all factors
        - Higher weight = more important factor

        **Example:**
        - Strategic Alignment: 30%
        - Innovation Potential: 25%
        - Risk Level: 20%
        - Market Impact: 15%
        - Technical Feasibility: 10%
        - **Total: 100%**

        ### 📊 Scoring Scale (Likert)

        **Default: 1-5**
        - 1 = Very Low / Strongly Disagree
        - 2 = Low / Disagree
        - 3 = Medium / Neutral
        - 4 = High / Agree
        - 5 = Very High / Strongly Agree

        **Custom Scales:**
        - Can use 0-10, 1-3, or any range
        - Consistent scale recommended within a portfolio

        ### 🎯 Strategic Alignment Calculation

        **Formula:**
        ```
        Alignment Score = Σ (Score / Max Score) × (Weight / 100) × 100%
        ```

        **Example:**
        For a project with:
        - Strategic Alignment: 5/5 (weight 30%) → 1.0 × 0.30 = 0.30
        - Innovation Potential: 4/5 (weight 25%) → 0.8 × 0.25 = 0.20
        - Risk Level: 3/5 (weight 20%) → 0.6 × 0.20 = 0.12
        - Market Impact: 5/5 (weight 15%) → 1.0 × 0.15 = 0.15
        - Technical Feasibility: 4/5 (weight 10%) → 0.8 × 0.10 = 0.08

        **Total: (0.30 + 0.20 + 0.12 + 0.15 + 0.08) × 100 = 85%**

        ### 🚦 Alignment Levels

        - **🟢 High (70-100%)**: Strong strategic alignment, high priority
        - **🟡 Medium (40-69%)**: Moderate alignment, consider improvements
        - **🔴 Low (0-39%)**: Weak alignment, review justification

        ### ✅ Best Practices

        1. **Define Clear Factors**: Use specific, measurable criteria
        2. **Limit Number**: 3-7 factors is optimal (too many dilutes focus)
        3. **Weight Thoughtfully**: Reflect true strategic priorities
        4. **Score Consistently**: Use same criteria across all projects
        5. **Review Regularly**: Update scores as projects evolve
        6. **Involve Stakeholders**: Get input from project sponsors
        7. **Document Rationale**: Explain why scores were assigned

        ### 📊 Using Alignment Scores

        **Portfolio Prioritization:**
        - Rank projects by alignment score
        - Allocate resources to high-alignment projects
        - Consider deprioritizing low-alignment projects

        **Decision Making:**
        - Gate criteria for project approval
        - Input to portfolio optimization
        - Justification for funding decisions

        **Performance Tracking:**
        - Track alignment over time
        - Identify projects drifting off strategy
        - Measure portfolio strategic health

        ### 🔗 Integration

        Strategic alignment scores can be combined with:
        - EVM metrics (cost/schedule performance)
        - SDG impact (sustainability alignment)
        - Risk assessments
        - Business case ROI

        ### 💡 Tips

        - **Start Simple**: Begin with 3-5 key factors
        - **Pilot Test**: Try with a few projects first
        - **Calibrate**: Discuss scoring with team to ensure consistency
        - **Iterate**: Refine factors based on experience
        - **Automate**: Use alignment scores in automated reports
        """)

# Footer
st.markdown("---")
st.caption("Strategic Factors - Portfolio Management Suite v1.0")

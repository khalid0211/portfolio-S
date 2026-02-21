"""
Portfolio Settings Management
Functions for managing portfolio-specific settings (controls, LLM config)
"""

import streamlit as st
import json
import pandas as pd
from datetime import datetime, date
from database.db_connection import get_db

def load_portfolio_settings(portfolio_id: int) -> dict:
    """Load settings for a specific portfolio from database"""
    db = get_db()

    # Get portfolio settings
    query = """
        SELECT
            default_curve_type,
            default_alpha,
            default_beta,
            default_inflation_rate,
            settings_json
        FROM portfolio
        WHERE portfolio_id = ?
    """

    result = db.execute(query, (portfolio_id,)).fetchone()

    if not result:
        return {}

    # Parse settings
    settings = {
        'curve_type': result[0] or 'linear',
        'alpha': float(result[1]) if result[1] else 2.0,
        'beta': float(result[2]) if result[2] else 2.0,
        'inflation_rate': float(result[3]) if result[3] else 0.0
    }

    # Parse JSON settings if they exist
    if result[4]:
        try:
            json_settings = json.loads(result[4])
            settings.update(json_settings)
        except:
            pass

    # Set defaults for missing values
    settings.setdefault('currency_symbol', '$')
    settings.setdefault('currency_postfix', '')
    settings.setdefault('date_format', 'YYYY-MM-DD')
    settings.setdefault('data_date', '2024-01-01')
    settings.setdefault('tier_config', {
        'cutoff_points': [4000, 8000, 15000],
        'tier_names': ['Tier 1', 'Tier 2', 'Tier 3', 'Tier 4'],
        'colors': ['#3498db', '#27ae60', '#f39c12', '#e74c3c']
    })
    settings.setdefault('duration_tier_config', {
        'cutoff_points': [6, 12, 24],
        'tier_names': ['Short', 'Medium', 'Long', 'Extra Long'],
        'colors': ['#3498db', '#27ae60', '#f39c12', '#e74c3c']
    })
    settings.setdefault('llm_config', {
        'provider': 'OpenAI',
        'model': 'gpt-4o-mini',
        'temperature': 0.2,
        'timeout': 60,
        'api_key': '',
        'has_api_key': False
    })
    settings.setdefault('voice_config', {
        'voice_enabled': False,
        'provider': 'OpenAI',  # 'OpenAI' or 'Replicate'
        'use_llm_openai_key': True,  # If true AND llm_config.provider is OpenAI, use llm_config.api_key
        'openai_api_key': '',  # For Whisper ASR + OpenAI TTS (only if not using LLM key)
        'replicate_api_key': ''  # For Replicate-based ASR/TTS
    })
    settings.setdefault('infographic_config', {
        'enabled': False,
        'use_voice_replicate_key': True,  # If true, use voice_config.replicate_api_key
        'replicate_api_key': '',  # Separate Replicate token for images (only if not sharing)
        'model': 'nano-banana'  # 'nano-banana', 'schnell', or 'dev'
    })

    # Migrate old voice_config format if needed
    settings = migrate_settings(settings)

    return settings


def save_portfolio_settings(portfolio_id: int, settings: dict):
    """Save settings for a specific portfolio to database"""
    db = get_db()
    conn = db.get_connection()

    # Extract fields that have dedicated columns
    curve_type = settings.get('curve_type', 'linear').lower()
    alpha = float(settings.get('alpha', 2.0))
    beta = float(settings.get('beta', 2.0))
    inflation_rate = float(settings.get('inflation_rate', 0.0))

    # Everything else goes into JSON
    json_settings = {
        'currency_symbol': settings.get('currency_symbol', '$'),
        'currency_postfix': settings.get('currency_postfix', ''),
        'date_format': settings.get('date_format', 'YYYY-MM-DD'),
        'data_date': settings.get('data_date', '2024-01-01'),
        'tier_config': settings.get('tier_config', {}),
        'duration_tier_config': settings.get('duration_tier_config', {}),
        'llm_config': settings.get('llm_config', {}),
        'voice_config': settings.get('voice_config', {}),
        'infographic_config': settings.get('infographic_config', {})
    }

    # Update portfolio
    conn.execute("""
        UPDATE portfolio
        SET
            default_curve_type = ?,
            default_alpha = ?,
            default_beta = ?,
            default_inflation_rate = ?,
            settings_json = ?
        WHERE portfolio_id = ?
    """, (curve_type, alpha, beta, inflation_rate, json.dumps(json_settings), portfolio_id))


def migrate_settings(settings: dict) -> dict:
    """
    Migrate old settings format to new structure.

    Old format:
        voice_config: { replicate_api_key, has_replicate_key, voice_enabled }

    New format:
        voice_config: { voice_enabled, provider, use_llm_openai_key, openai_api_key, replicate_api_key }
        infographic_config: { enabled, use_voice_replicate_key, replicate_api_key, model }
    """
    voice_config = settings.get('voice_config', {})
    llm_config = settings.get('llm_config', {})

    # Check if already migrated (has new fields like 'provider')
    if 'provider' in voice_config:
        return settings

    # Old format detected - migrate
    old_replicate_key = voice_config.get('replicate_api_key', '')
    old_voice_enabled = voice_config.get('voice_enabled', False)
    has_replicate = bool(old_replicate_key)

    # Determine voice provider based on what keys are available
    # If they had a Replicate key, keep using Replicate; otherwise default to OpenAI
    voice_provider = 'Replicate' if has_replicate else 'OpenAI'

    # Migrate voice_config
    new_voice_config = {
        'voice_enabled': old_voice_enabled,
        'provider': voice_provider,
        'use_llm_openai_key': llm_config.get('provider') == 'OpenAI',  # Default to sharing if LLM is OpenAI
        'openai_api_key': '',  # No separate OpenAI key in old format
        'replicate_api_key': old_replicate_key
    }

    # Create new infographic_config - share Replicate key by default
    infographic_config = settings.get('infographic_config', {})
    if 'enabled' not in infographic_config:
        new_infographic_config = {
            'enabled': has_replicate,  # Enable if they had Replicate key
            'use_voice_replicate_key': True,  # Default to sharing
            'replicate_api_key': '',  # Empty since sharing
            'model': 'nano-banana'
        }
        settings['infographic_config'] = new_infographic_config

    settings['voice_config'] = new_voice_config
    return settings


def get_voice_openai_key(llm_config: dict, voice_config: dict) -> str:
    """
    Get the OpenAI API key to use for voice features.

    Returns the appropriate OpenAI key based on voice_config settings:
    - If voice provider is OpenAI and use_llm_openai_key is True and LLM is OpenAI, return LLM key
    - Otherwise return voice_config's own openai_api_key

    Args:
        llm_config: LLM configuration dict
        voice_config: Voice configuration dict

    Returns:
        OpenAI API key string (may be empty)
    """
    if voice_config.get('provider') != 'OpenAI':
        return ''

    if voice_config.get('use_llm_openai_key') and llm_config.get('provider') == 'OpenAI':
        return llm_config.get('api_key', '')

    return voice_config.get('openai_api_key', '')


def get_voice_replicate_key(voice_config: dict) -> str:
    """
    Get the Replicate API key to use for voice features.

    Args:
        voice_config: Voice configuration dict

    Returns:
        Replicate API key string (may be empty)
    """
    if voice_config.get('provider') != 'Replicate':
        return ''
    return voice_config.get('replicate_api_key', '')


def get_infographic_replicate_key(voice_config: dict, infographic_config: dict) -> str:
    """
    Get the Replicate API key to use for infographic generation.

    Returns the appropriate Replicate key based on infographic_config settings:
    - If use_voice_replicate_key is True, return voice_config's replicate_api_key
    - Otherwise return infographic_config's own replicate_api_key

    Args:
        voice_config: Voice configuration dict
        infographic_config: Infographic configuration dict

    Returns:
        Replicate API key string (may be empty)
    """
    if infographic_config.get('use_voice_replicate_key'):
        return voice_config.get('replicate_api_key', '')
    return infographic_config.get('replicate_api_key', '')

"""
Configuration loader for WhatsApp integration.

Loads all settings from environment variables.
"""

import os
from dataclasses import dataclass, field
from typing import List, Optional
from dotenv import load_dotenv

# Load .env file from whatsapp directory or parent
load_dotenv()


@dataclass
class WhatsAppConfig:
    """Configuration for the WhatsApp service."""

    # Twilio credentials
    twilio_account_sid: str
    twilio_auth_token: str
    twilio_whatsapp_number: str

    # Database connection
    motherduck_token: str
    motherduck_database: str
    motherduck_connection_string: str

    # Authentication - allowed phone numbers
    allowed_phones: List[str]

    # LLM fallback (if portfolio has no LLM config)
    openai_api_key: Optional[str]

    # Server settings
    host: str
    port: int

    # Security
    validate_twilio_signature: bool

    @classmethod
    def from_env(cls) -> "WhatsAppConfig":
        """Load configuration from environment variables."""

        # Required Twilio settings
        twilio_account_sid = os.getenv("TWILIO_ACCOUNT_SID", "")
        twilio_auth_token = os.getenv("TWILIO_AUTH_TOKEN", "")
        twilio_whatsapp_number = os.getenv("TWILIO_WHATSAPP_NUMBER", "")

        # Database settings
        motherduck_token = os.getenv("MOTHERDUCK_TOKEN", "")
        motherduck_database = os.getenv("MOTHERDUCK_DATABASE", "portfolio_cloud")

        # Build connection string
        if motherduck_token:
            conn_string = f"md:{motherduck_database}?motherduck_token={motherduck_token}"
        else:
            conn_string = f"md:{motherduck_database}"

        # Allowed phone numbers (comma-separated)
        allowed_phones_str = os.getenv("ALLOWED_PHONES", "")
        allowed_phones = [
            phone.strip()
            for phone in allowed_phones_str.split(",")
            if phone.strip()
        ]

        # LLM fallback key
        openai_api_key = os.getenv("OPENAI_API_KEY", "")

        # Server settings
        host = os.getenv("HOST", "0.0.0.0")
        port = int(os.getenv("PORT", "8000"))

        # Security
        validate_signature = os.getenv("VALIDATE_TWILIO_SIGNATURE", "false").lower() == "true"

        return cls(
            twilio_account_sid=twilio_account_sid,
            twilio_auth_token=twilio_auth_token,
            twilio_whatsapp_number=twilio_whatsapp_number,
            motherduck_token=motherduck_token,
            motherduck_database=motherduck_database,
            motherduck_connection_string=conn_string,
            allowed_phones=allowed_phones,
            openai_api_key=openai_api_key if openai_api_key else None,
            host=host,
            port=port,
            validate_twilio_signature=validate_signature,
        )

    def is_phone_allowed(self, phone: str) -> bool:
        """Check if a phone number is in the whitelist."""
        # Normalize phone number (remove spaces, ensure + prefix)
        normalized = phone.strip().replace(" ", "").replace("-", "")
        if not normalized.startswith("+"):
            normalized = f"+{normalized}"

        for allowed in self.allowed_phones:
            allowed_normalized = allowed.strip().replace(" ", "").replace("-", "")
            if not allowed_normalized.startswith("+"):
                allowed_normalized = f"+{allowed_normalized}"
            if normalized == allowed_normalized:
                return True

        return False

    def validate(self) -> List[str]:
        """Validate configuration and return list of errors."""
        errors = []

        if not self.twilio_account_sid:
            errors.append("TWILIO_ACCOUNT_SID is required")
        if not self.twilio_auth_token:
            errors.append("TWILIO_AUTH_TOKEN is required")
        if not self.motherduck_token:
            errors.append("MOTHERDUCK_TOKEN is required")
        if not self.allowed_phones:
            errors.append("ALLOWED_PHONES is required (comma-separated phone numbers)")

        return errors

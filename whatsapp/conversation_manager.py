"""
Conversation state manager for WhatsApp integration.

Manages per-phone-number conversation state including:
- Selected portfolio and project
- Department hierarchy navigation
- Chat history
- Session timeout/expiry
"""

from typing import Dict, Optional, List
from datetime import datetime, timedelta
from enum import Enum
import threading


class MenuLevel(str, Enum):
    """Current navigation level in the hierarchy."""
    NONE = "NONE"              # No portfolio selected
    PORTFOLIO = "PORTFOLIO"    # Portfolio selected, showing overview
    DEPARTMENT = "DEPARTMENT"  # Viewing department drill-down
    PROJECT = "PROJECT"        # Specific project selected


class ConversationManager:
    """
    Thread-safe conversation state management.

    Each phone number has its own conversation state that includes:
    - Selected portfolio (id and name)
    - Selected project (optional, for project-specific mode)
    - Chat history (Q&A pairs for LLM context)
    - Last activity timestamp (for auto-expiry)
    """

    def __init__(self, history_limit: int = 10, ttl_minutes: int = 60):
        """
        Initialize the conversation manager.

        Args:
            history_limit: Maximum Q&A pairs to keep per conversation
            ttl_minutes: Auto-expire conversations after inactivity (minutes)
        """
        self._conversations: Dict[str, dict] = {}
        self._lock = threading.Lock()
        self._history_limit = history_limit
        self._ttl = timedelta(minutes=ttl_minutes)

    def _create_empty_state(self) -> dict:
        """Create a new empty conversation state."""
        return {
            "portfolio_id": None,
            "portfolio_name": None,
            "project_id": None,
            "project_name": None,
            "status_date": None,
            "history": [],
            "last_activity": datetime.utcnow(),
            "selected_department": None,
            "current_menu_level": MenuLevel.NONE.value,
            "last_list_type": None,
            "last_list_items": [],
        }

    def get_or_create(self, phone: str) -> dict:
        """
        Get or create conversation state for a phone number.

        Args:
            phone: Phone number (normalized)

        Returns:
            Conversation state dict with keys:
                - portfolio_id: int or None
                - portfolio_name: str or None
                - project_id: int or None
                - project_name: str or None
                - status_date: date or None
                - history: List of {"question": str, "answer": str}
                - last_activity: datetime
        """
        with self._lock:
            self._cleanup_expired()

            if phone not in self._conversations:
                self._conversations[phone] = self._create_empty_state()
            else:
                self._conversations[phone]["last_activity"] = datetime.utcnow()

            return self._conversations[phone]

    def get(self, phone: str) -> Optional[dict]:
        """
        Get conversation state if exists.

        Returns:
            Conversation state or None if not found
        """
        with self._lock:
            return self._conversations.get(phone)

    def set_portfolio(
        self,
        phone: str,
        portfolio_id: int,
        portfolio_name: str,
        status_date=None
    ):
        """
        Set the active portfolio for a conversation.

        Also clears project/department selection and history when portfolio changes.
        """
        with self._lock:
            # Ensure conversation exists
            if phone not in self._conversations:
                self._conversations[phone] = self._create_empty_state()

            conv = self._conversations[phone]
            conv["portfolio_id"] = portfolio_id
            conv["portfolio_name"] = portfolio_name
            conv["project_id"] = None
            conv["project_name"] = None
            conv["status_date"] = status_date
            conv["history"] = []  # Clear history on portfolio change
            conv["selected_department"] = None
            conv["current_menu_level"] = MenuLevel.PORTFOLIO.value
            conv["last_list_type"] = None
            conv["last_list_items"] = []
            conv["last_activity"] = datetime.utcnow()

    def set_project(self, phone: str, project_id: int, project_name: str):
        """
        Set the active project for a conversation (project-specific mode).
        """
        with self._lock:
            if phone not in self._conversations:
                self._conversations[phone] = self._create_empty_state()

            conv = self._conversations[phone]
            conv["project_id"] = project_id
            conv["project_name"] = project_name
            conv["history"] = []  # Clear history on project change
            conv["current_menu_level"] = MenuLevel.PROJECT.value
            conv["last_list_type"] = None
            conv["last_list_items"] = []
            conv["last_activity"] = datetime.utcnow()

    def clear_project(self, phone: str):
        """
        Clear project selection (return to portfolio/department mode).
        """
        with self._lock:
            if phone in self._conversations:
                conv = self._conversations[phone]
                conv["project_id"] = None
                conv["project_name"] = None
                conv["history"] = []
                # Return to department level if one was selected, else portfolio
                if conv.get("selected_department"):
                    conv["current_menu_level"] = MenuLevel.DEPARTMENT.value
                else:
                    conv["current_menu_level"] = MenuLevel.PORTFOLIO.value
                conv["last_activity"] = datetime.utcnow()

    def add_to_history(self, phone: str, question: str, answer: str):
        """
        Add a Q&A pair to the conversation history.

        Automatically trims to history_limit.
        """
        with self._lock:
            if phone in self._conversations:
                history = self._conversations[phone]["history"]
                history.append({"question": question, "answer": answer})

                # Trim to limit
                if len(history) > self._history_limit:
                    self._conversations[phone]["history"] = history[-self._history_limit:]

                self._conversations[phone]["last_activity"] = datetime.utcnow()

    def clear_history(self, phone: str):
        """Clear conversation history for a phone number."""
        with self._lock:
            if phone in self._conversations:
                self._conversations[phone]["history"] = []
                self._conversations[phone]["last_activity"] = datetime.utcnow()

    def clear_conversation(self, phone: str):
        """Completely reset a conversation (clear portfolio, project, history)."""
        with self._lock:
            if phone in self._conversations:
                del self._conversations[phone]

    def get_history(self, phone: str) -> List[Dict[str, str]]:
        """Get the conversation history for a phone number."""
        with self._lock:
            if phone in self._conversations:
                return list(self._conversations[phone]["history"])
            return []

    def set_department(self, phone: str, department: str):
        """
        Set the selected department for hierarchical navigation.
        """
        with self._lock:
            if phone not in self._conversations:
                self._conversations[phone] = self._create_empty_state()

            conv = self._conversations[phone]
            conv["selected_department"] = department
            conv["project_id"] = None
            conv["project_name"] = None
            conv["current_menu_level"] = MenuLevel.DEPARTMENT.value
            conv["history"] = []
            conv["last_activity"] = datetime.utcnow()

    def clear_department(self, phone: str):
        """
        Clear department selection (return to portfolio level).
        """
        with self._lock:
            if phone in self._conversations:
                conv = self._conversations[phone]
                conv["selected_department"] = None
                conv["project_id"] = None
                conv["project_name"] = None
                conv["current_menu_level"] = MenuLevel.PORTFOLIO.value
                conv["history"] = []
                conv["last_activity"] = datetime.utcnow()

    def set_last_list(self, phone: str, list_type: str, items: List[Dict]):
        """
        Store the last displayed numbered list for number-based selection.

        Args:
            phone: Phone number
            list_type: Type of list ("portfolios", "departments", "projects")
            items: List of dicts with item details for mapping
        """
        with self._lock:
            if phone not in self._conversations:
                self._conversations[phone] = self._create_empty_state()

            conv = self._conversations[phone]
            conv["last_list_type"] = list_type
            conv["last_list_items"] = items
            conv["last_activity"] = datetime.utcnow()

    def get_list_item_by_number(self, phone: str, number: int) -> Optional[Dict]:
        """
        Get an item from the last displayed list by 1-based index.

        Args:
            phone: Phone number
            number: 1-based index (user input)

        Returns:
            Dict with item details or None if invalid
        """
        with self._lock:
            if phone not in self._conversations:
                return None

            conv = self._conversations[phone]
            items = conv.get("last_list_items", [])

            if not items or number < 1 or number > len(items):
                return None

            return items[number - 1]

    def get_last_list_type(self, phone: str) -> Optional[str]:
        """Get the type of the last displayed list."""
        with self._lock:
            if phone in self._conversations:
                return self._conversations[phone].get("last_list_type")
            return None

    def get_last_list_count(self, phone: str) -> int:
        """Get the count of items in the last displayed list."""
        with self._lock:
            if phone in self._conversations:
                return len(self._conversations[phone].get("last_list_items", []))
            return 0

    def update_menu_level(self, phone: str, level: MenuLevel):
        """Update the current menu level."""
        with self._lock:
            if phone in self._conversations:
                self._conversations[phone]["current_menu_level"] = level.value
                self._conversations[phone]["last_activity"] = datetime.utcnow()

    def get_menu_level(self, phone: str) -> str:
        """Get the current menu level."""
        with self._lock:
            if phone in self._conversations:
                return self._conversations[phone].get("current_menu_level", MenuLevel.NONE.value)
            return MenuLevel.NONE.value

    def _cleanup_expired(self):
        """Remove expired conversations (called within lock)."""
        cutoff = datetime.utcnow() - self._ttl
        expired = [
            phone for phone, conv in self._conversations.items()
            if conv["last_activity"] < cutoff
        ]
        for phone in expired:
            del self._conversations[phone]

    def get_stats(self) -> dict:
        """Get statistics about active conversations."""
        with self._lock:
            return {
                "active_conversations": len(self._conversations),
                "total_history_items": sum(
                    len(conv["history"]) for conv in self._conversations.values()
                )
            }

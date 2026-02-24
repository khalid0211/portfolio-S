"""
FastAPI service for Twilio WhatsApp webhook integration.

This service receives WhatsApp messages via Twilio webhook,
processes them using portfolio data from MotherDuck/DuckDB,
and returns AI-generated answers.

Usage:
    uvicorn main:app --reload --host 0.0.0.0 --port 8000

For local testing with ngrok:
    ngrok http 8000
    # Set the ngrok URL as your Twilio WhatsApp webhook
"""

import os
import sys
import logging
import asyncio
import time
from typing import Optional

from fastapi import FastAPI, Form, HTTPException, Request, BackgroundTasks
from fastapi.responses import Response
from twilio.twiml.messaging_response import MessagingResponse
from twilio.request_validator import RequestValidator
from twilio.rest import Client as TwilioClient

# Add parent directory for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from whatsapp.config import WhatsAppConfig
from whatsapp.conversation_manager import ConversationManager, MenuLevel
from shared.portfolio_logic import PortfolioLogic, LLMConfig

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Initialize FastAPI app
app = FastAPI(
    title="Portfolio WhatsApp Bot",
    description="WhatsApp integration for Portfolio Analysis Suite",
    version="1.0.0"
)

# Global instances (initialized on startup)
config: Optional[WhatsAppConfig] = None
portfolio_logic: Optional[PortfolioLogic] = None
conversation_manager: Optional[ConversationManager] = None
twilio_client: Optional[TwilioClient] = None


def get_currency_formatter(portfolio_id: int):
    """Get currency formatting function for a portfolio."""
    if not portfolio_id:
        return lambda v: portfolio_logic.format_currency(v, "$", "")

    try:
        settings = portfolio_logic.get_portfolio_settings(portfolio_id)
        symbol = settings.get("currency_symbol", "$")
        postfix = settings.get("currency_postfix", "")
    except Exception:
        symbol, postfix = "$", ""

    return lambda v: portfolio_logic.format_currency(v, symbol, postfix)


@app.on_event("startup")
async def startup_event():
    """Initialize services on startup."""
    global config, portfolio_logic, conversation_manager, twilio_client

    logger.info("Starting Portfolio WhatsApp Bot...")

    # Load configuration
    config = WhatsAppConfig.from_env()

    # Validate configuration
    errors = config.validate()
    if errors:
        logger.error(f"Configuration errors: {errors}")
        raise RuntimeError(f"Configuration errors: {errors}")

    logger.info(f"Allowed phones: {config.allowed_phones}")
    logger.info(f"Database: {config.motherduck_database}")

    # Initialize services
    portfolio_logic = PortfolioLogic(config.motherduck_connection_string)
    conversation_manager = ConversationManager(history_limit=10, ttl_minutes=60)

    # Initialize Twilio client for sending messages
    twilio_client = TwilioClient(config.twilio_account_sid, config.twilio_auth_token)

    logger.info("Portfolio WhatsApp Bot started successfully")


@app.on_event("shutdown")
async def shutdown_event():
    """Cleanup on shutdown."""
    global portfolio_logic
    if portfolio_logic:
        portfolio_logic.close()
    logger.info("Portfolio WhatsApp Bot shut down")


@app.get("/health")
async def health_check():
    """Health check endpoint."""
    return {
        "status": "healthy",
        "service": "portfolio-whatsapp-bot",
        "conversations": conversation_manager.get_stats() if conversation_manager else {}
    }


def send_whatsapp_message(to_phone: str, message: str):
    """Send a WhatsApp message via Twilio API. Handles long messages by splitting."""
    global twilio_client

    # Twilio sandbox limit is 1600 chars, use 1500 for safety
    MAX_LENGTH = 1500

    # If message fits, send directly
    if len(message) <= MAX_LENGTH:
        try:
            twilio_client.messages.create(
                body=message,
                from_=f"whatsapp:{config.twilio_whatsapp_number}",
                to=f"whatsapp:{to_phone}"
            )
            logger.info(f"Sent message to {to_phone}: {message[:100]}...")
        except Exception as e:
            logger.error(f"Failed to send message to {to_phone}: {e}")
        return

    # Split long messages into chunks
    chunks = []
    remaining = message
    part_num = 1

    while remaining:
        if len(remaining) <= MAX_LENGTH:
            chunks.append(remaining)
            break
        else:
            # Find a good break point (newline or space)
            cut_point = remaining.rfind('\n', 0, MAX_LENGTH - 20)
            if cut_point < MAX_LENGTH // 2:
                cut_point = remaining.rfind(' ', 0, MAX_LENGTH - 20)
            if cut_point < MAX_LENGTH // 2:
                cut_point = MAX_LENGTH - 20

            chunks.append(remaining[:cut_point].strip())
            remaining = remaining[cut_point:].strip()

    # Send each chunk with delay to ensure order
    total_parts = len(chunks)
    for i, chunk in enumerate(chunks, 1):
        try:
            if total_parts > 1:
                chunk_with_indicator = f"[{i}/{total_parts}]\n{chunk}"
            else:
                chunk_with_indicator = chunk

            twilio_client.messages.create(
                body=chunk_with_indicator,
                from_=f"whatsapp:{config.twilio_whatsapp_number}",
                to=f"whatsapp:{to_phone}"
            )
            logger.info(f"Sent message part {i}/{total_parts} to {to_phone}")

            # Wait between chunks to ensure delivery order
            if i < total_parts:
                time.sleep(1.5)
        except Exception as e:
            logger.error(f"Failed to send message part {i} to {to_phone}: {e}")


async def process_llm_question_async(phone: str, question: str):
    """Process LLM question in background and send response via Twilio API."""
    try:
        # Always fetch fresh state in background task
        conv_state = conversation_manager.get_or_create(phone)
        response_text = await handle_question(phone, question, conv_state)
        send_whatsapp_message(phone, response_text)
    except Exception as e:
        logger.error(f"Error in async LLM processing: {e}", exc_info=True)
        send_whatsapp_message(phone, f"Sorry, an error occurred: {str(e)[:200]}")


async def process_overview_async(phone: str):
    """Process project overview (with LLM trend analysis) in background."""
    try:
        # Always fetch fresh state in background task
        conv_state = conversation_manager.get_or_create(phone)
        response_text = await handle_project_overview(phone, conv_state)
        send_whatsapp_message(phone, response_text)
    except Exception as e:
        logger.error(f"Error in async overview processing: {e}", exc_info=True)
        send_whatsapp_message(phone, f"Sorry, an error occurred: {str(e)[:200]}")


@app.post("/whatsapp")
async def whatsapp_webhook(
    request: Request,
    background_tasks: BackgroundTasks,
    Body: str = Form(...),
    From: str = Form(...),
    To: str = Form(...)
):
    """
    Twilio WhatsApp webhook endpoint.

    Receives messages from Twilio and returns TwiML responses.
    For LLM questions, responds immediately and sends answer via background task.

    Twilio sends:
    - Body: Message text from user
    - From: WhatsApp number (e.g., "whatsapp:+923008405479")
    - To: Your Twilio WhatsApp number
    """
    logger.info(f"Received message from {From}: {Body[:100]}...")

    # 1. Validate Twilio signature (if enabled)
    if config.validate_twilio_signature:
        signature = request.headers.get("X-Twilio-Signature", "")
        validator = RequestValidator(config.twilio_auth_token)
        form_data = await request.form()
        url = str(request.url)

        if not validator.validate(url, dict(form_data), signature):
            logger.warning(f"Invalid Twilio signature from {From}")
            raise HTTPException(status_code=403, detail="Invalid signature")

    # 2. Extract and validate phone number
    phone = From.replace("whatsapp:", "").strip()

    if not config.is_phone_allowed(phone):
        logger.warning(f"Unauthorized phone number: {phone}")
        twiml = MessagingResponse()
        twiml.message("Your phone number is not authorized to use this service. Contact admin.")
        return Response(content=str(twiml), media_type="text/xml")

    # 3. Get or create conversation state
    conv_state = conversation_manager.get_or_create(phone)

    message = Body.strip()
    message_lower = message.lower()

    # 4. Classify message type:
    #    - FAST commands: /portfolios, /departments, /projects, /status, /clear, /help, number selections
    #    - SLOW commands: /overview (requires LLM for trend analysis)
    #    - LLM questions: free-form text questions (require LLM)

    is_fast_command = (
        message_lower in ["/portfolios", "/departments", "/projects", "/portfolio", "/status", "/clear", "/help", "help", "?"]
        or message_lower.startswith("/select ")
        or message_lower.startswith("/project ")
        or message.strip().isdigit()
    )
    is_slow_command = message_lower == "/overview"
    is_llm_question = not is_fast_command and not is_slow_command

    logger.info(f"Message from {phone}: '{message[:50]}' | fast={is_fast_command}, slow={is_slow_command}, llm={is_llm_question}")

    # 5a. FAST commands - respond immediately (but check for overview trigger)
    if is_fast_command:
        try:
            response_text = await process_message(phone, message, conv_state)

            # Check if action triggered overview (requires LLM)
            if response_text == "__OVERVIEW__":
                # Schedule background task for overview
                background_tasks.add_task(process_overview_async, phone)

                twiml = MessagingResponse()
                twiml.message("Thinking... I'll send you the project overview shortly.")
                logger.info(f"Queued overview via action for {phone}")
                return Response(content=str(twiml), media_type="text/xml")

        except Exception as e:
            logger.error(f"Error processing command: {e}", exc_info=True)
            response_text = f"Sorry, an error occurred: {str(e)[:200]}"

        twiml = MessagingResponse()
        twiml.message(response_text)
        return Response(content=str(twiml), media_type="text/xml")

    # 5b. SLOW commands (/overview) - send heartbeat, process in background
    elif is_slow_command:
        if not conv_state.get("project_id"):
            twiml = MessagingResponse()
            twiml.message("Select a project first to see its performance overview.")
            return Response(content=str(twiml), media_type="text/xml")

        # Schedule background task for overview
        background_tasks.add_task(process_overview_async, phone)

        twiml = MessagingResponse()
        twiml.message("Thinking... I'll send you the project overview shortly.")
        logger.info(f"Queued overview for {phone}, project: {conv_state.get('project_name')}")
        return Response(content=str(twiml), media_type="text/xml")

    # 5c. LLM questions - send heartbeat, process in background
    else:
        if not conv_state.get("portfolio_id"):
            twiml = MessagingResponse()
            twiml.message("No portfolio selected. Use /portfolios to see available portfolios, then select one by number.")
            return Response(content=str(twiml), media_type="text/xml")

        # Schedule background task to process LLM question
        background_tasks.add_task(process_llm_question_async, phone, message)

        twiml = MessagingResponse()
        twiml.message("Thinking... I'll send you the answer shortly.")
        logger.info(f"Queued LLM question from {phone}: {message[:50]}...")
        return Response(content=str(twiml), media_type="text/xml")


async def process_message(phone: str, message: str, conv_state: dict) -> str:
    """
    Process incoming message and generate response.

    Command syntax:
    - /portfolios     - List available portfolios (numbered)
    - /departments    - List departments in current portfolio
    - /projects       - List projects (filtered by department if selected)
    - /overview       - Show project EVM history with trend analysis
    - /portfolio      - Return to portfolio level
    - /clear          - Clear conversation history
    - /status         - Show current selection
    - /help           - Show help
    - <number>        - Select from last displayed list
    - Any other text  - Ask question about current context
    """
    message_lower = message.lower().strip()

    # Check for number-only input FIRST (for numbered list selection)
    if message.strip().isdigit():
        number = int(message.strip())
        return await handle_number_selection(phone, number, conv_state)

    # Handle commands
    if message_lower == "/portfolios":
        return handle_list_portfolios(phone)

    elif message_lower.startswith("/select "):
        portfolio_name = message[8:].strip()
        return handle_select_portfolio(phone, portfolio_name)

    elif message_lower == "/departments":
        return handle_list_departments(phone, conv_state)

    elif message_lower == "/projects":
        return handle_list_projects(phone, conv_state)

    elif message_lower.startswith("/project "):
        project_name = message[9:].strip()
        return handle_select_project(phone, project_name, conv_state)

    elif message_lower == "/overview":
        return await handle_project_overview(phone, conv_state)

    elif message_lower == "/portfolio":
        return handle_portfolio_mode(phone, conv_state)

    elif message_lower == "/clear":
        return handle_clear_history(phone)

    elif message_lower == "/status":
        return handle_status(conv_state)

    elif message_lower in ["/help", "help", "?"]:
        return get_help_text()

    # Otherwise, ask question about portfolio/project
    return await handle_question(phone, message, conv_state)


def handle_list_portfolios(phone: str) -> str:
    """List available portfolios with numbered selection."""
    try:
        df = portfolio_logic.get_available_portfolios()

        if df.empty:
            return "No portfolios found."

        lines = ["*Available Portfolios:*\n"]
        items = []

        for idx, row in enumerate(df.itertuples(), 1):
            name = row.portfolio_name
            manager = getattr(row, 'portfolio_manager', '') or ''
            if manager:
                lines.append(f"{idx}. {name} (PM: {manager})")
            else:
                lines.append(f"{idx}. {name}")
            items.append({
                "portfolio_id": row.portfolio_id,
                "portfolio_name": name,
                "portfolio_manager": manager
            })

        lines.append("\nSelect a portfolio (1-{})".format(len(df)))

        # Store list context for number-based selection
        conversation_manager.set_last_list(phone, "portfolios", items)

        return "\n".join(lines)

    except Exception as e:
        logger.error(f"Error listing portfolios: {e}", exc_info=True)
        return f"Error loading portfolios: {str(e)[:100]}"


def handle_select_portfolio(phone: str, portfolio_name: str) -> str:
    """Select a portfolio by name."""
    try:
        portfolio = portfolio_logic.get_portfolio_by_name(portfolio_name)

        if not portfolio:
            return f"Portfolio '{portfolio_name}' not found. Use /portfolios to see available options."

        portfolio_id = portfolio["portfolio_id"]

        # Get currency formatter for this portfolio
        fmt = get_currency_formatter(portfolio_id)

        # Get portfolio summary
        summary = portfolio_logic.get_portfolio_summary(portfolio_id)

        # Get latest status date
        status_date = portfolio_logic.get_latest_status_date(portfolio_id)

        # Update conversation state
        conversation_manager.set_portfolio(
            phone,
            portfolio_id,
            portfolio["portfolio_name"],
            status_date
        )

        # Build response with summary
        lines = [f"Selected: *{portfolio['portfolio_name']}*\n"]
        lines.append(f"Total Projects: {summary.get('total_projects', 0)}")
        lines.append(f"Total Budget: {fmt(summary.get('total_budget', 0))}")
        lines.append(f"Departments: {summary.get('department_count', 0)}")

        if status_date:
            date_str = status_date.strftime('%Y-%m-%d') if hasattr(status_date, 'strftime') else str(status_date).split()[0]
            lines.append(f"Latest Data: {date_str}")

        # Numbered next steps
        lines.append("\n*What would you like to do?*")
        lines.append("1. View by Department")
        lines.append("2. View all Projects")
        lines.append("3. Back to Portfolios")
        lines.append("\nOr type a question about this portfolio")

        # Store action menu for number selection
        conversation_manager.set_last_list(phone, "portfolio_actions", [
            {"action": "departments"},
            {"action": "projects"},
            {"action": "portfolios"}
        ])

        return "\n".join(lines)

    except Exception as e:
        logger.error(f"Error selecting portfolio: {e}", exc_info=True)
        return f"Error selecting portfolio: {str(e)[:100]}"


def handle_list_projects(phone: str, conv_state: dict) -> str:
    """List projects in the current portfolio/department with numbered selection and executive summary."""
    portfolio_id = conv_state.get("portfolio_id")
    department = conv_state.get("selected_department")

    if not portfolio_id:
        return "Select a Portfolio first. Use /portfolios to see available portfolios."

    try:
        # Get currency formatter for this portfolio
        fmt = get_currency_formatter(portfolio_id)

        # Get projects - filtered by department if one is selected
        if department:
            df = portfolio_logic.get_projects_by_department(portfolio_id, department)
            context = f"*Projects in {department}:*\n"
        else:
            df = portfolio_logic.get_projects_in_portfolio(portfolio_id)
            context = f"*Projects in {conv_state.get('portfolio_name', 'Portfolio')}:*\n"

        if df.empty:
            return f"No projects found."

        lines = [context]
        items = []
        total_budget = 0

        for idx, row in enumerate(df.itertuples(), 1):
            name = row.project_name
            budget = getattr(row, 'current_budget', 0) or 0
            total_budget += budget
            budget_str = fmt(budget)

            lines.append(f"{idx}. {name} ({budget_str})")
            items.append({
                "project_id": row.project_id,
                "project_name": name,
                "current_budget": budget
            })

        # Executive Summary
        lines.append(f"\n*Summary:* {len(df)} Projects | Total Budget: {fmt(total_budget)}")
        lines.append("\nSelect a project (1-{}) or:".format(len(df)))

        # Add navigation options based on context
        nav_idx = len(df) + 1
        if department:
            lines.append(f"{nav_idx}. Back to Departments")
            items.append({"action": "departments"})
            nav_idx += 1

        lines.append(f"{nav_idx}. Back to {conv_state.get('portfolio_name', 'Portfolio')}")
        items.append({"action": "portfolio"})

        # Store list context for number-based selection
        conversation_manager.set_last_list(phone, "projects", items)

        return "\n".join(lines)

    except Exception as e:
        logger.error(f"Error listing projects: {e}", exc_info=True)
        return f"Error loading projects: {str(e)[:100]}"


def handle_select_project(phone: str, project_name: str, conv_state: dict) -> str:
    """Select a project for project-specific mode - shows quick confirmation with next steps."""
    portfolio_id = conv_state.get("portfolio_id")

    if not portfolio_id:
        return "No portfolio selected. Use /portfolios and /select first."

    try:
        project = portfolio_logic.get_project_by_name(portfolio_id, project_name)

        if not project:
            return f"Project '{project_name}' not found. Use /projects to see available options."

        # Update conversation state
        conversation_manager.set_project(
            phone,
            project["project_id"],
            project["project_name"]
        )

        # Get context for navigation
        department = conv_state.get("selected_department")
        portfolio_name = conv_state.get("portfolio_name", "Portfolio")

        # Quick confirmation with numbered next steps
        lines = [f"Selected: *{project['project_name']}*"]

        lines.append("\n*What would you like to do?*")
        lines.append("1. View Performance Overview")
        lines.append("2. Back to Projects")

        if department:
            lines.append("3. Back to Departments")
            lines.append(f"4. Back to {portfolio_name}")
            actions = [
                {"action": "overview"},
                {"action": "projects"},
                {"action": "departments"},
                {"action": "portfolio"}
            ]
        else:
            lines.append(f"3. Back to {portfolio_name}")
            lines.append("4. Back to Portfolios")
            actions = [
                {"action": "overview"},
                {"action": "projects"},
                {"action": "portfolio"},
                {"action": "portfolios"}
            ]

        lines.append("\nOr type a question about this project")

        # Store action menu for number selection
        conversation_manager.set_last_list(phone, "project_actions", actions)

        return "\n".join(lines)

    except Exception as e:
        logger.error(f"Error selecting project: {e}", exc_info=True)
        return f"Error selecting project: {str(e)[:100]}"


def handle_portfolio_mode(phone: str, conv_state: dict) -> str:
    """Return to portfolio overview mode."""
    if not conv_state.get("portfolio_id"):
        return "No portfolio selected. Use /portfolios to get started."

    portfolio_name = conv_state.get("portfolio_name")
    portfolio_id = conv_state.get("portfolio_id")

    conversation_manager.clear_project(phone)
    conversation_manager.clear_department(phone)

    # Get currency formatter for this portfolio
    fmt = get_currency_formatter(portfolio_id)

    # Get fresh summary
    summary = portfolio_logic.get_portfolio_summary(portfolio_id)

    lines = [f"*{portfolio_name}*\n"]
    lines.append(f"Total Projects: {summary.get('total_projects', 0)}")
    lines.append(f"Total Budget: {fmt(summary.get('total_budget', 0))}")
    lines.append(f"Departments: {summary.get('department_count', 0)}")

    # Numbered next steps
    lines.append("\n*What would you like to do?*")
    lines.append("1. View by Department")
    lines.append("2. View all Projects")
    lines.append("3. Back to Portfolios")
    lines.append("\nOr type a question about this portfolio")

    # Store action menu for number selection
    conversation_manager.set_last_list(phone, "portfolio_actions", [
        {"action": "departments"},
        {"action": "projects"},
        {"action": "portfolios"}
    ])

    return "\n".join(lines)


def handle_list_departments(phone: str, conv_state: dict) -> str:
    """List departments (responsible_organization) with project counts and budgets."""
    portfolio_id = conv_state.get("portfolio_id")

    if not portfolio_id:
        return "Select a Portfolio first. Use /portfolios to see available portfolios."

    try:
        # Get currency formatter for this portfolio
        fmt = get_currency_formatter(portfolio_id)

        df = portfolio_logic.get_departments_in_portfolio(portfolio_id)

        if df.empty:
            return f"No departments found in {conv_state.get('portfolio_name', 'this portfolio')}."

        lines = [f"*Departments in {conv_state.get('portfolio_name', 'Portfolio')}:*\n"]
        items = []
        total_projects = 0
        total_budget = 0

        for idx, row in enumerate(df.itertuples(), 1):
            dept = row.department
            count = row.project_count
            budget = row.total_budget or 0
            total_projects += count
            total_budget += budget

            lines.append(f"{idx}. {dept} ({count} projects, {fmt(budget)})")
            items.append({
                "department": dept,
                "project_count": count,
                "total_budget": budget
            })

        # Executive Summary
        lines.append(f"\n*Summary:* {len(df)} Departments | {total_projects} Projects | {fmt(total_budget)}")
        lines.append("\nSelect a department (1-{}) or:".format(len(df)))
        lines.append(f"{len(df) + 1}. View all Projects")
        lines.append(f"{len(df) + 2}. Back to {conv_state.get('portfolio_name', 'Portfolio')}")

        # Add navigation actions to the list
        items.append({"action": "projects"})
        items.append({"action": "portfolio"})

        # Store list context for number-based selection
        conversation_manager.set_last_list(phone, "departments", items)

        return "\n".join(lines)

    except Exception as e:
        logger.error(f"Error listing departments: {e}", exc_info=True)
        return f"Error loading departments: {str(e)[:100]}"


async def handle_number_selection(phone: str, number: int, conv_state: dict) -> str:
    """Handle numeric input for list selection or action menus."""
    last_list_type = conversation_manager.get_last_list_type(phone)
    list_count = conversation_manager.get_last_list_count(phone)

    if not last_list_type or list_count == 0:
        return "No options available. Type /help to see commands."

    if number < 1 or number > list_count:
        return f"Invalid selection. Please enter a number between 1 and {list_count}."

    item = conversation_manager.get_list_item_by_number(phone, number)

    if not item:
        return "Error retrieving selection. Please try again."

    # Check if this is a navigation action (embedded in any list type)
    if "action" in item:
        return await handle_action_selection(phone, item, conv_state)

    # Handle data list selections
    if last_list_type == "portfolios":
        return handle_select_portfolio_by_item(phone, item)
    elif last_list_type == "departments":
        return handle_select_department_by_item(phone, item, conv_state)
    elif last_list_type == "projects":
        return handle_select_project_by_item(phone, item, conv_state)

    # Handle action-only menus
    elif last_list_type in ["portfolio_actions", "department_actions", "project_actions"]:
        return await handle_action_selection(phone, item, conv_state)

    else:
        return "Unknown context."


async def handle_action_selection(phone: str, item: dict, conv_state: dict) -> str:
    """Handle action menu selections (numbered next steps).

    Returns:
        str: Response message, or special marker "__OVERVIEW__" to trigger background processing
    """
    action = item.get("action")

    if action == "departments":
        return handle_list_departments(phone, conv_state)
    elif action == "projects":
        return handle_list_projects(phone, conv_state)
    elif action == "portfolios":
        return handle_list_portfolios(phone)
    elif action == "portfolio":
        return handle_portfolio_mode(phone, conv_state)
    elif action == "overview":
        # Return special marker - webhook will handle background processing
        return "__OVERVIEW__"
    else:
        return "Unknown action."


def handle_select_portfolio_by_item(phone: str, item: dict) -> str:
    """Handle portfolio selection after number input."""
    try:
        portfolio_id = item["portfolio_id"]
        portfolio_name = item["portfolio_name"]

        # Get currency formatter for this portfolio
        fmt = get_currency_formatter(portfolio_id)

        # Get portfolio summary
        summary = portfolio_logic.get_portfolio_summary(portfolio_id)

        # Get latest status date
        status_date = portfolio_logic.get_latest_status_date(portfolio_id)

        # Update conversation state
        conversation_manager.set_portfolio(phone, portfolio_id, portfolio_name, status_date)

        # Build response with summary
        lines = [f"Selected: *{portfolio_name}*\n"]
        lines.append(f"Total Projects: {summary.get('total_projects', 0)}")
        lines.append(f"Total Budget: {fmt(summary.get('total_budget', 0))}")
        lines.append(f"Departments: {summary.get('department_count', 0)}")

        if status_date:
            date_str = status_date.strftime('%Y-%m-%d') if hasattr(status_date, 'strftime') else str(status_date).split()[0]
            lines.append(f"Latest Data: {date_str}")

        # Numbered next steps
        lines.append("\n*What would you like to do?*")
        lines.append("1. View by Department")
        lines.append("2. View all Projects")
        lines.append("3. Back to Portfolios")
        lines.append("\nOr type a question about this portfolio")

        # Store action menu for number selection
        conversation_manager.set_last_list(phone, "portfolio_actions", [
            {"action": "departments"},
            {"action": "projects"},
            {"action": "portfolios"}
        ])

        return "\n".join(lines)

    except Exception as e:
        logger.error(f"Error selecting portfolio by item: {e}", exc_info=True)
        return f"Error selecting portfolio: {str(e)[:100]}"


def handle_select_department_by_item(phone: str, item: dict, conv_state: dict) -> str:
    """Handle department selection after number input."""
    try:
        department = item["department"]
        portfolio_id = conv_state.get("portfolio_id")
        portfolio_name = conv_state.get("portfolio_name", "Portfolio")

        # Get currency formatter for this portfolio
        fmt = get_currency_formatter(portfolio_id)

        # Update state
        conversation_manager.set_department(phone, department)

        # Get department summary
        summary = portfolio_logic.get_department_summary(portfolio_id, department)

        lines = [f"Selected Department: *{department}*\n"]
        lines.append(f"Projects: {summary.get('project_count', 0)}")
        lines.append(f"Total Budget: {fmt(summary.get('total_budget', 0))}")

        # Numbered next steps
        lines.append("\n*What would you like to do?*")
        lines.append("1. View Projects in this Department")
        lines.append("2. Back to Departments")
        lines.append(f"3. Back to {portfolio_name}")
        lines.append("\nOr type a question about this department")

        # Store action menu for number selection
        conversation_manager.set_last_list(phone, "department_actions", [
            {"action": "projects"},
            {"action": "departments"},
            {"action": "portfolio"}
        ])

        return "\n".join(lines)

    except Exception as e:
        logger.error(f"Error selecting department: {e}", exc_info=True)
        return f"Error selecting department: {str(e)[:100]}"


def handle_select_project_by_item(phone: str, item: dict, conv_state: dict) -> str:
    """Handle project selection after number input - shows quick confirmation with next steps."""
    try:
        project_id = item["project_id"]
        project_name = item["project_name"]
        budget = item.get("current_budget", 0)

        # Get currency formatter for this portfolio
        portfolio_id = conv_state.get("portfolio_id")
        fmt = get_currency_formatter(portfolio_id)

        # Update state
        conversation_manager.set_project(phone, project_id, project_name)

        # Get context for navigation
        department = conv_state.get("selected_department")
        portfolio_name = conv_state.get("portfolio_name", "Portfolio")

        # Quick confirmation with numbered next steps
        lines = [f"Selected: *{project_name}*"]
        if budget:
            lines.append(f"Budget: {fmt(budget)}")

        lines.append("\n*What would you like to do?*")
        lines.append("1. View Performance Overview")
        lines.append("2. Back to Projects")

        if department:
            lines.append("3. Back to Departments")
            lines.append(f"4. Back to {portfolio_name}")
            actions = [
                {"action": "overview"},
                {"action": "projects"},
                {"action": "departments"},
                {"action": "portfolio"}
            ]
        else:
            lines.append(f"3. Back to {portfolio_name}")
            lines.append("4. Back to Portfolios")
            actions = [
                {"action": "overview"},
                {"action": "projects"},
                {"action": "portfolio"},
                {"action": "portfolios"}
            ]

        lines.append("\nOr type a question about this project")

        # Store action menu for number selection
        conversation_manager.set_last_list(phone, "project_actions", actions)

        return "\n".join(lines)

    except Exception as e:
        logger.error(f"Error selecting project: {e}", exc_info=True)
        return f"Error selecting project: {str(e)[:100]}"


async def handle_project_overview(phone: str, conv_state: dict) -> str:
    """Show project EVM history with AI-generated trend commentary."""
    project_id = conv_state.get("project_id")
    project_name = conv_state.get("project_name")

    if not project_id:
        return "Select a project first. Use /projects and select one by number."

    try:
        # Get currency formatter for this portfolio
        portfolio_id = conv_state.get("portfolio_id")
        fmt = get_currency_formatter(portfolio_id)

        # Get EVM history (last 10 periods)
        df = portfolio_logic.get_project_evm_history(project_id, limit=10)

        if df.empty:
            return f"No status reports found for {project_name}."

        lines = [f"*Performance History: {project_name}*\n"]

        # Format EVM data table
        for _, row in df.iterrows():
            # Format date without time
            status_dt = row['status_date']
            if hasattr(status_dt, 'strftime'):
                date_str = status_dt.strftime('%Y-%m-%d')
            else:
                date_str = str(status_dt).split()[0]  # Take only date part

            pv = fmt(row['pv'])
            ac = fmt(row['ac'])
            ev = fmt(row['ev'])
            spi = f"{row['spi']:.2f}" if row['spi'] is not None else "N/A"
            cpi = f"{row['cpi']:.2f}" if row['cpi'] is not None else "N/A"

            lines.append(f"*{date_str}*")
            lines.append(f"  PV: {pv} | AC: {ac} | EV: {ev}")
            lines.append(f"  SPI: {spi} | CPI: {cpi}")

        # Generate AI trend commentary if enough data points
        if len(df) >= 2:
            trend_commentary = await generate_trend_commentary(df, project_name, conv_state)
            lines.append(f"\n*Trend Analysis:*\n{trend_commentary}")
        else:
            lines.append("\n_Insufficient data points for trend analysis._")

        # Numbered next steps
        department = conv_state.get("selected_department")
        portfolio_name = conv_state.get("portfolio_name", "Portfolio")

        lines.append("\n*What would you like to do?*")
        lines.append("1. Back to Projects")
        if department:
            lines.append("2. Back to Departments")
            lines.append(f"3. Back to {portfolio_name}")
            actions = [
                {"action": "projects"},
                {"action": "departments"},
                {"action": "portfolio"}
            ]
        else:
            lines.append(f"2. Back to {portfolio_name}")
            lines.append("3. Back to Portfolios")
            actions = [
                {"action": "projects"},
                {"action": "portfolio"},
                {"action": "portfolios"}
            ]
        lines.append("\nOr type a question about this project")

        # Store action menu for number selection
        conversation_manager.set_last_list(phone, "project_actions", actions)

        return "\n".join(lines)

    except Exception as e:
        logger.error(f"Error generating project overview: {e}", exc_info=True)
        return f"Error loading project overview: {str(e)[:100]}"


async def generate_trend_commentary(evm_df, project_name: str, conv_state: dict) -> str:
    """Generate AI commentary on SPI/CPI trends."""
    try:
        # Build trend context for LLM
        trend_lines = []
        for _, row in evm_df.iterrows():
            # Format date without time
            status_dt = row['status_date']
            if hasattr(status_dt, 'strftime'):
                date_str = status_dt.strftime('%Y-%m-%d')
            else:
                date_str = str(status_dt).split()[0]
            spi = f"{row['spi']:.2f}" if row['spi'] is not None else "N/A"
            cpi = f"{row['cpi']:.2f}" if row['cpi'] is not None else "N/A"
            trend_lines.append(f"- {date_str}: SPI={spi}, CPI={cpi}")

        trend_context = f"""PROJECT: {project_name}
EVM PERFORMANCE HISTORY (oldest to newest):
{chr(10).join(trend_lines)}

Key metrics:
- SPI (Schedule Performance Index): >1 = ahead of schedule, <1 = behind schedule
- CPI (Cost Performance Index): >1 = under budget, <1 = over budget
"""

        # Get LLM config
        portfolio_id = conv_state.get("portfolio_id")
        settings = portfolio_logic.get_portfolio_settings(portfolio_id) if portfolio_id else {}
        llm_config_dict = settings.get("llm_config", {})

        api_key = llm_config_dict.get("api_key", "")
        provider = llm_config_dict.get("provider", "OpenAI")
        model = llm_config_dict.get("model", "gpt-4o-mini")

        # Fallback to environment variable
        if not api_key and config.openai_api_key:
            api_key = config.openai_api_key
            provider = "OpenAI"
            model = "gpt-4o-mini"

        if not api_key:
            return "LLM not configured for trend analysis."

        llm_config = LLMConfig(
            provider=provider,
            model=model,
            api_key=api_key,
            temperature=0.2,
            timeout=30
        )

        # Call LLM for trend analysis
        answer = portfolio_logic.ask_question(
            question="Provide a concise 2-3 sentence trend analysis of this project's schedule and cost performance. Focus on whether performance is improving or declining and any concerns.",
            context=trend_context,
            conversation_history=[],
            llm_config=llm_config
        )

        return answer

    except Exception as e:
        logger.error(f"Error generating trend commentary: {e}", exc_info=True)
        return "Unable to generate trend analysis."


def handle_clear_history(phone: str) -> str:
    """Clear conversation history."""
    conversation_manager.clear_history(phone)
    return "Conversation history cleared. Ask a new question!"


def handle_status(conv_state: dict) -> str:
    """Show current selection status."""
    portfolio_name = conv_state.get("portfolio_name")
    project_name = conv_state.get("project_name")
    department = conv_state.get("selected_department")
    status_date = conv_state.get("status_date")
    menu_level = conv_state.get("current_menu_level", "NONE")
    history_count = len(conv_state.get("history", []))

    if not portfolio_name:
        return "No portfolio selected. Use /portfolios to get started."

    lines = ["*Current Selection:*"]
    lines.append(f"Portfolio: {portfolio_name}")

    if department:
        lines.append(f"Department: {department}")

    if project_name:
        lines.append(f"Project: {project_name}")

    lines.append(f"Navigation Level: {menu_level}")

    if status_date:
        date_str = status_date.strftime('%Y-%m-%d') if hasattr(status_date, 'strftime') else str(status_date).split()[0]
        lines.append(f"Data date: {date_str}")

    lines.append(f"Chat history: {history_count} exchanges")

    return "\n".join(lines)


def get_help_text() -> str:
    """Return help text."""
    return """*Portfolio Bot - Help*

*Getting Started:*
Type /portfolios to see available portfolios, then navigate using numbers!

*Navigation:*
Every screen shows numbered options. Just reply with a number to:
• Select items from lists
• Choose next actions (View Overview, Back, etc.)
• Navigate between levels

*Example Flow:*
/portfolios → "1" (select portfolio) → "1" (view departments) → "2" (select dept) → "1" (view projects) → "3" (select project) → "1" (view overview)

*Quick Commands:*
/portfolios - Start here
/overview - View project performance
/status - Show current selection
/clear - Clear chat history
/help - Show this help

*Ask Questions:*
Type any question about your current selection:
• "What is the overall health?"
• "Which projects are at risk?"

_Note: Questions and overviews show "Thinking..." while loading._
"""


async def handle_question(phone: str, question: str, conv_state: dict) -> str:
    """Ask LLM about portfolio/project data."""
    portfolio_id = conv_state.get("portfolio_id")
    project_id = conv_state.get("project_id")
    status_date = conv_state.get("status_date")

    if not portfolio_id:
        return "No portfolio selected. Use /portfolios to see available portfolios, then /select <name> to choose one."

    # Get portfolio settings for LLM config and currency
    try:
        settings = portfolio_logic.get_portfolio_settings(portfolio_id)
    except Exception as e:
        logger.error(f"Error loading settings: {e}")
        settings = {"currency_symbol": "$", "currency_postfix": "", "llm_config": {}}

    currency = settings.get("currency_symbol", "$")
    postfix = settings.get("currency_postfix", "")

    # Build context based on mode
    try:
        if project_id:
            # Project-specific mode
            context = portfolio_logic.build_project_context(
                project_id,
                status_date,
                currency,
                postfix
            )
        else:
            # Portfolio overview mode
            context = portfolio_logic.build_portfolio_context(
                portfolio_id,
                status_date,
                currency,
                postfix
            )
    except Exception as e:
        logger.error(f"Error building context: {e}", exc_info=True)
        return f"Error loading data: {str(e)[:100]}"

    # Get LLM config
    llm_config_dict = settings.get("llm_config", {})

    # Build LLMConfig, using fallback if needed
    api_key = llm_config_dict.get("api_key", "")
    provider = llm_config_dict.get("provider", "OpenAI")
    model = llm_config_dict.get("model", "gpt-4o-mini")

    # Fallback to environment variable if no portfolio LLM config
    if not api_key and config.openai_api_key:
        api_key = config.openai_api_key
        provider = "OpenAI"
        model = "gpt-4o-mini"

    if not api_key:
        return (
            "LLM is not configured for this portfolio.\n"
            "Please configure LLM settings in Portfolio Management, "
            "or set OPENAI_API_KEY in the environment."
        )

    llm_config = LLMConfig(
        provider=provider,
        model=model,
        api_key=api_key,
        temperature=float(llm_config_dict.get("temperature", 0.2)),
        timeout=int(llm_config_dict.get("timeout", 60))
    )

    # Get conversation history
    history = conversation_manager.get_history(phone)

    # Generate answer
    try:
        answer = portfolio_logic.ask_question(
            question=question,
            context=context,
            conversation_history=history,
            llm_config=llm_config
        )

        # Update history
        conversation_manager.add_to_history(phone, question, answer)

        # Note: Long messages are automatically split by send_whatsapp_message()
        # Only truncate extremely long responses (>4500 chars) to avoid too many parts
        if len(answer) > 4500:
            answer = answer[:4400] + "\n\n_[Response truncated]_"

        return answer

    except Exception as e:
        logger.error(f"Error generating answer: {e}", exc_info=True)
        return f"Error generating answer: {str(e)[:200]}"


# Entry point for uvicorn
if __name__ == "__main__":
    import uvicorn

    # Load config for port
    cfg = WhatsAppConfig.from_env()
    uvicorn.run(
        "main:app",
        host=cfg.host,
        port=cfg.port,
        reload=True
    )

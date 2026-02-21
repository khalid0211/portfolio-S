# WhatsApp Integration for Portfolio Analysis Suite

FastAPI service that bridges Twilio WhatsApp with your portfolio data via LLM.

## Features

- Query portfolio and project EVM data via WhatsApp
- Multi-portfolio support with `/select` command
- Project-specific mode with `/project` command
- Conversation history for contextual follow-up questions
- Phone number whitelist authentication
- Multi-provider LLM support (OpenAI, Claude, Gemini, Kimi)

## Quick Start

### 1. Install Dependencies

```bash
cd whatsapp
pip install -r requirements.txt
```

### 2. Configure Environment

```bash
cp .env.example .env
# Edit .env with your actual credentials
```

Required settings:
- `TWILIO_ACCOUNT_SID` and `TWILIO_AUTH_TOKEN` from [Twilio Console](https://console.twilio.com/)
- `MOTHERDUCK_TOKEN` from [MotherDuck](https://motherduck.com/)
- `ALLOWED_PHONES` - comma-separated phone numbers with country code
- `OPENAI_API_KEY` (or configure LLM per portfolio in the main app)

### 3. Start the Server

```bash
# From the whatsapp directory
uvicorn main:app --reload --host 0.0.0.0 --port 8000

# Or run directly
python main.py
```

### 4. Set Up ngrok (for local testing)

```bash
# Install ngrok from https://ngrok.com/
ngrok http 8000
```

Copy the ngrok URL (e.g., `https://abc123.ngrok.io`)

### 5. Configure Twilio Sandbox

1. Go to [Twilio Console > WhatsApp Sandbox](https://console.twilio.com/us1/develop/sms/try-it-out/whatsapp-learn)
2. Set webhook URL to: `https://<your-ngrok-url>/whatsapp`
3. Method: POST
4. Save

### 6. Test via WhatsApp

1. Send "join <sandbox-keyword>" to your Twilio sandbox number
2. Send `/help` to see commands
3. Send `/portfolios` to list available portfolios
4. Send `/select <portfolio-name>` to select one
5. Ask questions like "What is the portfolio health?"

## Commands

| Command | Description |
|---------|-------------|
| `/portfolios` | List available portfolios |
| `/select <name>` | Select a portfolio |
| `/projects` | List projects in portfolio |
| `/project <name>` | Focus on specific project |
| `/portfolio` | Return to portfolio overview |
| `/status` | Show current selection |
| `/clear` | Clear conversation history |
| `/help` | Show help |

Any other text is treated as a question about the current portfolio/project.

## Architecture

```
whatsapp/
  main.py                    # FastAPI app with /whatsapp endpoint
  config.py                  # Configuration loader from .env
  conversation_manager.py    # Per-phone conversation state
  requirements.txt           # Python dependencies

shared/
  portfolio_logic.py         # Streamlit-free LLM + DB logic
```

## Security

- **Phone Whitelist**: Only numbers in `ALLOWED_PHONES` can use the bot
- **Twilio Signature**: Enable `VALIDATE_TWILIO_SIGNATURE=true` in production
- **No Hardcoded Keys**: All credentials come from environment variables

## Troubleshooting

### "Your phone number is not authorized"
Add your phone number to `ALLOWED_PHONES` in `.env` (with country code, e.g., `+923008405479`)

### "LLM is not configured"
Either:
1. Set `OPENAI_API_KEY` in `.env`, or
2. Configure LLM settings in Portfolio Management > Settings in the main app

### Database connection errors
Verify your `MOTHERDUCK_TOKEN` is valid and the database exists.

### Webhook not receiving messages
1. Check ngrok is running and URL is correct
2. Verify Twilio sandbox webhook is set to POST
3. Check Twilio console for webhook errors

## Production Deployment

For production:

1. Set `VALIDATE_TWILIO_SIGNATURE=true`
2. Use a production Twilio WhatsApp number (not sandbox)
3. Deploy behind HTTPS (required by Twilio)
4. Consider adding rate limiting
5. Use persistent storage for conversations (Redis) if scaling

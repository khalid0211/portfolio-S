# WhatsApp Integration Setup Guide

## Quick Start (Running Again)

```bash
# 1. Open terminal and navigate to whatsapp folder
cd F:\coding\portfolio-S\whatsapp

# 2. Start the FastAPI server
uvicorn main:app --reload --port 8000

# 3. In a NEW terminal, start ngrok tunnel
ngrok http 8000

# 4. Copy the ngrok URL (e.g., https://abc123.ngrok-free.app)
#    Go to Twilio Console > WhatsApp Sandbox
#    Update webhook to: https://<ngrok-url>/whatsapp
#    Method: POST
#    Save

# 5. Send message in WhatsApp to test
```

**Twilio Sandbox Settings:** https://console.twilio.com/us1/develop/sms/try-it-out/whatsapp-learn

---

## First Time Setup

### 1. Install Dependencies
```bash
cd F:\coding\portfolio-S\whatsapp
pip install -r requirements.txt
```

### 2. Configure Environment
```bash
cp .env.example .env
# Edit .env with your credentials
```

### 3. Required Environment Variables (.env)
```bash
# Twilio Configuration
TWILIO_ACCOUNT_SID=ACxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx
TWILIO_AUTH_TOKEN=your_auth_token_here
TWILIO_WHATSAPP_NUMBER=+14155238886

# Database Configuration
MOTHERDUCK_TOKEN=your_motherduck_token_here
MOTHERDUCK_DATABASE=portfolio_cloud

# Authentication - Allowed Phone Numbers
ALLOWED_PHONES=+923008405479,+923713044469

# LLM Configuration (Fallback)
OPENAI_API_KEY=sk-your_openai_key_here

# Server Configuration
HOST=0.0.0.0
PORT=8000
VALIDATE_TWILIO_SIGNATURE=false
```

---

## WhatsApp Commands

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

**Any other text** is treated as a question for the AI.

---

## Example Usage

1. Send `/portfolios` to see available portfolios
2. Send `/select World Bank Pakistan` to select a portfolio
3. Ask questions like:
   - "What is the overall health of the portfolio?"
   - "Which projects are at risk?"
   - "List all federal projects sorted by value"
   - "Write an email to the sponsor about top 5 projects"

**Note:** For AI questions, you'll first receive "Thinking..." then the answer arrives shortly after.

---

## File Structure

```
portfolio-S/
  whatsapp/
    main.py                 # FastAPI app with /whatsapp endpoint
    config.py               # Configuration loader
    conversation_manager.py # Per-phone conversation state
    .env                    # Your credentials (don't commit!)
    .env.example            # Template
    requirements.txt        # Dependencies
    README.md               # Detailed docs
  shared/
    portfolio_logic.py      # LLM + DB logic (Streamlit-free)
```

---

## Troubleshooting

### "Your phone number is not authorized"
Add your phone to `ALLOWED_PHONES` in `.env` (with country code, e.g., `+923008405479`)

### "LLM is not configured"
Set `OPENAI_API_KEY` in `.env`, or configure LLM in Portfolio Management > Settings

### No response in WhatsApp
- Check terminal for errors
- Verify ngrok is running
- Check Twilio webhook URL ends with `/whatsapp`
- Check Twilio logs: https://console.twilio.com/us1/monitor/logs/messaging

### Messages arrive out of order
Long messages are split into parts with 1.5s delay - they should arrive in order

### Sandbox session expired
Re-send the join message: `join <your-sandbox-keyword>`

---

## Architecture Notes

- **Commands** (starting with `/`): Respond immediately via TwiML
- **AI Questions**: Respond with "Thinking...", then send answer via Twilio API (avoids timeout)
- **Long Messages**: Auto-split into chunks of ~1500 chars with `[1/2]`, `[2/2]` indicators
- **Conversation History**: Maintained per phone number for contextual follow-ups

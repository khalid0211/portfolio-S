Portfolio Analysis Suite — User Guide (Quick Start)

Overview
- A Streamlit-based suite for project portfolio analysis and executive reporting.
- Sign in with Google, then navigate between tools like File Management, Manual Data Entry, Portfolio Analytics, Gantt Charts, Cash Flow Simulator, and EVM Calculator.

Quick Start
- Install Python 3.10+ and pip.
- Install dependencies: `pip install -r requirements.txt`.
- Run the app: `streamlit run Main.py`.
- Sign in with Google when prompted. If a page is locked, contact your admin for access.

Navigation
- Home shows tool cards. Click a card’s Open button to go to that page.
- Common flow: File Management → Portfolio Analysis/Charts → Export.

Data
- Supported: CSV or JSON. See `docs/data_dictionary.md:1` for required fields and types.
- Samples: `data/Portfolio_Demo.csv:1`, `data/Portfolio_Demo.json:1`.

Key Pages (For Users)
- File Management: import data, map columns, configure controls, run batch EVM, export results.
- Manual Data Entry: add/edit projects directly without files.
- Project/Portfolio Analysis: view metrics, comparisons, summaries.
- Portfolio Gantt: timeline visualization with filters and hover details.
- Cash Flow Simulator: model scenarios (delays, inflation, patterns) and compare to baseline.
- EVM Calculator: compute EVM metrics and forecasts interactively.

Help & Troubleshooting
- Can’t access a page: your admin must grant permission.
- OAuth error: confirm you used the sign-in button and try again.
- Data import issues: verify required fields (see `docs/data_dictionary.md:1`).

More Documentation
- Detailed user instructions: `docs/User_Guide.md:1`.

Documentation Site (Optional)
- A MkDocs config is included in `mkdocs.yml:1`.
- To preview locally:
  - Install: `pip install mkdocs mkdocs-material`
  - Serve: `mkdocs serve` and open the local URL shown
  - Main pages: Home, User Guide, First Run, Data Dictionary

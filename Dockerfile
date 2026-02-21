# Use full Python image (slim has glibc issues with MotherDuck extension)
FROM python:3.11-bookworm

# Set the working directory inside the container
WORKDIR /app

# 1. Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    ca-certificates \
    dos2unix \
    && rm -rf /var/lib/apt/lists/*

# 2. Install Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# 3. Pre-install MotherDuck extension for DuckDB
RUN python -c "import duckdb; duckdb.execute('INSTALL motherduck'); duckdb.execute('LOAD motherduck')"

# 4. Create a directory for local DuckDB files (ephemeral storage)
RUN mkdir -p /tmp/duckdb

# 5. Copy your code
COPY . .

# 6. Convert Windows line endings (CRLF) to Unix (LF)
RUN find /app -type f -name "*.py" -exec dos2unix -v {} \; && \
    echo "Line ending conversion complete"

# --- CRITICAL CONFIGS ---
ENV PYTHONUNBUFFERED=1
ENV PYTHONPATH=/app

# 7. Expose port
EXPOSE 8501

# 8. Run the Streamlit app
CMD ["streamlit", "run", "main.py", "--server.port=8080", "--server.address=0.0.0.0"]
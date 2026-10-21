FROM python:3.12-slim

ENV PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1
WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .
# Download the public dataset and build the lakehouse at image build time.
RUN python scripts/download_data.py && python -m lakehouse_agent.build_lakehouse

EXPOSE 8501
# Runs in offline demo mode unless ANTHROPIC_API_KEY (or another provider key) is passed with -e.
CMD ["streamlit", "run", "app.py", "--server.address=0.0.0.0", "--server.port=8501", "--server.headless=true"]

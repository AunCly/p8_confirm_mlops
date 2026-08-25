FROM python:3.13-slim

WORKDIR /app

RUN apt-get update && apt-get install -y \
    libpq-dev \
    gcc \
    && rm -rf /var/lib/apt/lists/*

COPY --from=ghcr.io/astral-sh/uv:latest /uv /uvx /bin/

# On copie directement depuis la racine du contexte
COPY pyproject.toml uv.lock ./
RUN uv sync --frozen

# On copie tout le reste du code
COPY . .

#FastApi
CMD ["uv", "run", "-m", "streamlit", "run", "dashboard/app.py"]
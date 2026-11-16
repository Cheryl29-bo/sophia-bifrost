FROM python:3.11-slim-bookworm

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    BIFROST_TRANSPORT_VERSION=v2.4.0

RUN apt-get update \
    && apt-get install -y --no-install-recommends ca-certificates nodejs npm \
    && rm -rf /var/lib/apt/lists/*

RUN npm install --global @maximhq/bifrost@1.6.3

WORKDIR /app

COPY requirements.txt ./
RUN python -m pip install --no-cache-dir -r requirements.txt

COPY main.py ./
COPY start.sh ./

RUN chmod 755 start.sh \
    && useradd --create-home --uid 10001 --shell /usr/sbin/nologin sophia \
    && mkdir -p /app/bifrost-data \
    && chown -R sophia:sophia /app/bifrost-data

USER sophia

RUN bifrost -help

EXPOSE 8000

CMD ["./start.sh"]

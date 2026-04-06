#!/bin/sh
set -e

echo "Running database migrations..."
/app/.venv/bin/python scripts/provision_db.py

echo "Starting combined Honcho app container..."
exec /app/.venv/bin/python /app/docker/run_combined.py

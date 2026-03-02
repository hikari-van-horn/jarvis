#!/usr/bin/env bash
# scripts/test.sh

echo "Running unit tests with pytest..."
poetry run pytest tests/ -v

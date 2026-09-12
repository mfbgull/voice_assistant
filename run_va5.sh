#!/bin/bash
# Launcher script for Voice Assistant v5
PROJECT_DIR="$(cd "$(dirname "$0")" && pwd)"
cd "$PROJECT_DIR"
exec va-env/bin/python va5.py "$@"

#!/bin/bash
# Voice Assistant Launcher

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
VENV_PYTHON="$SCRIPT_DIR/va-env/bin/python"
VENV_PYTHON_ALT="$SCRIPT_DIR/.va-env/bin/python"

# Find virtual environment Python
if [ -f "$VENV_PYTHON" ]; then
    PYTHON_PATH="$VENV_PYTHON"
elif [ -f "$VENV_PYTHON_ALT" ]; then
    PYTHON_PATH="$VENV_PYTHON_ALT"
else
    echo "========================================"
    echo "      Voice Assistant Launcher"
    echo "========================================"
    echo "ERROR: Virtual environment not found!"
    echo "Please create it first:"
    echo "  python -m venv va-env"
    echo "  source va-env/bin/activate"
    echo "  pip install -r requirements.txt"
    exit 1
fi

echo "========================================"
echo "      Voice Assistant Launcher"
echo "========================================"
echo ""
echo "  1. va1.py - Basic version"
echo "  2. va2.py - Enhanced (recommended)"
echo "  3. va3.py - Model selector version"
echo "  4. va4.py - Legacy version"
echo "  5. va5.py - New version"
echo "  6. va.py - Basic (default)"
echo ""
echo "  q. Quit"
echo ""
echo "========================================"

read -p "Select script to run: " choice

case "$choice" in
    1) "$PYTHON_PATH" "$SCRIPT_DIR/va1.py" ;;
    2) "$PYTHON_PATH" "$SCRIPT_DIR/va2.py" ;;
    3) "$PYTHON_PATH" "$SCRIPT_DIR/va3.py" ;;
    4) "$PYTHON_PATH" "$SCRIPT_DIR/va4.py" ;;
    5) "$PYTHON_PATH" "$SCRIPT_DIR/va5.py" ;;
    6) "$PYTHON_PATH" "$SCRIPT_DIR/va.py" ;;
    q|Q) echo "Goodbye!"; exit 0 ;;
    *) echo "Invalid choice!"; exit 1 ;;
esac
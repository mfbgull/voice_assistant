#!/bin/bash
# Voice Assistant Launcher - GUI Compatible Version

# Get the directory where this script is located
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# Define possible virtual environment Python paths
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
    echo ""
    echo "Press Enter to exit..."
    read
    exit 1
fi

echo "========================================"
echo "      Voice Assistant Launcher"
echo "========================================"
echo ""
echo " 1. va1.py - Basic version"
echo " 2. va2.py - Enhanced (recommended)"
echo " 3. va3.py - Model selector version"
echo " 4. va4.py - Legacy version"
echo ""
echo " q. Quit"
echo ""
echo "========================================"

read -p "Select script to run: " choice

case "$choice" in
    1) 
        echo "Launching va1.py..."
        "$PYTHON_PATH" "$SCRIPT_DIR/va1.py"
        echo ""
        echo "Press Enter to exit..."
        read
        ;;
    2) 
        echo "Launching va2.py (Enhanced version)..."
        "$PYTHON_PATH" "$SCRIPT_DIR/va2.py"
        echo ""
        echo "Press Enter to exit..."
        read
        ;;
    3) 
        echo "Launching va3.py..."
        "$PYTHON_PATH" "$SCRIPT_DIR/va3.py"
        echo ""
        echo "Press Enter to exit..."
        read
        ;;
    4) 
        echo "Launching va4.py..."
        "$PYTHON_PATH" "$SCRIPT_DIR/va4.py"
        echo ""
        echo "Press Enter to exit..."
        read
        ;;
    q|Q) 
        echo "Goodbye!"
        sleep 2
        exit 0
        ;;
    *) 
        echo "Invalid choice!"
        echo "Press Enter to exit..."
        read
        exit 1
        ;;
esac
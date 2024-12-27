#!/bin/bash
echo "==========================================="
echo "To activate the virtual environment, run:"
echo "    . venv/bin/activate"
echo
echo "Once activated, you can run:"
echo "    python tuning_t5.py"
echo "==========================================="
# Keep the container running in an interactive mode
exec "$@"


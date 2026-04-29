#!/bin/bash

# Run all Python scripts in the scripts folder sequentially

SCRIPTS_DIR="scripts"

# Get all Python files in the scripts directory and run them
for script in "$SCRIPTS_DIR"/*.py; do
    if [ -f "$script" ]; then
        echo "Running $script..."
        python3 "$script"
        if [ $? -ne 0 ]; then
            echo "Error: $script failed with exit code $?"
            exit 1
        fi
        echo "$script completed successfully."
        echo "---"
    fi
done

echo "All Python scripts in $SCRIPTS_DIR have finished executing."

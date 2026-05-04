#!/bin/bash

# Run all Python scripts in the scripts folder sequentially

SCRIPTS_DIR="scripts"

# List of scripts to run (excluding comparison script)
SCRIPTS=(
    "$SCRIPTS_DIR/analyze_detections.py"
    "$SCRIPTS_DIR/generate_debris_oes.py"
    "$SCRIPTS_DIR/plot_debris.py"
    "$SCRIPTS_DIR/plot_oe_analysis.py"
    "$SCRIPTS_DIR/plot_velocity_analysis.py"
)

# Run each script sequentially
for script in "${SCRIPTS[@]}"; do
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

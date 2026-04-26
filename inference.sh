#!/bin/bash

echo "Running script..."

# mode = eval
if [ "$1" == "--eval" ]; then
    python scripts/inference.py --eval "$2"

# mode = inference
else
    python scripts/inference.py "$@"
fi

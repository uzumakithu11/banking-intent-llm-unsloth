#!/bin/bash

echo "Running script..."

# mode = eval
if [ "$1" == "--eval" ]; then
    python inference.py --eval "$2"

# mode = inference
else
    python inference.py "$@"
fi

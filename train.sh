#!/bin/bash

echo "Installing dependencies..."
pip install -r requirements.txt

echo "Start training..."
python train.py 

echo "Done!"
#!/bin/bash

# Bash script to set up and run the project

echo "Starting setup..."

# 1. Install Python dependencies
echo "Installing Python dependencies..."
pip install -r requirements.txt

# 2. Configure AWS CLI (if not configured)
echo "Checking AWS CLI configuration..."
if ! aws sts get-caller-identity > /dev/null 2>&1; then
    echo "AWS CLI is not configured. Please configure it now."
    aws configure
else
    echo "AWS CLI is already configured."
fi

# 3. Download model from S3
echo "Downloading the model from S3..."
BUCKET_NAME="your-bucket-name"
MODEL_PATH="path/to/model/"
LOCAL_MODEL_DIR="./model/"

mkdir -p $LOCAL_MODEL_DIR
aws s3 cp s3://$BUCKET_NAME/$MODEL_PATH $LOCAL_MODEL_DIR --recursive

# 4. Run the backend server
echo "Starting the backend server..."
uvicorn backend:app --host 0.0.0.0 --port 8000 --reload

echo "Setup complete!"

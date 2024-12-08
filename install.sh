#!/bin/bash

# install.sh
echo "Installing all requirements..."

# Backend Python packages
echo "Installing Python packages..."
pip install fastapi==0.104.1 \
    uvicorn==0.24.0 \
    transformers==4.35.2 \
    torch==2.1.1 \
    boto3==1.29.3 \
    python-dotenv==1.0.0 \
    pydantic==2.4.2 \
    numpy==1.26.2 \
    tqdm==4.66.1 \
    requests==2.31.0 \
    typing-extensions==4.8.0

# Frontend packages
echo "Installing Node packages..."
npm install next@14.0.3 \
    react@18.2.0 \
    react-dom@18.2.0 \
    @radix-ui/react-slider@1.1.2 \
    @shadcn/ui \
    lucide-react@0.293.0 \
    class-variance-authority@0.7.0 \
    clsx@2.0.0 \
    tailwind-merge@2.0.0 \
    tailwindcss-animate@1.0.7 \
    typescript@5.3.2

echo "Installation complete!"
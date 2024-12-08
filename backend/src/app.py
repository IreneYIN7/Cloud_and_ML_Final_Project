# backend/app.py
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from transformers import AutoModelForCausalLM, AutoTokenizer
import boto3
import torch
import os
from dotenv import load_dotenv

load_dotenv()

app = FastAPI()

class TextInput(BaseModel):
    text: str
    max_length: int = 100

class ModelManager:
    def __init__(self):
        self.model = None
        self.tokenizer = None
        self.s3_client = boto3.client(
            's3',
            aws_access_key_id=os.getenv('AWS_ACCESS_KEY_ID'),
            aws_secret_access_key=os.getenv('AWS_SECRET_ACCESS_KEY')
        )

    def load_model_from_s3(self):
        """Load model and tokenizer from S3 if not already loaded"""
        if self.model is None or self.tokenizer is None:
            try:
                # Create a temporary directory for model files
                os.makedirs("tmp_model", exist_ok=True)
                
                # Download model files from S3
                files_to_download = [
                    "pytorch_model.bin",
                    "config.json",
                    "tokenizer.json",
                    "tokenizer_config.json",
                    "special_tokens_map.json"
                ]
                
                for file in files_to_download:
                    self.s3_client.download_file(
                        os.getenv('S3_BUCKET_NAME'),
                        f"{os.getenv('MODEL_PATH')}/{file}",
                        f"tmp_model/{file}"
                    )
                
                # Load model and tokenizer
                self.model = AutoModelForCausalLM.from_pretrained("tmp_model")
                self.tokenizer = AutoTokenizer.from_pretrained("tmp_model")
                
                # Move model to GPU if available
                if torch.cuda.is_available():
                    self.model = self.model.to('cuda')
                
                return True
            except Exception as e:
                print(f"Error loading model: {str(e)}")
                return False

    def generate_text(self, input_text: str, max_length: int) -> str:
        """Generate text using the loaded model"""
        if not self.load_model_from_s3():
            raise Exception("Failed to load model")
        
        try:
            inputs = self.tokenizer(input_text, return_tensors="pt")
            if torch.cuda.is_available():
                inputs = {k: v.to('cuda') for k, v in inputs.items()}
            
            outputs = self.model.generate(
                inputs["input_ids"],
                max_length=max_length,
                num_return_sequences=1,
                temperature=0.7,
                pad_token_id=self.tokenizer.eos_token_id
            )
            
            return self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        except Exception as e:
            raise Exception(f"Error generating text: {str(e)}")

# Initialize model manager
model_manager = ModelManager()

@app.post("/generate")
async def generate_text(input_data: TextInput):
    try:
        generated_text = model_manager.generate_text(
            input_data.text,
            input_data.max_length
        )
        return {"generated_text": generated_text}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/health")
async def health_check():
    return {"status": "healthy"}
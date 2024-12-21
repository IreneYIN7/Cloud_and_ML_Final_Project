import json
import boto3
import os
from typing import Dict, Any

def create_response(status_code: int, body: Dict[str, Any], cors_origin: str = '*') -> Dict[str, Any]:
    """Create a standardized API response"""
    return {
        'statusCode': status_code,
        'headers': {
            'Access-Control-Allow-Origin': cors_origin,
            'Access-Control-Allow-Headers': 'Content-Type,X-Amz-Date,Authorization,X-Api-Key',
            'Access-Control-Allow-Methods': 'OPTIONS,POST',
            'Content-Type': 'application/json'
        },
        'body': json.dumps(body)
    }

def lambda_handler(event: Dict[str, Any], context: Any) -> Dict[str, Any]:
    """
    Lambda handler for text generation requests
    """
    # Handle OPTIONS request for CORS
    if event.get('httpMethod') == 'OPTIONS':
        return create_response(200, {'message': 'CORS supported'})
        
    try:
        # Parse request body
        body = json.loads(event['body']) if isinstance(event.get('body'), str) else event.get('body', {})
        
        # Get input parameters with defaults
        text = body.get('text', '').strip()
        max_length = min(body.get('max_length', 50), 100)  # Cap max length at 100
        temperature = max(min(body.get('temperature', 0.7), 2.0), 0.1)  # Bound temperature
        
        # Input validation
        if not text:
            return create_response(400, {'error': 'Text input is required'})
            
        # Create SageMaker runtime client
        runtime = boto3.client('runtime.sagemaker')
        
        # Prepare payload
        payload = {
            'text': text,
            'max_length': max_length,
            'temperature': temperature
        }
        
        try:
            # Call SageMaker endpoint
            response = runtime.invoke_endpoint(
                EndpointName='llama-endpoint222',
                ContentType='application/json',
                Body=json.dumps(payload)
            )
            
            # Parse response
            result = json.loads(response['Body'].read().decode())
            
            # Return success response
            return create_response(200, {
                'input': text,
                'generated_text': result[0]['generated_text']
            })
            
        except runtime.exceptions.ModelError as e:
            return create_response(500, {'error': 'Model inference failed', 'details': str(e)})
        except Exception as e:
            return create_response(500, {'error': 'SageMaker endpoint error', 'details': str(e)})
            
    except json.JSONDecodeError:
        return create_response(400, {'error': 'Invalid JSON in request body'})
    except Exception as e:
        return create_response(500, {'error': 'Internal server error', 'details': str(e)}) 
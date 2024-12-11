# Llama3 Chat
This is a simple chat application that uses the Llama3 model to generate responses.
We use AWS SageMaker to host the Llama3 model.

## Folder Structure
- `llama3-app`: The main application.
    - `server`: The server code for the application.
        - `setup.sh`: The setup script for the server.  
        - `app.py`: The main server code.   
        - `requirements.txt`: The dependencies for the server.  
        - `app.py`: The main server code.   

    - `client`: The client code for the application.    
        - `setup.sh`: The setup script for the client.
        - `package.json`: The dependencies for the client.
        - `src`: The source code for the client.
            - `App.jsx`: The main client code.
            - `index.jsx`: The entry point for the client.
            - `App.css`: The styles for the client.
            - `index.css`: The styles for the client.

- `Llama-Training`: The training code for the Llama3 model.
    - `generate.py`: The code for generating the Llama3 model.
    - `model.py`: The code for the Llama3 model.
    - `train.py`: The code for training the Llama3 model.
    - `utils.py`: The code for the utils for the Llama3 model.

## Server
We use Flask to host the API.
### Setup

```
cd server
pip install -r requirements.txt
```

### Run

```
python app.py
```

## Client
We use React to build the client.
### Setup
```
cd client
npm install
npm install react-scripts
```

### Run

```
npm start
```

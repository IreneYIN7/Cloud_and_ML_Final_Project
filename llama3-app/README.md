# Llama3 Chat
This is a simple chat application that uses the Llama3 model to generate responses.
We use AWS SageMaker to host the Llama3 model.


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

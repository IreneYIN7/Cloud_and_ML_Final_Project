# Server setup
python -m venv venv
source venv/bin/activate  # or `venv\Scripts\activate` on Windows
pip install flask flask-cors python-dotenv boto3

# Client setup
npm create vite@latest client -- --template react
cd client
npm install

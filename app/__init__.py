from flask import Flask, jsonify
from flask_jwt_extended import JWTManager
from flask_cors import CORS
import os
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# Initialize the app
app = Flask(__name__)
CORS(app)
JWTManager(app)

@app.route('/')
def home():
    return jsonify({"Welcome":"This is  NutriVision API"}), 200

# Store MongoDB URI in app config
app.config['MONGO_URI'] = os.environ.get('MONGO_URI')

# Print out the MONGO_URI to ensure it's being read correctly
# print("MONGO_URI from app config:", app.config['MONGO_URI'])

# Import routes after the app is initialized to avoid circular imports
from app.routes import  model_routes , fetch_routes

app.register_blueprint(model_routes.bp)
app.register_blueprint(fetch_routes.bp)
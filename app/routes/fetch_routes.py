from flask import Blueprint, jsonify, request
from pymongo.errors import PyMongoError
from dotenv import load_dotenv
import os
from app.utils.connect_mongo import get_db
from app.utils.fetch_users_uid import extract_token, get_uid_and_email
from app.utils.logger import setup_logger

load_dotenv()

logger = setup_logger()

bp = Blueprint('fetch', __name__)

@bp.route('/fetch/saved_fruits_vegies', methods=["GET"])
def fetch_saved_fruits_vegies():
    logger.info("Received request for saved fruits and vegetables.")
    
    db = get_db()
    authorization_header = request.headers.get('Authorization')
    id_token = extract_token(authorization_header)
    logger.debug(f"Received Token: {id_token}")

    if not id_token:
        logger.warning("Authorization token missing")
        return jsonify({"error": "Authorization token is missing"}), 401
    
    try:
        uid, email = get_uid_and_email(id_token)
        logger.info(f"User UID: {uid}, Email: {email}")
    except ValueError as e:
        logger.error(f"Error getting UID and email: {str(e)}")
        return jsonify({"error": str(e)}), 401
    
    try:
        # Access user-specific collection
        collection_name = f"{email}_{uid}"
        user_collection = db[collection_name]
        logger.info(f"Accessing collection: {collection_name}")

        # Fetch only products of type 'image_classification' (fruits and vegetables)
        fruits_vegies = list(user_collection.find({"type": "image_classification"}))
        logger.info(f"Fetched {len(fruits_vegies)} fruits and vegetables from collection.")

        for item in fruits_vegies:
            item['_id'] = str(item['_id'])  # Convert ObjectId to string for JSON serialization

        logger.debug(f"Processed fruits and vegetables: {fruits_vegies}")
        return jsonify(fruits_vegies), 200
        
    except PyMongoError as e:
        logger.error(f"MongoDB error: {str(e)}")
        return jsonify({"error": f"MongoDB error: {str(e)}"}), 500
    except Exception as e:
        logger.error(f"Failed to fetch fruits and vegetables: {str(e)}")
        return jsonify({"error": f"Failed to fetch fruits and vegetables: {str(e)}"}), 500
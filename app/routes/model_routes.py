import os
import gc
from flask import Blueprint, jsonify, request
from PIL import Image
import torch
import torchvision.transforms as transforms
from pymongo import MongoClient
from pymongo.errors import PyMongoError
from dotenv import load_dotenv
from transformers import AutoModelForImageClassification
import psutil
from app.utils.fetch_users_uid import extract_token, get_uid_and_email
from app.utils.logger import setup_logger 

# Set up logger
logger = setup_logger()

load_dotenv()

# MongoDB setup
MONGO_URI = os.getenv("MONGO_URI")
client = MongoClient(MONGO_URI)
db = client['nutritional_db']

# Load the PyTorch model and labels
model_name = "jazzmacedo/fruits-and-vegetables-detector-36"
model = AutoModelForImageClassification.from_pretrained(model_name)
model.eval()  # Set the model to evaluation mode
labels = list(model.config.id2label.values())

# Check if GPU is available and set the device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)  # Move model to the correct device

def log_memory_usage():
    process = psutil.Process(os.getpid())
    mem_info = process.memory_info()
    logger.info(f"Memory usage - RSS: {mem_info.rss / 1024 ** 2:.2f} MB, VMS: {mem_info.vms / 1024 ** 2:.2f} MB")

# Define the image preprocessing transformation
preprocess = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

bp = Blueprint('model', __name__)

@bp.route('/model/classify_image', methods=['POST'])
def classifyImage():
    # Log start of request
    logger.info("Received a request to /model/classify_image")

    # Authentication
    authorization_header = request.headers.get('Authorization')
    id_token = extract_token(authorization_header)
    if not id_token:
        logger.warning("Authorization token is missing")
        return jsonify({"error": "Authorization token is missing"}), 401

    try:
        uid, email = get_uid_and_email(id_token)
    except ValueError as e:
        logger.warning(f"Invalid token: {str(e)}")
        return jsonify({"error": str(e)}), 401

    if 'image' not in request.files:
        logger.warning("No image file provided")
        return jsonify({'error': 'No image file provided'}), 400
    
    image_file = request.files['image']
    logger.info(f"Received image file: {image_file.filename}")

    try:
        # Open and preprocess the image
        img = Image.open(image_file).convert("RGB")
        input_tensor = preprocess(img).unsqueeze(0)  # Add batch dimension
        logger.info("Image processed and input tensor created")

        # Log memory usage before prediction
        log_memory_usage()

        # Perform prediction with mixed precision
        with torch.no_grad():
            with torch.autocast(device_type=device.type, dtype=torch.float16):
                outputs = model(input_tensor.to(device))
        
        # Log memory usage after prediction
        log_memory_usage()

        # Get the predicted class index and name
        predicted_idx = torch.argmax(outputs.logits, dim=1).item()
        predicted_class_name = labels[predicted_idx]
        confidence_scores = torch.softmax(outputs.logits, dim=1).cpu().numpy()[0]
        confidence_score = float(confidence_scores[predicted_idx])
        logger.info(f"Predicted class: {predicted_class_name}, Confidence: {confidence_score}")

        # MongoDB storage
        collection_name = f"{email}_{uid}"
        user_collection = db[collection_name]

        result = user_collection.update_one(
            {'image_filename': image_file.filename},
            {
                '$set': {
                    'type': 'image_classification',
                    'predicted_class': predicted_class_name,
                    'confidence': confidence_score,
                    'image_filename': image_file.filename
                }
            },
            upsert=True  
        )

        if result.upserted_id:
            product_id = result.upserted_id
        else:
            document = user_collection.find_one({'image_filename': image_file.filename})
            product_id = document['_id']

        logger.info(f"Classification result saved for {image_file.filename} with product ID {product_id}")

        return jsonify({
            'predicted_class': predicted_class_name,
            'confidence': confidence_score,
            'product_id': str(product_id)
        })
    except PyMongoError as e:
        logger.error(f"MongoDB error: {str(e)}")
        return jsonify({"error": "Failed to save classification result"}), 500
    except Exception as e:
        logger.error(f"Unexpected error: {str(e)}")
        return jsonify({"error": "An unexpected error occurred"}), 500
    finally:
        # Cleanup
        del input_tensor, outputs
        torch.cuda.empty_cache()
        gc.collect()
        # Log memory usage after cleanup
        log_memory_usage()
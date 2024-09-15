import os
import json
import firebase_admin
from firebase_admin import credentials, auth
from firebase_admin._token_gen import ExpiredIdTokenError

# Load the Firebase credentials JSON from the environment variable
firebase_credentials = os.getenv('FIREBASE_CREDENTIALS')

if not firebase_credentials:
    raise Exception("FIREBASE_CREDENTIALS environment variable not found.")

# Parse the JSON string into a Python dictionary
cred_dict = json.loads(firebase_credentials)

# Initialize the Firebase Admin SDK with the parsed credentials
cred = credentials.Certificate(cred_dict)
firebase_admin.initialize_app(cred)


def extract_token(authorization_header):
    if authorization_header and authorization_header.startswith("Bearer "):
        return authorization_header.split("Bearer ")[1]
    return None

def get_uid_and_email(id_token):
    try:
        decoded_token = auth.verify_id_token(id_token)
        uid = decoded_token['uid']
        email = decoded_token.get('email')
        if not email:
            raise ValueError("Email not found in token")
        return uid, email
    except ExpiredIdTokenError:
        raise ValueError("Token expired, please refresh your token and try again.")
    except Exception as e:
        raise ValueError(f"Invalid token or user information: {str(e)}")
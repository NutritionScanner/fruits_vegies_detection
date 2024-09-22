# Connect to mongodb using enviroment variables
import os
from pymongo import MongoClient


def get_db():
    MONGO_URI = os.getenv("MONGO_URI")
    client = MongoClient(MONGO_URI)
    return client['nutritional_db']
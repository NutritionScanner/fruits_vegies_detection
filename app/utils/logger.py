import logging
import os

def setup_logger():
    # Create a logger with the name 'nutrivision'
    logger = logging.getLogger('nutrivision')
    
    # Prevent adding multiple handlers if the logger is already configured
    if not logger.hasHandlers():
        # Set the logging level
        logger.setLevel(logging.INFO)
        
        # Create a console handler and set its level
        console_handler = logging.StreamHandler()
        console_handler.setLevel(logging.INFO)
        
        # Create a file handler to log messages to a file
        log_file_path = 'logs/nutrivision.log'
        os.makedirs(os.path.dirname(log_file_path), exist_ok=True)
        file_handler = logging.FileHandler(log_file_path)
        file_handler.setLevel(logging.INFO)
        
        # Create a formatter and set it for both console and file handlers
        formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        console_handler.setFormatter(formatter)
        file_handler.setFormatter(formatter)
        
        # Add handlers to the logger
        logger.addHandler(console_handler)
        logger.addHandler(file_handler)

    return logger

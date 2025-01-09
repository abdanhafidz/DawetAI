import os
import sys
import django
import redis
import torch
import json
import logging
from datetime import datetime
from transformers import TextIteratorStreamer
from app.services import PredictService
from threading import Thread
# Add the project root directory to Python path
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../../"))
sys.path.append(PROJECT_ROOT)

# Set Django settings module
os.environ.setdefault('DJANGO_SETTINGS_MODULE', 'dawetAI.settings')

# Initialize Redis client
redis_client = redis.StrictRedis(host='localhost', port=6379, db=0)

# Configure logging
def setup_logger():
    """Configure logging to both file and console"""
    # Create logs directory if it doesn't exist
    log_dir = os.path.join(os.path.dirname(__file__), 'logs')
    os.makedirs(log_dir, exist_ok=True)
    log_file = os.path.join(log_dir, 'worker.log')
    
    logger = logging.getLogger('WorkerLogger')
    logger.setLevel(logging.INFO)
    
    # Create formatters
    file_formatter = logging.Formatter(
        '%(asctime)s [%(levelname)s] %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    console_formatter = logging.Formatter(
        '%(asctime)s [%(levelname)s] %(message)s',
        datefmt='%H:%M:%S'
    )
    
    # File handler
    file_handler = logging.FileHandler(log_file)
    file_handler.setLevel(logging.INFO)
    file_handler.setFormatter(file_formatter)
    
    # Console handler
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(logging.INFO)
    console_handler.setFormatter(console_formatter)
    
    # Add handlers to logger
    logger.addHandler(file_handler)
    logger.addHandler(console_handler)
    
    return logger

logger = setup_logger()


def process_prediction_stream(user_message, session_id, model, tokenizer):
    """
    Process a single prediction request
    """
    logger.info(f"Processing new prediction task - Session ID: {session_id}")
    logger.info(f"Input message: {user_message}")
    
    messages = [{"role": "system", "content": "Kamu adalah Asisten AI bernama Dawet yang dikembangkan oleh Mahasiswa di ITS (Institut Teknologi Sepuluh Nopember) Surabaya, program studi RKA (Rekayasa Kecerdasan Artifisial). ITS ada di Surabaya."},
                {"role": "user", "content": user_message}]
    logger.debug(f"Formatted messages: {messages}")
    
    try:
        inputs = tokenizer.apply_chat_template(
            messages,
            tokenize=True,
            add_generation_prompt=True,
            return_tensors="pt"
        ).to("cuda")
        
        logger.info("Starting prediction generation...")

        token_count = 0
        streamer = TextIteratorStreamer(tokenizer, skip_prompt=True)
        inputs = tokenizer.apply_chat_template(
            messages,
            tokenize = True,
            add_generation_prompt = True, # Must add for generation
            return_tensors = "pt",
        ).to("cuda")
        
        generation_kwargs = dict(inputs=inputs, streamer=streamer, max_new_tokens=128,
                                 min_new_tokens=64, repetition_penalty=1.1)
        thread = Thread(target=model.generate, kwargs=generation_kwargs)
        thread.start()

        for output_text in streamer:
            # output_text = tokenizer.decode(token, skip_special_tokens=True)
            logger.info(f"Generated token {output_text}")
            redis_client.rpush(session_id, output_text)
            token_count += 1
            if token_count % 10 == 0:  # Log every 10 tokens
                logger.info(f"Generated {token_count} tokens")
        
        logger.info(f"Prediction completed - Total tokens generated: {token_count}")
        
    except Exception as e:
        logger.error(f"Error during prediction generation: {str(e)}", exc_info=True)
        raise
    
    finally:
        redis_client.rpush(session_id, "[END]")
        logger.info("Stream ended - Session complete")

def run_worker():
    """
    Persistent worker that processes tasks from Redis queue
    """
    logger.info("Initializing worker process")
    
    try:
        # Setup Django environment
        django.setup()
        logger.info("Django setup completed")

        
        
        # Initialize model and tokenizer
        logger.info("Loading model and tokenizer...")
        predict_service = PredictService()
        model = predict_service.getModel()
        tokenizer = predict_service.getTokenizer()
        
        # Move model to CUDA
        model.to("cuda")
        logger.info("Model loaded and moved to CUDA successfully")
        
        logger.info("Worker initialization complete - Ready to process tasks")
        
        task_counter = 0
        while True:
            try:
                logger.debug("Waiting for new task...")
                _, task = redis_client.blpop("prediction_tasks")
                task_data = json.loads(task)
                
                task_counter += 1
                logger.info(f"Starting task #{task_counter} - Session: {task_data['session_id']}")
                
                process_prediction_stream(
                    task_data['message'],
                    task_data['session_id'],
                    model,
                    tokenizer
                )
                
                logger.info(f"Task #{task_counter} completed successfully")
                
            except json.JSONDecodeError as e:
                logger.error(f"Invalid task data received: {str(e)}")
                continue
                
            except Exception as e:
                logger.error(f"Error processing task #{task_counter}: {str(e)}", exc_info=True)
                redis_client.rpush(
                    task_data['session_id'],
                    f"[ERROR] {str(e)}"
                )
                redis_client.rpush(task_data['session_id'], "[END]")
                
    except Exception as e:
        logger.critical(f"Critical worker error: {str(e)}", exc_info=True)
        raise

if __name__ == "__main__":
    try:
        run_worker()
    except KeyboardInterrupt:
        logger.info("Worker shutdown requested")
    except Exception as e:
        logger.critical(f"Fatal error: {str(e)}", exc_info=True)
    finally:
        logger.info("Worker process terminated")
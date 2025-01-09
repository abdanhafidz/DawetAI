import subprocess
import redis
import json
import signal
import time
import atexit
import threading
import os
python_interpreter = "/home/abdanhafidz/.pyenv/versions/dawetenv/bin/python"
worker_path = os.path.join("/home", "abdanhafidz", "projects", "DawetAI", "src", "dawetAI", "app", "worker", "worker.py")
class WorkerPool:
    _instance = None
    _lock = threading.Lock()
    
    def __init__(self, num_workers=1):
        self.num_workers = num_workers
        self.workers = []
        self.redis_client = redis.StrictRedis(host='localhost', port=6379, db=0)
        
        # Start workers
        self.start_workers()
        
        # Register cleanup on exit - only in main thread
        if threading.current_thread() is threading.main_thread():
            atexit.register(self.cleanup)
            # Only register signals in main thread
            self._setup_signal_handlers()

    def _setup_signal_handlers(self):
        """Setup signal handlers - should only be called from main thread"""
        signal.signal(signal.SIGINT, self.handle_interrupt)
        signal.signal(signal.SIGTERM, self.handle_interrupt)

    @classmethod
    def get_instance(cls, num_workers=1):
        """Thread-safe singleton pattern"""
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    cls._instance = cls(num_workers)
        return cls._instance

    def start_workers(self):
        """Start the specified number of worker processes"""
        print(f"Starting {self.num_workers} workers...")
            # Path ke root direktori proyek
        project_root = "/home/abdanhafidz/projects/DawetAI/src/dawetAI"
        env = os.environ.copy()
        env["PYTHONPATH"] = project_root
        for i in range(self.num_workers):
            try:
                worker = subprocess.Popen(
                            f"source /home/abdanhafidz/projects/DawetAI/pyenv/bin/activate && python app/worker/worker.py",
                            shell=True,
                            executable="/bin/bash",
                            stdout=subprocess.PIPE,
                            stderr=subprocess.PIPE,
                            universal_newlines=True,
                            env=env
                        )
                # Add immediate output check
                # stdout, stderr = worker.communicate(timeout=60)
                # if stderr:
                #     print(f"Worker {i+1} error: {stderr}")
                # if stdout:
                #     print(f"Worker {i+1} output: {stdout}")
                    
                self.workers.append(worker)
                print(f"Started worker {i+1} with PID: {worker.pid}")
            except Exception as e:
                print(f"Failed to start worker {i+1}: {str(e)}")

    def submit_task(self, user_message, session_id):
        """Submit a new prediction task to the queue"""
        task = {
            'message': user_message,
            'session_id': session_id
        }
        
        # Push task to Redis queue
        self.redis_client.rpush("prediction_tasks", json.dumps(task))
    
    def get_results(self, session_id, timeout=30):
        """Generator that yields results as they become available"""
        while True:
            # Wait for new result with timeout
            result = self.redis_client.blpop(session_id, timeout=timeout)
            
            if result is None:
                print("Timeout waiting for results")
                break
            
            _, token = result
            token = token.decode('utf-8')
            
            if token == "[END]":
                break
            
            yield token

    def check_worker_health(self):
        """Check if all workers are still running"""
        with self._lock:
            for i, worker in enumerate(self.workers):
                if worker.poll() is not None:
                    print(f"Worker {i+1} has died, restarting...")
                    # Restart the worker
                    new_worker = subprocess.Popen(
                        ["python", "worker.py"],
                        stdout=subprocess.PIPE,
                        stderr=subprocess.PIPE
                    )
                    self.workers[i] = new_worker

    def cleanup(self):
        """Cleanup worker processes"""
        print("Cleaning up workers...")
        with self._lock:
            for worker in self.workers:
                worker.terminate()
                worker.wait()
        print("All workers terminated")

    def handle_interrupt(self, signum, frame):
        """Handle interrupt signals"""
        print("Received interrupt signal, shutting down...")
        self.cleanup()
        exit(0)
import json
from channels.generic.websocket import AsyncWebsocketConsumer
import redis
import asyncio
import uuid
from time import sleep
from app.worker import WorkerPool

# Create a single WorkerPool instance to be shared across all connections
worker_pool = WorkerPool(num_workers=1)

class PredictConsumer(AsyncWebsocketConsumer):
    async def connect(self):
        self.redis_client = redis.StrictRedis(host='localhost', port=6379, db=0)
        await self.accept()

    async def disconnect(self, close_code):
        # Cleanup if needed
        pass

    async def receive(self, text_data):
        try:
            data = json.loads(text_data)
            user_message = data['message']
            session_id = str(uuid.uuid4())

            print(f"Received prompt for session {session_id}: {user_message}")

            # Submit task to worker pool
            worker_pool.submit_task(user_message, session_id)

            # Stream results back to client
            try:
                while True:
                    token = self.redis_client.blpop(session_id, timeout=5)
                    if not token:
                        print("No new tokens, retrying...")
                        await asyncio.sleep(0.1)
                        continue

                    token = token[1].decode('utf-8')

                    if token == "[END]":
                        # Send completion message
                        await self.send(text_data=json.dumps({
                            'status': 'complete'
                        }))
                        break
                    elif token.startswith("[ERROR]"):
                        # Handle error case
                        await self.send(text_data=json.dumps({
                            'status': 'error',
                            'error': token[7:]  # Remove [ERROR] prefix
                        }))
                        break
                    else:
                        # Send token
                        await self.send(text_data=json.dumps({
                            'status': 'streaming',
                            'response': token
                        }))
                    await asyncio.sleep(0.000001)

            except Exception as e:
                print(f"Streaming error: {str(e)}")
                await self.send(text_data=json.dumps({
                    'status': 'error',
                    'error': f"Streaming error: {str(e)}"
                }))

        except Exception as e:
            print(f"Processing error: {str(e)}")
            await self.send(text_data=json.dumps({
                'status': 'error',
                'error': f"Processing error: {str(e)}"
            }))
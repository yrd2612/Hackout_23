import asyncio
import websockets

async def send_video():
    video_filename = r"D:\code\received_video.mp4"  # Replace with the path to your video file
    chunk_size = 10000
    
    async with websockets.connect('ws://192.168.25.171:8765') as websocket:  # Replace 'server_ip' with the server's IP address
        with open(video_filename, 'rb') as video_file:
            while True:
                chunk = video_file.read(chunk_size)
                if not chunk:
                    break
                await websocket.send(chunk)

asyncio.get_event_loop().run_until_complete(send_video())
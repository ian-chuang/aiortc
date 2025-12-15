import argparse
import asyncio
import json
import logging
import cv2
import time
from av import VideoFrame
from aiohttp import web
from aiortc import RTCPeerConnection, RTCSessionDescription, VideoStreamTrack, RTCIceCandidate
from aiortc.sdp import candidate_from_sdp, candidate_to_sdp

class SignalingBye:
    pass

BYE = SignalingBye()

def object_from_string(message_str):
    message = json.loads(message_str)
    if message["type"] in ["answer", "offer"]:
        return RTCSessionDescription(**message)
    elif message["type"] == "candidate" and message["candidate"]:
        candidate = candidate_from_sdp(message["candidate"].split(":", 1)[1])
        candidate.sdpMid = message["id"]
        candidate.sdpMLineIndex = message["label"]
        return candidate
    elif message["type"] == "bye":
        return BYE
    return None

def object_to_string(obj):
    if isinstance(obj, RTCSessionDescription):
        message = {"sdp": obj.sdp, "type": obj.type}
    elif isinstance(obj, RTCIceCandidate):
        message = {
            "candidate": "candidate:" + candidate_to_sdp(obj),
            "id": obj.sdpMid,
            "label": obj.sdpMLineIndex,
            "type": "candidate",
        }
    elif obj is BYE:
        message = {"type": "bye"}
    else:
        return ""
    return json.dumps(message, sort_keys=True)

class WebSocketServerSignaling:
    def __init__(self, host, port):
        self._host = host
        self._port = port
        self._runner = None
        self._ws = None
        self._queue = asyncio.Queue()

    async def connect(self):
        app = web.Application()
        app.router.add_get("/", self._handler)
        self._runner = web.AppRunner(app)
        await self._runner.setup()
        site = web.TCPSite(self._runner, self._host, self._port)
        await site.start()
        print(f"Signaling server started at http://{self._host}:{self._port}")

    async def _handler(self, request):
        ws = web.WebSocketResponse()
        await ws.prepare(request)
        self._ws = ws
        
        async for msg in ws:
            if msg.type == web.WSMsgType.TEXT:
                await self._queue.put(object_from_string(msg.data))
        return ws

    async def send(self, obj):
        while self._ws is None:
            await asyncio.sleep(0.1)
        await self._ws.send_str(object_to_string(obj))

    async def receive(self):
        return await self._queue.get()

    async def close(self):
        if self._runner:
            await self._runner.cleanup()

class MetadataVideoStreamTrack(VideoStreamTrack):
    def __init__(self):
        super().__init__()
        self.cap = cv2.VideoCapture(0)
        self.metadata_len = 10 # Example length

    async def recv(self):
        pts, time_base = await self.next_timestamp()
        
        ret, frame = self.cap.read()
        if not ret:
            # Loop video or handle end
            self.cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
            ret, frame = self.cap.read()
        
        # Create VideoFrame
        frame = VideoFrame.from_ndarray(frame, format="bgr24")
        frame.pts = pts
        frame.time_base = time_base
        
        # Create metadata
        # Simple counter as metadata
        metadata = f"{pts % 10000000000:010d}".encode('utf-8') # 10 bytes
        
        return frame, metadata

    def stop(self):
        super().stop()
        self.cap.release()

async def run(pc, signaling):
    await signaling.connect()

    # Add track
    track = MetadataVideoStreamTrack()
    pc.addTrack(track)

    # Create offer
    await pc.setLocalDescription(await pc.createOffer())
    await signaling.send(pc.localDescription)

    # Receive answer
    while True:
        obj = await signaling.receive()
        if isinstance(obj, RTCSessionDescription):
            await pc.setRemoteDescription(obj)
        elif obj is BYE:
            break

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Video sender with metadata")
    parser.add_argument("--verbose", "-v", action="count")
    args = parser.parse_args()

    if args.verbose:
        logging.basicConfig(level=logging.DEBUG)
    else:
        logging.basicConfig(level=logging.INFO)

    signaling = WebSocketServerSignaling("127.0.0.1", 9999)
    pc = RTCPeerConnection()
    
    loop = asyncio.get_event_loop()
    try:
        loop.run_until_complete(run(pc, signaling))
    except KeyboardInterrupt:
        pass
    finally:
        loop.run_until_complete(pc.close())
        loop.run_until_complete(signaling.close())

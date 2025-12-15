import argparse
import asyncio
import json
import logging
import cv2
import numpy as np
import aiohttp
from aiortc import RTCPeerConnection, RTCSessionDescription, MediaStreamTrack, RTCIceCandidate
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

class WebSocketClientSignaling:
    def __init__(self, host, port):
        self._url = f"http://{host}:{port}"
        self._session = None
        self._ws = None
        self._queue = asyncio.Queue()

    async def connect(self):
        self._session = aiohttp.ClientSession()
        while True:
            try:
                self._ws = await self._session.ws_connect(self._url)
                break
            except Exception:
                print("Waiting for signaling server...")
                await asyncio.sleep(1)
        
        asyncio.create_task(self._read_loop())

    async def _read_loop(self):
        async for msg in self._ws:
            if msg.type == aiohttp.WSMsgType.TEXT:
                await self._queue.put(object_from_string(msg.data))

    async def send(self, obj):
        await self._ws.send_str(object_to_string(obj))

    async def receive(self):
        return await self._queue.get()

    async def close(self):
        if self._session:
            await self._session.close()

async def consume_video(track):
    while True:
        try:
            result = await track.recv()
            if isinstance(result, tuple):
                frame, metadata = result
                print(f"Received frame with metadata: {metadata}")
            else:
                frame = result
                print("Received frame without metadata")
            
            img = frame.to_ndarray(format="bgr24")
            cv2.imshow("Video", img)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
        except Exception as e:
            print(f"Error: {e}")
            break
    cv2.destroyAllWindows()

async def run(pc, signaling):
    await signaling.connect()

    @pc.on("track")
    def on_track(track):
        print("Track received", track.kind)
        if track.kind == "video":
            track.metadata_len = 10 # Must match sender
            asyncio.ensure_future(consume_video(track))

    # Receive offer
    while True:
        obj = await signaling.receive()
        if isinstance(obj, RTCSessionDescription):
            await pc.setRemoteDescription(obj)
            if obj.type == "offer":
                await pc.setLocalDescription(await pc.createAnswer())
                await signaling.send(pc.localDescription)
        elif obj is BYE:
            break

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Video receiver with metadata")
    parser.add_argument("--verbose", "-v", action="count")
    args = parser.parse_args()

    if args.verbose:
        logging.basicConfig(level=logging.DEBUG)
    else:
        logging.basicConfig(level=logging.INFO)

    signaling = WebSocketClientSignaling("127.0.0.1", 9999)
    pc = RTCPeerConnection()
    
    loop = asyncio.get_event_loop()
    try:
        loop.run_until_complete(run(pc, signaling))
    except KeyboardInterrupt:
        pass
    finally:
        loop.run_until_complete(pc.close())
        loop.run_until_complete(signaling.close())

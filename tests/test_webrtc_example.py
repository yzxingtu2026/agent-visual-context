"""真实 aiortc 协商/视频回传/结果通道冒烟测试；不依赖物理摄像头。"""

from __future__ import annotations

import asyncio
import json
from typing import Any

import pytest

pytest.importorskip("aiortc")
pytest.importorskip("aiohttp")

import numpy as np  # noqa: E402
from aiohttp.test_utils import TestClient, TestServer  # noqa: E402
from aiortc import RTCPeerConnection, RTCSessionDescription, VideoStreamTrack  # noqa: E402
from av import VideoFrame  # noqa: E402
from examples.webrtc_browser.server import create_app  # noqa: E402


class SyntheticVideo(VideoStreamTrack):
    async def recv(self) -> VideoFrame:
        pts, time_base = await self.next_timestamp()
        frame = VideoFrame.from_ndarray(np.zeros((240, 320, 3), dtype=np.uint8), format="bgr24")
        frame.pts = pts
        frame.time_base = time_base
        return frame


def test_webrtc_video_and_datachannel() -> None:
    async def scenario() -> None:
        client = TestClient(TestServer(create_app(fps=2)))
        await client.start_server()
        peer = RTCPeerConnection()
        got_video = asyncio.Event()
        got_result = asyncio.Event()
        payloads: list[dict[str, Any]] = []
        channel = peer.createDataChannel("results")

        @channel.on("message")
        def on_message(message: str) -> None:
            data = json.loads(message)
            if data.get("type") == "inference":
                payloads.append(data)
                got_result.set()

        @peer.on("track")
        def on_track(track: VideoStreamTrack) -> None:
            async def consume() -> None:
                received = await track.recv()
                assert isinstance(received, VideoFrame)
                assert received.width == 320 and received.height == 240
                got_video.set()

            asyncio.create_task(consume())

        peer.addTrack(SyntheticVideo())
        try:
            await peer.setLocalDescription(await peer.createOffer())
            response = await client.post(
                "/offer", json={"sdp": peer.localDescription.sdp, "type": "offer"}
            )
            assert response.status == 200
            answer = await response.json()
            await peer.setRemoteDescription(
                RTCSessionDescription(sdp=answer["sdp"], type=answer["type"])
            )
            async with asyncio.timeout(12):
                await asyncio.gather(got_video.wait(), got_result.wait())
            assert payloads[0]["frame_id"].startswith("frame-")
            assert payloads[0]["backend"] == "mock / mock"
            assert payloads[0]["server_ms"] >= 0
        finally:
            await peer.close()
            await client.close()

    asyncio.run(scenario())


def test_offer_validation_and_static_page() -> None:
    async def scenario() -> None:
        client = TestClient(TestServer(create_app()))
        await client.start_server()
        try:
            page = await client.get("/")
            assert page.status == 200
            assert "WebRTC" in await page.text()
            invalid = await client.post("/offer", json={"type": "offer"})
            assert invalid.status == 400
        finally:
            await client.close()

    asyncio.run(scenario())

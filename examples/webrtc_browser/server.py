"""本机 WebRTC 视频回传 + 独立推理示例；运行: uv run --extra webrtc python examples/webrtc_browser/server.py"""

from __future__ import annotations

import argparse
import asyncio
import json
import logging
import time
import uuid
from collections.abc import Awaitable, Callable
from pathlib import Path
from typing import Any

from aiohttp import web
from aiortc import (
    MediaStreamTrack,
    RTCDataChannel,
    RTCPeerConnection,
    RTCSessionDescription,
    VideoStreamTrack,
)
from av import VideoFrame

from agent_visual_context.config import load_config
from agent_visual_context.domain import Frame, utc_now
from agent_visual_context.logging_setup import configure_logging
from agent_visual_context.runtime.factory import build_mock_pipeline
from agent_visual_context.runtime.remote import PushFrameSource, RemoteInference

log = logging.getLogger("avc.examples.webrtc")
STATIC = Path(__file__).parent


class ReturnVideoTrack(VideoStreamTrack):
    """直通回传收到的视频帧；采样推理绝不在这里等待模型。"""

    def __init__(self, incoming: MediaStreamTrack, inference: RemoteInference) -> None:
        super().__init__()
        self.incoming = incoming
        self.inference = inference
        self.sequence = 0

    async def recv(self) -> VideoFrame:
        video_frame = await self.incoming.recv()
        if not isinstance(video_frame, VideoFrame):
            raise TypeError("收到非视频帧")
        self.sequence += 1
        if self.inference.observe():
            now = time.monotonic()
            try:
                image = video_frame.to_ndarray(format="bgr24")
                self.inference.offer(
                    Frame(
                        frame_id=f"frame-{self.sequence}",
                        source_id=self.inference.source.source_id,
                        captured_at=utc_now(),  # 接收时间，非浏览器采集时间
                        width=video_frame.width,
                        height=video_frame.height,
                        data=image,
                        metadata={"received_mono": now},
                    )
                )
            except Exception:
                log.exception("视频帧采样失败，继续回传视频")
        return video_frame


def create_app(*, fps: float = 2.0, detector: str = "mock", reasoner: str = "mock") -> web.Application:
    if fps <= 0 or fps > 30:
        raise ValueError("推理采样 FPS 须在 (0, 30] 范围内")
    app = web.Application(client_max_size=64 * 1024)
    peers: set[RTCPeerConnection] = set()
    closers: set[Callable[[], Awaitable[None]]] = set()

    async def index(_request: web.Request) -> web.FileResponse:
        return web.FileResponse(STATIC / "index.html")

    async def script(_request: web.Request) -> web.FileResponse:
        return web.FileResponse(STATIC / "client.js")

    async def offer(request: web.Request) -> web.Response:
        try:
            params = await request.json()
            if params.get("type") != "offer" or not isinstance(params.get("sdp"), str):
                raise ValueError("缺少有效 SDP offer")
        except (ValueError, json.JSONDecodeError) as exc:
            raise web.HTTPBadRequest(text=str(exc)) from exc

        if peers:
            raise web.HTTPConflict(text="单机示例一次只支持一个浏览器会话")
        session_id = uuid.uuid4().hex[:12]
        source = PushFrameSource(f"webrtc-{session_id}")
        config = load_config(
            scene_id=f"scene-{session_id}", source_id=source.source_id, target_fps=fps,
            detector_backend=detector, reasoner_backend=reasoner,
            persistence_min_hits=1, persistence_min_seconds=0.0,
        )
        pipeline = build_mock_pipeline(config, source=source)
        peer = RTCPeerConnection()
        peers.add(peer)
        channel: RTCDataChannel | None = None
        backend = f"{detector} / {reasoner}"

        def send_result(payload: dict[str, Any]) -> None:
            active_channel = channel
            if active_channel is not None and active_channel.readyState == "open":
                payload["backend"] = backend
                active_channel.send(json.dumps(payload, ensure_ascii=False))

        inference = RemoteInference(pipeline, source, target_fps=fps, on_result=send_result)
        inference.start()
        closing = False

        async def close() -> None:
            nonlocal closing
            if closing:
                return
            closing = True
            await inference.stop()
            await peer.close()
            peers.discard(peer)
            closers.discard(close)
            log.info("会话 %s 关闭", session_id)

        closers.add(close)

        @peer.on("datachannel")
        def on_datachannel(incoming_channel):  # type: ignore[no-untyped-def]
            nonlocal channel
            channel = incoming_channel

            @channel.on("message")
            def on_message(message):  # type: ignore[no-untyped-def]
                if message == "ping" and channel.readyState == "open":
                    channel.send("pong")

        @peer.on("track")
        def on_track(track):  # type: ignore[no-untyped-def]
            if track.kind == "video":
                peer.addTrack(ReturnVideoTrack(track, inference))
                track.on("ended", lambda: asyncio.create_task(close()))

        @peer.on("connectionstatechange")
        async def on_state_change() -> None:
            if peer.connectionState in {"failed", "closed"}:
                await close()

        try:
            await peer.setRemoteDescription(RTCSessionDescription(sdp=params["sdp"], type="offer"))
            if not any(sender.track is not None for sender in peer.getSenders()):
                raise ValueError("offer 必须含视频轨")
            answer = await peer.createAnswer()
            await peer.setLocalDescription(answer)
            log.info("会话 %s 已连接：detector=%s reasoner=%s fps=%.1f", session_id, detector, reasoner, fps)
            return web.json_response({"sdp": peer.localDescription.sdp, "type": peer.localDescription.type})
        except Exception as exc:
            await close()
            log.exception("WebRTC 信令失败")
            raise web.HTTPBadRequest(text=f"无法建立 WebRTC 会话: {exc}") from exc

    async def shutdown(_app: web.Application) -> None:
        await asyncio.gather(*(close() for close in list(closers)))

    app.router.add_get("/", index)
    app.router.add_get("/client.js", script)
    app.router.add_post("/offer", offer)
    app.on_shutdown.append(shutdown)
    return app


def main() -> None:
    parser = argparse.ArgumentParser(description="本机浏览器 WebRTC 连续视频 + 视觉推理示例")
    parser.add_argument("--port", type=int, default=8080)
    parser.add_argument("--fps", type=float, default=2.0, help="推理采样 FPS；视频仍连续回传")
    parser.add_argument("--detector", choices=["mock", "yolo-world"], default="mock")
    parser.add_argument("--reasoner", choices=["mock", "relate-anything"], default="mock")
    args = parser.parse_args()
    configure_logging("INFO")
    app = create_app(fps=args.fps, detector=args.detector, reasoner=args.reasoner)
    web.run_app(app, host="127.0.0.1", port=args.port)


if __name__ == "__main__":
    main()

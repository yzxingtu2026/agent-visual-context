# WebRTC 单机浏览器示例边界（Issue #24）

- 示例文件在 `examples/webrtc_browser/`；领域模型、推理流水线不依赖 WebRTC。`ReturnVideoTrack`
  接收浏览器视频并原样回传；采样的 `Frame` 通过 `PushFrameSource` 注入每连接独立的流水线。
- `RemoteInference` 与现有 `LatestFrameBuffer(maxsize=1)` 共用丢旧保新语义。接收/视频回传与
  模型执行分离；单连接推理串行运行于线程，处理过程中最多积压一个采样帧。视频和模型
  分别测吞吐，**视频 30 FPS 不意味着模型 30 FPS**。
- 结果的 `frame_id` 为服务端接收序号；`sampled_at` 是服务端收到帧时刻，不是客户端
  采集时刻。`server_ms` 为服务端处理时长，`queue_ms` 为排队等候时间；
  `queue_dropped` 仅是已入队后被覆盖的采样帧数，`received`/`sampled`/`processed`
  有不同口径。浏览器通过 WebRTC inbound-rtp 统计解码 FPS，DataChannel ping/pong
  的 RTT 不等于视频画面端到端延迟。**本示例未提供跨端精准采集到显示耗时**；
  可先对比左右画面中的动作评估体感，再用水印/同源时钟实验做精确基准。
- 检测框只叠加最近一次结果，不保证与实时回传帧像素严格同步。Mock 框为固定演示位置，
  不应解读为真实检测；若要严谨比较图像/标注同步，需扩展帧时间戳与客户端同步策略。
- 本机信令是 `POST /offer`，只绑定 loopback、不鉴权、不支持 TURN/多客户端。
  **不要直接绑定 `0.0.0.0` 部署内网**：先设计 HTTPS 安全上下文、访问控制、
  ICE 候选与 UDP 防火墙/TURN、资源限额和审计。浏览器摄像头从 `localhost` 页面授权。

## 人工验收

1. `uv sync --extra webrtc`，启动 README 中的服务，从本机打开 `http://localhost:8080`。
2. 点击启动并授权摄像头，检查本地与远端画面均连续变化；快速移动物体，比较回传滞后。
3. 检查解码 FPS、RTT、服务端耗时及处理/丢帧计数更新；调高 `--fps` 时视频仍持续播放。
4. 点击停止，确认摄像头指示灯熄灭；再次启动可重连。拒绝摄像头授权时应能看到错误。
5. 切换真实模型后检查结果及降级字段；首次加载权重的耗时与预热后的统计应分开看。

自动化测试不代替上述真实浏览器/摄像头/GPU 验收。

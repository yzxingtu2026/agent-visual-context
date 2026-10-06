# InsightFace 本地验证与许可证

## 依赖和权重

- SDK：`insightface>=0.7.3,<0.8`，来源 [deepinsight/insightface](https://github.com/deepinsight/insightface)，代码许可证 MIT；运行时还依赖 `onnxruntime>=1.18,<2`（MIT）与 `opencv-python>=4.10,<5`（Apache-2.0）。以本地安装包的 LICENSE 为最终依据。
- 权重：默认模型名 `buffalo_l`，由使用者从 [官方 model zoo](https://github.com/deepinsight/insightface/tree/master/python-package) 单独取得，放入 `~/.insightface/models/buffalo_l/`。本仓库不下载、缓存、提交或分发任何权重。
- 官方预训练模型的使用范围与 SDK 代码许可证不同：官方说明仅供**非商业研究/验证**。商用须单独取得权重及训练数据的相应授权，或更换获授权模型；安装 SDK 不等于获得权重商用许可。
- 运行时健康信息中的模型版本由 SDK 版本与模型包名构成；部署时还应记录本地 ONNX 文件的来源、发布日期及哈希，避免同名权重混用。

## 启用

`uv sync --extra vision --extra face` 安装可选依赖。设置 `AVC_TRACKER_BACKEND=short-term` 与
`AVC_FACE_BACKEND=insightface`，可选 `AVC_FACE_MODEL_DIR`、`AVC_FACE_MODEL_NAME`、
`AVC_FACE_TIMEOUT_SECONDS`。可视化示例可用：

```sh
uv run python examples/run_camera_visual.py --detector yolo-world --tracker short-term --face insightface --face-model-dir ~/.insightface
```

未安装 SDK、权重缺失、推理超时或异常均只降级人脸组件。视频/摄像头的原始像素与特征向量
不写入普通日志、测试仓库、时间线或 Agent 文本。保存标注帧会包含原始图像，须由本地操作者
控制其位置和删除期限。

## 解释与验证边界

`det_score` 仅作为检测质量近似指标，并非年龄/性别估计的校准概率。
SDK 若无对应置信度，输出 `null`。低质量、背脸、遮挡、多人候选、多人争同一人脸时
应保持未知。年龄仅输出十岁年龄段，外观性别只是模型推断，不能当作本人性别事实。
模型在不同年龄、性别表达、肤色、光照与设备上有偏差，应在目标设备和实际场景逐组验证。

Mock/合成时序测试无需权重；真实设备验证须使用已获许可的本地素材或摄像头，记录
设备、输入分辨率、FPS、每组件延迟、误关联案例与遮挡恢复。此项实测依赖本地模型与素材，
未取得前不可宣称真实链路验收通过。

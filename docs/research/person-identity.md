# 匿名人物匹配与受控特征库

Issue #18 的第一阶段实现是进程内 `InMemoryIdentityStore`。它默认关闭，不写磁盘、不进入
时间线、不进入普通 Agent 上下文，也不记录原始人脸或向量。启用后，宿主必须通过查询 API
显式登记并提供授权；登记生成或接收匿名 `person_id`，下游仍需自行完成真实账户绑定。

## 匹配边界

- 只匹配 `FaceObservation.embedding`，向量在内存中归一化；库中的记录严格按 `model_version`
  隔离，模型升级后不会静默混用旧向量。
- 分数低于 candidate 阈值返回 `unknown`；达到 candidate 但未达到 match 阈值返回
  `candidate`；达到 match 阈值还必须连续通过配置的多帧确认才返回 `matched`。
- 最优候选与次优候选差距小于歧义边界时返回 `unknown`。同一帧多个轨迹争用同一个
  `person_id` 时全部拒绝，不能因为相似度而合并轨迹。
- 结果只暴露匿名 ID、状态、分数、模型版本和有效期。`track_id` 仍然只是本次连续出现的
  短时轨迹，不是跨会话身份。

## 生命周期与授权

`register()` 与 `add_sample()` 都要求 `consent=True`。`delete()` 会移除全部样本及待确认
状态；`expire()` 按保留期清理记录。默认配置 `identity_matching_enabled=false`，适配器只会
返回 `unknown`，不会创建或保存特征。

InsightFace 的预训练权重仍仅用于本地受控非商业验证。商用启用长期特征保存前，需要单独
完成模型、训练数据、采集授权、访问控制和保留期限审查。本实现不推断真实姓名、账户或业务
权限，也不接入下游用户数据库。

# 匿名身份适配器约定

适用于实现 `IdentityStore` 或 `IdentityMatcher` 的使用方适配器；完整接线示例见
[`docs/research/person-identity.md`](../research/person-identity.md)。

- 统一通过 `build_mock_pipeline`、`build_offline_pipeline` 或 `build_live_runtime` 的
  `identity_store=` / `identity_matcher=` 注入，不在流水线或查询 API 中判断数据库类型。
- `IdentityStore.search()` 只接收调用期间的人脸向量，按 `model_version`、启用状态、
  有效期过滤，最多返回前两名不含向量的候选。候选分数应在 `[0, 1]` 内，匹配器会
  对失效、版本不符或非法分数再次拒绝。
- `register()` 和 `add_sample()` 必须校验明确授权；`delete()` 和 `disable()` 必须使
  对应人物立即不可检索，删除覆盖所有样本。使用方负责持久化加密、访问控制、保留期
  与过期清理，并在需要原子性的地方使用自己的数据库事务。
- `StoredIdentityMatcher` 统一处理质量过滤、阈值、歧义、多帧确认和同帧冲突。自定义
  matcher 只能输出 `PersonIdentityMatch`；不得把向量写入日志、时间线、事件或普通
  Agent 上下文。自定义 matcher 的异常文本也不会进入流水线日志和健康状态。
- `person_id` 是匿名稳定 ID。`person_id -> user_id` 映射属于使用方系统；`track_id`
  仅代表短时连续轨迹。进程重启后重新连接持久化 store，确认计数从新会话开始。

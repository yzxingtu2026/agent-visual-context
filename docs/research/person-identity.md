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

## 使用方存储与匹配协议（Issue #22）

`IdentityStore` 只定义受控登记、追加样本、删除、禁用、到期清理和 `search()`。
`search()` 必须按 `model_version`、启用状态与保留期过滤，返回按分数排序的前两名
`IdentityCandidate`，不能返回向量。登记和追加样本必须核验授权；删除及禁用后必须立即
停止检索，且持久化后端应实现加密、访问控制和保留期限。`InMemoryIdentityStore` 只是本地
测试实现，不应承载生产环境的全部身份向量。

`StoredIdentityMatcher` 使用 store 的候选执行质量过滤、候选/匹配阈值、第二名差距、
连续多帧确认及同帧多人冲突拒绝。使用方也可以直接实现 `IdentityMatcher.match_faces()`
并通过构建器的 `identity_matcher=` 注入。自定义 matcher 仅可在调用期间使用
`FaceObservation.embedding`，返回 `PersonIdentityMatch`，不得把向量写入日志、事件或
普通 Agent 上下文。流水线暴露的 `FrameResult.face_items` 会去掉 embedding。

下面是数据库适配器的接口示例；`VectorRepository` 由使用方实现，项目不连接具体数据库：

```python
from agent_visual_context.perception import IdentityCandidate, StoredIdentityMatcher
from agent_visual_context.runtime import build_live_runtime

class DatabaseIdentityStore:
    def __init__(self, repository):
        self.repository = repository

    def register(self, embedding, *, model_version, now, consent=False, person_id=None):
        if not consent:
            raise PermissionError("consent required")
        return self.repository.insert_person(embedding, model_version, now, person_id)

    def add_sample(self, person_id, embedding, *, model_version, now, consent=False):
        if not consent:
            raise PermissionError("consent required")
        self.repository.insert_sample(person_id, embedding, model_version, now)

    def delete(self, person_id):
        self.repository.delete_person_and_samples(person_id)

    def disable(self, person_id):
        self.repository.disable_person(person_id)

    def expire(self, *, now):
        return self.repository.delete_expired(now)

    def search(self, embedding, *, model_version, now, limit=2):
        rows = self.repository.search_active(
            embedding, model_version=model_version, expires_after=now, limit=limit
        )
        return tuple(IdentityCandidate(row.person_id, row.score, row.model_version,
                                       row.expires_at) for row in rows)

store = DatabaseIdentityStore(repository)
runtime = build_live_runtime(config, identity_store=store)
# 或注入自定义匹配器：build_live_runtime(config, identity_matcher=my_matcher)
```

例如 pgvector 检索应在数据库中限制 `model_version`、`enabled`、`expires_at > now`，
按向量距离排序并 `LIMIT 2`，并以相同事务处理删除和样本清理。库中稳定的匿名
`person_id` 可由使用方在自己的受控表中关联 `user_id`；该映射、真实姓名和业务权限
始终留在使用方系统。进程重启后重新构建 store 并连接同一持久化后端即可继续匹配，
`track_id` 和多帧确认计数则从新会话重新开始。

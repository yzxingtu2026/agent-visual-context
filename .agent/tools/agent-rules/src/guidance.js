// 为本地操作者生成通用协作说明，不依赖仓库中的身份清单。

function renderRoleGuide(_repo, roleName) {
  return `### ${roleName || "项目协作者"}
- 职责：按当前 Issue/PR 约定完成协作与交付。
- 权限：遵循仓库规则和项目授权。
- 协作偏好：不确定时先在 Issue/PR 中确认。`;
}

function renderCommonConstraints(_repo) {
  return "- 遵循仓库默认分支、Issue/PR、测试和合并规则。";
}

module.exports = { renderCommonConstraints, renderRoleGuide };

# 项目初始化检查清单

- [ ] 替换 `README.md` 中的项目名称、用途和启动说明。
- [ ] 运行 `npx -y @yz-xingtu/agent-rules@latest init`，填写当前使用者姓名、GitHub 用户名和提交邮箱。
- [ ] 运行 `npx -y @yz-xingtu/agent-rules@latest sync` 生成规则和技能文件。
- [ ] 更新 `.agent/rules/main.md` 的开发硬约束，再运行 `sync`。
- [ ] 更新 `.agent/rules/team-escalation.md` 的 Issue/PR 求助方式，再运行 `sync`。
- [ ] 明确哪些目录属于运行时代码，影响 `needs-testing` 判断。
- [ ] 配置 Issue 模板、PR 模板、标签和分支保护。
- [ ] 若使用 `tested-pass` 自动合并，配置对应 GitHub Actions。

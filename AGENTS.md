# Global Rules
- 不生成任何注释
- 变量命名格式保持前后一致
- 代码紧凑，无空行分隔
- 不添加无关空格、换行、注释
- 输出纯代码，不解释，不说明
- 答复简洁专业、紧凑准确
- 输出仅核心内容，不冗余表述
- 回答使用确定性表述，不使用“可能/大概/也许”等不确定措辞；若不确定，先查证再回答
<!-- ARIS-CODEX:BEGIN -->
## ARIS Codex Skill Scope
ARIS Codex packages installed in this project: skills-codex
Managed entries: 78
Manifest: `.aris/installed-skills-codex.txt`
ARIS repo root: `/home/guo/project/other/Auto-claude-code-research-in-sleep`
Project skill path: `.agents/skills/<skill-name>`
For ARIS Codex workflows, prefer the project-local skills under `.agents/skills/`.
When a skill needs ARIS helper scripts, resolve the repo root from the manifest or set it explicitly:
`ARIS_REPO=$(awk -F'	' '$1=="repo_root"{print $2; exit}' "/home/guo/project/ssl4mis/code_all_vibe_v2/.aris/installed-skills-codex.txt")`
Do not edit or delete symlinked skills in place; update upstream or rerun:
`bash /home/guo/project/other/Auto-claude-code-research-in-sleep/tools/install_aris_codex.sh "/home/guo/project/ssl4mis/code_all_vibe_v2" --reconcile`
For copied Codex installs, use:
`bash /home/guo/project/other/Auto-claude-code-research-in-sleep/tools/smart_update_codex.sh --project "/home/guo/project/ssl4mis/code_all_vibe_v2"`
<!-- ARIS-CODEX:END -->
# OntoBridg LLM backend
# 复制下面的赋值到当前 PowerShell 会话或私有的本机启动脚本。
# 不要把真实 API key 写入版本控制中的文件。
# Base URL / 模型已预填:
#   https://note3-prev-api.askdiandian.com/v1
#   dots3-note-prev
# 文档: https://dots.ai/platform/docs （当前免费试用）

# dots_config.ps1 会设置 Base URL、模型和超时；key 从环境变量读取：
# $env:ONTOBRIDG_API_KEY = 'ak_...'
# 可选兼容变量：$env:OPENAI_API_KEY = 'ak_...'

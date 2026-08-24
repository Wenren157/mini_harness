# Mini Harness

**一个最小但完整的 AI Agent Runtime 内核实现**  
从零构建，覆盖单 Agent 与 Multi-Agent 模式，支持工具调用、上下文管理、长期记忆、MCP 协议、沙箱隔离与全链路可观测性。

---

## 项目定位

Mini Harness 是一个用于学习和验证 Agent 系统核心原理的**最小实现**。它模拟了生产级 Agent Harness 的基础架构，但刻意保持轻量、可读、可测试，适合作为面试作品、教学案例或底层框架原型。

**核心公式**：`Model + Harness = Agent`  
**目标岗位**：Agent Harness / Infra 研发工程师

---

## 功能特性

- **ReAct Agent Loop**：异步状态机驱动，支持 LLM → 工具调用 → 结果回写 → 循环，直至最终回答。
- **工具系统**：ToolRegistry + SandboxExecutor，支持并发执行、超时重试、指数退避、死循环防御。
- **上下文管理**：滑动窗口裁剪、异步后台压缩（LLM 摘要）、Token 估算。
- **长期记忆**：LRU 淘汰 + JSON 持久化，支持手动 consolidate。
- **MCP 协议**：stdio JSON-RPC 通信，支持 initialize / tools/list / tools/call，工具能力标准化与进程级隔离。
- **Multi-Agent**：Planner + Executor + Orchestrator 三层隔离，AgentScope 资源隔离，MessageBus 通信。
- **可观测性**：EventBus（deque）记录全链路事件，自动导出 Trace JSON，支持执行时间线与统计摘要。
- **安全沙箱**：路径越权防护、敏感文件拦截、文件大小限制、只读模式。
- **真机 LLM 验证**：支持 DeepSeek/OpenAI API，完整端到端测试。
- **自我架构审计 Demo**：Agent 自主分析自身源码，生成架构审计报告（Day7 收官展示）。

---

## 技术栈

- **Python 3.10+**，asyncio 异步编程
- **pytest** 测试框架
- **OpenAI SDK**（兼容 DeepSeek API）
- **JSON-RPC 2.0** over stdio
- **tiktoken**（Token 估算，可选）
- **PyO3 / Rust**（规划中，用于高性能 Token 计数与 JSON 编解码）

---

## 目录结构

```
mini_harness/
├── src/mini_harness/
│   ├── core/                  # 核心调度层
│   │   ├── runtime.py         # HarnessRuntime（Agent Loop + 状态机）
│   │   ├── models.py          # 事件类型、Agent 状态模型
│   │   └── interfaces.py      # LLMClient 抽象接口
│   ├── agents/                # Multi-Agent 编排层
│   │   ├── planner.py         # Planner（任务拆解）
│   │   ├── executor.py        # Executor（顺序执行步骤）
│   │   ├── orchestrator.py    # Orchestrator（Plan→Execute→Aggregate）
│   │   ├── scope.py           # AgentScope（资源隔离）
│   │   └── message_bus.py     # MessageBus（Agent 通信）
│   ├── infra/                 # 基础设施层
│   │   ├── tools.py           # ToolRegistry + SandboxExecutor
│   │   ├── context.py         # ContextManager（滑动窗口 + 压缩）
│   │   ├── memory.py          # LongTermMemory（LRU + 持久化）
│   │   ├── llm_client.py      # OpenAI 兼容 LLM 客户端
│   │   ├── real_llm_client.py # 真实 LLM 客户端（Function Calling）
│   │   └── config.py          # RuntimeConfig 配置
│   ├── mcp/                   # MCP 协议层
│   │   ├── server.py          # MCP Server（子进程）
│   │   ├── client.py          # MCP Client（stdio JSON-RPC）
│   │   └── protocol.py        # JSON-RPC 消息协议
│   └── __init__.py
├── demo/
│   └── self_architecture_audit.py   # Day7 自我架构审计 Demo
├── tests/                     # 单元、集成、端到端、真机测试
├── traces/                    # 运行时自动生成的 Trace 文件
├── workspace/                 # 沙箱工作目录
├── pyproject.toml
└── README.md
```

---

## 快速开始

### 1. 环境准备

```bash
# 克隆仓库
git clone <your-repo-url>
cd mini_harness

# 创建虚拟环境
python -m venv .venv
source .venv/bin/activate  # Windows: .venv\Scripts\activate

# 安装依赖
pip install -r requirements.txt
```

### 2. 配置 LLM API Key

本机运行需要 DeepSeek 或 OpenAI 兼容的 API Key。在项目根目录创建 `.env` 文件：

```
DEEPSEEK_API_KEY=sk-xxx
```

或设置环境变量：

```bash
export DEEPSEEK_API_KEY=sk-xxx
```

### 3. 运行测试

```bash
# 运行全部测试（smoke + regression + e2e）
python tests/run_all.py all

# 仅运行回归测试
python tests/run_all.py regression

# 运行真机 LLM 测试（需要 API Key）
pytest tests/real_llm/ -v
```

### 4. 运行自我架构审计 Demo

```bash
python demo/self_architecture_audit.py
```

该 Demo 将启动一个只读审计 Agent，自动读取 `src/mini_harness` 下的核心源码，执行任务拆解、工具调用，最终生成 `architecture_audit_report.md` 审计报告，并导出 Trace 文件到 `traces/` 目录。

---

## 使用示例

### 单 Agent 运行（代码片段）

```python
import asyncio
from mini_harness.core.runtime import HarnessRuntime
from mini_harness.infra.real_llm_client import RealLLMClient
from mini_harness.infra.tools import ToolRegistry, SandboxExecutor
from mini_harness.infra.context import ContextManager
from mini_harness.infra.config import RuntimeConfig

async def main():
    # 初始化组件
    llm = RealLLMClient()
    sandbox = SandboxExecutor("workspace")
    tools = ToolRegistry(sandbox)
    tools.register_default_tools()
    
    context = ContextManager(llm_client=llm, max_tokens=8000)
    config = RuntimeConfig(workspace="workspace", max_iterations=10)
    
    runtime = HarnessRuntime(
        config=config,
        llm_client=llm,
        tool_registry=tools,
        context_manager=context,
    )
    
    result = await runtime.run("创建一个 hello.txt 文件，内容为 Hello")
    print(result["final_answer"])

asyncio.run(main())
```

### Multi-Agent 运行

```python
from mini_harness.agents.planner import Planner
from mini_harness.agents.orchestrator import Orchestrator

planner = Planner(llm_client=llm)
orchestrator = Orchestrator(runtime=runtime, planner=planner)
result = await orchestrator.run_goal("分析项目结构并生成报告")
```

---

## 架构设计决策

### 为什么 Runtime 不直接调用工具？

所有工具调用通过 MCP 协议或 ToolRegistry 统一管理，保证上下文记录、事件追踪、错误处理的一致性，避免执行路径分散。

### 为什么 Planner 和 Executor 分离？

Planner 只输出抽象步骤，Executor 负责驱动 Runtime 执行，两者解耦，可独立替换策略（例如用 LLM 规划或规则规划）。

### 为什么 Orchestrator 不污染 Runtime？

Runtime 核心循环保持稳定，Multi-Agent 编排逻辑通过 Orchestrator 注入，未来扩展（如 DAG 调度、消息队列）不会影响基础执行。

### 为什么砍掉向量检索和自动 consolidation？

项目定位为最小实现，ContextManager 已承担短期上下文管理，长期记忆只需 LRU + 持久化即可验证核心链路。语义检索和自动压缩作为 `# TODO` 扩展点保留。

---

## 测试覆盖

| 测试分组 | 覆盖范围 | 命令 |
|---------|---------|------|
| smoke | Runtime、MCP 基础链路 | `python tests/run_all.py smoke` |
| regression | 上下文、内存、沙箱、多 Agent 单元测试 | `python tests/run_all.py regression` |
| e2e | 完整链路、工具调用、多 Agent 集成 | `python tests/run_all.py e2e` |
| real_llm | 真实 LLM 端到端（需 API Key） | `pytest tests/real_llm/ -v` |

当前所有测试均通过（ALL TESTS PASSED）。

---

## 可观测性

每次运行自动生成 Trace JSON 文件（位于 `traces/`），包含：

- 用户输入
- LLM 请求/响应
- 工具调用请求/结果
- 错误事件
- 完成状态
- 时间戳与统计信息

可通过 Trace 文件进行执行回放和排障分析。

---

## 安全与约束

- 敏感文件黑名单：禁止读取 `.env`, `*.key`, `*.pem` 等。
- 路径白名单：仅允许访问 `src/mini_harness` 和 `demo` 目录（Demo 中可配置）。
- 沙箱路径防护：防止目录穿越（`../`）。
- 文件大小限制：默认 1MB 读取上限。
- 只读模式：Demo 中不包含写工具，确保审计过程不修改任何文件。

---

## 未来扩展（TODO）

- [ ] 高性能 Token 计数器（Rust/PyO3）
- [ ] JSON-RPC 序列化加速（Rust serde）
- [ ] 子进程资源限制（cgroup/rlimit）
- [ ] 向量检索与语义记忆
- [ ] DAG 调度与 Agent 间消息队列
- [ ] 指标监控与告警（Prometheus）
- [ ] 优雅退出与状态持久化

---

## 贡献

欢迎提 Issue 和 PR。本项目为个人学习与面试展示项目，但任何改进建议都值得讨论。

---

## 许可证

MIT License

---

**作者**：闻人 
**GitHub**：<https://github.com/Wenren157>  
**求职意向**：Agent Harness / Infra 研发工程师（DeepSeek Code Harness）
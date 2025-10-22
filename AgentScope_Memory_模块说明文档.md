# AgentScope Memory 模块详细说明文档

## 概述

AgentScope 的 Memory 模块提供了一套完整的记忆管理系统，支持短期记忆（InMemoryMemory）和长期记忆（Mem0LongTermMemory）功能。该模块允许 AI 智能体在对话过程中保持上下文信息，并能够跨会话持久化重要信息。

## 架构设计

Memory 模块采用分层设计，包含以下核心组件：

```
agentscope.memory/
├── __init__.py                     # 模块导出接口
├── _memory_base.py                 # 记忆基类定义
├── _in_memory_memory.py            # 内存记忆实现（短期记忆）
├── _long_term_memory_base.py       # 长期记忆基类
├── _mem0_long_term_memory.py       # 基于 mem0 的长期记忆实现
└── _mem0_utils.py                  # mem0 集成工具类
```

## 核心类说明

### 1. MemoryBase（记忆基类）

`MemoryBase` 是所有记忆类的抽象基类，继承自 `StateModule`，定义了记忆系统的基本接口：

```python
class MemoryBase(StateModule):
    async def add(*args, **kwargs) -> None:        # 添加记忆项
    async def delete(*args, **kwargs) -> None:     # 删除记忆项
    async def retrieve(*args, **kwargs) -> None:   # 检索记忆项
    async def size() -> int:                       # 获取记忆大小
    async def clear() -> None:                     # 清除记忆内容
    async def get_memory(*args, **kwargs) -> list[Msg]:  # 获取记忆内容
    def state_dict() -> dict:                      # 获取状态字典
    def load_state_dict(state_dict: dict, strict: bool = True) -> None:  # 加载状态
```

### 2. InMemoryMemory（短期记忆）

`InMemoryMemory` 是基于内存的短期记忆实现，主要用于存储当前会话的消息历史。

#### 主要特性：
- **内存存储**：所有数据存储在内存中，会话结束后数据丢失
- **消息管理**：存储和管理 `Msg` 对象列表
- **状态持久化**：支持通过 `state_dict()` 和 `load_state_dict()` 进行状态保存和恢复
- **重复检测**：可选择是否允许添加重复消息（基于消息 ID）

#### 核心方法：

```python
# 添加消息到记忆
await memory.add(
    memories=messages,           # Msg 对象或 Msg 对象列表
    allow_duplicates=False       # 是否允许重复消息
)

# 删除指定索引的消息
await memory.delete(index)       # 单个索引或索引列表

# 获取所有记忆内容
messages = await memory.get_memory()

# 获取记忆大小
size = await memory.size()

# 清除所有记忆
await memory.clear()
```

#### 使用示例：

```python
from agentscope.memory import InMemoryMemory
from agentscope.message import Msg

# 创建内存记忆实例
memory = InMemoryMemory()

# 添加消息
msg = Msg(role="user", content="Hello, how are you?", name="user")
await memory.add(msg)

# 获取所有消息
all_messages = await memory.get_memory()

# 获取记忆大小
size = await memory.size()
print(f"Memory contains {size} messages")
```

### 3. LongTermMemoryBase（长期记忆基类）

`LongTermMemoryBase` 定义了长期记忆系统的接口，支持跨会话的记忆管理：

#### 核心概念：
- **开发者方法**：`record()` 和 `retrieve()` - 供开发者在代码中调用
- **智能体工具方法**：`record_to_memory()` 和 `retrieve_from_memory()` - 作为工具供智能体主动调用

#### 主要方法：

```python
# 开发者使用的记录方法
await memory.record(msgs, **kwargs)

# 开发者使用的检索方法
result = await memory.retrieve(msg, **kwargs)

# 智能体工具：记录重要信息
response = await memory.record_to_memory(
    thinking="思考和推理过程",
    content=["要记住的内容列表"]
)

# 智能体工具：基于关键词检索记忆
response = await memory.retrieve_from_memory(
    keywords=["关键词列表"]
)
```

### 4. Mem0LongTermMemory（基于 mem0 的长期记忆）

`Mem0LongTermMemory` 是基于 mem0 库的长期记忆实现，提供了强大的语义记忆功能。

#### 核心特性：
- **语义搜索**：基于向量相似度的语义记忆检索
- **持久化存储**：支持多种向量数据库（默认使用 Qdrant）
- **多模型支持**：支持多种 LLM 和嵌入模型
- **用户分离**：基于 agent_name、user_name、run_name 进行记忆隔离
- **异步操作**：完全异步的记忆操作，不阻塞主线程

#### 初始化参数：

```python
long_term_memory = Mem0LongTermMemory(
    agent_name="Friday",                    # 智能体名称
    user_name="user_123",                   # 用户名称
    run_name="session_001",                 # 会话名称
    model=chat_model,                       # 聊天模型实例
    embedding_model=embedding_model,        # 嵌入模型实例
    vector_store_config=vector_config,      # 向量存储配置
    mem0_config=mem0_config,               # mem0 完整配置
    default_memory_type=None,              # 默认记忆类型
    on_disk=True                           # 是否持久化到磁盘
)
```

#### 记忆记录：

```python
# 记录单个消息
await memory.record(
    msgs=[Msg(role="user", content="我喜欢住民宿", name="user")],
    memory_type="semantic",  # 记忆类型：semantic, procedural_memory
    infer=True              # 是否推理记忆内容
)

# 使用工具方法记录（供智能体调用）
response = await memory.record_to_memory(
    thinking="用户表达了住宿偏好，这是重要的个人信息",
    content=["用户偏好住民宿而不是酒店"]
)
```

#### 记忆检索：

```python
# 基于消息内容检索
memories = await memory.retrieve(
    msg=Msg(role="user", content="我的住宿偏好是什么？", name="user"),
    limit=5  # 限制返回结果数量
)

# 使用工具方法检索（供智能体调用）
response = await memory.retrieve_from_memory(
    keywords=["住宿", "偏好", "民宿"],
    limit=5
)
```

## 长短期记忆的生成机制

### 短期记忆生成流程

1. **消息接收**：当智能体接收到新消息时，自动添加到 InMemoryMemory
2. **上下文维护**：InMemoryMemory 维护当前对话的完整上下文
3. **内存管理**：所有消息存储在内存中，提供快速访问
4. **会话结束**：会话结束后，短期记忆内容丢失（除非手动保存状态）

```python
# 短期记忆的典型使用流程
memory = InMemoryMemory()

# 1. 用户消息添加到记忆
user_msg = Msg(role="user", content="今天天气怎么样？", name="user")
await memory.add(user_msg)

# 2. 智能体生成回复并添加到记忆
agent_msg = Msg(role="assistant", content="今天是晴天，气温25度。", name="assistant")
await memory.add(agent_msg)

# 3. 获取完整对话历史用于下一轮对话
conversation_history = await memory.get_memory()
```

### 长期记忆生成流程

1. **内容分析**：mem0 使用 LLM 分析输入内容的重要性和语义
2. **向量化**：使用嵌入模型将内容转换为向量表示
3. **存储持久化**：将向量和元数据存储到向量数据库
4. **语义检索**：基于向量相似度进行语义搜索和检索

```python
# 长期记忆的生成流程示例

# 1. 智能体决定记录重要信息
await long_term_memory.record_to_memory(
    thinking="用户询问了天气，并且提到了偏好的温度单位",
    content=[
        "用户询问天气信息",
        "用户偏好摄氏度而不是华氏度",
        "当前日期：2024-01-15"
    ]
)

# 2. mem0 内部处理流程：
#    a) LLM 分析内容重要性
#    b) 提取关键信息和实体
#    c) 生成结构化记忆
#    d) 使用嵌入模型生成向量
#    e) 存储到向量数据库

# 3. 后续检索（可能在几天后的新会话中）
memories = await long_term_memory.retrieve_from_memory(
    keywords=["天气", "偏好", "温度单位"]
)
# 返回：["用户偏好摄氏度而不是华氏度"]
```

### 记忆层次结构

AgentScope 的记忆系统采用分层架构：

```
智能体记忆架构
├── 短期记忆 (InMemoryMemory)
│   ├── 当前会话消息
│   ├── 上下文窗口管理
│   └── 临时状态信息
│
└── 长期记忆 (Mem0LongTermMemory)
    ├── 用户偏好和特征
    ├── 重要事实和知识
    ├── 历史交互模式
    └── 跨会话持久信息
```

## 智能体集成

### ReActAgent 集成示例

```python
from agentscope.agent import ReActAgent
from agentscope.memory import InMemoryMemory, Mem0LongTermMemory

# 创建长期记忆
long_term_memory = Mem0LongTermMemory(
    agent_name="Friday",
    user_name="user_123",
    model=chat_model,
    embedding_model=embedding_model
)

# 创建智能体，同时配置短期和长期记忆
agent = ReActAgent(
    name="Friday",
    sys_prompt="You are a helpful assistant named Friday.",
    model=chat_model,
    memory=InMemoryMemory(),              # 短期记忆
    long_term_memory=long_term_memory,    # 长期记忆
    long_term_memory_mode="both"          # 记忆模式：both, agent_control, dev_control
)
```

### 记忆模式说明

- **agent_control**：智能体主动控制长期记忆的记录和检索
- **dev_control**：开发者在代码中控制长期记忆操作
- **both**：智能体和开发者都可以控制长期记忆

## 配置选项

### mem0 配置示例

```python
from mem0.configs.base import MemoryConfig
from mem0.embeddings.configs import EmbedderConfig
from mem0.llms.configs import LlmConfig
from mem0.vector_stores.configs import VectorStoreConfig

# 创建完整的 mem0 配置
mem0_config = MemoryConfig(
    llm=LlmConfig(
        provider="agentscope",
        config={"model": chat_model}
    ),
    embedder=EmbedderConfig(
        provider="agentscope", 
        config={"model": embedding_model}
    ),
    vector_store=VectorStoreConfig(
        provider="qdrant",
        config={
            "on_disk": True,
            "path": "./memory_data"
        }
    )
)

# 使用配置创建长期记忆
long_term_memory = Mem0LongTermMemory(
    agent_name="Friday",
    user_name="user_123", 
    mem0_config=mem0_config
)
```

## 最佳实践

### 1. 记忆内容设计
- **具体性**：记录具体、可操作的信息而非抽象概念
- **上下文完整**：包含足够的上下文信息便于后续理解
- **结构化**：使用结构化格式组织复杂信息

### 2. 检索优化
- **关键词选择**：使用具体、相关的关键词提高检索精度
- **限制结果数量**：根据使用场景设置合理的结果限制
- **错误处理**：优雅处理检索失败和空结果情况

### 3. 性能优化
- **批量操作**：尽可能将相关的记忆操作批量处理
- **定期清理**：定期清理过时或无关的记忆内容
- **配置调优**：根据使用场景优化向量存储和嵌入配置

## 示例代码

### 完整的记忆管理示例

```python
import asyncio
from agentscope.memory import InMemoryMemory, Mem0LongTermMemory
from agentscope.message import Msg
from agentscope.model import OpenAIChatModel
from agentscope.embedding import OpenAITextEmbedding

async def memory_example():
    # 创建短期记忆
    short_memory = InMemoryMemory()
    
    # 创建长期记忆
    long_memory = Mem0LongTermMemory(
        agent_name="Assistant",
        user_name="User123",
        model=OpenAIChatModel(model_name="gpt-4"),
        embedding_model=OpenAITextEmbedding(model_name="text-embedding-3-small")
    )
    
    # 短期记忆操作
    msg = Msg(role="user", content="我喜欢喝咖啡，不喜欢茶", name="user")
    await short_memory.add(msg)
    
    # 长期记忆记录
    await long_memory.record_to_memory(
        thinking="用户表达了饮品偏好，这是重要的个人信息",
        content=["用户喜欢咖啡", "用户不喜欢茶"]
    )
    
    # 长期记忆检索
    memories = await long_memory.retrieve_from_memory(
        keywords=["饮品", "偏好", "咖啡"]
    )
    
    print(f"检索到的记忆: {memories}")

# 运行示例
asyncio.run(memory_example())
```

## 故障排除

### 常见问题

1. **记忆未找到**
   - 检查 agent_name、user_name 等标识符是否一致
   - 验证记忆是否正确记录
   - 确认向量存储配置正确

2. **检索结果不准确**
   - 使用更具体的关键词
   - 检查嵌入模型配置
   - 验证记录时的内容格式

3. **性能问题**
   - 优化向量存储配置
   - 减少检索结果限制
   - 考虑使用磁盘存储处理大数据集

### 调试建议

```python
import logging
logging.basicConfig(level=logging.DEBUG)

# 启用调试日志以排查问题
```

## 总结

AgentScope 的 Memory 模块提供了完整的记忆管理解决方案，通过短期记忆维护对话上下文，通过长期记忆实现跨会话的知识积累。这种分层设计使得智能体能够在保持对话连贯性的同时，积累和利用长期知识，为构建更智能、更人性化的 AI 助手提供了强大的基础设施。

关键优势：
- **灵活性**：支持多种存储后端和模型配置
- **可扩展性**：基于向量数据库的高效语义搜索
- **易用性**：简洁的 API 设计和完善的集成支持
- **持久性**：跨会话的记忆保持和恢复能力

通过合理使用短期和长期记忆，开发者可以构建出具有学习能力和个性化体验的智能对话系统。

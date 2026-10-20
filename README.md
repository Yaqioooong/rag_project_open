# RAG 知识库问答系统

一个面向半导体材料、芯片制造与光刻技术文档的本地 RAG 项目，覆盖文档解析、语义分块、Milvus 稠密/稀疏双索引、混合检索、LangGraph 自检问答和网络搜索降级。

## 功能特性

- **文档入库**：使用 Unstructured 解析 Markdown，合并标题与正文，对超长章节做语义切片，并通过多进程队列解耦解析和写入。
- **混合检索**：BGE-small-zh 稠密向量 + Milvus 内置 BM25 稀疏向量，使用 RRF 融合排序。
- **Corrective RAG**：Agent 调用检索工具，检索后执行文档相关性评分、查询重写和答案生成。
- **Adaptive RAG**：根据问题主题在本地知识库与 Tavily 网络搜索之间动态路由，并执行文档、答案和幻觉评分。
- **模型适配**：通过 OpenAI 兼容接口接入 DeepSeek，使用 JSON Mode 完成结构化评分，兼容 Tool Calling。
- **本地优先**：默认使用 Milvus Lite，不需要单独部署数据库服务。

## 架构

```text
data/md
   │
   ▼
Unstructured 解析 ──► 标题/正文合并 ──► 语义分块 ──► 多进程批量入库
                                                        │
                                                        ▼
                                              Milvus Lite Collection
                                        ┌───────────────┴───────────────┐
                                        │                               │
                                  BGE 稠密向量                     BM25 稀疏向量
                                        └───────────────┬───────────────┘
                                                        ▼
                                                RRF 混合检索
                                                        │
                              ┌─────────────────────────┴─────────────────────────┐
                              ▼                                                   ▼
                       Corrective RAG                                      Adaptive RAG
                 Agent → Retrieve → Grade → Rewrite                    Route → Retrieve / Web
                              │                                                   │
                              └──────────────────────► Generate ◄─────────────────┘
                                                        │
                                                        ▼
                                                      回答
```

## 目录结构

```text
.
├── README.md
├── requirements.txt
├── pyproject.toml
├── .env.example
├── data/
│   └── md/                     # 示例知识库文档
├── src/
│   ├── rag_cli.py              # 统一命令行入口
│   ├── agent/                  # 普通 Tool Calling Agent
│   ├── documents/              # Markdown 解析、Milvus 建表与批量写入
│   ├── graph_adaptive/         # Adaptive RAG 工作流
│   ├── graph_corrective/       # Corrective RAG 工作流
│   ├── llm_models/             # DeepSeek / 嵌入模型配置
│   ├── tools/                  # Retriever Tool 封装
│   └── utils/                  # 环境变量、日志与打印工具
└── test/                       # 原理验证与调试脚本
```

## 快速开始

### 1. 创建环境

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
pip install -e .
```

建议使用 Python 3.11 及以上版本。首次运行会从 Hugging Face 下载 `BAAI/bge-small-zh-v1.5`，如果使用离线环境，请提前准备模型缓存并设置 `HF_HUB_OFFLINE=1`。

### 2. 配置环境变量

```bash
cp .env.example .env
```

填写 `.env` 中的 DeepSeek、阿里云嵌入接口和 Tavily 密钥。`.env` 已被 `.gitignore` 排除，不会提交到仓库。

| 变量 | 用途 |
| --- | --- |
| `DEEPSEEK_API_KEY` | DeepSeek API 密钥 |
| `DEEPSEEK_BASE_URL` | DeepSeek OpenAI 兼容接口地址 |
| `DEEPSEEK_MODEL` | 对话模型名称，默认 `deepseek-v4-flash` |
| `ALIBABA_API_KEY` / `ALIBABA_BASE_URL` | SemanticChunker 使用的嵌入接口 |
| `TAVILY_API_KEY` | Adaptive RAG 网络搜索 |
| `MILVUS_LITE_URI` | Milvus Lite 数据文件路径 |
| `COLLECTION_NAME` | 向量集合名称 |

### 3. 构建知识库

```bash
python -m rag_cli ingest --data-dir data/md
```

也可以直接运行：

```bash
python src/documents/write_milvus.py --data-dir data/md
```

### 4. 运行问答工作流

```bash
# Adaptive RAG：知识库 / 网络搜索动态路由
python -m rag_cli adaptive

# Corrective RAG：检索自检与查询重写
python -m rag_cli corrective

# 普通 Tool Calling Agent
python -m rag_cli agent --query "什么是EUV光刻机？"
```

安装 `pyproject.toml` 后也可以使用统一入口：

```bash
rag-demo --help
rag-demo adaptive
rag-demo corrective
```

## 文档与元数据

默认扫描 `data/md` 下的 Markdown 文件。写入 Milvus 前会补齐以下字段：

- `title`：缺失时使用文件名
- `category_depth`：缺失时使用 `0`
- `category`、`source`、`filename`、`filetype`：由解析器生成

如果整批文档中有一条记录缺少 Milvus Schema 中的必填字段，整批插入会失败，因此入库前应保持元数据字段一致。

## 常见问题

### DeepSeek 报 `response_format type is unavailable`

当前 DeepSeek 接口不支持 OpenAI 的 `json_schema` 结构化输出。项目中的评分链使用 `method="json_mode"`，并在提示词中要求模型只返回 JSON。

### 报 `only drop_ratio_build=0 is supported`

BM25 稀疏索引在当前 Milvus Lite 中要求 `drop_ratio_build=0`，同时兼容性参数需要在 `search_params` 中显式配置。

### 报 `another process holds the lock`

Milvus Lite 是嵌入式单进程数据库。父进程和子进程不能同时打开同一个 `.db` 文件，入库时应只让写入进程打开数据库，或改用独立部署的 Milvus Server。

## 说明

- `test/` 下的文件是原理验证和调试脚本，不是完整的自动化测试套件。
- 示例知识库仅用于演示，正式使用时应替换为自己的文档并确认数据授权。
- 本项目默认不包含模型权重、运行数据库和生成的输出文件。

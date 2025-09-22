# openGauss 向量数据库与 Arize Phoenix 的对接与评估最佳实践

在构建高效的 RAG 系统时，结合强大的向量数据库和先进的评估框架是实现性能优化的关键。openGauss 凭借其高效的向量存储与检索能力，为 RAG 管道提供了可靠的数据支持，而 Arize Phoenix 则通过其全面的评估工具，帮助用户量化 RAG 管道的性能表现，识别潜在瓶颈并优化系统。

本文将为您提供一份**从 openGauss 部署到 Arize Phoenix 集成的完整指南**，帮助您掌握 openGauss 在容器中的部署方法、RAG 管道的构建，以及使用 Arize Phoenix 进行检索和生成评估的最佳实践，从而构建更高效、可靠的 RAG 系统。

## 1. 环境准备

+ 已部署 7.0.0-RC1 及以上版本的openGauss实例，容器部署参考[容器镜像安装](https://docs.opengauss.org/zh/docs/latest-lite/docs/InstallationGuide/容器镜像安装.html)

+ 已安装 3.10 及以上版本的Python环境

+ 已安装涉及的Python库

  ```shell
  pip3 install pandas tqdm psycopg2 requests openai arize-phoenix openinference-instrumentation-openai
  ```

> 若出现报错`ImportError: cannot import name 'LoopSetupType' from 'uvicorn.config'`，可参考[Issue #9569 · Arize-ai/phoenix](https://github.com/Arize-ai/phoenix/issues/9569)，将`uvicorn`降版本即可。经测试`uvicorn`为`0.34.0`可用，即`pip install uvicorn==0.34.0`。

在本例程中，我们使用DeepSeek作为LLM，因此需要将配置文件中环境变量`OPENAI_API_KEY`设置为DeepSeek API Key；此外，本例中使用的嵌入模型为硅基流动平台的`Qwen/Qwen3-Embedding-0.6B`模型，故还需要设置硅基流动的API Key，嵌入模型选择参考[嵌入模型](https://docs.opengauss.org/zh/docs/latest/docs/DataVec/embedding-bgem3.html)。

```python
# DeepSeek API key
DEEPEVAL_API_KEY = "sk-xxxxxxxxxx"

# SiliconFlow API key
SILICONFLOW_API_KEY = "sk-xxxxxxxxxx"

DB_CONFIG = {
    "host": "localhost",  # 数据库服务器地址
    "port": 8888,  # 数据库服务端口号
    "database": "YourDbName",  # 要连接的数据库名称
    "user": "YourUserName",  # 数据库用户名
    "password": "YourUserPassword",  # 数据库密码
}
```

## 2. 定义 RAG 管道

本节中我们将定义一个使用openGauss DataVec作为向量存储，DeepSeek作为LLM的RAG类。该类包含如下方法：

+ `load`: 将文本数据转换为向量嵌入并存储到 openGauss 数据库中
+ `retrieve`: 根据问题检索最相似的`top_k`个文本片段
+ `answer`: 结合检索到的文本和 LLM 生成最终答案

```python
class RAG:
    """
    基于 DeepSeek、Qwen/Qwen3-Embedding-0.6B 和 openGauss 构建的 RAG 类。
    """

    def __init__(
        self, openai_client: OpenAI, db_config: dict, table_name: str = "rag_table"
    ):
        self._prepare_openai(openai_client)
        self._prepare_opengauss(db_config, table_name)

    def _emb_text(self, text: str) -> List[float]:
        """使用 Qwen/Qwen3-Embedding-0.6B 模型通过 API 生成文本的向量嵌入"""
        try:
            payload = {"model": "Qwen/Qwen3-Embedding-0.6B", "input": text}
            headers = {
                "Authorization": f"Bearer {SILICONFLOW_API_KEY}",
                "Content-Type": "application/json",
            }

            response = requests.post(
                SILICONFLOW_BASE_URL, json=payload, headers=headers
            )
            response.raise_for_status()

            result = response.json()
            return result["data"][0]["embedding"]
        except Exception as e:
            print(f"生成嵌入向量时出错：{e}")
            raise e

    def _prepare_openai(self, openai_client: OpenAI, llm_model: str = "deepseek-chat"):
        """初始化 OpenAI 客户端和模型配置"""
        self.openai_client = openai_client
        self.llm_model = llm_model
        self.SYSTEM_PROMPT = (
            "你是一个AI助手。你能够从提供的上下文段落片段中找到问题的答案。"
        )
        self.USER_PROMPT = """使用以下用<context>标签包围的信息片段来回答用<question>标签包围的问题。
                            <context>
                            {context}
                            </context>
                            <question>
                            {question}
                            </question>"""

    def _prepare_opengauss(self, db_config: dict, table_name: str):
        """初始化 openGauss 数据库连接和表结构"""
        self.db_config = db_config
        self.table_name = table_name
        self.conn = psycopg2.connect(**db_config)
        self.cursor = self.conn.cursor()

        # 创建 schema
        self.schema_name = "test_schema"
        try:
            self.cursor.execute("CREATE SCHEMA IF NOT EXISTS test_schema;")
            self.conn.commit()
            print(f"使用 schema: {self.schema_name}")
        except psycopg2.Error as e:
            print(f"Schema 创建失败，使用默认: {e}")
            self.schema_name = "public"
            self.conn.rollback()

        # 获取嵌入维度并创建表和索引
        print("正在检测向量维度...")
        self.embedding_dim = len(self._emb_text("示例"))
        print(f"向量维度：{self.embedding_dim}")

        self._create_table(self.table_name, self.embedding_dim)
        self._create_index(self.table_name, f"idx_{self.table_name}")

    def _create_table(self, table_name: str, dim: int):
        """创建存储文本和向量嵌入的表"""
        self.cursor.execute(
            sql.SQL(
                "CREATE TABLE IF NOT EXISTS {schema}.{table_name} (id BIGINT PRIMARY KEY, text TEXT, embedding vector({dim}));"
            ).format(
                schema=sql.Identifier(self.schema_name),
                table_name=sql.Identifier(table_name),
                dim=sql.Literal(dim),
            )
        )
        self.conn.commit()
        print(f"表 {self.schema_name}.{table_name} 创建成功！")

    def _create_index(self, table_name: str, index_name: str):
        """为向量字段创建 HNSW 索引以提高检索性能"""
        self.cursor.execute(
            sql.SQL(
                "CREATE INDEX IF NOT EXISTS {index_name} ON {schema}.{table_name} USING hnsw (embedding vector_l2_ops);"
            ).format(
                index_name=sql.Identifier(index_name),
                schema=sql.Identifier(self.schema_name),
                table_name=sql.Identifier(table_name),
            )
        )
        self.conn.commit()
        print(f"索引 {index_name} 创建成功！")

    def load(self, texts: List[str]):
        """将文本数据加载到 openGauss 数据库中"""
        print("开始加载数据...")

        # 清空现有数据
        self.cursor.execute(
            sql.SQL("DELETE FROM {schema}.{table_name};").format(
                schema=sql.Identifier(self.schema_name),
                table_name=sql.Identifier(self.table_name),
            )
        )
        self.conn.commit()
        print("已清空现有数据")

        # 生成嵌入向量并插入数据
        for i, text in enumerate(tqdm(texts, desc="正在创建向量嵌入")):
            emb = self._emb_text(text)
            self.cursor.execute(
                sql.SQL(
                    "INSERT INTO {schema}.{table_name} (id, text, embedding) VALUES (%s, %s, %s);"
                ).format(
                    schema=sql.Identifier(self.schema_name),
                    table_name=sql.Identifier(self.table_name),
                ),
                (i, text, emb),
            )
        self.conn.commit()
        print("数据插入成功！")

    def retrieve(self, question: str, top_k: int = 3) -> List[str]:
        """检索与给定问题最相似的文本数据"""
        query_emb = self._emb_text(question)

        # 设置查询参数以优化向量搜索
        self.cursor.execute("SET enable_seqscan = off;")

        # 执行向量相似度搜索（使用 L2 距离）
        self.cursor.execute(
            sql.SQL(
                "SELECT text FROM {schema}.{table_name} ORDER BY embedding <-> %s::vector ASC LIMIT %s;"
            ).format(
                schema=sql.Identifier(self.schema_name),
                table_name=sql.Identifier(self.table_name),
            ),
            (query_emb, top_k),
        )

        results = self.cursor.fetchall()
        return [row[0] for row in results] if results else []

    def answer(
        self,
        question: str,
        retrieval_top_k: int = 3,
        return_retrieved_text: bool = False,
    ):
        """基于检索到的知识回答给定的问题"""
        retrieved_texts = self.retrieve(question, top_k=retrieval_top_k)
        user_prompt = self.USER_PROMPT.format(
            context="\n".join(retrieved_texts), question=question
        )
        response = self.openai_client.chat.completions.create(
            model=self.llm_model,
            messages=[
                {"role": "system", "content": self.SYSTEM_PROMPT},
                {"role": "user", "content": user_prompt},
            ],
        )
        if not return_retrieved_text:
            return response.choices[0].message.content
        else:
            return response.choices[0].message.content, retrieved_texts

    def __del__(self):
        """清理数据库连接"""
        try:
            if hasattr(self, "cursor") and self.cursor:
                self.cursor.close()
            if hasattr(self, "conn") and self.conn:
                self.conn.close()
        except Exception:
            pass
```

使用OpenAI实例和openGauss DataVec对此 RAG 类进行初始化如下。

```python
from openai import OpenAI

openai_client = OpenAI(
    api_key=os.environ["OPENAI_API_KEY"], base_url=os.environ["OPENAI_BASE_URL"]
)

db_config = DB_CONFIG

# 创建 RAG 实例
rag = RAG(openai_client, db_config)
```

## 3. 测试 RAG 管道

在本例中，我们选用[openGauss社区官方文档](https://gitcode.com/opengauss/docs)作为RAG知识库，可以使用命令

```shell
git clone https://gitcode.com/opengauss/docs.git
```

下载对应的文档，随后将这些文档加载至 RAG 管道中。

```python
def read_markdown_files(docs_path: str) -> List[str]:
    """
    读取指定路径下的所有markdown文件，按#分割成块。
    """
    texts = []
    for file_path in glob.glob(os.path.join(docs_path, "**", "*.md"), recursive=True):
        with open(file_path, "r", encoding="utf-8") as file:
            file_text = file.read()
        # 按#分割，过滤空块
        text_lines = [line.strip() for line in file_text.split("# ") if line.strip()]
        for line in text_lines:
            if len(line) > 500:  
                # 分割成子块，按句子或段落分割以保留语义
                sentences = re.split(r"(?<=[.!?])\s+", line)  # 按句子分割
                chunk = ""
                for sentence in sentences:
                    if len(chunk + sentence) > 500:
                        if chunk:
                            texts.append(chunk.strip())
                        chunk = sentence
                    else:
                        chunk += " " + sentence
                if chunk:
                    texts.append(chunk.strip())
            else:
                texts.append(line)
    return texts

# 设置文档路径
docs_path = "/your/opengauss/docs"

texts = read_markdown_files(docs_path)

rag = RAG(openai_client, db_config, table_name="opengauss_docs_rag")

rag.load(texts)
```

在本例中，我们使用了openGauss文档中`docs\content\docs-lite\zh\docs\GettingStarted`文件夹下的所有markdown文件作为知识库，嵌入结果如下：

```
正在创建向量嵌入: 100%|█████████████████████████████████████████████████████████████████████████████████████| 117/117 [00:57<00:00,  2.04it/s]
```

接下来可以定义一组基于openGauss文档的测试问题，然后使用`answer` 方法获取答案和检索到的上下文文本。

```python
# 测试问题列表
test_questions = [
    "openGauss数据库是什么？",
    "openGauss数据库具备哪些主要特性？",
]
```

```
问题：openGauss数据库是什么？
回答：openGauss是一款全面友好开放的企业级开源关系型数据库管理系统。它由华为结合多年数据库研发经验打造，具备高性能、高可用、高安全、易运维和全开
放的特点。openGauss支持标准的SQL语言规范，并采用木兰宽松许可证V2，允许用户免费使用、修改和分发源代码。

问题：openGauss数据库具备哪些主要特性？
回答：openGauss数据库具备以下主要特性：

1. **高性能**：提供面向多核架构的并发控制技术，结合鲲鹏硬件优化方案，在两路鲲鹏下TPCC Benchmark可达150万tpmc；采用Numa-Aware数据结构、Sql-bypass智能快速引擎技术以及Ustore存储引擎，适用于数据频繁更新场景。

2. **高可用**：支持主备同步、异步及级联备机多种部署模式；具备数据页CRC校验和自动修复功能；备机并行恢复可在10秒内升主提供服务；基于Paxos分布式一
致性协议实现日志复制及选主框架。

3. **高安全**：支持访问控制、加密认证、数据库审计、动态数据脱敏等安全特性，提供全方位端到端的数据安全保护。

4. **易运维**：基于AI的智能参数调优和索引推荐，提供自动参数推荐；支持慢SQL诊断和多维性能自监控视图；提供在线自学习的SQL时间预测功能。

5. **全开放**：采用木兰宽松许可证协议，允许自由修改、使用和引用代码；数据库内核能力全面开放；提供丰富的伙伴认证、培训体系和高校课程。
```

## 4. 使用 Arize Phoenix 进行评估

我们使用 Arize Phoenix 评估框架对基于 openGauss DataVec 的检索增强生成（RAG）管道进行性能评估，重点关注以下两个关键指标：

- **幻觉评估**：判断生成内容是否基于检索到的上下文，确保回答不包含未被支持的信息（幻觉），从而保证数据完整性和可靠性，并提供为什么回答被判断为“事实”或“幻觉”的详细说明。
- **问答评估**：评估 RAG 管道对输入问题的回答是否准确，与预期答案（ground truth）一致，并说明回答被判断为“正确”或“错误”的原因。

在本例中，我们采用DeepSeek作为评估模型。

```python
import phoenix as px
from phoenix.otel import register
from openinference.instrumentation.openai import OpenAIInstrumentor

# 启动 Phoenix 服务器
session = px.launch_app()

tracer_provider = register()
OpenAIInstrumentor().instrument(tracer_provider=tracer_provider)
```

服务器正常启动后应显示：

```
🌍 To view the Phoenix app in your browser, visit http://localhost:6006/
📖 For more information on how to use Phoenix, check out https://arize.com/docs/phoenix
```

可访问http://localhost:6006/查看 Phoenix 仪表盘。

![](figures/arize_phoenix/%E4%BB%AA%E8%A1%A8%E7%9B%98.png)

```python
from phoenix.evals import HallucinationEvaluator, OpenAIModel, QAEvaluator, run_evals
from RAG import RAG, openai_client, db_config
from config import DEEPSEEK_API_KEY

eval_model = OpenAIModel(
    model="deepseek-chat",
    api_key=DEEPSEEK_API_KEY,
    base_url="https://api.deepseek.com",
)

hallucination_evaluator = HallucinationEvaluator(eval_model)
qa_evaluator = QAEvaluator(eval_model)

# 创建 RAG 实例
rag = RAG(openai_client, db_config)
```

```python
texts = [
    "openGauss 是华为开源的企业级关系型数据库。openGauss 内核早期基于 PostgreSQL 开发，融合了华为在数据库领域多年的经验，在架构、事务、存储引擎、优化器及 ARM 架构上进行了适配与优化。作为一个开源数据库，openGauss 希望与广泛的开发者一起构建一个能够融合多元化技术架构的企业级开源数据库社区。",
	...
]

# 加载数据到 RAG
rag.load(texts)

questions = [
    "openGauss 数据库是什么？",
    "如何安装 openGauss？",
    ...
]


results = []
for question in questions:
    answer, retrieved_texts = rag.answer(question, return_retrieved_text=True)
    results.append(
        {
            "question": question,
            "answer": answer,
            "contexts": retrieved_texts,
            "reference": "\n".join(retrieved_texts),  
        }
    )

# 创建 DataFrame
df = pd.DataFrame(results)
df["context"] = df["contexts"]
df["reference"] = df["contexts"]
df.rename(columns={"question": "input", "answer": "output"}, inplace=True)


assert all(
    column in df.columns for column in ["output", "input", "context", "reference"]
)

# 运行评估
hallucination_eval_df, qa_eval_df = run_evals(
    dataframe=df,
    evaluators=[hallucination_evaluator, qa_evaluator],
    provide_explanation=True,
)

results_df = df.copy()
results_df["hallucination_eval"] = hallucination_eval_df["label"]
results_df["hallucination_explanation"] = hallucination_eval_df["explanation"]
results_df["qa_eval"] = qa_eval_df["label"]
results_df["qa_explanation"] = qa_eval_df["explanation"]


print(results_df.head())
results_df.to_csv("evaluation_results.csv", index=False)
```

> 程序运行结束后若出现报错` PermissionError: [WinError 32] 另一个程序正在使用此文件，进程无法访问。`或`NotADirectoryError: [WinError 267] 目录名称无效。: 'C:\\Users\\Username\\AppData\\Local\\Temp\\tmpqcfbqxhf\\phoenix.db' `，对结果不影响，可以忽略。

最终得到的csv结果文件示例如下：

![](figures/arize_phoenix/%E8%A1%A8%E6%A0%BC%E7%BB%93%E6%9E%9C.png)

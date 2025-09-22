# openGauss 向量数据库与 DeepEval 的 RAG 评估最佳实践

在当今数据驱动的时代，高效、可靠的数据库系统和先进的评估框架成为企业优化 RAG 管道的关键。将 openGauss 与 DeepEval 集成，不仅可以充分发挥 openGauss 的高效向量存储与检索优势，还能借助 DeepEval 的评估指标量化 RAG 管道的性能，从而构建更可靠、智能化的数据应用系统。

本文旨在提供一份从部署到集成评估的完整指南。通过本文的指导，**读者将能够掌握 openGauss 在容器中的部署方法、RAG 管道的构建，以及使用 DeepEval 进行全面评估的最佳实践**，最终实现高性能的 RAG 系统优化。

## 1. 环境准备

+ 已部署 7.0.0-RC1 及以上版本的openGauss实例，容器部署参考[容器镜像安装](https://docs.opengauss.org/zh/docs/latest-lite/docs/InstallationGuide/容器镜像安装.html)

+ 已安装 3.10 及以上版本的Python环境

+ 已安装涉及的Python库

  ```shell
  pip3 install pandas tqdm psycopg2 requests openai deepeval langchain_openai
  ```


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

## 4. 检索器评估

在进行评估前，首先要准备一些问题及其真实答案

在评估 LLM 系统中的检索器时，以下几个关键点尤为重要：

- **排名相关性**：检索器是否能有效优先排序相关信息，过滤掉无关数据。
- **上下文检索能力**：检索器根据输入查询捕获和提取上下文相关信息的能力。
- **平衡性**：检索器如何在文本块大小和检索范围之间取得平衡，以减少无关内容的干扰。

这些因素共同决定了检索器在优先级排序、上下文捕获及信息呈现方面的整体表现。如下的代码即用于评估 LLM 系统的上述三个性能。

```python
from deepeval.metrics import (
    ContextualPrecisionMetric,
    ContextualRecallMetric,
    ContextualRelevancyMetric,
)
from deepeval.test_case import LLMTestCase
from deepeval import evaluate


def evaluate_retriever(df):
    """评估检索器性能"""

    # 创建检索器评估指标
    contextual_precision = ContextualPrecisionMetric()
    contextual_recall = ContextualRecallMetric()
    contextual_relevancy = ContextualRelevancyMetric()

    # 创建测试用例
    test_cases = []
    for index, row in df.iterrows():
        test_case = LLMTestCase(
            input=row["question"],
            actual_output=row["answer"],
            expected_output=row["ground_truth"],
            retrieval_context=row["contexts"],
        )
        test_cases.append(test_case)

    # 运行评估
    try:
        result = evaluate(
            test_cases=test_cases,
            metrics=[contextual_precision, contextual_recall, contextual_relevancy],
        )
        return result

    except Exception as e:
        print(f"检索器评估失败：{e}")
        return None
```

> 由于Deepeval默认使用Chatgpt模型，更换其他模型API或本地模型可参考[更换LLM模型](https://deepeval.com/guides/guides-using-custom-llms)。在本例中提供一种将DeepSeek作为Deepeval评估模型的方法，更适合国内用户使用。

```python
from deepeval.metrics import (
    ContextualPrecisionMetric,
    ContextualRecallMetric,
    ContextualRelevancyMetric,
)
from deepeval.test_case import LLMTestCase
from deepeval import evaluate
from langchain_openai import ChatOpenAI
from deepeval.models.base_model import DeepEvalBaseLLM


class DeepSeekLLM(DeepEvalBaseLLM):
    def __init__(self, model_name="deepseek-chat", api_key="your_deepseek_api_key"):
        self.model = ChatOpenAI(
            model=model_name,
            openai_api_key=api_key,
            base_url="https://api.deepseek.com/v1",  # DeepSeek API 端点
        )

    def load_model(self):
        return self.model

    def generate(self, prompt: str) -> str:
        chat_model = self.load_model()
        return chat_model.invoke(prompt).content

    async def a_generate(self, prompt: str) -> str:
        chat_model = self.load_model()
        res = await chat_model.ainvoke(prompt)
        return res.content

    def get_model_name(self):
        return "DeepSeek Model"


def evaluate_retriever_ds(df):
    """评估检索器性能"""

    custom_llm = DeepSeekLLM(api_key="your_deepseek_api_key")

    # 创建检索器评估指标
    contextual_precision = ContextualPrecisionMetric(model=custom_llm)
    contextual_recall = ContextualRecallMetric(model=custom_llm)
    contextual_relevancy = ContextualRelevancyMetric(model=custom_llm)

    # 创建测试用例
    test_cases = []
    for index, row in df.iterrows():
        test_case = LLMTestCase(
            input=row["question"],
            actual_output=row["answer"],
            expected_output=row["ground_truth"],
            retrieval_context=row["contexts"],
        )
        test_cases.append(test_case)

    # 运行评估
    try:
        result = evaluate(
            test_cases=test_cases,
            metrics=[contextual_precision, contextual_recall, contextual_relevancy],
        )
        return result

    except Exception as e:
        print(f"检索器评估失败：{e}")
        return None

```
检索器的评估流程如下图所示。

![](figures/deepeval/%E6%A3%80%E7%B4%A2.png)

![](figures/deepeval/%E6%A3%80%E7%B4%A2%E7%BB%93%E6%9E%9C.png)

## 5. 生成器评估

在评估 LLM 生成输出的质量时，需重点关注以下两个方面：

- **相关性**：评估输入提示是否能有效引导模型生成与上下文契合且有帮助的回答。
- **忠实性**：衡量生成内容的准确性，确保输出信息与事实一致，避免出现幻觉或矛盾，并与检索上下文中的事实信息保持一致。

这两个方面共同确保了生成内容的可靠性和相关性。

```python
from deepeval.metrics import (
    AnswerRelevancyMetric,
    FaithfulnessMetric,
)
from deepeval.test_case import LLMTestCase
from deepeval import evaluate


def evaluate_generation(df):
    """评估生成质量"""

    # 创建生成评估指标
    answer_relevancy = AnswerRelevancyMetric()
    faithfulness = FaithfulnessMetric()

    # 创建测试用例
    test_cases = []
    for index, row in df.iterrows():
        test_case = LLMTestCase(
            input=row["question"],
            actual_output=row["answer"],
            expected_output=row["ground_truth"],
            retrieval_context=row["contexts"],
        )
        test_cases.append(test_case)
    # 运行评估
    try:
        result = evaluate(
            test_cases=test_cases,
            metrics=[answer_relevancy, faithfulness],
        )
        return result

    except Exception as e:
        print(f"生成质量评估失败：{e}")
        return None

```

> 由于Deepeval默认使用Chatgpt模型，更换其他模型API或本地模型可参考[更换LLM模型](https://deepeval.com/guides/guides-using-custom-llms)。在本例中提供一种将DeepSeek作为Deepeval评估模型的方法，更适合国内用户使用。

```python
from deepeval.metrics import (
    AnswerRelevancyMetric,
    FaithfulnessMetric,
)
from deepeval.test_case import LLMTestCase
from deepeval import evaluate
from langchain_openai import ChatOpenAI
from deepeval.models.base_model import DeepEvalBaseLLM


class DeepSeekLLM(DeepEvalBaseLLM):
    def __init__(self, model_name="deepseek-chat", api_key="your_deepseek_api_key"):
        self.model = ChatOpenAI(
            model=model_name,
            openai_api_key=api_key,
            base_url="https://api.deepseek.com/v1",  
        )

    def load_model(self):
        return self.model

    def generate(self, prompt: str) -> str:
        chat_model = self.load_model()
        return chat_model.invoke(prompt).content

    async def a_generate(self, prompt: str) -> str:
        chat_model = self.load_model()
        res = await chat_model.ainvoke(prompt)
        return res.content

    def get_model_name(self):
        return "DeepSeek Model"

def evaluate_generation_ds(df):

    custom_llm = DeepSeekLLM(api_key="your_deepseek_api_key")
    answer_relevancy = AnswerRelevancyMetric(model=custom_llm)
    faithfulness = FaithfulnessMetric(model=custom_llm)

    test_cases = []
    for index, row in df.iterrows():
        test_case = LLMTestCase(
            input=row["question"],
            actual_output=row["answer"],
            expected_output=row["ground_truth"],
            retrieval_context=row["contexts"],
        )
        test_cases.append(test_case)

    try:
        result = evaluate(
            test_cases=test_cases,
            metrics=[answer_relevancy, faithfulness],
        )
        return result
    except Exception as e:
        print(f"生成质量评估失败：{e}")
        return None![](figures/deepeval/%E7%94%9F%E6%88%90.png)
```
生成器的评估流程如下图所示。

![](figures/deepeval/%E7%94%9F%E6%88%90.png)

![](figures/deepeval/%E7%94%9F%E6%88%90%E7%BB%93%E6%9E%9C.png)

import phoenix as px
from phoenix.otel import register
from openinference.instrumentation.openai import OpenAIInstrumentor
from phoenix.evals import HallucinationEvaluator, OpenAIModel, QAEvaluator, run_evals
import pandas as pd
from RAG import RAG, openai_client, db_config
from config import DEEPSEEK_API_KEY
import csv
import time

# 启动 Phoenix 服务器
session = px.launch_app()

tracer_provider = register()
OpenAIInstrumentor().instrument(tracer_provider=tracer_provider)

eval_model = OpenAIModel(
    model="deepseek-chat",
    api_key=DEEPSEEK_API_KEY,
    base_url="https://api.deepseek.com",
)

hallucination_evaluator = HallucinationEvaluator(eval_model)
qa_evaluator = QAEvaluator(eval_model)

# 创建 RAG 实例
rag = RAG(openai_client, db_config)

texts = [
    "openGauss 是华为开源的企业级关系型数据库。openGauss 内核早期基于 PostgreSQL 开发，融合了华为在数据库领域多年的经验，在架构、事务、存储引擎、优化器及 ARM 架构上进行了适配与优化。作为一个开源数据库，openGauss 希望与广泛的开发者一起构建一个能够融合多元化技术架构的企业级开源数据库社区。",
    "openGauss 安装前需要检查硬件和软件要求。硬件要求包括：至少2GB内存，推荐8GB以上；至少10GB磁盘空间用于安装。软件要求包括：支持x86_64或ARM64架构的Linux系统。安装步骤：下载安装包，解压到指定目录，配置环境变量，执行安装脚本。",
    "openGauss 支持丰富的数据类型：数值类型包括 TINYINT、SMALLINT、INTEGER、BIGINT、DECIMAL、NUMERIC、REAL、DOUBLE PRECISION；字符类型包括 CHAR、VARCHAR、TEXT、CLOB；日期时间类型包括 DATE、TIME、TIMETZ、TIMESTAMP、TIMESTAMPTZ、INTERVAL；布尔类型 BOOLEAN；二进制类型 BYTEA、BLOB；数组类型；JSON类型；几何类型；网络地址类型；向量类型 VECTOR。",
    "在 openGauss 中创建表的语法：CREATE TABLE [schema.]table_name (column_name data_type [column_constraint], ...)[table_constraint]; 例如：CREATE TABLE employees (id INTEGER PRIMARY KEY, name VARCHAR(100) NOT NULL, email VARCHAR(255) UNIQUE, salary DECIMAL(10,2), hire_date DATE DEFAULT CURRENT_DATE);",
    "openGauss 主要特性：1. 高性能：支持多核并行处理，优化的查询执行引擎；2. 高可用：支持主备复制、读写分离；3. 安全性：提供透明数据加密、访问控制、审计功能；4. 兼容性：兼容 SQL 标准和 PostgreSQL 生态；5. 扩展性：支持分布式架构；6. AI能力：内置向量数据类型，支持AI应用。",
    "openGauss 数据库备份方法：1. 逻辑备份：使用 gs_dump 命令备份单个数据库，gs_dumpall 备份整个实例；2. 物理备份：使用 gs_basebackup 进行在线物理备份；3. 增量备份：基于WAL日志的增量备份；4. 备份策略：可配置自动备份、定期备份、压缩备份等选项。",
    "openGauss 支持向量数据类型 VECTOR，可以存储固定长度的浮点数向量。向量类型支持欧几里德距离、余弦距离等相似度计算。可以创建向量索引（如IVF、HNSW）来加速向量检索。语法：VECTOR(dimension)，例如 VECTOR(128) 表示128维向量。向量类型适用于AI、机器学习、图像识别等应用场景。",
    "openGauss 连接参数配置：主要参数包括 host（主机地址）、port（端口号，默认5432）、database（数据库名）、user（用户名）、password（密码）。可以通过连接字符串配置：postgresql://user:password@host:port/database。也可以通过环境变量或配置文件设置连接参数。连接池参数包括 max_connections、connection_timeout 等。",
]

# 加载数据到 RAG
rag.load(texts)

questions = [
    "openGauss 数据库是什么？",
    "如何安装 openGauss？",
    "openGauss 支持哪些数据类型？",
    "如何在 openGauss 中创建表？",
    "openGauss 的主要特性有哪些？",
    "如何进行 openGauss 数据库的备份？",
    "openGauss 支持向量数据类型吗？",
    "如何配置 openGauss 的连接参数？",
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


def clean_text_for_csv(text):
    """清理文本内容，将换行符替换为空格，避免CSV格式问题"""
    if isinstance(text, str):
        return text.replace("\n", " ").replace("\r", " ")
    return text

text_columns = [
    "input",
    "output",
    "contexts",
    "reference",
    "context",
    "hallucination_explanation",
    "qa_explanation",
]
for col in text_columns:
    if col in results_df.columns:
        results_df[col] = results_df[col].apply(clean_text_for_csv)

results_df.to_csv(
    "./evaluation_results.csv", index=False, quoting=csv.QUOTE_ALL, escapechar="\\"
)

from RAG import RAG, openai_client, db_config
from datasets import Dataset
import pandas as pd
from tqdm import tqdm
from deepeval.metrics import (
    ContextualPrecisionMetric,
    ContextualRecallMetric,
    ContextualRelevancyMetric,
    AnswerRelevancyMetric,
    FaithfulnessMetric,
)
from deepeval.test_case import LLMTestCase
from deepeval import evaluate
import os
from evaluate_retriever import evaluate_retriever
from evaluate_generation import evaluate_generation
from evaluate_generation_ds import evaluate_generation_ds
from evaluate_retriever_ds import evaluate_retriever_ds


# 针对 openGauss 的测试问题和标准答案
question_list = [
    "openGauss 数据库是什么？",
    "如何安装 openGauss？",
    "openGauss 支持哪些数据类型？",
    "如何在 openGauss 中创建表？",
    "openGauss 的主要特性有哪些？",
    "如何进行 openGauss 数据库的备份？",
    "openGauss 支持向量数据类型吗？",
    "如何配置 openGauss 的连接参数？",
]

ground_truth_list = [
    "openGauss 是一款开源关系型数据库管理系统，具有高性能、高可靠性和高安全性的特点。它支持SQL标准，提供了丰富的数据类型和功能，适用于企业级应用场景。",
    "安装 openGauss 需要先下载安装包，配置系统环境，创建用户和用户组，然后运行安装脚本进行安装和初始化。具体步骤包括检查系统要求、设置环境变量、执行安装命令等。",
    "openGauss 支持多种数据类型，包括数值类型（INTEGER、BIGINT、DECIMAL等）、字符类型（CHAR、VARCHAR、TEXT等）、日期时间类型（DATE、TIME、TIMESTAMP等）、布尔类型、数组类型，以及扩展的向量类型等。",
    "在 openGauss 中创建表使用 CREATE TABLE 语句，需要指定表名、列名、数据类型和约束条件。例如：CREATE TABLE table_name (column1 datatype, column2 datatype, ...);",
    "openGauss 的主要特性包括：高性能并发处理、ACID事务支持、多种索引类型、SQL标准兼容、高可用性、安全机制、扩展性、向量数据支持等。",
    "openGauss 数据库备份可以使用 gs_dump 工具进行逻辑备份，或使用 gs_basebackup 进行物理备份。还可以配置自动备份策略和增量备份。",
    "是的，openGauss 支持向量数据类型，可以存储高维向量数据，支持向量相似度搜索，适用于AI和机器学习应用场景。",
    "openGauss 的连接参数包括主机地址、端口号、数据库名、用户名、密码等，可以通过连接字符串或配置文件进行设置。常用端口为5432。",
]


def load_sample_data():
    """加载示例数据到RAG系统"""
    sample_texts = [
        "openGauss 是华为开源的企业级关系型数据库。openGauss 内核早期基于 PostgreSQL 开发，融合了华为在数据库领域多年的经验，在架构、事务、存储引擎、优化器及 ARM 架构上进行了适配与优化。作为一个开源数据库，openGauss 希望与广泛的开发者一起构建一个能够融合多元化技术架构的企业级开源数据库社区。",
        "openGauss 安装前需要检查硬件和软件要求。硬件要求包括：至少2GB内存，推荐8GB以上；至少10GB磁盘空间用于安装。软件要求包括：支持x86_64或ARM64架构的Linux系统。安装步骤：下载安装包，解压到指定目录，配置环境变量，执行安装脚本。",
        "openGauss 支持丰富的数据类型：数值类型包括 TINYINT、SMALLINT、INTEGER、BIGINT、DECIMAL、NUMERIC、REAL、DOUBLE PRECISION；字符类型包括 CHAR、VARCHAR、TEXT、CLOB；日期时间类型包括 DATE、TIME、TIMETZ、TIMESTAMP、TIMESTAMPTZ、INTERVAL；布尔类型 BOOLEAN；二进制类型 BYTEA、BLOB；数组类型；JSON类型；几何类型；网络地址类型；向量类型 VECTOR。",
        "在 openGauss 中创建表的语法：CREATE TABLE [schema.]table_name (column_name data_type [column_constraint], ...)[table_constraint]; 例如：CREATE TABLE employees (id INTEGER PRIMARY KEY, name VARCHAR(100) NOT NULL, email VARCHAR(255) UNIQUE, salary DECIMAL(10,2), hire_date DATE DEFAULT CURRENT_DATE);",
        "openGauss 主要特性：1. 高性能：支持多核并行处理，优化的查询执行引擎；2. 高可用：支持主备复制、读写分离；3. 安全性：提供透明数据加密、访问控制、审计功能；4. 兼容性：兼容 SQL 标准和 PostgreSQL 生态；5. 扩展性：支持分布式架构；6. AI能力：内置向量数据类型，支持AI应用。",
        "openGauss 数据库备份方法：1. 逻辑备份：使用 gs_dump 命令备份单个数据库，gs_dumpall 备份整个实例；2. 物理备份：使用 gs_basebackup 进行在线物理备份；3. 增量备份：基于WAL日志的增量备份；4. 备份策略：可配置自动备份、定期备份、压缩备份等选项。",
        "openGauss 支持向量数据类型 VECTOR，可以存储固定长度的浮点数向量。向量类型支持欧几里德距离、余弦距离等相似度计算。可以创建向量索引（如IVF、HNSW）来加速向量检索。语法：VECTOR(dimension)，例如 VECTOR(128) 表示128维向量。向量类型适用于AI、机器学习、图像识别等应用场景。",
        "openGauss 连接参数配置：主要参数包括 host（主机地址）、port（端口号，默认5432）、database（数据库名）、user（用户名）、password（密码）。可以通过连接字符串配置：postgresql://user:password@host:port/database。也可以通过环境变量或配置文件设置连接参数。连接池参数包括 max_connections、connection_timeout 等。",
    ]
    return sample_texts


if __name__ == "__main__":
    # 创建 RAG 实例并加载示例数据
    rag = RAG(openai_client, db_config)
    sample_texts = load_sample_data()
    rag.load(sample_texts)

    # 初始化列表
    contexts_list = []
    answer_list = []

    # 生成答案和上下文
    for question in tqdm(question_list, desc="Answering questions"):
        answer, contexts = rag.answer(question, return_retrieved_text=True)
        contexts_list.append(contexts)
        answer_list.append(answer)

    # 创建 DataFrame
    df = pd.DataFrame(
        {
            "question": question_list,
            "contexts": contexts_list,
            "answer": answer_list,
            "ground_truth": ground_truth_list,
        }
    )
    rag_results = Dataset.from_pandas(df)

    # 评估检索器
    # retriever_result = evaluate_retriever(df)
    retriever_result = evaluate_retriever_ds(df)

    # 评估生成质量
    # generation_result = evaluate_generation(df)
    generation_result = evaluate_generation_ds(df)

    print("Evaluation completed!")

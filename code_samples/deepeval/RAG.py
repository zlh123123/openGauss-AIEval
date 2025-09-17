from typing import List
from tqdm import tqdm
from openai import OpenAI
import psycopg2
from psycopg2 import sql
import os
import requests



openai_client = OpenAI(
    api_key=os.environ["OPENAI_API_KEY"], base_url=os.environ["OPENAI_BASE_URL"]
)


db_config = {
    "host": "localhost",
    "port": 8888,
    "database": "postgres",
    "user": "test",
    "password": "Enigma#2025",
}


class RAG:
    """
    基于 DeepSeek、BAAI/bge-large-zh-v1.5 和 openGauss 构建的 RAG（检索增强生成）类。
    """

    def __init__(self, openai_client: OpenAI, db_config: dict):
        self._prepare_openai(openai_client)
        self._prepare_opengauss(db_config)

    def _emb_text(self, text: str) -> List[float]:
        """使用 BAAI/bge-large-zh-v1.5 模型通过 API 生成文本的向量嵌入"""
        try:
            payload = {"model": "BAAI/bge-large-zh-v1.5", "input": text}
            headers = {
                "Authorization": f"Bearer {SILICONFLOW_API_KEY}",
                "Content-Type": "application/json",
            }

            response = requests.post(
                SILICONFLOW_BASE_URL, json=payload, headers=headers
            )
            response.raise_for_status()  # 检查HTTP错误

            result = response.json()
            return result["data"][0]["embedding"]
        except Exception as e:
            print(f"生成嵌入向量时出错：{e}")
            raise e

    def _prepare_openai(
        self,
        openai_client: OpenAI,
        embedding_model: str = "BAAI/bge-large-zh-v1.5",
        llm_model: str = "deepseek-chat",
    ):
        """初始化 OpenAI 客户端和模型配置"""
        self.openai_client = openai_client
        self.embedding_model = embedding_model
        self.llm_model = llm_model
        self.SYSTEM_PROMPT = """
            你是一个AI助手。你能够从提供的上下文段落片段中找到问题的答案。
        """
        self.USER_PROMPT = """
            使用以下用<context>标签包围的信息片段来回答用<question>标签包围的问题。
            <context>
            {context}
            </context>
            <question>
            {question}
            </question>
        """

    def _check_vector_support(self) -> bool:
        """检查数据库是否支持向量类型"""
        try:
            # 尝试创建一个临时的向量类型测试
            self.cursor.execute("SELECT typname FROM pg_type WHERE typname = 'vector';")
            result = self.cursor.fetchone()
            if result:
                print("检测到数据库支持 vector 类型")
                return True
            else:
                print("数据库不支持 vector 类型，将使用数组类型")
                return False
        except psycopg2.Error as e:
            print(f"检查向量支持时出错：{e}")
            self.conn.rollback()
            return False

    def _prepare_opengauss(self, db_config: dict, table_name: str = "rag_table"):
        """初始化 openGauss 数据库连接和表结构"""
        self.db_config = db_config
        self.table_name = table_name
        self.conn = psycopg2.connect(**db_config)
        self.cursor = self.conn.cursor()

        # 创建用户自己的 schema（如果不存在）
        try:
            self.cursor.execute("CREATE SCHEMA IF NOT EXISTS test_schema;")
            self.conn.commit()
            self.schema_name = "test_schema"
            print(f"使用 schema: {self.schema_name}")
        except psycopg2.Error:
            # 如果无法创建 schema，使用默认的用户 schema
            self.schema_name = db_config["user"]
            self.conn.rollback()
            print(f"使用默认用户 schema: {self.schema_name}")

        self.vector_supported = self._check_vector_support()

        # 获取嵌入维度并创建表和索引
        print("正在检测向量维度...")
        self.embedding_dim = len(self._emb_text("示例"))
        print(f"向量维度：{self.embedding_dim}")

        self.create_table(self.conn, self.cursor, self.table_name, self.embedding_dim)
        self.create_index(
            self.conn, self.cursor, self.table_name, f"idx_{self.table_name}"
        )

    def create_table(self, conn, cursor, table_name: str, dim: int):
        """创建存储文本和向量嵌入的表"""
        cursor.execute(
            sql.SQL(
                "CREATE TABLE IF NOT EXISTS {schema}.{table_name} (id BIGINT PRIMARY KEY, text TEXT, embedding vector({dim}));"
            ).format(
                schema=sql.Identifier(self.schema_name),
                table_name=sql.Identifier(table_name),
                dim=sql.Literal(dim),
            )
        )
        conn.commit()
        print(f"表 {self.schema_name}.{table_name} 创建成功！")

    def create_index(self, conn, cursor, table_name: str, index_name: str):
        """为向量字段创建 HNSW 索引以提高检索性能"""
        cursor.execute(
            sql.SQL(
                "CREATE INDEX IF NOT EXISTS {index_name} ON {schema}.{table_name} USING hnsw (embedding vector_l2_ops);"
            ).format(
                index_name=sql.Identifier(index_name),
                schema=sql.Identifier(self.schema_name),
                table_name=sql.Identifier(table_name),
            )
        )
        conn.commit()
        print(f"索引 {index_name} 创建成功！")

    def load(self, texts: List[str]):
        """
        将文本数据加载到 openGauss 数据库中。
        """
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
        ids = list(range(len(texts)))

        # 生成嵌入向量
        embeddings = []
        for text in tqdm(texts, desc="正在创建向量嵌入"):
            emb = self._emb_text(text)
            embeddings.append(emb)

        # 准备数据并插入
        data = list(zip(ids, texts, embeddings))
        self.cursor.executemany(
            sql.SQL(
                "INSERT INTO {schema}.{table_name} (id, text, embedding) VALUES (%s, %s, %s);"
            ).format(
                schema=sql.Identifier(self.schema_name),
                table_name=sql.Identifier(self.table_name),
            ),
            data,
        )
        self.conn.commit()
        print("数据插入成功！")

    def retrieve(self, question: str, top_k: int = 3) -> List[str]:
        """
        检索与给定问题最相似的文本数据。
        """
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
        """
        基于检索到的知识回答给定的问题。
        """
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
        """
        清理数据库连接。
        """
        try:
            if hasattr(self, "cursor") and self.cursor:
                self.cursor.close()
            if hasattr(self, "conn") and self.conn:
                self.conn.close()
        except Exception:
            pass


# 使用示例
if __name__ == "__main__":
    # 创建 RAG 实例
    rag = RAG(openai_client, db_config)

    # 示例文本数据
    texts = [
        "openGauss 是一款开源数据库",
        "DataVec是一个基于openGauss的向量数据库",
        "nomic-embed-text是一个专门用于文本转化为高维向量表示的高性能嵌入模型",
    ]

    # 加载数据
    rag.load(texts)

    # 测试检索和回答
    question = "openGauss数据库是什么？"
    answer = rag.answer(question)
    print(f"问题：{question}")
    print(f"回答：{answer}")

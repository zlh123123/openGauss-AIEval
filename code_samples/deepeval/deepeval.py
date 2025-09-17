from typing import List
from tqdm import tqdm
from openai import OpenAI
import psycopg2
from psycopg2 import sql
from psycopg2.extras import execute_multi_search, init_conn_pool, close_conn_pool


class RAG:
    def __init__(self, openai_client: OpenAI, db_config: dict):
        self._prepare_openai(openai_client)
        self._prepare_opengauss(db_config)

    def _emb_text(self, text: str) -> List[float]:
        return (
            self.openai_client.embeddings.create(
                input=text, model="text-embedding-3-small"
            )
            .data[0]
            .embedding
        )

    def _prepare_openai(
        self,
        openai_client: OpenAI,
        embedding_model: str = "text-embedding-3-small",
        llm_model: str = "gpt-4o-mini",
    ):
        self.openai_client = openai_client
        self.embedding_model = embedding_model
        self.llm_model = llm_model
        self.SYSTEM_PROMPT = "Human: You are an AI assistant. You are able to find answers to the questions from the contextual passage snippets provided."
        self.USER_PROMPT = "Use the following pieces of information enclosed in <context> tags to provide an answer to the question enclosed in <question> tags.\n<context>\n{context}\n</context>\n<question>\n{question}\n</question>"

    def _prepare_opengauss(self, db_config: dict, table_name: str = "rag_table"):
        self.db_config = db_config
        self.table_name = table_name
        self.embedding_dim = len(self._emb_text("demo"))
        self.conn = psycopg2.connect(**db_config)
        self.cursor = self.conn.cursor()
        self.create_table(self.conn, self.cursor, self.table_name, self.embedding_dim)
        self.create_index(
            self.conn, self.cursor, self.table_name, f"idx_{self.table_name}"
        )

    def create_table(self, conn, cursor, table_name: str, dim: int):
        cursor.execute(
            sql.SQL(
                "CREATE TABLE IF NOT EXISTS public.{table_name} (id BIGINT PRIMARY KEY, text TEXT, embedding vector({dim}));"
            ).format(table_name=sql.Identifier(table_name), dim=sql.Literal(dim))
        )
        conn.commit()

    def create_index(self, conn, cursor, table_name: str, index_name: str):
        cursor.execute(
            sql.SQL(
                "CREATE INDEX IF NOT EXISTS {index_name} ON public.{table_name} USING hnsw (embedding vector_ip_ops);"
            ).format(
                index_name=sql.Identifier(index_name),
                table_name=sql.Identifier(table_name),
            )
        )
        conn.commit()

    def load(self, texts: List[str]):
        ids = list(range(len(texts)))
        embeddings = [
            self._emb_text(text) for text in tqdm(texts, desc="Creating embeddings")
        ]
        data = list(zip(ids, texts, embeddings))
        self.cursor.executemany(
            sql.SQL(
                "INSERT INTO public.{table_name} (id, text, embedding) VALUES (%s, %s, %s);"
            ).format(table_name=sql.Identifier(self.table_name)),
            data,
        )
        self.conn.commit()

    def retrieve(self, question: str, top_k: int = 3) -> List[str]:
        query_emb = self._emb_text(question)
        sql_template = f"SELECT text FROM public.{self.table_name} ORDER BY embedding <#> %s::vector ASC LIMIT %s;"
        argslist = [(query_emb, top_k)]
        scan_params = {"enable_seqscan": "off", "hnsw_ef_search": 40}
        conn_pool_mgr = init_conn_pool(
            self.db_config, max_workers=2, scan_params=scan_params
        )
        results = execute_multi_search(
            self.db_config,
            conn_pool_mgr,
            sql_template,
            argslist,
            scan_params,
            max_workers=2,
        )
        close_conn_pool(conn_pool_mgr)
        return [row[0] for row in results[0]] if results else []

    def answer(
        self,
        question: str,
        retrieval_top_k: int = 3,
        return_retrieved_text: bool = False,
    ):
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
        if hasattr(self, "cursor") and hasattr(self, "conn"):
            self.cursor.close()
            self.conn.close()

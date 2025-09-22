import os
import glob
from typing import List
import re
from RAG import RAG, openai_client, db_config


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


def test_rag_pipeline():
    """
    测试RAG管道
    """

    # 设置文档路径
    docs_path = "/your/opengauss/docs"

    texts = read_markdown_files(docs_path)

    rag = RAG(openai_client, db_config, table_name="opengauss_docs_rag")

    rag.load(texts)

    # 测试问题列表
    test_questions = [
        "openGauss数据库是什么？",
        "openGauss数据库具备哪些主要特性？",
    ]

    for question in test_questions:
        print(f"\n问题：{question}")
        try:
            answer, retrieved_texts = rag.answer(question, return_retrieved_text=True)
            print(f"回答：{answer}")
            print("检索到的上下文：")
            for idx, ctx in enumerate(retrieved_texts, 1):
                print(f"上下文 {idx}：{ctx}\n")
                print("-" * 30)
            print("-" * 30)
        except Exception as e:
            print(f"回答问题时出错：{e}")
            print("-" * 30)


if __name__ == "__main__":
    test_rag_pipeline()

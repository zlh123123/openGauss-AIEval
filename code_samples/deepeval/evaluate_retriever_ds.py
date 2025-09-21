from deepeval.metrics import (
    ContextualPrecisionMetric,
    ContextualRecallMetric,
    ContextualRelevancyMetric,
)
from deepeval.test_case import LLMTestCase
from deepeval import evaluate
from langchain_openai import ChatOpenAI
from deepeval.models.base_model import DeepEvalBaseLLM
from config import DEEPEVAL_API_KEY


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

    custom_llm = DeepSeekLLM(api_key=DEEPEVAL_API_KEY)

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

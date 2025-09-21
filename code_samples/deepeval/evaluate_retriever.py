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

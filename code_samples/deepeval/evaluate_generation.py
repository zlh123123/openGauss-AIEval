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

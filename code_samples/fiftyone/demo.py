
import fiftyone as fo
import fiftyone.brain as fob
import fiftyone.zoo as foz


def main():

    # 加载数据集
    print("Loading dataset...")
    dataset = foz.load_zoo_dataset("quickstart")
    print(f"Loaded {len(dataset)} samples")

    # 创建openGauss相似性索引
    print("Creating similarity index...")
    opengauss_index = fob.compute_similarity(
        dataset,
        brain_key="opengauss_index",
        backend="opengauss",
        embeddings="clip",  
        metric="cosine",  # 距离度量
        table_name="quickstart_vectors",  # openGauss表名
    )
    print("Similarity index created successfully!")

    # 执行相似性搜索
    print("Performing similarity search...")
    query_sample = dataset.first()
    print(f"Query sample ID: {query_sample.id}")
    # print(query_sample)

    similar_view = dataset.sort_by_similarity(
        query_sample.id, brain_key="opengauss_index", k=10
    )

    print(f"Found {len(similar_view)} similar images")

    # 显示前几个相似样本的信息
    for i, sample in enumerate(similar_view.limit(5)):
        print(f"  Similar sample {i+1}: {sample.id}")

    # 可视化结果
    session = fo.launch_app(similar_view)
    session.wait()


if __name__ == "__main__":
    main()

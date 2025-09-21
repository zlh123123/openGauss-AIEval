# DeepEval Code Samples

这是一个用于openGauss AI评估的DeepEval代码示例文件夹。DeepEval是一个用于评估大型语言模型（LLM）的开源库。

## 功能特性
- 提供LLM评估的示例代码。
- 支持多种评估指标，如准确性、相关性等。
- 集成openGauss数据库进行数据存储和查询。

## 文件说明


- **`config.py`** - 配置文件，包含API密钥、数据库连接信息等配置参数
- **`RAG.py`** - RAG系统的核心实现，负责文档检索和生成
- **`test_RAG.py`** - RAG系统的测试文件，包含功能测试用例
- **`evaluate.py`** - 主要评估脚本，统一调用各种评估功能
- **`evaluate_generation.py`** - 文本生成质量评估，评估LLM生成内容的质量（Openai API）
- **`evaluate_generation_ds.py`** - 文本生成质量评估，评估LLM生成内容的质量（DeepSeek API）
- **`evaluate_retriever.py`** - 检索器性能评估，评估信息检索的准确性（Openai API）
- **`evaluate_retriever_ds.py`** - 检索器性能评估，评估信息检索的准确性（DeepSeek API）

## 使用方法
1. 克隆仓库：
   ```
   git clone https://github.com/zlh123123/openGauss-AIEval.git
   cd openGauss-AIEval/code_samples/deepeval
   ```

2. 安装依赖：
   ```
   pip3 install pandas tqdm psycopg2 requests openai deepeval langchain_openai
   ```

3. 配置API keys：在`config.py`中设置你的API keys。



# Arize Phoenix Code Samples

这是一个用于openGauss AI评估的Arize Phoenix代码示例文件夹。Arize Phoenix是一个开源的AI可观测性平台，专门用于LLM应用的监控、评估和调试。

## 功能特性
- 提供基于Phoenix的LLM评估和监控示例代码
- 支持多种评估指标，如幻觉检测、问答质量评估等
- 集成openGauss数据库进行RAG系统的向量存储和检索
- 实时追踪和可视化LLM应用的性能指标
- 支持分布式追踪和性能分析

## 文件说明

- **`config.py`** - 配置文件，包含DeepSeek API密钥、SiliconFlow API密钥、数据库连接信息等配置参数
- **`RAG.py`** - RAG系统的核心实现，基于openGauss向量数据库的检索增强生成系统
- **`phoenix_integration.py`** - Phoenix集成主脚本，包含LLM监控、评估和结果导出功能

## 评估指标
- **幻觉检测 (Hallucination Detection)** - 检测LLM生成内容是否包含虚假或不准确信息
- **问答质量评估 (QA Evaluation)** - 评估问答系统回答的准确性和完整性
- **检索质量评估** - 评估RAG系统检索相关文档的效果

## 使用方法

1. 克隆仓库：
   ```bash
   git clone https://github.com/zlh123123/openGauss-AIEval.git
   cd openGauss-AIEval/code_samples/arize_phoenix
   ```

2. 安装依赖：
   ```bash
    pip install pandas tqdm psycopg2 requests openai arize-phoenix openinference-instrumentation-openai
   ```

3. 配置API keys：在`config.py`中设置你的API keys：
   ```python
   # DeepSeek API key (用于LLM对话)
   DEEPSEEK_API_KEY = "sk-your-deepseek-key"
   
   # SiliconFlow API key (用于文本嵌入)
   SILICONFLOW_API_KEY = "sk-your-siliconflow-key"
   
   # openGauss数据库配置
   DB_CONFIG = {
       "host": "localhost",
       "port": 8888,
       "database": "YourDbName",
       "user": "YourUserName", 
       "password": "YourUserPassword",
   }
   ```

4. 运行评估：
   ```bash
   python phoenix_integration.py
   ```

5. 查看结果：
   - Phoenix监控界面会自动启动，可在浏览器中查看实时监控数据
   - 评估结果会保存到 `evaluation_results.csv` 文件中

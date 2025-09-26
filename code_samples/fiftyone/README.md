# FiftyOne Code Samples

这是一个用于openGauss AI评估的FiftyOne代码示例文件夹。FiftyOne是一个开源的计算机视觉数据集管理工具，支持数据集可视化、相似性搜索和模型评估等功能。

## 功能特性
- 提供基于FiftyOne的数据集管理和相似性搜索示例代码
- 集成openGauss向量数据库作为相似性搜索后端
- 支持多种距离度量方式，如余弦相似度、欧几里得距离等
- 提供向量索引和高效的近邻搜索功能
- 支持数据集可视化和交互式探索

## 文件说明

- **`config.py`** - 配置文件，包含openGauss数据库连接信息等配置参数
- **`opengauss_backend.py`** - openGauss相似性后端实现，包含配置类、索引类和相似性搜索逻辑
- **`demo.py`** - 演示脚本，展示如何使用FiftyOne加载数据集、创建相似性索引并执行相似性搜索

## 使用方法

1. 克隆仓库：
   ```bash
   git clone https://github.com/zlh123123/openGauss-AIEval.git
   cd openGauss-AIEval/code_samples/fiftyone
   ```

2. 安装依赖：
   ```bash
   pip install tqdm psycopg2 fiftyone torch torchvision clip
   ```

3. 配置数据库：在`config.py`中设置你的openGauss数据库连接信息：
   ```python
   DB_CONFIG = {
       "host": "localhost",
       "port": 8888,
       "database": "YourDbName",
       "user": "YourUserName",
       "password": "YourUserPassword",
   }
   ```

4. 运行演示：
   ```bash
   python demo.py
   ```

5. 查看结果：
   - FiftyOne应用会自动启动，可在浏览器中查看数据集和相似性搜索结果
   - 控制台会输出相似样本的信息

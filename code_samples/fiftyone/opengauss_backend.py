import logging
import numpy as np
import psycopg2
import psycopg2.extras as psy_extras
from psycopg2 import sql

import fiftyone.core.utils as fou
import fiftyone.brain.internal.core.utils as fbu
from fiftyone.brain.similarity import (
    SimilarityConfig,
    Similarity,
    SimilarityIndex,
)
import ast

from config import DB_CONFIG

logger = logging.getLogger(__name__)


def _is_str(value):
    """检查值是否为字符串"""
    return isinstance(value, str)


# openGauss支持的向量距离度量
_SUPPORTED_METRICS = {
    "cosine": "<=>",
    "euclidean": "<->",
    "inner_product": "<#>",  # 内积
}

# openGauss向量索引操作符
_INDEX_OPERATORS = {
    "cosine": "vector_cosine_ops",
    "euclidean": "vector_l2_ops",
    "inner_product": "vector_ip_ops",
}


class OpenGaussSimilarityConfig(SimilarityConfig):
    """openGauss相似性配置类"""

    def __init__(
        self,
        host=None,
        port=None,
        database=None,
        user=None,
        password=None,
        table_name=None,
        index_name=None,
        metric="cosine",
        schema=None,
        **kwargs,
    ):
        """
        Args:
            host: 数据库主机地址
            port: 数据库端口
            database: 数据库名称
            user: 用户名
            password: 密码
            table_name: 表名，如果为None会自动生成
            index_name: 索引名，如果为None会自动生成
            metric: 距离度量方式，支持 "cosine", "euclidean", "inner_product"
            schema: 数据库模式，如果为None使用用户名作为模式
        """
        if metric not in _SUPPORTED_METRICS:
            raise ValueError(
                f"Unsupported metric '{metric}'. "
                f"Supported values are {tuple(_SUPPORTED_METRICS.keys())}"
            )

        super().__init__(**kwargs)

        # 使用配置文件中的默认值
        self.host = host or DB_CONFIG["host"]
        self.port = port or DB_CONFIG["port"]
        self.database = database or DB_CONFIG["database"]
        self.user = user or DB_CONFIG["user"]
        self.password = password or DB_CONFIG["password"]

        # 使用用户名作为默认模式
        self.schema = schema or self.user

        self.table_name = table_name
        self.index_name = index_name
        self.metric = metric

    @property
    def method(self):
        """返回后端标识符"""
        return "opengauss"

    @property
    def connection_string(self):
        """构建连接字符串"""
        return f"host={self.host} port={self.port} dbname={self.database} user={self.user} password={self.password}"

    @property
    def max_k(self):
        """最大k值限制"""
        return 10000

    @property
    def supports_least_similarity(self):
        """是否支持最不相似查询"""
        return True

    @property
    def supported_aggregations(self):
        """支持的聚合方法"""
        return ("mean",)


class OpenGaussSimilarity(Similarity):
    """openGauss相似性工厂类"""

    def ensure_requirements(self):
        """检查必需的包"""
        fou.ensure_package("psycopg2|psycopg2-binary")

    def ensure_usage_requirements(self):
        """运行时依赖检查"""
        fou.ensure_package("psycopg2|psycopg2-binary")

    def initialize(self, samples, brain_key):
        """初始化索引实例"""
        return OpenGaussSimilarityIndex(samples, self.config, brain_key, backend=self)


class OpenGaussSimilarityIndex(SimilarityIndex):
    """openGauss相似性索引类"""

    def __init__(self, samples, config, brain_key, backend=None):
        super().__init__(samples, config, brain_key, backend=backend)

        self._dataset = samples._dataset
        self._conn = None
        self._cur = None
        self._initialize()

    @property
    def is_external(self):
        """是否为外部索引"""
        return True

    @property
    def total_index_size(self):
        """获取索引大小"""
        if self._conn is None or self._conn.closed:
            self._initialize()

        try:
            self._cur.execute(
                sql.SQL("SELECT COUNT(*) FROM {schema}.{table_name}").format(
                    schema=sql.Identifier(self.config.schema),
                    table_name=sql.Identifier(self.config.table_name),
                )
            )
            return self._cur.fetchone()[0]
        except psycopg2.Error:
            return 0

    def _initialize(self):
        """初始化数据库连接和表结构"""
        try:
            logger.info(
                f"Connecting to openGauss database at {self.config.host}:{self.config.port}"
            )
            self._conn = psycopg2.connect(self.config.connection_string)
            self._cur = self._conn.cursor()

            # 创建用户模式（如果不存在）
            try:
                self._cur.execute(
                    sql.SQL("CREATE SCHEMA IF NOT EXISTS {schema}").format(
                        schema=sql.Identifier(self.config.schema)
                    )
                )
                self._conn.commit()
                logger.info(f"Created/ensured schema {self.config.schema}")
            except psycopg2.Error as e:
                logger.warning(
                    f"Could not create user schema {self.config.schema}: {e}"
                )

            # 生成表名和索引名
            if self.config.table_name is None:
                existing_tables = self._get_table_names()
                self.config.table_name = fbu.get_unique_name(
                    f"fiftyone_vectors_{self.brain_key[:8]}", existing_tables
                )
                self.save_config()

            if self.config.index_name is None:
                self.config.index_name = f"{self.config.table_name}_vector_idx"
                self.save_config()

        except psycopg2.Error as e:
            logger.error(f"Failed to connect to openGauss: {e}")
            raise

    def _get_table_names(self):
        """获取用户模式中的所有表名"""
        try:
            # 首先尝试查找用户模式中的表
            self._cur.execute(
                sql.SQL(
                    "SELECT table_name FROM information_schema.tables WHERE table_schema = {schema}"
                ).format(schema=sql.Literal(self.config.schema))
            )
            user_tables = [row[0] for row in self._cur.fetchall()]

            # 如果用户模式没有表，也检查public模式作为后备
            if not user_tables:
                self._cur.execute(
                    "SELECT table_name FROM information_schema.tables WHERE table_schema = 'public'"
                )
                public_tables = [row[0] for row in self._cur.fetchall()]
                return public_tables

            return user_tables
        except psycopg2.Error as e:
            logger.warning(f"Error getting table names: {e}")
            return []

    def _create_table(self, dimension):
        """创建向量表"""
        try:
            # 使用用户模式创建表
            self._cur.execute(
                sql.SQL(
                    "CREATE TABLE IF NOT EXISTS {schema}.{table_name} "
                    "(id TEXT PRIMARY KEY, sample_id TEXT NOT NULL, embedding vector({dim}));"
                ).format(
                    schema=sql.Identifier(self.config.schema),
                    table_name=sql.Identifier(self.config.table_name),
                    dim=sql.Literal(dimension),
                )
            )
            self._conn.commit()
            logger.info(
                f"Created table {self.config.schema}.{self.config.table_name} with dimension {dimension}"
            )
        except psycopg2.Error as e:
            logger.error(f"Failed to create table: {e}")
            raise

    def _create_vector_index(self):
        """创建向量索引"""
        try:
            # 使用openGauss的HNSW索引
            index_operator = _INDEX_OPERATORS[self.config.metric]

            self._cur.execute(
                sql.SQL(
                    "CREATE INDEX IF NOT EXISTS {index_name} ON {schema}.{table_name} "
                    "USING hnsw (embedding {operator});"
                ).format(
                    index_name=sql.Identifier(self.config.index_name),
                    schema=sql.Identifier(self.config.schema),
                    table_name=sql.Identifier(self.config.table_name),
                    operator=sql.SQL(index_operator),
                )
            )
            self._conn.commit()
            logger.info(f"Created vector index {self.config.index_name}")
        except psycopg2.Error as e:
            logger.warning(f"Failed to create vector index: {e}")

    def add_to_index(
        self,
        embeddings,
        sample_ids,
        label_ids=None,
        overwrite=True,
        allow_existing=True,
        warn_existing=False,
        reload=True,
        batch_size=1000,
    ):
        """添加向量到索引"""
        if self._conn is None or self._conn.closed:
            self._initialize()

        # 确保表存在
        if self.config.table_name not in self._get_table_names():
            self._create_table(embeddings.shape[1])

        # 准备数据
        if label_ids is not None:
            ids = list(label_ids)
        else:
            ids = list(sample_ids)

        sample_ids = list(sample_ids)
        embeddings = [e.tolist() for e in embeddings]

        # 检查重复ID
        existing_ids = set(self._get_existing_ids(ids))

        if existing_ids:
            if not allow_existing:
                raise ValueError(f"Found {len(existing_ids)} existing IDs")
            if warn_existing:
                logger.warning(f"Found {len(existing_ids)} existing IDs")

        try:
            # 分批处理数据
            for batch_embeddings, batch_ids, batch_sample_ids in zip(
                fou.iter_batches(embeddings, batch_size),
                fou.iter_batches(ids, batch_size),
                fou.iter_batches(sample_ids, batch_size),
            ):
                # 分离新增和更新的数据
                insert_data = []
                update_data = []

                for bid, bsid, bemb in zip(
                    batch_ids, batch_sample_ids, batch_embeddings
                ):
                    if bid in existing_ids:
                        if overwrite:
                            update_data.append((bsid, bemb, bid))
                    else:
                        insert_data.append((bid, bsid, bemb))

                # 执行插入操作 - 使用用户模式
                if insert_data:
                    self._cur.executemany(
                        sql.SQL(
                            "INSERT INTO {schema}.{table_name} (id, sample_id, embedding) VALUES(%s, %s, %s);"
                        ).format(
                            schema=sql.Identifier(self.config.schema),
                            table_name=sql.Identifier(self.config.table_name),
                        ),
                        insert_data,
                    )

                # 执行更新操作
                if update_data:
                    self._cur.executemany(
                        sql.SQL(
                            "UPDATE {schema}.{table_name} SET sample_id = %s, embedding = %s WHERE id = %s;"
                        ).format(
                            schema=sql.Identifier(self.config.schema),
                            table_name=sql.Identifier(self.config.table_name),
                        ),
                        update_data,
                    )

                self._conn.commit()

            # 创建向量索引
            self._create_vector_index()

            logger.info(f"Added {len(embeddings)} embeddings to index")

        except psycopg2.Error as e:
            self._conn.rollback()
            logger.error(f"Failed to add embeddings to index: {e}")
            raise

        if reload:
            self.reload()

    def remove_from_index(
        self,
        sample_ids=None,
        label_ids=None,
        allow_missing=True,
        warn_missing=False,
        reload=True,
    ):
        """从索引中删除向量"""
        if self._conn is None or self._conn.closed:
            self._initialize()

        # 确定要删除的ID
        if label_ids is not None:
            ids_to_delete = list(label_ids)
        else:
            ids_to_delete = list(sample_ids)

        # 检查缺失的ID
        if warn_missing or not allow_missing:
            existing_ids = self._get_existing_ids(ids_to_delete)
            missing_ids = set(ids_to_delete) - set(existing_ids)
            if missing_ids and not allow_missing:
                raise ValueError(f"IDs not found in index: {missing_ids}")
            if missing_ids and warn_missing:
                logger.warning(f"IDs not found in index: {missing_ids}")

        # 删除数据 - 使用用户模式
        try:
            if ids_to_delete:
                self._cur.execute(
                    sql.SQL(
                        "DELETE FROM {schema}.{table_name} WHERE id IN ({ids});"
                    ).format(
                        schema=sql.Identifier(self.config.schema),
                        table_name=sql.Identifier(self.config.table_name),
                        ids=sql.SQL(",").join(map(sql.Literal, ids_to_delete)),
                    )
                )
                delete_count = self._cur.rowcount
                self._conn.commit()
                logger.info(f"Removed {delete_count} embeddings from index")
        except psycopg2.Error as e:
            logger.error(f"Failed to remove embeddings: {e}")
            raise

        if reload:
            self.reload()

    def get_embeddings(
        self,
        sample_ids=None,
        label_ids=None,
        allow_missing=True,
        warn_missing=False,
    ):
        """获取向量数据"""
        if self._conn is None or self._conn.closed:
            self._initialize()

        if label_ids is not None:
            query_ids = list(label_ids)
            self._cur.execute(
                sql.SQL(
                    "SELECT id, sample_id, embedding FROM {schema}.{table_name} WHERE id IN ({ids});"
                ).format(
                    schema=sql.Identifier(self.config.schema),
                    table_name=sql.Identifier(self.config.table_name),
                    ids=sql.SQL(",").join(map(sql.Literal, query_ids)),
                )
            )
        elif sample_ids is not None:
            query_ids = list(sample_ids)
            self._cur.execute(
                sql.SQL(
                    "SELECT id, sample_id, embedding FROM {schema}.{table_name} WHERE sample_id IN ({ids});"
                ).format(
                    schema=sql.Identifier(self.config.schema),
                    table_name=sql.Identifier(self.config.table_name),
                    ids=sql.SQL(",").join(map(sql.Literal, query_ids)),
                )
            )
        else:
            query_ids = None
            self._cur.execute(
                sql.SQL(
                    "SELECT id, sample_id, embedding FROM {schema}.{table_name};"
                ).format(
                    schema=sql.Identifier(self.config.schema),
                    table_name=sql.Identifier(self.config.table_name),
                )
            )

        results = self._cur.fetchall()

        if not results:
            return np.array([]), np.array([]), np.array([])

        ids, sample_ids_result, embeddings = zip(*results)
        embeddings = np.array([np.array(ast.literal_eval(emb)) for emb in embeddings])

        # 处理缺失的ID
        if query_ids is not None:
            found_ids = set(ids)
            missing_ids = set(query_ids) - found_ids
            if missing_ids:
                if not allow_missing:
                    raise ValueError(f"IDs not found: {missing_ids}")
                if warn_missing:
                    logger.warning(f"IDs not found: {missing_ids}")

        return embeddings, np.array(sample_ids_result), np.array(ids)

    def _kneighbors(
        self,
        query=None,
        k=None,
        reverse=False,
        aggregation=None,
        return_dists=False,
    ):
        """K近邻查询"""
        if self._conn is None or self._conn.closed:
            self._initialize()

        if query is None:
            raise ValueError(
                "openGauss backend does not support full index neighbors queries"
            )

        if aggregation not in (None, "mean"):
            raise ValueError(f"Unsupported aggregation: {aggregation}")

        if k is None:
            k = self.config.max_k

        # 解析查询向量
        try:
            query = self._parse_neighbors_query(query)
        except Exception as e:
            logger.error(f"Error parsing query: {e}")
            raise

        # 处理聚合
        if aggregation == "mean" and query.ndim == 2:
            query = query.mean(axis=0)

        single_query = query.ndim == 1
        if single_query:
            query = query.reshape(1, -1)

        # 验证向量维度
        if query.size == 0 or (query.ndim > 1 and query.shape[1] == 0):
            raise ValueError("Query vectors have zero dimensions")

        # 获取距离操作符
        distance_op = _SUPPORTED_METRICS[self.config.metric]

        # 构建查询
        sample_ids = []
        label_ids = []
        dists = []

        try:
            for q in query:
                # 确保查询向量是列表格式，然后转换为字符串表示
                q_list = q.tolist() if hasattr(q, "tolist") else list(q)
                q_str = str(q_list)  # 转换为字符串，如 '[1.0, 2.0, 3.0]'

                results = []  # 初始化 results

                if self.has_view:
                    # 如果有视图过滤，需要获取当前视图的ID
                    index_ids = self._get_view_ids()
                    if index_ids:  # 只有当有视图ID时才执行查询
                        self._cur.execute(
                            sql.SQL(
                                "SELECT id, sample_id, embedding {distance_op} %s::vector AS distance "
                                "FROM {schema}.{table_name} WHERE id IN ({ids}) "
                                "ORDER BY distance {order} LIMIT %s::int;"
                            ).format(
                                distance_op=sql.SQL(distance_op),
                                schema=sql.Identifier(self.config.schema),
                                table_name=sql.Identifier(self.config.table_name),
                                ids=sql.SQL(",").join(map(sql.Literal, index_ids)),
                                order=sql.SQL("DESC" if reverse else "ASC"),
                            ),
                            (q_str, k),  # 传递字符串表示的向量
                        )
                        results = self._cur.fetchall()
                    # 如果 index_ids 为空，results 保持为空列表
                else:
                    self._cur.execute(
                        sql.SQL(
                            "SELECT id, sample_id, embedding {distance_op} %s::vector AS distance "
                            "FROM {schema}.{table_name} ORDER BY distance {order} LIMIT %s::int;"
                        ).format(
                            distance_op=sql.SQL(distance_op),
                            schema=sql.Identifier(self.config.schema),
                            table_name=sql.Identifier(self.config.table_name),
                            order=sql.SQL("DESC" if reverse else "ASC"),
                        ),
                        (q_str, k),  # 传递字符串表示的向量
                    )
                    results = self._cur.fetchall()

                # 处理查询结果
                if results:
                    batch_label_ids, batch_sample_ids, batch_dists = zip(*results)
                    sample_ids.append(list(batch_sample_ids))
                    label_ids.append(list(batch_label_ids))
                    dists.append(list(batch_dists))
                else:
                    sample_ids.append([])
                    label_ids.append([])
                    dists.append([])

        except psycopg2.Error as e:
            logger.error(f"Database error during similarity search: {e}")
            raise
        except Exception as e:
            logger.error(f"Error during similarity search: {e}")
            raise

        if single_query:
            sample_ids = sample_ids[0] if sample_ids else []
            label_ids = label_ids[0] if label_ids else []
            dists = dists[0] if dists else []

        result = (sample_ids, label_ids)

        if return_dists:
            return result, dists

        return result

    def _parse_neighbors_query(self, query):
        """解析查询参数"""
        if _is_str(query):
            # 单个ID字符串
            query_ids = [query]
            single_query = True
        elif hasattr(query, "__iter__") and not isinstance(query, (np.ndarray, list)):
            # 可迭代对象但不是数组或列表，转换为列表
            query_ids = list(query)
            single_query = False
        else:
            try:
                # 尝试作为向量数组处理
                query = np.asarray(query, dtype=float)
                if query.ndim in (1, 2) and query.size > 0:
                    # 检查是否为有效的向量数据
                    if query.ndim == 1 or (query.ndim == 2 and query.shape[1] > 1):
                        return query
            except (ValueError, TypeError):
                pass

            # 如果不是向量数据，作为ID列表处理
            if hasattr(query, "__iter__"):
                query_ids = list(query)
                single_query = len(query_ids) == 1
            else:
                query_ids = [query]
                single_query = True

        # 根据ID获取向量
        try:
            embeddings, _, found_ids = self.get_embeddings(label_ids=query_ids)

            if len(embeddings) == 0:
                raise ValueError(f"No embeddings found for query IDs: {query_ids}")

            # 确保返回的向量顺序与查询ID顺序一致
            if len(found_ids) != len(query_ids):
                missing_ids = set(query_ids) - set(found_ids)
                logger.warning(f"Some query IDs not found: {missing_ids}")

            query = embeddings
            if single_query and len(embeddings) > 0:
                query = embeddings[0]

        except Exception as e:
            logger.error(f"Error retrieving embeddings for query IDs {query_ids}: {e}")
            raise ValueError(f"Failed to retrieve embeddings for query: {e}")

        return query

    def _get_existing_ids(self, ids):
        """获取已存在的ID"""
        if not ids:
            return []

        self._cur.execute(
            sql.SQL("SELECT id FROM {schema}.{table_name} WHERE id IN ({ids});").format(
                schema=sql.Identifier(self.config.schema),
                table_name=sql.Identifier(self.config.table_name),
                ids=sql.SQL(",").join(map(sql.Literal, ids)),
            )
        )
        return [row[0] for row in self._cur.fetchall()]

    def _get_view_ids(self):
        """获取当前视图的ID列表"""
        # 这里需要根据FiftyOne的视图机制来实现
        # 暂时返回空列表，实际实现需要与FiftyOne的视图系统集成
        return []

    def cleanup(self):
        """清理资源"""
        if self._cur and not self._cur.closed:
            try:
                # 删除索引
                self._cur.execute(
                    sql.SQL("DROP INDEX IF EXISTS {schema}.{index_name};").format(
                        schema=sql.Identifier(self.config.schema),
                        index_name=sql.Identifier(self.config.index_name),
                    )
                )
                # 删除表
                self._cur.execute(
                    sql.SQL("DROP TABLE IF EXISTS {schema}.{table_name};").format(
                        schema=sql.Identifier(self.config.schema),
                        table_name=sql.Identifier(self.config.table_name),
                    )
                )
                self._conn.commit()
                logger.info(
                    f"Cleaned up table {self.config.schema}.{self.config.table_name} and index {self.config.index_name}"
                )
            except psycopg2.Error as e:
                logger.error(f"Error during cleanup: {e}")
            finally:
                self._cur.close()

        if self._conn and not self._conn.closed:
            self._conn.close()

    def reload(self):
        """重新加载索引"""
        super().reload()

    @classmethod
    def _from_dict(cls, d, samples, config, brain_key):
        """从字典创建实例"""
        return cls(samples, config, brain_key)

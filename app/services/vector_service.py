"""
Qdrant 向量数据库服务
负责文档的向量化存储和语义检索
"""

import logging
import uuid
from typing import Dict, List, Optional, Any, Tuple
from datetime import datetime

try:
    from qdrant_client import QdrantClient
    from qdrant_client.http import models
    from qdrant_client.http.models import (
        Distance, VectorParams, CreateCollection, PointStruct,
        Filter, FieldCondition, MatchValue, SearchRequest
    )
except ImportError:
    raise ImportError("请安装qdrant-client库: pip install qdrant-client")

from app.core.config import settings
from app.models.responses import ErrorCode
from app.core.exceptions import create_service_exception

logger = logging.getLogger("rag-anything")


class VectorService:
    """Qdrant 向量数据库服务"""
    
    def __init__(self):
        self.client: Optional[QdrantClient] = None
        self.default_collection = settings.QDRANT_COLLECTION_NAME
        self._connected = False
        
    async def initialize(self):
        """初始化Qdrant连接"""
        if self._connected:
            return
            
        try:
            # 创建Qdrant客户端 - 增加超时时间处理3072维向量
            self.client = QdrantClient(
                host=settings.QDRANT_HOST,
                port=settings.QDRANT_PORT,
                timeout=120  # 增加到2分钟，处理大维度向量
            )
            
            # 测试连接
            self.client.get_collections()
            
            # 确保默认集合存在
            await self._ensure_collection_exists(self.default_collection)
            
            self._connected = True
            logger.info(f"Qdrant 服务初始化成功，连接到 {settings.QDRANT_HOST}:{settings.QDRANT_PORT}")
            
        except Exception as e:
            logger.error(f"Qdrant 服务初始化失败: {e}")
            raise create_service_exception(
                ErrorCode.INTERNAL_SERVER_ERROR,
                f"Qdrant 连接失败: {str(e)}"
            )
    
    async def _ensure_collection_exists(self, collection_name: str, vector_size: int = None):
        """确保集合存在"""
        try:
            # 使用配置中的向量维度
            if vector_size is None:
                vector_size = settings.EMBEDDING_DIMENSION
                
            # 检查集合是否已存在
            collections = self.client.get_collections()
            existing_collections = [col.name for col in collections.collections]
            
            if collection_name not in existing_collections:
                # 创建新集合
                self.client.create_collection(
                    collection_name=collection_name,
                    vectors_config=VectorParams(
                        size=vector_size,  # 使用Qwen3-Embedding-8B的维度：3072
                        distance=Distance.COSINE  # 使用余弦相似度
                    )
                )
                logger.info(f"创建向量集合: {collection_name} - 维度: {vector_size}")
            else:
                logger.info(f"向量集合已存在: {collection_name}")
                
        except Exception as e:
            logger.error(f"确保集合存在失败: {collection_name} - {e}")
            raise
    
    async def create_collection(self, collection_name: str, vector_size: int = None) -> bool:
        """创建向量集合"""
        if not self._connected:
            await self.initialize()
            
        try:
            # 使用配置中的向量维度
            if vector_size is None:
                vector_size = settings.EMBEDDING_DIMENSION
                
            self.client.create_collection(
                collection_name=collection_name,
                vectors_config=VectorParams(
                    size=vector_size,
                    distance=Distance.COSINE
                )
            )
            logger.info(f"创建向量集合成功: {collection_name} - 维度: {vector_size}")
            return True
            
        except Exception as e:
            logger.error(f"创建向量集合失败: {collection_name} - {e}")
            return False
    
    async def delete_collection(self, collection_name: str) -> bool:
        """删除向量集合"""
        if not self._connected:
            await self.initialize()
            
        try:
            self.client.delete_collection(collection_name)
            logger.info(f"删除向量集合成功: {collection_name}")
            return True
            
        except Exception as e:
            logger.error(f"删除向量集合失败: {collection_name} - {e}")
            return False
    
    async def list_collections(self) -> List[str]:
        """列出所有集合"""
        if not self._connected:
            await self.initialize()
            
        try:
            collections = self.client.get_collections()
            return [col.name for col in collections.collections]
            
        except Exception as e:
            logger.error(f"列出集合失败: {e}")
            return []
    
    async def add_points(
        self,
        collection_name: str,
        points: List[Dict[str, Any]],
        vectors: List[List[float]]
    ) -> List[str]:
        """添加向量点"""
        if not self._connected:
            await self.initialize()
            
        if len(points) != len(vectors):
            raise ValueError("点数据和向量数量不匹配")
            
        try:
            # 生成点ID
            point_ids = [str(uuid.uuid4()) for _ in points]
            
            # 构建PointStruct
            point_structs = []
            for i, (point_data, vector) in enumerate(zip(points, vectors)):
                # 添加时间戳到payload
                point_data["created_at"] = datetime.now().isoformat()
                point_data["point_id"] = point_ids[i]
                
                point_structs.append(
                    PointStruct(
                        id=point_ids[i],
                        vector=vector,
                        payload=point_data
                    )
                )
            
            # 分批插入 - 处理3072维向量时避免超时
            batch_size = 50  # 每批处理50个向量点
            total_points = len(point_structs)
            
            for i in range(0, total_points, batch_size):
                batch_points = point_structs[i:i + batch_size]
                try:
                    self.client.upsert(
                        collection_name=collection_name,
                        points=batch_points
                    )
                    logger.info(f"批次 {i//batch_size + 1}: 插入 {len(batch_points)} 个向量点")
                except Exception as batch_e:
                    logger.error(f"批次插入失败: {i//batch_size + 1} - {batch_e}")
                    raise batch_e
            
            logger.info(f"添加向量点成功: {collection_name} - {len(point_ids)}个点，分{(total_points + batch_size - 1) // batch_size}批完成")
            return point_ids
            
        except Exception as e:
            logger.error(f"添加向量点失败: {collection_name} - {e}")
            raise create_service_exception(
                ErrorCode.INTERNAL_SERVER_ERROR,
                f"添加向量点失败: {str(e)}"
            )
    
    async def search_vectors(
        self,
        collection_name: str,
        query_vector: List[float],
        limit: int = 10,
        score_threshold: float = 0.0,
        filter_conditions: Optional[Dict[str, Any]] = None
    ) -> List[Dict[str, Any]]:
        """向量搜索"""
        if not self._connected:
            await self.initialize()
        
        # 🔧 验证collection_name参数
        if not isinstance(collection_name, str):
            logger.error(f"collection_name必须是字符串，当前类型: {type(collection_name)}, 值: {collection_name}")
            raise ValueError(f"collection_name必须是字符串，当前类型: {type(collection_name)}")
            
        try:
            # 构建过滤条件
            search_filter = None
            if filter_conditions:
                conditions = []
                for key, value in filter_conditions.items():
                    conditions.append(
                        FieldCondition(
                            key=key,
                            match=MatchValue(value=value)
                        )
                    )
                search_filter = Filter(must=conditions)
            
            # 调试：记录搜索参数
            logger.info(f"🔍 向量搜索调试:")
            logger.info(f"   集合: {collection_name}")
            logger.info(f"   向量维度: {len(query_vector)}")
            logger.info(f"   向量前5个值: {query_vector[:5]}")
            logger.info(f"   限制: {limit}")
            logger.info(f"   阈值: {score_threshold}")
            logger.info(f"   过滤条件: {filter_conditions}")
            
            # 执行搜索
            search_results = self.client.search(
                collection_name=collection_name,
                query_vector=query_vector,
                limit=limit,
                score_threshold=score_threshold,
                query_filter=search_filter,
                with_payload=True,
                with_vectors=False
            )
            
            # 处理结果
            results = []
            for hit in search_results:
                result = {
                    "id": hit.id,
                    "score": hit.score,
                    "payload": hit.payload
                }
                results.append(result)
                # 调试：记录找到的结果
                logger.info(f"   📄 找到结果: score={hit.score:.4f}, text='{hit.payload.get('text', '')[:50]}...'")
            
            logger.info(f"🔍 向量搜索完成: {collection_name} - 找到{len(results)}个结果")
            return results
            
        except Exception as e:
            logger.error(f"向量搜索失败: {collection_name} - {e}")
            raise create_service_exception(
                ErrorCode.INTERNAL_SERVER_ERROR,
                f"向量搜索失败: {str(e)}"
            )
    
    async def get_point(self, collection_name: str, point_id: str) -> Optional[Dict[str, Any]]:
        """获取单个点"""
        if not self._connected:
            await self.initialize()
            
        try:
            points = self.client.retrieve(
                collection_name=collection_name,
                ids=[point_id],
                with_payload=True,
                with_vectors=True
            )
            
            if not points:
                return None
                
            point = points[0]
            return {
                "id": point.id,
                "vector": point.vector,
                "payload": point.payload
            }
            
        except Exception as e:
            logger.error(f"获取向量点失败: {collection_name} - {point_id} - {e}")
            return None
    
    async def delete_points(self, collection_name: str, point_ids: List[str]) -> bool:
        """删除向量点"""
        if not self._connected:
            await self.initialize()
            
        try:
            self.client.delete(
                collection_name=collection_name,
                points_selector=models.PointIdsList(
                    points=point_ids
                )
            )
            
            logger.info(f"删除向量点成功: {collection_name} - {len(point_ids)}个点")
            return True
            
        except Exception as e:
            logger.error(f"删除向量点失败: {collection_name} - {e}")
            return False
    
    async def update_point_payload(
        self,
        collection_name: str,
        point_id: str,
        payload: Dict[str, Any]
    ) -> bool:
        """更新点的payload"""
        if not self._connected:
            await self.initialize()
            
        try:
            # 添加更新时间
            payload["updated_at"] = datetime.now().isoformat()
            
            self.client.set_payload(
                collection_name=collection_name,
                payload=payload,
                points=[point_id]
            )
            
            logger.debug(f"更新向量点payload成功: {collection_name} - {point_id}")
            return True
            
        except Exception as e:
            logger.error(f"更新向量点payload失败: {collection_name} - {point_id} - {e}")
            return False
    
    async def count_points(self, collection_name: str) -> int:
        """统计点数量"""
        if not self._connected:
            await self.initialize()
            
        try:
            info = self.client.get_collection(collection_name)
            return info.points_count or 0
            
        except Exception as e:
            logger.error(f"统计向量点数量失败: {collection_name} - {e}")
            return 0
    
    async def scroll_points(
        self,
        collection_name: str,
        limit: int = 100,
        offset: Optional[str] = None,
        filter_conditions: Optional[Dict[str, Any]] = None
    ) -> Tuple[List[Dict[str, Any]], Optional[str]]:
        """滚动获取点（分页）"""
        if not self._connected:
            await self.initialize()
            
        try:
            # 构建过滤条件
            scroll_filter = None
            if filter_conditions:
                conditions = []
                for key, value in filter_conditions.items():
                    conditions.append(
                        FieldCondition(
                            key=key,
                            match=MatchValue(value=value)
                        )
                    )
                scroll_filter = Filter(must=conditions)
            
            # 执行滚动查询
            result, next_page_offset = self.client.scroll(
                collection_name=collection_name,
                limit=limit,
                offset=offset,
                scroll_filter=scroll_filter,
                with_payload=True,
                with_vectors=False
            )
            
            # 处理结果
            points = []
            for point in result:
                points.append({
                    "id": point.id,
                    "payload": point.payload
                })
            
            return points, next_page_offset
            
        except Exception as e:
            logger.error(f"滚动获取向量点失败: {collection_name} - {e}")
            return [], None
    
    async def search_by_text_filter(
        self,
        collection_name: str,
        text_field: str,
        search_text: str,
        limit: int = 100
    ) -> List[Dict[str, Any]]:
        """根据文本字段搜索"""
        if not self._connected:
            await self.initialize()
            
        try:
            # 使用滚动查询配合客户端过滤
            points, _ = await self.scroll_points(
                collection_name=collection_name,
                limit=limit
            )
            
            # 客户端文本匹配过滤
            filtered_points = []
            search_text_lower = search_text.lower()
            
            for point in points:
                if text_field in point["payload"]:
                    field_value = str(point["payload"][text_field]).lower()
                    if search_text_lower in field_value:
                        filtered_points.append(point)
            
            return filtered_points[:limit]
            
        except Exception as e:
            logger.error(f"文本搜索失败: {collection_name} - {e}")
            return []
    
    async def get_collection_info(self, collection_name: str) -> Optional[Dict[str, Any]]:
        """获取集合信息"""
        if not self._connected:
            await self.initialize()
            
        try:
            info = self.client.get_collection(collection_name)
            return {
                "name": collection_name,
                "points_count": info.points_count,
                "segments_count": info.segments_count,
                "vectors_count": info.vectors_count,
                "status": info.status,
                "optimizer_status": info.optimizer_status,
                "config": {
                    "params": info.config.params.__dict__ if info.config.params else None,
                    "hnsw_config": info.config.hnsw_config.__dict__ if info.config.hnsw_config else None,
                    "optimizer_config": info.config.optimizer_config.__dict__ if info.config.optimizer_config else None,
                }
            }
            
        except Exception as e:
            logger.error(f"获取集合信息失败: {collection_name} - {e}")
            return None
    
    async def health_check(self) -> bool:
        """健康检查"""
        try:
            if not self._connected:
                await self.initialize()
            self.client.get_collections()
            return True
        except:
            return False
    
    # ===================
    # 特定业务操作
    # ===================
    
    async def add_document_chunks(
        self,
        file_id: str,
        chunks: List[Dict[str, Any]],
        vectors: List[List[float]],
        collection_name: Optional[str] = None
    ) -> List[str]:
        """添加文档块到向量数据库"""
        if collection_name is None:
            collection_name = self.default_collection
            
        # 🔧 确保集合存在 - 修复删除后重新解析的bug
        await self._ensure_collection_exists(collection_name)
            
        # 为每个块添加文件ID和块信息
        enriched_chunks = []
        for i, chunk in enumerate(chunks):
            enriched_chunk = {
                "file_id": file_id,
                "chunk_index": i,
                "chunk_id": f"{file_id}_{i}",
                **chunk
            }
            enriched_chunks.append(enriched_chunk)
        
        return await self.add_points(collection_name, enriched_chunks, vectors)
    
    async def search_documents(
        self,
        query_vector: List[float],
        file_ids: Optional[List[str]] = None,
        limit: int = 10,
        score_threshold: float = 0.1,  # 🔧 大幅降低默认阈值确保能找到结果
        collection_name: Optional[str] = None
    ) -> List[Dict[str, Any]]:
        """搜索文档"""
        if collection_name is None:
            collection_name = self.default_collection
        
        # 🔧 增强调试信息
        logger.debug(f"search_documents调用参数: collection_name={collection_name}, file_ids={file_ids}, limit={limit}, score_threshold={score_threshold}")
        
        # 构建过滤条件
        filter_conditions = {}
        if file_ids:
            # 注意：这里的实现可能需要根据Qdrant的具体API调整
            # 当前实现仅支持单个文件ID过滤
            if len(file_ids) == 1:
                filter_conditions["file_id"] = file_ids[0]
        
        return await self.search_vectors(
            collection_name=collection_name,
            query_vector=query_vector,
            limit=limit,
            score_threshold=score_threshold,
            filter_conditions=filter_conditions
        )
    
    async def delete_document(self, file_id: str, collection_name: Optional[str] = None) -> bool:
        """删除文档的所有向量"""
        if collection_name is None:
            collection_name = self.default_collection
        
        try:
            # 首先查找该文件的所有点
            points, _ = await self.scroll_points(
                collection_name=collection_name,
                limit=1000,  # 假设单个文件不会超过1000个块
                filter_conditions={"file_id": file_id}
            )
            
            if not points:
                logger.info(f"文件 {file_id} 在向量数据库中未找到")
                return True
            
            # 提取点ID
            point_ids = [point["id"] for point in points]
            
            # 删除所有相关点
            result = await self.delete_points(collection_name, point_ids)
            
            if result:
                logger.info(f"删除文件向量成功: {file_id} - 删除了{len(point_ids)}个向量点")
            
            return result
            
        except Exception as e:
            logger.error(f"删除文件向量失败: {file_id} - {e}")
            return False


# 全局向量服务实例
vector_service = VectorService()


async def get_vector_service() -> VectorService:
    """获取向量服务实例"""
    return vector_service 
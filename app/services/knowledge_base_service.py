"""
知识库管理服务
提供知识库的创建、管理、检索等功能
"""

import uuid
import asyncio
from datetime import datetime
from typing import Optional, List, Dict, Any, Tuple
from app.core.config import settings
from app.core.exceptions import create_service_exception
from app.models.responses import ErrorCode
from app.models.knowledge_base import (
    KnowledgeBase, KnowledgeBaseCreate, KnowledgeBaseUpdate, 
    KnowledgeBaseStatus, KnowledgeBaseStats, QdrantConfig
)
from app.services.cache_service import get_cache_service
from app.services.vector_service import get_vector_service
from app.services.document_service import get_document_service
from app.services.minio_service import get_minio_service
from app.utils.logger import logger


class KnowledgeBaseService:
    """知识库管理服务"""
    
    def __init__(self):
        self.cache_service = None
        self.vector_service = None
        self.document_service = None
        self.minio_service = None
        
    async def _get_services(self):
        """延迟初始化服务依赖"""
        if not self.cache_service:
            self.cache_service = await get_cache_service()
        if not self.vector_service:
            self.vector_service = await get_vector_service()
        if not self.document_service:
            self.document_service = await get_document_service()
        if not self.minio_service:
            self.minio_service = await get_minio_service()
    
    def _generate_kb_id(self) -> str:
        """生成知识库ID"""
        return f"kb_{uuid.uuid4().hex[:12]}"
    
    def _generate_collection_name(self, kb_id: str) -> str:
        """生成Qdrant集合名称"""
        return f"kb_{kb_id}"
    
    async def create_knowledge_base(self, request: KnowledgeBaseCreate) -> KnowledgeBase:
        """创建知识库"""
        await self._get_services()
        
        try:
            # 生成知识库ID
            kb_id = self._generate_kb_id()
            collection_name = self._generate_collection_name(kb_id)
            
            # 创建Qdrant配置
            qdrant_config = QdrantConfig(
                collection_name=collection_name,
                vector_size=request.vector_size,
                distance_metric=request.distance_metric,
                top_k=request.top_k,
                score_threshold=request.score_threshold
            )
            
            # 创建知识库对象
            knowledge_base = KnowledgeBase(
                kb_id=kb_id,
                name=request.name,
                description=request.description,
                tags=request.tags,
                category=request.category,
                qdrant_config=qdrant_config,
                status=KnowledgeBaseStatus.ACTIVE,
                created_at=datetime.now(),
                updated_at=datetime.now()
            )
            
            # 创建Qdrant集合
            success = await self.vector_service.create_collection(
                collection_name=collection_name,
                vector_size=request.vector_size
            )
            
            if not success:
                raise create_service_exception(
                    ErrorCode.INTERNAL_SERVER_ERROR,
                    f"创建向量集合失败: {collection_name}"
                )
            
            # 保存知识库元数据
            await self.cache_service.save_data(
                f"knowledge_base:{kb_id}",
                knowledge_base.dict()
            )
            
            # 添加到知识库列表
            await self.cache_service.redis.sadd("knowledge_bases", kb_id)
            
            logger.info(f"知识库创建成功: {kb_id} - {request.name}")
            return knowledge_base
            
        except Exception as e:
            logger.error(f"知识库创建失败: {e}")
            raise create_service_exception(
                ErrorCode.INTERNAL_SERVER_ERROR,
                f"知识库创建失败: {str(e)}"
            )
    
    async def get_knowledge_base(self, kb_id: str) -> Optional[KnowledgeBase]:
        """获取知识库信息"""
        await self._get_services()
        
        try:
            kb_data = await self.cache_service.get_data(f"knowledge_base:{kb_id}")
            if not kb_data:
                return None
            
            return KnowledgeBase(**kb_data)
            
        except Exception as e:
            logger.error(f"获取知识库失败: {kb_id} - {e}")
            return None
    
    async def update_knowledge_base(self, kb_id: str, request: KnowledgeBaseUpdate) -> Optional[KnowledgeBase]:
        """更新知识库"""
        await self._get_services()
        
        try:
            # 获取现有知识库
            knowledge_base = await self.get_knowledge_base(kb_id)
            if not knowledge_base:
                raise create_service_exception(
                    ErrorCode.NOT_FOUND,
                    f"知识库不存在: {kb_id}"
                )
            
            # 更新字段
            update_data = request.dict(exclude_unset=True)
            for field, value in update_data.items():
                if field in ["top_k", "score_threshold", "hnsw_ef_search"]:
                    # 更新Qdrant配置
                    setattr(knowledge_base.qdrant_config, field, value)
                else:
                    setattr(knowledge_base, field, value)
            
            # 更新时间戳
            knowledge_base.updated_at = datetime.now()
            
            # 保存更新后的数据
            await self.cache_service.save_data(
                f"knowledge_base:{kb_id}",
                knowledge_base.dict()
            )
            
            logger.info(f"知识库更新成功: {kb_id}")
            return knowledge_base
            
        except Exception as e:
            logger.error(f"知识库更新失败: {kb_id} - {e}")
            raise create_service_exception(
                ErrorCode.INTERNAL_SERVER_ERROR,
                f"知识库更新失败: {str(e)}"
            )
    
    async def delete_knowledge_base(self, kb_id: str, delete_files: bool = False) -> bool:
        """删除知识库"""
        await self._get_services()
        
        try:
            # 获取知识库信息
            knowledge_base = await self.get_knowledge_base(kb_id)
            if not knowledge_base:
                logger.warning(f"知识库不存在，无法删除: {kb_id}")
                return True
            
            collection_name = knowledge_base.qdrant_config.collection_name
            
            # 删除向量集合
            await self.vector_service.delete_collection(collection_name)
            
            if delete_files:
                # 获取知识库中的所有文件
                file_ids = await self.get_knowledge_base_files(kb_id)
                
                # 删除所有文件
                for file_id in file_ids:
                    await self.document_service.delete_file(
                        file_id, 
                        delete_parsed_data=True, 
                        delete_vector_data=False  # 向量集合已删除
                    )
            
            # 删除知识库元数据
            await self.cache_service.delete_data(f"knowledge_base:{kb_id}")
            
            # 从知识库列表中移除
            await self.cache_service.redis.srem("knowledge_bases", kb_id)
            
            logger.info(f"知识库删除成功: {kb_id}")
            return True
            
        except Exception as e:
            logger.error(f"知识库删除失败: {kb_id} - {e}")
            return False
    
    async def list_knowledge_bases(
        self, 
        limit: int = 20, 
        offset: int = 0,
        status_filter: Optional[KnowledgeBaseStatus] = None
    ) -> Tuple[List[KnowledgeBase], int]:
        """列出知识库"""
        await self._get_services()
        
        try:
            # 获取所有知识库ID
            kb_ids = await self.cache_service.redis.smembers("knowledge_bases")
            kb_ids = [kb_id.decode() if isinstance(kb_id, bytes) else kb_id for kb_id in kb_ids]
            
            # 批量获取知识库数据
            knowledge_bases = []
            for kb_id in kb_ids:
                kb = await self.get_knowledge_base(kb_id)
                if kb:
                    # 状态过滤
                    if status_filter is None or kb.status == status_filter:
                        knowledge_bases.append(kb)
            
            # 排序（按创建时间倒序）
            knowledge_bases.sort(key=lambda x: x.created_at, reverse=True)
            
            # 分页
            total = len(knowledge_bases)
            paginated = knowledge_bases[offset:offset + limit]
            
            return paginated, total
            
        except Exception as e:
            logger.error(f"列出知识库失败: {e}")
            return [], 0
    
    async def add_file_to_knowledge_base(self, kb_id: str, file_id: str) -> bool:
        """将文件添加到知识库"""
        await self._get_services()
        
        try:
            # 验证知识库存在
            knowledge_base = await self.get_knowledge_base(kb_id)
            if not knowledge_base:
                raise create_service_exception(
                    ErrorCode.NOT_FOUND,
                    f"知识库不存在: {kb_id}"
                )
            
            # 更新文件元数据，添加知识库关联
            file_metadata = await self.document_service.get_file_info(file_id)
            if not file_metadata:
                raise create_service_exception(
                    ErrorCode.FILE_NOT_FOUND,
                    f"文件不存在: {file_id}"
                )
            
            file_metadata["kb_id"] = kb_id
            file_metadata["kb_name"] = knowledge_base.name
            await self.cache_service.save_file_metadata(file_id, file_metadata)
            
            # 添加到知识库文件列表
            await self.cache_service.redis.sadd(f"kb_files:{kb_id}", file_id)
            
            # 更新知识库统计
            await self._update_knowledge_base_stats(kb_id)
            
            logger.info(f"文件添加到知识库成功: {file_id} -> {kb_id}")
            return True
            
        except Exception as e:
            logger.error(f"添加文件到知识库失败: {file_id} -> {kb_id} - {e}")
            return False
    
    async def remove_file_from_knowledge_base(self, kb_id: str, file_id: str) -> bool:
        """从知识库中移除文件"""
        await self._get_services()
        
        try:
            # 从文件元数据中移除知识库关联
            file_metadata = await self.document_service.get_file_info(file_id)
            if file_metadata:
                file_metadata.pop("kb_id", None)
                file_metadata.pop("kb_name", None)
                await self.cache_service.save_file_metadata(file_id, file_metadata)
            
            # 从知识库文件列表中移除
            await self.cache_service.redis.srem(f"kb_files:{kb_id}", file_id)
            
            # 删除该文件的向量数据（从知识库的向量集合中）
            knowledge_base = await self.get_knowledge_base(kb_id)
            if knowledge_base:
                await self.vector_service.delete_document(
                    file_id, 
                    knowledge_base.qdrant_config.collection_name
                )
            
            # 更新知识库统计
            await self._update_knowledge_base_stats(kb_id)
            
            logger.info(f"文件从知识库移除成功: {file_id} <- {kb_id}")
            return True
            
        except Exception as e:
            logger.error(f"从知识库移除文件失败: {file_id} <- {kb_id} - {e}")
            return False
    
    async def get_knowledge_base_files(self, kb_id: str) -> List[str]:
        """获取知识库中的文件列表"""
        await self._get_services()
        
        try:
            file_ids = await self.cache_service.redis.smembers(f"kb_files:{kb_id}")
            return [file_id.decode() if isinstance(file_id, bytes) else file_id for file_id in file_ids]
        except Exception as e:
            logger.error(f"获取知识库文件失败: {kb_id} - {e}")
            return []
    
    async def _update_knowledge_base_stats(self, kb_id: str):
        """更新知识库统计信息"""
        try:
            knowledge_base = await self.get_knowledge_base(kb_id)
            if not knowledge_base:
                return
            
            # 获取文件列表
            file_ids = await self.get_knowledge_base_files(kb_id)
            
            # 统计信息
            total_size = 0
            parsed_files = 0
            vectorized_files = 0
            
            for file_id in file_ids:
                file_metadata = await self.document_service.get_file_info(file_id)
                if file_metadata:
                    total_size += file_metadata.get("file_size", 0)
                    if file_metadata.get("parse_status") == "completed":
                        parsed_files += 1
                    if file_metadata.get("vector_status") == "completed":
                        vectorized_files += 1
            
            # 获取向量数量
            vector_count = await self.vector_service.count_points(
                knowledge_base.qdrant_config.collection_name
            )
            
            # 更新知识库统计
            knowledge_base.file_count = len(file_ids)
            knowledge_base.document_count = parsed_files
            knowledge_base.vector_count = vector_count
            knowledge_base.total_size = total_size
            knowledge_base.updated_at = datetime.now()
            
            # 保存更新
            await self.cache_service.save_data(
                f"knowledge_base:{kb_id}",
                knowledge_base.dict()
            )
            
        except Exception as e:
            logger.error(f"更新知识库统计失败: {kb_id} - {e}")
    
    async def get_knowledge_base_stats(self, kb_id: str) -> Optional[KnowledgeBaseStats]:
        """获取知识库详细统计信息"""
        await self._get_services()
        
        try:
            knowledge_base = await self.get_knowledge_base(kb_id)
            if not knowledge_base:
                return None
            
            # 获取文件详细统计
            file_ids = await self.get_knowledge_base_files(kb_id)
            
            file_type_distribution = {}
            parse_status_distribution = {}
            vector_status_distribution = {}
            total_size = 0
            last_indexed = None
            
            for file_id in file_ids:
                file_metadata = await self.document_service.get_file_info(file_id)
                if file_metadata:
                    # 文件类型分布
                    file_type = file_metadata.get("content_type", "unknown")
                    file_type_distribution[file_type] = file_type_distribution.get(file_type, 0) + 1
                    
                    # 解析状态分布
                    parse_status = file_metadata.get("parse_status", "pending")
                    parse_status_distribution[parse_status] = parse_status_distribution.get(parse_status, 0) + 1
                    
                    # 向量化状态分布
                    vector_status = file_metadata.get("vector_status", "pending")
                    vector_status_distribution[vector_status] = vector_status_distribution.get(vector_status, 0) + 1
                    
                    # 累计大小
                    total_size += file_metadata.get("file_size", 0)
                    
                    # 最后索引时间
                    vectorized_at = file_metadata.get("vectorized_at")
                    if vectorized_at:
                        indexed_time = datetime.fromisoformat(vectorized_at)
                        if not last_indexed or indexed_time > last_indexed:
                            last_indexed = indexed_time
            
            # 计算平均文件大小
            avg_file_size = total_size / len(file_ids) if file_ids else 0
            
            stats = KnowledgeBaseStats(
                kb_id=kb_id,
                name=knowledge_base.name,
                status=knowledge_base.status,
                file_count=len(file_ids),
                document_count=knowledge_base.document_count,
                vector_count=knowledge_base.vector_count,
                total_size=total_size,
                avg_file_size=avg_file_size,
                file_type_distribution=file_type_distribution,
                parse_status_distribution=parse_status_distribution,
                vector_status_distribution=vector_status_distribution,
                created_at=knowledge_base.created_at,
                last_updated=knowledge_base.updated_at,
                last_indexed=last_indexed
            )
            
            return stats
            
        except Exception as e:
            logger.error(f"获取知识库统计失败: {kb_id} - {e}")
            return None


# 全局知识库服务实例
knowledge_base_service = KnowledgeBaseService()


async def get_knowledge_base_service() -> KnowledgeBaseService:
    """获取知识库服务实例"""
    return knowledge_base_service 
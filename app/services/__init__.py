"""
服务管理模块
统一管理所有基础服务的初始化和生命周期
"""

import logging
from typing import Dict, Any
import asyncio

from app.services.storage_service import MinIOService, get_minio_service
from app.services.cache_service import CacheService, get_cache_service
from app.services.vector_service import VectorService, get_vector_service
from app.services.document_service import DocumentService, get_document_service
from app.services.search_service import SearchService, get_search_service
from app.services.task_service import TaskService, get_task_service

logger = logging.getLogger("rag-anything")


class ServiceManager:
    """服务管理器"""
    
    def __init__(self):
        self._services = {}
        self._initialized = False
    
    async def initialize_all_services(self):
        """初始化所有服务"""
        if self._initialized:
            return
        
        logger.info("开始初始化所有服务...")
        
        try:
            # 初始化基础服务（按依赖顺序）
            logger.info("初始化基础存储服务...")
            
            # 1. 初始化MinIO服务
            minio_service = await get_minio_service()
            await minio_service.initialize()
            self._services['minio'] = minio_service
            logger.info("✓ MinIO服务初始化完成")
            
            # 2. 初始化Redis缓存服务
            cache_service = await get_cache_service()
            await cache_service.initialize()
            self._services['cache'] = cache_service
            logger.info("✓ Redis缓存服务初始化完成")
            
            # 3. 初始化Qdrant向量服务
            vector_service = await get_vector_service()
            await vector_service.initialize()
            self._services['vector'] = vector_service
            logger.info("✓ Qdrant向量服务初始化完成")
            
            # 4. 初始化任务管理服务
            task_service = await get_task_service()
            await task_service.initialize()
            self._services['task'] = task_service
            logger.info("✓ 任务管理服务初始化完成")
            
            # 初始化应用服务
            logger.info("初始化应用服务...")
            
            # 5. 初始化文档服务
            document_service = await get_document_service()
            self._services['document'] = document_service
            logger.info("✓ 文档处理服务初始化完成")
            
            # 6. 初始化搜索服务
            search_service = await get_search_service()
            self._services['search'] = search_service
            logger.info("✓ 搜索服务初始化完成")
            
            self._initialized = True
            logger.info("所有服务初始化完成！")
            
        except Exception as e:
            logger.error(f"服务初始化失败: {e}")
            # 清理已初始化的服务
            await self.cleanup_services()
            raise
    
    async def cleanup_services(self):
        """清理所有服务"""
        logger.info("开始清理服务...")
        
        # 按反向顺序清理服务
        cleanup_order = ['search', 'document', 'task', 'vector', 'cache', 'minio']
        
        for service_name in cleanup_order:
            if service_name in self._services:
                try:
                    service = self._services[service_name]
                    if hasattr(service, 'cleanup'):
                        await service.cleanup()
                    logger.info(f"✓ {service_name}服务清理完成")
                except Exception as e:
                    logger.error(f"清理{service_name}服务失败: {e}")
        
        self._services.clear()
        self._initialized = False
        logger.info("所有服务清理完成")
    
    async def get_service_status(self) -> Dict[str, Any]:
        """获取所有服务状态"""
        status = {
            "initialized": self._initialized,
            "services": {}
        }
        
        for service_name, service in self._services.items():
            try:
                if hasattr(service, 'health_check'):
                    service_status = await service.health_check()
                else:
                    service_status = {"status": "unknown", "message": "Health check not implemented"}
                
                status["services"][service_name] = service_status
                
            except Exception as e:
                status["services"][service_name] = {
                    "status": "error",
                    "message": str(e)
                }
        
        return status
    
    def get_service(self, service_name: str):
        """获取指定服务实例"""
        return self._services.get(service_name)
    
    @property
    def is_initialized(self) -> bool:
        """检查是否已初始化"""
        return self._initialized


# 全局服务管理器实例
service_manager = ServiceManager()


async def initialize_services():
    """初始化所有服务"""
    await service_manager.initialize_all_services()


async def cleanup_services():
    """清理所有服务"""
    await service_manager.cleanup_services()


async def get_service_status():
    """获取服务状态"""
    return await service_manager.get_service_status()


# 服务获取函数（向后兼容）
async def get_minio_service_instance() -> MinIOService:
    """获取MinIO服务实例"""
    return service_manager.get_service('minio') or await get_minio_service()


async def get_cache_service_instance() -> CacheService:
    """获取缓存服务实例"""
    return service_manager.get_service('cache') or await get_cache_service()


async def get_vector_service_instance() -> VectorService:
    """获取向量服务实例"""
    return service_manager.get_service('vector') or await get_vector_service()


async def get_document_service_instance() -> DocumentService:
    """获取文档服务实例"""
    return service_manager.get_service('document') or await get_document_service()


async def get_search_service_instance() -> SearchService:
    """获取搜索服务实例"""
    return service_manager.get_service('search') or await get_search_service()


async def get_task_service_instance() -> TaskService:
    """获取任务服务实例"""
    return service_manager.get_service('task') or await get_task_service() 
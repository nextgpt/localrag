"""
向量化任务处理器 - 参考RAG-Anything官方架构
负责从Redis队列中消费向量化任务，执行向量化并入库到Qdrant
"""

import asyncio
import json
import logging
from typing import Dict, Any, Optional
from datetime import datetime

from app.services.cache_service import get_cache_service
from app.services.document_service import get_document_service
from app.models.responses import ErrorCode
from app.core.exceptions import create_service_exception

logger = logging.getLogger("rag-anything")


class VectorizeWorker:
    """向量化任务处理器"""
    
    def __init__(self):
        self.cache_service = None
        self.document_service = None
        self.running = False
        self.queue_name = "document_vectorize"
        
    async def initialize(self):
        """初始化服务依赖"""
        self.cache_service = await get_cache_service()
        self.document_service = await get_document_service()
        logger.info("向量化任务处理器初始化完成")
        
    async def start(self):
        """启动任务处理器"""
        await self.initialize()
        self.running = True
        logger.info("🚀 向量化任务处理器启动，开始监听队列...")
        
        while self.running:
            try:
                # 1. 尝试获取优先级任务
                task_data = await self.cache_service.get_priority_task(self.queue_name)
                
                # 2. 如果没有优先级任务，获取普通任务  
                if not task_data:
                    task_json = await self.cache_service.redis.lpop(self.queue_name)
                    if task_json:
                        task_data = json.loads(task_json)
                
                # 3. 处理任务
                if task_data:
                    await self.process_task(task_data)
                else:
                    # 没有任务，等待1秒
                    await asyncio.sleep(1)
                    
            except Exception as e:
                logger.error(f"任务处理器运行异常: {e}")
                await asyncio.sleep(5)  # 出错时等待5秒后重试
                
    async def stop(self):
        """停止任务处理器"""
        self.running = False
        logger.info("向量化任务处理器已停止")
        
    async def process_task(self, task_data: Dict[str, Any]):
        """处理单个向量化任务"""
        task_id = task_data.get("task_id")
        file_id = task_data.get("file_id")
        filename = task_data.get("filename", "未知文件")
        
        try:
            logger.info(f"🔄 开始处理向量化任务: {task_id}")
            logger.info(f"   📄 文件: {filename} ({file_id})")
            
            # 1. 更新任务状态为运行中
            await self.update_task_status(task_id, file_id, "running", "开始向量化处理...")
            
            # 2. 执行向量化
            result = await self.document_service.vectorize_document(file_id)
            
            # 3. 更新任务状态为完成
            await self.update_task_status(task_id, file_id, "completed", "向量化完成", result)
            
            logger.info(f"✅ 向量化任务完成: {task_id}")
            logger.info(f"   📊 生成向量: {result.get('vector_count', 0)}个")
            logger.info(f"   📝 文本块: {result.get('chunk_count', 0)}个")
            
        except Exception as e:
            # 更新任务状态为失败
            error_msg = str(e)
            await self.update_task_status(task_id, file_id, "failed", f"向量化失败: {error_msg}")
            
            logger.error(f"❌ 向量化任务失败: {task_id}")
            logger.error(f"   🔍 错误: {error_msg}")
            
    async def update_task_status(
        self, 
        task_id: str, 
        file_id: str, 
        status: str, 
        message: str = "", 
        result: Optional[Dict[str, Any]] = None
    ):
        """更新任务和文件状态"""
        try:
            current_time = datetime.utcnow().isoformat()
            
            # 更新任务状态
            task_update = {
                "status": status,
                "updated_at": current_time,
                "message": message
            }
            
            if result:
                task_update["result"] = result
                
            await self.cache_service.set_task_info(task_id, task_update)
            
            # 更新文件向量化状态
            await self.cache_service.hset_field(f"file:{file_id}", "vectorize_status", status)
            await self.cache_service.hset_field(f"file:{file_id}", "vectorize_updated_at", current_time)
            
            if result:
                await self.cache_service.hset_field(f"file:{file_id}", "vector_count", result.get("vector_count", 0))
                await self.cache_service.hset_field(f"file:{file_id}", "chunk_count", result.get("chunk_count", 0))
                
        except Exception as e:
            logger.error(f"更新任务状态失败: {task_id} - {e}")


# 全局任务处理器实例
_vectorize_worker = None


async def get_vectorize_worker() -> VectorizeWorker:
    """获取向量化任务处理器实例"""
    global _vectorize_worker
    if _vectorize_worker is None:
        _vectorize_worker = VectorizeWorker()
    return _vectorize_worker


async def start_vectorize_worker():
    """启动向量化任务处理器（后台运行）"""
    worker = await get_vectorize_worker()
    # 在后台运行，不阻塞主进程
    asyncio.create_task(worker.start())
    logger.info("向量化任务处理器已在后台启动")


async def stop_vectorize_worker():
    """停止向量化任务处理器"""
    global _vectorize_worker
    if _vectorize_worker and _vectorize_worker.running:
        await _vectorize_worker.stop()
        logger.info("向量化任务处理器已停止") 
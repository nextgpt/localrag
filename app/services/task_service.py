"""
任务管理服务
负责管理系统中的异步任务，包括任务创建、状态跟踪、结果获取等
使用Redis进行任务状态的持久化存储
"""

import asyncio
import uuid
import time
from typing import Dict, Any, Optional, List, Callable, Awaitable
from datetime import datetime, timedelta
from enum import Enum
import logging
import json

from app.models.requests import TaskStatus
from app.models.responses import ErrorCode
from app.core.exceptions import create_task_exception
from app.core.config import settings
from app.services.cache_service import get_cache_service

logger = logging.getLogger("rag-anything")


class TaskInfo:
    """任务信息类"""
    
    def __init__(self, task_id: str, task_name: str, created_by: str = "system"):
        self.task_id = task_id
        self.task_name = task_name
        self.status = TaskStatus.PENDING
        self.created_at = datetime.utcnow()
        self.started_at: Optional[datetime] = None
        self.completed_at: Optional[datetime] = None
        self.created_by = created_by
        self.progress = 0
        self.result: Optional[Any] = None
        self.error: Optional[str] = None
        self.metadata: Dict[str, Any] = {}
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "TaskInfo":
        """从字典创建TaskInfo实例"""
        task_info = cls(
            task_id=data["task_id"],
            task_name=data["task_name"],
            created_by=data.get("created_by", "system")
        )
        
        task_info.status = data.get("status", TaskStatus.PENDING)
        
        # 解析时间字段 - 🔧 修复：处理Redis可能返回字符串"None"的情况
        if data.get("created_at") and data["created_at"] != "None":
            task_info.created_at = datetime.fromisoformat(data["created_at"])
        if data.get("started_at") and data["started_at"] != "None":
            task_info.started_at = datetime.fromisoformat(data["started_at"])
        if data.get("completed_at") and data["completed_at"] != "None":
            task_info.completed_at = datetime.fromisoformat(data["completed_at"])
            
        task_info.progress = data.get("progress", 0)
        task_info.result = data.get("result")
        task_info.error = data.get("error")
        task_info.metadata = data.get("metadata", {})
        
        return task_info
    
    def to_dict(self) -> Dict[str, Any]:
        """转换为字典格式"""
        return {
            "task_id": self.task_id,
            "task_name": self.task_name,
            "status": self.status,
            "created_at": self.created_at.isoformat(),
            "started_at": self.started_at.isoformat() if self.started_at else None,
            "completed_at": self.completed_at.isoformat() if self.completed_at else None,
            "created_by": self.created_by,
            "progress": self.progress,
            "result": self.result,
            "error": self.error,
            "metadata": self.metadata
        }


class TaskService:
    """任务管理服务"""
    
    def __init__(self):
        # 运行中的任务保持在内存中，用于取消等操作
        self.running_tasks: Dict[str, asyncio.Task] = {}
        self.cleanup_task: Optional[asyncio.Task] = None
        self.cache_service = None
        self._initialized = False
    
    async def initialize(self):
        """初始化任务服务"""
        if self._initialized:
            return
        
        await self._get_cache_service()
        await self.start_cleanup_task()
        self._initialized = True
        logger.info("任务管理服务初始化完成")
    
    async def cleanup(self):
        """清理任务服务"""
        await self.stop_cleanup_task()
        
        # 取消所有运行中的任务
        for task_id, task in self.running_tasks.items():
            if not task.done():
                task.cancel()
                logger.info(f"取消运行中的任务: {task_id}")
        
        # 等待所有任务完成
        if self.running_tasks:
            await asyncio.gather(*self.running_tasks.values(), return_exceptions=True)
        
        self.running_tasks.clear()
        self._initialized = False
        logger.info("任务管理服务清理完成")
    
    async def health_check(self) -> Dict[str, Any]:
        """健康检查"""
        try:
            cache_service = await self._get_cache_service()
            
            # 检查Redis连接
            await cache_service.ping()
            
            # 统计任务数量
            task_count = await self.get_task_count()
            
            return {
                "status": "healthy",
                "running_tasks": len(self.running_tasks),
                "task_counts": task_count,
                "cleanup_task_running": self.cleanup_task and not self.cleanup_task.done()
            }
        except Exception as e:
            return {
                "status": "unhealthy",
                "error": str(e)
            }
    
    async def _get_cache_service(self):
        """获取缓存服务"""
        if self.cache_service is None:
            self.cache_service = await get_cache_service()
        return self.cache_service
    
    async def start_cleanup_task(self):
        """启动清理任务"""
        if self.cleanup_task is None or self.cleanup_task.done():
            self.cleanup_task = asyncio.create_task(self._cleanup_loop())
            logger.info("任务清理程序已启动")
    
    async def stop_cleanup_task(self):
        """停止清理任务"""
        if self.cleanup_task and not self.cleanup_task.done():
            self.cleanup_task.cancel()
            try:
                await self.cleanup_task
            except asyncio.CancelledError:
                pass
            logger.info("任务清理程序已停止")
    
    async def _cleanup_loop(self):
        """清理任务循环"""
        while True:
            try:
                await asyncio.sleep(settings.TASK_CLEANUP_INTERVAL)
                await self._cleanup_old_tasks()
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"任务清理过程中发生错误: {e}")
    
    async def _cleanup_old_tasks(self):
        """清理过期任务"""
        try:
            cache_service = await self._get_cache_service()
            cutoff_time = datetime.utcnow() - timedelta(seconds=settings.TASK_MAX_RETENTION)
            
            # 获取所有任务keys
            keys = []
            cursor = 0
            while True:
                cursor, new_keys = await cache_service.redis.scan(cursor, match="task:*", count=100)
                keys.extend(new_keys)
                if cursor == 0:
                    break
            
            tasks_to_remove = []
            
            for key in keys:
                task_data = await cache_service.hgetall(key)
                if task_data and task_data.get("created_at"):
                    try:
                        created_at = datetime.fromisoformat(task_data["created_at"])
                        status = task_data.get("status")
                        
                        if (created_at < cutoff_time and 
                            status in [TaskStatus.COMPLETED, TaskStatus.FAILED]):
                            task_id = key.replace("task:", "")
                            tasks_to_remove.append(task_id)
                    except ValueError:
                        # 时间格式解析错误，跳过
                        continue
            
            for task_id in tasks_to_remove:
                await self.remove_task(task_id)
                logger.debug(f"已清理过期任务: {task_id}")
            
            if tasks_to_remove:
                logger.info(f"已清理 {len(tasks_to_remove)} 个过期任务")
                
        except Exception as e:
            logger.error(f"清理过期任务失败: {e}")
    
    async def create_task(
        self,
        task_func: Callable[..., Awaitable[Any]], 
        task_name: str,
        created_by: str = "system",
        *args,
        **kwargs
    ) -> str:
        """创建并启动异步任务"""
        task_id = str(uuid.uuid4())
        
        # 创建任务信息
        task_info = TaskInfo(task_id, task_name, created_by)
        
        # 保存到Redis
        cache_service = await self._get_cache_service()
        await cache_service.save_task(task_id, task_info.to_dict())
        
        # 启动异步任务
        async_task = asyncio.create_task(
            self._run_task(task_id, task_func, *args, **kwargs)
        )
        
        # 保存任务引用
        self.running_tasks[task_id] = async_task
        
        logger.info(f"任务已创建并启动: {task_id} - {task_name}")
        return task_id
    
    async def _run_task(
        self,
        task_id: str,
        task_func: Callable[..., Awaitable[Any]],
        *args,
        **kwargs
    ):
        """运行任务的内部方法"""
        cache_service = await self._get_cache_service()
        
        try:
            # 更新任务状态为运行中
            task_data = await cache_service.get_task(task_id)
            if task_data:
                task_info = TaskInfo.from_dict(task_data)
                task_info.status = TaskStatus.RUNNING
                task_info.started_at = datetime.utcnow()
                await cache_service.save_task(task_id, task_info.to_dict())
            
            # 执行任务
            result = await task_func(*args, **kwargs)
            
            # 更新任务状态为完成
            if task_data:
                task_info = TaskInfo.from_dict(task_data)
                task_info.status = TaskStatus.COMPLETED
                task_info.completed_at = datetime.utcnow()
                task_info.progress = 100
                task_info.result = result
                await cache_service.save_task(task_id, task_info.to_dict())
            
            logger.info(f"任务执行完成: {task_id}")
            
        except asyncio.CancelledError:
            # 任务被取消
            task_data = await cache_service.get_task(task_id)
            if task_data:
                task_info = TaskInfo.from_dict(task_data)
                task_info.status = TaskStatus.CANCELLED
                task_info.completed_at = datetime.utcnow()
                task_info.error = "任务被取消"
                await cache_service.save_task(task_id, task_info.to_dict())
            
            logger.info(f"任务被取消: {task_id}")
            raise
            
        except Exception as e:
            # 任务执行失败
            task_data = await cache_service.get_task(task_id)
            if task_data:
                task_info = TaskInfo.from_dict(task_data)
                task_info.status = TaskStatus.FAILED
                task_info.completed_at = datetime.utcnow()
                task_info.error = str(e)
                await cache_service.save_task(task_id, task_info.to_dict())
            
            logger.error(f"任务执行失败: {task_id} - {e}")
            
        finally:
            # 从运行任务列表中移除
            if task_id in self.running_tasks:
                del self.running_tasks[task_id]
    
    async def get_task_status(self, task_id: str) -> Optional[Dict[str, Any]]:
        """获取任务状态"""
        cache_service = await self._get_cache_service()
        return await cache_service.get_task(task_id)
    
    async def get_task_result(self, task_id: str) -> Any:
        """获取任务结果"""
        task_data = await self.get_task_status(task_id)
        if not task_data:
            raise create_task_exception(
                ErrorCode.TASK_NOT_FOUND,
                f"任务不存在: {task_id}"
            )
        
        task_info = TaskInfo.from_dict(task_data)
        
        if task_info.status == TaskStatus.PENDING:
            raise create_task_exception(
                ErrorCode.TASK_NOT_READY,
                f"任务尚未开始执行: {task_id}"
            )
        elif task_info.status == TaskStatus.RUNNING:
            raise create_task_exception(
                ErrorCode.TASK_NOT_READY,
                f"任务正在执行中: {task_id}"
            )
        elif task_info.status == TaskStatus.FAILED:
            raise create_task_exception(
                ErrorCode.TASK_FAILED,
                f"任务执行失败: {task_info.error}"
            )
        elif task_info.status == TaskStatus.CANCELLED:
            raise create_task_exception(
                ErrorCode.TASK_CANCELLED,
                f"任务已被取消: {task_id}"
            )
        
        return task_info.result
    
    async def cancel_task(self, task_id: str) -> bool:
        """取消任务"""
        # 检查任务是否存在
        task_data = await self.get_task_status(task_id)
        if not task_data:
            raise create_task_exception(
                ErrorCode.TASK_NOT_FOUND,
                f"任务不存在: {task_id}"
            )
        
        task_info = TaskInfo.from_dict(task_data)
        
        # 检查任务状态
        if task_info.status in [TaskStatus.COMPLETED, TaskStatus.FAILED, TaskStatus.CANCELLED]:
            raise create_task_exception(
                ErrorCode.TASK_NOT_CANCELLABLE,
                f"任务已完成，无法取消: {task_id}"
            )
        
        # 取消运行中的任务
        if task_id in self.running_tasks:
            async_task = self.running_tasks[task_id]
            if not async_task.done():
                async_task.cancel()
                logger.info(f"已取消运行中的任务: {task_id}")
                return True
        
        # 取消待执行的任务
        if task_info.status == TaskStatus.PENDING:
            cache_service = await self._get_cache_service()
            task_info.status = TaskStatus.CANCELLED
            task_info.completed_at = datetime.utcnow()
            task_info.error = "任务被手动取消"
            await cache_service.save_task(task_id, task_info.to_dict())
            logger.info(f"已取消待执行的任务: {task_id}")
            return True
        
        return False
    
    async def remove_task(self, task_id: str) -> bool:
        """删除任务记录"""
        try:
            # 先尝试取消任务（如果仍在运行）
            task_data = await self.get_task_status(task_id)
            if task_data:
                task_info = TaskInfo.from_dict(task_data)
                if task_info.status in [TaskStatus.PENDING, TaskStatus.RUNNING]:
                    await self.cancel_task(task_id)
            
            # 从Redis删除任务记录
            cache_service = await self._get_cache_service()
            await cache_service.delete(f"task:{task_id}")
            
            # 从运行任务列表移除
            if task_id in self.running_tasks:
                del self.running_tasks[task_id]
            
            logger.info(f"任务记录已删除: {task_id}")
            return True
            
        except Exception as e:
            logger.error(f"删除任务记录失败: {task_id} - {e}")
            return False
    
    async def list_tasks(
        self,
        status_filter: Optional[TaskStatus] = None,
        created_by: Optional[str] = None,
        limit: int = 10,
        offset: int = 0
    ) -> List[Dict[str, Any]]:
        """列出任务"""
        try:
            cache_service = await self._get_cache_service()
            
            # 获取所有任务keys
            keys = []
            cursor = 0
            while True:
                cursor, new_keys = await cache_service.redis.scan(cursor, match="task:*", count=100)
                keys.extend(new_keys)
                if cursor == 0:
                    break
            
            tasks = []
            for key in keys:
                task_data = await cache_service.hgetall(key)
                if task_data:
                    # 应用过滤条件
                    if status_filter and task_data.get("status") != status_filter:
                        continue
                    if created_by and task_data.get("created_by") != created_by:
                        continue
                    
                    tasks.append(task_data)
            
            # 按创建时间排序（最新的在前）
            tasks.sort(key=lambda x: x.get("created_at", ""), reverse=True)
            
            # 应用分页
            start = offset
            end = offset + limit
            return tasks[start:end]
            
        except Exception as e:
            logger.error(f"列出任务失败: {e}")
            return []
    
    async def get_task_count(self) -> Dict[str, int]:
        """获取各状态任务数量统计"""
        try:
            cache_service = await self._get_cache_service()
            
            # 获取所有任务keys
            keys = []
            cursor = 0
            while True:
                cursor, new_keys = await cache_service.redis.scan(cursor, match="task:*", count=100)
                keys.extend(new_keys)
                if cursor == 0:
                    break
            
            # 统计各状态数量
            counts = {
                "total": 0,
                "pending": 0,
                "running": 0,
                "completed": 0,
                "failed": 0,
                "cancelled": 0
            }
            
            for key in keys:
                task_data = await cache_service.hgetall(key)
                if task_data:
                    counts["total"] += 1
                    status = task_data.get("status", "")
                    if status in counts:
                        counts[status] += 1
            
            return counts
            
        except Exception as e:
            logger.error(f"获取任务统计失败: {e}")
            return {"total": 0, "pending": 0, "running": 0, "completed": 0, "failed": 0, "cancelled": 0}
    
    async def update_task_progress(self, task_id: str, progress: int, metadata: Dict[str, Any] = None):
        """更新任务进度"""
        cache_service = await self._get_cache_service()
        task_data = await cache_service.get_task(task_id)
        
        if task_data:
            task_info = TaskInfo.from_dict(task_data)
            task_info.progress = max(0, min(100, progress))  # 确保进度在0-100之间
            
            if metadata:
                task_info.metadata.update(metadata)
            
            await cache_service.save_task(task_id, task_info.to_dict())
            logger.debug(f"任务进度已更新: {task_id} - {progress}%")


# 全局任务服务实例
task_service = TaskService()


async def get_task_service() -> TaskService:
    """获取任务服务实例"""
    return task_service


# 向后兼容的TaskManager类
class TaskManager(TaskService):
    """任务管理器 - 向后兼容"""
    
    _instance = None
    _lock = asyncio.Lock()
    
    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance


async def get_task_manager() -> TaskManager:
    """获取任务管理器实例（向后兼容）"""
    return TaskManager() 
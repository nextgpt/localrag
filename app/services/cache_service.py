"""
Redis 缓存服务
负责任务状态管理、文件元数据缓存和会话管理
"""

import json
import asyncio
import logging
from typing import Any, Dict, List, Optional, Union
from datetime import datetime, timedelta
from urllib.parse import quote_plus

try:
    import redis.asyncio as aioredis
    from redis.asyncio import Redis
except ImportError:
    raise ImportError("请安装redis库: pip install redis")

from app.core.config import settings
from app.models.responses import ErrorCode
from app.core.exceptions import create_service_exception

logger = logging.getLogger("rag-anything")


class CacheService:
    """Redis 缓存服务"""
    
    def __init__(self):
        self.redis: Optional[Redis] = None
        self._connected = False
        
    async def initialize(self):
        """初始化Redis连接"""
        if self._connected:
            return
            
        try:
            # URL编码密码以处理特殊字符
            encoded_password = quote_plus(settings.REDIS_PASSWORD) if settings.REDIS_PASSWORD else ""
            
            # 构建Redis连接URL
            if encoded_password:
                redis_url = f"redis://:{encoded_password}@{settings.REDIS_HOST}:{settings.REDIS_PORT}/{settings.REDIS_DB}"
            else:
                redis_url = f"redis://{settings.REDIS_HOST}:{settings.REDIS_PORT}/{settings.REDIS_DB}"
            
            # 创建Redis连接池
            self.redis = aioredis.from_url(
                redis_url,
                encoding='utf-8',
                decode_responses=True,
                max_connections=20,
                retry_on_timeout=True
            )
            
            # 测试连接
            await self.redis.ping()
            
            self._connected = True
            logger.info(f"Redis 服务初始化成功，连接到 {settings.REDIS_HOST}:{settings.REDIS_PORT}")
            
        except Exception as e:
            logger.error(f"Redis 服务初始化失败: {e}")
            raise create_service_exception(
                ErrorCode.REDIS_CONNECTION_ERROR,
                f"Redis 连接失败: {str(e)}"
            )
    
    async def close(self):
        """关闭Redis连接"""
        if self.redis:
            await self.redis.close()
            self._connected = False
            logger.info("Redis 连接已关闭")
    
    # ===================
    # 基础操作
    # ===================
    
    async def set(self, key: str, value: Any, expire: Optional[int] = None) -> bool:
        """设置键值对"""
        if not self._connected:
            await self.initialize()
            
        try:
            # 序列化值
            if isinstance(value, (dict, list)):
                value = json.dumps(value, ensure_ascii=False)
            elif not isinstance(value, str):
                value = str(value)
                
            result = await self.redis.set(key, value, ex=expire)
            return result
            
        except Exception as e:
            logger.error(f"Redis SET 操作失败: {key} - {e}")
            return False
    
    async def get(self, key: str) -> Optional[str]:
        """获取值"""
        if not self._connected:
            await self.initialize()
            
        try:
            value = await self.redis.get(key)
            return value
            
        except Exception as e:
            logger.error(f"Redis GET 操作失败: {key} - {e}")
            return None
    
    async def get_json(self, key: str) -> Optional[Union[Dict, List]]:
        """获取JSON值"""
        value = await self.get(key)
        if value is None:
            return None
            
        try:
            return json.loads(value)
        except json.JSONDecodeError:
            logger.warning(f"无法解析JSON: {key} - {value}")
            return None
    
    async def delete(self, *keys: str) -> int:
        """删除键"""
        if not self._connected:
            await self.initialize()
            
        try:
            return await self.redis.delete(*keys)
        except Exception as e:
            logger.error(f"Redis DELETE 操作失败: {keys} - {e}")
            return 0
    
    async def exists(self, key: str) -> bool:
        """检查键是否存在"""
        if not self._connected:
            await self.initialize()
            
        try:
            return bool(await self.redis.exists(key))
        except Exception as e:
            logger.error(f"Redis EXISTS 操作失败: {key} - {e}")
            return False
    
    async def expire(self, key: str, seconds: int) -> bool:
        """设置过期时间"""
        if not self._connected:
            await self.initialize()
            
        try:
            return await self.redis.expire(key, seconds)
        except Exception as e:
            logger.error(f"Redis EXPIRE 操作失败: {key} - {e}")
            return False
    
    async def ttl(self, key: str) -> int:
        """获取过期时间"""
        if not self._connected:
            await self.initialize()
            
        try:
            return await self.redis.ttl(key)
        except Exception as e:
            logger.error(f"Redis TTL 操作失败: {key} - {e}")
            return -1
    
    # ===================
    # 哈希操作
    # ===================
    
    async def hset(self, name: str, mapping: Dict[str, Any]) -> int:
        """设置哈希字段"""
        if not self._connected:
            await self.initialize()
            
        try:
            # 序列化值
            serialized_mapping = {}
            for k, v in mapping.items():
                if isinstance(v, (dict, list)):
                    serialized_mapping[k] = json.dumps(v, ensure_ascii=False)
                else:
                    serialized_mapping[k] = str(v)
                    
            return await self.redis.hset(name, mapping=serialized_mapping)
            
        except Exception as e:
            logger.error(f"Redis HSET 操作失败: {name} - {e}")
            return 0
    
    async def hget(self, name: str, key: str) -> Optional[str]:
        """获取哈希字段值"""
        if not self._connected:
            await self.initialize()
            
        try:
            return await self.redis.hget(name, key)
        except Exception as e:
            logger.error(f"Redis HGET 操作失败: {name}.{key} - {e}")
            return None
    
    async def hgetall(self, name: str) -> Dict[str, str]:
        """获取所有哈希字段"""
        if not self._connected:
            await self.initialize()
            
        try:
            return await self.redis.hgetall(name)
        except Exception as e:
            logger.error(f"Redis HGETALL 操作失败: {name} - {e}")
            return {}
    
    async def hdel(self, name: str, *keys: str) -> int:
        """删除哈希字段"""
        if not self._connected:
            await self.initialize()
            
        try:
            return await self.redis.hdel(name, *keys)
        except Exception as e:
            logger.error(f"Redis HDEL 操作失败: {name}.{keys} - {e}")
            return 0
    
    async def hexists(self, name: str, key: str) -> bool:
        """检查哈希字段是否存在"""
        if not self._connected:
            await self.initialize()
            
        try:
            return bool(await self.redis.hexists(name, key))
        except Exception as e:
            logger.error(f"Redis HEXISTS 操作失败: {name}.{key} - {e}")
            return False
    
    async def hset_field(self, name: str, key: str, value: Any) -> int:
        if not self._connected:
            await self.initialize()
        try:
            if isinstance(value, (dict, list)):
                value = json.dumps(value, ensure_ascii=False)
            else:
                value = str(value)
            return await self.redis.hset(name, key, value)
        except Exception as e:
            logger.error(f"Redis hset_field 操作失败: {name} - {e}")
            return 0
    
    # ===================
    # 列表操作
    # ===================
    
    async def lpush(self, name: str, *values: Any) -> int:
        """左侧推入列表"""
        if not self._connected:
            await self.initialize()
            
        try:
            # 序列化值
            serialized_values = []
            for v in values:
                if isinstance(v, (dict, list)):
                    serialized_values.append(json.dumps(v, ensure_ascii=False))
                else:
                    serialized_values.append(str(v))
                    
            return await self.redis.lpush(name, *serialized_values)
            
        except Exception as e:
            logger.error(f"Redis LPUSH 操作失败: {name} - {e}")
            return 0
    
    async def rpush(self, name: str, *values: Any) -> int:
        """右侧推入列表"""
        if not self._connected:
            await self.initialize()
            
        try:
            # 序列化值
            serialized_values = []
            for v in values:
                if isinstance(v, (dict, list)):
                    serialized_values.append(json.dumps(v, ensure_ascii=False))
                else:
                    serialized_values.append(str(v))
                    
            return await self.redis.rpush(name, *serialized_values)
            
        except Exception as e:
            logger.error(f"Redis RPUSH 操作失败: {name} - {e}")
            return 0
    
    async def lpop(self, name: str) -> Optional[str]:
        """左侧弹出列表元素"""
        if not self._connected:
            await self.initialize()
            
        try:
            return await self.redis.lpop(name)
        except Exception as e:
            logger.error(f"Redis LPOP 操作失败: {name} - {e}")
            return None
    
    async def rpop(self, name: str) -> Optional[str]:
        """右侧弹出列表元素"""
        if not self._connected:
            await self.initialize()
            
        try:
            return await self.redis.rpop(name)
        except Exception as e:
            logger.error(f"Redis RPOP 操作失败: {name} - {e}")
            return None
    
    async def lrange(self, name: str, start: int, end: int) -> List[str]:
        """获取列表范围"""
        if not self._connected:
            await self.initialize()
            
        try:
            return await self.redis.lrange(name, start, end)
        except Exception as e:
            logger.error(f"Redis LRANGE 操作失败: {name} - {e}")
            return []
    
    async def llen(self, name: str) -> int:
        """获取列表长度"""
        if not self._connected:
            await self.initialize()
            
        try:
            return await self.redis.llen(name)
        except Exception as e:
            logger.error(f"Redis LLEN 操作失败: {name} - {e}")
            return 0
    
    # ===================
    # 特定业务操作
    # ===================
    
    async def save_task(self, task_id: str, task_data: Dict[str, Any], expire: int = 86400) -> bool:
        """保存任务信息"""
        task_key = f"task:{task_id}"
        
        # 添加时间戳
        task_data["updated_at"] = datetime.now().isoformat()
        
        result = await self.hset(task_key, task_data)
        if result and expire > 0:
            await self.expire(task_key, expire)
            
        return result > 0
    
    async def get_task(self, task_id: str) -> Optional[Dict[str, Any]]:
        """获取任务信息"""
        task_key = f"task:{task_id}"
        task_data = await self.hgetall(task_key)
        
        if not task_data:
            return None
            
        # 解析JSON字段
        for key, value in task_data.items():
            if key in ["metadata", "result", "error_details"]:
                try:
                    task_data[key] = json.loads(value) if value else None
                except json.JSONDecodeError:
                    pass
                    
        return task_data
    
    async def update_task_status(self, task_id: str, status: str, **kwargs) -> bool:
        """更新任务状态"""
        task_key = f"task:{task_id}"
        
        update_data = {
            "status": status,
            "updated_at": datetime.now().isoformat()
        }
        update_data.update(kwargs)
        
        return await self.hset(task_key, update_data) > 0
    
    async def save_file_metadata(self, file_id: str, metadata: Dict[str, Any], expire: int = 2592000) -> bool:
        """保存文件元数据 (默认30天过期)"""
        file_key = f"file:{file_id}"
        
        metadata["updated_at"] = datetime.now().isoformat()
        
        result = await self.hset(file_key, metadata)
        if result and expire > 0:
            await self.expire(file_key, expire)
            
        return result > 0
    
    async def get_file_metadata(self, file_id: str) -> Optional[Dict[str, Any]]:
        """获取文件元数据"""
        file_key = f"file:{file_id}"
        metadata = await self.hgetall(file_key)
        
        if not metadata:
            return None
            
        # 解析JSON字段
        for key, value in metadata.items():
            if key in ["tags", "custom_fields", "parse_result"]:
                try:
                    metadata[key] = json.loads(value) if value else None
                except json.JSONDecodeError:
                    pass
                    
        return metadata
    
    async def add_task_to_queue(self, queue_name: str, task_id: str) -> bool:
        """添加任务到队列"""
        queue_key = f"queue:{queue_name}"
        try:
            await self.rpush(queue_key, task_id)
            return True
        except Exception as e:
            logger.error(f"添加任务到队列失败: {queue_name} - {task_id} - {e}")
            return False
    
    async def get_task_from_queue(self, queue_name: str) -> Optional[str]:
        """从队列获取任务"""
        queue_key = f"queue:{queue_name}"
        return await self.lpop(queue_key)
    
    async def get_queue_length(self, queue_name: str) -> int:
        """获取队列长度"""
        queue_key = f"queue:{queue_name}"
        return await self.llen(queue_key)
    
    async def increment_counter(self, key: str, amount: int = 1) -> int:
        """递增计数器"""
        if not self._connected:
            await self.initialize()
            
        try:
            return await self.redis.incrby(key, amount)
        except Exception as e:
            logger.error(f"Redis INCRBY 操作失败: {key} - {e}")
            return 0
    
    async def health_check(self) -> bool:
        """健康检查"""
        try:
            await self.redis.ping()
            return True
        except:
            return False
    
    # ===================
    # 高级任务队列操作 - 参考mineru-web的异步机制
    # ===================
    
    async def add_priority_task(self, queue_name: str, task_data: Dict[str, Any], priority: int = 0) -> bool:
        """添加优先级任务到队列 - 参考mineru-web的任务调度"""
        try:
            # 使用有序集合实现优先级队列
            priority_queue = f"{queue_name}:priority"
            task_json = json.dumps(task_data, ensure_ascii=False)
            
            # 优先级越高，分数越小（先处理）
            score = -priority  
            await self.redis.zadd(priority_queue, {task_json: score})
            
            logger.info(f"添加优先级任务: {queue_name} - 优先级{priority}")
            return True
            
        except Exception as e:
            logger.error(f"添加优先级任务失败: {queue_name} - {e}")
            return False
    
    async def get_priority_task(self, queue_name: str) -> Optional[Dict[str, Any]]:
        """获取最高优先级任务"""
        try:
            priority_queue = f"{queue_name}:priority"
            
            # 获取最高优先级的任务（分数最小）
            result = await self.redis.zpopmin(priority_queue)
            if result:
                task_json, score = result[0]
                return json.loads(task_json)
            
            return None
            
        except Exception as e:
            logger.error(f"获取优先级任务失败: {queue_name} - {e}")
            return None
    
    async def get_queue_stats(self, queue_name: str) -> Dict[str, int]:
        """获取队列统计信息 - 类似mineru-web的任务监控"""
        try:
            regular_queue = f"queue:{queue_name}"
            priority_queue = f"{queue_name}:priority"
            
            stats = {
                "pending_tasks": await self.llen(regular_queue),
                "priority_tasks": await self.redis.zcard(priority_queue),
                "running_tasks": 0,
                "completed_tasks": 0,
                "failed_tasks": 0
            }
            
            # 统计不同状态的任务
            pattern = f"task:*:status"
            async for key in self.redis.scan_iter(match=pattern):
                status = await self.redis.get(key)
                if status == "running":
                    stats["running_tasks"] += 1
                elif status == "completed":
                    stats["completed_tasks"] += 1
                elif status == "failed":
                    stats["failed_tasks"] += 1
            
            return stats
            
        except Exception as e:
            logger.error(f"获取队列统计失败: {queue_name} - {e}")
            return {"pending_tasks": 0, "priority_tasks": 0, "running_tasks": 0, "completed_tasks": 0, "failed_tasks": 0}
    
    async def batch_update_tasks(self, task_updates: List[Dict[str, Any]]) -> int:
        """批量更新任务状态 - 提高性能"""
        try:
            success_count = 0
            
            # 使用管道批量更新
            pipe = self.redis.pipeline()
            
            for update in task_updates:
                task_id = update["task_id"]
                status = update["status"]
                
                # 更新任务状态
                pipe.hset(f"task:{task_id}", "status", status)
                
                # 更新时间戳
                pipe.hset(f"task:{task_id}", "updated_at", datetime.utcnow().isoformat())
                
                # 如果有额外数据，也一并更新
                if "result" in update:
                    pipe.hset(f"task:{task_id}", "result", json.dumps(update["result"], ensure_ascii=False))
                
                if "error" in update:
                    pipe.hset(f"task:{task_id}", "error", update["error"])
            
            await pipe.execute()
            success_count = len(task_updates)
            
            logger.info(f"批量更新任务完成: {success_count}个任务")
            return success_count
            
        except Exception as e:
            logger.error(f"批量更新任务失败: {e}")
            return 0
    
    async def setup_task_retry(self, task_id: str, max_retries: int = 3, delay: int = 60) -> bool:
        """设置任务重试机制 - 参考mineru-web的错误恢复"""
        try:
            retry_key = f"task:{task_id}:retry"
            retry_data = {
                "max_retries": max_retries,
                "current_retries": 0,
                "delay": delay,
                "next_retry": (datetime.utcnow() + timedelta(seconds=delay)).isoformat()
            }
            
            await self.redis.hset(retry_key, mapping=retry_data)
            await self.redis.expire(retry_key, max_retries * delay + 3600)  # 额外1小时缓冲
            
            return True
            
        except Exception as e:
            logger.error(f"设置任务重试失败: {task_id} - {e}")
            return False
    
    async def get_failed_tasks_for_retry(self, queue_name: str) -> List[Dict[str, Any]]:
        """获取需要重试的失败任务"""
        try:
            retry_tasks = []
            current_time = datetime.utcnow()
            
            # 扫描重试队列
            pattern = f"task:*:retry"
            async for key in self.redis.scan_iter(match=pattern):
                retry_data = await self.redis.hgetall(key)
                
                if retry_data:
                    next_retry = datetime.fromisoformat(retry_data.get("next_retry", ""))
                    current_retries = int(retry_data.get("current_retries", 0))
                    max_retries = int(retry_data.get("max_retries", 3))
                    
                    # 检查是否到了重试时间且未超过最大重试次数
                    if current_time >= next_retry and current_retries < max_retries:
                        task_id = key.split(":")[1]  # 从 task:xxx:retry 中提取任务ID
                        retry_tasks.append({
                            "task_id": task_id,
                            "current_retries": current_retries,
                            "max_retries": max_retries,
                            "delay": int(retry_data.get("delay", 60))
                        })
            
            return retry_tasks
            
        except Exception as e:
            logger.error(f"获取重试任务失败: {queue_name} - {e}")
            return []
    
    async def set_task_info(self, task_id: str, task_data: Dict[str, Any], expire: int = 86400) -> bool:
        """保存任务信息到Redis（task:{task_id}）"""
        if not self._connected:
            await self.initialize()
        try:
            # 任务信息全部序列化为字符串
            serialized_data = {}
            for k, v in task_data.items():
                if isinstance(v, (dict, list)):
                    serialized_data[k] = json.dumps(v, ensure_ascii=False)
                else:
                    serialized_data[k] = str(v)
            await self.redis.hset(f"task:{task_id}", mapping=serialized_data)
            await self.redis.expire(f"task:{task_id}", expire)
            return True
        except Exception as e:
            logger.error(f"Redis set_task_info 操作失败: {task_id} - {e}")
            return False
    
    async def add_to_queue(self, queue_name: str, task_data: dict) -> bool:
        """将任务数据推入Redis队列（RPUSH）"""
        if not self._connected:
            await self.initialize()
        try:
            await self.redis.rpush(queue_name, json.dumps(task_data, ensure_ascii=False))
            return True
        except Exception as e:
            logger.error(f"Redis add_to_queue 操作失败: {queue_name} - {e}")
            return False

    # ===================
    # 数据操作便利方法 - 用于知识库管理
    # ===================
    
    async def save_data(self, key: str, data: Any, expire: Optional[int] = None) -> bool:
        """保存数据（支持字典、列表等复杂类型）"""
        try:
            if isinstance(data, (dict, list)):
                serialized_data = json.dumps(data, ensure_ascii=False)
            else:
                serialized_data = str(data)
            
            return await self.set(key, serialized_data, expire)
        except Exception as e:
            logger.error(f"保存数据失败: {key} - {e}")
            return False
    
    async def get_data(self, key: str) -> Optional[Any]:
        """获取数据（自动反序列化JSON）"""
        try:
            value = await self.get(key)
            if value is None:
                return None
            
            # 尝试解析为JSON
            try:
                return json.loads(value)
            except json.JSONDecodeError:
                # 如果不是JSON，返回原始字符串
                return value
                
        except Exception as e:
            logger.error(f"获取数据失败: {key} - {e}")
            return None
    
    async def delete_data(self, key: str) -> bool:
        """删除数据"""
        try:
            result = await self.delete(key)
            return result > 0
        except Exception as e:
            logger.error(f"删除数据失败: {key} - {e}")
            return False


# 全局缓存服务实例
cache_service = CacheService()


async def get_cache_service() -> CacheService:
    """获取缓存服务实例"""
    return cache_service 
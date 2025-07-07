"""
MinIO 对象存储服务
负责文件的分布式存储和管理
"""

import os
import asyncio
import logging
from typing import Dict, List, Optional, BinaryIO, Any
from datetime import datetime, timedelta
import uuid

try:
    from minio import Minio
    from minio.error import S3Error
except ImportError:
    raise ImportError("请安装minio库: pip install minio")

from app.core.config import settings
from app.models.responses import ErrorCode
from app.core.exceptions import create_service_exception

logger = logging.getLogger("rag-anything")


class MinIOService:
    """MinIO 对象存储服务"""
    
    def __init__(self):
        self.client: Optional[Minio] = None
        self.bucket_name = settings.MINIO_BUCKET_NAME
        self._connected = False
        
    async def initialize(self):
        """初始化MinIO连接"""
        if self._connected:
            return
            
        try:
            # 创建MinIO客户端
            self.client = Minio(
                f"{settings.MINIO_HOST}:{settings.MINIO_PORT}",
                access_key=settings.MINIO_ACCESS_KEY,
                secret_key=settings.MINIO_SECRET_KEY,
                secure=settings.MINIO_SECURE
            )
            
            # 检查连接是否正常
            await self._check_connection()
            
            # 确保存储桶存在
            await self._ensure_bucket_exists()
            
            self._connected = True
            logger.info(f"MinIO 服务初始化成功，连接到 {settings.MINIO_HOST}:{settings.MINIO_PORT}")
            
        except Exception as e:
            logger.error(f"MinIO 服务初始化失败: {e}")
            raise create_service_exception(
                ErrorCode.MINIO_CONNECTION_ERROR,
                f"MinIO 连接失败: {str(e)}"
            )
    
    async def _check_connection(self):
        """检查MinIO连接"""
        loop = asyncio.get_event_loop()
        
        def _sync_check():
            # 尝试列出存储桶来验证连接
            list(self.client.list_buckets())
            
        await loop.run_in_executor(None, _sync_check)
    
    async def _ensure_bucket_exists(self):
        """确保存储桶存在"""
        loop = asyncio.get_event_loop()
        
        def _sync_ensure_bucket():
            if not self.client.bucket_exists(self.bucket_name):
                self.client.make_bucket(self.bucket_name)
                logger.info(f"创建存储桶: {self.bucket_name}")
            else:
                logger.info(f"存储桶已存在: {self.bucket_name}")
                
        await loop.run_in_executor(None, _sync_ensure_bucket)
    
    async def upload_file(self, object_name: str, file_data: bytes, content_type: str = None) -> str:
        """上传文件到MinIO"""
        if not self._connected:
            await self.initialize()
            
        try:
            loop = asyncio.get_event_loop()
            
            def _sync_upload():
                from io import BytesIO
                
                # 🔧 修复作用域问题：在内部函数开始就确定content_type的值
                final_content_type = content_type
                if not final_content_type:
                    ext = os.path.splitext(object_name)[1].lower()
                    content_type_map = {
                        '.pdf': 'application/pdf',
                        '.doc': 'application/msword',
                        '.docx': 'application/vnd.openxmlformats-officedocument.wordprocessingml.document',
                        '.txt': 'text/plain',
                        '.md': 'text/markdown',
                        '.jpg': 'image/jpeg',
                        '.jpeg': 'image/jpeg',
                        '.png': 'image/png',
                    }
                    final_content_type = content_type_map.get(ext, 'application/octet-stream')
                
                # 调试日志
                logger.debug(f"MinIO上传: object_name={object_name}, content_type={content_type} -> final_content_type={final_content_type}")
                
                # 上传文件
                self.client.put_object(
                    bucket_name=self.bucket_name,
                    object_name=object_name,
                    data=BytesIO(file_data),
                    length=len(file_data),
                    content_type=final_content_type
                )
                
                return f"minio://{self.bucket_name}/{object_name}"
            
            file_url = await loop.run_in_executor(None, _sync_upload)
            logger.info(f"文件上传成功: {object_name}")
            return file_url
            
        except Exception as e:
            logger.error(f"文件上传失败: {object_name} - {e}")
            raise create_service_exception(
                ErrorCode.MINIO_CONNECTION_ERROR,
                f"文件上传失败: {str(e)}"
            )
    
    async def download_file(self, object_name: str) -> bytes:
        """从MinIO下载文件"""
        if not self._connected:
            await self.initialize()
            
        try:
            loop = asyncio.get_event_loop()
            
            def _sync_download():
                response = self.client.get_object(self.bucket_name, object_name)
                try:
                    return response.read()
                finally:
                    response.close()
                    response.release_conn()
            
            file_data = await loop.run_in_executor(None, _sync_download)
            logger.debug(f"文件下载成功: {object_name}")
            return file_data
            
        except Exception as e:
            logger.error(f"文件下载失败: {object_name} - {e}")
            raise create_service_exception(
                ErrorCode.MINIO_CONNECTION_ERROR,
                f"文件下载失败: {str(e)}"
            )
    
    async def delete_file(self, object_name: str) -> bool:
        """从MinIO删除文件"""
        if not self._connected:
            await self.initialize()
            
        try:
            loop = asyncio.get_event_loop()
            
            def _sync_delete():
                self.client.remove_object(self.bucket_name, object_name)
                
            await loop.run_in_executor(None, _sync_delete)
            logger.info(f"文件删除成功: {object_name}")
            return True
            
        except Exception as e:
            logger.error(f"文件删除失败: {object_name} - {e}")
            return False
    
    async def file_exists(self, object_name: str) -> bool:
        """检查文件是否存在"""
        if not self._connected:
            await self.initialize()
            
        try:
            loop = asyncio.get_event_loop()
            
            def _sync_stat():
                try:
                    self.client.stat_object(self.bucket_name, object_name)
                    return True
                except S3Error as e:
                    if e.code == "NoSuchKey":
                        return False
                    raise
                    
            exists = await loop.run_in_executor(None, _sync_stat)
            return exists
            
        except Exception as e:
            logger.error(f"检查文件存在性失败: {object_name} - {e}")
            return False
    
    async def get_file_info(self, object_name: str) -> Optional[Dict]:
        """获取文件信息"""
        if not self._connected:
            await self.initialize()
            
        try:
            loop = asyncio.get_event_loop()
            
            def _sync_stat():
                stat = self.client.stat_object(self.bucket_name, object_name)
                return {
                    "object_name": object_name,
                    "size": stat.size,
                    "etag": stat.etag,
                    "last_modified": stat.last_modified.isoformat(),
                    "content_type": stat.content_type,
                    "metadata": stat.metadata
                }
                
            file_info = await loop.run_in_executor(None, _sync_stat)
            return file_info
            
        except Exception as e:
            logger.error(f"获取文件信息失败: {object_name} - {e}")
            return None
    
    async def list_files(self, prefix: str = "", limit: int = 100) -> List[Dict]:
        """列出文件"""
        if not self._connected:
            await self.initialize()
            
        try:
            loop = asyncio.get_event_loop()
            
            def _sync_list():
                objects = self.client.list_objects(
                    self.bucket_name, 
                    prefix=prefix, 
                    recursive=True
                )
                
                files = []
                count = 0
                for obj in objects:
                    if count >= limit:
                        break
                        
                    files.append({
                        "object_name": obj.object_name,
                        "size": obj.size,
                        "etag": obj.etag,
                        "last_modified": obj.last_modified.isoformat(),
                        "is_dir": obj.is_dir
                    })
                    count += 1
                    
                return files
                
            files = await loop.run_in_executor(None, _sync_list)
            return files
            
        except Exception as e:
            logger.error(f"列出文件失败: {e}")
            return []
    
    async def generate_presigned_url(self, object_name: str, expires: timedelta = None) -> str:
        """生成预签名URL"""
        if not self._connected:
            await self.initialize()
            
        if expires is None:
            expires = timedelta(hours=1)  # 默认1小时过期
            
        try:
            loop = asyncio.get_event_loop()
            
            def _sync_presign():
                return self.client.presigned_get_object(
                    bucket_name=self.bucket_name,
                    object_name=object_name,
                    expires=expires
                )
                
            url = await loop.run_in_executor(None, _sync_presign)
            return url
            
        except Exception as e:
            logger.error(f"生成预签名URL失败: {object_name} - {e}")
            raise create_service_exception(
                ErrorCode.MINIO_CONNECTION_ERROR,
                f"生成预签名URL失败: {str(e)}"
            )
    
    async def health_check(self) -> bool:
        """健康检查"""
        try:
            await self._check_connection()
            return True
        except:
            return False
    
    async def create_bucket_policy(self, bucket_name: str, policy_type: str = "read"):
        """设置存储桶策略 - 参考mineru-web的公共访问设置"""
        try:
            if policy_type == "public_read":
                # 设置公共读权限，便于文件分享和预览
                policy = {
                    "Version": "2012-10-17",
                    "Statement": [
                        {
                            "Effect": "Allow",
                            "Principal": {"AWS": "*"},
                            "Action": ["s3:GetObject"],
                            "Resource": [f"arn:aws:s3:::{bucket_name}/*"]
                        }
                    ]
                }
                
                import json
                self.client.set_bucket_policy(bucket_name, json.dumps(policy))
                logger.info(f"存储桶 {bucket_name} 设置为公共读权限")
            
        except Exception as e:
            logger.warning(f"设置存储桶策略失败: {bucket_name} - {e}")
    
    async def get_file_url(self, object_name: str, expires: int = 3600) -> str:
        """获取文件预签名URL - 支持文件预览和分享"""
        try:
            from datetime import timedelta
            
            # 生成预签名URL，支持临时访问
            url = self.client.presigned_get_object(
                bucket_name=self.bucket_name,
                object_name=object_name,
                expires=timedelta(seconds=expires)
            )
            return url
            
        except Exception as e:
            logger.error(f"生成预签名URL失败: {object_name} - {e}")
            raise create_service_exception(
                ErrorCode.MINIO_CONNECTION_ERROR,
                f"生成文件访问URL失败: {str(e)}"
            )
    
    async def batch_upload_files(self, files_data: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """批量上传文件 - 参考mineru-web的批量处理机制"""
        results = []
        
        for file_data in files_data:
            try:
                object_name = file_data["object_name"]
                file_content = file_data["file_data"]
                content_type = file_data.get("content_type")
                
                file_url = await self.upload_file(object_name, file_content, content_type)
                
                results.append({
                    "object_name": object_name,
                    "success": True,
                    "file_url": file_url,
                    "size": len(file_content)
                })
                
            except Exception as e:
                results.append({
                    "object_name": file_data.get("object_name", "unknown"),
                    "success": False,
                    "error": str(e)
                })
                logger.error(f"批量上传失败: {file_data.get('object_name')} - {e}")
        
        logger.info(f"批量上传完成: 成功{sum(1 for r in results if r['success'])}个")
        return results
    
    async def get_file_categories(self) -> Dict[str, int]:
        """获取文件分类统计 - 类似mineru-web的文件管理界面"""
        try:
            objects = self.client.list_objects(self.bucket_name, recursive=True)
            categories = {
                "documents": 0,  # PDF, Word, Excel等
                "images": 0,     # 图片文件
                "texts": 0,      # 纯文本文件
                "parsed": 0,     # 已解析的文件
                "others": 0
            }
            
            for obj in objects:
                file_ext = obj.object_name.split('.')[-1].lower()
                if file_ext in ['pdf', 'doc', 'docx', 'ppt', 'pptx', 'xls', 'xlsx']:
                    categories["documents"] += 1
                elif file_ext in ['jpg', 'jpeg', 'png', 'gif', 'bmp', 'tiff', 'webp']:
                    categories["images"] += 1
                elif file_ext in ['txt', 'md', 'csv']:
                    categories["texts"] += 1
                elif 'parsed' in obj.object_name:
                    categories["parsed"] += 1
                else:
                    categories["others"] += 1
            
            return categories
            
        except Exception as e:
            logger.error(f"获取文件分类失败: {e}")
            return {"documents": 0, "images": 0, "texts": 0, "parsed": 0, "others": 0}


# 全局MinIO服务实例
minio_service = MinIOService()


async def get_minio_service() -> MinIOService:
    """获取MinIO服务实例"""
    return minio_service 
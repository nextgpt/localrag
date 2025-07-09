"""
文档处理服务
负责文档的上传、解析、向量化和管理，使用分布式存储架构
"""

import os
import uuid
import logging
from typing import Dict, List, Optional, Any, Tuple
from datetime import datetime
import asyncio
from pathlib import Path

from app.core.config import settings
from app.models.responses import ErrorCode
from app.core.exceptions import create_service_exception
from app.services.storage_service import get_minio_service
from app.services.cache_service import get_cache_service
from app.services.vector_service import get_vector_service

# 先定义logger
logger = logging.getLogger("rag-anything")

# RAGAnything 相关导入
try:
    from raganything import RAGAnything
    from raganything.modalprocessors import (
        ImageModalProcessor, 
        TableModalProcessor, 
        EquationModalProcessor, 
        GenericModalProcessor
    )
except ImportError:
    logger.warning("RAGAnything 未安装，部分功能可能不可用")
    RAGAnything = None
    ImageModalProcessor = None
    TableModalProcessor = None
    EquationModalProcessor = None
    GenericModalProcessor = None


class DocumentService:
    """文档处理服务"""
    
    def __init__(self):
        self.minio_service = None
        self.cache_service = None
        self.vector_service = None
        self.rag_processor = None
    
    async def _get_services(self):
        """获取依赖服务"""
        if self.minio_service is None:
            self.minio_service = await get_minio_service()
        if self.cache_service is None:
            self.cache_service = await get_cache_service()
        if self.vector_service is None:
            self.vector_service = await get_vector_service()
        
        # 初始化RAG处理器
        if self.rag_processor is None and RAGAnything is not None:
            try:
                # RAGAnything需要模型函数，不是API参数
                # 暂时使用None，等后续需要时再配置具体的模型函数
                self.rag_processor = RAGAnything(
                    llm_model_func=None,  # 需要时再配置具体的LLM函数
                    embedding_func=None,  # 需要时再配置具体的嵌入函数  
                    vision_model_func=None,  # 需要时再配置具体的视觉函数
                    config=None  # 使用默认配置
                )
                logger.info("RAGAnything 处理器初始化成功")
            except Exception as e:
                logger.error(f"RAGAnything 处理器初始化失败: {e}")
                # 如果初始化失败，设置为None，不影响基本的文件上传功能
                self.rag_processor = None
    
    async def upload_file(
        self, 
        file_content: bytes = None,
        filename: str = None, 
        content_type: str = None,
        metadata: Optional[Dict[str, Any]] = None,
        original_name: Optional[str] = None,
        description: Optional[str] = None,
        **kwargs  # 接收额外参数，提高兼容性
    ) -> str:
        """上传文件到MinIO"""
        # 💡 添加方法入口日志 - 立即记录所有传入参数
        logger.info(f"🔍 upload_file 方法被调用！")
        logger.info(f"📋 传入参数详情:")
        logger.info(f"  - file_content: {'存在' if file_content else '缺失'} (长度: {len(file_content) if file_content else 0})")
        logger.info(f"  - filename: '{filename}' (类型: {type(filename)})")
        logger.info(f"  - content_type: '{content_type}'")
        logger.info(f"  - original_name: '{original_name}'")
        logger.info(f"  - description: '{description}'")
        logger.info(f"  - metadata: {metadata}")
        logger.info(f"  - kwargs: {kwargs}")
        
        # 参数验证和兼容性处理 - 参考mineru-web的实现
        if not file_content:  # 检查None或空内容
            # 尝试从kwargs获取
            file_content = kwargs.get('data', kwargs.get('file_data'))
            if not file_content:  # 检查None或空内容
                raise create_service_exception(
                    ErrorCode.INVALID_REQUEST,
                    "缺少文件内容参数 (file_content)"
                )
        
        if not filename:  # 检查None或空字符串
            # 尝试从kwargs获取
            filename = kwargs.get('name', kwargs.get('file_name'))
            if not filename:  # 检查None或空字符串
                raise create_service_exception(
                    ErrorCode.INVALID_REQUEST,
                    "缺少文件名参数 (filename)"
                )
        
        # 如果还没有content_type，尝试从kwargs获取
        if content_type is None:
            content_type = kwargs.get('content_type', kwargs.get('mime_type'))
        
        # 调试日志 - 记录参数信息
        logger.info(f"upload_file 调用参数: filename='{filename}', content_type='{content_type}', "
                    f"file_size={len(file_content) if file_content else 0}, "
                    f"original_name='{original_name}', description='{description}', "
                    f"kwargs={list(kwargs.keys()) if kwargs else []}")
        
        await self._get_services()
        
        # 生成文件ID
        file_id = str(uuid.uuid4())
        file_extension = Path(filename).suffix.lower()
        
        # 生成对象名（包含路径结构）
        upload_date = datetime.now().strftime("%Y/%m/%d")
        object_name = f"documents/{upload_date}/{file_id}{file_extension}"
        
        try:
            # 上传到MinIO
            file_url = await self.minio_service.upload_file(
                object_name=object_name,
                file_data=file_content,
                content_type=content_type
            )
            
            # 准备文件元数据
            display_name = original_name or filename
            
            # 处理description参数
            final_metadata = metadata or {}
            if description:
                final_metadata = {**final_metadata, "description": description}
            
            file_metadata = {
                "file_id": file_id,
                "filename": filename,
                "original_name": display_name,
                "object_name": object_name,
                "file_url": file_url,
                "file_size": len(file_content),
                "content_type": content_type,
                "file_extension": file_extension,
                "upload_date": datetime.now().isoformat(),
                "status": "uploaded",
                "parse_status": "pending",
                "custom_metadata": final_metadata
            }
            
            # 保存元数据到Redis
            await self.cache_service.save_file_metadata(file_id, file_metadata)
            
            logger.info(f"文件上传成功: {filename} -> {file_id}")
            return file_id
            
        except Exception as e:
            logger.error(f"文件上传失败: {filename} - {e}")
            raise create_service_exception(
                ErrorCode.FILE_UPLOAD_FAILED,
                f"文件上传失败: {str(e)}"
            )
    
    async def get_file_info(self, file_id: str) -> Optional[Dict[str, Any]]:
        """获取文件信息"""
        await self._get_services()
        
        # 从Redis获取文件元数据
        metadata = await self.cache_service.get_file_metadata(file_id)
        if not metadata:
            return None
        
        # 补充MinIO中的实时信息
        try:
            object_name = metadata.get("object_name")
            if object_name:
                minio_info = await self.minio_service.get_file_info(object_name)
                if minio_info:
                    metadata.update({
                        "minio_etag": minio_info.get("etag"),
                        "minio_last_modified": minio_info.get("last_modified"),
                        "actual_size": minio_info.get("size")
                    })
        except Exception as e:
            logger.warning(f"获取MinIO文件信息失败: {file_id} - {e}")
        
        return metadata
    
    async def download_file(self, file_id: str) -> Tuple[bytes, Dict[str, Any]]:
        """下载文件"""
        await self._get_services()
        
        # 获取文件元数据
        metadata = await self.get_file_info(file_id)
        if not metadata:
            raise create_service_exception(
                ErrorCode.FILE_NOT_FOUND,
                f"文件不存在: {file_id}"
            )
        
        object_name = metadata.get("object_name")
        if not object_name:
            raise create_service_exception(
                ErrorCode.FILE_NOT_FOUND,
                f"文件对象名不存在: {file_id}"
            )
        
        try:
            # 从MinIO下载文件
            file_data = await self.minio_service.download_file(object_name)
            return file_data, metadata
            
        except Exception as e:
            logger.error(f"文件下载失败: {file_id} - {e}")
            raise create_service_exception(
                ErrorCode.FILE_DOWNLOAD_FAILED,
                f"文件下载失败: {str(e)}"
            )
    
    async def delete_file(
        self, 
        file_id: str, 
        delete_parsed_data: bool = True, 
        delete_vector_data: bool = True
    ) -> bool:
        """删除文件"""
        await self._get_services()
        
        try:
            # 获取文件元数据
            metadata = await self.get_file_info(file_id)
            if not metadata:
                logger.warning(f"文件元数据不存在: {file_id}")
                return False
            
            success = True
            
            # 删除向量数据库中的数据
            if delete_vector_data:
                try:
                    await self.vector_service.delete_document(file_id)
                except Exception as e:
                    logger.error(f"删除向量数据失败: {file_id} - {e}")
                    success = False
            
            # 删除解析数据
            if delete_parsed_data:
                try:
                    # 删除解析结果缓存
                    await self.cache_service.delete(f"parse_result:{file_id}")
                    await self.cache_service.delete(f"text_chunks:{file_id}")
                except Exception as e:
                    logger.error(f"删除解析数据失败: {file_id} - {e}")
                    success = False
            
            # 删除MinIO中的文件
            object_name = metadata.get("object_name")
            if object_name:
                try:
                    await self.minio_service.delete_file(object_name)
                except Exception as e:
                    logger.error(f"删除MinIO文件失败: {file_id} - {e}")
                    success = False
            
            # 删除Redis中的元数据
            try:
                await self.cache_service.delete(f"file:{file_id}")
            except Exception as e:
                logger.error(f"删除Redis元数据失败: {file_id} - {e}")
                success = False
            
            if success:
                logger.info(f"文件删除成功: {file_id}")
            else:
                logger.warning(f"文件删除部分失败: {file_id}")
            
            return success
            
        except Exception as e:
            logger.error(f"文件删除失败: {file_id} - {e}")
            return False
    
    async def start_parse_task(self, file_id: str, priority: int = 0) -> str:
        """启动文档解析任务 - 支持优先级设置"""
        await self._get_services()
        
        try:
            # 获取任务服务
            from app.services.task_service import get_task_service
            task_service = await get_task_service()
            
            # 🔧 修复：定义解析任务函数，然后传递给create_task
            async def parse_task_func():
                """实际的解析任务函数"""
                try:
                    logger.info(f"开始解析文档: {file_id}")
                    result = await self.parse_document(file_id)
                    logger.info(f"文档解析完成: {file_id}")
                    return result
                except Exception as e:
                    logger.error(f"文档解析任务失败: {file_id} - {e}")
                    raise
            
            # 🔧 修复：正确调用create_task，传递函数作为第一个参数
            task_id = await task_service.create_task(
                task_func=parse_task_func,  # ✅ 传递函数作为第一个参数
                task_name=f"解析文档 {file_id}",
                created_by="document_service"
            )
            
            logger.info(f"文档解析任务已启动: {task_id} - {file_id} - 优先级: {priority}")
            return task_id
            
        except Exception as e:
            logger.error(f"启动文档解析任务失败: {file_id} - {e}")
            raise create_service_exception(
                ErrorCode.TASK_CREATION_FAILED,
                f"启动解析任务失败: {str(e)}"
            )
    
    async def _run_mineru_with_sglang(self, input_file: str, file_id: str, original_filename: str = None) -> Dict[str, Any]:
        """使用MinerU和SGLang服务解析文档 - 🔧 优化：解析结果直接存储到MinIO"""
        try:
            import subprocess
            import tempfile
            import shutil
            
            # 创建临时工作目录
            with tempfile.TemporaryDirectory() as temp_dir:
                # 🔧 修复：保持原始文件名而不是使用"input"
                if original_filename:
                    # 使用原始文件名，但清理特殊字符以避免路径问题
                    safe_filename = "".join(c for c in original_filename if c.isalnum() or c in '.-_')
                    temp_input = os.path.join(temp_dir, safe_filename)
                else:
                    # 兜底方案：使用file_id作为文件名
                    temp_input = os.path.join(temp_dir, f"{file_id}" + Path(input_file).suffix)
                
                shutil.copy2(input_file, temp_input)
                logger.info(f"📄 使用文件名进行解析: {os.path.basename(temp_input)}")
                
                # 创建临时输出目录
                temp_output = os.path.join(temp_dir, "output")
                os.makedirs(temp_output, exist_ok=True)
                
                # 构建MinerU命令
                cmd = [
                    "mineru",
                    "-p", temp_input,
                    "-o", temp_output,
                    "-b", "vlm-sglang-client",
                    "-u", settings.SGLANG_API_BASE
                ]
                
                logger.info(f"执行MinerU命令: {' '.join(cmd)}")
                
                # 🔧 修复：增加超时时间并添加更详细的日志
                logger.info(f"⏱️  开始执行MinerU解析，预计需要10-15分钟...")
                
                # 执行命令 - 增加超时到20分钟，适应大文件处理
                result = subprocess.run(
                    cmd,
                    capture_output=True,
                    text=True,
                    timeout=1200  # 🔧 20分钟超时，适应大文件
                )
                
                logger.info(f"✅ MinerU命令执行完成，返回码: {result.returncode}")
                
                if result.returncode != 0:
                    logger.error(f"MinerU执行失败: {result.stderr}")
                    raise Exception(f"MinerU执行失败: {result.stderr}")
                
                # 🔧 优化：将解析结果上传到MinIO而不是保存到本地
                minio_files = []
                content_blocks = []
                
                logger.info(f"📤 开始将解析结果上传到MinIO...")
                
                if os.path.exists(temp_output):
                    # 统计要上传的文件
                    total_files = sum([len(files) for _, _, files in os.walk(temp_output)])
                    logger.info(f"📊 发现{total_files}个解析结果文件，开始上传...")
                    
                    uploaded_count = 0
                    # 扫描临时输出目录并上传所有文件到MinIO
                    for root, dirs, files in os.walk(temp_output):
                        for file in files:
                            local_file_path = os.path.join(root, file)
                            
                            # 计算相对路径，保持目录结构
                            rel_path = os.path.relpath(local_file_path, temp_output)
                            
                            # MinIO中的路径：parsed/{file_id}/{相对路径}
                            minio_path = f"parsed/{file_id}/{rel_path}"
                            
                            try:
                                # 读取文件内容
                                with open(local_file_path, 'rb') as f:
                                    file_content = f.read()
                                
                                # 上传到MinIO
                                content_type = "text/markdown" if file.endswith('.md') else \
                                             "application/json" if file.endswith('.json') else \
                                             "application/octet-stream"
                                
                                await self.minio_service.upload_file(
                                    object_name=minio_path,
                                    file_data=file_content,
                                    content_type=content_type
                                )
                                
                                minio_files.append(minio_path)
                                uploaded_count += 1
                                
                                # 添加到内容块列表
                                if file.endswith('.md'):
                                    content_blocks.append({
                                        "type": "markdown",
                                        "minio_path": minio_path,
                                        "local_filename": file,
                                        "size": len(file_content)
                                    })
                                elif file.endswith('.json'):
                                    content_blocks.append({
                                        "type": "json", 
                                        "minio_path": minio_path,
                                        "local_filename": file,
                                        "size": len(file_content)
                                    })
                                
                                logger.info(f"✅ [{uploaded_count}/{total_files}] 已上传: {minio_path} ({len(file_content)} 字节)")
                                
                            except Exception as e:
                                logger.error(f"❌ 上传解析结果失败: {local_file_path} -> {minio_path} - {e}")
                                continue
                    
                    if uploaded_count > 0:
                        logger.info(f"🎉 MinIO上传完成: {uploaded_count}/{total_files} 个文件上传成功")
                    else:
                        logger.warning(f"⚠️  没有文件被上传到MinIO")
                else:
                    logger.warning(f"⚠️  MinerU输出目录不存在: {temp_output}")
                
                # 解析结果
                parse_result = {
                    "status": "success",
                    "minio_base_path": f"parsed/{file_id}",
                    "minio_files": minio_files,
                    "stdout": result.stdout,
                    "stderr": result.stderr,
                    "content_blocks": content_blocks,
                    "uploaded_files_count": len(minio_files),
                    "processing_time": "查看任务日志获取详细时间"
                }
                
                logger.info(f"🎉 MinerU解析完成: 找到{len(content_blocks)}个内容块，已上传{len(minio_files)}个文件到MinIO")
                return parse_result
                
        except subprocess.TimeoutExpired as e:
            logger.error(f"⏰ MinerU解析超时（20分钟）: {e}")
            return {
                "status": "failed",
                "error": f"解析超时，大文件处理需要更长时间: {str(e)}",
                "content_blocks": [],
                "timeout": True
            }
        except Exception as e:
            logger.error(f"❌ MinerU解析失败: {e}")
            import traceback
            logger.error(f"🔍 详细错误信息: {traceback.format_exc()}")
            return {
                "status": "failed",
                "error": str(e),
                "content_blocks": [],
                "detailed_error": traceback.format_exc()
            }
    
    async def parse_document(self, file_id: str) -> Dict[str, Any]:
        """解析文档内容"""
        await self._get_services()
        
        # 获取文件信息
        metadata = await self.get_file_info(file_id)
        if not metadata:
            raise create_service_exception(
                ErrorCode.FILE_NOT_FOUND,
                f"文件不存在: {file_id}"
            )
        
        try:
            # 🔧 修复文件大小计算错误
            filename = metadata.get("filename", "未知文件")
            file_size = metadata.get("file_size", 0)
            
            # 确保file_size是数字类型
            try:
                file_size_num = int(file_size) if isinstance(file_size, str) else file_size
                file_size_mb = round(file_size_num / (1024*1024), 2) if file_size_num else 0
                estimated_time = file_size_mb * 0.5
            except (ValueError, TypeError):
                file_size_mb = 0
                estimated_time = 5  # 默认估计5分钟
            
            logger.info(f"🚀 开始解析文档: {filename}")
            logger.info(f"   📊 文件大小: {file_size_mb} MB")
            logger.info(f"   🆔 文件ID: {file_id}")
            logger.info(f"   ⏱️  预计处理时间: {estimated_time:.1f} 分钟（大文件需要更长时间）")
            
            # 更新解析状态
            await self.cache_service.save_file_metadata(
                file_id, 
                {**metadata, "parse_status": "parsing", "parse_started_at": datetime.now().isoformat()}
            )
            
            # 下载文件到临时目录
            logger.info(f"📥 从MinIO下载文件进行解析...")
            file_data, _ = await self.download_file(file_id)
            
            # 创建临时文件
            import tempfile
            with tempfile.NamedTemporaryFile(
                suffix=metadata.get("file_extension", ".pdf"), 
                delete=False
            ) as temp_file:
                temp_file.write(file_data)
                temp_input_path = temp_file.name
            
            try:
                # 🔧 优化：不再需要本地输出目录，直接使用MinIO存储
                
                # 使用MinerU解析 - 🔧 传递原始文件名
                original_filename = metadata.get("filename", metadata.get("original_filename"))
                parse_result = await self._run_mineru_with_sglang(temp_input_path, file_id, original_filename)
                
                # 更新元数据 - 🔧 优化：保存MinIO路径信息而不是本地路径
                # 🔧 修复：同时更新status和parse_status字段以保持API兼容性
                is_success = parse_result["status"] == "success"
                updated_metadata = {
                    **metadata,
                    "status": "parsed" if is_success else "parse_failed",  # API层检查的字段
                    "parse_status": "completed" if is_success else "failed",  # Service层使用的字段
                    "parse_result": parse_result,
                    "parsed_at": datetime.now().isoformat(),
                    "minio_base_path": parse_result.get("minio_base_path"),  # MinIO中的基础路径
                    "parsed_files_count": len(parse_result.get("content_blocks", [])),
                    "storage_location": "minio"  # 标记存储位置
                }
                
                await self.cache_service.save_file_metadata(file_id, updated_metadata)
                
                # 🔧 添加详细的完成日志
                if parse_result.get("status") == "success":
                    uploaded_files = parse_result.get("uploaded_files_count", 0)
                    content_blocks = len(parse_result.get("content_blocks", []))
                    minio_path = parse_result.get("minio_base_path")
                    
                    logger.info(f"🎉 文档解析完全成功: {file_id}")
                    logger.info(f"   📁 MinIO存储路径: {minio_path}")
                    logger.info(f"   📄 解析出内容块: {content_blocks}个")
                    logger.info(f"   💾 上传文件数量: {uploaded_files}个")
                    logger.info(f"   ✅ 所有数据已安全存储到MinIO")
                else:
                    error_msg = parse_result.get("error", "未知错误")
                    logger.error(f"❌ 文档解析失败: {file_id}")
                    logger.error(f"   🔍 错误信息: {error_msg}")
                    if parse_result.get("timeout"):
                        logger.warning(f"   ⏰ 建议：大文件可能需要更长处理时间")
                
                return parse_result
                
            finally:
                # 清理临时文件
                try:
                    os.unlink(temp_input_path)
                except:
                    pass
                    
        except Exception as e:
            # 更新解析状态为失败 - 🔧 修复：同时更新status和parse_status
            await self.cache_service.save_file_metadata(
                file_id,
                {
                    **metadata, 
                    "status": "parse_failed",  # API层检查的字段
                    "parse_status": "failed",  # Service层使用的字段
                    "parse_error": str(e)
                }
            )
            
            logger.error(f"文档解析失败: {file_id} - {e}")
            raise create_service_exception(
                ErrorCode.FILE_PARSE_FAILED,
                f"文档解析失败: {str(e)}"
            )
    
    async def extract_text_chunks(self, file_id: str) -> List[Dict[str, Any]]:
        """从解析结果中提取文本块 - 🔧 升级：智能表格感知分块"""
        await self._get_services()
        
        # 获取文件信息
        metadata = await self.get_file_info(file_id)
        if not metadata:
            raise create_service_exception(
                ErrorCode.FILE_NOT_FOUND,
                f"文件不存在: {file_id}"
            )
        
        parse_result = metadata.get("parse_result")
        if not parse_result or parse_result.get("status") != "success":
            raise create_service_exception(
                ErrorCode.FILE_PARSE_FAILED,
                f"文档未解析或解析失败: {file_id}"
            )
        
        chunks = []
        content_blocks = parse_result.get("content_blocks", [])
        
        for i, block in enumerate(content_blocks):
            if block.get("type") == "markdown":
                try:
                    # 🔧 优化：从MinIO读取文件而不是本地文件系统
                    minio_path = block.get("minio_path")
                    if minio_path:
                        # 从MinIO下载文件内容
                        file_content = await self.minio_service.download_file(minio_path)
                        content = file_content.decode('utf-8')
                        
                        # 🚀 新增：智能表格感知分块算法
                        smart_chunks = self._smart_chunk_content(content, file_id, i, minio_path)
                        chunks.extend(smart_chunks)
                                
                except Exception as e:
                    logger.warning(f"从MinIO读取解析文件失败: {minio_path} - {e}")
        
        logger.info(f"智能分块完成: {file_id} - {len(chunks)}个语义块")
        return chunks
    
    def _smart_chunk_content(self, content: str, file_id: str, block_index: int, minio_path: str) -> List[Dict[str, Any]]:
        """🚀 智能表格感知分块算法 - 🎯 招标书专用优化"""
        import re
        
        chunks = []
        current_pos = 0
        
        # 🎯 招标书专用配置
        max_chunk_size = 2000  # 增大块尺寸，适应招标书复杂内容
        min_chunk_size = 150   # 提高最小块尺寸，确保信息完整
        overlap_size = 200     # 增大重叠，确保关键信息不丢失
        
        # 1️⃣ 招标书关键结构识别
        tender_sections = self._identify_tender_sections(content)
        
        # 2️⃣ 表格边界检测（继承原有逻辑）
        table_start_pattern = r'<table[^>]*>'
        table_end_pattern = r'</table>'
        
        table_ranges = []
        for match in re.finditer(table_start_pattern, content, re.IGNORECASE):
            start_pos = match.start()
            remaining_content = content[start_pos:]
            end_match = re.search(table_end_pattern, remaining_content, re.IGNORECASE)
            
            if end_match:
                end_pos = start_pos + end_match.end()
                table_ranges.append((start_pos, end_pos))
        
        # 3️⃣ 关键信息区域检测
        key_info_ranges = self._detect_key_info_ranges(content)
        
        logger.info(f"🏗️ 招标书结构分析: {len(tender_sections)}个章节, {len(table_ranges)}个表格, {len(key_info_ranges)}个关键信息区域")
        
        # 4️⃣ 智能分块策略（招标书优化版）
        while current_pos < len(content):
            chunk_start = current_pos
            chunk_end = min(chunk_start + max_chunk_size, len(content))
            
            # 5️⃣ 优先保护关键信息完整性
            protected_chunk = None
            chunk_type = "text"
            
            # 检查是否与关键信息区域重叠
            for info_range in key_info_ranges:
                info_start, info_end, info_type = info_range
                if (chunk_start < info_end and chunk_end > info_start):
                    # 扩展到包含完整关键信息
                    if info_end - chunk_start <= max_chunk_size * 1.5:
                        chunk_start = max(0, info_start - 50)  # 包含前文上下文
                        chunk_end = min(len(content), info_end + 50)  # 包含后文上下文
                        protected_chunk = info_range
                        chunk_type = f"key_info_{info_type}"
                        logger.info(f"🔑 保护关键信息区域: {info_type} ({info_start}-{info_end})")
                        break
            
            # 6️⃣ 表格完整性保护（继承原有逻辑）
            if not protected_chunk:
                for table_start, table_end in table_ranges:
                    if (chunk_start < table_end and chunk_end > table_start and 
                        not (chunk_start <= table_start and chunk_end >= table_end)):
                        
                        if table_start >= chunk_start and table_start < chunk_end:
                            if table_end - chunk_start <= max_chunk_size * 2:
                                chunk_end = table_end
                                protected_chunk = (table_start, table_end)
                                chunk_type = "table"
                                logger.info(f"📊 扩展块以包含完整表格: {table_start}-{table_end}")
                            else:
                                chunk_end = table_start
                            break
                        
                        elif table_end > chunk_start and table_end <= chunk_end:
                            chunk_start = table_start
                            chunk_end = table_end
                            protected_chunk = (table_start, table_end)
                            chunk_type = "table"
                            logger.info(f"📊 调整块以对齐表格边界: {table_start}-{table_end}")
                            break
            
            # 7️⃣ 章节边界优化
            if not protected_chunk:
                section_boundary = self._find_section_boundary(content, chunk_end, tender_sections)
                if section_boundary:
                    chunk_end = section_boundary
                    chunk_type = "section_aligned"
            
            # 8️⃣ 语义边界优化（招标书专用）
            if not protected_chunk and chunk_end < len(content):
                boundary_chars = [
                    '\n\n', '。\n', '：\n', ';\n',  # 段落边界
                    '）\n', ')\n', '、\n',          # 列表项边界
                    '要求：', '规定：', '说明：',    # 招标书常见分隔词
                    '。', '！', '？'               # 句子边界
                ]
                best_boundary = chunk_end
                
                search_end = min(chunk_end + 300, len(content))  # 扩大搜索范围
                for boundary in boundary_chars:
                    pos = content.find(boundary, chunk_end, search_end)
                    if pos != -1:
                        best_boundary = pos + len(boundary)
                        break
                
                chunk_end = best_boundary
            
            # 9️⃣ 创建增强型块
            chunk_text = content[chunk_start:chunk_end].strip()
            
            if len(chunk_text) >= min_chunk_size:
                # 🎯 招标书专用内容增强
                enhanced_text = self._enhance_tender_chunk(chunk_text, chunk_type, protected_chunk)
                
                # 🔍 提取结构化信息
                structured_info = self._extract_structured_info(chunk_text)
                
                chunk_data = {
                    "chunk_id": f"{file_id}_{block_index}_{len(chunks)}",
                    "text": enhanced_text,
                    "chunk_index": len(chunks),
                    "source_minio_path": minio_path,
                    "block_type": chunk_type,
                    "start_pos": chunk_start,
                    "end_pos": chunk_end,
                    "size": len(chunk_text),
                    "tender_info": {
                        "section_type": self._identify_section_type(chunk_text),
                        "has_dates": bool(structured_info.get("dates")),
                        "has_amounts": bool(structured_info.get("amounts")),
                        "has_requirements": bool(structured_info.get("requirements")),
                        "structured_data": structured_info,
                        "importance_score": self._calculate_importance_score(chunk_text, chunk_type)
                    },
                    "protection_info": {
                        "is_protected": protected_chunk is not None,
                        "protection_type": chunk_type,
                        "protected_range": protected_chunk
                    }
                }
                
                chunks.append(chunk_data)
                logger.debug(f"✅ 创建招标书{chunk_type}块: {chunk_start}-{chunk_end} ({len(chunk_text)}字符)")
            
            # 🔄 位置更新策略
            if protected_chunk or chunk_type.startswith("key_info"):
                # 关键信息块后不重叠
                current_pos = chunk_end
            else:
                # 普通块使用重叠策略
                current_pos = max(chunk_end - overlap_size, chunk_start + 1)
        
        # 📊 生成分块质量报告
        self._generate_chunking_report(chunks, file_id)
        
        return chunks
    
    def _extract_table_content(self, html_content: str) -> str:
        """从HTML表格中提取纯文本内容，保持结构性"""
        import re
        
        try:
            # 移除HTML标签，但保持表格结构
            content = html_content
            
            # 表格行分隔
            content = re.sub(r'</tr[^>]*>', '\n', content)
            content = re.sub(r'<tr[^>]*>', '', content)
            
            # 表格单元格分隔  
            content = re.sub(r'</td[^>]*>', ' | ', content)
            content = re.sub(r'<td[^>]*>', '', content)
            
            # 移除其他HTML标签
            content = re.sub(r'<[^>]+>', '', content)
            
            # 清理空白
            content = re.sub(r'\n\s*\n', '\n', content)
            content = re.sub(r'\s+', ' ', content)
            
            # 移除空行和多余的分隔符
            lines = [line.strip() for line in content.split('\n') if line.strip()]
            filtered_lines = []
            
            for line in lines:
                # 移除只包含分隔符的行
                if line.replace('|', '').replace(' ', ''):
                    # 清理连续的分隔符
                    line = re.sub(r'\s*\|\s*\|\s*', ' | ', line)
                    line = re.sub(r'^\s*\|\s*', '', line)
                    line = re.sub(r'\s*\|\s*$', '', line)
                    filtered_lines.append(line)
            
            result = '\n'.join(filtered_lines)
            return result.strip()
            
        except Exception as e:
            logger.warning(f"表格内容提取失败: {e}")
            # 兜底：简单移除HTML标签
            return re.sub(r'<[^>]+>', ' ', html_content).strip()
    
    def _identify_tender_sections(self, content: str) -> List[Dict[str, Any]]:
        """🏗️ 识别招标书标准章节结构"""
        import re
        
        # 招标书常见章节模式
        section_patterns = [
            # 一级标题模式
            (r'第[一二三四五六七八九十]\s*章[^；。]*', 'chapter'),
            (r'第[0-9]+\s*章[^；。]*', 'chapter'),
            (r'[一二三四五六七八九十]、[^；。]*', 'major_section'),
            (r'[0-9]+、[^；。]*', 'major_section'),
            
            # 关键章节识别
            (r'(投标须知|招标公告|技术规范|商务条款|合同条款|评标办法|工程量清单)', 'key_chapter'),
            (r'(项目概况|工程概况|建设规模|投标人资格|投标文件)', 'important_section'),
            (r'(技术要求|质量标准|施工方案|材料设备|工期要求)', 'technical_section'),
            (r'(报价要求|付款条件|保证金|履约保证|违约责任)', 'commercial_section'),
            
            # 子章节模式
            (r'\([一二三四五六七八九十]\)[^；。]*', 'subsection'),
            (r'\([0-9]+\)[^；。]*', 'subsection'),
            (r'[0-9]+\.[0-9]+[^；。]*', 'subsection'),
        ]
        
        sections = []
        for pattern, section_type in section_patterns:
            for match in re.finditer(pattern, content, re.MULTILINE):
                sections.append({
                    'start': match.start(),
                    'end': match.end(),
                    'title': match.group().strip(),
                    'type': section_type,
                    'level': self._get_section_level(section_type)
                })
        
        # 按位置排序
        sections.sort(key=lambda x: x['start'])
        logger.info(f"📋 识别招标书章节: {len(sections)}个")
        return sections
    
    def _detect_key_info_ranges(self, content: str) -> List[tuple]:
        """🔍 检测关键信息区域（日期、金额、工期等）"""
        import re
        from datetime import datetime
        
        key_ranges = []
        
        # 1️⃣ 日期信息检测
        date_patterns = [
            r'[0-9]{4}年[0-9]{1,2}月[0-9]{1,2}日',
            r'[0-9]{4}-[0-9]{1,2}-[0-9]{1,2}',
            r'[0-9]{4}\.[0-9]{1,2}\.[0-9]{1,2}',
            r'截标时间[：:][^；。\n]*',
            r'开标时间[：:][^；。\n]*',
            r'工期[：:]?[^；。\n]*天',
            r'交工日期[：:][^；。\n]*',
            r'竣工日期[：:][^；。\n]*'
        ]
        
        for pattern in date_patterns:
            for match in re.finditer(pattern, content):
                start = max(0, match.start() - 30)
                end = min(len(content), match.end() + 30)
                key_ranges.append((start, end, 'date_info'))
        
        # 2️⃣ 金额信息检测
        amount_patterns = [
            r'[0-9,]+\.?[0-9]*\s*万元',
            r'[0-9,]+\.?[0-9]*\s*元',
            r'预算[：:]?[^；。\n]*元',
            r'投标限价[：:]?[^；。\n]*',
            r'保证金[：:]?[^；。\n]*元',
            r'人民币[^；。\n]*元'
        ]
        
        for pattern in amount_patterns:
            for match in re.finditer(pattern, content):
                start = max(0, match.start() - 50)
                end = min(len(content), match.end() + 50)
                key_ranges.append((start, end, 'amount_info'))
        
        # 3️⃣ 技术要求信息
        tech_patterns = [
            r'技术标准[：:]?[^；。\n]{20,}',
            r'质量等级[：:]?[^；。\n]*',
            r'施工工艺[：:]?[^；。\n]{20,}',
            r'材料要求[：:]?[^；。\n]{20,}',
            r'设备规格[：:]?[^；。\n]{20,}'
        ]
        
        for pattern in tech_patterns:
            for match in re.finditer(pattern, content):
                start = max(0, match.start() - 30)
                end = min(len(content), match.end() + 100)
                key_ranges.append((start, end, 'tech_requirement'))
        
        # 4️⃣ 资格要求信息
        qualification_patterns = [
            r'资质要求[：:]?[^；。\n]{20,}',
            r'业绩要求[：:]?[^；。\n]{20,}',
            r'人员要求[：:]?[^；。\n]{20,}',
            r'注册资金[：:]?[^；。\n]*'
        ]
        
        for pattern in qualification_patterns:
            for match in re.finditer(pattern, content):
                start = max(0, match.start() - 30)
                end = min(len(content), match.end() + 100)
                key_ranges.append((start, end, 'qualification'))
        
        # 去重和合并重叠区域
        key_ranges = self._merge_overlapping_ranges(key_ranges)
        logger.info(f"🔑 检测关键信息区域: {len(key_ranges)}个")
        return key_ranges
    
    def _find_section_boundary(self, content: str, position: int, sections: List[Dict]) -> int:
        """🎯 查找最佳章节边界"""
        # 查找position后最近的章节开始位置
        for section in sections:
            if section['start'] > position and section['start'] - position < 500:
                return section['start']
        return None
    
    def _enhance_tender_chunk(self, text: str, chunk_type: str, protected_info: Any) -> str:
        """🎯 招标书内容增强处理"""
        enhanced_text = text
        
        # 1️⃣ 表格内容增强
        if chunk_type == "table":
            clean_table = self._extract_table_content(text)
            if clean_table:
                enhanced_text = f"📊 表格内容:\n{clean_table}\n\n原始格式:\n{text}"
        
        # 2️⃣ 关键信息增强
        elif chunk_type.startswith("key_info"):
            info_type = chunk_type.replace("key_info_", "")
            enhanced_text = f"🔑 关键信息类型: {info_type}\n\n{text}"
            
            # 添加结构化标注
            if info_type == "date_info":
                dates = self._extract_dates(text)
                if dates:
                    enhanced_text += f"\n\n📅 提取的日期信息: {', '.join(dates)}"
            
            elif info_type == "amount_info":
                amounts = self._extract_amounts(text)
                if amounts:
                    enhanced_text += f"\n\n💰 提取的金额信息: {', '.join(amounts)}"
        
        # 3️⃣ 章节类型增强
        section_type = self._identify_section_type(text)
        if section_type != "unknown":
            enhanced_text = f"📋 章节类型: {section_type}\n\n{enhanced_text}"
        
        return enhanced_text
    
    def _extract_structured_info(self, text: str) -> Dict[str, Any]:
        """🔍 提取结构化信息"""
        info = {
            "dates": self._extract_dates(text),
            "amounts": self._extract_amounts(text),
            "requirements": self._extract_requirements(text),
            "deadlines": self._extract_deadlines(text),
            "specifications": self._extract_specifications(text)
        }
        return {k: v for k, v in info.items() if v}  # 只返回非空项
    
    def _identify_section_type(self, text: str) -> str:
        """📋 识别章节类型"""
        import re
        
        section_keywords = {
            "project_overview": ["项目概况", "工程概况", "建设规模", "项目性质"],
            "bidding_notice": ["投标须知", "投标说明", "投标人须知"],
            "technical_specs": ["技术规范", "技术要求", "施工标准", "质量标准"],
            "commercial_terms": ["商务条款", "报价要求", "付款条件", "合同条款"],
            "qualification": ["资格要求", "投标人资格", "资质要求", "业绩要求"],
            "schedule": ["工期要求", "进度安排", "里程碑", "节点计划"],
            "materials": ["材料要求", "设备规格", "材料标准"],
            "evaluation": ["评标办法", "评标标准", "评标程序"],
            "contract": ["合同条款", "履约要求", "违约责任"]
        }
        
        text_lower = text.lower()
        for section_type, keywords in section_keywords.items():
            if any(keyword in text for keyword in keywords):
                return section_type
        
        return "unknown"
    
    def _calculate_importance_score(self, text: str, chunk_type: str) -> float:
        """📊 计算内容重要性分数（0-1）"""
        score = 0.5  # 基础分数
        
        # 1️⃣ 块类型权重
        type_weights = {
            "key_info_date_info": 0.9,
            "key_info_amount_info": 0.95,
            "key_info_tech_requirement": 0.85,
            "key_info_qualification": 0.8,
            "table": 0.75,
            "section_aligned": 0.6,
            "text": 0.5
        }
        score = type_weights.get(chunk_type, 0.5)
        
        # 2️⃣ 关键词权重调整
        high_importance_keywords = [
            "截标时间", "开标时间", "投标截止", "工期", "预算", "投标限价",
            "保证金", "资质要求", "技术标准", "质量等级", "违约责任"
        ]
        
        medium_importance_keywords = [
            "技术要求", "施工方案", "材料规格", "人员配置", "安全要求",
            "环保要求", "验收标准", "付款方式"
        ]
        
        keyword_count = sum(1 for keyword in high_importance_keywords if keyword in text)
        score += keyword_count * 0.1
        
        keyword_count = sum(1 for keyword in medium_importance_keywords if keyword in text)
        score += keyword_count * 0.05
        
        # 3️⃣ 文本长度调整（适中长度更重要）
        text_length = len(text)
        if 200 <= text_length <= 1000:
            score += 0.1
        elif text_length > 2000:
            score -= 0.05
        
        return min(1.0, score)
    
    def _generate_chunking_report(self, chunks: List[Dict], file_id: str):
        """📊 生成分块质量报告"""
        total_chunks = len(chunks)
        table_chunks = sum(1 for c in chunks if c.get('block_type') == 'table')
        key_info_chunks = sum(1 for c in chunks if c.get('block_type', '').startswith('key_info'))
        
        avg_importance = sum(c.get('tender_info', {}).get('importance_score', 0) for c in chunks) / total_chunks if total_chunks > 0 else 0
        
        high_importance_chunks = sum(1 for c in chunks if c.get('tender_info', {}).get('importance_score', 0) > 0.8)
        
        structured_data_chunks = sum(1 for c in chunks if c.get('tender_info', {}).get('structured_data'))
        
        logger.info(f"📊 招标书分块质量报告 - {file_id}:")
        logger.info(f"   📄 总块数: {total_chunks}")
        logger.info(f"   📊 表格块: {table_chunks} ({table_chunks/total_chunks*100:.1f}%)")
        logger.info(f"   🔑 关键信息块: {key_info_chunks} ({key_info_chunks/total_chunks*100:.1f}%)")
        logger.info(f"   ⭐ 高重要性块: {high_importance_chunks} ({high_importance_chunks/total_chunks*100:.1f}%)")
        logger.info(f"   🏗️ 结构化数据块: {structured_data_chunks} ({structured_data_chunks/total_chunks*100:.1f}%)")
        logger.info(f"   📈 平均重要性分数: {avg_importance:.3f}")
    
    # 🔧 辅助方法
    def _get_section_level(self, section_type: str) -> int:
        """获取章节层级"""
        level_map = {
            'chapter': 1,
            'key_chapter': 1,
            'major_section': 2,
            'important_section': 2,
            'technical_section': 2,
            'commercial_section': 2,
            'subsection': 3
        }
        return level_map.get(section_type, 3)
    
    def _merge_overlapping_ranges(self, ranges: List[tuple]) -> List[tuple]:
        """合并重叠的范围"""
        if not ranges:
            return []
        
        # 按开始位置排序
        sorted_ranges = sorted(ranges, key=lambda x: x[0])
        merged = [sorted_ranges[0]]
        
        for current in sorted_ranges[1:]:
            last = merged[-1]
            # 如果重叠，合并
            if current[0] <= last[1]:
                merged[-1] = (last[0], max(last[1], current[1]), last[2])
            else:
                merged.append(current)
        
        return merged
    
    def _extract_dates(self, text: str) -> List[str]:
        """提取日期信息"""
        import re
        dates = []
        
        date_patterns = [
            r'[0-9]{4}年[0-9]{1,2}月[0-9]{1,2}日',
            r'[0-9]{4}-[0-9]{1,2}-[0-9]{1,2}',
            r'[0-9]{4}\.[0-9]{1,2}\.[0-9]{1,2}'
        ]
        
        for pattern in date_patterns:
            dates.extend(re.findall(pattern, text))
        
        return list(set(dates))  # 去重
    
    def _extract_amounts(self, text: str) -> List[str]:
        """提取金额信息"""
        import re
        amounts = []
        
        amount_patterns = [
            r'[0-9,]+\.?[0-9]*\s*万元',
            r'[0-9,]+\.?[0-9]*\s*元',
            r'人民币[^；。\n]*元'
        ]
        
        for pattern in amount_patterns:
            amounts.extend(re.findall(pattern, text))
        
        return list(set(amounts))
    
    def _extract_requirements(self, text: str) -> List[str]:
        """提取要求信息"""
        import re
        requirements = []
        
        req_patterns = [
            r'[^；。]*要求[：:]?[^；。\n]+',
            r'[^；。]*标准[：:]?[^；。\n]+',
            r'[^；。]*规范[：:]?[^；。\n]+'
        ]
        
        for pattern in req_patterns:
            requirements.extend(re.findall(pattern, text))
        
        return requirements[:5]  # 限制数量
    
    def _extract_deadlines(self, text: str) -> List[str]:
        """提取截止时间信息"""
        import re
        deadlines = []
        
        deadline_patterns = [
            r'截标时间[：:]?[^；。\n]*',
            r'投标截止[：:]?[^；。\n]*',
            r'开标时间[：:]?[^；。\n]*'
        ]
        
        for pattern in deadline_patterns:
            deadlines.extend(re.findall(pattern, text))
        
        return list(set(deadlines))
    
    def _extract_specifications(self, text: str) -> List[str]:
        """提取规格参数信息"""
        import re
        specs = []
        
        spec_patterns = [
            r'[^；。]*规格[：:]?[^；。\n]+',
            r'[^；。]*型号[：:]?[^；。\n]+',
            r'[^；。]*参数[：:]?[^；。\n]+'
        ]
        
        for pattern in spec_patterns:
            specs.extend(re.findall(pattern, text))
        
        return specs[:3]  # 限制数量
    
    async def vectorize_document(self, file_id: str) -> Dict[str, Any]:
        """将文档向量化并存储到向量数据库"""
        await self._get_services()
        
        if not self.rag_processor:
            raise create_service_exception(
                ErrorCode.INTERNAL_SERVER_ERROR,
                "RAGAnything 处理器未初始化"
            )
        
        try:
            # 获取文件元数据，检查是否属于知识库
            file_metadata = await self.get_file_info(file_id)
            if not file_metadata:
                raise create_service_exception(
                    ErrorCode.FILE_NOT_FOUND,
                    f"文件不存在: {file_id}"
                )
            
            # 确定向量集合名称
            collection_name = None
            kb_id = file_metadata.get("kb_id")
            
            if kb_id:
                # 文件属于知识库，获取知识库的集合名称
                from app.services.knowledge_base_service import get_knowledge_base_service
                kb_service = await get_knowledge_base_service()
                knowledge_base = await kb_service.get_knowledge_base(kb_id)
                if knowledge_base:
                    collection_name = knowledge_base.qdrant_config.collection_name
                    logger.info(f"文件 {file_id} 属于知识库 {kb_id}，使用集合: {collection_name}")
                else:
                    logger.warning(f"文件 {file_id} 关联的知识库 {kb_id} 不存在，使用默认集合")
            
            # 提取文本块
            chunks = await self.extract_text_chunks(file_id)
            
            if not chunks:
                raise create_service_exception(
                    ErrorCode.FILE_PARSE_FAILED,
                    f"文档没有可提取的内容: {file_id}"
                )
            
            # 生成向量
            texts = [chunk["text"] for chunk in chunks]
            
            # 使用RAGAnything生成embeddings
            embeddings = []
            for text in texts:
                # _get_embedding方法内部已经包含了fallback逻辑
                embedding = await self._get_embedding(text)
                embeddings.append(embedding)
            
            # 存储到向量数据库（使用知识库的集合或默认集合）
            point_ids = await self.vector_service.add_document_chunks(
                file_id=file_id,
                chunks=chunks,
                vectors=embeddings,
                collection_name=collection_name  # 🔧 使用知识库的集合名称
            )
            
            # 更新文件元数据
            if file_metadata:
                updated_metadata = {
                    **file_metadata,
                    "vector_status": "completed",
                    "vectorized_at": datetime.now().isoformat(),
                    "chunk_count": len(chunks),
                    "vector_point_ids": point_ids,
                    "vector_collection": collection_name or "rag_documents"  # 记录向量集合名称
                }
                await self.cache_service.save_file_metadata(file_id, updated_metadata)
            
            result = {
                "file_id": file_id,
                "chunk_count": len(chunks),
                "vector_count": len(embeddings),
                "point_ids": point_ids
            }
            
            logger.info(f"文档向量化完成: {file_id} - {len(chunks)}个块")
            return result
            
        except Exception as e:
            # 更新向量化状态为失败
            try:
                current_metadata = await self.get_file_info(file_id)
                if current_metadata:
                    updated_metadata = {
                        **current_metadata,
                        "vector_status": "failed",
                        "vector_error": str(e)
                    }
                    await self.cache_service.save_file_metadata(file_id, updated_metadata)
            except Exception as meta_error:
                logger.error(f"更新失败状态元数据失败: {file_id} - {meta_error}")
            
            logger.error(f"文档向量化失败: {file_id} - {e}")
            raise create_service_exception(
                ErrorCode.INTERNAL_SERVER_ERROR,
                f"文档向量化失败: {str(e)}"
            )
    
    async def _get_embedding(self, text: str) -> List[float]:
        """获取文本的embedding向量"""
        try:
            # 检查embedding API配置
            if not settings.EMBEDDING_API_BASE or not settings.EMBEDDING_API_KEY:
                logger.warning("Embedding API配置不完整，使用本地fallback方案")
                raise ValueError("Embedding API配置不完整")
            
            # 使用外部embedding API
            import httpx
            
            logger.info(f"DocumentService使用embedding API: {settings.EMBEDDING_API_BASE}")
            
            async with httpx.AsyncClient() as client:
                response = await client.post(
                    f"{settings.EMBEDDING_API_BASE}/embeddings",
                    json={
                        "model": settings.EMBEDDING_MODEL_NAME,
                        "input": text
                    },
                    headers={
                        "Authorization": f"Bearer {settings.EMBEDDING_API_KEY}",
                        "Content-Type": "application/json"
                    },
                    timeout=30
                )
                response.raise_for_status()
                
                data = response.json()
                embedding = data["data"][0]["embedding"]
                
                logger.info(f"DocumentService获取embedding成功: {len(embedding)}维")
                return embedding
                
        except Exception as e:
            logger.warning(f"DocumentService Embedding API失败: {e}")
            # 使用本地embedding作为fallback
            from app.services.search_service import SearchService
            search_service = SearchService()
            fallback_embedding = await search_service._get_local_embedding(text)
            logger.info(f"DocumentService使用本地embedding生成向量: {len(fallback_embedding)}维")
            return fallback_embedding
    
    async def list_files(
        self,
        limit: int = 20,
        offset: int = 0,
        status_filter: Optional[str] = None
    ) -> List[Dict[str, Any]]:
        """列出文件"""
        await self._get_services()
        
        try:
            # 获取所有文件keys
            keys = []
            cursor = 0
            while True:
                cursor, new_keys = await self.cache_service.redis.scan(
                    cursor, match="file:*", count=100
                )
                keys.extend(new_keys)
                if cursor == 0:
                    break
            
            files = []
            for key in keys:
                file_data = await self.cache_service.hgetall(key)
                if file_data:
                    # 应用状态过滤
                    if status_filter and file_data.get("status") != status_filter:
                        continue
                    files.append(file_data)
            
            # 按上传时间排序
            files.sort(key=lambda x: x.get("upload_date", ""), reverse=True)
            
            # 应用分页
            start = offset
            end = offset + limit
            return files[start:end]
            
        except Exception as e:
            logger.error(f"列出文件失败: {e}")
            return []
    
    async def start_vectorize_task(self, file_id: str, priority: int = 0) -> str:
        """启动文档向量化任务 - 参考mineru-web的后台处理机制"""
        try:
            await self._get_services()
            
            # 检查文件是否存在
            file_info = await self.get_file_info(file_id)
            if not file_info:
                raise create_service_exception(
                    ErrorCode.FILE_NOT_FOUND,
                    f"文件不存在: {file_id}"
                )
            
            # 检查文件是否已解析 - 🔧 修复：兼容两种状态字段
            status = file_info.get("status")
            parse_status = file_info.get("parse_status")
            
            # 检查API层状态字段或Service层状态字段
            is_parsed = (status == "parsed") or (parse_status == "completed")
            
            if not is_parsed:
                current_status = status or parse_status or "unknown"
                raise create_service_exception(
                    ErrorCode.INVALID_REQUEST,
                    f"文件尚未解析完成，当前状态: {current_status}"
                )
            
            # 生成任务ID
            task_id = f"vectorize_{file_id}_{uuid.uuid4().hex[:8]}"
            current_time = datetime.utcnow().isoformat()
            
            # 准备任务数据
            task_data = {
                "task_id": task_id,
                "task_type": "vectorize",
                "file_id": file_id,
                "filename": file_info.get("filename"),
                "status": "pending",
                "created_at": current_time,
                "priority": priority,
                "metadata": {
                    "file_size": file_info.get("file_size"),
                    "content_type": file_info.get("content_type"),
                    "parse_result": file_info.get("parse_result")
                }
            }
            
            # 保存任务信息
            await self.cache_service.set_task_info(task_id, task_data)
            
            # 根据优先级添加到不同队列
            if priority > 0:
                await self.cache_service.add_priority_task("document_vectorize", task_data, priority)
            else:
                await self.cache_service.add_to_queue("document_vectorize", task_data)
            
            # 更新文件状态
            await self.cache_service.hset_field(f"file:{file_id}", "vectorize_status", "pending")
            await self.cache_service.hset_field(f"file:{file_id}", "vectorize_task_id", task_id)
            
            logger.info(f"向量化任务已创建: {task_id} - 文件: {file_id}")
            return task_id
            
        except Exception as e:
            logger.error(f"创建向量化任务失败: {file_id} - {e}")
            if hasattr(e, 'code'):
                raise e
            else:
                raise create_service_exception(
                    ErrorCode.TASK_CREATION_FAILED,
                    f"创建向量化任务失败: {str(e)}"
                )
    
    async def batch_process_files(self, file_ids: List[str], operations: List[str], priority: int = 0) -> Dict[str, List[str]]:
        """批量处理文件 - 类似mineru-web的批量操作"""
        try:
            results = {
                "parse_tasks": [],
                "vectorize_tasks": [],
                "failed_operations": []
            }
            
            for file_id in file_ids:
                try:
                    if "parse" in operations:
                        task_id = await self.start_parse_task(file_id, priority)
                        results["parse_tasks"].append(task_id)
                    
                    if "vectorize" in operations:
                        # 如果同时有解析任务，等解析完成后再进行向量化
                        # 这里可以设置依赖关系或延迟处理
                        task_id = await self.start_vectorize_task(file_id, priority)
                        results["vectorize_tasks"].append(task_id)
                        
                except Exception as e:
                    logger.error(f"批量处理失败: {file_id} - {e}")
                    results["failed_operations"].append(f"{file_id}: {str(e)}")
            
            logger.info(f"批量处理完成 - 解析任务: {len(results['parse_tasks'])}, 向量化任务: {len(results['vectorize_tasks'])}")
            return results
            
        except Exception as e:
            logger.error(f"批量处理文件失败: {e}")
            raise create_service_exception(
                ErrorCode.INTERNAL_SERVER_ERROR,
                f"批量处理失败: {str(e)}"
            )
    
    async def get_file_processing_status(self, file_id: str) -> Dict[str, Any]:
        """获取文件完整处理状态 - 类似mineru-web的状态监控"""
        try:
            await self._get_services()
            
            # 获取基本文件信息
            file_info = await self.get_file_info(file_id)
            if not file_info:
                return {"exists": False}
            
            # 获取解析状态
            parse_status = file_info.get("parse_status", "pending")
            parse_task_id = file_info.get("parse_task_id")
            
            # 获取向量化状态
            vectorize_status = file_info.get("vectorize_status", "pending")
            vectorize_task_id = file_info.get("vectorize_task_id")
            
            # 获取任务详细信息
            parse_task_info = None
            vectorize_task_info = None
            
            if parse_task_id:
                parse_task_info = await self.cache_service.get_task_info(parse_task_id)
            
            if vectorize_task_id:
                vectorize_task_info = await self.cache_service.get_task_info(vectorize_task_id)
            
            # 计算总体进度
            progress = 0
            if parse_status == "completed":
                progress += 50
            elif parse_status == "running":
                progress += 25
            
            if vectorize_status == "completed":
                progress += 50
            elif vectorize_status == "running":
                progress += 25
            
            status_data = {
                "exists": True,
                "file_id": file_id,
                "filename": file_info.get("filename"),
                "upload_date": file_info.get("upload_date"),
                "file_size": file_info.get("file_size"),
                "total_progress": progress,
                "parse": {
                    "status": parse_status,
                    "task_id": parse_task_id,
                    "task_info": parse_task_info,
                    "result": file_info.get("parse_result")
                },
                "vectorize": {
                    "status": vectorize_status,
                    "task_id": vectorize_task_id,
                    "task_info": vectorize_task_info,
                    "vector_count": file_info.get("vector_count", 0)
                },
                "last_updated": file_info.get("updated_at", file_info.get("upload_date"))
            }
            
            return status_data
            
        except Exception as e:
            logger.error(f"获取文件处理状态失败: {file_id} - {e}")
            raise create_service_exception(
                ErrorCode.INTERNAL_SERVER_ERROR,
                f"获取处理状态失败: {str(e)}"
            )
    
    async def get_processing_statistics(self) -> Dict[str, Any]:
        """获取处理统计信息 - 参考mineru-web的监控面板"""
        try:
            await self._get_services()
            
            # 获取队列统计
            parse_stats = await self.cache_service.get_queue_stats("document_parse")
            vectorize_stats = await self.cache_service.get_queue_stats("document_vectorize")
            
            # 获取文件统计
            files = await self.list_files(limit=1000)
            
            # 统计各种状态的文件
            status_counts = {
                "total_files": len(files),
                "parsed_files": 0,
                "vectorized_files": 0,
                "processing_files": 0,
                "failed_files": 0
            }
            
            file_sizes = []
            processing_times = []
            
            for file in files:
                parse_status = file.get("parse_status", "pending")
                vectorize_status = file.get("vectorize_status", "pending")
                
                if parse_status == "completed":
                    status_counts["parsed_files"] += 1
                
                if vectorize_status == "completed":
                    status_counts["vectorized_files"] += 1
                
                if parse_status == "running" or vectorize_status == "running":
                    status_counts["processing_files"] += 1
                
                if parse_status == "failed" or vectorize_status == "failed":
                    status_counts["failed_files"] += 1
                
                # 收集文件大小用于统计
                if file.get("file_size"):
                    file_sizes.append(file["file_size"])
            
            # 计算统计指标
            avg_file_size = sum(file_sizes) / len(file_sizes) if file_sizes else 0
            success_rate = (status_counts["parsed_files"] / status_counts["total_files"] * 100) if status_counts["total_files"] > 0 else 0
            
            statistics = {
                "file_statistics": status_counts,
                "queue_statistics": {
                    "parse_queue": parse_stats,
                    "vectorize_queue": vectorize_stats
                },
                "performance_metrics": {
                    "average_file_size_mb": round(avg_file_size / (1024 * 1024), 2),
                    "success_rate_percent": round(success_rate, 1),
                    "total_storage_mb": round(sum(file_sizes) / (1024 * 1024), 2)
                },
                "system_health": {
                    "active_tasks": parse_stats["running_tasks"] + vectorize_stats["running_tasks"],
                    "pending_tasks": parse_stats["pending_tasks"] + vectorize_stats["pending_tasks"],
                    "failed_tasks": parse_stats["failed_tasks"] + vectorize_stats["failed_tasks"]
                },
                "updated_at": datetime.utcnow().isoformat()
            }
            
            return statistics
            
        except Exception as e:
            logger.error(f"获取处理统计失败: {e}")
            raise create_service_exception(
                ErrorCode.INTERNAL_SERVER_ERROR,
                f"获取统计信息失败: {str(e)}"
            )


# 全局文档服务实例
document_service = DocumentService()


async def get_document_service() -> DocumentService:
    """获取文档服务实例"""
    return document_service 
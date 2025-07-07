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
        """从解析结果中提取文本块"""
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
                        
                        # 简单的文本分块（可以后续优化）
                        chunk_size = 1000
                        overlap = 200
                        
                        for j in range(0, len(content), chunk_size - overlap):
                            chunk_text = content[j:j + chunk_size]
                            if len(chunk_text.strip()) > 50:  # 忽略太短的块
                                chunks.append({
                                    "chunk_id": f"{file_id}_{i}_{j}",
                                    "text": chunk_text.strip(),
                                    "chunk_index": len(chunks),
                                    "source_minio_path": minio_path,  # 🔧 改为MinIO路径
                                    "block_type": "markdown",
                                    "start_pos": j,
                                    "end_pos": j + len(chunk_text)
                                })
                                
                except Exception as e:
                    logger.warning(f"从MinIO读取解析文件失败: {minio_path} - {e}")
        
        logger.info(f"提取文本块完成: {file_id} - {len(chunks)}个块")
        return chunks
    
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
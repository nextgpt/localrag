"""
文件上传API端点
处理文件上传和相关操作
"""

from fastapi import APIRouter, UploadFile, File, Depends, Form, HTTPException
from typing import Optional, List
import logging
from datetime import datetime

from app.models.responses import SuccessResponse, ErrorCode
from app.models.requests import FileUploadRequest
from app.core.exceptions import create_file_exception
from app.services.document_service import get_document_service, DocumentService

router = APIRouter(prefix="/upload", tags=["文件上传"])
logger = logging.getLogger("rag-anything")


@router.post("/file", response_model=SuccessResponse, summary="上传文件")
async def upload_file(
    file: UploadFile = File(..., description="要上传的文件"),
    description: Optional[str] = Form(None, description="文件描述"),
    auto_parse: bool = Form(True, description="是否自动解析文件"),
    document_service: DocumentService = Depends(get_document_service)
):
    """
    上传文件接口
    
    **功能说明:**
    - 支持多种文件格式：PDF、Word、PowerPoint、Excel、图像、文本等
    - 自动验证文件类型和大小
    - 可选择是否自动解析文件
    - 返回唯一文件ID
    
    **请求参数:**
    - file: 要上传的文件（必填）
    - description: 文件描述（可选）
    - auto_parse: 是否自动解析（默认true）
    
    **响应数据:**
    - file_id: 唯一文件标识
    - original_name: 原始文件名
    - file_size: 文件大小（字节）
    - status: 文件状态
    - uploaded_at: 上传时间
    """
    
    if not file.filename:
        raise create_file_exception(
            ErrorCode.INVALID_REQUEST,
            "文件名不能为空"
        )
    
    try:
        # 💡 API层调试日志
        logger.info(f"🚀 API upload_file 开始处理文件")
        logger.info(f"📄 文件信息: filename='{file.filename}', content_type='{file.content_type}', description='{description}'")
        
        # 读取文件内容
        file_content = await file.read()
        logger.info(f"📊 文件内容读取成功，大小: {len(file_content)} 字节")
        
        # 上传文件
        logger.info(f"🔄 开始调用 document_service.upload_file")
        file_id = await document_service.upload_file(
            file_content=file_content,
            filename=file.filename,
            content_type=file.content_type,
            metadata={"description": description} if description else None
        )
        logger.info(f"✅ document_service.upload_file 调用成功，返回 file_id: {file_id}")
        
        # 获取文件信息
        file_info = await document_service.get_file_info(file_id)
        
        # 如果需要自动解析，启动解析任务
        parse_task_id = None
        if auto_parse:
            parse_task_id = await document_service.start_parse_task(file_id)
        
        result_data = {
            **file_info,
            "auto_parse": auto_parse,
            "parse_task_id": parse_task_id
        }
        
        logger.info(f"文件上传成功: {file_id} - {file.filename}")
        
        return SuccessResponse(
            data=result_data,
            message="文件上传成功"
        )
        
    except Exception as e:
        logger.error(f"文件上传失败: {e}")
        if hasattr(e, 'code'):
            raise e
        else:
            raise create_file_exception(
                ErrorCode.FILE_UPLOAD_FAILED,
                f"文件上传失败: {str(e)}"
            )


@router.post("/batch", response_model=SuccessResponse, summary="批量上传文件")
async def upload_files_batch(
    files: List[UploadFile] = File(..., description="要上传的文件列表"),
    description: Optional[str] = Form(None, description="批次描述"),
    auto_parse: bool = Form(True, description="是否自动解析所有文件"),
    document_service: DocumentService = Depends(get_document_service)
):
    """
    批量上传文件接口
    
    **功能说明:**
    - 支持一次上传多个文件
    - 每个文件独立处理，部分失败不影响其他文件
    - 返回每个文件的处理结果
    
    **请求参数:**
    - files: 文件列表（必填，最多10个）
    - description: 批次描述（可选）
    - auto_parse: 是否自动解析所有文件（默认true）
    
    **响应数据:**
    - total_files: 总文件数
    - successful_uploads: 成功上传数
    - failed_uploads: 失败上传数
    - results: 每个文件的详细结果
    """
    
    if len(files) > 10:
        raise create_file_exception(
            ErrorCode.INVALID_REQUEST,
            "批量上传最多支持10个文件"
        )
    
    if not files:
        raise create_file_exception(
            ErrorCode.INVALID_REQUEST,
            "至少需要上传一个文件"
        )
    
    results = []
    successful_uploads = 0
    failed_uploads = 0
    
    for file in files:
        try:
            if not file.filename:
                results.append({
                    "filename": "未知文件",
                    "success": False,
                    "error": "文件名不能为空"
                })
                failed_uploads += 1
                continue
            
            # 读取文件内容
            file_content = await file.read()
            
            # 上传文件
            file_id = await document_service.upload_file(
                file_content=file_content,
                filename=file.filename,
                content_type=file.content_type,
                metadata={"description": description} if description else None
            )
            
            # 获取文件信息
            file_info = await document_service.get_file_info(file_id)
            
            # 如果需要自动解析，启动解析任务
            parse_task_id = None
            if auto_parse:
                parse_task_id = await document_service.start_parse_task(file_id)
            
            results.append({
                "filename": file.filename,
                "success": True,
                "file_id": file_id,
                "file_info": file_info,
                "parse_task_id": parse_task_id
            })
            successful_uploads += 1
            
        except Exception as e:
            logger.error(f"批量上传中文件失败: {file.filename} - {e}")
            results.append({
                "filename": file.filename if file.filename else "未知文件",
                "success": False,
                "error": str(e)
            })
            failed_uploads += 1
    
    result_data = {
        "total_files": len(files),
        "successful_uploads": successful_uploads,
        "failed_uploads": failed_uploads,
        "auto_parse": auto_parse,
        "batch_description": description,
        "results": results,
        "uploaded_at": datetime.utcnow().isoformat()
    }
    
    logger.info(f"批量上传完成: 成功{successful_uploads}个，失败{failed_uploads}个")
    
    return SuccessResponse(
        data=result_data,
        message=f"批量上传完成，成功{successful_uploads}个文件"
    ) 
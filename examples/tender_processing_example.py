#!/usr/bin/env python3
"""
招标书处理完整流程示例
演示从上传PDF到专业分析的完整工作流
"""

import requests
import time
import json
from pathlib import Path
from typing import Dict, Any, Optional


class TenderDocumentProcessor:
    """招标书处理客户端"""
    
    def __init__(self, base_url: str = "http://localhost:8000"):
        self.base_url = base_url
        self.api_base = f"{base_url}/api/v1"
    
    def upload_file(self, file_path: str, description: str = None) -> Dict[str, Any]:
        """
        步骤1: 上传招标书PDF文件
        
        Args:
            file_path: PDF文件路径
            description: 文件描述
            
        Returns:
            包含file_id的响应数据
        """
        print(f"📤 正在上传文件: {file_path}")
        
        with open(file_path, 'rb') as f:
            files = {'file': f}
            data = {
                'description': description or f"招标书文件: {Path(file_path).name}",
                'auto_parse': 'true'
            }
            
            response = requests.post(f"{self.api_base}/upload/file", files=files, data=data)
            
        if response.status_code == 200:
            result = response.json()
            if result['success']:
                file_id = result['data']['file_id']
                parse_task_id = result['data'].get('parse_task_id')
                print(f"✅ 文件上传成功!")
                print(f"   文件ID: {file_id}")
                print(f"   解析任务: {parse_task_id}")
                return result['data']
            else:
                raise Exception(f"上传失败: {result}")
        else:
            raise Exception(f"上传请求失败: {response.status_code} - {response.text}")
    
    def wait_for_parsing(self, task_id: str, timeout: int = 300) -> bool:
        """
        步骤2: 等待文件解析完成
        
        Args:
            task_id: 解析任务ID
            timeout: 超时时间（秒）
            
        Returns:
            解析是否成功
        """
        print(f"⏳ 等待解析完成: {task_id}")
        
        start_time = time.time()
        while time.time() - start_time < timeout:
            response = requests.get(f"{self.api_base}/tasks/{task_id}")
            
            if response.status_code == 200:
                result = response.json()
                if result['success']:
                    status = result['data']['status']
                    progress = result['data'].get('progress', 0)
                    
                    print(f"   解析状态: {status} ({progress}%)")
                    
                    if status == 'completed':
                        print("✅ 文件解析完成!")
                        return True
                    elif status == 'failed':
                        print(f"❌ 解析失败: {result['data'].get('error')}")
                        return False
            
            time.sleep(5)  # 每5秒检查一次
        
        print("⏰ 解析超时")
        return False
    
    def vectorize_document(self, file_id: str) -> Dict[str, Any]:
        """
        步骤3: 向量化文档
        
        Args:
            file_id: 文件ID
            
        Returns:
            向量化任务信息
        """
        print(f"🧮 开始向量化文档: {file_id}")
        
        data = {'file_id': file_id}
        response = requests.post(f"{self.api_base}/documents/index", json=data)
        
        if response.status_code == 200:
            result = response.json()
            if result['success']:
                task_id = result['data']['task_id']
                print(f"✅ 向量化任务已启动: {task_id}")
                return result['data']
            else:
                raise Exception(f"向量化失败: {result}")
        else:
            raise Exception(f"向量化请求失败: {response.status_code} - {response.text}")
    
    def wait_for_vectorization(self, task_id: str, timeout: int = 600) -> bool:
        """
        等待向量化完成
        """
        print(f"⏳ 等待向量化完成: {task_id}")
        
        start_time = time.time()
        while time.time() - start_time < timeout:
            response = requests.get(f"{self.api_base}/tasks/{task_id}")
            
            if response.status_code == 200:
                result = response.json()
                if result['success']:
                    status = result['data']['status']
                    progress = result['data'].get('progress', 0)
                    
                    print(f"   向量化状态: {status} ({progress}%)")
                    
                    if status == 'completed':
                        print("✅ 向量化完成!")
                        return True
                    elif status == 'failed':
                        print(f"❌ 向量化失败: {result['data'].get('error')}")
                        return False
            
            time.sleep(10)  # 每10秒检查一次
        
        print("⏰ 向量化超时")
        return False
    
    def analyze_tender_document(self, 
                              file_id: str, 
                              query: str,
                              analysis_type: str = "general") -> Dict[str, Any]:
        """
        步骤4: 专业招标书分析
        
        Args:
            file_id: 文件ID
            query: 查询问题
            analysis_type: 分析类型 (general/project_info/technical_specs/commercial_terms/risks)
            
        Returns:
            分析结果
        """
        print(f"🔍 开始分析招标书...")
        print(f"   文件ID: {file_id}")
        print(f"   查询内容: {query}")
        print(f"   分析类型: {analysis_type}")
        
        data = {
            'query': query,
            'file_ids': [file_id],  # 使用实际的file_id
            'analysis_type': analysis_type,
            'limit': 20,
            'score_threshold': 0.4
        }
        
        response = requests.post(f"{self.api_base}/search/tender", json=data)
        
        if response.status_code == 200:
            result = response.json()
            if result['success']:
                print("✅ 分析完成!")
                return result['data']
            else:
                raise Exception(f"分析失败: {result}")
        else:
            raise Exception(f"分析请求失败: {response.status_code} - {response.text}")
    
    def comprehensive_analysis(self, file_id: str) -> Dict[str, Any]:
        """
        步骤5: 全面分析招标书
        
        Args:
            file_id: 文件ID
            
        Returns:
            全面分析结果
        """
        print("🎯 开始全面分析招标书...")
        
        # 多维度分析查询
        analysis_queries = [
            ("项目基本信息和工期要求", "project_info"),
            ("技术规格和材料要求", "technical_specs"),
            ("商务条款和资格要求", "commercial_terms"),
            ("风险识别和合规检查", "risks")
        ]
        
        results = {}
        
        for query, analysis_type in analysis_queries:
            try:
                result = self.analyze_tender_document(file_id, query, analysis_type)
                results[analysis_type] = result
                print(f"   ✓ {analysis_type} 分析完成")
                time.sleep(1)  # 避免请求过快
            except Exception as e:
                print(f"   ✗ {analysis_type} 分析失败: {e}")
                results[analysis_type] = {"error": str(e)}
        
        return results


def main():
    """主函数 - 演示完整流程"""
    
    # 初始化处理器
    processor = TenderDocumentProcessor()
    
    # 招标书文件路径（请替换为您的实际文件路径）
    pdf_file_path = "path/to/your/tender_document.pdf"
    
    if not Path(pdf_file_path).exists():
        print("❌ 请将 pdf_file_path 替换为您的实际招标书PDF文件路径")
        print("示例文件路径: /Users/yourname/Documents/招标文件.pdf")
        return
    
    try:
        # 步骤1: 上传文件
        upload_result = processor.upload_file(pdf_file_path, "重要招标项目文件")
        file_id = upload_result['file_id']
        parse_task_id = upload_result.get('parse_task_id')
        
        # 步骤2: 等待解析完成
        if parse_task_id:
            if not processor.wait_for_parsing(parse_task_id):
                print("❌ 文件解析失败，流程终止")
                return
        
        # 步骤3: 向量化文档
        vectorize_result = processor.vectorize_document(file_id)
        vectorize_task_id = vectorize_result['task_id']
        
        # 等待向量化完成
        if not processor.wait_for_vectorization(vectorize_task_id):
            print("❌ 向量化失败，流程终止")
            return
        
        # 步骤4: 开始专业分析
        print("\n" + "="*60)
        print("🎯 开始招标书专业分析")
        print("="*60)
        
        # 单个查询示例
        result = processor.analyze_tender_document(
            file_id=file_id,
            query="项目名称、建设地点和工期要求",
            analysis_type="project_info"
        )
        
        print("\n📋 项目信息分析结果:")
        print("-" * 40)
        if 'structured_analysis' in result:
            analysis = result['structured_analysis']
            print(f"📊 关键信息: {json.dumps(analysis.get('key_information', {}), ensure_ascii=False, indent=2)}")
        
        # 步骤5: 全面分析（可选）
        print("\n🔍 进行全面分析...")
        comprehensive_result = processor.comprehensive_analysis(file_id)
        
        print("\n" + "="*60)
        print("✅ 招标书处理流程完成!")
        print("="*60)
        print(f"📁 文件ID: {file_id}")
        print("📊 您现在可以使用这个file_id进行更多查询")
        
    except Exception as e:
        print(f"❌ 处理过程中出现错误: {e}")


if __name__ == "__main__":
    main() 
#!/usr/bin/env python3
"""
RAG-Anything 服务器启动脚本
"""

import os
import sys
import subprocess
import argparse
from pathlib import Path

def check_python_version():
    """检查Python版本"""
    if sys.version_info < (3, 8):
        print("错误: 需要 Python 3.8 或更高版本")
        sys.exit(1)

def install_requirements():
    """安装依赖包"""
    print("正在安装依赖包...")
    try:
        subprocess.check_call([sys.executable, "-m", "pip", "install", "-r", "requirements.txt"])
        print("依赖包安装完成")
    except subprocess.CalledProcessError:
        print("错误: 依赖包安装失败")
        sys.exit(1)

def create_directories():
    """创建必要的目录"""
    dirs = [
        "./data",
        "./data/uploads", 
        "./data/parsed",
        "./data/rag_workdir"
    ]
    
    for dir_path in dirs:
        Path(dir_path).mkdir(parents=True, exist_ok=True)
    
    print("目录创建完成")

def check_env_file():
    """检查环境配置文件"""
    if not os.path.exists(".env"):
        if os.path.exists(".env.example"):
            print("警告: .env 文件不存在，请复制 .env.example 为 .env 并配置相关参数")
            print("cp .env.example .env")
        else:
            print("警告: 环境配置文件不存在")
        return False
    return True

def start_server(host="0.0.0.0", port=8000, reload=True, workers=1):
    """启动服务器"""
    print(f"正在启动 RAG-Anything 服务器...")
    print(f"地址: http://{host}:{port}")
    print(f"API文档: http://{host}:{port}/docs")
    print(f"重载模式: {'开启' if reload else '关闭'}")
    print(f"工作进程: {workers}")
    print("-" * 50)
    
    cmd = [
        sys.executable, "-m", "uvicorn",
        "app.main:app",
        "--host", host,
        "--port", str(port)
    ]
    
    if reload and workers == 1:
        cmd.append("--reload")
    elif workers > 1:
        cmd.extend(["--workers", str(workers)])
    
    try:
        subprocess.run(cmd)
    except KeyboardInterrupt:
        print("\n服务器已停止")

def main():
    parser = argparse.ArgumentParser(description="RAG-Anything 服务器启动脚本")
    parser.add_argument("--host", default="0.0.0.0", help="服务器地址 (默认: 0.0.0.0)")
    parser.add_argument("--port", type=int, default=8000, help="服务器端口 (默认: 8000)")
    parser.add_argument("--no-reload", action="store_true", help="禁用热重载")
    parser.add_argument("--workers", type=int, default=1, help="工作进程数 (默认: 1)")
    parser.add_argument("--install-deps", action="store_true", help="安装依赖包")
    parser.add_argument("--setup-only", action="store_true", help="仅执行初始化设置")
    
    args = parser.parse_args()
    
    print("RAG-Anything 多模态文档处理和检索系统")
    print("=" * 50)
    
    # 检查Python版本
    check_python_version()
    
    # 安装依赖包
    if args.install_deps:
        install_requirements()
    
    # 创建目录
    create_directories()
    
    # 检查环境配置
    check_env_file()
    
    if args.setup_only:
        print("初始化设置完成")
        return
    
    # 启动服务器
    start_server(
        host=args.host,
        port=args.port,
        reload=not args.no_reload,
        workers=args.workers
    )

if __name__ == "__main__":
    main() 
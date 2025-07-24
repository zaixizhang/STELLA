#!/usr/bin/env python3
"""
🌟 Stella AI Assistant - 基础版启动器
最稳定的核心功能，适合日常使用
"""

import sys
import os
import subprocess
import time
import socket

def get_local_ip():
    """Get local IP address"""
    try:
        s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        s.connect(("8.8.8.8", 80))
        local_ip = s.getsockname()[0]
        s.close()
        return local_ip
    except:
        return "localhost"

def check_port(port):
    """Check if port is available"""
    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    result = sock.connect_ex(('localhost', port))
    sock.close()
    return result == 0

def main():
    print("🌟" + "=" * 60 + "🌟")
    print("   🚀 Stella AI Assistant - 基础版 (最稳定)")
    print("🌟" + "=" * 60 + "🌟")
    print()
    print("🎯 基础配置:")
    print("   ✅ 核心 AI 代理: ENABLED")
    print("   ✅ Web 用户界面: ENABLED")  
    print("   ✅ 预定义工具集: ENABLED")
    print("   ✅ PubMed 文献检索: ENABLED")
    print("   ⚠️ 高级记忆功能: DISABLED (减少冲突)")
    print()
    
    # Check if port is already in use
    if check_port(7860):
        print("⚠️  Port 7860 已被占用!")
        print("🔧 尝试停止现有进程...")
        try:
            subprocess.run(["pkill", "-f", "stella"], check=False)
            time.sleep(2)
        except:
            pass
    
    print("🚀 启动 Stella AI Assistant...")
    
    # Get network information
    local_ip = get_local_ip()
    
    print()
    print("🌐 访问地址:")
    print(f"   📍 本地:     http://localhost:7860")
    print(f"   📍 网络:     http://{local_ip}:7860")
    print()
    
    try:
        # Use the English UI directly (most stable)
        from stella_ui_english import StellaEnglishUI
        
        print("✅ 初始化基础界面...")
        print("🤖 使用稳定的核心功能")
        print("🚀 启动 Web 界面...")
        print()
        
        stella = StellaEnglishUI()
        
        # Launch with sharing enabled
        stella.launch(
            server_name="0.0.0.0",
            server_port=7860,
            share=True,  # 创建公共分享链接
            debug=False,
            show_error=True,
            inbrowser=False  # 不自动打开浏览器
        )
        
    except KeyboardInterrupt:
        print("\n🤖 用户请求停止 Stella")
        print("💫 感谢使用 Stella AI Assistant!")
    except ImportError as e:
        print(f"❌ 导入错误: {e}")
        print("💡 请确保安装了必要的依赖:")
        print("   pip install gradio requests markdownify smolagents")
        print("   pip install numpy pandas matplotlib seaborn")
        sys.exit(1)
    except Exception as e:
        print(f"❌ 启动错误: {e}")
        print("🔧 请检查:")
        print("   1. 网络连接")
        print("   2. Python 版本 (3.8+)")
        print("   3. 依赖库安装")
        sys.exit(1)

if __name__ == "__main__":
    main() 
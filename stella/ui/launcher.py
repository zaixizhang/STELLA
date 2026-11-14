#!/usr/bin/env python3
"""
🌟 Stella AI Assistant - Web Interface Launcher
Simple launcher with multiple access options and enhanced error handling
"""

import sys
import os
import subprocess
import time
import socket
import logging

# 设置日志记录
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def get_local_ip():
    """Get local IP address"""
    try:
        # Connect to a remote address to determine local IP
        s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        s.connect(("8.8.8.8", 80))
        local_ip = s.getsockname()[0]
        s.close()
        return local_ip
    except Exception as e:
        logger.warning(f"无法获取本地 IP: {e}")
        return "localhost"

def check_port(port):
    """Check if port is available"""
    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    result = sock.connect_ex(('localhost', port))
    sock.close()
    return result == 0

def check_dependencies():
    """检查必要的依赖包"""
    required_packages = [
        'gradio', 'requests', 'markdownify', 'smolagents', 
        'numpy', 'pandas', 'matplotlib', 'seaborn'
    ]
    
    # 特殊处理的包名映射
    package_mappings = {
        'sklearn': 'scikit-learn'  # sklearn 是 scikit-learn 的导入名
    }
    
    missing_packages = []
    for package in required_packages:
        try:
            __import__(package)
        except ImportError:
            missing_packages.append(package)
    
    # 检查 scikit-learn (导入时使用 sklearn)
    try:
        import sklearn
    except ImportError:
        missing_packages.append('scikit-learn')
    
    if missing_packages:
        logger.error(f"❌ 缺少必要的依赖包: {', '.join(missing_packages)}")
        logger.info("💡 请运行以下命令安装:")
        logger.info(f"   pip install {' '.join(missing_packages)}")
        return False
    
    logger.info("✅ 所有必要依赖包已安装")
    return True

def main():
    print("🤖" + "=" * 60 + "🤖")
    print("   🌟 Stella AI Assistant - Enhanced Web Launcher 🌟")
    print("🤖" + "=" * 60 + "🤖")
    print()
    
    # 检查依赖
    if not check_dependencies():
        sys.exit(1)
    
    print("🎯 Enhanced Features Enabled:")
    print("   ✅ Template Learning: ENABLED")
    print("   ✅ Mem0 Memory: ENABLED (with fallback)")
    print("   ✅ English Interface: ENABLED")
    print("   ✅ Biomedical Tools: ENABLED")
    print("   ✅ Error Handling: ENHANCED")
    print()
    
    # Check if port is already in use
    if check_port(7860):
        print("⚠️  Port 7860 is already in use!")
        print("🔧 Trying to stop existing process...")
        try:
            subprocess.run(["pkill", "-f", "launch_stella_english"], check=False)
            time.sleep(2)
        except Exception as e:
            logger.warning(f"无法停止现有进程: {e}")
    
    print("🚀 Starting Stella AI Assistant...")
    print("📡 Configuring network access...")
    
    # Get network information
    local_ip = get_local_ip()
    
    print()
    print("🌐 Access URLs:")
    print(f"   📍 Local:     http://localhost:7860")
    print(f"   📍 Network:   http://{local_ip}:7860")
    print(f"   📍 External:  http://192.222.54.136:7860")
    print()
    print()
    print("⏳ Starting interface... (this may take a moment)")
    print("🌟 The interface will also generate a public sharing link!")
    print()
    
    try:
        # Set command line arguments to enable template and mem0
        original_argv = sys.argv.copy()
        sys.argv = [sys.argv[0], "--use_template", "--use_mem0"]
        
        # Import and start Stella UI with enhanced features
        logger.info("正在导入 Stella UI 模块...")
        from stella.ui.web import main as stella_ui_main
        
        print("✅ Stella core initialized with enhanced memory!")
        print("🧠 Template Learning: Active")
        print("🤖 Mem0 Memory System: Active (with graceful fallback)")
        print("🚀 Launching English UI interface...")
        print()
        
        # Launch Stella English UI with enhanced features
        stella_ui_main()
        
        # Restore original argv
        sys.argv = original_argv
        
    except KeyboardInterrupt:
        print("\n🤖 Stella shutdown requested by user")
        print("💫 Thank you for using Stella AI Assistant!")
        # Restore original argv in case of interruption
        sys.argv = original_argv
    except ImportError as e:
        logger.error(f"❌ Import error: {e}")
        print("💡 Please ensure all dependencies are installed:")
        print("   pip install gradio requests markdownify smolagents mem0ai")
        print("   pip install numpy pandas scikit-learn matplotlib seaborn")
        print("💡 For biomedical tools:")
        print("   pip install biopython")
        # Restore original argv
        sys.argv = original_argv
        sys.exit(1)
    except Exception as e:
        logger.error(f"❌ Error starting Stella: {e}")
        print("🔧 Please check the error details above")
        print("💡 If memory initialization fails, the system will use fallback mechanisms")
        print("💡 If embedding model fails, try updating your OpenRouter API configuration")
        # Restore original argv
        sys.argv = original_argv
        sys.exit(1)

if __name__ == "__main__":
    main() 
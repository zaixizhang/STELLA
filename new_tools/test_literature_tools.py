#!/usr/bin/env python3
"""
测试脚本：literature_tools.py 中的所有工具函数

这个脚本测试每个文献查询工具的基本功能，包括：
- 基本查询测试
- 错误处理测试
- 网络连接测试

注意：某些工具需要网络连接才能正常工作
"""

import sys
import os
import traceback
import time
from typing import Dict, Any, List

# 导入所有工具函数
from literature_tools import (
    fetch_supplementary_info_from_doi,
    query_arxiv,
    query_scholar,
    query_pubmed,
    search_google,
    extract_url_content,
    extract_pdf_content
)

class LiteratureToolTester:
    """文献工具测试类"""
    
    def __init__(self):
        self.results = {}
        self.success_count = 0
        self.fail_count = 0
        self.skip_count = 0
        
    def log_result(self, test_name: str, status: str, message: str = "", result: Any = None):
        """记录测试结果"""
        self.results[test_name] = {
            'status': status,
            'message': message,
            'result': result
        }
        
        if status == 'PASS':
            self.success_count += 1
            print(f"✅ {test_name}: {status}")
        elif status == 'FAIL':
            self.fail_count += 1
            print(f"❌ {test_name}: {status} - {message}")
        elif status == 'SKIP':
            self.skip_count += 1
            print(f"⏭️  {test_name}: {status} - {message}")
            
        if message:
            print(f"   详情: {message}")
    
    def test_function(self, func, test_name: str, *args, **kwargs):
        """通用函数测试器"""
        try:
            print(f"\n🧪 测试 {test_name}...")
            result = func(*args, **kwargs)
            
            # 检查结果类型和内容
            if isinstance(result, str):
                if result and not result.startswith("Error") and not result.startswith("No "):
                    self.log_result(test_name, 'PASS', f"返回内容长度: {len(result)} 字符", result[:200] + "..." if len(result) > 200 else result)
                elif result.startswith("Error"):
                    self.log_result(test_name, 'FAIL', f"函数错误: {result}")
                elif result.startswith("No "):
                    self.log_result(test_name, 'PASS', f"正常的空结果: {result}")
                else:
                    self.log_result(test_name, 'FAIL', f"返回空字符串或异常格式")
            else:
                self.log_result(test_name, 'PASS', f"返回数据类型: {type(result)}", result)
                
        except Exception as e:
            error_msg = f"异常: {str(e)}"
            self.log_result(test_name, 'FAIL', error_msg)
            if "timeout" in str(e).lower() or "connection" in str(e).lower():
                print(f"   网络相关错误，可能是网络连接问题")
    
    def run_all_tests(self):
        """运行所有测试"""
        print("📚 文献工具测试脚本")
        print("这个脚本将测试 literature_tools.py 中的所有工具函数")
        print("注意：某些测试可能需要网络连接")
        
        print("\n🚀 开始测试所有文献工具...")
        print("=" * 80)
        
        # 1. 测试 arXiv 查询
        self.test_function(
            query_arxiv,
            "arXiv查询",
            query="machine learning",
            max_papers=2
        )
        
        # 2. 测试 Google Scholar (可能失败，因为需要特殊配置)
        try:
            self.test_function(
                query_scholar,
                "Google Scholar查询",
                query="deep learning"
            )
        except:
            self.log_result("Google Scholar查询", "SKIP", "需要特殊配置或可能被阻止")
        
        # 3. 测试 PubMed 查询
        self.test_function(
            query_pubmed,
            "PubMed查询",
            query="cancer treatment",
            max_papers=2
        )
        
        # 4. 测试 Google 搜索
        self.test_function(
            search_google,
            "Google搜索",
            query="COVID-19 research",
            num_results=2
        )
        
        # 5. 测试网页内容提取
        self.test_function(
            extract_url_content,
            "网页内容提取",
            url="https://httpbin.org/html"  # 使用测试网站
        )
        
        # 6. 测试PDF内容提取 (使用一个公开的测试PDF)
        self.test_function(
            extract_pdf_content,
            "PDF内容提取",
            url="https://www.w3.org/WAI/ER/tests/xhtml/testfiles/resources/pdf/dummy.pdf"
        )
        
        # 7. 测试DOI补充信息获取 (可能失败，因为需要特定的DOI)
        try:
            self.test_function(
                fetch_supplementary_info_from_doi,
                "DOI补充信息获取",
                doi="10.1038/nature12373",  # 使用一个真实的DOI
                output_dir="test_supplementary"
            )
        except:
            self.log_result("DOI补充信息获取", "SKIP", "可能需要特定的DOI格式或网络访问")
        
        # 测试错误处理
        print(f"\n🔍 测试错误处理...")
        
        # 测试无效URL
        self.test_function(
            extract_url_content,
            "无效URL处理",
            url="https://invalid-url-that-does-not-exist.com"
        )
        
        # 测试无效PDF URL  
        self.test_function(
            extract_pdf_content,
            "无效PDF URL处理",
            url="https://httpbin.org/html"  # 不是PDF的URL
        )
        
        self.print_summary()
    
    def print_summary(self):
        """打印测试总结"""
        print("\n" + "=" * 80)
        print("📊 测试总结")
        print("=" * 80)
        print(f"✅ 成功: {self.success_count}")
        print(f"❌ 失败: {self.fail_count}")  
        print(f"⏭️  跳过: {self.skip_count}")
        print(f"📝 总计: {self.success_count + self.fail_count + self.skip_count}")
        
        if self.success_count + self.fail_count > 0:
            success_rate = (self.success_count / (self.success_count + self.fail_count)) * 100
            print(f"📈 成功率: {success_rate:.1f}%")
        
        # 显示失败的测试
        failed_tests = [name for name, result in self.results.items() if result['status'] == 'FAIL']
        if failed_tests:
            print(f"\n❌ 失败的测试:")
            for test_name in failed_tests:
                result = self.results[test_name]
                print(f"   - {test_name}: {result['message']}")


def main():
    """主函数"""
    # 设置工作目录
    if not os.path.exists('test_output'):
        os.makedirs('test_output')
    
    tester = LiteratureToolTester()
    tester.run_all_tests()


if __name__ == "__main__":
    main() 
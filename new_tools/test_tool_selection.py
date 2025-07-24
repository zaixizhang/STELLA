#!/usr/bin/env python3
"""
测试智能工具选择功能的脚本
"""

import sys
import os
sys.path.append('..')  # 添加上级目录到路径

def test_tool_selection():
    """测试智能工具选择功能"""
    try:
        # 导入必要的函数
        from stella_core import analyze_query_and_load_relevant_tools
        
        print("🧪 测试智能工具选择功能")
        print("=" * 50)
        
        # 测试用例
        test_queries = [
            "查找关于CRISPR-Cas9在癌症治疗中的最新研究",
            "分析蛋白质结构和功能",
            "从PubMed搜索COVID-19相关论文",
            "查询UniProt数据库中的胰岛素蛋白信息",
            "提取PDF文献的内容并分析"
        ]
        
        for i, query in enumerate(test_queries, 1):
            print(f"\n📋 测试用例 {i}: {query}")
            print("-" * 50)
            
            try:
                result = analyze_query_and_load_relevant_tools(query, max_tools=20)
                print(result)
                
                if "✅" in result:
                    print("✅ 测试通过")
                else:
                    print("❌ 测试可能有问题")
                    
            except Exception as e:
                print(f"❌ 测试失败: {str(e)}")
        
        print("\n" + "=" * 50)
        print("📊 测试完成")
        
    except Exception as e:
        print(f"❌ 无法运行测试: {str(e)}")
        print("💡 请确保在agents/STELLA/目录下运行此脚本")

if __name__ == "__main__":
    test_tool_selection() 
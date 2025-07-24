#!/usr/bin/env python3
"""
测试脚本：database_tools.py 中的所有工具函数

这个脚本测试每个数据库查询工具的基本功能，包括：
- 基本查询测试
- 错误处理测试
- 网络连接测试

注意：某些工具需要API密钥或者特定的网络连接才能正常工作
"""

import sys
import os
import traceback
import time
from typing import Dict, Any, List

# 添加模块路径
sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), 'agents', 'STELLA', 'new_tools'))

# 导入所有工具函数
from database_tools import (
    query_uniprot, query_alphafold, query_interpro, query_pdb, query_pdb_identifiers,
    query_kegg, query_stringdb, query_paleobiology, query_jaspar,
    query_worms, query_cbioportal, query_clinvar, query_geo, query_dbsnp,
    query_ucsc, query_ensembl, query_opentarget_genetics, query_opentarget,
    query_gwas_catalog, query_gnomad, blast_sequence, query_reactome,
    query_regulomedb, query_pride, query_gtopdb, region_to_ccre_screen,
    get_genes_near_ccre, query_remap, query_mpd, query_emdb, get_hpo_names
)

class DatabaseToolTester:
    """数据库工具测试类"""
    
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
            if isinstance(result, dict):
                if 'error' in result:
                    self.log_result(test_name, 'FAIL', f"API错误: {result.get('error', '未知错误')}")
                elif 'success' in result and result['success']:
                    self.log_result(test_name, 'PASS', "API调用成功", result)
                elif result:  # 非空字典被视为成功
                    self.log_result(test_name, 'PASS', "返回有效数据", result)
                else:
                    self.log_result(test_name, 'FAIL', "返回空数据")
            elif isinstance(result, str) and result:
                self.log_result(test_name, 'PASS', "返回字符串数据", result)
            elif result is not None:
                self.log_result(test_name, 'PASS', f"返回数据类型: {type(result)}", result)
            else:
                self.log_result(test_name, 'FAIL', "函数返回None")
                
        except Exception as e:
            error_msg = f"异常: {str(e)}"
            self.log_result(test_name, 'FAIL', error_msg)
            if "timeout" in str(e).lower() or "connection" in str(e).lower():
                print(f"   网络相关错误，可能是网络连接问题")
    
    def run_all_tests(self):
        """运行所有测试"""
        print("🚀 开始测试所有数据库工具...")
        print("=" * 80)
        
        # 1. 测试 UniProt
        self.test_function(
            query_uniprot,
            "UniProt查询",
            prompt="Find information about human insulin protein",
            max_results=2
        )
        
        # 2. 测试 AlphaFold
        self.test_function(
            query_alphafold,
            "AlphaFold查询",
            uniprot_id="P01308",
            endpoint="prediction"
        )
        
        # 3. 测试 InterPro
        self.test_function(
            query_interpro,
            "InterPro查询",
            prompt="Find information about kinase domains",
            max_results=2
        )
        
        # 4. 测试 PDB
        self.test_function(
            query_pdb,
            "PDB查询",
            prompt="Find structures of human insulin",
            max_results=2
        )
        
        # 5. 测试 KEGG
        self.test_function(
            query_kegg,
            "KEGG查询",
            prompt="Find human pathways related to glycolysis",
            verbose=False
        )
        
        # 6. 测试 STRING数据库
        self.test_function(
            query_stringdb,
            "STRING数据库查询",
            prompt="Show protein interactions for BRCA1 in humans",
            verbose=False
        )
        
        
        # 8. 测试古生物学数据库
        self.test_function(
            query_paleobiology,
            "古生物学数据库查询",
            prompt="Find fossil records of Tyrannosaurus rex",
            verbose=False
        )
        
        # 9. 测试 JASPAR
        self.test_function(
            query_jaspar,
            "JASPAR查询",
            prompt="Find transcription factor matrices for human",
            verbose=False
        )
        
        # 10. 测试 WoRMS
        self.test_function(
            query_worms,
            "WoRMS查询",
            prompt="Find information about the blue whale",
            verbose=False
        )
        
        # 11. 测试 cBioPortal
        self.test_function(
            query_cbioportal,
            "cBioPortal查询",
            prompt="Find mutations in BRCA1 for breast cancer",
            verbose=False
        )
        
        # 12. 测试 ClinVar
        self.test_function(
            query_clinvar,
            "ClinVar查询",
            prompt="Find pathogenic BRCA1 variants",
            max_results=2
        )
        
        # 13. 测试 GEO
        self.test_function(
            query_geo,
            "GEO查询",
            prompt="Find RNA-seq datasets for breast cancer",
            max_results=2
        )
        
        # 14. 测试 dbSNP
        self.test_function(
            query_dbsnp,
            "dbSNP查询",
            prompt="Find pathogenic variants in BRCA1",
            max_results=2
        )
        
        # 15. 测试 UCSC
        self.test_function(
            query_ucsc,
            "UCSC查询",
            prompt="Get DNA sequence of chromosome M positions 1-100 in human genome",
            verbose=False
        )
        
        # 16. 测试 Ensembl
        self.test_function(
            query_ensembl,
            "Ensembl查询",
            prompt="Get information about the human BRCA2 gene",
            verbose=False
        )
        
        # 17. 测试 OpenTargets Genetics
        self.test_function(
            query_opentarget_genetics,
            "OpenTargets Genetics查询",
            prompt="Get information about variant 1_154453788_C_T",
            verbose=False
        )
        
        # 18. 测试 OpenTargets Platform
        self.test_function(
            query_opentarget,
            "OpenTargets Platform查询",
            prompt="Find drug targets for Alzheimer's disease",
            verbose=False
        )
        
        # 19. 测试 GWAS Catalog
        self.test_function(
            query_gwas_catalog,
            "GWAS Catalog查询",
            prompt="Find GWAS studies related to Type 2 diabetes",
            max_results=2
        )
        
        # 20. 测试 gnomAD
        self.test_function(
            query_gnomad,
            "gnomAD查询",
            gene_symbol="BRCA1",
            verbose=False
        )
        
        # 21. 测试 BLAST (简单序列)
        self.test_function(
            blast_sequence,
            "BLAST序列查询",
            sequence="ATGCGATCGATCGATCG",  # 简单测试序列
            database="core_nt",
            program="blastn"
        )
        
        # 22. 测试 Reactome
        self.test_function(
            query_reactome,
            "Reactome查询",
            prompt="Find pathways related to DNA repair",
            verbose=False
        )
        
        # 23. 测试 RegulomeDB
        self.test_function(
            query_regulomedb,
            "RegulomeDB查询",
            prompt="Find regulatory elements for rs35675666",
            verbose=False
        )
        
        # 24. 测试 PRIDE
        self.test_function(
            query_pride,
            "PRIDE查询",
            prompt="Find proteomics data related to breast cancer",
            max_results=2
        )
        
        # 25. 测试 GtoPdb
        self.test_function(
            query_gtopdb,
            "GtoPdb查询",
            prompt="Find ligands that target the beta-2 adrenergic receptor",
            verbose=False
        )
        
        # 26. 测试 cCRE SCREEN
        self.test_function(
            region_to_ccre_screen,
            "cCRE SCREEN查询",
            coord_chrom="chr1",
            coord_start=1000000,
            coord_end=1001000,
            assembly="GRCh38"
        )
        
        # 27. 测试获取cCRE附近基因
        try:
            self.test_function(
                get_genes_near_ccre,
                "cCRE附近基因查询",
                accession="EH38E1516980",
                assembly="GRCh38",
                chromosome="chr12",
                k=5
            )
        except:
            self.log_result("cCRE附近基因查询", "SKIP", "需要有效的cCRE accession")
        
        # 28. 测试 ReMap
        self.test_function(
            query_remap,
            "ReMap查询",
            prompt="Find CTCF binding sites in chromosome 1",
            verbose=False
        )
        
        # 29. 测试鼠类表型数据库
        self.test_function(
            query_mpd,
            "鼠类表型数据库查询",
            prompt="Find phenotype data for C57BL/6J mice related to blood glucose",
            verbose=False
        )
        
        # 30. 测试电子显微镜数据库
        self.test_function(
            query_emdb,
            "电子显微镜数据库查询",
            prompt="Find cryo-EM structures of ribosomes at resolution better than 3Å",
            verbose=False
        )
        
        # 31. 测试HPO术语名称获取
        try:
            self.test_function(
                get_hpo_names,
                "HPO术语名称查询",
                hpo_terms=["HP:0001250", "HP:0000707"]
            )
        except:
            self.log_result("HPO术语名称查询", "SKIP", "需要HPO数据文件")

    def print_summary(self):
        """打印测试总结"""
        print("\n" + "=" * 80)
        print("📊 测试总结")
        print("=" * 80)
        print(f"✅ 成功: {self.success_count}")
        print(f"❌ 失败: {self.fail_count}")
        print(f"⏭️  跳过: {self.skip_count}")
        print(f"📝 总计: {len(self.results)}")
        print(f"📈 成功率: {self.success_count/len(self.results)*100:.1f}%")
        
        # 显示失败的测试
        failed_tests = [name for name, result in self.results.items() if result['status'] == 'FAIL']
        if failed_tests:
            print(f"\n❌ 失败的测试:")
            for test_name in failed_tests:
                result = self.results[test_name]
                print(f"   - {test_name}: {result['message']}")
        
        # 显示跳过的测试
        skipped_tests = [name for name, result in self.results.items() if result['status'] == 'SKIP']
        if skipped_tests:
            print(f"\n⏭️  跳过的测试:")
            for test_name in skipped_tests:
                result = self.results[test_name]
                print(f"   - {test_name}: {result['message']}")

def main():
    """主函数"""
    print("🧬 数据库工具测试脚本")
    print("这个脚本将测试 database_tools.py 中的所有工具函数")
    print("注意：某些测试可能需要网络连接和API密钥\n")
    
    tester = DatabaseToolTester()
    
    try:
        tester.run_all_tests()
    except KeyboardInterrupt:
        print("\n⚠️  测试被用户中断")
    except Exception as e:
        print(f"\n💥 测试过程中发生错误: {str(e)}")
        traceback.print_exc()
    finally:
        tester.print_summary()

if __name__ == "__main__":
    main() 
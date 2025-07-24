
# _core_guard/sequence_scanner.py
import re

NUC_RE = re.compile(r"[ACGT]{60,}", re.I)
AA_RE  = re.compile(r"[ACDEFGHIKLMNPQRSTVWY]{60,}", re.I)

def sequence_check(text: str) -> bool:
    """
    目前仅用正则检测长序列（可扩展为 mmseqs2/blastn 本地比对）
    """
    return bool(NUC_RE.search(text) or AA_RE.search(text))

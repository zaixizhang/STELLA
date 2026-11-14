
# _core_guard/embedding_filter.py
from sentence_transformers import SentenceTransformer, util
import numpy as np

# 初始化模型（推荐小模型：MiniLM 或 BioBERT）
model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")

# 示例高风险文本向量（真实部署请加载文件）
HIGH_RISK_TEXTS = [
    "reverse genetic manipulation of coronavirus",
    "designing airborne toxins",
    "synthetic viral vector for gene delivery"
]
HIGH_RISK_EMBEDDINGS = model.encode(HIGH_RISK_TEXTS, convert_to_tensor=True)

def embedding_check(text: str, threshold: float = 0.75) -> bool:
    embedding = model.encode(text, convert_to_tensor=True)
    scores = util.cos_sim(embedding, HIGH_RISK_EMBEDDINGS)
    return bool((scores > threshold).any())

from typing import List
import numpy as np
from sentence_transformers import SentenceTransformer
import torch


class Embedder:
    def __init__(self, model_name: str, device: str = "auto", batch_size: int = 64):
        if device == "auto":
            device = "cuda" if torch.cuda.is_available() else "cpu"
        self.device = device
        self.model = SentenceTransformer(model_name, device=self.device)
        self.batch_size = batch_size
        # 推理输出为 float32，便于 Faiss

    def encode(self, texts: List[str]) -> np.ndarray:
        embs = self.model.encode(
            texts,
            batch_size=self.batch_size,
            convert_to_numpy=True,
            normalize_embeddings=False,
            show_progress_bar=False,
        )
        return embs.astype("float32")

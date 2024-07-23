# ranker.py

import faiss
import numpy as np
import torch

class Ranker:
    def __init__(self, job_embeddings):
        self.job_embeddings = job_embeddings.cpu().numpy()
        self.index = self._create_index()

    def _create_index(self):
        index = faiss.IndexFlatL2(self.job_embeddings.shape[1])
        index.add(self.job_embeddings)
        return index

    def rank(self, kpi_embedding):
        kpi_embedding_np = kpi_embedding.cpu().numpy().reshape(1, -1)
        D, I = self.index.search(kpi_embedding_np, len(self.job_embeddings))
        return I[0]
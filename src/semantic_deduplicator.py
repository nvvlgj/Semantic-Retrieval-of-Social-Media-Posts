"""
Semantic Deduplicator

Uses sentence embeddings to identify semantically
similar documents and keeps one representative
from each cluster.
"""

from __future__ import annotations

from typing import List

import numpy as np
import pandas as pd
from sentence_transformers import SentenceTransformer
from sklearn.neighbors import NearestNeighbors

from .config import EMBEDDING_MODEL


class SemanticDeduplicator:

    def __init__(
        self,
        model_name: str = EMBEDDING_MODEL,
        similarity_threshold: float = 0.95,
        batch_size: int = 32,
    ):

        self.model = SentenceTransformer(model_name)

        self.threshold = similarity_threshold

        self.batch_size = batch_size

    # -------------------------------------------------------

    def remove_semantic_duplicates(
        self,
        dataframe: pd.DataFrame,
        text_column: str,
    ) -> pd.DataFrame:

        print("\nRunning semantic deduplication...\n")

        texts = dataframe[text_column].tolist()

        embeddings = self.model.encode(
            texts,
            batch_size=self.batch_size,
            convert_to_numpy=True,
            normalize_embeddings=True,
            show_progress_bar=True,
        )

        nn = NearestNeighbors(
            metric="cosine",
            algorithm="auto",
        )

        nn.fit(embeddings)

        distances, neighbors = nn.kneighbors(
            embeddings,
            n_neighbors=10,
        )

        keep = []

        removed = set()

        duplicate_groups = []

        for i in range(len(texts)):

            if i in removed:
                continue

            keep.append(i)

            current_cluster = [i]

            for distance, neighbor in zip(
                distances[i][1:],
                neighbors[i][1:],
            ):

                similarity = 1 - distance

                if similarity >= self.threshold:

                    removed.add(neighbor)

                    current_cluster.append(neighbor)

            if len(current_cluster) > 1:

                duplicate_groups.append(current_cluster)

        print(f"Semantic duplicates removed : {len(removed)}")

        print(f"Clusters found              : {len(duplicate_groups)}")

        cleaned = dataframe.iloc[keep].reset_index(drop=True)

        return cleaned
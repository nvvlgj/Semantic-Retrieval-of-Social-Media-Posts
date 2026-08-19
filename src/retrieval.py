"""
Task 3 - Semantic Retrieval

Responsibilities
----------------
1. Load embedding bank
2. Build NearestNeighbors index
3. Encode search query
4. Retrieve Top-K documents
5. Save retrieval results

Author: Lalitha Nandiraju
"""

from __future__ import annotations

import json

import numpy as np
import pandas as pd
from sentence_transformers import SentenceTransformer
from sklearn.neighbors import NearestNeighbors

from .config import (
    EMBEDDING_MODEL,
    BATCH_SIZE,
    NORMALIZE_EMBEDDINGS,
    EMBEDDING_FILE,
    METADATA_FILE,
    RESULT_DIRECTORY,
    RESULT_FILE,
    RETRIEVAL_INFO_FILE,
    EVALUATION_FILE,
    TOP_K,
)


class SemanticRetriever:
    """
    Performs semantic retrieval over a precomputed embedding bank.
    """

    def __init__(
        self,
        model_name: str,
        top_k: int = 30,
    ):

        self.model_name = model_name
        self.top_k = top_k

        print(f"Loading embedding model: {model_name}")

        self.model = SentenceTransformer(model_name)

        self.index = None

    # -------------------------------------------------------
    # Embedding Bank
    # -------------------------------------------------------

    def load_embedding_bank(self):

        print("Loading embedding bank...")

        embeddings = np.load(EMBEDDING_FILE)

        metadata = pd.read_csv(METADATA_FILE)

        print(f"Loaded {len(metadata)} documents.")

        return embeddings, metadata

    # -------------------------------------------------------
    # Index
    # -------------------------------------------------------

    def build_index(
        self,
        embeddings: np.ndarray,
    ):

        print("Building NearestNeighbors index...")

        self.index = NearestNeighbors(
            metric="cosine",
            algorithm="auto",
        )

        self.index.fit(embeddings)

        print("Index created successfully.")

    # -------------------------------------------------------
    # Query
    # -------------------------------------------------------

    def encode_query(
        self,
        query: str,
    ) -> np.ndarray:

        embedding = self.model.encode(
            query,
            convert_to_numpy=True,
            normalize_embeddings=True,
        )

        return embedding.astype(np.float32)

    # -------------------------------------------------------
    # Search
    # -------------------------------------------------------

    def search(
        self,
        query: str,
        metadata: pd.DataFrame,
    ) -> pd.DataFrame:

        query_embedding = self.encode_query(query)

        distances, indices = self.index.kneighbors(
            query_embedding.reshape(1, -1),
            n_neighbors=self.top_k,
        )

        similarities = 1.0 - distances[0]

        results = metadata.iloc[
            indices[0]
        ].copy()

        results.insert(
            0,
            "rank",
            np.arange(
                1,
                len(results) + 1,
            ),
        )

        results.insert(
            1,
            "similarity",
            similarities,
        )

        return results

    # -------------------------------------------------------
    # Save
    # -------------------------------------------------------

    def save_results(
        self,
        query: str,
        results: pd.DataFrame,
    ):

        RESULT_DIRECTORY.mkdir(
            parents=True,
            exist_ok=True,
        )

        results.to_csv(
            RESULT_FILE,
            index=False,
        )

        info = {
            "query": query,
            "embedding_model": self.model_name,
            "retrieval_method": "NearestNeighbors (Cosine)",
            "top_k": self.top_k,
            "documents_retrieved": len(results),
            "normalized_embeddings": True,
        }

        with open(
            RETRIEVAL_INFO_FILE,
            "w",
            encoding="utf-8",
        ) as file:

            json.dump(
                info,
                file,
                indent=4,
            )

        print("\nResults saved successfully.")
        print(f"Results : {RESULT_FILE}")
        print(f"Info    : {RETRIEVAL_INFO_FILE}")

    def evaluate_results(
        self,
        results: pd.DataFrame,
    ):

        evaluation = {

            "documents_retrieved": len(results),

            "highest_similarity": float(
                results["similarity"].max()
            ),

            "lowest_similarity": float(
                results["similarity"].min()
            ),

            "average_similarity": float(
                results["similarity"].mean()
            ),

            "median_similarity": float(
                results["similarity"].median()
            ),
        }

        with open(
            EVALUATION_FILE,
            "w",
            encoding="utf-8",
        ) as file:

            json.dump(
                evaluation,
                file,
                indent=4,
            )

        print("\nRetrieval Evaluation")
        print("----------------------------")

        for key, value in evaluation.items():

            print(f"{key:<25}: {value}")

        return evaluation

def main():

    retriever = SemanticRetriever(
        model_name=EMBEDDING_MODEL,
        top_k=TOP_K,
    )

    embeddings, metadata = retriever.load_embedding_bank()

    retriever.build_index(
        embeddings,
    )

    query = (
        "Participants rallied behind the victim and her family, "
        "discussed details of her life, and called for stronger "
        "gun-control measures. Some argued that the tragedy could "
        "have been prevented through stricter laws or firearm bans. "
        "Others argued that criminals would obtain guns regardless "
        "of restrictions and that bans are therefore ineffective."
    )

    results = retriever.search(
        query,
        metadata,
    )

    retriever.save_results(
        query,
        results,
    )

    retriever.evaluate_results(
        results,
    )

    print("\nTop Results\n")

    print(
        results[
            [
                "rank",
                "similarity",
                "title",
            ]
        ]
    )


if __name__ == "__main__":
    main()
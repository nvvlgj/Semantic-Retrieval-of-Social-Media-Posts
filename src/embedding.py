"""
Task 2 - Embedding Generation and Storage

Responsibilities
----------------
1. Load cleaned dataset
2. Generate sentence embeddings
3. Save embedding bank
4. Save metadata
5. Save embedding information

Author: Lalitha Nandiraju
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sentence_transformers import SentenceTransformer

from .config import (
    PROCESSED_DATA_PATH,
    EMBEDDING_MODEL,
    BATCH_SIZE,
    NORMALIZE_EMBEDDINGS,
    EMBEDDING_DIRECTORY,
    EMBEDDING_FILE,
    METADATA_FILE,
    EMBEDDING_INFO_FILE,
    TEXT_COLUMN,
)


class EmbeddingGenerator:
    """
    Generates sentence embeddings for the cleaned dataset
    and stores them for semantic retrieval.
    """

    def __init__(
        self,
        model_name: str,
        batch_size: int = 32,
        normalize_embeddings: bool = True,
    ):

        self.model_name = model_name
        self.batch_size = batch_size
        self.normalize_embeddings = normalize_embeddings

        self.device = "cuda" if torch.cuda.is_available() else "cpu"

        print(f"Using device : {self.device}")
        print(f"Loading model : {model_name}")

        self.model = SentenceTransformer(
            model_name,
            device=self.device,
        )

    # -------------------------------------------------------
    # Public API
    # -------------------------------------------------------

    def create_embedding_bank(
        self,
        dataframe: pd.DataFrame,
        text_column: str,
    ) -> np.ndarray:
        """
        Generates embeddings for every document.
        """

        texts = dataframe[text_column].tolist()

        print(f"\nGenerating embeddings for {len(texts)} documents...\n")

        embeddings = self.model.encode(
            texts,
            batch_size=self.batch_size,
            show_progress_bar=True,
            convert_to_numpy=True,
            normalize_embeddings=self.normalize_embeddings,
        )

        embeddings = embeddings.astype(np.float32)

        print("\nEmbedding generation complete.")
        print(f"Embedding shape : {embeddings.shape}")

        return embeddings

    def save_embedding_bank(
        self,
        embeddings: np.ndarray,
        metadata: pd.DataFrame,
    ) -> None:
        """
        Saves embeddings, metadata and embedding information.
        """

        EMBEDDING_DIRECTORY.mkdir(
            parents=True,
            exist_ok=True,
        )

        np.save(
            EMBEDDING_FILE,
            embeddings,
        )

        metadata.to_csv(
            METADATA_FILE,
            index=False,
        )

        self._save_embedding_info(
            embeddings,
            len(metadata),
        )

        print("\nEmbedding bank saved successfully.\n")
        print(f"Embeddings : {EMBEDDING_FILE}")
        print(f"Metadata   : {METADATA_FILE}")
        print(f"Info       : {EMBEDDING_INFO_FILE}")

    def load_embedding_bank(
        self,
    ) -> tuple[np.ndarray, pd.DataFrame]:
        """
        Loads previously generated embeddings.
        """

        embeddings = np.load(
            EMBEDDING_FILE,
        )

        metadata = pd.read_csv(
            METADATA_FILE,
        )

        return embeddings, metadata

    # -------------------------------------------------------
    # Private Helpers
    # -------------------------------------------------------

    def _save_embedding_info(
        self,
        embeddings: np.ndarray,
        number_of_documents: int,
    ) -> None:

        info = {
            "model": self.model_name,
            "embedding_dimension": int(embeddings.shape[1]),
            "documents": number_of_documents,
            "normalized": self.normalize_embeddings,
            "batch_size": self.batch_size,
            "dtype": str(embeddings.dtype),
            "device": self.device,
        }

        with open(
            EMBEDDING_INFO_FILE,
            "w",
            encoding="utf-8",
        ) as file:

            json.dump(
                info,
                file,
                indent=4,
            )


def main():

    print("Loading cleaned dataset...\n")

    dataframe = pd.read_csv(
        PROCESSED_DATA_PATH,
    )

    generator = EmbeddingGenerator(
        model_name=EMBEDDING_MODEL,
        batch_size=BATCH_SIZE,
        normalize_embeddings=NORMALIZE_EMBEDDINGS,
    )

    embeddings = generator.create_embedding_bank(
        dataframe,
        TEXT_COLUMN,
    )

    generator.save_embedding_bank(
        embeddings,
        dataframe,
    )

if __name__ == "__main__":
    main()
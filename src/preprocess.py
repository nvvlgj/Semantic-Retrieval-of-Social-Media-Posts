"""
Task 1 - Data Preprocessing & Normalization

Responsibilities
----------------
1. Load dataset
2. Normalize text
3. Remove empty rows
4. Remove exact duplicates
5. Remove near duplicates
6. Save cleaned dataset
"""

from __future__ import annotations

import re
import unicodedata
from pathlib import Path

import pandas as pd

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

from .config import *

from .semantic_deduplicator import SemanticDeduplicator


class DatasetPreprocessor:

    def __init__(
        self,
        text_column: str,
        remove_urls: bool = True,
        near_duplicate_threshold: float = 0.95,
    ):
        self.text_column = text_column
        self.remove_urls = remove_urls
        self.threshold = near_duplicate_threshold
        self.near_duplicate_pairs = []

        self.stats = {
            "original_rows": 0,
            "missing_removed": 0,
            "exact_duplicates_removed": 0,
            "near_duplicates_removed": 0,
            "final_rows": 0,
        }

    # -------------------------------------------------------
    # Public API
    # -------------------------------------------------------

    def preprocess(
        self,
        input_path: Path,
    ):

        print(f"Loading dataset from {input_path}")

        df = pd.read_csv(input_path)

        self.stats["original_rows"] = len(df)

        df = self._drop_unused_columns(df)
        df = self._clean_text(df)
        df = self._remove_empty_posts(df)
        df = self._remove_exact_duplicates(df)

        if ENABLE_NEAR_DUPLICATE_REMOVAL:
            df = self._remove_near_duplicates(df)

        self.stats["final_rows"] = len(df)

        self._print_summary()

        return df

    # -------------------------------------------------------
    # Cleaning
    # -------------------------------------------------------

    def _drop_unused_columns(self, df):

        # Drop columns that are not needed.
        if "display_url" in df.columns:
            df = df.drop(columns=["display_url"])

        return df

    def _clean_text(self, df):

        df[self.text_column] = (
            df[self.text_column]
            .fillna("")
            .astype(str)
            .apply(self._normalize_text)
        )

        return df

    def _normalize_text(self, text: str):

        text = unicodedata.normalize("NFKC", text)

        if self.remove_urls:
            text = re.sub(r"http\S+|www\S+", "", text)

        text = text.strip()

        text = re.sub(r"\s+", " ", text)

        return text

    # -------------------------------------------------------
    # Exact duplicates
    # -------------------------------------------------------

    def _remove_empty_posts(self, df):

        before = len(df)

        df = df[df[self.text_column] != ""]

        self.stats["missing_removed"] = before - len(df)

        return df

    def _remove_exact_duplicates(self, df):

        before = len(df)

        df = df.drop_duplicates(subset=[self.text_column])

        self.stats["exact_duplicates_removed"] = before - len(df)

        return df

    # -------------------------------------------------------
    # Near duplicates
    # -------------------------------------------------------

    def _remove_near_duplicates(self, df):

        print("Detecting near duplicates...")

        texts = df[self.text_column].tolist()

        vectorizer = TfidfVectorizer(
            stop_words="english",
            ngram_range=(1, 2),
            min_df=2,
        )

        vectors = vectorizer.fit_transform(texts)

        similarity_matrix = cosine_similarity(vectors)

        keep = []
        removed = set()

        self.near_duplicate_pairs = []

        for i in range(len(texts)):

            if i in removed:
                continue

            keep.append(i)

            for j in range(i + 1, len(texts)):

                if j in removed:
                    continue

                similarity = similarity_matrix[i, j]

                if similarity >= self.threshold:

                    removed.add(j)

                    self.near_duplicate_pairs.append({
                        "kept_index": i,
                        "removed_index": j,
                        "similarity": round(float(similarity), 4),

                        "kept_title": df.iloc[i]["title"],
                        "removed_title": df.iloc[j]["title"],

                        "kept_text": texts[i],
                        "removed_text": texts[j]
                    })

        self.stats["near_duplicates_removed"] = len(removed)

        cleaned_df = df.iloc[keep].reset_index(drop=True)

        self._save_near_duplicates()

        return cleaned_df

    def _save_near_duplicates(self):

        if not self.near_duplicate_pairs:
            return

        duplicate_df = pd.DataFrame(self.near_duplicate_pairs)

        output_path = (
            PROJECT_ROOT
            / "data"
            / "analysis"
            / "near_duplicates.csv"
        )

        output_path.parent.mkdir(parents=True, exist_ok=True)

        duplicate_df.to_csv(output_path, index=False)

        print(f"\nSaved {len(duplicate_df)} near duplicate pairs")
        print(f"Location: {output_path}")

    # -------------------------------------------------------
    # Reporting
    # -------------------------------------------------------

    def _print_summary(self):

        print("\n" + "=" * 50)
        print("Preprocessing Summary")
        print("=" * 50)

        for key, value in self.stats.items():
            print(f"{key:<30}: {value}")

        print("=" * 50)

    def save_dataset(
        self,
        dataframe: pd.DataFrame,
        output_path: Path,
    ):

        output_path.parent.mkdir(
            parents=True,
            exist_ok=True,
        )

        dataframe.to_csv(
            output_path,
            index=False,
        )

        print(f"\nSaved cleaned dataset to {output_path}")

def main():

    print("Starting preprocessing pipeline...\n")

    preprocessor = DatasetPreprocessor(
        text_column=TEXT_COLUMN,
        remove_urls=REMOVE_URLS,
        near_duplicate_threshold=NEAR_DUPLICATE_THRESHOLD,
    )

    # Step 1: Text preprocessing
    cleaned_df = preprocessor.preprocess(
        RAW_DATA_PATH,
    )

    # Step 2: Semantic deduplication
    deduplicator = SemanticDeduplicator(
        model_name=EMBEDDING_MODEL,
        similarity_threshold=NEAR_DUPLICATE_THRESHOLD,
    )

    cleaned_df = deduplicator.remove_semantic_duplicates(
        cleaned_df,
        TEXT_COLUMN,
    )

    # Step 3: Save cleaned dataset
    preprocessor.save_dataset(
        cleaned_df,
        PROCESSED_DATA_PATH,
    )

    print("\nPreprocessing pipeline completed successfully.")


if __name__ == "__main__":
    main()
"""
Semantic Retrieval Pipeline

Runs the complete pipeline in the correct order:

1. Preprocessing
2. Semantic Deduplication
3. Embedding Generation
4. Semantic Retrieval
"""

from .preprocess import main as preprocess_main
from .embedding import main as embedding_main
from .retrieval import main as retrieval_main


def main():

    print("=" * 60)
    print("Semantic Retrieval Pipeline")
    print("=" * 60)

    print("\nStep 1: Preprocessing")
    preprocess_main()

    print("\nStep 2: Embedding Generation")
    embedding_main()

    print("\nStep 3: Semantic Retrieval")
    retrieval_main()

    print("\nPipeline completed successfully!")


if __name__ == "__main__":
    main()
import argparse
import json

from utils.file_utils import get_index_paths
from commands.hybrid_search_commands import rrf_search
from utils.types import EvaluationDataset

golden_dataset_path = get_index_paths()["golden_dataset_path"]


def main() -> None:
    parser = argparse.ArgumentParser(description="Search Evaluation CLI")
    parser.add_argument(
        "--limit",
        type=int,
        default=5,
        help="Number of results to evaluate (k for precision@k, recall@k)",
    )

    args = parser.parse_args()
    limit = args.limit

    with golden_dataset_path.open() as dataset:
        dataset_dict: EvaluationDataset = json.load(dataset)
        test_cases = dataset_dict["test_cases"]
        print(f"k={limit}")
        for test_case in test_cases:
            query = test_case["query"]
            result = rrf_search(query, limit, 60, None, None, False)
            relevant_docs = test_case["relevant_docs"]
            ground_truth = {doc.lower().strip() for doc in relevant_docs}
            total_retrieved = len(result)
            total_relevant = len(relevant_docs)
            titles = [movie["title"] for movie in result]
            relevant_retrieved = sum(
                1 for movie in result if movie["title"].lower().strip() in ground_truth
            )
            precision = (
                0.0 if total_retrieved == 0 else (relevant_retrieved / total_retrieved)
            )
            recall = (
                0.0 if total_relevant == 0 else (relevant_retrieved / total_relevant)
            )
            f1 = 2 * (precision * recall) / (precision + recall)
            print(f"- Query: {query}")
            print(f"  - Precision@{limit}: {precision:.4f}")
            print(f"  - Recall@{limit}: {recall:.4f}")
            print(f"  - F1 Score: {f1:.4f}")
            print(f"  - Retrieved: {", ".join(titles)}")
            print(f"  - Relevant: {", ".join(relevant_docs)}")


if __name__ == "__main__":
    main()

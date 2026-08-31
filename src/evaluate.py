from collections import defaultdict

from search import VisualSearchEngine
from utils import load_pickle
from config import METADATA_PKL


def precision_at_k(retrieved, query_category, k):
    if k == 0:
        return 0.0
    relevant = sum(1 for item in retrieved[:k] if item.get("category") == query_category)
    return relevant / k


def recall_at_k(retrieved, query_category, total_relevant, k):
    if total_relevant == 0:
        return 0.0
    relevant = sum(1 for item in retrieved[:k] if item.get("category") == query_category)
    return relevant / total_relevant


def evaluate(k=5, max_queries=100):
    metadata = load_pickle(METADATA_PKL)
    engine = VisualSearchEngine()

    category_counts = defaultdict(int)
    for item in metadata:
        category = item.get("category")
        if category:
            category_counts[category] += 1

    precisions = []
    recalls = []
    evaluated = 0

    for item in metadata:
        image_path = item.get("image_path")
        query_category = item.get("category")

        if not image_path or not query_category:
            continue

        results = engine.search(image_path, top_k=k + 1)
        results = [r for r in results if r.get("image_path") != image_path][:k]

        total_relevant = max(category_counts[query_category] - 1, 0)

        p_at_k = precision_at_k(results, query_category, k)
        r_at_k = recall_at_k(results, query_category, total_relevant, k)

        precisions.append(p_at_k)
        recalls.append(r_at_k)

        evaluated += 1
        if evaluated >= max_queries:
            break

    avg_precision = sum(precisions) / len(precisions) if precisions else 0
    avg_recall = sum(recalls) / len(recalls) if recalls else 0

    print(f"Evaluated queries: {evaluated}")
    print(f"Average Precision@{k}: {avg_precision:.4f}")
    print(f"Average Recall@{k}: {avg_recall:.4f}")


if __name__ == "__main__":
    evaluate(k=5, max_queries=100)
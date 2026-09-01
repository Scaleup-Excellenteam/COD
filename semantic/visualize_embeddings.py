import argparse
import json
from pathlib import Path

import numpy as np
import plotly.graph_objects as go
from sklearn.manifold import TSNE


def load_embeddings(path: Path, limit: int | None = None):
    records = []

    with path.open("r", encoding="utf-8") as file:
        for line in file:
            if limit is not None and len(records) >= limit:
                break

            record = json.loads(line)
            records.append(record)

    if not records:
        raise ValueError("No embeddings found.")

    embeddings = np.asarray(
        [record["embedding"] for record in records],
        dtype=np.float32,
    )

    return records, embeddings


def reduce_to_3d(embeddings: np.ndarray) -> np.ndarray:
    # t-SNE perplexity must be smaller than the number of samples.
    perplexity = min(30, max(2, len(embeddings) - 1))

    tsne = TSNE(
        n_components=3,
        perplexity=perplexity,
        random_state=42,
        init="pca",
        learning_rate="auto",
    )

    return tsne.fit_transform(embeddings)


def visualize(records, points_3d: np.ndarray):
    hover_text = [
        (
            f"<b>{record['sentence']}</b><br>"
            f"Source: {record['source_text']}<br>"
            f"Offset: {record['offset']}"
        )
        for record in records
    ]

    figure = go.Figure(
        data=[
            go.Scatter3d(
                x=points_3d[:, 0],
                y=points_3d[:, 1],
                z=points_3d[:, 2],
                mode="markers",
                text=hover_text,
                hoverinfo="text",
                marker={"size": 5},
            )
        ]
    )

    figure.update_layout(
        title="Semantic Embeddings — 3D t-SNE",
        scene={
            "xaxis_title": "t-SNE 1",
            "yaxis_title": "t-SNE 2",
            "zaxis_title": "t-SNE 3",
        },
    )

    figure.show()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--input",
        type=Path,
        default=Path("data/semantic_embeddings.jsonl"),
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
    )

    args = parser.parse_args()

    records, embeddings = load_embeddings(args.input, args.limit)

    print(f"Loaded: {len(records)} sentences")
    print(f"Original dimensions: {embeddings.shape[1]}")
    print("Reducing to 3D with t-SNE...")

    points_3d = reduce_to_3d(embeddings)

    visualize(records, points_3d)


if __name__ == "__main__":
    main()
    
#uv run python -m semantic.visualize_embeddings
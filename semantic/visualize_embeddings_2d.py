import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.express as px
from sklearn.cluster import KMeans
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
        raise ValueError("No embedding records found.")

    embeddings = np.asarray(
        [record["embedding"] for record in records],
        dtype=np.float32,
    )

    return records, embeddings


def reduce_to_2d(embeddings: np.ndarray) -> np.ndarray:
    sample_count = len(embeddings)
    perplexity = min(30, max(2, sample_count - 1))

    reducer = TSNE(
        n_components=2,
        perplexity=perplexity,
        random_state=42,
        init="pca",
        learning_rate="auto",
    )

    return reducer.fit_transform(embeddings)


def cluster_embeddings(embeddings: np.ndarray, cluster_count: int) -> np.ndarray:
    if cluster_count < 1:
        raise ValueError("cluster_count must be positive.")

    if cluster_count > len(embeddings):
        raise ValueError(
            f"cluster_count ({cluster_count}) cannot be larger than the number of records ({len(embeddings)})."
        )

    model = KMeans(n_clusters=cluster_count, random_state=42, n_init=10)
    return model.fit_predict(embeddings)


def build_dataframe(records, points_2d: np.ndarray, cluster_labels: np.ndarray) -> pd.DataFrame:
    rows = []

    for record, point, cluster_label in zip(records, points_2d, cluster_labels):
        rows.append(
            {
                "x": point[0],
                "y": point[1],
                "topic": f"Topic {cluster_label + 1}",
                "sentence": record["sentence"],
                "source_text": record["source_text"],
                "offset": record["offset"],
            }
        )

    return pd.DataFrame(rows)


def visualize(dataframe: pd.DataFrame, output_html: Path | None = None) -> None:
    figure = px.scatter(
        dataframe,
        x="x",
        y="y",
        color="topic",
        hover_data={
            "sentence": True,
            "source_text": True,
            "offset": True,
            "x": False,
            "y": False,
        },
        title="Semantic Embeddings — 2D Topic View",
    )

    figure.update_traces(marker={"size": 9})
    figure.update_layout(
        xaxis_title="t-SNE Dimension 1",
        yaxis_title="t-SNE Dimension 2",
        legend_title="Clustered Topics",
    )

    if output_html is not None:
        figure.write_html(output_html)
        print(f"Saved interactive visualization to: {output_html}")

    figure.show()


def main():
    parser = argparse.ArgumentParser(description="Visualize semantic embeddings in 2D.")
    parser.add_argument(
        "--input",
        type=Path,
        default=Path("data/semantic_embeddings.jsonl"),
        help="Path to the embeddings JSONL file.",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Optional number of records to load.",
    )
    parser.add_argument(
        "--clusters",
        type=int,
        default=6,
        help="Number of topic clusters to color.",
    )
    parser.add_argument(
        "--output-html",
        type=Path,
        default=None,
        help="Optional path to save an interactive HTML file.",
    )

    args = parser.parse_args()

    records, embeddings = load_embeddings(args.input, args.limit)
    print(f"Loaded {len(records)} embeddings with dimension {embeddings.shape[1]}")

    cluster_labels = cluster_embeddings(embeddings, args.clusters)
    print(f"Grouped into {args.clusters} topic clusters")

    points_2d = reduce_to_2d(embeddings)
    dataframe = build_dataframe(records, points_2d, cluster_labels)

    visualize(dataframe, args.output_html)


if __name__ == "__main__":
    main()
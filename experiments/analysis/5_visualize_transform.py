import os
import random
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
from codebase.load_data.config import DATASETS, DATASETS_FREQ

ORIGINAL_DATA_DIR = Path("./assets/original_data")
TRANSFORMED_DATA_DIR = Path("./assets/transformed_data")

N_SAMPLES = 3


def plot_transformed_vs_original(dataset_name: str, group: str, transformation: str, version: int, n_samples: int = N_SAMPLES):
    """Plots a few series from the original and transformed datasets for comparison."""

    data_cls = DATASETS[dataset_name]
    try:
        original_ds = data_cls.load_data(group)
    except FileNotFoundError as e:
        print(f"Original data for {dataset_name} - {group} not found: {e}")
        return

    transformed_file_path = TRANSFORMED_DATA_DIR / dataset_name / f"{dataset_name}_{group}_{transformation}_v{version}.csv"
    try:
        transformed_ds = pd.read_csv(transformed_file_path)
    except FileNotFoundError as e:
        print(f"Transformed data for {dataset_name} - {group} - {transformation} - v{version} not found: {e}")
        return

    original_ds['ds'] = pd.to_datetime(original_ds['ds'], errors='coerce')
    transformed_ds['ds'] = pd.to_datetime(transformed_ds['ds'], errors='coerce')

    # randomly select a few series (unique_ids) to visualize
    unique_ids = original_ds['unique_id'].unique()
    selected_ids = random.sample(list(unique_ids), min(n_samples, len(unique_ids)))

    fig, axes = plt.subplots(n_samples, 1, figsize=(10, 4 * n_samples))
    fig.suptitle(f"Original vs Transformed ({transformation} v{version}) - {dataset_name} - {group}", fontsize=16)

    for i, unique_id in enumerate(selected_ids):
        original_series = original_ds[original_ds['unique_id'] == unique_id]
        transformed_series = transformed_ds[transformed_ds['unique_id'] == unique_id]

        ax = axes[i] if n_samples > 1 else axes
        ax.plot(original_series['ds'], original_series['y'], label='Original', color='blue')
        ax.plot(transformed_series['ds'], transformed_series['y'], label='Transformed', color='red', linestyle='--')

        ax.set_title(f"Series {unique_id}")
        ax.set_xlabel("Date")
        ax.set_ylabel("Value")
        ax.legend()
        ax.grid(True)

    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.show()


def main():
    """Main function to visualize transformed vs original datasets."""

    transformations = ["jitter", "scaling", "magnitude_warp", "time_warp"]
    n_versions = 6

    for data_name, groups in DATASETS_FREQ.items():
        for group in groups:
            for transformation in transformations:
                for version in range(1, n_versions + 1):
                    print(f"Visualizing {data_name} - {group} | Transformation: {transformation}, Version: {version}")
                    plot_transformed_vs_original(data_name, group, transformation, version, n_samples=N_SAMPLES)


if __name__ == "__main__":
    main()

import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.preprocessing import StandardScaler
from codebase.load_data.config import DATASETS, DATASETS_FREQ
from rhiots.transformations import ManipulateData

ORIGINAL_DATA_DIR = Path("assets/original_data")
OUTPUT_DIR = Path("assets/transformed_data")


def transform_and_save(
    dataset_name: str, group: str, freq: str, transformation: str, version: int
):
    """Transforms each series in the dataset independently and saves in long format"""

    output_file = (
        OUTPUT_DIR
        / dataset_name
        / f"{dataset_name}_{group}_{transformation}_v{version}.csv"
    )

    if output_file.exists():
        print(
            f"Transformed dataset for {dataset_name} - {group} - {transformation} - v{version} already exists. Skipping..."
        )
        return

    data_cls = DATASETS[dataset_name]
    ds = data_cls.load_data(group)

    transformed_data = []

    # apply transformations to each series independently
    for unique_id in ds["unique_id"].unique():
        series_data = ds[ds["unique_id"] == unique_id].copy()

        series_values = series_data[["y"]].values  # extract `y` as 2D array
        scaler = StandardScaler()
        scaled_values = scaler.fit_transform(series_values)

        manipulator = ManipulateData(
            x=scaled_values,
            transformation=transformation,
            parameters=[0.3, 0.5, 0.1, 0.1],
            version=version,
        )
        transformed_scaled_series = manipulator.apply_transf()

        transformed_series = scaler.inverse_transform(transformed_scaled_series)

        transformed_df = pd.DataFrame(
            {
                "unique_id": unique_id,
                "ds": (
                    series_data["ds"]
                    if dataset_name == "M4"
                    else series_data["ds"].dt.date.values
                ),
                "y": transformed_series.flatten(),
            }
        )

        transformed_data.append(transformed_df)

    transformed_long_df = pd.concat(transformed_data, ignore_index=True)

    output_path = OUTPUT_DIR / dataset_name
    output_path.mkdir(parents=True, exist_ok=True)
    original_path = ORIGINAL_DATA_DIR / dataset_name
    original_path.mkdir(parents=True, exist_ok=True)
    transformed_long_df.to_csv(
        output_path / f"{dataset_name}_{group}_{transformation}_v{version}.csv",
        index=False,
    )
    ds.to_csv(original_path / f"{dataset_name}_{group}.csv", index=False)


def main():
    """Main function to loop through datasets and apply transformations."""

    transformations = ["jitter", "scaling", "magnitude_warp", "time_warp"]
    n_versions = 6

    for data_name, groups in DATASETS_FREQ.items():
        for group in groups:
            data_cls = DATASETS[data_name]
            freq = data_cls.frequency_pd[group]

            for transformation in transformations:
                for version in range(1, n_versions + 1):
                    print(
                        f"Transforming {data_name} - {group} with {transformation} version {version}"
                    )
                    transform_and_save(data_name, group, freq, transformation, version)


if __name__ == "__main__":
    main()

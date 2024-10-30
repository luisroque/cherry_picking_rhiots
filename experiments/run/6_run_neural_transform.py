import os
import pandas as pd
from neuralforecast import NeuralForecast
from neuralforecast.auto import (
    AutoRNN,
    AutoTCN,
    AutoDeepAR,
    AutoNHITS,
    AutoTiDE,
    AutoInformer,
)
from codebase.load_data.config import DATASETS, DATASETS_FREQ

TRANSFORMED_DATA_DIR = "./assets/transformed_data"
RESULTS_DIR = "./assets/results/transformed_by_group"

os.makedirs(RESULTS_DIR, exist_ok=True)

for data_name, groups in DATASETS_FREQ.items():
    for group in groups:
        data_cls = DATASETS[data_name]
        h = data_cls.horizons_map[group]
        n_lags = data_cls.context_length[group]

        if data_name == "M4":
            freq = data_cls.frequency_map.get(group)
        else:
            freq = data_cls.frequency_pd[group]
        season_len = data_cls.frequency_map[group]

        transformations = ["jitter", "scaling", "magnitude_warp", "time_warp"]
        n_versions = 6

        for transformation in transformations:
            for version in range(1, n_versions + 1):
                print(
                    f"Processing {data_name} - {group} | Transformation: {transformation}, Version: {version}"
                )

                transformed_file_path = f"{TRANSFORMED_DATA_DIR}/{data_name}/{data_name}_{group}_{transformation}_v{version}.csv"
                output_file_path = f"{RESULTS_DIR}/{data_name}_{group}_{transformation}_v{version}_neural.csv"

                if os.path.exists(output_file_path):
                    print(
                        f"Output file for {data_name} - {group} - {transformation} - v{version} already exists. Skipping..."
                    )
                    continue

                try:
                    ds = pd.read_csv(
                        transformed_file_path,
                        dtype={"unique_id": str, "y": float},
                        parse_dates=["ds"],
                    )
                except FileNotFoundError as e:
                    print(
                        f"Error loading transformed data for {data_name} - {group} - {transformation} - v{version}: {e}"
                    )
                    continue

                n_series = ds["unique_id"].nunique()

                models = [
                    AutoRNN(h=h, cpus=15, gpus=0),
                    AutoTCN(h=h, cpus=15, gpus=0),
                    AutoDeepAR(h=h, cpus=15, gpus=0),
                    AutoNHITS(h=h, cpus=15, gpus=0),
                    AutoTiDE(h=h, cpus=15, gpus=0),
                    AutoInformer(h=h, cpus=15, gpus=0),
                ]

                nf = NeuralForecast(models=models, freq=freq)

                try:
                    cv_nf = nf.cross_validation(df=ds, test_size=h, n_windows=None)
                    cv_nf = cv_nf.reset_index()

                    cv_nf.to_csv(output_file_path, index=False)
                    print(
                        f"Results saved for {data_name} - {group} - {transformation} - v{version}"
                    )

                except Exception as e:
                    print(
                        f"Error during cross-validation for {data_name} - {group} - {transformation} - v{version}: {e}"
                    )

import os
import numpy as np
import pandas as pd


class RopeAnalysis:

    @staticmethod
    def get_vector_probs(diff_vec: pd.Series, rope: float):
        left = (diff_vec < -rope).mean()
        right = (diff_vec > rope).mean()
        mid = np.mean([-rope < x_ < rope for x_ in diff_vec])

        return left, mid, right

    @classmethod
    def get_probs(cls, results: pd.DataFrame, rope: float, reference: str):
        output_dir = "./assets/metrics/by_group/"
        os.makedirs(output_dir, exist_ok=True)

        # get the dataset
        results["Dataset"] = results["Series"].str.extract(r"([a-zA-Z0-9]+)")
        results["Dataset_Frequency"] = results["Series"].str.extract(
            r"([a-zA-Z0-9]+_[a-zA-Z0-9])"
        )

        dataset_names = "_".join(results["Dataset"].unique())
        if rope == 0:
            output_path = os.path.join(output_dir, f"{dataset_names}_rope0.csv")
        else:
            output_path = os.path.join(output_dir, f"{dataset_names}_rope.csv")
        if os.path.exists(output_path):
            print(f"File {output_path} already exists. Loading existing data.")
            return pd.read_csv(output_path)
        results_pd = cls.calc_percentage_diff(results, reference)

        prob_df = results_pd.apply(lambda x: cls.get_vector_probs(x, rope=rope), axis=0)
        prob_df = prob_df.T.reset_index()

        outcome_names = [f"{reference} loses", "draw", f"{reference} wins"]

        prob_df.columns = ["Method"] + outcome_names

        loc = prob_df.query(f'Method=="{reference}"').index[0]

        prob_df = prob_df.drop(loc).reset_index(drop=True)

        df_melted = prob_df.melt("Method")
        df_melted["variable"] = pd.Categorical(
            df_melted["variable"], categories=outcome_names
        )

        df_melted.columns = ["Model", "Result", "Probability"]
        df_melted.to_csv(output_path, index=False)
        print(f"Storing ROPE on {output_path}")

        return df_melted

    @classmethod
    def calc_percentage_diff(cls, results: pd.DataFrame, reference: str):
        results_pd = results.copy()
        results_pivot = results_pd.pivot(
            index="Series", columns="Model", values="Error"
        )
        if reference not in results_pivot.columns:
            raise ValueError(
                f"Reference model '{reference}' not found in the DataFrame columns."
            )

        percentage_diff_df = results_pivot.apply(
            lambda row: cls.percentage_diff(row, row[reference]), axis=1
        )
        return percentage_diff_df

    @staticmethod
    def percentage_diff(x, y):
        return ((x - y) / abs(y)) * 100

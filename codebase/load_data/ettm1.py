import pandas as pd
from datasetsforecast.long_horizon import LongHorizon

from codebase.load_data.base import LoadDataset


class ETTm1Dataset(LoadDataset):
    DATASET_NAME = "ETTm1"

    @classmethod
    def load_data(cls, group):
        self = cls()
        ds, *_ = LongHorizon.load(cls.DATASET_PATH, group=self.DATASET_NAME)
        ds["ds"] = pd.to_datetime(ds["ds"])
        return ds
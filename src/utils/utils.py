import os
from typing import List

import clhs as cl
import pandas as pd
from tqdm import tqdm


def get_cLHS_samples(dataf: pd.DataFrame, features: list, nb_samples: int) -> List[int]:
    """Function use to get a sample of LSOAs based on a set of features using a CLHS algorithm"""

    # cLHS
    sampled = cl.clhs(dataf[features], nb_samples, max_iterations=10000)
    tqdm._instances.clear()
    return list(sampled["sample_indices"])


def load_data(full_path):
    """Load data"""
    dataf = pd.read_csv(full_path, index_col=0)
    dataf.reset_index(inplace=True)
    return dataf


def get_validation_data(case):
    path_validation_data = r"D:\OneDrive - Cardiff University\04 - Projects\20 - UKERC\00 - data\validation data"
    file = "House temp.xlsx"
    dataf: pd.DataFrame = pd.read_excel(
        path_validation_data + os.path.sep + file, sheet_name=case, index_col=0
    )
    dataf = dataf.loc[dataf.index.dropna(), "Indoor temperature (C)"].to_frame()
    dataf.rename(
        columns={"Indoor temperature (C)": "Indoor_temperature_degreeC",}, inplace=True
    )

    return dataf


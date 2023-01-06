import datetime as dt
import os
from dataclasses import dataclass, field
from typing import Callable, Dict, List, Tuple

import numpy as np
import pandas as pd

from ..utils import folder_system
from . import dwelling_functions
from . import dwellings_model as dm
from . import enums, errors

TemperatureFunction = Callable[[int], float]


@dataclass
class FlexibilityModel:

    lsoa_dataf: pd.DataFrame
    folder_system: folder_system.FolderSystem
    capacity_factor: int = 0
    outside_air_temperature: float = 0
    efficiency_heating_system: float = 3  # cop of heating systems
    nb_dwellings_per_cluster: int = 5  # avg number of dwellings per cluster
    nb_slices: int = 10  # to split the analysis in multiple chunks to avoid creating very large results files
    scenario: enums.Scenario = enums.Scenario.HP100
    thermal_capacity_level: str = "medium"
    temperature_func: TemperatureFunction = lambda x: 19.0
    temperature_threshold: float = 18
    demand_increase = False

    def set_temperature_func(self,
                             use_pdf: bool,
                             mean_indoor_temperature: float = 19.0) -> None:
        if use_pdf:
            (
                self.temperature_func,
                self.temperature_threshold,
                self.demand_increase,
            ) = dwelling_functions.get_pdf_inside_air_temperature_func(
                self.capacity_factor)
        else:
            (
                self.temperature_func,
                self.temperature_threshold,
                self.demand_increase,
            ) = dwelling_functions.get_mean_inside_air_temperature_func(
                self.capacity_factor,
                mean_indoor_temperature=mean_indoor_temperature)

    def get_number_dwellings(self, index_lsoas: List[int]) -> int:
        nb_dwellings_cols: List[str] = [
            c for c in self.lsoa_dataf.columns
            if "Number of" in c and "2018" in c
        ]
        return self.lsoa_dataf.loc[index_lsoas, nb_dwellings_cols].sum().sum()

    def get_number_clusters(self, nb_dwellings: int) -> int:
        nb_clusters = int(
            round(nb_dwellings / self.nb_dwellings_per_cluster, 0))
        if nb_clusters < 1:
            raise errors.NumberClustersOverflowError(
                nb_clusters=nb_clusters,
                nb_dwellings=nb_dwellings,
                message=
                f"The number of clusters is too low. Number of clusters is {nb_clusters} and there are {nb_dwellings} dwellings.",
            )
        return nb_clusters

    def run_model(
            self, index_lsoas: List[int],
            list_methods: List[enums.Method]) -> Tuple[dm.Dwellings, int, int]:
        """Create and run a Dwellings object based on a list of lsoas"""

        nb_dwellings = self.get_number_dwellings(index_lsoas)
        nb_clusters = self.get_number_clusters(nb_dwellings)

        dwellings_obj = dm.Dwellings(
            index_lsoas,
            self.temperature_threshold,
            self.capacity_factor,
            self.demand_increase,
            scenario=self.scenario,
            thermal_capacity_level=self.thermal_capacity_level,
            outside_air_temperature=self.outside_air_temperature,
        )

        if enums.Method.INDIVIDUAL in list_methods and len(list_methods) == 1:
            dwellings_obj = (dwellings_obj.create_dwellings(
                self.lsoa_dataf).add_variables(
                    self.temperature_func).get_duration_service(
                        enums.Method.INDIVIDUAL).concat_results())
        elif (enums.Method.INDIVIDUAL in list_methods
              and enums.Method.REPRESENTATIVE in list_methods
              and len(list_methods) == 2):
            dwellings_obj = (
                dwellings_obj.create_dwellings(self.lsoa_dataf).add_variables(
                    self.temperature_func).get_duration_service(
                        enums.Method.INDIVIDUAL).cluster_dwellings(nb_clusters)
                .create_representative_dwellings_df(
                    enums.Keyword.SUM).get_duration_service(
                        enums.Method.REPRESENTATIVE).concat_results())
        elif enums.Method.REPRESENTATIVE in list_methods and len(
                list_methods) == 1:
            dwellings_obj = (dwellings_obj.create_dwellings(
                self.lsoa_dataf).add_variables(
                    self.temperature_func).cluster_dwellings(
                        nb_clusters).create_representative_dwellings_df(
                            enums.Keyword.SUM).get_duration_service(
                                enums.Method.REPRESENTATIVE).concat_results())
        else:

            print(f"the methods {list_methods} is/are not recognised")

        return dwellings_obj, nb_dwellings, nb_clusters

    def get_iter(self, by_LA: bool, nb_lsoas: int) -> Dict[str, List[int]]:
        """Return a dictionnary where keys are the name of the columns and values the list of indices of LSOAs"""
        self.lsoa_dataf["Index"] = self.lsoa_dataf.index
        if by_LA:
            index_map = (self.lsoa_dataf.groupby("Local Authority")
                         ["Index"].apply(list).to_dict())
        else:
            index_arr = self.lsoa_dataf["Index"].values
            index_arr = np.array_split(
                index_arr,
                len(index_arr) /
                nb_lsoas)  # split the index_arr into multiple arrays
            index_map = {
                f"{list(temp_arr)}": list(temp_arr)
                for temp_arr in index_arr
            }

        return index_map

    def run_model_multiple_lsoas(
        self,
        methods: List[enums.Method],
        by_LA: bool = True,
        nb_lsoas: int = 1,
        filename: str = "",
    ) -> pd.DataFrame:
        """Run the model for all LSOAs in the input dataframe:dataf
        nb_slices: number of chunks in which the original dataset is split into. This is to avoid to create very large csv results file.
        nb_lsoas: number of lsoas that are inputted in the model at the same time (e.g., default is 1 by 1)
        by_LA: run the model by grouping LSOAs by LAs (nb_lsoas is ignored in that case)
        """
        index_lsoa_map = self.get_iter(by_LA, nb_lsoas)
        time_start = dt.datetime.now()

        map_keys = np.array([*index_lsoa_map
                             ])  # convert dict keys into an array
        result_df = pd.DataFrame()

        for slice, list_keys in enumerate(
                np.array_split(map_keys, self.nb_slices)):
            time_start_loop = dt.datetime.now()
            final_frames: List[pd.DataFrame] = []
            for key in list_keys:
                index_lsoas = index_lsoa_map[key]

                print(
                    f"Create Dwellings object based on {len(index_lsoas)} lsoas"
                )
                dwellings_obj, unused_var1, unused_var2 = self.run_model(
                    index_lsoas, methods)
                temp_df = dwellings_obj.concat_results_df[
                    "Flexibility_provided_(Individual_dwellings)_kW"].to_frame(
                    )
                temp_df = temp_df[temp_df.abs() > 0].dropna()
                temp_df.index = np.round(temp_df.index, 0)
                temp_df.columns = [key]
                final_frames.append(temp_df)
            time_end_loop = dt.datetime.now()
            delta_time = time_end_loop - time_start_loop
            print(
                f"running the model for slice {slice} took a total of {delta_time} or {delta_time.seconds}s"
            )
            full_path = self.folder_system.get_path_with_filename(
                slice, self.capacity_factor, self.scenario, filename)
            print(f"The results are stored in {full_path}")
            if len(final_frames) > 0:
                result_df = pd.concat(final_frames, axis=1)
            else:
                result_df = final_frames[0]

            result_df.to_csv(full_path)
        time_end = dt.datetime.now()
        print(
            f"running the model for took a total of {time_end-time_start} or {(time_end-time_start).seconds}s"
        )
        return result_df

    def calculate_accuracy_methods(
        self,
        list_lsoas: List[int],
    ) -> pd.DataFrame:
        # Monte Carlo type of method: picked X number of randomly selected LSOA and calculate the duration of service
        # using the three methods: individual dwellings, agg dwelling and representative dwellings.
        # the accuracy scores are calculated at each step and aggregated together
        frame_score = []

        for index_lsoas in list_lsoas:

            dwellings_obj, nb_dwellings, nb_clusters = self.run_model(
                [index_lsoas],
                [enums.Method.INDIVIDUAL, enums.Method.REPRESENTATIVE],
            )

            temp_scores_df = dwellings_obj.calculate_scores()
            temp_scores_df[
                "Nb_dwellings_per_cluster"] = self.nb_dwellings_per_cluster
            temp_scores_df["Nb_dwellings"] = nb_dwellings
            temp_scores_df["Nb_clusters"] = nb_clusters
            temp_scores_df["LSOA_index"] = index_lsoas
            frame_score.append(temp_scores_df)

        final_scores_df = pd.concat(frame_score).reset_index()

        return final_scores_df

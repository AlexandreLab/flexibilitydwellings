import datetime as dt
import types
from dataclasses import dataclass, field
from typing import Callable, List, Type

import numpy as np
import pandas as pd
from sklearn.cluster import MiniBatchKMeans
from sklearn.preprocessing import StandardScaler

from . import dwelling_functions, scores_functions
# import flexibility.thermal_model as tm
from .enums import Keyword, Method, Scenario


@dataclass
class Dwellings:
    list_lsoas: List[int]
    limit_indoor_air_temperature: float = 24
    capacity_factor: float = 0
    demand_increase: bool = True
    scenario: Scenario = Scenario.HP100
    categories: List[str] = field(default_factory=lambda: [
        "flat oil boiler",
        "detached gas boiler",
        "detached resistance heating",
        "detached oil boiler",
        "detached biomass boiler",
        "semi-detached gas boiler",
        "semi-detached resistance heating",
        "semi-detached oil boiler",
        "semi-detached biomass boiler",
        "terraced gas boiler",
        "terraced resistance heating",
        "terraced oil boiler",
        "terraced biomass boiler",
        "flat gas boiler",
        "flat resistance heating",
        "flat biomass boiler",
    ])

    infinite_duration_value: int = 2 * 24 * 60 * 60  # duration of services is limited to 24 hours
    outside_air_temperature: float = 5
    nb_dwellings: int = 0
    thermal_capacity_level: str = "medium"
    initial_indoor_air_temperature: float = 19
    target_indoor_air_temperature: float = 21  # ideal indoor air temperature (used to size the heating system)
    efficiency_heating_system: float = 0
    dwellings_df: pd.DataFrame = pd.DataFrame()
    representative_dwellings_df: pd.DataFrame = pd.DataFrame(
    )  # store the parameters and results of the representative dwellings based on the clustering method
    summary_representative_dwellings: pd.DataFrame = pd.DataFrame(
    )  # summary of the results from using representative dellings
    summary_individual_dwellings: pd.DataFrame = pd.DataFrame(
    )  # summary of the results from using individual dellings
    concat_results_df: pd.DataFrame = pd.DataFrame()
    temp_df: pd.DataFrame = pd.DataFrame(
    )  # store the results of the get duration function

    def __post_init__(self) -> None:
        self.efficiency_heating_system: float = dwelling_functions.get_heating_systems_efficiency(
            self.outside_air_temperature)  # COP of ASHPs

    def get_scenario_correction_factor(self) -> float:
        factor = 1
        if self.scenario is Scenario.HP75:
            factor = 0.75
        elif self.scenario is Scenario.HP50:
            factor = 0.5
        return factor

    # @logger.log_time_class
    def create_dwellings(self, dataf: pd.DataFrame, func_C=None, func_R=None):
        # Create dataset of individual dwellings based on the data from the dwelling categories
        frames = []
        factor = self.get_scenario_correction_factor()
        for lsoa_index in self.list_lsoas:
            for cat in self.categories:
                number_col = f"Number of {cat} in 2018"
                temp_thermal_losses_col = f"Average thermal losses {cat} kW/K"
                temp_thermal_cap_col = (
                    f"Average {self.thermal_capacity_level} thermal capacity {cat} kJ/K"
                )
                nb_val = int(
                    round(dataf.loc[lsoa_index, number_col] * factor, 0))
                if nb_val > 0:
                    lsoa_name = dataf.loc[lsoa_index, "LSOA11CD"]
                    design_temp_val = dataf.loc[lsoa_index,
                                                "Design_temperature_degreeC"]
                    thermal_cap_val = dataf.loc[lsoa_index,
                                                temp_thermal_cap_col]  # kJ/K
                    thermal_loss_val = 1 / (
                        dataf.loc[lsoa_index, temp_thermal_losses_col])  # K/kW
                    #                     print(f'{c} {nb_val}, {thermal_loss_val} and {thermal_cap_val}')
                    characteristics = [
                        lsoa_name,
                        cat,
                        thermal_cap_val,
                        thermal_loss_val,
                        design_temp_val,
                    ]
                    stock_characteristics = [
                        characteristics for _ in range(nb_val)
                    ]
                    self.nb_dwellings += nb_val
                    frames.append(stock_characteristics)
        frames = [l for sublist in frames for l in sublist]
        self.dwellings_df = pd.DataFrame(
            frames,
            columns=[
                "LSOA",
                "Dwelling_category",
                "C_kJ/K",
                "R_K/kW",
                "Design_temperature_ºC",
            ],
        )
        self.dwellings_df.index.name = "Index"
        if isinstance(func_C, types.LambdaType):
            self.dwellings_df["C_kJ/K"] = [
                func_C(t) for t in self.dwellings_df["C_kJ/K"]
            ]

        if isinstance(func_R, types.LambdaType):
            self.dwellings_df["R_K/kW"] = [
                func_R(t) for t in self.dwellings_df["R_K/kW"]
            ]

        return self

    def set_dwellings_df(self, dataf: pd.DataFrame):
        """Replace the dataset of individual dwellings"""
        # the columns of dataf should include at minimum:
        # "C_kJ/K",
        # "R_K/kW",
        # "Design_temperature_ºC",
        self.dwellings_df = dataf.copy()
        return self

    # @logger.log_time_class
    def create_representative_dwellings_df(self, key: Keyword):
        """Create representative dwellings of each cluster. Specific aggregation method can be selected usings UPPER, LOWER and AVG"""

        dataf = self.get_parameters_representative_dwellings()

        # default values are the aggregated values
        columns_to_extract = [
            "agg_C_kJ/K",
            "agg_R_K/kW",
            "agg_Initial_indoor_temperature_ºC",
            "agg_Outdoor_temperature_ºC",
            "agg_Design_temperature_ºC",
        ]
        if self.demand_increase:
            # Demand increase
            if key is Keyword.UPPER:
                # upper band
                # the lower R is the higher the amount of flexibility can be provided
                # The higher C is the longer the amount of flexibility can be provided
                # The higher the initial indoor temperature is, the more heating output can be increased
                # the lower the outside air temperature, the longer it will take to heat up the dwelling
                # the lower the design temperature, the larger the heating system installed will be
                columns_to_extract = [
                    "max_C_kJ/K",
                    "min_R_K/kW",
                    "min_Initial_indoor_temperature_ºC",
                    "min_Outdoor_temperature_ºC",
                    "min_Design_temperature_ºC",
                ]
            elif key is Keyword.LOWER:
                # lower band
                # the higher R is the lower the amount of flexibility can be provided
                # The lower C is the longer the amount of flexibility can be provided
                # The higher the initial indoor temperature is, the more heating output can be increased
                # the higher the outside air temperature, the longer it will take to heat up the dwelling
                # the higher the design temperature, the larger the heating system installed will be
                columns_to_extract = [
                    "min_C_kJ/K",
                    "max_R_K/kW",
                    "max_Initial_indoor_temperature_ºC",
                    "max_Outdoor_temperature_ºC",
                    "max_Design_temperature_ºC",
                ]
            elif key is Keyword.SUM:
                columns_to_extract = [
                    "agg_C_kJ/K",
                    "agg_R_K/kW",
                    "agg_Initial_indoor_temperature_ºC",
                    "agg_Outdoor_temperature_ºC",
                    "agg_Design_temperature_ºC",
                ]
        else:
            # Demand reduction
            if key is Keyword.UPPER:
                # upper band
                # the higher R is the lower the amount of flexibility can be provided
                # The lower C is the longer the amount of flexibility can be provided
                # The lower the initial indoor temperature is, the less the heating output can be decreased
                # the lower the outside air temperature, the faster it will take to cool down the dwelling
                # the lower the design temperature, the larger the heating system installed will be
                columns_to_extract = [
                    "min_C_kJ/K",
                    "max_R_K/kW",
                    "min_Initial_indoor_temperature_ºC",
                    "min_Outdoor_temperature_ºC",
                    "min_Design_temperature_ºC",
                ]
            elif key is Keyword.LOWER:
                # lower band
                # the lower R is the higher the amount of flexibility can be provided
                # The higher C is the longer the amount of flexibility can be provided
                # The higher the initial indoor temperature is, the more heating output can be decreased
                # the higher the outside air temperature, the longer it will take to cool down the dwelling
                # the higher the design temperature, the larger the heating system installed will be
                columns_to_extract = [
                    "max_C_kJ/K",
                    "min_R_K/kW",
                    "max_Initial_indoor_temperature_ºC",
                    "max_Outdoor_temperature_ºC",
                    "max_Design_temperature_ºC",
                ]
            elif key is Keyword.SUM:
                columns_to_extract = [
                    "agg_C_kJ/K",
                    "agg_R_K/kW",
                    "agg_Initial_indoor_temperature_ºC",
                    "agg_Outdoor_temperature_ºC",
                    "agg_Design_temperature_ºC",
                ]

        representative_dwellings_df = dataf[columns_to_extract +
                                            ["Cluster"]].copy()
        representative_dwellings_df.columns = (
            representative_dwellings_df.columns.str.replace(
                "max_", "").str.replace("min_", "").str.replace("agg_", ""))

        representative_dwellings_df["Capacity_factor_%"] = self.capacity_factor
        representative_dwellings_df["Keyword"] = key.name
        representative_dwellings_df[
            "Final_indoor_temperature_ºC"] = self.limit_indoor_air_temperature
        representative_dwellings_df[
            "Org_Initial_indoor_temperature_ºC"] = representative_dwellings_df[
                "Initial_indoor_temperature_ºC"]
        self.representative_dwellings_df = self.calculate_initial_variables(
            representative_dwellings_df)
        return self

    # @logger.log_time_class
    def get_parameters_representative_dwellings(self) -> pd.DataFrame:
        dataf = self.dwellings_df

        # extract parameters of the representative virtual dwelling of each cluster
        frames = []

        for _, cl in enumerate(dataf["Cluster"].unique()):
            temp_df = dataf.loc[dataf["Cluster"] == cl].copy()
            nb_dwellings = len(temp_df)
            vd_agg_R = 1 / ((1 / temp_df["R_K/kW"]).sum())
            vd_agg_C = temp_df["C_kJ/K"].sum()
            vd_agg_current_heating_output = temp_df[
                "Initial_heating_outputk_kW"].sum()
            vd_agg_outdoor_temperature = temp_df[
                "Outdoor_temperature_ºC"].mean()
            vd_agg_design_temperature = temp_df["Design_temperature_ºC"].mean()
            vd_agg_indoor_temperature = dwelling_functions.calculate_max_temperature(
                vd_agg_outdoor_temperature, vd_agg_current_heating_output,
                vd_agg_R)

            vd_min_R = 1 / ((1 / temp_df["R_K/kW"].min()) * nb_dwellings)
            vd_max_R = 1 / ((1 / temp_df["R_K/kW"].max()) * nb_dwellings)
            vd_min_C = temp_df["C_kJ/K"].min() * nb_dwellings
            vd_max_C = temp_df["C_kJ/K"].max() * nb_dwellings
            vd_min_indoor_temperature = temp_df[
                "Initial_indoor_temperature_ºC"].min()
            vd_max_indoor_temperature = temp_df[
                "Initial_indoor_temperature_ºC"].max()
            vd_min_outdoor_temperature = temp_df["Outdoor_temperature_ºC"].min(
            )
            vd_max_outdoor_temperature = temp_df["Outdoor_temperature_ºC"].max(
            )
            vd_min_design_temperature = temp_df["Design_temperature_ºC"].min()
            vd_max_design_temperature = temp_df["Design_temperature_ºC"].max()
            cluster_number = cl
            frames.append([
                vd_agg_R,
                vd_min_R,
                vd_max_R,
                vd_agg_C,
                vd_min_C,
                vd_max_C,
                vd_agg_indoor_temperature,
                vd_min_indoor_temperature,
                vd_max_indoor_temperature,
                vd_agg_outdoor_temperature,
                vd_min_outdoor_temperature,
                vd_max_outdoor_temperature,
                vd_agg_design_temperature,
                vd_min_design_temperature,
                vd_max_design_temperature,
                nb_dwellings,
                cluster_number,
            ])

        columns = [
            "agg_R_K/kW",
            "min_R_K/kW",
            "max_R_K/kW",
            "agg_C_kJ/K",
            "min_C_kJ/K",
            "max_C_kJ/K",
            "agg_Initial_indoor_temperature_ºC",
            "min_Initial_indoor_temperature_ºC",
            "max_Initial_indoor_temperature_ºC",
            "agg_Outdoor_temperature_ºC",
            "min_Outdoor_temperature_ºC",
            "max_Outdoor_temperature_ºC",
            "agg_Design_temperature_ºC",
            "min_Design_temperature_ºC",
            "max_Design_temperature_ºC",
            "number_of_dwellings",
            "Cluster",
        ]

        parameters_virtual_dwellings_df = pd.DataFrame(frames, columns=columns)
        return parameters_virtual_dwellings_df

    def get_initial_heating_output(
        self,
        R: float,
        initial_indoor_air_temperature: float,
        outdoor_air_temperature: float,
    ) -> float:
        delta_T = initial_indoor_air_temperature - outdoor_air_temperature
        init_P_out = 1 / R * delta_T
        if init_P_out < 0:
            init_P_out = 0
        return init_P_out

    # @logger.log_time_class
    def add_variables(
            self, func_temperature: dwelling_functions.TemperatureFunction):
        dataf = self.dwellings_df
        dataf[
            "Initial_indoor_temperature_ºC"] = self.initial_indoor_air_temperature

        if isinstance(func_temperature, types.LambdaType):
            dataf["Initial_indoor_temperature_ºC"] = [
                func_temperature(t)
                for t in dataf["Initial_indoor_temperature_ºC"]
            ]
        else:
            #             print(f'The indoor temperature input param is not recognized, the default indoor temperature will be used.')
            dataf[
                "Initial_indoor_temperature_ºC"] = self.initial_indoor_air_temperature

        #         dataf['Org_Initial_indoor_temperature_ºC'] = dataf['Initial_indoor_temperature_ºC']
        dataf["Outdoor_temperature_ºC"] = self.outside_air_temperature

        dataf[
            "Final_indoor_temperature_ºC"] = self.limit_indoor_air_temperature
        dataf["Capacity_factor_%"] = self.capacity_factor
        dataf["Cluster"] = 0
        self.dwellings_df = self.calculate_initial_variables(dataf)
        return self

    def calculate_initial_variables(self, dataf: pd.DataFrame) -> pd.DataFrame:

        dataf["Flexibility_provided_kW"] = 0
        lambda_func = lambda row: self.get_initial_heating_output(
            row["R_K/kW"],
            row["Initial_indoor_temperature_ºC"],
            row["Outdoor_temperature_ºC"],
        )
        dataf["Initial_heating_outputk_kW"] = dataf.apply(lambda_func, axis=1)
        dataf["Max_heating_output_kW"] = (
            21 - dataf["Design_temperature_ºC"]) / dataf["R_K/kW"]
        filt = dataf["Initial_heating_outputk_kW"] > dataf[
            "Max_heating_output_kW"]
        dataf.loc[
            filt,
            "Initial_heating_outputk_kW"] = dataf["Max_heating_output_kW"]

        dataf.loc[:, "Flexibility_provided_kW"] = (
            dataf["Max_heating_output_kW"] * dataf["Capacity_factor_%"] -
            dataf["Initial_heating_outputk_kW"]
        ) / self.efficiency_heating_system

        return dataf

    # @logger.log_time_class
    def get_duration_service(self, method: Method):
        dataf = None
        if method is Method.REPRESENTATIVE:
            dataf = self.representative_dwellings_df.copy()
        elif method is Method.INDIVIDUAL:
            dataf = self.dwellings_df.copy()
        else:  # not recognised
            raise ValueError(f"The method {method} was not recognised.")

        dataf["Duration_s"] = dataf.apply(
            lambda row: dwelling_functions.calculate_duration_service(
                row["Initial_indoor_temperature_ºC"],
                row["Outdoor_temperature_ºC"],
                row["Final_indoor_temperature_ºC"],
                row["Max_heating_output_kW"] * row["Capacity_factor_%"],
                row["Max_heating_output_kW"],
                row["R_K/kW"],
                row["C_kJ/K"],
            ),
            axis=1,
        )

        dataf["Scenario"] = self.scenario.name
        dataf.dropna(subset=["Duration_s"], axis=0, inplace=True)

        dataf.loc[dataf["Duration_s"] == -1, "Duration_s"] = (
            self.infinite_duration_value
        )  ## the service can be provided for an infinite amount of time
        #         dataf['Initial_indoor_temperature_ºC'] = -1 # only for aggregation purposes using the summarise_results() function

        if self.demand_increase:  # negative flexibility is set to 0
            filt = dataf["Flexibility_provided_kW"] < 0
        else:
            filt = dataf["Flexibility_provided_kW"] > 0
        dataf.loc[filt, "Flexibility_provided_kW"] = 0

        # dataf = dataf.loc[dataf["Duration_s"] > 0, :]

        self.temp_df = dataf.copy()

        summary_df = self.summarise_results(dataf)

        if method is Method.REPRESENTATIVE:
            self.summary_representative_dwellings = summary_df
        elif method is Method.INDIVIDUAL:
            self.summary_individual_dwellings = summary_df

        return self

    # @logger.log_time_class
    def cluster_dwellings(self, nb_cluster: int):
        """Groups the dwellings into N clusters based on a set of variables."""

        dataf = self.dwellings_df
        clustering_variables = [
            "C_kJ/K",
            "R_K/kW",
            "Initial_indoor_temperature_ºC",
            "Outdoor_temperature_ºC",
            "Design_temperature_ºC",
        ]
        X = dataf[clustering_variables].values
        standardized_data = StandardScaler().fit_transform(X)
        kmeans = MiniBatchKMeans(
            n_clusters=nb_cluster, random_state=0, batch_size=1024 * 3).fit(
                standardized_data
            )  # batch size set to 1024*3 to avoid memory leak on windows
        dataf["Cluster"] = kmeans.labels_
        self.dwellings_df = dataf
        return self

    def summarise_results(self, dataf: pd.DataFrame):
        """Aggregate the results obtained from the different dwellings into a single column."""
        agg_df = dataf[["Duration_s", "Flexibility_provided_kW"]].copy()
        agg_df = agg_df.sort_values(["Duration_s"], ascending=False)
        agg_df["Flexibility_provided_kW"] = agg_df[
            "Flexibility_provided_kW"].cumsum()

        if self.demand_increase:
            agg_df = (agg_df.groupby("Duration_s").agg({
                "Flexibility_provided_kW":
                "max"
            }).reset_index())
        else:
            agg_df = (agg_df.groupby("Duration_s").agg({
                "Flexibility_provided_kW":
                "min"
            }).reset_index())

        agg_df.sort_values("Duration_s", inplace=True)

        # add extra records for plotting purposes
        copy_agg_df = agg_df.copy()
        copy_agg_df["Flexibility_provided_kW"] = copy_agg_df[
            "Flexibility_provided_kW"].shift(-1)
        copy_agg_df["Duration_s"] = copy_agg_df["Duration_s"] + 0.00001

        agg_df = pd.concat([agg_df, copy_agg_df], axis=0).fillna(0)

        if self.demand_increase:
            init_value = agg_df["Flexibility_provided_kW"].max()
        else:
            init_value = agg_df["Flexibility_provided_kW"].min()
        agg_df = agg_df.append(
            {
                "Duration_s": 0,
                "Flexibility_provided_kW": init_value
            },
            ignore_index=True)  # add origin point

        if self.demand_increase:
            agg_df.sort_values("Flexibility_provided_kW",
                               ascending=False,
                               inplace=True)
        else:
            agg_df.sort_values("Flexibility_provided_kW",
                               ascending=True,
                               inplace=True)

        agg_df = agg_df.loc[~agg_df["Duration_s"].duplicated(keep="first")]

        agg_df.sort_values("Duration_s", inplace=True)
        agg_df.set_index("Duration_s", inplace=True)
        return agg_df

    def concat_results(self, time_resolution: str = "10s"):
        """Concatenate the results from the different strategies into a single dataframe"""
        list_dfs = [
            self.summary_individual_dwellings,
            self.summary_representative_dwellings,
        ]
        cols_name = [
            "Flexibility_provided_(Individual_dwellings)_kW",
            "Flexibility_provided_(Representative_dwellings)_kW",
        ]

        new_frame = []
        for ii, df in enumerate(list_dfs):
            if len(df) > 0:
                temp_df = df.copy()
                temp_df.rename(
                    columns={"Flexibility_provided_kW": cols_name[ii]},
                    inplace=True)
                new_frame.append(temp_df)

        concat_df = pd.concat(new_frame, axis=1).fillna(method="ffill")
        concat_df = concat_df.groupby("Duration_s").sum()
        ## Resample the results so accuracy scores (e.g., RMSE, MAPE) can be calculated
        concat_df.reset_index(inplace=True)
        concat_df["Time_delta"] = pd.to_timedelta(concat_df["Duration_s"],
                                                  unit="s")
        concat_df.set_index("Time_delta", inplace=True)
        concat_df.sort_index(inplace=True)

        concat_df = concat_df.resample(time_resolution).mean()
        concat_df.fillna(method="bfill", inplace=True)
        concat_df["Time_delta"] = concat_df.index
        concat_df["Duration_s"] = concat_df.index.total_seconds()
        concat_df.set_index("Duration_s", inplace=True)
        self.concat_results_df = concat_df
        return self

    def calculate_scores(self):
        dataf = self.concat_results_df
        actual_col = "Flexibility_provided_(Individual_dwellings)_kW"
        filters = dataf[actual_col].abs() > 0
        cols_name = [
            "Flexibility_provided_(Individual_dwellings)_kW",
            "Flexibility_provided_(Representative_dwellings)_kW",
        ]

        actual_values = dataf.loc[filters, actual_col, ].values
        scores = pd.DataFrame({
            "RMSE_kW": pd.Series([], dtype="float"),
            "MAPE_%": pd.Series([], dtype="float"),
            "MAE_kW": pd.Series([], dtype="float"),
        })

        scores.index.name = "Predicted_column_name"
        if len(actual_values) > 0:
            for predicted_col in cols_name:
                if predicted_col in dataf.columns:
                    predicted_values = dataf.loc[filters, predicted_col].values
                    RMSE_val = scores_functions.RMSE(actual_values,
                                                     predicted_values)
                    if len(actual_values[actual_values == 0]) == 0:
                        MAPE_val = scores_functions.MAPE(
                            actual_values, predicted_values)
                    else:
                        MAPE_val = np.nan
                    MAE_val = scores_functions.MAE(actual_values,
                                                   predicted_values)
                #             print(predicted_col)
                #             print(RMSE_val, MAPE_val, MAE_val)
                else:
                    RMSE_val = np.nan
                    MAPE_val = np.nan
                    MAE_val = np.nan
                scores.loc[predicted_col, :] = [RMSE_val, MAPE_val, MAE_val]

        return scores

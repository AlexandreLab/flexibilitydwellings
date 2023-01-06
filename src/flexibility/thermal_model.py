from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from . import dwelling_functions


@dataclass
class ThermalModel:
    """Create a thermal model of a building."""
    R: float = 0  # K/kW
    C: float = 0  # kJ/K
    outdoor_air_temperature: float = 0
    initial_indoor_air_temperature: float = 0
    limit_indoor_air_temperature: float = 0
    design_temperature: float = 0
    target_indoor_air_temperature: float = 21
    initial_heating_output: float = 0
    max_heating_output: float = 0
    RC_model_df = pd.DataFrame()

    def init_parameters(self):
        self.initial_heating_output = self.get_initial_heating_output()
        self.max_heating_output = round(
            1 / self.R *
            (self.target_indoor_air_temperature - self.design_temperature),
            3,
        )
        return self

    def get_initial_heating_output(self) -> float:
        delta_T = self.initial_indoor_air_temperature - self.outdoor_air_temperature
        init_P_out = 1 / self.R * delta_T
        if init_P_out < 0:
            init_P_out = 0
        return round(init_P_out, 3)

    def get_duration_service(self, current_heating_output):

        return dwelling_functions.calculate_duration_service(
            self.initial_indoor_air_temperature,
            self.outdoor_air_temperature,
            self.limit_indoor_air_temperature,
            current_heating_output,
            self.max_heating_output,
            self.R,
            self.C,
        )

    def get_max_temperature(self, current_heating_output):
        return dwelling_functions.calculate_max_temperature(
            self.outdoor_air_temperature, current_heating_output, self.R)

    def get_heating_output(self, duration):
        #     print(f'Dwelling with R={R}, C={C}, T_in={T_in}, T_out={T_out}, duration={duration} sec')
        B = 1 - 1 / (self.R * self.C)
        P_out = ((self.limit_indoor_air_temperature -
                  self.initial_indoor_air_temperature * B**duration -
                  self.outdoor_air_temperature / (self.R * self.C) *
                  (1 - B**duration) / (1 - B)) * self.C * (1 - B) /
                 (1 - B**duration))
        #     if P_out<0:
        #         P_out = 0
        return P_out

    def RC_model_data(self,
                      heating_output: float = 0,
                      length_index: int = 30 * 60 * 60) -> pd.DataFrame:
        # Initialisation of the data

        dataf = pd.DataFrame(
            columns=[
                "Indoor_temperature_degreeC",
                "Outdoor_temperature_degreeC",
                "Heating_output_kW",
            ],
            index=np.arange(length_index),
        )
        dataf.fillna(0.0, inplace=True)
        dataf.loc[
            0,
            "Indoor_temperature_degreeC"] = self.initial_indoor_air_temperature
        dataf.loc[:,
                  "Outdoor_temperature_degreeC"] = self.outdoor_air_temperature
        dataf.loc[:, "Heating_output_kW"] = heating_output
        dataf = dataf.astype('float32')
        self.RC_model_df = dataf
        return dataf

    def run_RC_model(self, dataf):
        time_index = dataf.index.values
        T_in = dataf["Indoor_temperature_degreeC"].values
        T_out = dataf["Outdoor_temperature_degreeC"].values
        P_out = dataf["Heating_output_kW"].values

        for t in np.arange(1, len(time_index)):
            dt = time_index[t] - time_index[t - 1]
            T_in[t] = T_in[t - 1] + dt / self.C * (
                (T_out[t - 1] - T_in[t - 1]) / self.R + P_out[t - 1])
        dataf["Indoor_temperature_degreeC"] = T_in
        self.RC_model_df = dataf
        return dataf

    def estimate_heating_demand(
            self,
            dataf,
            min_indoor_air_temperature: float = 19,
            max_indoor_air_temperature: float = 24) -> pd.DataFrame:
        """Estimate the heating/cooling output required to maintain indoor air temperature between limits."""
        time_index = dataf.index.values
        indoor_air_temperatures = dataf["Indoor_temperature_degreeC"].values
        outdoor_air_temperatures = dataf["Outdoor_temperature_degreeC"].values
        heating_system_outputs = dataf["Heating_output_kW"].values

        for t in np.arange(1, len(time_index)):
            dt = time_index[t] - time_index[t - 1]
            #calculate the expected indoor air temperature if no cooling is used
            temp_indoor_air_temperature = indoor_air_temperatures[
                t -
                1] + dt / self.C * (outdoor_air_temperatures[t - 1] -
                                    indoor_air_temperatures[t - 1]) / self.R
            if temp_indoor_air_temperature > max_indoor_air_temperature:
                #Cooling required
                heating_system_outputs[
                    t - 1] = -(outdoor_air_temperatures[t - 1] -
                               indoor_air_temperatures[t - 1]) / self.R
            elif temp_indoor_air_temperature < min_indoor_air_temperature:
                #heating required
                heating_system_outputs[
                    t - 1] = -(outdoor_air_temperatures[t - 1] -
                               indoor_air_temperatures[t - 1]) / self.R
            else:
                heating_system_outputs[t - 1] = 0
            indoor_air_temperatures[t] = indoor_air_temperatures[
                t - 1] + dt / self.C * (
                    (outdoor_air_temperatures[t - 1] -
                     indoor_air_temperatures[t - 1]) / self.R +
                    heating_system_outputs[t - 1])

        dataf["Indoor_temperature_degreeC"] = indoor_air_temperatures
        dataf["Heating_output_kW"] = heating_system_outputs
        self.RC_model_df = dataf
        return dataf

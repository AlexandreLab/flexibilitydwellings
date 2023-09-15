from __future__ import annotations

import os
from calendar import monthrange
from dataclasses import dataclass, field
from typing import List, Tuple

import pandas as pd


def get_new_names(list_cols: List[str],
                  list_categories: List[str]) -> List[str]:
    """Return the name of the dwelling categorie of each columns"""
    new_names = []
    for col in list_cols:
        for cat in list_categories:
            if " " + cat in col:
                new_names.append(cat)
    return new_names


def get_tidy_df(dataf, cats, list_keywords):
    """Return a tidy dataframe of the columns attached based on the list of dwelling categories"""
    frame = []

    for keywords in list_keywords:
        temp_cols: List[str] = dataf.columns
        for k in keywords:
            temp_cols = [x for x in temp_cols if k in x]
        new_cols_names = get_new_names(temp_cols, cats)
        temp_df = dataf[temp_cols].copy()
        temp_df.columns = new_cols_names
        temp_df = temp_df.stack().to_frame().reset_index()
        temp_df.columns = ["LSOA_index", "Dwelling category", "Value"]
        temp_df.set_index(["LSOA_index", "Dwelling category"],
                          inplace=True,
                          drop=True)
        frame.append(temp_df)

    return pd.concat(frame, axis=1)


def dwelling_form_lookup_table(dataf: pd.DataFrame):
    list_dwelling_forms = ["detached", "semi-detached", "terraced", "flat"]
    lookup_dict = {}
    for k in dataf["Dwelling category"].unique():
        for v in list_dwelling_forms:
            if len(k.split(v)[0]) == 0:
                lookup_dict[k] = v
    return lookup_dict


def heating_system_lookup_table(dataf: pd.DataFrame):
    list_heating_systems = [
        "gas boiler",
        "oil boiler",
        "resistance heating",
        "biomass boiler",
    ]
    lookup_dict = {}
    for k in dataf["Dwelling category"].unique():
        for v in list_heating_systems:
            if len(k.split(v)) > 1:
                lookup_dict[k] = v
    return lookup_dict


# def design_temperature_lookup_table(dataf: pd.DataFrame):
#     """Return a lookup table that links LSOA index to design temperature"""
#     lookup_dict = {}
#     for k in dataf["LSOA11CD"].index:
#         lookup_dict[k] = dataf.loc[k, "Design_temperature_degreeC"]
#     return lookup_dict

# def region_lookup_table(dataf: pd.DataFrame):
#     """Return a lookup table that links LSOA index to design temperature"""
#     lookup_dict = {}
#     for k in dataf["LSOA11CD"].index:
#         lookup_dict[k] = dataf.loc[k, "Region"]
#     return lookup_dict


def get_lookup_table(dataf: pd.DataFrame, target_col: str):
    """Return a lookup table that links LSOA index to the target col values"""
    lookup_dict = {}
    for k in dataf["LSOA11CD"].index:
        lookup_dict[k] = dataf.loc[k, target_col]
    return lookup_dict


def add_columns_tidy_df(org_dataf: pd.DataFrame,
                        dataf: pd.DataFrame) -> pd.DataFrame:
    dataf = dataf.copy()
    dataf = dataf.loc[dataf["Number of dwellings"] > 0]
    dataf["Total annual heat demand kWh"] = (
        dataf["Average annual heat demand kWh"] * dataf["Number of dwellings"])
    lookup_dict = dwelling_form_lookup_table(dataf)
    dataf["Dwelling forms"] = dataf["Dwelling category"].map(lookup_dict)
    print('Dwelling form')
    lookup_dict = heating_system_lookup_table(dataf)
    dataf["Heating systems"] = dataf["Dwelling category"].map(lookup_dict)
    print('Heating system')
    lookup_dict = get_lookup_table(org_dataf, "Design_temperature_degreeC")
    dataf["Outdoor air design temperature degreeC"] = dataf["LSOA_index"].map(
        lookup_dict)
    print('Outdoor air design temperature degreeC')
    lookup_dict = get_lookup_table(org_dataf, "Region")
    dataf["Region"] = dataf["LSOA_index"].map(lookup_dict)
    print('Region')
    lookup_dict = get_lookup_table(org_dataf, "Local Authority")
    dataf["Local Authority"] = dataf["LSOA_index"].map(lookup_dict)
    print('Local authority')
    lookup_dict = get_lookup_table(org_dataf, "LSOA11CD")
    dataf["LSOA_code"] = dataf["LSOA_index"].map(lookup_dict)
    print('LSOA_code')
    dataf["Average size of heating system kW"] = dataf[
        "Average thermal losses kW/K"] * (
            21 - dataf["Outdoor air design temperature degreeC"])
    print("Average size of heating system kW")
    dataf["Total capacity installed of heating systems GW"] = (
        dataf["Average size of heating system kW"] *
        dataf["Number of dwellings"] / 1_000_000)
    print("Total capacity installed of heating systems GW")
    return dataf


def get_concat_tidy_df(
    dataf: pd.DataFrame,
    categories: List[str],
    thermal_capacity_levels: List[str],
    before_after: str = "before",
) -> pd.DataFrame:
    frame = []
    new_cols_name = [
        "Average thermal capacity kJ/K", "Average thermal losses kW/K",
        "Number of dwellings", "Average annual heat demand kWh",
        "Average floor area m2"
    ]

    for level in thermal_capacity_levels:
        list_keys = [
            [f"Average {level} thermal capacity"], ["Average thermal losses"],
            ["Number", "2018"],
            [
                f"Average heat demand {before_after} energy efficiency measures for"
            ], ["Average floor area of"]
        ]

        temp_df = get_tidy_df(dataf, categories, list_keys)
        temp_df.columns = new_cols_name
        temp_df["Thermal capacity level"] = level
        frame.append(temp_df.reset_index())

    return pd.concat(frame, axis=0)


def get_nb_degree_days(temperatures_month: List[float],
                       year: int = 2010) -> float:
    # from sap 2012
    base_temperature = 15.5
    nb_days_month = [monthrange(year, x)[1] for x in range(1, 13)]
    nb_degree_days = 0
    for ii, temp in enumerate(temperatures_month):
        if temp < base_temperature:
            nb_degree_days = (nb_degree_days +
                              (base_temperature - temp) * nb_days_month[ii])
    return nb_degree_days


def get_LA_to_region_dict() -> dict:
    path_OA_lookup = r"D:\OneDrive - Cardiff University\04 - Projects\03 - PhD\03 - Analysis\03 - LSOAs\00 - Data\Lookup tables"
    OA_lookup_file = r"Postcode to OA LSOA MSOA LAD\PCD11_OA11_LSOA11_MSOA11_LAD11_EW_LU_feb_2018.csv"
    OA_lookup_df = pd.read_csv(path_OA_lookup + os.path.sep + OA_lookup_file,
                               low_memory=False)
    OA_lookup_df.drop(
        [
            "pcd8",
            "pcds",
            "dointr",
            "doterm",
            "usertype",
            "lsoa11nm",
            "msoa11nm",
            "ladnmw",
            "FID",
        ],
        axis=1,
        inplace=True,
    )
    OA_lookup_df.drop_duplicates(inplace=True)

    region_lookup_file = r"laregionlookup376las.xls"
    region_lookup_df = pd.read_excel(
        path_OA_lookup + os.path.sep + region_lookup_file,
        sheet_name=1,
        header=6,
        usecols=[0, 1, 3],
    )

    OA_lookup_df = pd.merge(
        OA_lookup_df,
        region_lookup_df,
        left_on="ladcd",
        right_on="la_code",
        how="left",
    )

    OA_lookup_df = pd.merge(
        OA_lookup_df,
        region_lookup_df,
        left_on="ladnm",
        right_on="la_name",
        how="left",
    )
    OA_lookup_df["la_code_x"].fillna(OA_lookup_df["la_code_y"], inplace=True)
    OA_lookup_df["la_name_x"].fillna(OA_lookup_df["la_name_y"], inplace=True)
    OA_lookup_df["region_name_x"].fillna(OA_lookup_df["region_name_y"],
                                         inplace=True)
    OA_lookup_df.drop(
        ["ladcd", "ladnm", "la_code_y", "la_name_y", "region_name_y"],
        axis=1,
        inplace=True,
    )
    OA_lookup_df.dropna(subset=["la_name_x"], inplace=True)
    OA_lookup_df["pcd7"] = OA_lookup_df["pcd7"].str.replace(" ", "")
    OA_lookup_df.columns = [
        "PCD7",
        "OA11CD",
        "LSOA11CD",
        "MSOA11CD",
        "LAD11CD",
        "Local Authority",
        "Region",
    ]
    LA_Region_df = OA_lookup_df[["Local Authority", "Region"]].copy()
    LA_Region_df = LA_Region_df.drop_duplicates().reset_index(drop=True)
    dict_map = dict(
        zip(LA_Region_df["Local Authority"], LA_Region_df["Region"]))
    return dict_map


def get_degreedays_by_region(ukerc_path_data: str) -> Tuple[dict, dict]:
    temperature_data = pd.read_csv(ukerc_path_data + os.path.sep +
                                   "mean_outside_air_temperature_SAP2012.csv")
    month_cols = [
        "Jan",
        "Feb",
        "Mar",
        "Apr",
        "May",
        "Jun",
        "Jul",
        "Aug",
        "Sep",
        "Oct",
        "Nov",
        "Dec",
    ]
    temperature_data["Degree_days"] = temperature_data[month_cols].apply(
        get_nb_degree_days, axis=1)
    temperature_data.fillna("", inplace=True)
    degree_day_dict = {}
    design_temperature_dict = {}
    for _, row in temperature_data.iterrows():
        lookup_value = row["Lookup_values"]
        degree_day = row["Degree_days"]
        design_temp = row["Design_temperature"]
        if str(lookup_value) != "":
            #         print(lookup_value)
            for val in lookup_value.split("/"):
                print(val)
                degree_day_dict[val] = degree_day
                design_temperature_dict[val] = design_temp
    return design_temperature_dict, degree_day_dict


@dataclass
class ThermalCharacteristics:
    path_data: str = ""
    filename: str = ""
    scenario: str = "before"
    if filename == "":
        filename = "LSOAs_in_England_Wales_before_EE_heat_demand.csv"

    lsoa_data: pd.DataFrame = pd.DataFrame()

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

    def remove_replace_outliers(self) -> ThermalCharacteristics:
        # remove values above the 99th quantile with the value of the 99th quantile
        # remove values below the 1th quantile with the value of the 1th quantile
        dataf = self.lsoa_data
        for c in self.categories:
            heat_demand_col = (
                f"Average heat demand {self.scenario} energy efficiency measures for {c} (kWh)")
            floor_col = f"Average floor area of {c} (m2)"

            max_heat_value = dataf[heat_demand_col].quantile(0.99)
            min_heat_value = dataf[heat_demand_col].quantile(0.01)
            max_floor_value = dataf[floor_col].quantile(0.99)
            min_floor_value = dataf[floor_col].quantile(0.01)

            dataf.loc[dataf[heat_demand_col] > max_heat_value,
                      heat_demand_col] = max_heat_value
            dataf.loc[dataf[heat_demand_col] > max_heat_value,
                      floor_col] = max_floor_value

            dataf.loc[dataf[heat_demand_col] < min_heat_value,
                      heat_demand_col] = min_heat_value
            dataf.loc[dataf[heat_demand_col] < min_heat_value,
                      floor_col] = min_floor_value

            dataf.loc[dataf[floor_col] > max_floor_value,
                      heat_demand_col] = max_heat_value
            dataf.loc[dataf[floor_col] > max_floor_value,
                      floor_col] = max_floor_value

            dataf.loc[dataf[floor_col] < min_floor_value,
                      heat_demand_col] = min_heat_value
            dataf.loc[dataf[floor_col] < min_floor_value,
                      floor_col] = min_floor_value

        return self

    def load_data(self) -> ThermalCharacteristics:
        self.lsoa_data = pd.read_csv(self.path_data + os.path.sep +
                                     self.filename,
                                     low_memory=False,
                                     index_col=0)
        return self

    def map_LA_to_region(self) -> ThermalCharacteristics:
        dict_map = get_LA_to_region_dict()
        self.lsoa_data["Region"] = self.lsoa_data["Local Authority"].map(
            dict_map)
        return self

    def add_DD_and_design_temp(self,
                               ukerc_path_data: str) -> ThermalCharacteristics:
        # add design temeprature and degree days data
        design_temp, dd_dict = get_degreedays_by_region(ukerc_path_data)
        self.lsoa_data["Degree_days"] = self.lsoa_data["Region"].map(dd_dict)
        self.lsoa_data["Design_temperature_degreeC"] = pd.to_numeric(self.lsoa_data[
            "Region"].map(design_temp))
        
        return self

    def calculate_thermal_losses(self) -> ThermalCharacteristics:

        for c in self.categories:
            heat_demand_col = (
                f"Average heat demand {self.scenario} energy efficiency measures for {c} (kWh)")
            temp_col_name = f"Average thermal losses {c} kW/K"
            self.lsoa_data[temp_col_name] = self.lsoa_data[heat_demand_col] / (
                24 * self.lsoa_data["Degree_days"])
        return self

    def calculate_thermal_capacity(self) -> ThermalCharacteristics:
        dataf = self.lsoa_data
        for c in self.categories:
            floor_col = f"Average floor area of {c} (m2)"
            dataf[floor_col].fillna(dataf[floor_col].mean(), inplace=True)
            print(f"0.9 Quantile of {c} is {dataf[floor_col].quantile(0.99)}")
            print(f"Max value of {c} is {dataf[floor_col].max()}")
            temp_col_name = f"Average low thermal capacity {c} kJ/K"
            dataf[temp_col_name] = dataf[
                floor_col] * 100  # kJ/m2/K (p196 of SAP 2012)
            temp_col_name = "Average medium thermal capacity " + c + " kJ/K"
            dataf[temp_col_name] = dataf[
                floor_col] * 250  # kJ/m2/K (p196 of SAP 2012)
            temp_col_name = "Average high thermal capacity " + c + " kJ/K"
            dataf[temp_col_name] = dataf[
                floor_col] * 450  # kJ/m2/K (p196 of SAP 2012)

            temp_col_name = "Average medium+10% thermal capacity " + c + " kJ/K"
            dataf[temp_col_name] = dataf[floor_col] * 250 * 1.1  # kJ/m2/K

            temp_col_name = "Average medium-10% thermal capacity " + c + " kJ/K"
            dataf[temp_col_name] = dataf[floor_col] * 250 * 0.9  # kJ/m2/K
        # self.lsoa_data = dataf
        return self

    def sum_thermal_parameters(self) -> ThermalCharacteristics:
        # total thermal capacity
        dataf = self.lsoa_data
        thermal_capacity_list = [
            "low", "medium", "high", "medium+10%", "medium-10%"
        ]
        for thermal_capacity_level in thermal_capacity_list:
            cols_to_sum = []
            for c in self.categories:
                number_col = f"Number of {c} in 2018"
                temp_thermal_cap_col = (
                    f"Average {thermal_capacity_level} thermal capacity {c} kJ/K"
                )
                temp_total_col = (
                    f"Total {thermal_capacity_level} thermal capacity {c} kJ/K"
                )
                cols_to_sum.append(temp_total_col)
                dataf[temp_total_col] = dataf[number_col] * dataf[
                    temp_thermal_cap_col]

            dataf[f"Total {thermal_capacity_level} thermal capacity GJ/K"] = (
                dataf[cols_to_sum].sum(axis=1) / 1000000)

        # total thermal losses
        cols_to_sum = []
        for c in self.categories:
            number_col = f"Number of {c} in 2018"
            temp_thermal_losses_col = f"Average thermal losses {c} kW/K"
            temp_total_col = f"Total thermal losses {c} kW/K"
            dataf[temp_total_col] = dataf[number_col] * dataf[
                temp_thermal_losses_col]
            cols_to_sum.append(temp_total_col)
        dataf["Total thermal losses kW/K"] = dataf[cols_to_sum].sum(axis=1)

        return self

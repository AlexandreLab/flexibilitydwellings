import os

import numpy as np
import pandas as pd
from matplotlib.style import use

import flexibility.flexibility_model as fm
from flexibility.enums import Method, Scenario
from utils import utils
from utils.folder_system import FolderSystem


def accuracy_calculation():
    data_path = r"D:\OneDrive - Cardiff University\04 - Projects\03 - PhD\03 - Analysis\03 - LSOAs\00 - Data\Domestic_EPC_results\EPC_thermal_capacity\\"  # computer_path
    file = "Thermal_characteristics_beforeEE_LSOA_EandW.csv"

    saving_path = os.getcwd()
    saving_path = os.path.join(saving_path, "data")
    saving_path = os.path.join(saving_path, "results_accuracy")
    lsoa_data = utils.load_data(data_path + file)

    OAT = 0
    capacity_factor = 0
    nb_lsoas = 4
    features = [
        "N_Households 2018",
        "Total thermal losses kW/K",
        "Total medium thermal capacity GJ/K",
    ]
    list_lsoas = utils.get_cLHS_samples(lsoa_data, features, nb_lsoas)

    list_nb_dwellings_per_cluster = np.divide(
        100, np.concatenate([[1], np.arange(2, 22, 2)]))

    accuracy_frames = []
    for nb_dwellings_per_cluster in list_nb_dwellings_per_cluster:

        flexiblity = fm.FlexibilityModel(
            lsoa_data,
            saving_path=saving_path,
            outside_air_temperature=OAT,
            capacity_factor=capacity_factor,
            nb_dwellings_per_cluster=nb_dwellings_per_cluster,
        )
        temp_accuracy_df = flexiblity.calculate_accuracy_methods(list_lsoas)
        accuracy_frames.append(temp_accuracy_df)
    final_scores_df = pd.concat(accuracy_frames)
    final_scores_df.sort_values("Nb_clusters", inplace=True)
    filename = f"final_scores_{nb_lsoas}_cf_{capacity_factor}_lsoas.csv"
    final_scores_df.to_csv(os.path.join(saving_path, filename))


def flexibility_calculation() -> None:
    data_path = r"D:\OneDrive - Cardiff University\04 - Projects\03 - PhD\03 - Analysis\03 - LSOAs\00 - Data\Domestic_EPC_results\EPC_thermal_capacity\\"  # computer_path

    after_EE = False
    if after_EE:
        file = "Thermal_characteristics_afterEE_LSOA_EandW.csv"
    else:
        file = "Thermal_characteristics_beforeEE_LSOA_EandW.csv"
    lsoa_data = utils.load_data(data_path + file)

    methods = [Method.INDIVIDUAL]
    use_pdf = True

    for mean_indoor_temperature in [19]:
        for thermal_capacity_level in ["medium"
                                       ]:  #, "medium+10%", "medium-10%"]:
            for scenario in [Scenario.HP100]:
                for outdoor_air_temperature in [-5.0, 0.0, 5.0, 10.0]:
                    folder_system = FolderSystem(
                        outside_air_temperature=outdoor_air_temperature,
                        use_pdf=use_pdf,
                        inside_air_temperature=mean_indoor_temperature,
                        after_energy_efficiency=after_EE,
                        thermal_capacity_level=thermal_capacity_level,
                    )
                    folder_system.set_path(True)
                    for capacity_factor in [0, 1]:

                        flexiblity = fm.FlexibilityModel(
                            lsoa_data,
                            folder_system=folder_system,
                            outside_air_temperature=outdoor_air_temperature,
                            capacity_factor=capacity_factor,
                            nb_slices=5,
                            scenario=scenario,
                            thermal_capacity_level=thermal_capacity_level,
                        )
                        flexiblity.set_temperature_func(
                            use_pdf, mean_indoor_temperature
                        )  # use mean inside air temperature for all dwellings
                        flexiblity.run_model_multiple_lsoas(methods,
                                                            nb_lsoas=200)


def accuracy_loop():
    # load LSOA data
    data_path = r"D:\OneDrive - Cardiff University\04 - Projects\03 - PhD\03 - Analysis\03 - LSOAs\00 - Data\Domestic_EPC_results\EPC_thermal_capacity\\"  # computer_path
    file = "Thermal_characteristics_LSOA_EandW.csv"

    saving_path = os.getcwd()
    saving_path = os.path.join(saving_path, "data")
    saving_path = os.path.join(saving_path, "results_accuracy")
    # saving_path = os.path.join(saving_path, "OAT_0")
    lsoa_data = utils.load_data(data_path + file)

    # run the model to calculate the accuracy of the different methods
    # instead of choosing random LSOAs, a latin hypercube method is used to sample

    nb_lsoa = 300
    features = [
        "N_Households 2018",
        "Total thermal losses kW/K",
        "Total medium thermal capacity GJ/K",
    ]
    list_lsoas = utils.get_cLHS_samples(lsoa_data, features, nb_lsoa)
    list_ratios = np.concatenate([[1], np.arange(2, 22, 2)
                                  ])  # number of clusters per 100 dwellings

    for capacity_factor in [0, 1]:
        filename = f"final_scores_{nb_lsoa}_cf_{capacity_factor}_lsoas.csv"
        final_scores_df = utils.calculate_accuracy_methods(
            lsoa_data, list_lsoas, list_ratios, capacity_factor)
        final_scores_df.to_csv(os.path.join(saving_path, filename))

    print(
        f"The results of the model have been stored in the folder: {saving_path}"
    )


if __name__ == "__main__":
    flexibility_calculation()

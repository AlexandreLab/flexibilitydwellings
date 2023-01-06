import os
from typing import List, Optional

import geopandas as gpd
import matplotlib.pyplot as plt
import matplotlib.ticker as tkr
import numpy as np
import pandas as pd
import seaborn as sns

from ..flexibility import enums
from ..utils import folder_system as fs

sns.set_palette(sns.color_palette("colorblind", 15))
plt.rcParams["font.family"] = "Times New Roman"
font_size = 12
params = {
    "axes.labelsize": font_size + 2,
    "axes.titlesize": font_size + 4,
    "legend.fontsize": font_size,
    "xtick.labelsize": font_size,
    "ytick.labelsize": font_size,
    "font.size": font_size,
}
plt.rcParams.update(params)
sns.set_style("whitegrid")
figsize = (12, 3.5)
figsize_map = (10, 6)

markers = ["P", "o", "^", "s", "*", "v"]
linewidth = 0.1
edgecolor = "black"


def plot_bar_chart(dataf: pd.DataFrame,
                   correction_factor: int = 1,
                   remove_legend: bool = True):
    f, ax = plt.subplots(figsize=figsize)
    (dataf / correction_factor).plot.bar(ax=ax, rot=0)
    if remove_legend:
        ax.legend().remove()
    return f, ax


def plot_thermal_characteristics_lsoa_level(dataf: pd.DataFrame,
                                            list_parameters: List[str]):
    f, ax = plt.subplots(figsize=figsize)
    for ii, p in enumerate(list_parameters):
        x_arr = dataf[f"Total {p} thermal capacity GJ/K"].values
        y_arr = dataf["Total thermal losses kW/K"].values
        ax.scatter(
            x=x_arr,
            y=y_arr,
            marker=markers[ii],
            linewidths=linewidth,
            edgecolors=edgecolor,
            label=p,
        )

    ax.get_yaxis().set_major_formatter(
        tkr.FuncFormatter(lambda x, p: "{:,.0f}".format(x)))
    ax.set_xlabel("Thermal capacity (GJ/K)")
    ax.set_ylabel("Thermal losses (kW/K)")
    ax.legend(title="Thermal capacity level")
    return f, ax


def plot_bar_chart(
    dataf: pd.DataFrame,
    correction_factor: int = 1,
    remove_legend: bool = True,
    stacked: bool = False,
):
    f, ax = plt.subplots(figsize=(figsize[0] / 2, figsize[1]))
    (dataf / correction_factor).plot.bar(ax=ax, rot=0, stacked=stacked)
    if remove_legend:
        ax.legend().remove()
    return f, ax


def plot_thermal_characteristics_dwelling_category(dataf: pd.DataFrame,
                                                   list_parameters: List[str],
                                                   dwelling_category: str):
    f, ax = plt.subplots(figsize=figsize)
    for ii, p in enumerate(list_parameters):
        filt = dataf["Thermal capacity level"] == p
        x_arr = dataf.loc[filt, f"Average thermal capacity kJ/K"].values
        y_arr = dataf.loc[filt, "Average thermal losses kW/K"].values
        ax.scatter(
            x=x_arr,
            y=y_arr,
            marker=markers[ii],
            linewidths=linewidth,
            edgecolors=edgecolor,
            label=p,
        )

    ax.set_title(f"Dwelling category: {dwelling_category}")
    ax.get_yaxis().set_major_formatter(
        tkr.FuncFormatter(lambda x, p: "{:,.2f}".format(x)))
    ax.get_xaxis().set_major_formatter(
        tkr.FuncFormatter(lambda x, p: "{:,.0f}".format(x / 1000)))
    ax.set_ylabel("Average thermal losses (kW/K)")
    ax.set_xlabel("Average thermal capacity (MJ/K)")
    ax.set_xlim(0, None)
    ax.set_ylim(0, None)
    ax.legend(title="Thermal capacity level")
    return f, ax


def plot_flexibility_results(
    dataf: pd.DataFrame,
    max_index: int = 0,
    time_resolution: enums.TimeResolution = enums.TimeResolution.SECONDS,
    power_resolution: enums.PowerResolution = enums.PowerResolution.MEGAWATT,
    smoothing: bool = False,
):

    time_correction_factor = time_resolution.factor
    power_scaling_factor = power_resolution.factor

    cols_name = [
        "Flexibility_provided_(Individual_dwellings)_kW",
        "Flexibility_provided_(Representative_dwellings)_kW",
    ]

    f, ax = plt.subplots(figsize=figsize)
    x_max_value = 0
    y_max_abs_value = 0
    for ii, col in enumerate(cols_name):

        if (len(dataf) > 0) & (col in dataf.columns):
            if smoothing:
                x_arr = dataf.index.rolling(window=4, min_periods=1).mean()
                y_arr = dataf.loc[:, col].rolling(window=4,
                                                  min_periods=1).mean()
            else:
                x_arr = dataf.index
                y_arr = dataf.loc[:, col]

            x_arr = x_arr / time_correction_factor
            y_arr = y_arr / power_scaling_factor
            ax.plot(x_arr, y_arr, color=sns.color_palette()[ii], label=col)
            ax.fill_between(
                x_arr,
                np.zeros(len(y_arr)),
                y_arr,
                color=sns.color_palette()[ii],
                alpha=0.2,
            )

            if np.max(x_arr) > x_max_value:
                x_max_value = np.max(x_arr)

            if np.max(np.abs(y_arr)) > y_max_abs_value:
                y_max_abs_value = np.max(np.abs(y_arr))

    if max_index == 0:
        ax.set_xlim(0, x_max_value / time_correction_factor)
    else:
        ax.set_xlim(0, max_index / time_correction_factor)

    if y_max_abs_value / power_scaling_factor > 100:
        func_formatter = lambda x, p: "{:,.0f}".format(x)
    else:
        func_formatter = lambda x, p: "{:,.2f}".format(x)
    ax.get_yaxis().set_major_formatter(tkr.FuncFormatter(func_formatter))

    ax.get_xaxis().set_major_formatter(
        tkr.FuncFormatter(lambda x, p: "{:,.0f}".format(x)))
    ax.legend()
    ax.set_ylabel(f"Flexibility provided ({power_resolution.label})")
    ax.set_xlabel(f"Duration ({time_resolution.label})")
    return f, ax


def rename(newname: str):

    def decorator(f):
        f.__name__ = newname
        return f

    return decorator


def q_at(y: float):

    @rename(f"quantile_{y:0.2f}")
    def q(x):
        return x.quantile(y)

    return q


def plot_accuracy_plots(dataf: pd.DataFrame):
    frames = []
    for sc_name in ["RMSE_kW", "MAPE_%", "MAE_kW"]:

        f, ax = plt.subplots(figsize=figsize)
        ax2 = ax.twinx()
        funcs = {sc_name: ["median", q_at(0.1), q_at(0.9)]}
        filt = (dataf["Predicted_column_name"] ==
                "Flexibility_provided_(Representative_dwellings)_kW")
        temp_df = dataf.loc[filt, :].groupby("Ratio").agg(funcs)
        temp_df = temp_df.droplevel(0, axis=1)
        x_arr = temp_df.index
        ax.plot(x_arr, temp_df["median"], label="Accuracy score")
        ax.fill_between(x_arr,
                        temp_df["quantile_0.10"],
                        temp_df["quantile_0.90"],
                        alpha=0.2)

        # plot percentage change
        y2_arr = np.abs(((temp_df["median"].shift(1) - temp_df["median"]) /
                         temp_df["median"]).fillna(1).values)

        # ax2.plot(x_arr, y2_arr, linestyle="--", color="black", label="Change")

        print(sc_name, temp_df["median"])
        if "%" in sc_name:
            ax.get_yaxis().set_major_formatter(
                tkr.FuncFormatter(lambda x, p: "{:,.0%}".format(x)))
            ax.set_ylim(0, 1)
        else:
            ax.get_yaxis().set_major_formatter(
                tkr.FuncFormatter(lambda x, p: "{:,.0f}".format(x)))
            ax.set_ylim(0, None)

        ax2.get_yaxis().set_major_formatter(
            tkr.FuncFormatter(lambda x, p: "{:,.0%}".format(x)))
        ax2.set_ylim(0, 1)
        ax2.set_ylabel("Absolute % change")
        ax2.grid(False)

        ax.set_ylabel(sc_name.replace("_", " "))
        ax.set_title(sc_name)
        ax.set_xlabel("Number of clusters per 100 dwellings")
        ax.margins(0, None)
        frames.append([f, ax])

        handles, labels = ax.get_legend_handles_labels()
        handles2, labels2 = ax2.get_legend_handles_labels()

        ax.legend(handles=handles + handles2, labels=labels + labels2)
    return frames


def plot_duration_individual_dwellings(
    dataf: pd.DataFrame,
    xlim: int = 0,
    time_resolution: enums.TimeResolution = enums.TimeResolution.SECONDS,
    power_resolution: enums.PowerResolution = enums.PowerResolution.MEGAWATT,
):
    """Plot the results from individual dwellings"""

    time_correction_factor = time_resolution.factor
    power_scaling_factor = power_resolution.factor

    dataf.sort_values(["Index", "Duration_s"], inplace=True)

    f, ax = plt.subplots(figsize=figsize)

    #     sns.lineplot(data=temp_df, x='Duration_s', y='Flexibility_provided_kW', hue='Amount_of_total_power_%')

    for ii, index_dwelling in enumerate(dataf.index.unique()):
        x_val = dataf.loc[index_dwelling,
                          "Duration_s"] / time_correction_factor
        y_val = (dataf.loc[index_dwelling, "Flexibility_provided_kW"] /
                 power_scaling_factor)
        C_val = dataf.loc[index_dwelling, "C_kJ/K"].mean()
        R_val = dataf.loc[index_dwelling, "R_K/kW"].mean()
        T_in_val = dataf.loc[index_dwelling,
                             "Initial_indoor_temperature_ºC"].mean()

        x_arr = [0, x_val, x_val + 0.1]
        y_arr = [y_val, y_val, 0]
        ax.plot(
            x_arr,
            y_arr,
            label=f"idx({ii}) with R={R_val}, C={C_val}, T_in={T_in_val}ºC",
        )
        ax.fill_between(x_arr, np.zeros(len(y_arr)), y_arr, alpha=0.2)

    if xlim == 0:
        ax.set_xlim(0, dataf["Duration_s"].max() / time_correction_factor)
    else:
        ax.set_xlim(0, xlim / time_correction_factor)

    #     ax.set_title(f'Initial indoor T of {init_T_in}ºC, indoor T limits [{lower_T_in}, {upper_T_in}]ºC, outdoor T {T_out}ºC')

    ax.get_xaxis().set_major_formatter(
        tkr.FuncFormatter(lambda x, p: "{:,.0f}".format(x)))
    ax.get_yaxis().set_major_formatter(
        tkr.FuncFormatter(lambda x, p: "{:,.2f}".format(x)))
    ax.set_ylabel(f"Flexibility provided ({power_resolution.label})")
    ax.set_xlabel(f"Duration ({time_resolution.label})")
    return f, ax


def plot_probability_function(func):
    mean_indoor_temperature = 19
    std_indoor_temperature = 2.5
    max_indoor_temperature = 24
    min_indoor_temperature = 14

    x_arr = np.linspace(12, 28, 200)
    y_arr = [
        func(
            x,
            mean_indoor_temperature,
            std_indoor_temperature,
            min_indoor_temperature,
            max_indoor_temperature,
        ) for x in x_arr
    ]

    f, ax = plt.subplots(figsize=figsize)
    ax.plot(x_arr, y_arr)
    ax.set_ylabel("Probability density")
    ax.set_xlabel("Indoor air temperature (ºC)")
    return f, ax


# from the thermal model function
def plot_thermal_model_results(orgdf, validf):

    f, ax = plt.subplots(figsize=figsize)
    orgdf["Indoor_temperature_degreeC"].plot(ax=ax,
                                             marker=markers[0],
                                             markevery=10000,
                                             label="Python_indoor_temperature")
    if len(validf) > 0:
        validf["Indoor_temperature_degreeC"].plot(
            ax=ax,
            marker=markers[1],
            markevery=9500,
            label="Simulink_indoor_temperature",
        )
    orgdf["Outdoor_temperature_degreeC"].plot(ax=ax,
                                              marker=markers[2],
                                              markevery=10000,
                                              label="Outdoor_temperature")
    ax.set_ylim(-10, None)
    ax.margins(0, None)
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Temperature (ºC)")
    ax.get_xaxis().set_major_formatter(
        tkr.FuncFormatter(lambda x, p: "{:,.0f}".format(x)))
    ax.legend()
    return f, ax


def create_filter(dataf: pd.DataFrame, dict_values: dict):
    filt = np.full(len(dataf), True)
    for k, v in dict_values.items():
        filt = filt & (dataf[k] == v)
    return filt


def plot_multiple_scenarios_flexibility_results(
    to_plot_df: pd.DataFrame,
    filter_variables: List[str],
    hue: str,
    time_resolution: enums.TimeResolution,
    power_resolution: enums.PowerResolution,
    ax: plt.Axes,
) -> plt.Axes:
    """Plot the results of the modelling for multiple outside air temperature and capacity factors"""
    time_correction_factor = time_resolution.factor
    power_scaling_factor = power_resolution.factor
    max_duration = to_plot_df.index.max() / time_correction_factor
    records = to_plot_df[filter_variables].drop_duplicates().to_dict(
        orient="records")
    hue_records = to_plot_df[hue].unique()
    # hue_records.sort()

    for single_record in records:
        print(single_record)
        hue_val = single_record[hue]
        ii = np.where(hue_records == hue_val)[0][0]
        filt = create_filter(to_plot_df, single_record)
        temp_df = to_plot_df.loc[filt, :]
        x_arr = list(temp_df.index / time_correction_factor) + [max_duration]
        y_arr = list(temp_df["Flexibility_kW"] / power_scaling_factor) + [0]
        if single_record["Capacity_factor"] == 0:  # add label
            ax.plot(
                x_arr,
                y_arr,
                label=hue_val,
                color=sns.color_palette()[ii],
                marker=markers[ii],
                markevery=1000,
            )
        else:  # no label
            ax.plot(
                x_arr,
                y_arr,
                color=sns.color_palette()[ii],
                marker=markers[ii],
                markevery=1000,
            )

    ax.get_xaxis().set_major_formatter(
        tkr.FuncFormatter(lambda x, p: "{:,.1f}".format(x)))
    ax.get_yaxis().set_major_formatter(
        tkr.FuncFormatter(lambda x, p: "{:,.0f}".format(x)))
    ax.set_ylabel(f"Flexibility provided ({power_resolution.label})")
    ax.set_xlabel(f"Duration ({time_resolution.label})")
    ax.margins(0, None)
    return ax


def get_legend_parameters(ax,
                          cunit: enums.Cunit,
                          within_chart: bool = True,
                          horizontal: bool = False) -> dict:
    handles, labels = ax.get_legend_handles_labels()
    print(labels)
    if cunit.variable_type == "number":
        labels = [f"{float(x):,.0f}{cunit.unit}" for x in labels]

    if horizontal:
        ncol = len(labels)
    else:
        ncol = 1

    if within_chart:
        legend_dict = dict(
            handles=handles,
            labels=labels,
            facecolor="white",
            framealpha=1,
            frameon=True,
            title=cunit.label,
            loc=4,
            ncol=ncol,
        )

        # ax.legend(
        #     lines,
        #     labels,
        #     facecolor="white",
        #     framealpha=1,
        #     frameon=True,
        #     title=cunit.label,
        #     loc=4,
        #     ncol=ncol,
        # )
    else:
        legend_dict = dict(
            handles=handles,
            labels=labels,
            facecolor="white",
            framealpha=1,
            title=cunit.label,
            ncol=ncol,
            bbox_to_anchor=(0.5, -0.38),
        )
        # ax.legend(
        #     lines,
        #     labels,
        #     facecolor="white",
        #     framealpha=1,
        #     title=cunit.label,
        #     ncol=ncol,
        #     bbox_to_anchor=(0.5, -0.38),
        # )
    return legend_dict


def distribution_thermal_characteristics(dataf: pd.DataFrame,
                                         thermal_capacity_level: str = "medium"
                                         ):
    """Visualisation of the number of dwellings for different values of the thermal losses and thermal capacity"""

    f, axs = plt.subplots(1, 2, figsize=(figsize[0], round(figsize[1])))

    hue_order = ["flat", "terraced", "semi-detached", "detached"]

    filt = dataf["Thermal capacity level"] == thermal_capacity_level
    nb_bins: int = 50

    plt.subplots_adjust(wspace=0.02, hspace=0)
    ax = axs[0]
    sns.histplot(
        data=dataf.loc[filt],
        x="Average thermal capacity kJ/K",
        hue="Dwelling forms",
        hue_order=hue_order,
        weights="Number of dwellings",
        ax=ax,
        element="poly",
        bins=nb_bins,
    )

    ax.set_ylabel("Number of dwellings (millions)")
    ax.get_yaxis().set_major_formatter(
        tkr.FuncFormatter(lambda x, p: "{:,.2f}".format(x / 1_000_000)))
    ax.get_xaxis().set_major_formatter(
        tkr.FuncFormatter(lambda x, p: "{:,.0f}".format(x / 1_000)))
    ax.set_xlabel("Average thermal capacity (MJ/K)")

    ax = axs[1]
    sns.histplot(
        data=dataf.loc[filt],
        x="Average thermal losses kW/K",
        hue="Dwelling forms",
        hue_order=hue_order,
        weights="Number of dwellings",
        ax=ax,
        element="poly",
        bins=nb_bins,
    )
    ax.set(yticklabels=[])
    ax.set_ylabel("")
    ax.legend().remove()
    ax.set_xlabel("Average thermal losses (kW/K)")
    return f, ax


def get_concat_results(
    scenario: enums.Scenario,
    use_pdf: bool,
    mean_indoor_temperature: float = 19,
    after_EE: bool = True,
    thermal_capacity_level: str = "medium",
) -> pd.DataFrame:
    """Return a dataframe with the results from the model for a scenario and a choice of assignement of inside air temperature of dwellings"""
    final_frames = []

    for OAT in [-5, 0, 5, 10]:
        for capacity_factor in [0, 1]:
            temp_df = get_results_data(
                scenario,
                use_pdf,
                mean_indoor_temperature,
                after_EE,
                thermal_capacity_level,
                OAT,
                capacity_factor,
            )
            temp_df = temp_df.sum(axis=1).to_frame()
            temp_df.columns = ["Flexibility_kW"]
            temp_df["Capacity_factor"] = capacity_factor
            temp_df["PDF_inside_air_temperature"] = False
            temp_df["Outside_air_temperature_ºC"] = OAT
            temp_df["Scenario"] = scenario.name
            final_frames.append(temp_df)
    if len(final_frames) > 0:
        results_df = pd.concat(final_frames, axis=0)
    else:
        results_df = pd.DataFrame()
    return results_df


def get_results_data(
    scenario: enums.Scenario,
    use_pdf: bool,
    mean_indoor_temperature: float,
    after_EE: bool,
    thermal_capacity_level: str,
    outside_air_temperature: float,
    capacity_factor: float,
    nb_slices: int = 5,
) -> pd.DataFrame:

    frames = []
    folder_system = fs.FolderSystem(
        outside_air_temperature=outside_air_temperature,
        use_pdf=use_pdf,
        inside_air_temperature=mean_indoor_temperature,
        after_energy_efficiency=after_EE,
        thermal_capacity_level=thermal_capacity_level,
    )
    saving_path = folder_system.set_path(create_if_missing=False)
    if saving_path == "":
        print("The results of this model run are not available.")
    else:
        print(f"Looking for results files in: {saving_path}")
        for slice in range(nb_slices):
            full_path = folder_system.get_path_with_filename(
                slice, capacity_factor, scenario)
            if folder_system.file_exist(full_path):
                print(f"Opening: {full_path}")
                slice_lsoas_df = pd.read_csv(full_path, index_col=0)
                frames.append(slice_lsoas_df)
    if len(frames) > 0:
        temp_df = pd.concat(frames, axis=1)
    else:
        temp_df = pd.DataFrame()
    return temp_df


def get_lookup_LA():
    ## get lookup table of local authority

    path = r"D:\OneDrive - Cardiff University\04 - Projects\03 - PhD\03 - Analysis\03 - LSOAs\00 - Data\Lookup tables"
    OA_lookup_file = r"Postcode to OA LSOA MSOA LAD\PCD11_OA11_LSOA11_MSOA11_LAD11_EW_LU_feb_2018.csv"
    OA_lookup_df = pd.read_csv(path + os.path.sep + OA_lookup_file,
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
        path + os.path.sep + region_lookup_file,
        sheet_name=1,
        header=6,
        usecols=[0, 1, 3],
    )

    OA_lookup_df = pd.merge(OA_lookup_df,
                            region_lookup_df,
                            left_on="ladcd",
                            right_on="la_code",
                            how="left")

    OA_lookup_df = pd.merge(OA_lookup_df,
                            region_lookup_df,
                            left_on="ladnm",
                            right_on="la_name",
                            how="left")
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

    OA_lookup_df = OA_lookup_df[["OA11CD",
                                 "Local Authority"]].drop_duplicates()

    OA_lookup_2019_file = "Output_Area_to_Ward_to_Local_Authority_District_(December_2019)_Lookup_in_England_and_Wales.csv"
    OA_lookup_2019_df = pd.read_csv(path + os.path.sep + OA_lookup_2019_file,
                                    low_memory=False)
    OA_lookup_df = pd.merge(
        OA_lookup_df,
        OA_lookup_2019_df[["OA11CD", "LAD19NM"]],
        left_on="OA11CD",
        right_on="OA11CD",
        how="left",
    )

    LA_lookup_dict = dict(
        zip(OA_lookup_df["Local Authority"], OA_lookup_df["LAD19NM"]))
    return LA_lookup_dict


def get_min_max_flexibility_results(dataf: pd.DataFrame,
                                    capacity_factor: float) -> pd.DataFrame:
    dataf.fillna(0, inplace=True)
    if capacity_factor == 0:
        dataf = dataf.min().to_frame().reset_index()
    else:
        dataf = dataf.max().to_frame().reset_index()
    dataf.columns = ["Local_authority", "Flexibility_kW"]
    LA_lookup_dict = get_lookup_LA()
    dataf["Local Authority (2019)"] = dataf["Local_authority"].map(
        LA_lookup_dict)
    dataf["Flexibility_MW"] = dataf["Flexibility_kW"] / 1_000
    return dataf


def get_color_bar(fig: plt.Figure, vmin, vmax, cmap: str):
    sm = plt.cm.ScalarMappable(cmap=cmap,
                               norm=plt.Normalize(vmin=vmin, vmax=vmax))
    sm._A = []
    cbar = fig.colorbar(sm)
    return cbar


def create_map_LA(
    df: pd.DataFrame,
    target: str,
    path_map: str,
    fig: plt.Figure,
    ax: plt.Axes,
    vmin: int = 0,
    vmax: int = 0,
    cmap=None,
):

    map_df = gpd.read_file(path_map)
    map_df.crs = {"init": "epsg:4326"}

    map_df = pd.merge(
        map_df,
        df[["Local Authority (2019)", target]],
        left_on="lad19nm",
        right_on="Local Authority (2019)",
        how="left",
    )
    map_df = map_df.dropna(subset=[target])
    cbar = None
    if len(map_df) == len(df):

        # create figure and axes for Matplotlib
        if vmin == 0:
            if map_df[target].min() < 0:
                vmin = map_df[target].min()
        if vmax == 0:
            if map_df[target].max() > 0:
                vmax = map_df[target].max()

        ax.axis("off")
        ax.tick_params(left=False,
                       labelleft=False,
                       bottom=False,
                       labelbottom=False)

        map_df = map_df.to_crs({"init": "epsg:3395"})  # mercator projections

        if cmap == None:
            cmap = "Blues"

        map_df.plot(
            column=target,
            cmap=cmap,
            linewidth=0.01,
            ax=ax,
            edgecolor="black",
            vmin=vmin,
            vmax=vmax,
        )  # ,

        cbar = get_color_bar(fig, vmin, vmax, cmap)

        plt.close()
    else:
        print("The number of LA does not match")
    return cbar

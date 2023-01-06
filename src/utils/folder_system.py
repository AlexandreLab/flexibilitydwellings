import os
from dataclasses import dataclass, field
from pathlib import Path

from ..flexibility import enums


@dataclass
class FolderSystem:

    outside_air_temperature: float
    use_pdf: bool  # use probability density function to assign inside air temperature
    inside_air_temperature: float
    after_energy_efficiency: bool  # based on annual heat demand before (False) or after (True) energy efficiency measures
    thermal_capacity_level: str  # thermal_capacity_level to be used
    main_saving_folder: str = ""
    current_path: str = ""

    def __post_init__(self) -> None:
        if os.path.basename(os.getcwd()) == "notebooks":
            self.main_saving_folder = os.path.dirname(os.getcwd())
        else:
            self.main_saving_folder = os.getcwd()
        self.main_saving_folder = os.path.join(self.main_saving_folder, "data")

    def set_path(self, create_if_missing: bool = True) -> None:
        """ Return the path where the results are stored or will be stored 
        Return empty path if the directory does not exist and was not requested to be created
        
        """
        temp_path = os.path.join(self.main_saving_folder, "results_modelling_EW")
        if self.after_energy_efficiency:
            temp_path = os.path.join(temp_path, "after_energy_efficiency")
        else:
            temp_path = os.path.join(temp_path, "before_energy_efficiency")
        if self.use_pdf:
            temp_path = os.path.join(
                temp_path,
                f"OAT_{int(self.outside_air_temperature)}_pdf_temp_thermal_capacity_{self.thermal_capacity_level}",
            )
        else:
            temp_path = os.path.join(
                temp_path,
                f"OAT_{int(self.outside_air_temperature)}_mean_temp_{int(self.inside_air_temperature)}_thermal_capacity_{self.thermal_capacity_level}",
            )

        if create_if_missing:
            if Path(temp_path).is_dir():  # the folder already exists
                print(f"{temp_path} already exists.")
            else:
                self.create_folder(temp_path)
        else:
            if not Path(temp_path).is_dir():
                temp_path = ""

        self.current_path = temp_path

    def create_folder(self, new_path: str,) -> None:
        """Create folders in the given path if the folder does not exist"""
        Path(new_path).mkdir(parents=True, exist_ok=True)
        return None

    def get_path_with_filename(
        self,
        slice_number: int,
        capacity_factor: float,
        scenario: enums.Scenario,
        filename: str = "",
    ) -> str:
        """Return the name of the full path of the where the file is and its name.
        if filename is provided it will be used instead of the default one.
        """

        if filename == "":
            filename = f"Results_slice_{slice_number}_cf_{capacity_factor}_sc_{scenario.name}.csv"
        full_path = Path(self.current_path) / filename
        return full_path.__str__()

    def file_exist(self, temp_path: str):
        temp_path_obj = Path(temp_path)
        return temp_path_obj.is_file()

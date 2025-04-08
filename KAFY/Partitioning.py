"""Partioning Module Definition"""

from os import path as os_path
from os import makedirs as os_makedirs
from json import load as json_load
from json import dump as json_dump
from typing import List, Tuple
from shapely.geometry import Point
# from .utilFunctions import load_metadata, load_tokenized_trajectories
from h3 import cell_to_latlng
import logging
# Configure the logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)
from numpy import sqrt as np_sqrt
# I will have one pyramid pointing to all models for all operations, no pretraining and finetuning pyramids.
# 

class UnexpectedError(Exception):
    """Custom exception to signal unexpected errors."""
    def __init__(self, message="An unexpected error occurred. This should not happen."):
        super().__init__(message)

class PartitioningModule:
    """
    A class to manage a hierarchical pyramid structure for storing and updating models
    based on trajectory datasets. The pyramid structure is defined by levels and cells,
    with each cell potentially containing a model. The class handles the initialization
    of the pyramid structure, loading configuration parameters, and updating the model
    repository with new datasets.
    """

    def __init__(self, project_path):
        """
        Initializes the PartitioningModule with configurations read from a JSON file.
        """

        self.models_repo_path = os_path.join(project_path,"modelsRepo")

        self.config_file = os_path.join(project_path, "pyramidConfigs.json")
        self.pyramid_path= os_path.join(
           project_path,  "partioningPyramid.json"
        )
        self.pyramid = {}
        self.tokens_threshold_per_cell = 100
        self.load_config()
        if self.build_pyramid_flag:
            self.build_pyramid(self.pyramid_path)
            with open(self.config_file, 'r+') as file:
                data = json_load(file); data["build_pyramid_from_scratch"] = False; file.seek(0); json_dump(data, file, indent=4); file.truncate()
        elif not self.build_pyramid_flag:
            # pyramid was build before, so just load it
             self.pyramid = self.load_pyramid(self.pyramid_path)
        else:
            raise UnexpectedError()


    def load_config(self):
        """
        Loads the configuration parameters H and L from the JSON file.
        """
        default_configs = {"H": 5, "L": 3, "build_pyramid_from_scratch": True}
        if not os_path.isfile(self.config_file):
            with open(self.config_file, "w", encoding="utf-8") as file:
                json_dump(default_configs, file, indent=4)
            raise Warning(
                "Pyramid Configurations File were not found\n Will assign default configurations."
            )
        with open(self.config_file, "r", encoding="utf-8") as file:
            config = json_load(file)
            self.pyramid_height = config.get("H", 5)  # Default to 5 if not specified
            self.pyramid_levels = config.get("L", 3)  # Default to 3 if not specified
            self.build_pyramid_flag = config.get("build_pyramid_from_scratch")
       
    def _calculate_bounds(self, h, index):
        """
        Calculates the bounds for a cell at a given height and index.
        """
        # Total number of cells at height h
        num_cells = 4**h
        # Determine the number of cells per side (sqrt(num_cells))
        cells_per_side = int(np_sqrt(num_cells))

        # Calculate the size of each cell
        lat_step = 180 / cells_per_side
        lon_step = 360 / cells_per_side
        # Calculate row and column of the cell
        row = index // cells_per_side
        col = index % cells_per_side

        # Calculate bounds
        min_lat =  round(90 - (row * lat_step), 6)
        max_lat = round(min_lat - lat_step, 6)
        min_lon = round(-180 + (col * lon_step), 6)
        max_lon = round(min_lon + lon_step, 6)

        return (min_lat, max_lat, min_lon, max_lon)


    def _generate_cells(self, h):
        """
        Generates cells for a given height h.
        """
        num_cells = 4**h
        cells = {}
        for i in range(num_cells):
            cells[i] = {
                "height": h,
                "index": i,
                "bounds": self._calculate_bounds(h, i),
                "occupied": False,
                "model_path": None, #Any model mentioned in metadata should be stored in this directory in modelsRepo and dataset in the same directory under TrajStore
                "metadata":{},  #This will store the nameOfDataset:model , e.g. classification:["Bert","DistillBert"], prediction:["Bert"],...
                "num_tokens": 0,
            }
        return cells

    def build_pyramid(self,location):
        """
        Builds the pyramid data structure for the models repository.
        """
        self.pyramid = {}  # Reset the pyramid structure
        for l in range(self.pyramid_height + 1):
            self.pyramid[l] = self._generate_cells(l)
        # Save the pyramid to the JSON file
        # Ensure the directory exists
        os_makedirs(os_path.dirname(location), exist_ok=True)
        with open(location, "w", encoding="utf-8") as file:
            json_dump(self.pyramid, file, indent=4)
        logging.info("Successfully built the partitioning Pyramid from scratch based on configurations in pyramidConfigs.json")
    def load_pyramid(self, pyramid_path=None):
        """
        Loads the pyramid data structure from the JSON file given a path to the pyramid.
        If no path is provided it defaults to the self.pyramid_path which is basically
        a combination of the modelsRepoPath/operation , i.e. I use this by default
        but if I want specifically to load a pyramid I pass the argument
        """

        if os_path.exists(pyramid_path):
            with open(pyramid_path, "r", encoding="utf-8") as file:
                pyramid = json_load(file)
        else:
            raise FileNotFoundError(f"Pyramid file not found at {self.pyramid_path}")

        return pyramid

    def save_pyramid(self):
        os_makedirs(os_path.dirname(self.pyramid_path), exist_ok=True)
        with open(self.pyramid_path, "w", encoding="utf-8") as file:
            json_dump(self.pyramid, file, indent=4)

    def calculate_mbr_gps(self,trajectory_list: List[List[Point]]) -> Tuple[float, float, float, float]:
        """
        Calculates the minimum bounding rectangle (MBR) for a set of trajectories.
        
        Args:
            trajectory_list (List[List[Tuple[float, float]]]): List of trajectories, 
                where each trajectory is a list of (latitude, longitude) tuples.

        Returns:
            Tuple[float, float, float, float]: A tuple representing the MBR (min_lat, max_lat, min_lon, max_lon).
        """
        min_lat = min_lon = float("inf")
        max_lat = max_lon = float("-inf")
        # Iterate over each trajectory and each GPS point to find the min/max latitude and longitude
        for trajectory in trajectory_list:
            for point in trajectory:
                min_lat = round(min(min_lat, float(point.y)), 6)
                max_lat = round(max(max_lat, float(point.y)), 6)
                min_lon = round(min(min_lon, float(point.x)), 6)
                max_lon = round(max(max_lon, float(point.x)), 6)
        
        return (min_lat, max_lat, min_lon, max_lon)    
    def _is_bounding_rectangle_enclosed(self, rectangle, cell_bounds):
        """
        Checks if a bounding rectangle is fully enclosed within the cell bounds.
        """
        lat_min, lat_max, lon_min, lon_max = rectangle
        cell_lat_max, cell_lat_min, cell_lon_min, cell_lon_max = cell_bounds
        return (
            lat_min >= cell_lat_min
            and lat_max <= cell_lat_max
            and lon_min >= cell_lon_min
            and lon_max <= cell_lon_max
        )


    def _find_enclosing_cell_of_trajectory_list(self, trajectory_list: List[List[Point]]):
        """
        Finds the smallest cell that fully encloses a given set of trajectories.
        """
        # First find the MBR of the trajectory

        bounding_rectangle = self.calculate_mbr_gps(trajectory_list)
        for l in reversed(range(self.pyramid_height + 1)):
            for i, cell in self.pyramid[str(l)].items():
                if self._is_bounding_rectangle_enclosed(
                    bounding_rectangle, cell["bounds"]
                ):
                    return cell
        return None
    def _update_cell_with_model(self, operation, cell, num_tokens):
        """
        Updates the cell with a new model and stores it in the models repository.
        """
        l = cell["height"]
        index = cell["index"]
        cell_path = os_path.join(self.models_repo_path, operation, f"{l}_{index}")

        # Create the directory if it doesn't exist
        if not os_path.exists(cell_path):
            os_makedirs(cell_path)

        # Define the model path
        cell["model_path"] = cell_path
        cell["occupied"] = True
        # @YoussefDo: I need to think about the logic of integrating two datasets together
        # and linking the dataset in the trajectory story to this cell
        cell["num_tokens"] = num_tokens



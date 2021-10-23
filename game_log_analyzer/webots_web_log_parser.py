import re
import os
import json
import numpy as np
import xml.etree.ElementTree as ET
from functools import cache, cached_property

import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

class WebotsGameLogParser:
    """
    All the gamelogs for a given game folder
    """
    def __init__(self, log_folder: str, verbose=True):
        super().__init__()

        self.log_folder = log_folder

        x3d_file = [file for file in os.listdir(self.log_folder) if file.endswith(".x3d")][0]
        self.x3d = X3DParser(os.path.join(log_folder, x3d_file))

        game_json_file = [file for file in os.listdir(self.log_folder) if file.endswith(".json")][0]
        self.game_data = GameJsonParser(os.path.join(log_folder, game_json_file))

        if verbose:
            print(f"Duration: {self.get_max_player_timestamp() / 60 :.2f} Minutes")
            print(f"Players: {', '.join(self.x3d.get_player_names())}")

    def plot_player_paths(self):
        """
        Creates a combined plot with the paths for all the players
        """
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')
        for bot in self.x3d.get_player_names():
            ax.plot(*self.game_data.get_translations_for_id(self.x3d.get_object_id(bot)).T)
        plt.show()

    def plot_path(self, id: int):
        """
        Creates a plot with the path for a certain object
        """
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')
        ax.plot(*self.game_data.get_translations_for_id(id).T)
        plt.show()

    def get_max_player_timestamp(self) -> float:
        """
        Returns the last timestamp where any of the players has movement data
        """
        return max(self.game_data.get_timestamps_for_id(id).max() for id in self.x3d.get_player_ids())


class GameJsonParser:
    """
    Parses time series information from the .json log
    """
    def __init__(self, jsonfile: str):
        self.jsonfile = jsonfile

        # Load file contents
        with open(self.jsonfile, "r") as f:
            self.data = json.load(f)

    @cached_property
    def get_time_step_size(self) -> int:
        """
        Returns the basic time step in ms.
        """
        return self.data["basicTimeStep"]

    def _parse_str_vector(self, vec: str) -> np.ndarray:
        """
        Converts a space divided string vector into a NumPy array.
        """
        return np.array([float(num) for num in vec.split(" ")], dtype=float)

    @cache
    def get_poses_for_id(self, id: int) -> [dict]:
        """
        Gets all the pose data for an object and
        return a list of dicts containing the time and the pose.
        """
        poses = []
        for frame in self.data["frames"]:
            time = frame["time"]
            if frame["poses"]:
                for sim_object in frame["poses"]:
                    if sim_object["id"] == id:
                        if "translation" in sim_object.keys():
                            translation = self._parse_str_vector(sim_object["translation"])
                        if "rotation" in sim_object.keys():
                            rotation = self._parse_str_vector(sim_object["rotation"])
                        poses.append({
                            "time": time,
                            "trans": translation,
                            "rot": rotation
                        })
                        break
        return poses

    def get_translations_for_id(self, id: int) -> np.ndarray:
        """
        Returns an NumPy array with all translations for an object
        """
        translations = list(map(lambda x: x["trans"], self.get_poses_for_id(id)))
        return np.array(translations, dtype=float)

    def get_timestamps_for_id(self, id: int) -> np.ndarray:
        """
        Returns an NumPy array with all timesteps for an object
        """
        timesteps = list(map(lambda x: x["time"], self.get_poses_for_id(id)))
        return (np.array(timesteps, dtype=np.float) / 1000)


class X3DParser:
    """
    Parses the .x3d file in the gamelogs, that contains all the 3d object information.
    """
    def __init__(self, x3dfile: str):
        super(X3DParser).__init__()
        self.x3d_file_path = x3dfile

        # Load file contents
        with open(self.x3d_file_path, "r") as f:
            x3d_file_raw_content = f.read()

        self.xml_root = ET.fromstring(x3d_file_raw_content)

    @cache
    def get_players(self) -> list[dict]:
        """
        Get the names of the loaded players
        """
        def is_player(node: ET.Element) -> bool:
            return "player" in node.attrib.get("name", "")

        def simplify_dict(node: ET.Element) -> dict:
            return {"id": int(node.attrib["id"][1:]),
                    "name": str(node.attrib["name"])}

        return list(map(simplify_dict, filter(is_player, self.xml_root.iter("Transform"))))

    def get_player_names(self) -> [str]:
        """
        Returns a list with all player names
        """
        return list(map(lambda x: x["name"], self.get_players()))

    def get_object_id(self, name: str) -> int or None:
        """
        The object id for a given name
        """
        return (list(map(lambda x: x["id"], filter(lambda x: x["name"] == name, self.get_players()))) + [None])[0]

    def get_player_ids(self) -> [int]:
        """
        Returns a list with all player object ids
        """
        return list(map(self.get_object_id, self.get_player_names()))


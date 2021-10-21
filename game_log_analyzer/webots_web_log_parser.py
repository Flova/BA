import re
import os
import json
import numpy as np
import xml.etree.ElementTree as ET
from functools import cache, cached_property

class WebotsGameLogParser:
    """
    All the gamelogs for a given game folder
    """
    def __init__(self, log_folder: str):
        super().__init__()

        self.log_folder = log_folder

        x3d_file = [file for file in os.listdir(self.log_folder) if file.endswith(".x3d")][0]
        self.x3d = X3DParser(os.path.join(log_folder, x3d_file))

        game_json_file = [file for file in os.listdir(self.log_folder) if file.endswith(".json")][0]
        self.game_data = GameJsonParser(os.path.join(log_folder, game_json_file))

    def plot_paths(self):
        import matplotlib.pyplot as plt
        from mpl_toolkits.mplot3d import Axes3D
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')
        for bot in self.x3d.get_player_names():
            ax.plot(*self.game_data.get_translations_for_id(self.x3d.get_player_id(bot)).T)
        fig.show()

    def plot_path(self, id: int):
        import matplotlib.pyplot as plt
        from mpl_toolkits.mplot3d import Axes3D
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')
        ax.plot(*self.game_data.get_translations_for_id(id).T)
        fig.show()


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
        return self.data["basicTimeStep"]

    def _parse_str_vector(self, vec: str) -> np.ndarray:
        return np.array([float(num) for num in vec.split(" ")], dtype=np.float)

    @cache
    def get_poses_for_id(self, id: int) -> dict:
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
        translations = list(map(lambda x: x["trans"], self.get_poses_for_id(id)))
        return np.array(translations, dtype=np.float)

    def get_timestamps_for_id(self, id: int) -> np.ndarray:
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
        return list(map(lambda x: x["name"], self.get_players()))

    def get_player_id(self, name: str) -> int or None:
        return (list(map(lambda x: x["id"], filter(lambda x: x["name"] == name, self.get_players()))) + [None])[0]

    def get_player_ids(self) -> [int]:
        return list(map(self.get_player_id, self.get_player_names()))

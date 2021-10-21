import re
import os
import json
import numpy as np

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

    def plot_path(id: int):
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

    def get_time_step_size(self) -> int:
        return self.data["basicTimeStep"]

    def _parse_str_vector(self, vec: str) -> np.ndarray:
        return np.array([float(num) for num in vec.split(" ")], dtype=np.float)

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
        translations = []
        for pose in self.get_poses_for_id(id):
            translations.append(pose["trans"])
        return np.array(translations)


class X3DParser:
    """
    Parses the .x3d file in the gamelogs, that contains all the 3d object information.
    """
    def __init__(self, x3dfile: str):
        super(X3DParser).__init__()
        self.x3d_file_path = x3dfile

        # Load file contents
        with open(self.x3d_file_path, "r") as f:
            self.x3d_file_raw_content = f.read()

    def get_object_names(self) -> list[str]:
        """
        Get the names of all 3d objects
        """
        return re.findall( r'name=\'(.*?)\'', self.x3d_file_raw_content)

    def get_players(self) -> list[str]:
        """
        Get the names of the loaded players
        """
        def is_player(name: str) -> bool:
            return "player" in name
        return list(filter(is_player, self.get_object_names()))

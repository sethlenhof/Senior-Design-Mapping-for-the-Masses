# scene_manager.py

import pandas as pd
import numpy as np
import os

class SceneManager:
    def __init__(self, upload_folder, download_folder):
        self.upload_folder = upload_folder
        self.download_folder = download_folder

    def load_point_cloud(self, filename):
        """Load a point cloud file into a dataframe."""
        file_path = os.path.join(self.upload_folder, filename)
        return pd.read_csv(file_path, header=None, delimiter=' ', skiprows=[0])

    def center_point_cloud(self, dataframe):
        """Center the point cloud to the origin."""
        dataframe.loc[:, 1] = dataframe[1] - dataframe[1].min()
        dataframe.loc[:, 0] = dataframe[0] - dataframe[0].min()
        dataframe.loc[:, 2] = dataframe[2] - dataframe[2].min()
        return dataframe

    def save_point_cloud(self, df, filename):
        """Save a point cloud to the download folder."""
        output_path = os.path.join(self.download_folder, filename)
        df.to_csv(output_path, header=False, index=False, sep=' ')
        print(f"Point cloud saved to {output_path}")

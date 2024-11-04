import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import mpl_toolkits.mplot3d.art3d as art3d
import numpy as np
import pandas as pd
import os
import math
from typing import List
import scipy.spatial

class Node:
    def __init__(self, df: pd.DataFrame):
        self.df: pd.DataFrame = df
        self.value: int = None
        self.shiftInX: float = df[1].min()
        self.shiftInY: float = df[0].min()
        self.shiftInZ: float = df[2].min()


def load_xyz_file(filepath: str) -> pd.DataFrame:
    try:
        return pd.read_csv(filepath, header=None, delimiter=' ', skiprows=[0])
    except Exception as e:
        print(f"Error loading file {filepath}: {e}")
        sys.exit(1)


def rotate_data(data: pd.DataFrame, angle_deg: float, axis: str = 'y') -> pd.DataFrame:
    angle_rad = np.radians(angle_deg)
    if axis == 'y':
        rotation_matrix = np.array([
            [np.cos(angle_rad), 0, np.sin(angle_rad)],
            [0, 1, 0],
            [-np.sin(angle_rad), 0, np.cos(angle_rad)]
        ])
    return pd.DataFrame(np.dot(data, rotation_matrix))


def bring_to_center(dataframe: pd.DataFrame) -> None:
    dataframe.loc[:, 1] = dataframe[1] - dataframe[1].min()
    dataframe.loc[:, 0] = dataframe[0] - dataframe[0].min()
    dataframe.loc[:, 2] = dataframe[2] - dataframe[2].min()


def plot_environment(ax: Axes3D, data: pd.DataFrame, color: str, label: str) -> None:
    ax.scatter(data[0], data[2], data[1], color=color, s=5, label=label)


def setup_plot() -> Axes3D:
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.set_xlabel('x')
    ax.set_ylabel('y')
    ax.set_zlabel('z')
    return ax


def build_subsection_list(littleDF: pd.DataFrame, dataframe: pd.DataFrame) -> List[pd.DataFrame]:
    maxx = math.ceil(dataframe[0].max())
    maxy = math.ceil(dataframe[2].max())
    littlex = math.ceil(littleDF[0].max())
    littley = math.ceil(littleDF[2].max())

    dataframes = []

    i = 0
    increment = 0.5
    while i < maxx:
        j = 0
        while j < maxy:
            i_df = dataframe[(dataframe[0] < i + littlex) & (dataframe[0] > i) & 
                             (dataframe[2] < j + littley) & (dataframe[2] > j)]
            dataframes.append(i_df)
            j += increment
        i += increment

    return dataframes


def find_location(littleDF: pd.DataFrame, nodeList: List[Node]) -> int:
    firstResult = scipy.spatial.distance.directed_hausdorff(littleDF, nodeList[0].df, seed=0)
    bestHausdorff = firstResult[0]
    retval = 0

    for i in range(len(nodeList)):
        result = scipy.spatial.distance.directed_hausdorff(littleDF, nodeList[i].df, seed=0)
        current_hausdorff = result[0]

        if current_hausdorff < bestHausdorff:
            bestHausdorff = current_hausdorff
            retval = i
    return retval


def align_blueprint_to_user(little: pd.DataFrame, blueprint: pd.DataFrame, matched_node: Node) -> pd.DataFrame:
    """
    Aligns the blueprint data to match the user's location.

    Args:
        little (pd.DataFrame): User environment point cloud data.
        blueprint (pd.DataFrame): Blueprint point cloud data.
        matched_node (Node): The node representing the matched subsection.

    Returns:
        pd.DataFrame: Shifted blueprint data.
    """
    # Bring matched node to user location
    shift_in_x = little[1].min() - matched_node.shiftInX
    shift_in_y = little[0].min() - matched_node.shiftInY
    shift_in_z = little[2].min() - matched_node.shiftInZ

    blueprint.loc[:, 1] = blueprint[1] + shift_in_x
    blueprint.loc[:, 0] = blueprint[0] + shift_in_y
    blueprint.loc[:, 2] = blueprint[2] + shift_in_z

    return blueprint


def load_files(download_folder: str, upload_folder: str) -> plt.Figure:
    pd.set_option('mode.chained_assignment', None)
    
    # Load data
    blueprint = load_xyz_file(os.path.join(upload_folder, 'blueprint.xyz'))
    little = load_xyz_file(os.path.join(upload_folder, 'userEnvironment.xyz'))

    # Rotate data and bring to center
    blueprint[0] *= -1
    blueprint = rotate_data(blueprint, 0)
    bring_to_center(blueprint)
    bring_to_center(little)

    # Build subsections
    dflist = build_subsection_list(little, blueprint)
    nodes = [Node(df) for df in dflist]

    # Find the matching subsection
    result_index = find_location(little, nodes)
    matched_node = nodes[result_index]

    # Align the blueprint to the user environment
    aligned_blueprint = align_blueprint_to_user(little, blueprint, matched_node)

    # Set up plotting
    ax = setup_plot()
    # Adjust view to be top-down, birds-eye
    ax.view_init(elev=90, azim=0)
    circle = plt.Circle((0, 0), 0.1, color='r')
    ax.add_patch(circle)
    art3d.pathpatch_2d_to_3d(circle, z=0, zdir='z')
    plot_environment(ax, aligned_blueprint, 'b', 'Blueprint')
    plot_environment(ax, little, 'y', 'User Environment')
    plt.savefig(os.path.join(download_folder, 'export.png'))

    return plt

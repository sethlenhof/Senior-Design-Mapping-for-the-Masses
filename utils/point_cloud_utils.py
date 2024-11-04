import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import numpy as np
import pandas as pd
import math
import scipy.spatial
import os
import mpl_toolkits.mplot3d.art3d as art3d

# ignore warnings
pd.set_option('mode.chained_assignment', None)

def generate_plot(upload_folder: str, download_folder: str) -> None:
    """
    Loads, processes, and visualizes point cloud data from specified folders, then saves the output plot.

    Args:
        upload_folder (str): Path to the folder containing input files.
        download_folder (str): Path to the folder for saving output files.
    """
    # Load point cloud data
    blueprintfile = os.path.join(upload_folder, 'blueprint.xyz')
    blueprint = pd.read_csv(blueprintfile, header=None, delimiter=' ', skiprows=[0])

    userfile = os.path.join(upload_folder, 'userEnvironment.xyz')
    little = pd.read_csv(userfile, header=None, delimiter=' ', skiprows=[0])

    # Reverse x values for blueprint and user environment if needed
    blueprint[0] = blueprint[0] * -1
    little[0] = little[0] * -1

    # Record shifts to revert later
    little_shift_in_x = little[1].min()
    little_shift_in_y = little[0].min()
    little_shift_in_z = little[2].min()

    # Rotate blueprint (if necessary)
    angle_deg = 0
    yaw_angle_rad = np.radians(angle_deg)
    rotation_matrix = np.array([
        [np.cos(yaw_angle_rad), 0, np.sin(yaw_angle_rad)],
        [0, 1, 0],
        [-np.sin(yaw_angle_rad), 0, np.cos(yaw_angle_rad)]
    ])
    blueprint = pd.DataFrame(np.dot(blueprint, rotation_matrix))

    # Function to bring DataFrame to (0, 0, 0)
    def bring_to_center(dataframe):
        dataframe.loc[:, 1] = dataframe[1] - dataframe[1].min()
        dataframe.loc[:, 0] = dataframe[0] - dataframe[0].min()
        dataframe.loc[:, 2] = dataframe[2] - dataframe[2].min()

    # Bring dataframes to (0, 0, 0)
    bring_to_center(blueprint)
    bring_to_center(little)

    # Build subsection list of blueprint
    def build_subsection_list(little_df, dataframe):
        maxx = math.ceil(dataframe[0].max())
        maxy = math.ceil(dataframe[2].max())
        littlex = math.ceil(little_df[0].max())
        littley = math.ceil(little_df[2].max())

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

    dflist = build_subsection_list(little, blueprint)

    # Create Node class to hold subsection dataframes
    class Node:
        def __init__(self, df):
            self.df = df
            self.value = None
            self.shift_in_x = df[1].min()
            self.shift_in_y = df[0].min()
            self.shift_in_z = df[2].min()

    # Turn subsections into nodes and bring each to center
    nodes = []
    for i in range(len(dflist)):
        temp = Node(dflist[i])
        temp.value = i
        bring_to_center(temp.df)
        nodes.append(temp)

    # Find the most similar node using Hausdorff distance
    def find_location(little_df, node_list):
        first_result = scipy.spatial.distance.directed_hausdorff(little_df, node_list[0].df, seed=0)
        best_hausdorff = first_result[0]
        retval = 0

        for i in range(len(node_list)):
            result = scipy.spatial.distance.directed_hausdorff(little_df, node_list[i].df, seed=0)
            current_hausdorff = result[0]
            if current_hausdorff < best_hausdorff:
                best_hausdorff = current_hausdorff
                retval = i
        return retval

    result_index = find_location(little, nodes)

    # Move user environment back to original spot
    little.loc[:, 1] = little[1] + little_shift_in_x
    little.loc[:, 0] = little[0] + little_shift_in_y
    little.loc[:, 2] = little[2] + little_shift_in_z

    # Move the entire blueprint to match user location
    blueprint.loc[:, 1] = blueprint[1] - nodes[result_index].shift_in_x
    blueprint.loc[:, 0] = blueprint[0] - nodes[result_index].shift_in_y
    blueprint.loc[:, 2] = blueprint[2] - nodes[result_index].shift_in_z

    blueprint.loc[:, 1] = blueprint[1] + little_shift_in_x
    blueprint.loc[:, 0] = blueprint[0] + little_shift_in_y
    blueprint.loc[:, 2] = blueprint[2] + little_shift_in_z

    # Set up plot
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.set_xlabel('x')
    ax.set_ylabel('y')
    ax.set_zlabel('z')

    # Plot blueprint in blue
    ax.scatter(blueprint[0], blueprint[2], blueprint[1], color="b", s=5)

    # Plot user environment in yellow
    ax.scatter(little[0], little[2], little[1], color="y", s=5)

    # Show results from bird's eye view and add a red dot for user's location
    ax.view_init(elev=90, azim=0)
    circle = plt.Circle((0, 0), 0.1, color='r')
    ax.add_patch(circle)
    art3d.pathpatch_2d_to_3d(circle, z=0, zdir='z')
    ax.axis("equal")

    # Save the plot to the download folder
    downloadfile = os.path.join(download_folder, 'export.png')
    plt.savefig(downloadfile)
    plt.close()
import math
import numpy as np
import pandas as pd
import scipy
import os

# ignore warnings
pd.set_option('mode.chained_assignment', None)

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
UPLOAD_FOLDER = os.path.join(BASE_DIR, '../uploads')
DOWNLOAD_FOLDER = os.path.join(BASE_DIR, '../downloads')

def process_point_clouds():
    # Load blueprint and user environment point clouds
    blueprintfile = os.path.join(UPLOAD_FOLDER, 'blueprint.xyz')
    blueprint = pd.read_csv(blueprintfile, header=None, delimiter=' ', skiprows=[0])

    userfile = os.path.join(UPLOAD_FOLDER, 'userEnvironment.xyz')
    little = pd.read_csv(userfile, header=None, delimiter=' ', skiprows=[0])

    # Reverse the x values for both point clouds
    blueprint[0] = blueprint[0] * -1
    little[0] = little[0] * -1

    # Record shifts for the user environment
    littleShiftInX = little[1].min()
    littleShiftInY = little[0].min()
    littleShiftInZ = little[2].min()

    # Rotate the blueprint if necessary
    angle_deg = 0  # Change this if needed
    yaw_angle_rad = np.radians(angle_deg)
    rotation_matrix = np.array([
        [np.cos(yaw_angle_rad), 0, np.sin(yaw_angle_rad)],
        [0, 1, 0],
        [-np.sin(yaw_angle_rad), 0, np.cos(yaw_angle_rad)]
    ])
    tempnp = np.dot(blueprint, rotation_matrix)
    blueprint = pd.DataFrame(tempnp)

    # Bring dataframes to the center (0, 0, 0)
    def bringToCenter(dataframe):
        dataframe.loc[:, 1] = dataframe[1] - dataframe[1].min()
        dataframe.loc[:, 0] = dataframe[0] - dataframe[0].min()
        dataframe.loc[:, 2] = dataframe[2] - dataframe[2].min()

    bringToCenter(blueprint)
    bringToCenter(little)

    # Define Node class for subsections
    class Node:
        def __init__(self, df):
            self.df = df
            self.value = None
            self.shiftInX = df[1].min()
            self.shiftInY = df[0].min()
            self.shiftInZ = df[2].min()

    # Build subsections of the blueprint
    def buildSubsectionList(littleDF, dataframe):
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

    dflist = buildSubsectionList(little, blueprint)

    # Create nodes for each subsection
    nodes = []
    for i in range(len(dflist)):
        temp = Node(dflist[i])
        temp.val = i
        bringToCenter(temp.df)
        nodes.append(temp)

    # Find the location of the best matching subsection using Hausdorff distance
    def findLocation(littleDF, nodeList):
        firstResult = scipy.spatial.distance.directed_hausdorff(littleDF.values, nodeList[0].df.values)
        bestHausdorff = firstResult[0]
        retval = 0
        for i in range(len(nodeList)):
            result = scipy.spatial.distance.directed_hausdorff(littleDF.values, nodeList[i].df.values)
            current_hausdorff = result[0]
            if current_hausdorff < bestHausdorff:
                bestHausdorff = current_hausdorff
                retval = i
        return retval

    result = findLocation(little, nodes)

    # Move user environment back to its original spot
    little.loc[:, 1] = little[1] + littleShiftInX
    little.loc[:, 0] = little[0] + littleShiftInY
    little.loc[:, 2] = little[2] + littleShiftInZ

    # Move the entire blueprint to match the user environment's location
    blueprint.loc[:, 1] = blueprint[1] - nodes[result].shiftInX
    blueprint.loc[:, 0] = blueprint[0] - nodes[result].shiftInY
    blueprint.loc[:, 2] = blueprint[2] - nodes[result].shiftInZ
    blueprint.loc[:, 1] = blueprint[1] + littleShiftInX
    blueprint.loc[:, 0] = blueprint[0] + littleShiftInY
    blueprint.loc[:, 2] = blueprint[2] + littleShiftInZ

    # Return blueprint and little for further use (for PNG or PLY generation)
    return blueprint, little

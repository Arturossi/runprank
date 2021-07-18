#!/usr/bin/python3

# Description
###############################################################################
'''
runprank.py is a script to run the software p2rank and then convert its output
to box coordinates to be used as input to docking software like Vina

Created by: Artur Duque Rossi
Version: 0.4
'''

import os
import time
import subprocess
import numpy as np
import pandas as pd

def __cart2pol(x, y):
    '''
    Transform Cartesian to polar coordinates
    Input:
     x [double] - The x coordinate
     y [double] - The y coordinate
    Return:
      theta [double] - The theta angle
      rho   [double] - The rho radial coordinate
    '''
    theta = np.arctan2(y, x)
    rho = np.hypot(x, y)
    return theta, rho

def __pol2cart(theta, rho):
    '''
    Transform polar to Cartesian coordinates
    Input:
     theta [double] - The theta angle
     rho   [double] - The rho radial coordinate
    Return:
      x [double] - The x coordinate
      y [double] - The y coordinate
    '''
    x = rho * np.cos(theta)
    y = rho * np.sin(theta)
    return x, y

def __cart2sph(x, y, z):
    '''
    Transform Cartesian to spherical coordinates
    Input:
     x [double] - The x coordinate
     y [double] - The y coordinate
     z [double] - The z coordinate
    Return:
      az [double] - The azimuth
      el [double] - The elevation
      r  [double] - The r radial coordinate
    '''
    hxy = np.hypot(x, y)
    r = np.hypot(hxy, z)
    el = np.arctan2(z, hxy)
    az = np.arctan2(y, x)
    return az, el, r

def __sph2cart(az, el, r):
    '''
    Transform spherical to Cartesian coordinates
    Input:
     az [double] - The azimuth
     el [double] - The elevation
     r  [double] - The radial coordinate
    Return:
      x [double] - The x coordinate
      y [double] - The y coordinate
      z [double] - The z coordinate
    '''
    rcos_theta = r * np.cos(el)
    x = rcos_theta * np.cos(az)
    y = rcos_theta * np.sin(az)
    z = r * np.sin(el)
    return x, y, z

def __safe_create_dir(dirname):
    '''
    Function to create a dir if not exists
    Input:
     dirname [string] - File path to be untarred
    Return:
      0 if success
      1 if folder exists
     -1 if any problem has occurred
     -2 should not appear
    '''

    try:
        if not os.path.isdir(dirname):
            os.makedirs(dirname)
            return 0
        else:
            print(f"The dir {dirname} already exists, aborting its creation")
            return 1
    except Exception as e:
        print(f"Error! Exception: {e}")
        exit(-1)
    return -2

def __process_cluster(clustering, coordinates, fout, suffix = "", coordSystem = "cartesian", spacing = 3.0, maxCutoff = 0.5):
    '''
    Function to process the cluster object and print a box file
    Input:
     clustering   [cluster result object]       - SciKit clustering object resulted from any clustering function after fitting
     coordinates  [np.array(np.array(float))]   - NumPy array of numpy arrays of 3 floats containg the X Y Z coordinates
     fout         [string]                      - The path to output box files
     suffix       [string] DEFAULT: ""          - The suffix to append to box files and to create containing folders
     coordSystem  [string] DEFAULT: "cartesian" - The coordinate system to be used. The options are cartesian, polar, spherical
     spacing      [float]  DEFAULT: 3.0         - Expansion size of the box in angstroms
     maxCutoff    [float]  DEFAULT: 0.5         - If the probability value from p2rank is above this value, the pocket WILL be considered as valid, even if its value is below the cutoff (use 1.0 to disable this feature)
    Return:
        Nothing
    '''
    # Fetch each element label
    labels = clustering.labels_

    # Find which labels exists removing repeated elements
    labels_unique = np.unique(labels)

    # Convert coordinates (if necessary)
    if coordSystem.lower() == "polar": # if is polar
        # For each element in coordinates array
        for i, coordinate in enumerate(coordinates):
            # Convert the first two elements (theta, ro) to cartesian (x, y). There is no need to convert z
            coordinates[i][0], coordinates[i][1] = __pol2cart(coordinates[i][0], coordinates[i][1]) #

    elif coordSystem.lower() == "spherical": # if is spherical
        # For each element in coordinates array
        for i, coordinate in enumerate(coordinates):
            # Convert the first three elements (azimuth, elevation, radial coordinate) to cartesian (x, y, z)
            coordinates[i][0], coordinates[i][1], coordinates[i][2] = __sph2cart(coordinates[i][0], coordinates[i][1], coordinates[i][2])

    # Create a dataframe containing x, y, z coordinates and the probability and the rank from P2Rank
    clusteringdf = pd.DataFrame(coordinates,  columns=['x', 'y', 'z', 'probability', 'rank'])

    # Add label column to the clusteringdf dataframe
    clusteringdf['label'] = labels

    # If the variable suffix is set
    if suffix:
        # Set the folder variable
        folder = f"/{suffix}"
        # Create the folder
        __safe_create_dir(f"{fout}{folder}")
        # Change the suffix (to concatenate in box filename)
        suffix = f"_{suffix}"
    else:
        # Set the folder variable as empty
        folder = ""

    # Set the cutoff as the mean of the probabilities (from P2Rank)
    cutoff = clusteringdf['probability'].mean()

    # Force the cutoff to be at maximum the maxCutoff variable
    cutoff = cutoff if cutoff < maxCutoff else maxCutoff

    # For each unique label (after removing repeated labels)
    for label_unique in labels_unique:
        # If the label is -1 (means that its an outlier) or if no probability of the set is above the cutoff
        if str(label_unique) == "-1" or not (clusteringdf[clusteringdf['label'] == label_unique]['probability'] >= cutoff).any():
            # Next iteration
            continue

        # Get min/max of the x/y/z coordinates (round to 3 decimals)
        min_x = round(clusteringdf[clusteringdf['label'] == label_unique]['x'].min() - spacing, 3)
        max_x = round(clusteringdf[clusteringdf['label'] == label_unique]['x'].max() + spacing, 3)
        min_y = round(clusteringdf[clusteringdf['label'] == label_unique]['y'].min() - spacing, 3)
        max_y = round(clusteringdf[clusteringdf['label'] == label_unique]['y'].max() + spacing, 3)
        min_z = round(clusteringdf[clusteringdf['label'] == label_unique]['z'].min() - spacing, 3)
        max_z = round(clusteringdf[clusteringdf['label'] == label_unique]['z'].max() + spacing, 3)

        # Get dimensions for each axis and its center (round to 3 decimals)
        dim_x = round(abs(min_x)+abs(max_x), 3)
        dim_y = round(abs(min_y)+abs(max_y), 3)
        dim_z = round(abs(min_z)+abs(max_z), 3)
        center_x = round(dim_x/2, 3)
        center_y = round(dim_y/2, 3)
        center_z = round(dim_z/2, 3)

        # Convert the values found above to string with 8 chars (complete with spaces to the left) as the .pdb file model
        min_x = " "*(8-len(str(min_x))) + str(min_x)
        max_x = " "*(8-len(str(max_x))) + str(max_x)
        min_y = " "*(8-len(str(min_y))) + str(min_y)
        max_y = " "*(8-len(str(max_y))) + str(max_y)
        min_z = " "*(8-len(str(min_z))) + str(min_z)
        max_z = " "*(8-len(str(max_z))) + str(max_z)

        dim_x = " "*(8-len(str(dim_x))) + str(dim_x)
        dim_y = " "*(8-len(str(dim_y))) + str(dim_y)
        dim_z = " "*(8-len(str(dim_z))) + str(dim_z)

        center_x = " "*(8-len(str(center_x))) + str(center_x)
        center_y = " "*(8-len(str(center_y))) + str(center_y)
        center_z = " "*(8-len(str(center_z))) + str(center_z)

        # Write out the box file (following the one given in the DUD-E database)
        with open(f'{fout}{folder}/box{label_unique}{suffix}.pdb', 'w') as f:
            f.write(f"HEADER    CORNERS OF BOX      {min_x}{min_y}{min_z}{min_y}{max_y}{max_z}\n")
            f.write(f"REMARK    CENTER (X Y Z)      {center_x}{center_y}{center_z}\n")
            f.write(f"REMARK    DIMENSIONS (X Y Z)  {dim_x}{dim_y}{dim_z}\n")
            f.write(f"ATOM      1  DUA BOX     1    {min_x}{min_y}{min_z}\n")
            f.write(f"ATOM      2  DUB BOX     1    {max_x}{min_y}{min_z}\n")
            f.write(f"ATOM      3  DUC BOX     1    {max_x}{min_y}{max_z}\n")
            f.write(f"ATOM      4  DUD BOX     1    {min_x}{min_y}{max_z}\n")
            f.write(f"ATOM      5  DUE BOX     1    {min_x}{max_y}{min_z}\n")
            f.write(f"ATOM      6  DUF BOX     1    {max_x}{max_y}{min_z}\n")
            f.write(f"ATOM      7  DUG BOX     1    {max_x}{max_y}{max_z}\n")
            f.write(f"ATOM      8  DUH BOX     1    {min_x}{max_y}{max_z}\n")
            f.write("CONECT    1    2    4    5\n")
            f.write("CONECT    2    1    3    6\n")
            f.write("CONECT    3    2    4    7\n")
            f.write("CONECT    4    1    3    8\n")
            f.write("CONECT    5    1    6    8\n")
            f.write("CONECT    6    2    5    7\n")
            f.write("CONECT    7    3    6    8\n")
            f.write("CONECT    8    4    5    7\n")

def run_prank(filein, outpath, algorithms={"AffinityPropagation": False, "AgglomerativeClustering": True, "Birch": False, "DBSCAN": False, "KMeans": False, "MeanShift": False, "MiniBatchKMeans": False, "OPTICS": False, "SpectralClustering": False}, prank = "", threads = 1, coordSystem = "cartesian", spacing = 3.0, maxCutoff = 0.5, verbose=False, debug=False):
    '''
    Function to run p2rank and process its results, converting to a box space to be used in Vina
    Input:
     filein       [string]                    - Input pdb file
     outpath      [string]                    - Output dir (a new folder will be created)
     algorithms   [dict[string] bool]
                    DEFAULT: {
                                "AffinityPropagation": False,
                                "AgglomerativeClustering": True,
                                "Birch": False,
                                "DBSCAN": False,
                                "KMeans": False,
                                "MeanShift": False,
                                "MiniBatchKMeans": False,
                                "OPTICS": False,
                                "SpectralClustering": False"
                             }                - Dictionary of trues and falses of each implemented algorithm. The options are: AffinityPropagation, AgglomerativeClustering, Birch, DBSCAN, KMeans, MeanShift, MiniBatchKMeans, OPTICS, SpectralClustering
     prank        [string]  DEFAULT: ""          - p2rank file
     threads      [int]     DEFAULT: 1           - Number of threads that the p2rank should use
     coordSystem  [string]  DEFAULT: "cartesian" - The coordinate system to be used. The options are cartesian, polar, spherical
     spacing      [float]   DEFAULT: 3.0         - Expansion size of the box in angstroms
     maxCutoff    [float]   DEFAULT: 0.5         - If the probability value from p2rank is above this value, the pocket WILL be considered as valid, even if its value is below the cutoff (use 1.0 to disable this feature)
     verbose      [bool]    DEFAULT: False       - Verbose mode on/off
     debug        [bool]    DEFAULT: False       - Debug on/off
    Return:
        Nothing
    '''

    # If the prank variable is set
    if prank:
        # If the verbose mode is on
        if verbose:
            # Show the command
            print(f"P2Rank execution command: {' '.join([prank, 'predict','-threads', str(threads),  '-f', filein, '-o', outpath])}")
        # Execute the P2Rank
        subprocess.run([prank, 'predict','-threads', str(threads),  '-f', filein, '-o', outpath], stdout=subprocess.DEVNULL)

    # Get the input file name (which will be used to read the output from P2Rank)
    fname = os.path.basename(os.path.splitext(filein)[0])

    # Read the output
    data = pd.read_csv(f"{outpath}/{fname}.pdb_predictions.csv")

    # Remove spaces from the column names
    data.columns = data.columns.str.replace(' ', '')

    # Initialize the atom/probabilities list
    preatoms = []

    # For each line in the surf_atom_ids column
    for index, row in data.iterrows():
        # Split the elements using space and strip each element
        innerAtoms = [s.strip() for s in row['surf_atom_ids'].split()]

        # Add them to the preatoms list with the probability and the rank relative to the atom
        preatoms += list(((innerAtom, row['probability'], row['rank']) for innerAtom in innerAtoms))

    # Create two empty numpy arrays (one will be used to input to the clustering algorithms and the other will be passed to the analysis. Don't worry, the order of the array elements is the same in both!)
    coordinates = np.empty((0,4), float)
    coordinatesFull = np.empty((0,5), float)

    # Initialize the statistics list
    statistics = []

    # Initialize the atoms, probabilities and rank (to ensure that the data is in the same order)
    atoms = [i[0] for i in preatoms]
    probabilities = [i[1] for i in preatoms]
    rank = [i[2] for i in preatoms]

    # Read the .pdb file to capture the x/y/z coordinates
    with open(filein, 'r') as f:
        # For each line in the file
        for line in f:
            # If the atom ID is in the atom list
            if line[7:11].strip() in atoms:
                # Finds if the atom index is in the atom list (again to ensure that the right probability is assigned to the right atom)
                idx = atoms.index(line[7:11].strip())

                # Check and convert (if needed) the coordinates cartesian/polar/spherical
                if coordSystem.lower() == "cartesian": # if is cartesian, just read the values
                    v1 = line[31:38]
                    v2 = line[39:46]
                    v3 = line[47:54]
                elif coordSystem.lower() == "polar": # if is polar, convert x and y, but keep z
                    v1, v2 = __cart2pol(line[31:38], line[39:46])
                    v3 = line[47:54]
                elif coordSystem.lower() == "spherical": # if is spherical, convert x, y and z
                    v1, v2, v3 = __cart2sph(line[31:38], line[39:46], line[47:54])
                else: # if the user has typed something wrong, show a warning message and use cartesian
                    print("WARNING: Unknown, coordinate system, using cartesian!")
                    v1 = line[31:38]
                    v2 = line[39:46]
                    v3 = line[47:54]

                # Add the data to the numpy array as a list containing the coordinates + extra data  [X, Y, Z]/[therta, rho, z]/[az, el, r]
                coordinates = np.append(coordinates, np.array([[v1, v2, v3, rank[idx]]], float), axis=0)
                coordinatesFull = np.append(coordinatesFull, np.array([[v1, v2, v3, probabilities[idx], rank[idx]]], float), axis=0)
                #coordinates = np.append(coordinates, np.array([[line[31:38], line[39:46], line[47:54], rank[idx]]], float), axis=0)
                #coordinatesFull = np.append(coordinatesFull, np.array([[line[31:38], line[39:46], line[47:54], probabilities[idx], rank[idx]]], float), axis=0)

    ############################################################################
    # Now the code will have the samme pattern:                                #
    ############################################################################
    # 1) Print the algorithm name                                              #
    # 2) Start a timer                                                         #
    # 3) Execute the algoritm                                                  #
    # 4) Check the execution time                                              #
    # 5) Process the output (all files have the same final processing)         #
    # 6) Check the total execution time (algoeithm + file processing)          #
    ############################################################################

    # Affinity Propagation
    if algorithms["AffinityPropagation"]:
        from sklearn.cluster import AffinityPropagation

        print("Running Affinity Propagation")
        start_time = time.time()
        clustering = AffinityPropagation(random_state=0).fit(coordinates)

        if debug:
            with open(f"{outpath}/statistics.txt", "a") as f:
                f.write(f"ap\t{round(time.time() - start_time, 2)}\n")
        if verbose:
            print(f"Tempo de execução do Affinity Propagation sozinho: {round(time.time() - start_time, 2)} segundos.")

        __process_cluster(clustering, coordinatesFull, outpath, suffix = "ap", coordSystem = coordSystem, spacing = spacing, maxCutoff = maxCutoff)

        if debug:
            with open(f"{outpath}/statistics.txt", "a") as f:
                f.write(f"ap+pa\t{round(time.time() - start_time, 2)}\n")
        if verbose:
            print(f"Tempo de execução do Affinity Propagation + processamento de arquivos: {round(time.time() - start_time, 2)} segundos.\n")

    # Agglomerative clustering
    if algorithms["AgglomerativeClustering"]:
        from sklearn.cluster import AgglomerativeClustering

        print("Running Agglomerative Clustering")
        start_time = time.time()
        clustering = AgglomerativeClustering().fit(coordinates)

        if debug:
            with open(f"{outpath}/statistics.txt", "a") as f:
                f.write(f"ac\t{round(time.time() - start_time, 2)}\n")
        if verbose:
            print(f"Tempo de execução do Agglomerative Clustering sozinho: {round(time.time() - start_time, 2)} segundos.")

        __process_cluster(clustering, coordinatesFull, outpath, suffix = "ac", coordSystem = coordSystem, spacing = spacing, maxCutoff = maxCutoff)

        if debug:
            with open(f"{outpath}/statistics.txt", "a") as f:
                f.write(f"ac+pa\t{round(time.time() - start_time, 2)}\n")
        if verbose:
            print(f"Tempo de execução do Agglomerative Clustering + processamento de arquivos: {round(time.time() - start_time, 2)} segundos.\n")

    # Birch
    if algorithms["Birch"]:
        from sklearn.cluster import Birch

        print("Running Birch")
        start_time = time.time()
        clustering = Birch().fit(coordinates)

        if debug:
            with open(f"{outpath}/statistics.txt", "a") as f:
                f.write(f"bi\t{round(time.time() - start_time, 2)}\n")
        if verbose:
            print(f"Tempo de execução do Birch sozinho: {round(time.time() - start_time, 2)} segundos.")

        __process_cluster(clustering, coordinatesFull, outpath, suffix = "bi", coordSystem = coordSystem, spacing = spacing, maxCutoff = maxCutoff)

        if debug:
            with open(f"{outpath}/statistics.txt", "a") as f:
                f.write(f"bi+pa\t{round(time.time() - start_time, 2)}\n")
        if verbose:
            print(f"Tempo de execução do Birch + processamento de arquivos: {round(time.time() - start_time, 2)} segundos.\n")

    # DBSCAN
    if algorithms["DBSCAN"]:
        from sklearn.cluster import DBSCAN

        print("Running DBSCAN")
        start_time = time.time()
        clustering = DBSCAN(eps=5, min_samples=5).fit(coordinates)

        if debug:
            with open(f"{outpath}/statistics.txt", "a") as f:
                f.write(f"db\t{round(time.time() - start_time, 2)}\n")
        if verbose:
            print(f"Tempo de execução do DBSCAN sozinho: {round(time.time() - start_time, 2)} segundos.")

        __process_cluster(clustering, coordinatesFull, outpath, suffix = "db", coordSystem = coordSystem, spacing = spacing, maxCutoff = maxCutoff)

        if debug:
            with open(f"{outpath}/statistics.txt", "a") as f:
                f.write(f"db+pa\t{round(time.time() - start_time, 2)}\n")
        if verbose:
            print(f"Tempo de execução do DBSCAN + processamento de arquivos: {round(time.time() - start_time, 2)} segundos.\n")

    # KMeans
    if algorithms["KMeans"]:
        from sklearn.cluster import KMeans

        print("Running KMeans")
        start_time = time.time()
        clustering = KMeans(n_clusters=2, random_state=0).fit(coordinates)

        if debug:
            with open(f"{outpath}/statistics.txt", "a") as f:
                f.write(f"km\t{round(time.time() - start_time, 2)}\n")
        if verbose:
            print(f"Tempo de execução do KMeans sozinho: {round(time.time() - start_time, 2)} segundos.")

        __process_cluster(clustering, coordinatesFull, outpath, suffix = "km", coordSystem = coordSystem, spacing = spacing, maxCutoff = maxCutoff)

        if debug:
            with open(f"{outpath}/statistics.txt", "a") as f:
                f.write(f"km+pa\t{round(time.time() - start_time, 2)}\n")
        if verbose:
            print(f"Tempo de execução do KMeans + processamento de arquivos: {round(time.time() - start_time, 2)} segundos.\n")

    # Meanshift
    if algorithms["MeanShift"]:
        from sklearn.cluster import MeanShift, estimate_bandwidth

        print("Running Mean Shift")
        start_time = time.time()
        bandwidth = estimate_bandwidth(coordinates, quantile=0.2, n_samples=len(coordinates))
        clustering = MeanShift(bandwidth=bandwidth).fit(coordinates)

        if debug:
            with open(f"{outpath}/statistics.txt", "a") as f:
                f.write(f"ms\t{round(time.time() - start_time, 2)}\n")
        if verbose:
            print(f"Tempo de execução do Mean Shift sozinho: {round(time.time() - start_time, 2)} segundos.")

        __process_cluster(clustering, coordinatesFull, outpath, suffix = "ms", coordSystem = coordSystem, spacing = spacing, maxCutoff = maxCutoff)

        if debug:
            with open(f"{outpath}/statistics.txt", "a") as f:
                f.write(f"ms+pa\t{round(time.time() - start_time, 2)}\n")
        if verbose:
            print(f"Tempo de execução do Mean Shift + processamento de arquivos: {round(time.time() - start_time, 2)} segundos.\n")

    # Mini Batch KMeans
    if algorithms["MiniBatchKMeans"]:
        from sklearn.cluster import MiniBatchKMeans

        print("Running Mini Batch KMeans")
        start_time = time.time()
        clustering = MiniBatchKMeans(n_clusters=2).fit(coordinates)

        if debug:
            with open(f"{outpath}/statistics.txt", "a") as f:
                f.write(f"mb\t{round(time.time() - start_time, 2)}\n")
        if verbose:
            print(f"Tempo de execução do Mini Batch KMeans sozinho: {round(time.time() - start_time, 2)} segundos.")

        __process_cluster(clustering, coordinatesFull, outpath, suffix = "mb", coordSystem = coordSystem, spacing = spacing, maxCutoff = maxCutoff)

        if debug:
            with open(f"{outpath}/statistics.txt", "a") as f:
                f.write(f"mb+pa\t{round(time.time() - start_time, 2)}\n")
        if verbose:
            print(f"Tempo de execução do Mini Batch KMeans + processamento de arquivos: {round(time.time() - start_time, 2)} segundos.\n")

    # OPTICS
    if algorithms["OPTICS"]:
        from sklearn.cluster import OPTICS

        print("Running OPTICS")
        start_time = time.time()
        clustering = OPTICS(min_samples=5).fit(coordinates)

        if debug:
            with open(f"{outpath}/statistics.txt", "a") as f:
                f.write(f"op\t{round(time.time() - start_time, 2)}\n")
        if verbose:
            print(f"Tempo de execução do OPTICS sozinho: {round(time.time() - start_time, 2)} segundos.")

        __process_cluster(clustering, coordinatesFull, outpath, suffix = "op", coordSystem = coordSystem, spacing = spacing, maxCutoff = maxCutoff)

        if debug:
            with open(f"{outpath}/statistics.txt", "a") as f:
                f.write(f"op+pa\t{round(time.time() - start_time, 2)}\n")
        if verbose:
            print(f"Tempo de execução do OPTICS + processamento de arquivos: {round(time.time() - start_time, 2)} segundos.\n")

    # Spectral Clustering
    if algorithms["SpectralClustering"]:
        from sklearn.cluster import SpectralClustering

        print("Running Spectral Clustering")
        start_time = time.time()
        clustering = SpectralClustering(n_clusters=2, random_state=0).fit(coordinates)

        if debug:
            with open(f"{outpath}/statistics.txt", "a") as f:
                f.write(f"sc\t{round(time.time() - start_time, 2)}\n")
        if verbose:
            print(f"Tempo de execução do Spectral Clustering sozinho: {round(time.time() - start_time, 2)} segundos.")

        __process_cluster(clustering, coordinatesFull, outpath, suffix = "sc", coordSystem = coordSystem, spacing = spacing, maxCutoff = maxCutoff)

        if debug:
            with open(f"{outpath}/statistics.txt", "a") as f:
                f.write(f"sc+pa\t{round(time.time() - start_time, 2)}\n")
        if verbose:
            print(f"Tempo de execução do Spectral Clustering + processamento de arquivos: {round(time.time() - start_time, 2)} segundos.\n")

# Execute the script
if __name__ == "__main__":
    # Variables to be manually adjusted to run the script from prompt
    prank = "/mnt/d/Documents/OCDocker/software/search/p2rank_2.3/prank"
    fname = "receptor"
    basePath = "/mnt/d/Documents/OCDocker/docking"
    fin = f"{basePath}/{fname}.pdb"
    fout = f"{basePath}/prank"
    threads = 8
    coordSystem = "cartesian"
    spacing = 3.0
    maxCutoff = 0.5
    debug = True
    verbose = True

    # Algorith list
    algorithms = {
        "AffinityPropagation": True,
        "AgglomerativeClustering": True,
        "Birch": True,
        "DBSCAN": True,
        "KMeans": True,
        "MeanShift": True,
        "MiniBatchKMeans": True,
        "OPTICS": True,
        "SpectralClustering": True
    }

    run_prank(fin, fout, algorithms = algorithms, prank = prank, threads = threads, coordSystem = coordSystem, spacing = spacing, maxCutoff = maxCutoff, verbose = verbose, debug = debug)

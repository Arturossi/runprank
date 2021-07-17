#!/usr/bin/python3

# Description
###############################################################################
'''
processDUDEz.py is a script to perform the benchmark using the DUDEz database
to help to find which clustering algorithm (and its parameters) should be used
in the future. This script runs the run_prank function from the runprank
library and plot its results for each protein and the entire database.

Created by: Artur Duque Rossi
Version: 0.2
'''

# gambiarra pra funcionar sem visual
import matplotlib
matplotlib.use('Agg')

import os
import shutil
import runprank
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

from glob import glob

# Algorithms to be analyzed
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

# Number of threads
threads = 11
# Number of execution times
execTimes = 15
# Debug mode
debug = True
# Verbose mode
verbose = False
# Dir where the DUDEz database is located
database = "/mnt/d/Documents/OCDocker/OCDocker/data/ocdb/DUDEZ"
# Output folder
foutBase = "./processDUDEz"
# Output dir inside the output folder to hold the results of all proteins
foutAll = f"{foutBase}/All_proteins"

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
            #exit(1)
            return 1
    except Exception as e:
        print(f"Error! Exception: {e}")
        exit(-1)
    return -2

def __print_graphic(data, name, fout, x="Time mean (s)", y="Algorithm", hue="Algorithm + File processing"):
    '''
    Function to print a graphic using the seaborn library
    Input:
     data    [pd.DrataFrame]                                  - Dataframe containing data to be ploted
     name    [string]                                         - Name of the graphic to be used in title and filename
     fout    [string]                                         - Path where the graphic should be put (DO NOT END THE STRING WITH A /)
     x       [string]  DEFAULT: "Time mean (s)"               - Dataframe column name to be used in the X axis
     y       [string]  DEFAULT: "Algorithm"                   - Dataframe column name to be used in the Y axis
     hue     [string]  DEFAULT: "Algorithm + File processing" - Dataframe column name to be used to separate the columns
    Return:
      Nothing
    '''
    # Set the style
    sns.set(style="darkgrid")
    # Create a figure object and set its size
    fig = plt.figure(figsize=(8, 8))
    # Create the barplot using passed args
    ax = sns.barplot(x=x, y=y, hue=hue, data=data)
    # Set the title of the graphic
    ax.set_title(name)
    # Fit the data into given space (to avoid the data to be cut)
    plt.tight_layout()
    # Save the plot
    plt.savefig(f"{fout}/{name.replace(' ', '_')}.png")
    # Close the fig
    plt.close('all')


# Get all dirs paths in the database
dirs = glob(f"{database}/*")

# List and Dict with algorithm acronyms and its relation to its name
acronyms = ("ap", "ac", "bi", "db", "km", "ms", "mb", "op", "sc")
acronymsDict = {"ap": "Affinity Propagation", "ac": "Agglomerative Clustering", "bi": "Birch", "db": "DBSCAN", "km": "KMeans", "ms": "Mean Shift", "mb": "Mini Batch KMeans", "op": "OPTICS", "sc": "SpectralClustering"}

# Create the dirs to hold the data
__safe_create_dir(foutBase)
__safe_create_dir(foutAll)

# For each directory in the database folder
for d in dirs:
    # Set the input file name path
    fin = f"{d}/rec.crg.pdb"
    # Find the protein name
    ptn = d.split("/")[-1]
    # Set the output path
    fout = f"{foutBase}/{ptn}"

    # Print informative data of the protein
    print(f"############################")
    print(f"\tProtein {ptn}")
    print(f"############################")

    # If the protein has already been processed (this saves time)
    if os.path.isfile(f"{fout}/statistics_{ptn}.csv"):
        # Skip to the next protein
        continue

    # Create the output dir
    __safe_create_dir(fout)

    # Copy the protein and its ligand to the output dir
    shutil.copyfile(fin, f"{fout}/rec.crg.pdb")
    shutil.copyfile(f"{d}/xtal-lig.pdb", f"{fout}/xtal-lig.pdb")

    # For each execution
    for i in range(execTimes):
        # Print informative data of the run
        print(f"\n\t   Run {i + 1}")
        print(f"~~~~~~~~~~~~~~~~~~~~~~~~~~~~\n")

        # If its the first run
        if i == 0:
            # Run prank
            prank = "/mnt/d/Documents/OCDocker/software/search/p2rank_2.3/prank"
        else:
            # Do not run to save time
            prank = ""

        # Run the run_prank function from runprank.py
        runprank.run_prank(fin, fout, algorithms, prank=prank, threads=threads, debug=debug, verbose=verbose)

        # If its not the last execution
        if i < (execTimes - 1):
            # For each acronym in list
            for acronym in acronyms:
                # Parameterize the output path
                toCheck = f"{fout}/{acronym}"
                # If the dir exists
                if os.path.isdir(toCheck):
                    # Delete it and its contents (write the file is differnt to overwrite, this is more to measure the performance)
                    shutil.rmtree(toCheck)

    # Create an empty dataframe to hold the statistics
    statistics = pd.DataFrame({'Algorithm': pd.Series([], dtype='string'), 'Algorithm + File processing': pd.Series([], dtype='string'), 'Time mean (s)': pd.Series([], dtype='float')})

    # Open the statistics file to read (this file is generated in the run_prank function)
    with open(f"{fout}/statistics.txt", "r") as f:
        # For each line in the file
        for line in f:
            # Append to the statistics dataframe
            statistics = statistics.append(pd.Series([acronymsDict[str(line[:2])], str("Yes") if "+" in line else str("No"), float(line.split("\t")[1].strip())], index = statistics.columns), ignore_index = True)

    # Write a csv file with statistics (will be used later to create the statistics of all proteins)
    statistics.to_csv(f"{fout}/statistics_{ptn}.csv", index = False)

    # Print the protein graphic
    __print_graphic(statistics, ptn, fout, x="Time mean (s)", y="Algorithm", hue="Algorithm + File processing")

    # Move the created graphic + statistics file to all proteins dir
    shutil.copyfile(f"{fout}/{ptn.replace(' ', '_')}.png", f"{foutAll}/{ptn}.png")
    shutil.copyfile(f"{fout}/statistics_{ptn}.csv", f"{foutAll}/statistics_{ptn}.csv")

    # Nothing special here, just to keep things organized
    print(f"\n")

# Recreate the statistics dataframe
statistics = pd.DataFrame({'Algorithm': pd.Series([], dtype='string'), 'Algorithm + File processing': pd.Series([], dtype='string'), 'Time mean (s)': pd.Series([], dtype='float')})

# For each csv inside the output script dir
for csv in glob(f"{foutAll}/*.csv"):
    # Concatenate the statistics dataframe with its data
    statistics = pd.concat([statistics, pd.read_csv(csv)])

# Print its results
__print_graphic(statistics, "All Proteins", foutAll, x="Time mean (s)", y="Algorithm", hue="Algorithm + File processing")

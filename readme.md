runprank/processDUDEz
=====================

Created by Artur Rossi (2021)

The runprank is a script which encapsulates the [P2Rank software](https://github.com/rdk/p2rank) and process its output, making use of Artificial Intelligence to cluster the data and then parse the results into a box, ready which data may be used by [Vina](http://vina.scripps.edu). It recieves a pdb file as an input.

The processDUDEz is a script to process data from the [DUDEz database](https://dudez.docking.org) and perform a benchmark of which clustering algorithm performs better.


## Libraries used

### runprank

- NumPy
- Matplotlib
- Pandas
- Sklearn

### processDUDEz

- NumPy
- Pandas
- Seaborn
- Matplotlib


## Citing

Unfortunately both scripts does not have a way to cite it... yet... But hopefully in the future it might!

Although...

If you use runprank, please cite [P2Rank](https://doi.org/10.1186/s13321-018-0285-8) in JChem about P2Rank pocket prediction tool (2018)
Krivak R., Hoksza D. *P2Rank: machine learning based tool for rapid and accurate prediction of ligand binding sites from protein structure.*

If you use the processDUDEz with the [DUDEz database](https://doi.org/10.1021/acs.jcim.0c00598) in JChem about the DUDEz database (2021)
Stein R.M., *et al*. *Property-Unmatched Decoys in Docking Benchmarks*


## Installation

```sh
git clone https://github.com/Arturossi/runprank

# If using pip
python3 -m pip install numpy matplotlib pandas seaborn scikit-learn

# If using conda
conda install numpy pandas seaborn
conda install -c conda-forge matplotlib scikit-learn
```


## Usage

Firstly put the runprank file in the same folder as your script and then call the run_prank function. Its input parameters are:
```sh
- filein [string] - Input pdb file
- outpath [string] - Output dir (a new folder will be created)
- algorithms [dict[string] bool]
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
  } - Dictionary of trues and falses of each implemented algorithm. The options are: AffinityPropagation, AgglomerativeClustering, Birch, DBSCAN, KMeans, MeanShift, MiniBatchKMeans, OPTICS, SpectralClustering
- prank [string]
  DEFAULT: "" - p2rank file
- threads [int]
  DEFAULT: 1 - Number of threads that the p2rank should use
- verbose [bool]
  DEFAULT: False - Verbose mode on/off
- debug [bool]
  DEFAULT: False - Debug on/off
```

Here a sample of the usage

```python
import runprank

# This command will NOT run prank in sample.pdb, just cluster its results, only with Agglomerative Clustering and then output the results in the ./output folder
run_prank("sample.pdb", "./output")

# This command will run prank in sample.pdb, and will cluster its results using all supported algorithms and will generate files with statistics data (this is useful to perform benchmarks) to the ./output folder
run_prank("sample.pdb", "./output", {"AffinityPropagation": True, "AgglomerativeClustering": True,"Birch": True, "DBSCAN": True, "KMeans": True, "MeanShift": True, "MiniBatchKMeans": True, "OPTICS": True, "SpectralClustering": True"}, "/path/to/prank", 1, False, True)
```


## Changelog

### runprank

* V 0.1
	* Initial release
	* Able to run p2rank, process the output and create a box file using various clustering algorithms

* V 0.2
	* Allowed to call the runprank function from importing the library from another python script
	* Added box expansion
	* Minor bug fixes and optimizations

* V 0.3
	* Added probability filter
	* Fixed a problem that caused a pocket to be split into more than one cluster, now a pocket should always be entirely inside the same cluster
	* Minor bug fixes and optimizations

### processDUDEz

* V 0.1
	* Initial release
	* Generation of the statistics graphics for each protein

* V 0.2
	* Generation of the statistics graphic for the ENTIRE dataset
	* Fixed a bug where the ligand were being overwrited by the receptor rather being copied to the destination folder
	* Minor bug fixes and optimizations


## License

Apache 2.0

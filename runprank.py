#!/usr/bin/python3

# Description
###############################################################################
'''
runprank.py is a script to run the software p2rank and then convert its output
to box coordinates to be used as input to docking software like Vina

Created by: Artur Duque Rossi
Version: 0.3
'''

import os
import time
import subprocess
import numpy as np
import pandas as pd

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

def __process_cluster(clustering, coordinates, fout, suffix = "", spacing = 3.0, maxCutoff = 0.5):
    '''
    Function to process the cluster object and print a box file
    Input:
     clustering   [cluster result object]     - SciKit clustering object resulted from any clustering function after fitting
     coordinates  [np.array(np.array(float))] - NumPy array of numpy arrays of 3 floats containg the X Y Z coordinates
     fout         [string]                    - The path to output box files
     suffix       [string] DEFAULT: ""        - The suffix to append to box files and to create containing folders
     spacing      [float]  DEFAULT: 3.0       - Expansion size of the box in angstroms
     maxCutoff    [float]  DEFAULT: 0.5       - If the probability value from p2rank is above this value, the pocket WILL be considered as valid, even if its value is below the cutoff
    Return:
        Nothing
    '''
    # Pega os labels de cada elemento
    labels = clustering.labels_

    # Descobre quais labels existem retirando as repetições
    labels_unique = np.unique(labels)

    # Cria um dataframe com apenas as coordenadas xyz
    clusteringdf = pd.DataFrame(coordinates,  columns=['x', 'y', 'z', 'probability', 'rank'])

    # Adiciona a coluna dos labels para cada átomo
    clusteringdf['label'] = labels

    if suffix:
        folder = f"/{suffix}"
        __safe_create_dir(f"{fout}{folder}")
        suffix = f"_{suffix}"
    else:
        folder = ""

    # Seta o cutoff como a média das probabilidades
    cutoff = clusteringdf['probability'].mean()

    # Faz o cutoff ser maior que o valor de cutoff mínimo
    cutoff = cutoff if cutoff < maxCutoff else maxCutoff
    cutoff
    # Para cada label único (depois de remover os labels repetidos)
    for label_unique in labels_unique:
        # Se pertencer ao grupo -1 é outlier ou se nenhum elemento do conjunto possuir uma probabilidade maior ou igual ao cutoff e o valor de probabilidade do p2rank seja inferior ao parâmetro safeProb, ignore
        if str(label_unique) == "-1" or not (clusteringdf[clusteringdf['label'] == label_unique]['probability'] >= cutoff).any():
            continue

        # Calcule o min/max das coordenadas x/y/z (arredondando com 3 casas decimais)
        min_x = round(clusteringdf[clusteringdf['label'] == label_unique]['x'].min() - spacing, 3)
        max_x = round(clusteringdf[clusteringdf['label'] == label_unique]['x'].max() + spacing, 3)
        min_y = round(clusteringdf[clusteringdf['label'] == label_unique]['y'].min() - spacing, 3)
        max_y = round(clusteringdf[clusteringdf['label'] == label_unique]['y'].max() + spacing, 3)
        min_z = round(clusteringdf[clusteringdf['label'] == label_unique]['z'].min() - spacing, 3)
        max_z = round(clusteringdf[clusteringdf['label'] == label_unique]['z'].max() + spacing, 3)

        # Calcula as dimensões da caixa nos 3 eixos e o centro (arredondando com 3 casas decimais)
        dim_x = round(abs(min_x)+abs(max_x), 3)
        dim_y = round(abs(min_y)+abs(max_y), 3)
        dim_z = round(abs(min_z)+abs(max_z), 3)
        center_x = round(dim_x/2, 3)
        center_y = round(dim_y/2, 3)
        center_z = round(dim_z/2, 3)

        # Padroniza o tamanho da string das coordenadas min/max, dimensões e o centro de acordo com o padrão do arquivo .pdb
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

        # Escreve o arquivo da caixa (igual ao exemplo que o DUD-E fornece)
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
        #print(f'{fout}{folder}/box{label_unique}{suffix}.pdb')

def run_prank(filein, outpath, algorithms={"AffinityPropagation": False, "AgglomerativeClustering": True, "Birch": False, "DBSCAN": False, "KMeans": False, "MeanShift": False, "MiniBatchKMeans": False, "OPTICS": False, "SpectralClustering": False}, prank="", threads = 1, verbose=False, debug=False):
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
     prank        [string]  DEFAULT: ""       - p2rank file
     threads      [int]     DEFAULT: 1        - Number of threads that the p2rank should use
     verbose      [bool]    DEFAULT: False    - Verbose mode on/off
     debug        [bool]    DEFAULT: False    - Debug on/off
    Return:
        Nothing
    '''

    # Rodando o prank
    if prank:
        # Debug do comando para rodar o prank
        if verbose:
            print(f"P2rank execution command: {' '.join([prank, 'predict','-threads', str(threads),  '-f', filein, '-o', outpath])}")
        subprocess.run([prank, 'predict','-threads', str(threads),  '-f', filein, '-o', outpath], stdout=subprocess.DEVNULL)

    # Descobre o nome do arquivo (que será usado para ler os resultados do prank)
    fname = os.path.basename(os.path.splitext(filein)[0])

    # Lê o resultado e remove os espaços
    data = pd.read_csv(f"{outpath}/{fname}.pdb_predictions.csv")
    data.columns = data.columns.str.replace(' ', '')

    # Inicializa lista de átomos e probabilidades
    preatoms = []

    # Para cada linha na coluna surf_atom_ids
    for index, row in data.iterrows():
        # Separe os elementos por espaços e limpe cada elemento
        innerAtoms = [s.strip() for s in row['surf_atom_ids'].split()]
        # Adicione-os à lista juntamente com o sua probabilidade relativa e o rank (pra evitar que átomos no mesmo pocket sejam separados)
        preatoms += list(((innerAtom, row['probability'], row['rank']) for innerAtom in innerAtoms))

    # Para cada linha na coluna surf_atom_ids
    #for index, row in data['surf_atom_ids'].iteritems():
        # Separe os elementos por espaços e limpe cada elemento, depois adicione à lista de átomos
    #    atoms += [s.strip() for s in row.split()]

    # Inicializa um vetor numpy de coordenadas vazia
    coordinates = np.empty((0,4), float)
    coordinatesFull = np.empty((0,5), float)

    # Coleta estatística
    statistics = []
    probabilities = []

    # Inicializa lista de átomos, probabilidades e rank (por causa da ordem)
    atoms = [i[0] for i in preatoms]
    probabilities = [i[1] for i in preatoms]
    rank = [i[2] for i in preatoms]

    # Leia o arquivo pdb para capturar as coordenadas xyz
    with open(filein, 'r') as f:
        # Para cada linha do arquivo
        for line in f:
            # Se o ID do átomo estiver dentro da lista de átomos
            if line[7:11].strip() in atoms:
                # Descobre o índice do átomo na lista de átomos (para juntar com a probabilidade certa)
                idx = atoms.index(line[7:11].strip())
                # Adicione o dado no vetor numpy em forma de lista contendo as 3 coordenadas +  [X, Y, Z]
                coordinates = np.append(coordinates, np.array([[line[31:38], line[39:46], line[47:54], rank[idx]]], float), axis=0)
                coordinatesFull = np.append(coordinatesFull, np.array([[line[31:38], line[39:46], line[47:54], probabilities[idx], rank[idx]]], float), axis=0)

    ################################################################################
    # Aqui abaixo todos os códigos terão o mesmo padrão:                           #
    ################################################################################
    # 1) Imprimir o nome do algoritmo a ser executado                              #
    # 2) Iniciar um cronometro                                                     #
    # 3) Executar o algoritmo                                                      #
    # 4) Checar o tempo de execução do algoritmo apenas                            #
    # 5) Processar a saída (todos possuem o mesmo processamento final)             #
    # 6) Checar o tempo total de execução do algoritmo + processamento de arquivo  #
    ################################################################################

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

        __process_cluster(clustering, coordinatesFull, outpath, "ap")

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

        __process_cluster(clustering, coordinatesFull, outpath, "ac")

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

        __process_cluster(clustering, coordinatesFull, outpath, "bi")

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

        __process_cluster(clustering, coordinatesFull, outpath, "db")

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

        __process_cluster(clustering, coordinatesFull, outpath, "km")

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

        __process_cluster(clustering, coordinatesFull, outpath, "ms")

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

        __process_cluster(clustering, coordinatesFull, outpath, "mb")

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

        __process_cluster(clustering, coordinatesFull, outpath, "op")

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

        __process_cluster(clustering, coordinatesFull, outpath, "sc")

        if debug:
            with open(f"{outpath}/statistics.txt", "a") as f:
                f.write(f"sc+pa\t{round(time.time() - start_time, 2)}\n")
        if verbose:
            print(f"Tempo de execução do Spectral Clustering + processamento de arquivos: {round(time.time() - start_time, 2)} segundos.\n")

# Executa o script
if __name__ == "__main__":
    # Variáveis gerais que devem ser ajustadas pra cada pessoa que executar
    prank = "/mnt/d/Documents/OCDocker/software/search/p2rank_2.3/prank"
    fname = "receptor"
    basePath = "/mnt/d/Documents/OCDocker/docking"
    fin = f"{basePath}/{fname}.pdb"
    fout = f"{basePath}/prank"
    threads = 8
    debug = True
    verbose = True

    # Lista de algoritmos
    algorithms = {
        "AffinityPropagation": True, # Ruim
        "AgglomerativeClustering": True, # MARAVILHOSO
        "Birch": True, # MARAVILHOSO
        "DBSCAN": True, # Meh, junta tudo
        "KMeans": True, # Ok
        "MeanShift": True, # Bom
        "MiniBatchKMeans": True, # Ok
        "OPTICS": True, # Ruim
        "SpectralClustering": True # Meio bizarro o resultado
    }

    run_prank(fin, fout, algorithms, prank, threads, debug, verbose)

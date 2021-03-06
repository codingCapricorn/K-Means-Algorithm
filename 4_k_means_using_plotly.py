# -*- coding: utf-8 -*-
"""4.K-Means_Using_Plotly.ipynb



# Implementation of K-Means algorithm from the scratch using Numpy, Pandas and Plotly ::::

Data set that is taken here consists of 788 data points and have 7 shapes that can be identified visually.
"""

import pandas as pd
import numpy as np
import plotly.offline as plt
import plotly.graph_objs as go

pd.set_option('display.max_columns', 10)


def k_means_clustering(path, k):
    data = pd.read_csv(path)
    data = data[['V1', 'V2']]
    k_means = (data.sample(k, replace=False))
    k_means2 = pd.DataFrame()
    clusters = pd.DataFrame()
    print('Initial means:\n', k_means)

    while not k_means2.equals(k_means):

        # distance matrix
        cluster_count = 0
        for idx, k_mean in k_means.iterrows():
            clusters[cluster_count] = (data[k_means.columns] - np.array(k_mean)).pow(2).sum(1).pow(0.5)
            cluster_count += 1

        # update cluster
        data['MDCluster'] = clusters.idxmin(axis=1)

        # store previous cluster
        k_means2 = k_means
        k_means = pd.DataFrame()
        k_means_frame = data.groupby('MDCluster').agg(np.mean)

        k_means[k_means_frame.columns] = k_means_frame[k_means_frame.columns]

        print(k_means.equals(k_means2))

    # plotting
    print('Plotting...')
    data_graph = [go.Scatter(
        x=data['V1'],
        y=data['V2'].where(data['MDCluster'] == c),
        mode='markers',
        name='Cluster: ' + str(c)
    ) for c in range(k)]

    data_graph.append(
        go.Scatter(
            x=k_means['V1'],
            y=k_means['V2'],
            mode='markers',
            marker=dict(
                size=10,
                color='#000000',
            ),
            name='Centroids of Clusters'
        )
    )

    plt.plot(data_graph)
    print('Check out -->> temp-plot.html....')
    

if __name__ == '__main__':
    k_means_clustering(path='/content/k_means_clustering_test_1.csv', k=7)

    
    
"""

###1.Check-out "temp-plot.html" once for "k_means_clustering_test_1.csv" file first. 

###2.Repeat the same for "k_means_clustering_test_2.csv"

"""


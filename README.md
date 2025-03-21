# GENLINK

## Tutorial

Tutorial on how to use our code is available on Google Colab [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/alexeyshmelev/GENLINK/blob/main/notebooks/genlink_tutorial.ipynb)


## Downstream tasks

In progress

## Supported list of models

```
{
    "gnn":{"GL_GATConv_3l_512h":["graph_based","one_hot"],
           "GL_GATConv_3l_128h":["one_hot", "graph_based"],
           "GL_GATConv_9l_512h":["one_hot", "graph_based"],
           "GL_GATConv_9l_128h":["one_hot", "graph_based"],
           "GL_GINConv_3l_512h":["one_hot", "graph_based"],
           "GL_GINConv_3l_128h":["one_hot", "graph_based"],
           "GL_GINConv_9l_512h":["one_hot", "graph_based"],
           "GL_GINConv_9l_128h":["one_hot", "graph_based"],
           "GL_TAGConv_3l_512h_w_k3":["one_hot", "graph_based"],
           "GL_TAGConv_3l_128h_w_k3":["one_hot", "graph_based"],
           "GL_TAGConv_9l_128h_w_k3":["one_hot", "graph_based"],
           "GL_TAGConv_9l_512h_w_k3":["one_hot", "graph_based"],
           "GL_MLP_3l_512h":["graph_based"],
           "GL_MLP_3l_128h":["graph_based"],
           "GL_MLP_9l_512h":["graph_based"],
           "GL_MLP_9l_128h":["graph_based"],
           "GL_TAGConv_3l_512h_w_k3_gnorm":["one_hot", "graph_based"],
           "GL_TAGConv_3l_512h_w_k3_gnorm_gelu":["one_hot", "graph_based"],
           "GL_TAGConv_3l_512h_w_k3_gnorm_leaky_relu":["one_hot", "graph_based"],
           "GL_TAGConv_3l_512h_w_k3_gnorm_relu":["one_hot", "graph_based"],
           "GL_TAGConv_3l_512h_nw_k3_gnorm":["one_hot", "graph_based"],
           "GL_TAGConv_3l_512h_nw_k3_gnorm_gelu":["one_hot", "graph_based"],
           "GL_TAGConv_3l_512h_nw_k3_gnorm_leaky_relu":["one_hot", "graph_based"],
           "GL_TAGConv_3l_512h_nw_k3_gnorm_relu":["one_hot", "graph_based"],
           "GL_GCNConv_3l_512h_w":["one_hot", "graph_based"],
           "GL_GCNConv_3l_128h_w":["one_hot", "graph_based"],
           "GL_GCNConv_9l_512h_w":["one_hot", "graph_based"],
           "GL_GCNConv_9l_128h_w":["one_hot", "graph_based"],
           "GL_SAGEConv_3l_512h":["one_hot", "graph_based"],
           "GL_SAGEConv_3l_128h":["one_hot", "graph_based"],
           "GL_SAGEConv_9l_512h":["one_hot", "graph_based"],
           "GL_SAGEConv_9l_128h":["one_hot", "graph_based"]},
    "heuristic":["MaxEdgeCount", "MaxEdgeCountPerClassSize", "MaxSegmentCount", "LongestIbd", "MaxIbdSum", "MaxIbdSumPerClassSize"],
    "community_detection":["LabelPropagation", "AgglomerativeClustering", "SpectralClustering", "RelationalNeighborClassifier"]
}

```
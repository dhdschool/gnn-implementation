### Dataset 
Our data consists of data gathered from wikipedia, provided by the karateclub library. In this dataset, pages are represented by vertices, mutual links are represented by edges, with some data including user traffic associated with nodes. 

Formally, this is the graph $G(V, E),\: V = \{v_i: i \in 1,2,...,n\},\; E = \{(v_i, v_j): v_i,\, v_j \in V,\, v_i \neq v_j\}$, a vector $t \in \mathbb{Z}_2^n$ with $t_i$ being the whether a page refers primarily to crocodiles at vertice $v_i$, and the binary feature matrix $S \in \mathbb{Z}_2^{m \times n}$ with a number of nouns in $w \in \mathbb{Z}_2^m$, with a 1 in $s_{ij}$ if noun $w_j$ is present in vertice $v_i$ and a 0 otherwise. 

### Problem
We wish to predict whether or not a wikipedia page is primarily refering to crocodiles given its position in the graph and the presence noun features in each page. In order to do this, the nodes will be embedded into $\mathbb{R}^k$ using an embedding neural network, then these nodes will undergo multiple equivariant graph neural network layers, then finally be output into a single neuron layer as a predicition.

### Sampling
In order to test the accuracy of our model, it is critical to create a method that provides accurate samples of the entire dataset. The method that we will implement is known as Snow Ball Sampling (SBS). Given a random sample of $k_0$ vertices from $V_0$, we take up to $k$ neighboring verticies $\forall v \in V_0$, then repeat this process $t$ times, with all vertices sampled at step $i$ placed into set $V_i$. One such larger sample will be taken as the training set, and a smaller sample taken from the remaining graph acts as the testing set. 

### Embedding
In order to take advantage of machine learning tools provided by libraries such as PyTorch, the native graph data much be embedded into $\mathbb{R}^k$ in a way that is equivariant with respect to adjacency matrix permutation as well as with respect to embedded vector translation, rotation, and reflection; that is, permuting the order of the input nodes should result in an equivalent permutation of the output vectors and different embeddings of the same graph should affect output similarly (but not exactly as modern embedding methods attempt to make a "good enough" approximation of actual permutation operations).

In order to achieve this first embedding step, we will use Multi-scale Attributed Node Embedding (MUSAE) created by Rozemberczki, Allen, and Sarkar [here](https://arxiv.org/abs/1909.13021). This method takes random walks over a graph and attempts to minimize loss between neighboring nodes over both the feature and spatial dimensions using a graph machine learning algorithm called Skipgram Negative Sampling (SGNS). 

### Predicting
Once the data is embedded from the vertice-space into a vector-space, we will learn features from the graph using two equivariant graph convolutional layers, then output will be pooled into a single linear neuron with the softmax activation function. These graph convolution layers create new node embedding steps by taking the mean of a node with its neighbors, then learns values through a typical affine-nonlinear function composition. This approach allows graphs that have different structures but the same information to give a "similar enough" output.
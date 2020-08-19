# Graph neural networks for community detection (GNN4CD)
Graph Neural Networks and Line Graph Neural Networks for community detection in graphs, as described in the paper **_Supervised community detection with graph neural networks_** by Zhengdao Chen, Lisha Li and Joan Bruna, which appeared in ICLR 2019. 

Here is [the latest version of the paper](https://arxiv.org/pdf/1705.08415.pdf) on ArXiv. 

The implementation is in Python (3.6) with Pytorch (0.3.1), and partially adapted from https://github.com/alexnowakvila/QAP_pt.

Running [_script_5SBM_gnn.sh_](https://github.com/zhengdao-chen/GNN4CD/blob/master/src/script_5SBM_gnn.sh) and [_script_5SBM_lgnn.sh_](https://github.com/zhengdao-chen/GNN4CD/blob/master/src/script_5SBM_lgnn.sh) will perform the experiments of community detection for 5-class dissortive stochastic block models using a graph neural network (GNN) and a line graph neural network (LGNN), respectively.

# Graph neural networks for community detection (GNN4CD)
Graph Neural Networks and Line Graph Neural Networks for community detection in graphs, as described in the paper [**_Supervised community detection with graph neural networks_**](https://arxiv.org/pdf/1705.08415.pdf) by Zhengdao Chen, Lisha Li and Joan Bruna, which appeared in ICLR 2019. 

The implementation is in _Python (3.6)_ with _Pytorch (0.3.1)_, and partially adapted from the code [here](https://github.com/alexnowakvila/QAP_pt). The latest version using _Python 3.7_ and _PyTorch 1.4.0_ can be found in the main branch.

Running [_script_5SBM_gnn.sh_](https://github.com/zhengdao-chen/GNN4CD/blob/master/src/script_5SBM_gnn.sh) and [_script_5SBM_lgnn.sh_](https://github.com/zhengdao-chen/GNN4CD/blob/master/src/script_5SBM_lgnn.sh) will perform the experiments of community detection for 5-class dissortive stochastic block models using a graph neural network (GNN) and a line graph neural network (LGNN), respectively.

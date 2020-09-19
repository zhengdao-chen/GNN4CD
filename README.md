# Graph neural networks for community detection (GNN4CD)
Graph Neural Networks and Line Graph Neural Networks for community detection in graphs, as described in the paper [*_Supervised community detection with graph neural networks_*](https://arxiv.org/pdf/1705.08415.pdf) by Zhengdao Chen, Lisha Li and Joan Bruna, which appeared in ICLR 2019. 

The implementation in the main branch uses _Python (3.7)_ with _NumPy (1.18.1)_ and _PyTorch (1.4.0)_. A previou version compatible with _Python 3.6_ and _PyTorch 0.3.1_ can be found in the branch [pytorch_0.3.1](https://github.com/zhengdao-chen/GNN4CD/tree/pytorch_0.3.1). The code is partially adapted from the code [here](https://github.com/alexnowakvila/QAP_pt). 

Running [_script_5SBM_gnn.sh_](https://github.com/zhengdao-chen/GNN4CD/blob/master/src/script_5SBM_gnn.sh) and [_script_5SBM_lgnn.sh_](https://github.com/zhengdao-chen/GNN4CD/blob/master/src/script_5SBM_lgnn.sh) will perform the experiments of community detection for 5-class dissortive stochastic block models using a graph neural network (GNN) and a line graph neural network (LGNN), respectively.

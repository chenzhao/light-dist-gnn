## A lightweight distributed GNN library for full batch node property prediction. 


#### Features/Changelog 
- Complete refactoring of CAGNET. 
- Distributed utilities such as log, timer, etc. 
- Node feature cached training.
- Partitioned graph cache on disk.
- More datasets. Most large graphs from pyg, dgl, ogb supported.
- Training depends on pytorch only.
- Distributed GAT training.
- Latest pytorch version supported. 
- CSR graph supported.
- Half precision training supported.


#### Getting started

1. Setup a clean environment.
```
conda create --name gnn
conda activate gnn
```
2. Install pytorch (needed for training) and other libraries (needed for downloading datasets). 


```
// Cuda 10:
conda install pytorch torchvision torchaudio cudatoolkit=10.2 -c pytorch-lts
conda install -c dglteam dgl-cuda10.2
conda install pyg -c pyg -c conda-forge
pip install ogb
```

```
// Cuda 11:
conda install pytorch torchvision torchaudio cudatoolkit=11.1 -c pytorch-lts -c nvidia
conda install -c dglteam dgl-cuda11.1
conda install pyg -c pyg -c conda-forge
pip install ogb
```

3. Compile and install spmm. (Optional. CUDA dev environment needed.)
```
cd spmm_cpp
python setup.py install
```

4. Prepare datasets (edit the code according to your needs).
```
//This may take a while.
python prepare_data.py
```
5. Train.
```
python main.py
```


#### Experiments for Sancus: Staleness-Aware Communication-Avoiding Full-Graph Decentralized Training in Large-Scale Graph Neural Networks
1. Check the steps in **Getting started** .
2. Check dataset, epoch, and num of GPUs in main.py.
3. Check model settings in dist_train.py 
4. Check cache methods in models.
5. Run and see the result. 


#### Contact

Contact chenzhao@ust.hk for any problems.


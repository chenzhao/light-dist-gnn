
conda create --name gnn

conda activate gnn

conda install pytorch torchvision torchaudio cudatoolkit=10.2 -c pytorch-lts

conda install -c dglteam dgl-cuda10.2

conda install pyg -c pyg -c conda-forge




conda install pytorch torchvision torchaudio cudatoolkit=11.1 -c pytorch-lts -c nvidia
conda install -c dglteam dgl-cuda11.1
conda install pyg -c pyg -c conda-forge

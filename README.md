# HyMN
Balancing Efficiency and Expressiveness: Subgraph GNNs with Walk-Based Centrality

In this work we leverage walk-based centrality measures, both as a powerful form of SE and also as a subgraph selection strategy for Subgraph GNNs. For the code, we make use of the GraphGPS framework (https://github.com/rampasek/GraphGPS/tree/main). 


### Python environment setup with Conda

```bash
conda create -n hymn python=3.10
conda activate hymn

conda install pytorch=1.13 torchvision torchaudio pytorch-cuda=11.7 -c pytorch -c nvidia
conda install pyg=2.2 -c pyg -c conda-forge
pip install pyg-lib -f https://data.pyg.org/whl/torch-1.13.0+cu117.html
  
conda install openbabel fsspec rdkit -c conda-forge

pip install pytorch-lightning yacs torchmetrics
pip install performer-pytorch
pip install tensorboardX
pip install ogb
pip install wandb

conda clean --all
```


### Running HyMN
```bash
conda activate hymn

# Running HyMN (GIN, T=1) with CSE on Peptides-func
python main.py --cfg configs/peptides-func-gin.yaml dataset.node_encoder_name Atom+NodeCentrality model.type colour_gnn gnn.num_samples 2 gnn.layer_type gineconv wandb.use False

# Running HyMN (GIN, T=2) without CSE on Peptides-func.
python main.py --cfg configs/peptides-func-gin.yaml dataset.node_encoder_name Atom model.type colour_gnn gnn.num_samples 3 gnn.layer_type gineconv wandb.use False

```

## Running HyMN for counting substructures

```
cd counting_substructures
wandb sweep wandb_sweep/counting.yaml
wandb agent [...]
```



### W&B logging
To use W&B logging, set `wandb.use True` and change it to whatever else you like by setting `wandb.entity`).



## Citation

If you find this work useful, please cite our ArXiv paper:
```bibtex
@article{
}
```

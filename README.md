# Balancing Efficiency and Expressiveness: Subgraph GNNs with Walk-Based Centrality

In this work we leverage walk-based centrality measures, both as a powerful form of SE and also as a subgraph selection strategy for Subgraph GNNs. We dub our architecture "HyMN" (Hybrid Marking Network). For the code, we make use of the GraphGPS framework (https://github.com/rampasek/GraphGPS/tree/main). 


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
To reproduce the results on real-world benchmarks reported in our experimental section, please run `main.py` providing the desired yaml config file via the `cfg` argument (they are all found in folder `configs/`). For example:
```bash
conda activate hymn

# Running HyMN (GIN, T=2) with CSE on MolHiv
python main.py --cfg configs/molhiv_with_cse.yaml gnn.num_samples 3 wandb.use False

# Running HyMN (GIN, T=5) without CSE on MolHiv.
python main.py --cfg configs/molhiv_without_cse.yaml gnn.num_samples 6 wandb.use False

Note: Number of samples is one extra as equals original + T

```

## Running HyMN for counting substructures
Experiments on the synthetic substructure counting benchmarks are best run via a wandb sweep. The following commands will generate results for random subgraph sampling and sampling based on the Subgraph Centrality:
```
cd counting_substructures
wandb sweep wandb_sweep/counting.yaml
wandb agent [...]
```
Note: it may take a while to run the above.

## Perturbation analysis in paper
Jupyter Notebooks for the analysis in the paper on the amount of perturbation with node marking and how this is correlated with substructures can be found in the folder 'notebooks' (3.2 EFFECTIVE NODE MARKING).



### W&B logging
To use W&B logging, set `wandb.use True` and change it to whatever else you like by setting `wandb.entity`).



## Citation

If you find this work useful, please cite our ArXiv paper:
```bibtex
@article{
}
```

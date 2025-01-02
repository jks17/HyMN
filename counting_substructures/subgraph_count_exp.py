import argparse
import time
import os
import torch
import numpy as np
import random
import wandb

from constants import OUT_DIMS, IN_DIMS, SUBGRAPH_COUNT_TASKS as task_mapping, PATHS_MAX_LENGTH, MAX_SIZES

from torch.nn import functional as F
from tqdm import tqdm
from pathlib import Path
from counting_data import CountingSubstructures
from torch_geometric.loader import DataLoader

from instantiation_utils import get_model, get_painters

from utils import str2bool, check_colour_args
from transform import RandomSampling
from transform import SubgraphTransform

def task2idx(task):
    if task not in task_mapping:
        raise ValueError(f"Unknown substructure '{task}'")
    return task_mapping[task]


def train(train_loader, model, optimiser, loss_fn, device, painter=None):

    model.train()
    
    total_loss = 0.0
    num_graphs = 0
    for data in tqdm(train_loader):
        
        optimiser.zero_grad()

        # TODO: consider painting after moving to device for more efficiency?
        if painter is not None:
            data = painter.paint(data)
        data = data.to(device)
        preds = model(data)
        loss = loss_fn(preds, data.y)

        loss.backward()
        optimiser.step()

        total_loss += loss.item() * len(data.y)
        num_graphs += len(data.y)

    return total_loss / num_graphs


def evaluate(loader, model, metric_fn, device, painter=None):

    model.eval()

    y_pred, y_true = [], []
    for data in tqdm(loader):

        # TODO: consider painting after moving to device for more efficiency?
        if painter is not None:
            data = painter.paint(data)
        data = data.to(device)
        preds = model(data)

        y_pred.append(preds.detach().cpu())
        y_true.append(data.y.detach().cpu())

    y_pred = torch.cat(y_pred, dim=0)
    y_true = torch.cat(y_true, dim=0)

    return metric_fn(y_true, y_pred)


def evaluate_trivial(train_loader, val_loader, test_loader, metric_fn):

    y_true = []
    for data in tqdm(train_loader):
        y_true.append(data.y.detach().cpu())
    y_true = torch.cat(y_true, dim=0)
    mean = torch.mean(y_true, dim=0, keepdims=True)
    
    y_pred = torch.cat([mean for y in y_true], dim=0)
    train_perf = metric_fn(y_true, y_pred)

    y_true = []
    for data in tqdm(val_loader):
        y_true.append(data.y.detach().cpu())
    y_true = torch.cat(y_true, dim=0)
    y_pred = torch.cat([mean for y in y_true], dim=0)
    val_perf = metric_fn(y_true, y_pred)

    y_true = []
    for data in tqdm(test_loader):
        y_true.append(data.y.detach().cpu())
    y_true = torch.cat(y_true, dim=0)
    y_pred = torch.cat([mean for y in y_true], dim=0)
    test_perf = metric_fn(y_true, y_pred)

    return (train_perf, val_perf, test_perf)


def run_exp(args, device, wandb_run=None):

    if wandb_run is not None:
        exp_name = wandb_run.name
    else:
        # get a name based on timestamp
        exp_name = str(time.time())[-5:]

    # verbose
    msg = f"The current experiment has name {exp_name}\n"
    print(msg)

    # seeds
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    random.seed(args.seed)

    # retrieve dataset
    if args.colouring_scheme in ['indexrandom']:
        pretransform = RandomSampling()
        largest_component=True # whether to just looks at largest connected component
        name = 'random_sampling'
    elif args.colouring_scheme in ['indexsubgraph']:
        pretransform = SubgraphTransform()
        largest_component=True
        name = 'subgraph_sampling'

    dataset = CountingSubstructures(Path(args.data_dir), name, include_paths=('path' in args.task), parameters=args.gen_parameters,
        largest_component=largest_component, pre_transform=pretransform)
    # I have given a seperate folder as different colouring schemes may produce different pre-calculations on datasets
    dataset.data.y = dataset.data.y[:, args.task_idx:args.task_idx+1]
    split_idx = dataset.separate_data(args.seed)  # mind the seed is not actually used inside (split is given)

    # verbose
    msg = f"Dataset loaded with substructure '{args.task}' (idx: {args.task_idx}) as target.\n"
    print(msg)

    # startup data loaders
    if not args.overfit:
        train_loader = DataLoader(dataset[split_idx["train"]], batch_size=args.batch_size, num_workers=args.num_workers, shuffle=True)
        train_loader_eval = DataLoader(dataset[split_idx["train"]], batch_size=args.batch_size, num_workers=args.num_workers, shuffle=False)
        val_loader = DataLoader(dataset[split_idx["val"]], batch_size=args.batch_size, num_workers=args.num_workers, shuffle=False)
        test_loader = DataLoader(dataset[split_idx["test"]], batch_size=args.batch_size, num_workers=args.num_workers, shuffle=False)
    else:
        train_loader = DataLoader(dataset[split_idx["train"]][:32], batch_size=args.batch_size, num_workers=args.num_workers, shuffle=True)
        train_loader_eval = DataLoader(dataset[split_idx["train"]][:32], batch_size=args.batch_size, num_workers=args.num_workers, shuffle=False)
        val_loader = DataLoader(dataset[split_idx["train"]][:32], batch_size=args.batch_size, num_workers=args.num_workers, shuffle=False)
        test_loader = DataLoader(dataset[split_idx["train"]][:32], batch_size=args.batch_size, num_workers=args.num_workers, shuffle=False)

    # verbose
    msg = f"Data loaders instantiated with batch size {args.batch_size}, and {args.num_workers} workers.\n"
    print(msg)

    # instantiate painters
    train_painter, train_eval_painter, val_painter, test_painter = get_painters(args, device)

    # instantiate model
    model = get_model(args)
    model = model.to(device)
    if wandb_run is not None:
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        wandb.run.summary["Model Parameters"] = trainable_params
        wandb_run.watch(model, log_graph=True)

    # verbose
    msg = f"Model {model} instantiated.\n"
    print(msg)

    # instantiate optimiser and scheduler
    loss_fn = F.l1_loss
    metric_fn = F.l1_loss
    optimiser = torch.optim.Adam(model.parameters(), lr=args.learning_rate)
    scheduler = (
        torch.optim.lr_scheduler.StepLR(optimiser, step_size=args.decay_step, gamma=args.decay_rate)
        if args.scheduler
        else None)

    # verbose
    msg = f"The training procedure will utilise optimiser {optimiser} with lr scheduler {scheduler}.\n"
    print(msg)

    # verbose
    msg = f"Training starting now ...\n"
    print(msg)

    # training loop
    train_curve = []
    valid_curve = []
    test_curve = []
    for epoch in range(1, args.epochs + 1):

        # verbose
        msg = f"\nEpoch {epoch} ...\n"
        print(msg)

        train(train_loader, model, optimiser, loss_fn, device, painter=train_painter)

        if train_eval_painter is not None and args.reset_eval_painters:
            train_eval_painter.reset()
            val_painter.reset()
            test_painter.reset()
        train_perf = evaluate(train_loader_eval, model, metric_fn, device, painter=train_eval_painter)
        valid_perf = evaluate(val_loader, model, metric_fn, device, painter=val_painter)
        test_perf = evaluate(test_loader, model, metric_fn, device, painter=test_painter)

        if args.scheduler:
            scheduler.step()

        # logging
        train_curve.append(train_perf)
        valid_curve.append(valid_perf)
        test_curve.append(test_perf)
        if wandb_run is not None:
            wandb_run.log({
                f'Metric/train': train_perf,
                f'Metric/valid': valid_perf,
                f'Metric/test': test_perf})

        # verbose
        msg = f"Train metric:\t{train_perf}\n"
        msg += f"Val metric:\t{valid_perf}\n"
        msg += f"Test metric:\t{test_perf}\n"
        print(msg)

    # compute perf of trivial predictor
    triv_train_perf, triv_valid_perf, triv_test_perf = evaluate_trivial(train_loader, val_loader, test_loader, metric_fn)

    return exp_name, (train_curve, valid_curve, test_curve), (triv_train_perf, triv_valid_perf, triv_test_perf)


def main():

    parser = argparse.ArgumentParser(description='Subgraph counting synthetic benchmarking')

    # architectural (hyper)parameters
    parser.add_argument('--model', type=str,
                        help='type of model {GNN, ColourCatSharedGNN}')
    parser.add_argument('--layer', type=str,
                        help='type of convolution {gin, colourcat_gin, colourcat_shared_gin}')
    parser.add_argument('--num_layers', type=int,
                        help='number of message-passing layers')
    parser.add_argument('--emb_dim', type=int,
                        help='dimensionality of hidden units in GNNs')
    parser.add_argument('--jk', type=str2bool,
                        help='whether to apply jumping knowledge, strategy is concatenation')
    parser.add_argument('--residual', type=str2bool,
                        help='Whether to use residual connections')
    parser.add_argument('--readout', type=str,
                        help='readout strategy to use, e.g. sum or mean')
    parser.add_argument('--bn', type=str2bool,
                        help='whether to use batch normalisation in each mlp')
    parser.add_argument('--bn_between_convs', type=str2bool,
                        help='whether to use batch normalisation in between convolutional layers')
    parser.add_argument('--multiplier', type=int,
                        help='growth factor for hidden dimensions in mlps')
    parser.add_argument('--predictor', type=str,
                        help='how to perform the final predictions, e.g., id, linear, mlp')

    # colouring (hyper)parameters
    parser.add_argument('--averaging', type=str,
                        help='averaging strategy to use, e.g. sum or mean')
    parser.add_argument('--sample_aggregation', type=str, default='none',
                        help='strategy to use to aggregate samples in a DSS layer, e.g. sum or mean')
    parser.add_argument('--num_colours', type=int,
                        help='number of colours to use')
    parser.add_argument('--nodes_to_colour', type=int,
                        help='number of nodes to colour  (centrality)', default=1)
    parser.add_argument('--num_samples', type=int,
                        help='number of colouring samples to use for each graph')
    parser.add_argument('--colouring_scheme', type=str, default='uniform',
                        help='colouring distribution')
    parser.add_argument('--encoding_scheme', type=str, default='onehot',
                        help='how colours are encoded in tensor format')
    parser.add_argument('--inject_colours', type=str2bool,
                        help='whether to inject colours at every layer')
    parser.add_argument('--reset_eval_painters', type=str2bool, default='false',
                        help='whether to reset painters at evaluation time to ensure the same colouring \
                        for inference graphs; default: false')
    parser.add_argument('--colour_readout', type=str, default='none',
                        help='readout strategy to use for colour feats, e.g. sum or mean')
    parser.add_argument('--colour_jk', type=str2bool, default='false',
                        help='whether to apply jumping knowledge to colour embs, strategy is concatenation')
    parser.add_argument('--colour_picker_aggregator', type=str, default='none',
                        help='aggregation strategy when summarising colours, e.g. sum or mean')
    parser.add_argument('--colour_emb_dim', type=int, default=1,
                        help='embedding dimension for colour features, default 1')
    parser.add_argument('--picker_b4_readout', type=str2bool, default='false',
                        help='whether to apply colour picking before colour readout')
    parser.add_argument('--subsampling_ratio', type=float, default=0.0,
                        help='whether to adapt the number of samples in a way that depends on the number of nodes; if 0.0 no adaptive sampling is applied')

    # training parameters
    parser.add_argument('--batch_size', type=int,
                        help='input batch size for training')
    parser.add_argument('--learning_rate', type=float,
                        help='learning rate for training')
    parser.add_argument('--epochs', type=int, default=100,
                        help='number of epochs to train (default: 100)')
    parser.add_argument('--num_workers', type=int, default=0,
                        help='number of workers (default: 0)')
    parser.add_argument('--scheduler', type=str2bool,
                        help='whether to decay learning rate')
    parser.add_argument('--decay_rate', type=float, default=0.5,
                        help='decay rate for training (default: 0.5)')
    parser.add_argument('--decay_step', type=int, default=50,
                        help='decay step for training (default: 50)')

    # session parameters
    parser.add_argument('--device', type=int, default=0,
                        help='which gpu to use if any (default: 0)')
    parser.add_argument('--task', type=str,
                        help='substructure to count')
    parser.add_argument('--data_dir', type=str, default="./datasets/",
                        help='directory where to store the data (default: ./datasets/)')
    parser.add_argument('--gen_parameters', type=str, default="none",
                        help='parameters for the generation of synthetic RR graphs; if `none`, the default dataset is used')
    parser.add_argument('--seed', type=int, default=0,
                        help='random seed (default: 0)')
    parser.add_argument('--res_folder', type=str, default='./results/subgraph_count/',
                        help='folder to output result')
    parser.add_argument('--wandb', type=str2bool, default='true',
                        help='whether to log over wandb')
    parser.add_argument('--verbose_data_pipe', type=str2bool, default='false',
                        help='whether to use verbose data feeding approach')
    parser.add_argument('--overfit', type=str2bool, default='false',
                        help='whether to only try and overfit a small sample of the original dataset')

    args = parser.parse_args()
    #setattr(args, 'sha', get_git_revision_short_hash()) 
    setattr(args, 'task_idx', task2idx(args.task))
    setattr(args, 'out_dim', OUT_DIMS['subgraph_count'])
    setattr(args, 'in_dim', IN_DIMS['subgraph_count'])
    if args.gen_parameters == 'none':
        setattr(args, 'gen_parameters', None)
    setattr(args, 'max_num_nodes', MAX_SIZES[f'subgraph_count_{args.gen_parameters}'])
    if args.subsampling_ratio > 0.0:  # adapt num_samples to save memory
        setattr(args, 'num_samples', int(args.max_num_nodes * args.subsampling_ratio))
    dargs = vars(args)

    # verbose
    msg = "\nHey :)\n\nStarting a new subgraph counting experiment ...\n\n"
    msg += "====================== arguments ======================\n"
    for key in dargs:
        msg += f"{key}: {dargs[key]}\n"
    msg += "===============================================\n"
    print(msg)

    # check consistency of colouring arguments
    check_colour_args(args)

    # device
    device = torch.device("cuda:" + str(args.device)) if torch.cuda.is_available() else torch.device("cpu")

    # wandb
    wandb_run = wandb.init(config=args) if args.wandb else None

    # run experiment
    exp_name, curves, baseline_perf = run_exp(args, device, wandb_run=wandb_run)
    train_curve, valid_curve, test_curve = curves
    triv_train_perf, triv_valid_perf, triv_test_perf = baseline_perf

    # extract results
    best_val_epoch = np.argmin(valid_curve)
    best_train = min(train_curve)
    results = {
        'Metric/train': train_curve[best_val_epoch],
        'Metric/valid': valid_curve[best_val_epoch],
        'Metric/test': test_curve[best_val_epoch],
        'Metric/best_train': best_train,
        'Metric/trivial_train': triv_train_perf,
        'Metric/trivial_valid': triv_valid_perf,
        'Metric/trivial_test': triv_test_perf}

    # verbose
    msg = f"====================== final results ======================\n"
    msg += f"Train:\t\t{train_curve[best_val_epoch]}\n"
    msg += f"Val:\t\t{valid_curve[best_val_epoch]}\n"
    msg += f"Test:\t\t{test_curve[best_val_epoch]}\n"
    msg += f"Best train:\t{best_train}\n"
    msg += f"------------------- trivial predictor --------------------\n"
    msg += f"Train:\t\t{triv_train_perf}\n"
    msg += f"Val:\t\t{triv_valid_perf}\n"
    msg += f"Test:\t\t{triv_test_perf}\n"
    print(msg)

    # save results
    if args.wandb:
        for key in results:
            wandb_run.summary[key] = results[key]
    else:
        outdir = Path(f"{args.res_folder}")
        outdir.mkdir(parents=True, exist_ok=True)
        with open(os.path.join(outdir, exp_name+'.txt'), 'w') as handle:
            for key in results:
                msg = f"{key}: {results[key]} \n"
                handle.write(msg)
            handle.write("===============================================\n")
            for key in dargs:
                msg = f"{key}: {dargs[key]} \n"
                handle.write(msg)

if __name__ == "__main__":
    main()

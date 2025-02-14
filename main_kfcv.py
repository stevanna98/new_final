import torch
import numpy as np
import lightning as L
import warnings
import json

from argparse import ArgumentParser
from torch.utils.data import Subset
from torch_geometric.data import DataLoader as torch_g_dataloader
from torch.utils.data import DataLoader as torch_dataloader
from lightning.pytorch.callbacks import LearningRateMonitor
from lightning.pytorch.loggers import TensorBoardLogger
from lightning.pytorch import seed_everything
from sklearn.model_selection import StratifiedKFold, ParameterGrid, train_test_split

from models.network import Model
from models.gnn import GNN
from src.utils import GraphDataset, MyDataset

# Set random seed
seed_value = 42
torch.manual_seed(seed=seed_value)
seed_everything(seed_value, workers=True)

if torch.cuda.is_available():
    device = 'cuda'
elif torch.backends.mps.is_available():
    device = 'mps'
else:
    device = 'cpu'

warnings.filterwarnings('ignore')

def main():
    parser = ArgumentParser()
    parser.add_argument('--data_dir', type=str, help='Data directory')
    parser.add_argument('--label_dir', type=str, help='Labels directory')
    parser.add_argument('--thr', type=int, default=5, help='Threshold')

    parser.add_argument('--epochs', type=int, default=50, help='Number of epochs')
    parser.add_argument('--lr', type=float, default=1e-4, help='Learning rate')
    parser.add_argument('--batch_size', type=int, default=64, help='Batch size')
    parser.add_argument('--kfolds', type=int, default=5, help='Number of folds')

    parser.add_argument('--model_type', type=str, default='network', help='Model type')

    parser.add_argument('--dim_hidden', type=int, default=1024, help='Hidden dimension')
    parser.add_argument('--dim_hidden_', type=int, default=512, help='Hidden dimension')
    parser.add_argument('--dim_hidden_sparser', type=int, default=256, help='Hidden dimension')
    parser.add_argument('--num_heads', type=int, default=4, help='Number of heads')
    parser.add_argument('--num_seeds', type=int, default=32, help='Number of seeds')
    parser.add_argument('--ln', type=bool, default=True, help='Layer normalization')

    parser.add_argument('--conv_type', type=str, default='gcn', help='Convolution type')
    parser.add_argument('--gnn_intermediate_dim', type=int, default=512, help='GNN intermediate dimension')
    parser.add_argument('--gnn_output_node_dim', type=int, default=256, help='GNN output node dimension')
    parser.add_argument('--num_layers', type=int, default=3, help='Number of layers')
    parser.add_argument('--gat_heads', type=int, default=8, help='GAT heads')

    parser.add_argument('--dropout_ratio', type=float, default=0.5, help='Dropout ratio')
    parser.add_argument('--output_intermediate_dim', type=int, default=64, help='Output intermediate dimension')
    parser.add_argument('--dim_output', type=int, default=1, help='Output dimension')

    parser.add_argument('--alpha', type=float, default=1, help='Alpha')

    args = parser.parse_args()

    # LOAD DATA #
    matrices = np.load(args.data_dir)
    labels = np.load(args.label_dir)

    # TRAIN-TEST SPLIT FOR GRID SEARCH #
    train_idx, test_idx = train_test_split(
        np.arange(len(labels)), test_size=0.2, stratify=labels, random_state=seed_value
    )

    # Create dataset
    train_set = MyDataset(data=matrices[train_idx], labels=labels[train_idx]) if args.model_type == 'network' else GraphDataset(func_matrices=matrices[train_idx], labels=labels[train_idx], threshold=args.thr)
    test_set = MyDataset(data=matrices[test_idx], labels=labels[test_idx]) if args.model_type == 'network' else GraphDataset(func_matrices=matrices[test_idx], labels=labels[test_idx], threshold=args.thr)

    test_loader = torch_dataloader(test_set, batch_size=args.batch_size, shuffle=False) if args.model_type == 'network' else torch_g_dataloader(test_set, batch_size=args.batch_size, shuffle=False)

    print(f'\nTrain set: {len(train_set)} subjects')
    print(f'Test set: {len(test_set)} subjects')
    print('Functional Connectivity Matrices shape:', matrices.shape)
    print('Number of classes:', len(np.unique(labels)))

    # HYPERPARAMETERS GRID #
    # Network
    param_grid_1 = {
        'l0_lambda': [1e-6, 1e-7, 1e-8],
        'l1_lambda': [1e-4, 1e-5],
        'lambda_sym': [1e-4, 1e-5]
    }

    # GNN
    param_grid_2 = {
        'readout': ['mean', 'sum', 'max'],
        'l1_lambda': [1e-4, 1e-5]
    }

    best_params = None
    best_val_scores = float('-inf')

    for params in ParameterGrid(param_grid=param_grid_1) if args.model_type == 'network' else ParameterGrid(param_grid=param_grid_2):
        print(f'Tuning with params: {params}')

        # K-FOLD CROSS-VALIDATION #
        skf = StratifiedKFold(n_splits=args.kfolds, shuffle=True, random_state=seed_value)
        fold_results = []

        for fold, (train_idx, val_idx) in enumerate(skf.split(matrices[np.arange(len(train_set))], labels[np.arange(len(train_set))])) if args.model_type == 'network' else enumerate(skf.split(train_set, train_set.y)):
            print(f'\n=== Fold {fold + 1}/{args.kfolds} ===')

            training_set = Subset(train_set, train_idx) if args.model_type == 'network' else train_set[train_idx]
            validation_set = Subset(train_set, val_idx) if args.model_type == 'network' else train_set[val_idx]

            train_loader = torch_dataloader(training_set, batch_size=args.batch_size, shuffle=True) if args.model_type == 'network' else torch_g_dataloader(training_set, batch_size=args.batch_size, shuffle=True)
            val_loader = torch_dataloader(validation_set, batch_size=args.batch_size, shuffle=False) if args.model_type == 'network' else torch_g_dataloader(validation_set, batch_size=args.batch_size, shuffle=False)

            # MODEL DEFINITION #
            n_features = matrices.shape[1]

            if args.model_type == 'network':
                model = Model(
                    dim_input=n_features,
                    dim_output=args.dim_output,
                    dim_hidden=args.dim_hidden,
                    dim_hidden_=args.dim_hidden_,
                    dim_hidden_sparser=args.dim_hidden_sparser,
                    output_intermediate_dim=args.output_intermediate_dim,
                    num_heads=args.num_heads,
                    num_seeds=args.num_seeds,
                    lr=args.lr,
                    ln=args.ln,
                    dropout_ratio=args.dropout_ratio,
                    alpha=args.alpha,
                    l0_lambda=params['l0_lambda'],
                    l1_lambda=params['l1_lambda'],
                    lambda_sym=params['lambda_sym']
                ).to(device)
            else:
                model = GNN(
                    conv_type=args.conv_type,
                    in_channels=n_features,
                    gnn_intermediate_dim=args.gnn_intermediate_dim,
                    gnn_output_node_dim=args.gnn_output_node_dim,
                    output_nn_intermediate_dim=args.output_nn_intermediate_dim,
                    output_nn_out_dim=args.dim_output,
                    readout=params['readout'],
                    gat_heads=args.gat_heads,
                    gat_dropouts=args.dropout_ratio,
                    lr=args.lr,
                    num_layers=args.num_layers,
                    l1_lambda=params['l1_lambda']
                ).to(device)

            # TRAINING #
            monitor = 'val_f1'
            lr_monitor = LearningRateMonitor(logging_interval='epoch')
            callbacks = [lr_monitor]

            trainer = L.Trainer(
                max_epochs=args.epochs,
                callbacks=callbacks,
                accelerator=device,
                enable_progress_bar=True
            )

            trainer.fit(model, train_dataloaders=train_loader, val_dataloaders=val_loader)
            fold_results.append(model.validation_metrics_per_epoch)

        val_f1_score = []
        for fold in range(args.kfolds):
            for key, value in fold_results[fold].items():
                val_f1_score.append(value[2])

        avg_val_score = np.mean(val_f1_score)
        if avg_val_score > best_val_scores:
            print('BEST PARAMETERS UPDATED')
            best_val_scores = avg_val_score
            best_params = params

    # TESTING #
    print(f'\nBest parameters: {best_params}')
    print('Testing with best parameters...')
    if args.model_type == 'network':
        model = Model(
            dim_input=n_features,
            dim_output=args.dim_output,
            dim_hidden=args.dim_hidden,
            dim_hidden_=args.dim_hidden_,
            output_intermediate_dim=args.output_intermediate_dim,
            num_heads=args.num_heads,
            num_seeds=args.num_seeds,
            ln=args.ln,
            dropout_ratio=args.dropout_ratio,
            alpha=args.alpha,
            l0_lambda=best_params['l0_lambda'],
            l1_lambda=best_params['l1_lambda'],
            l2_lambda=best_params['l2_lambda'],
            lambda_sym=best_params['lambda_sym']
        ).to(device)
    else:
        model = GNN(
            conv_type=args.conv_type,
            in_channels=n_features,
            gnn_intermediate_dim=args.gnn_intermediate_dim,
            gnn_output_node_dim=args.gnn_output_node_dim,
            output_nn_intermediate_dim=args.output_nn_intermediate_dim,
            output_nn_out_dim=args.dim_output,
            readout=best_params['readout'],
            gat_heads=args.gat_heads,
            gat_dropouts=args.dropout_ratio,
            lr=args.lr,
            num_layers=args.num_layers,
            l1_lambda=best_params['l1_lambda'],
            l2_lambda=best_params['l2_lambda']
        ).to(device)

    train_loader = torch_dataloader(train_set, batch_size=args.batch_size, shuffle=True) if args.model_type == 'network' else torch_g_dataloader(train_set, batch_size=args.batch_size, shuffle=True)

    lr_monitor = LearningRateMonitor(logging_interval='epoch')
    callbacks = [lr_monitor]

    trainer = L.Trainer(
        max_epochs=args.epochs,
        callbacks=callbacks,
        accelerator=device,
        enable_progress_bar=True
    )

    trainer.fit(model, train_dataloaders=train_loader)
    trainer.test(model, test_dataloaders=test_loader)

    print(model.test_metrics_per_epoch)

    # SAVE BEST PARAMETERS #
    with open('best_params.json', 'w') as f:
        json.dump(best_params, f, indent=4)
    print(f"Best parameters saved to best_params.json")
    
if __name__ == '__main__':
    print(f'Using device: {device}')
    main()








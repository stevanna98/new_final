import torch
import numpy as np
import lightning as L
import warnings
import sys

from argparse import ArgumentParser
from torch.utils.data import Subset
from torch_geometric.data import DataLoader as torch_g_dataloader
from torch.utils.data import DataLoader as torch_dataloader
from lightning.pytorch.callbacks import EarlyStopping, LearningRateMonitor
from lightning.pytorch.loggers import TensorBoardLogger
from lightning.pytorch import seed_everything
from sklearn.model_selection import train_test_split, StratifiedKFold

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
device = 'cpu'
warnings.filterwarnings('ignore')

def main():
    parser = ArgumentParser()
    parser.add_argument('--data_dir', type=str, help='Data directory')
    parser.add_argument('--label_dir', type=str, help='Labels directory')
    parser.add_argument('--log_dir', type=str, help='Log directory')
    parser.add_argument('--threshold', type=int, default=5, help='Threshold')

    parser.add_argument('--epochs', type=int, default=100, help='Number of epochs')
    parser.add_argument('--lr', type=float, default=1e-3, help='Learning rate')
    parser.add_argument('--batch_size', type=int, default=64, help='Batch size')
    parser.add_argument('--kfolds', type=int, default=5, help='Number of folds')

    parser.add_argument('--model_type', type=str, default='network', help='Model type')

    parser.add_argument('--dim_hidden', type=int, default=1024, help='Hidden dimension')
    parser.add_argument('--dim_hidden_', type=int, default=512, help='Hidden dimension')
    parser.add_argument('--dim_hidden_sparser', type=int, default=1024, help='Hidden dimension')
    parser.add_argument('--num_heads', type=int, default=16, help='Number of heads')
    parser.add_argument('--num_seeds', type=int, default=16, help='Number of seeds')
    parser.add_argument('--ln', type=bool, default=True, help='Layer normalization')

    parser.add_argument('--conv_type', type=str, default='gatv2', help='Convolution type')
    parser.add_argument('--gnn_intermediate_dim', type=int, default=512, help='GNN intermediate dimension')
    parser.add_argument('--gnn_output_node_dim', type=int, default=256, help='GNN output node dimension')
    parser.add_argument('--gat_heads', type=int, default=8, help='GAT heads')
    parser.add_argument('--num_layers', type=int, default=3, help='Number of layers')
    parser.add_argument('--readout', type=str, default='mean', help='Readout function')

    parser.add_argument('--dropout_ratio', type=float, default=0.5, help='Dropout ratio')
    parser.add_argument('--output_intermediate_dim', type=int, default=64, help='Output intermediate dimension')
    parser.add_argument('--dim_output', type=int, default=1, help='Output dimension')

    parser.add_argument('--alpha', type=float, default=1, help='Alpha')
    parser.add_argument('--l0_lambda', type=float, default=1, help='L0 lambda')
    parser.add_argument('--l1_lambda', type=float, default=1e-5, help='L1 lambda')
    parser.add_argument('--l2_lambda', type=float, default=1e-5, help='L2 lambda')
    parser.add_argument('--lambda_sym', type=float, default=1e-5, help='Symmetric lambda')

    args = parser.parse_args()

    # LOAD DATA #
    matrices = np.load(args.data_dir)
    labels = np.load(args.label_dir)

    # Create dataset
    dataset = MyDataset(data=matrices, labels=labels) if args.model_type == 'network' else GraphDataset(func_matrices=matrices, labels=labels, threshold=args.threshold)

    print(f'\nDataset: {len(dataset)} subjects')
    print('Functional Connectivity Matrices shape:', matrices.shape)
    print('Number of classes:', len(np.unique(labels)))

    # K-FOLD CROSS-VALIDATION #
    skf = StratifiedKFold(n_splits=args.kfolds, shuffle=True, random_state=seed_value)
    fold_results = []   

    for fold, (train_idx, test_idx) in enumerate(skf.split(matrices, labels)) if args.model_type == 'network' else enumerate(skf.split(dataset, dataset.y)):
        print(f'\n=== Fold {fold + 1}/{args.kfolds} ===')

        train_set = Subset(dataset, train_idx) if args.model_type == 'network' else dataset[train_idx.tolist()]
        test_set = Subset(dataset, test_idx) if args.model_type == 'network' else dataset[test_idx.tolist()]

        train_len = int(0.8 * len(train_set))
        val_len = len(train_set) - train_len
        train_subset, val_subset = torch.utils.data.random_split(train_set, [train_len, val_len])

        train_loader = torch_dataloader(train_subset, batch_size=args.batch_size, shuffle=True) if args.model_type == 'network' else torch_g_dataloader(train_subset, batch_size=args.batch_size, shuffle=True)
        val_loader = torch_dataloader(val_subset, batch_size=args.batch_size, shuffle=False) if args.model_type == 'network' else torch_g_dataloader(val_subset, batch_size=args.batch_size, shuffle=False)
        test_loader = torch_dataloader(test_set, batch_size=args.batch_size, shuffle=False) if args.model_type == 'network' else torch_g_dataloader(test_set, batch_size=args.batch_size, shuffle=False)

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
                dropout_ratio=args.dropout_ratio,
                num_heads=args.num_heads,
                num_seeds=args.num_seeds,
                ln=args.ln,
                lr=args.lr,
                alpha=args.alpha,
                l0_lambda=args.l0_lambda,
                l1_lambda=args.l1_lambda,
                l2_lambda=args.l2_lambda,
                lambda_sym=args.lambda_sym
            ).to(device)
        else:
            model = GNN(
                conv_type=args.conv_type,
                in_channels=n_features,
                gnn_intermediate_dim=args.gnn_intermediate_dim,
                gnn_output_node_dim=args.gnn_output_node_dim,
                output_nn_intermediate_dim=args.output_intermediate_dim,
                output_nn_out_dim=args.dim_output,
                readout=args.readout,
                gat_heads=args.gat_heads,
                dropout_ratio=args.dropout_ratio,
                gat_dropouts=args.dropout_ratio,
                lr=args.lr,
                num_layers=args.num_layers,
                alpha=args.alpha,
                l1_lambda=args.l1_lambda,
                l2_lambda=args.l2_lambda
            ).to(device)

        # TRAINING #
        # Callbacks
        monitor = 'val_f1'
        early_stopping = EarlyStopping(monitor=monitor, patience=30, mode='max')
        lr_monitor = LearningRateMonitor(logging_interval='epoch')
        callbacks = [early_stopping, lr_monitor]

        # Logger
        tensorboard_logger = TensorBoardLogger(args.log_dir, name=f'logs_{args.model_type}_{fold}')

        trainer = L.Trainer(
                max_epochs=args.epochs,
                callbacks=callbacks,
                accelerator=device,
                logger=tensorboard_logger,
                enable_progress_bar=True
            )
        
        trainer.fit(model, train_dataloaders=train_loader, val_dataloaders=val_loader)

        # TESTING #
        trainer.test(model, dataloaders=test_loader)
        fold_results.append(model.test_metrics_per_epoch)

    # AVERAGE METRICS #
    print('\n=== Average Metrics: ===')
    accs = []
    f1s = []
    roc_aucs = []
    mccs = []
    for fold in range(args.kfolds):
        for key, value in fold_results[fold].items():
            accs.append(value[1])
            f1s.append(value[2])
            mccs.append(value[3])
            roc_aucs.append(value[4])

    print(f'\nAverage Accuracy: {np.mean(accs):.4f} ± {np.std(accs):.4f}')
    print(f'Average F1-Score: {np.mean(f1s):.4f} ± {np.std(f1s):.4f}')
    print(f'Average MCC: {np.mean(mccs):.4f} ± {np.std(mccs):.4f}')
    print(f'Average ROC-AUC: {np.mean(roc_aucs):.4f} ± {np.std(roc_aucs):.4f}')

if __name__ == '__main__':
    print(f'Using device: {device}')
    main()





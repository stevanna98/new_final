import torch
import torch.nn as nn
import torch.nn.functional as F
import lightning.pytorch as pl
import sys
import os

from collections import defaultdict

from torch import Tensor
from torch_geometric.utils.repeat import repeat 

from src.modules import *
from src.mask_modules import *
from src.sparser import Sparser
from src.utils import *

class Model(pl.LightningModule):
    def __init__(self,
                 dim_input: int,
                 dim_output: int,
                 dim_hidden_sparser: int,
                 dim_hidden: int,
                 dim_hidden_: int,
                 output_intermediate_dim: int,
                 dropout_ratio: float,
                 num_heads: int,
                 sparser_num_heads: int,
                 num_seeds: int,
                 ln: bool,
                 lr: float,
                 alpha: float,
                 l0_lambda: float,
                 l1_lambda: float,
                 lambda_sym: float):
        super(Model, self).__init__()
        self.save_hyperparameters()
        self.dim_input = dim_input
        self.dim_output = dim_output
        self.dim_hidden_sparser = dim_hidden_sparser
        self.dim_hidden = dim_hidden
        self.dim_hidden_ = dim_hidden_
        self.output_intermediate_dim = output_intermediate_dim
        self.dropout_ratio = dropout_ratio
        self.num_heads = num_heads
        self.sparser_num_heads = sparser_num_heads
        self.num_seeds = num_seeds
        self.ln = ln
        self.lr = lr

        self.alpha = alpha
        self.l0_lambda = l0_lambda
        self.l1_lambda = l1_lambda  
        self.lambda_sym = lambda_sym

        # ENCODER #
        self.sparser = Sparser(
            dim_Q=dim_input,
            dim_K=dim_input,
            dim_out=dim_hidden_sparser,
            sparser_num_heads=sparser_num_heads,
            ln=ln,
            dropout_ratio=dropout_ratio,
            l0_lambda=l0_lambda
        )

        self.enc_msab1 = MSAB(dim_input, dim_hidden, num_heads, ln, dropout_ratio)
        self.enc_msab2 = MSAB(dim_hidden, dim_hidden, num_heads, ln, dropout_ratio)
        self.enc_msab3 = MSAB(dim_hidden, dim_hidden_, num_heads, ln, dropout_ratio)
        self.enc_sab2 = SAB(dim_hidden_, dim_hidden_, num_heads, ln, dropout_ratio)

        # DECODER #
        self.pma = PMA(dim_hidden_, num_heads, num_seeds, ln, dropout_ratio)
        if self.num_seeds > 1:
            self.dec_sab = SAB(dim_hidden_, output_intermediate_dim, num_heads, ln, dropout_ratio)

        # CLASSIFIER #
        self.output_mlp = nn.Sequential(
            nn.Linear(output_intermediate_dim * self.num_seeds, output_intermediate_dim),
            nn.ReLU(),
            nn.Dropout(dropout_ratio),
            nn.Linear(output_intermediate_dim, dim_output)
        )

        # Storage
        self.train_outputs = defaultdict(list)
        self.validation_outputs = defaultdict(list)
        self.test_outputs = defaultdict(list)

        self.train_metrics_per_epoch = {}
        self.validation_metrics_per_epoch = {}
        self.test_metrics_per_epoch = {}

    def forward(self, X):
        mask, l0_penalty = self.sparser(X, X)

        

        enc1 = self.enc_msab1(X, mask)
        enc2 = self.enc_msab2(enc1, mask) + enc1
        enc3 = self.enc_msab3(enc2, mask) + enc2
        enc4 = self.enc_sab2(enc3) + enc3

        encoded = self.pma(enc4)
        if self.num_seeds > 1:
            # decoded = self.dec_sab(encoded)
            # readout = torch.mean(decoded, dim=1, keepdim=True)
            # readout = readout.flatten(start_dim=1)
            decoded = self.dec_sab(encoded)
            readout = decoded.flatten(start_dim=1)

            out = self.output_mlp(readout)
        else:
            out = self.output_mlp(encoded)

        return out, mask, l0_penalty
    
    def loss_function(self, y_true, y_pred, mask, l0_penalty):
        # Binary Cross Entropy Loss
        y_true = y_true.view(y_pred.shape)
        bce_loss = self.alpha * F.binary_cross_entropy_with_logits(y_pred.float(), y_true.float())
        
        # Symmetry Regularization
        sym_reg = self.lambda_sym * F.mse_loss(mask, mask.transpose(1, 2), reduction='mean')

        # L0 Regularization
        # l0_reg = self.l0_lambda * l0_penalty
        l0_reg = l0_penalty

        # L1 Regularization
        l1_norm = self.l1_lambda * sum(p.abs().sum() for p in self.parameters())

        loss = bce_loss + sym_reg + l0_reg + l1_norm

        return loss, (bce_loss, sym_reg, l0_reg, l1_norm)
    
    def _step(self, batch, batch_idx):
        X, y = batch
        out, mask, l0_penalty = self.forward(X)
        loss, loss_terms = self.loss_function(y, out, mask, l0_penalty)

        y_pred = torch.sigmoid(out).detach()
        y_pred = torch.where(y_pred > 0.5, 1.0, 0.0).long()

        metrics = get_classification_metrics(
            y_true=y.long().detach().cpu().numpy(),
            y_pred=y_pred.detach().cpu().numpy()
        )

        return loss, y, out, metrics, loss_terms, mask
    
    def training_step(self, batch, batch_idx):
        loss, ys, outs, metrics, loss_terms, mask = self._step(batch, batch_idx)

        self.log('train_loss', loss)
        self.log('train_acc', metrics[1], on_step=True, on_epoch=True)
        self.log('train_f1', metrics[2], on_step=True, on_epoch=True)

        self.train_outputs[self.current_epoch].append({'y_true': ys, 'y_pred': outs, 
                                                       'mask': mask, 'loss': loss,
                                                       'bce_loss': loss_terms[0],
                                                       'sym_reg': loss_terms[1],
                                                       'l0_reg': loss_terms[2],
                                                       'l1_norm': loss_terms[3]})

        return loss
    
    def validation_step(self, batch, batch_idx):
        loss, ys, outs, metrics, loss_terms, mask = self._step(batch, batch_idx)

        self.log('val_loss', loss)
        self.log('val_acc', metrics[1], on_step=True, on_epoch=True)
        self.log('val_f1', metrics[2], on_step=True, on_epoch=True)

        self.validation_outputs[self.current_epoch].append({'y_true': ys, 'y_pred': outs, 
                                                            'mask': mask, 'loss': loss,
                                                            'bce_loss': loss_terms[0],
                                                            'sym_reg': loss_terms[1],
                                                            'l0_reg': loss_terms[2],
                                                            'l1_norm': loss_terms[3]})

        return loss
    
    def test_step(self, batch, batch_idx):
        loss, ys, outs, metrics, loss_terms, mask = self._step(batch, batch_idx)

        self.log('test_loss', loss)
        self.log('test_acc', metrics[1], on_step=True, on_epoch=True)
        self.log('test_f1', metrics[2], on_step=True, on_epoch=True)

        self.test_outputs[self.current_epoch].append({'y_true': ys, 'y_pred': outs, 
                                                      'mask': mask, 'loss': loss,
                                                      'bce_loss': loss_terms[0],
                                                      'sym_reg': loss_terms[1],
                                                      'l0_reg': loss_terms[2],
                                                      'l1_norm': loss_terms[3]})

        return loss
    
    def _get_metrics_epoch_end(self, all_y_true, all_y_pred):
        all_y_pred = torch.sigmoid(all_y_pred.float())
        all_y_pred = torch.where(all_y_pred > 0.5, 1.0, 0.0).long()
        return get_classification_metrics(y_true=all_y_true.long().detach().cpu().numpy(), y_pred=all_y_pred.detach().cpu().numpy())
    
    def on_train_epoch_end(self, unused=None):
        all_y_true = [elem['y_true'] for elem in self.train_outputs[self.current_epoch]]
        all_y_pred = [elem['y_pred'] for elem in self.train_outputs[self.current_epoch]]

        all_y_true = torch.cat(all_y_true, dim=0)
        all_y_pred = torch.cat(all_y_pred, dim=0)

        metrics = self._get_metrics_epoch_end(all_y_true, all_y_pred)

        self.train_metrics_per_epoch[self.current_epoch] = metrics

        weights = torch.cat([p.view(-1) for p in self.parameters()])
        mean_weight = weights.mean().item()
        std_weight = weights.std().item()

        print(f"Epoch {self.current_epoch} - Mean Weight: {mean_weight:.6f}, Std: {std_weight:.6f}")

        # Print metrics
        print(f"Epoch {self.current_epoch} - Training Metrics:")
        print_classification_metrics(metrics)

        total_loss = [loss['loss'] for loss in self.train_outputs[self.current_epoch]]
        bce_loss = [loss['bce_loss'] for loss in self.train_outputs[self.current_epoch]]
        sym_reg = [loss['sym_reg'] for loss in self.train_outputs[self.current_epoch]]
        l0_reg = [loss['l0_reg'] for loss in self.train_outputs[self.current_epoch]]
        l1_norm = [loss['l1_norm'] for loss in self.train_outputs[self.current_epoch]]


        print('\n')
        print_loss(total_loss[-1], bce_loss[-1], sym_reg[-1], l0_reg[-1], l1_norm[-1])
        
        all_masks = [elem['mask'] for elem in self.train_outputs[self.current_epoch]]
        num_zeri = np.count_nonzero(all_masks[0].detach().cpu().numpy() == 0)
        print(f"Numero di zeri nella matrice: {num_zeri}")

        # all_masks = [elem['mask'] for elem in self.train_outputs[self.current_epoch]]

        # if all_masks:  # Controlla che ci siano maschere salvate
        #     random_index = torch.randint(len(all_masks), (1,)).item()
        #     random_mask = all_masks[random_index].detach().cpu().numpy()  # Converti in NumPy

        #     os.makedirs("masks", exist_ok=True)  # Crea la cartella se non esiste
        # np.save(f"masks/train_mask_epoch_{self.current_epoch}.npy", random_mask)

        del self.train_outputs[self.current_epoch]
        del all_y_true
        del all_y_pred

    def on_validation_epoch_end(self, unused=None):
        all_y_true = [elem['y_true'] for elem in self.validation_outputs[self.current_epoch]]
        all_y_pred = [elem['y_pred'] for elem in self.validation_outputs[self.current_epoch]]

        all_y_true = torch.cat(all_y_true, dim=0)
        all_y_pred = torch.cat(all_y_pred, dim=0)

        metrics = self._get_metrics_epoch_end(all_y_true, all_y_pred)

        self.validation_metrics_per_epoch[self.current_epoch] = metrics

        # Print metrics
        print(f"Epoch {self.current_epoch} - Validation Metrics:")
        print_classification_metrics(metrics)

        total_loss = [loss['loss'] for loss in self.validation_outputs[self.current_epoch]]
        bce_loss = [loss['bce_loss'] for loss in self.validation_outputs[self.current_epoch]]
        sym_reg = [loss['sym_reg'] for loss in self.validation_outputs[self.current_epoch]]
        l0_reg = [loss['l0_reg'] for loss in self.validation_outputs[self.current_epoch]]
        l1_norm = [loss['l1_norm'] for loss in self.validation_outputs[self.current_epoch]]

        print('\n')
        print_loss(total_loss[-1], bce_loss[-1], sym_reg[-1], l0_reg[-1], l1_norm[-1])

        # all_masks = [elem['mask'] for elem in self.validation_outputs[self.current_epoch]]

        # if all_masks:
        #     random_index = torch.randint(len(all_masks), (1,)).item()
        #     random_mask = all_masks[random_index].detach().cpu().numpy()

        #     os.makedirs("masks", exist_ok=True)
        #     np.save(f"masks/val_mask_epoch_{self.current_epoch}.npy", random_mask)

        del self.validation_outputs[self.current_epoch]
        del all_y_true
        del all_y_pred

    def on_test_epoch_end(self, unused=None):     
        all_y_true = [elem['y_true'] for elem in self.test_outputs[self.current_epoch]]
        all_y_pred = [elem['y_pred'] for elem in self.test_outputs[self.current_epoch]]

        all_y_true = torch.cat(all_y_true, dim=0)
        all_y_pred = torch.cat(all_y_pred, dim=0)

        metrics = self._get_metrics_epoch_end(all_y_true, all_y_pred)

        self.test_metrics_per_epoch[self.current_epoch] = metrics

        # Print metrics
        print(f"Epoch {self.current_epoch} - Test Metrics:")
        print_classification_metrics(metrics)

        total_loss = [loss['loss'] for loss in self.test_outputs[self.current_epoch]]
        bce_loss = [loss['bce_loss'] for loss in self.test_outputs[self.current_epoch]]
        sym_reg = [loss['sym_reg'] for loss in self.test_outputs[self.current_epoch]]
        l0_reg = [loss['l0_reg'] for loss in self.test_outputs[self.current_epoch]]
        l1_norm = [loss['l1_norm'] for loss in self.test_outputs[self.current_epoch]]

        print('\n')
        print_loss(total_loss[-1], bce_loss[-1], sym_reg[-1], l0_reg[-1], l1_norm[-1])

        del self.test_outputs[self.current_epoch]
        del all_y_true
        del all_y_pred

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.lr, weight_decay=1e-3)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer=optimizer, mode='max', factor=0.05, patience=10, verbose=True)

        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "monitor": "val_f1"
            }
        }






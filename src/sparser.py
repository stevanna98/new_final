import torch 
import torch.nn as nn
import torch.nn.functional as F
import math
import lightning.pytorch as pl
import sys

from src.l0_regularizer import *

class Sparser(pl.LightningModule):
    def __init__(self,
                 dim_Q: int,
                 dim_K: int,
                 dim_out: int,
                 num_heads: int,
                 ln: bool,
                 dropout_ratio: float,
                 N: int = 1079,
                 weight_decay = 1,
                 droprate_init = 0.5,
                 temperature = 2./3.,
                 lamba = 1.,
                 local_rep = False):
        super(Sparser, self).__init__()
        self.dim_Q = dim_Q
        self.dim_K = dim_K
        self.dim_out = dim_out

        self.N = N
        self.num_heads = num_heads
        self.dim_head = dim_out // num_heads

        l0_params = {
            'weight_decay': weight_decay,
            'droprate_init': droprate_init,
            'temperature': temperature,
            'lamba': lamba,
            'local_rep': local_rep
        }

        self.W_q = nn.ModuleList([
            L0Linear(self.dim_Q, self.dim_head, **l0_params) for _ in range(num_heads)
        ])
        self.W_k = nn.ModuleList([
            L0Linear(self.dim_K, self.dim_head, **l0_params) for _ in range(num_heads)
        ])

        self.dropout = nn.Dropout(dropout_ratio)

        if ln:
            self.ln_q = nn.LayerNorm(self.dim_Q)
            self.ln_k = nn.LayerNorm(self.dim_K)

    def _regularization(self, penalty):
        regularization = 0.
        regularization += - (1. / self.N) * penalty
        if torch.cuda.is_available():
            regularization = regularization.cuda()
        return regularization

    def forward(self, Q, K):
        Q_norm = Q if getattr(self, 'ln_q', None) is None else self.ln_q(Q)
        K_norm = K if getattr(self, 'ln_k', None) is None else self.ln_k(K)

        head_outputs = []
        total_penalty = 0
        for head in range(self.num_heads):
            Q_ = self.dropout(self.W_q[head](Q_norm))
            K_ = self.dropout(self.W_k[head](K_norm))

            l0_q = self.W_q[head].regularization()
            l0_q_reg = self._regularization(l0_q)

            l0_k = self.W_k[head].regularization()
            l0_k_reg = self._regularization(l0_k)

            tot_reg = (l0_q_reg + l0_k_reg) / 2
            total_penalty += tot_reg

            A = torch.softmax(Q_.bmm(K_.transpose(1,2))/math.sqrt(self.dim_head), 2)

            head_outputs.append(A)

        O = torch.stack(head_outputs, dim=1)
        O = O.mean(dim=1)

        return O, total_penalty / self.num_heads
            




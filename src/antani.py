import torch
import torch.nn as nn
import torch.nn.functional as F
import lightning.pytorch as pl
import math

class SparseMatrixAttention(pl.LightningModule):
    def __init__(self, dim_Q, dim_K, dim_V, dim_out, sparser_num_heads, 
                 init_percentile=90, lambda_sparsity=0.01):
        super(SparseMatrixAttention, self).__init__()
        self.dim_Q = dim_Q
        self.dim_K = dim_K
        self.dim_V = dim_V
        self.sparser_num_heads = sparser_num_heads
        self.dim_head = dim_out // sparser_num_heads    
        self.init_percentile = init_percentile  
        
        # Parametri base
        self.W_q = nn.ModuleList([
            nn.Linear(self.dim_Q, self.dim_head) for _ in range(sparser_num_heads)
        ])
        self.W_k = nn.ModuleList([
            nn.Linear(self.dim_K, self.dim_head) for _ in range(sparser_num_heads)
        ])
        self.W_v = nn.ModuleList([
            nn.Linear(self.dim_K, self.dim_head) for _ in range(sparser_num_heads)
        ])
        
        # Threshold learnable per ogni testa
        self.tau = nn.Parameter(
            torch.tensor(init_percentile / 100.0)
        )
        self.lambda_sparsity = lambda_sparsity
        
        # Inizializzazione
        for head in range(sparser_num_heads):
            nn.init.xavier_uniform_(self.W_q[head].weight)
            nn.init.xavier_uniform_(self.W_k[head].weight)

    def forward(self, Q, K):
        head_outputs = []
        masks = []
        
        for head in range(self.sparser_num_heads):
            Q_ = self.W_q[head](Q)
            K_ = self.W_k[head](K)
            V_ = self.W_v[head](K)

            A = torch.softmax(Q_.bmm(K_.transpose(1, 2)) / math.sqrt(self.dim_head), 2)

            current_thr = torch.quantile(A, self.tau)
            mask = (A > current_thr).float()

            sparsity_loss = self.lambda_sparsity * mask.sum()

            out = A.bmm(V_)
            head_outputs.append(out)
            masks.append(mask)
        
        O = torch.cat(head_outputs, 2)
        masks = torch.stack(masks, dim=1)
        masks_ = masks.mean(dim=1)

        return O, masks_, sparsity_loss
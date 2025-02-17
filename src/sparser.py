import torch
import torch.nn as nn
import torch.nn.functional as F
import lightning.pytorch as pl
import math

class SparseMatrixAttention(pl.LightningModule):
    def __init__(self, dim_Q, dim_K, dim_out, sparser_num_heads, 
                 init_percentile=90, lambda_sparsity=1e-3, dropout_ratio=0.5):
        super(SparseMatrixAttention, self).__init__()
        self.dim_Q = dim_Q
        self.dim_K = dim_K
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
            nn.Linear(self.dim_K, self.dim_K) for _ in range(sparser_num_heads)
        ])
        
        # Threshold learnable per ogni testa
        self.tau = nn.Parameter(
            torch.tensor(init_percentile / 100.0)
        )
        self.lambda_sparsity = lambda_sparsity
        
        for head in range(sparser_num_heads):
            nn.init.xavier_uniform_(self.W_q[head].weight)
            nn.init.xavier_uniform_(self.W_k[head].weight)
            nn.init.xavier_uniform_(self.W_v[head].weight)

        self.ln_q = nn.LayerNorm(self.dim_Q)
        self.ln_k = nn.LayerNorm(self.dim_K)
        self.ln_v = nn.LayerNorm(self.dim_K)

        self.dropout = nn.Dropout(dropout_ratio)

    def forward(self, Q, K):
        masks = []
        total_sparsity_loss = 0

        Q_norm = self.ln_q(Q)
        K_norm = self.ln_k(K)
        V_norm = self.ln_v(K)

        for head in range(self.sparser_num_heads):
            Q_ = self.dropout(self.W_q[head](Q_norm))
            K_ = self.dropout(self.W_k[head](K_norm))
            V_ = self.dropout(self.W_v[head](V_norm))

            A = torch.softmax(Q_.bmm(K_.transpose(1, 2)) / math.sqrt(self.dim_head), 2)
            attn_output = F.tanh(A.bmm(V_))

            current_thr = torch.quantile(attn_output, self.tau)
            mask = attn_output
            mask[torch.abs(attn_output) < current_thr] = 0

            num_elements = mask.numel()  
            num_zeros = num_elements - mask.sum() 
            sparsity_loss = self.lambda_sparsity * num_zeros
            total_sparsity_loss += sparsity_loss

            masks.append(mask)

        masks_ = torch.stack(masks, dim=1)
        masks_ = masks_.mean(dim=1)

        return masks_, total_sparsity_loss / self.sparser_num_heads
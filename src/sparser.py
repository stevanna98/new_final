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
        
        # Threshold learnable per ogni testa
        self.tau = nn.Parameter(
            torch.tensor(init_percentile / 100.0)
        )
        self.lambda_sparsity = lambda_sparsity
        
        for head in range(sparser_num_heads):
            nn.init.xavier_uniform_(self.W_q[head].weight)
            nn.init.xavier_uniform_(self.W_k[head].weight)

        self.ln_q = nn.LayerNorm(self.dim_Q)
        self.ln_k = nn.LayerNorm(self.dim_K)

        self.dropout = nn.Dropout(dropout_ratio)

        self.lambda_sparsity = 1e-3
        self.lambda_entropy = 1e-4
        self.lambda_smoothness = 1e-4

    def compute_entropy_loss(self, mask):
        """Calcola la loss di entropia per promuovere una distribuzione bilanciata"""
        eps = 1e-8
        p = mask.mean(dim=(1,2))
        entropy = -(p * torch.log(p + eps) + (1-p) * torch.log(1-p + eps))
        return -entropy.mean()  # Negativo perché vogliamo massimizzare l'entropia

    def compute_smoothness_loss(self, mask):
        """Calcola la loss di smoothness per promuovere pattern continui"""
        # Differenze orizzontali
        h_diff = torch.abs(mask[:, :, 1:] - mask[:, :, :-1])
        # Differenze verticali 
        v_diff = torch.abs(mask[:, 1:, :] - mask[:, :-1, :])
        
        smoothness = (h_diff.mean() + v_diff.mean()) / 2
        return smoothness

    def forward(self, Q, K):
        masks = []
        total_loss = 0

        Q_norm = self.ln_q(Q)
        K_norm = self.ln_k(K)

        for head in range(self.sparser_num_heads):
            Q_ = self.dropout(self.W_q[head](Q_norm))
            K_ = self.dropout(self.W_k[head](K_norm))

            A = torch.softmax(Q_.bmm(K_.transpose(1, 2)) / math.sqrt(self.dim_head), 2)

            current_thr = torch.quantile(A, self.tau)
            mask = (A > current_thr).float()

            # num_elements = mask.numel()  
            # num_zeros = num_elements - mask.sum() 
            # sparsity_loss = self.lambda_sparsity * num_zeros
            # total_sparsity_loss += sparsity_loss
            num_elements = mask.numel()
            num_zeros = num_elements - mask.sum()
            sparsity_loss = self.lambda_sparsity * (num_zeros / num_elements)
            entropy_loss = self.lambda_entropy * self.compute_entropy_loss(mask)
            smoothness_loss = self.lambda_smoothness * self.compute_smoothness_loss(mask)
            
            # Loss totale per questa head
            head_loss = sparsity_loss + entropy_loss + smoothness_loss
            total_loss += head_loss

            masks.append(mask)

        masks_ = torch.stack(masks, dim=1)
        masks_ = masks_.mean(dim=1)

        return masks_, total_loss / self.sparser_num_heads
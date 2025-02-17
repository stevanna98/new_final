import torch
import torch.nn as nn
import torch.nn.functional as F
import lightning.pytorch as pl
import math

class SparseMatrixAttention(pl.LightningModule):
    def __init__(self, dim_Q, dim_K, dim_out, sparser_num_heads,
                 init_percentile=90, lambda_sparsity=1e-3, dropout_ratio=0.5,
                 target_sparsity=0.5, sparsity_momentum=0.9):
        super(SparseMatrixAttention, self).__init__()
        self.dim_Q = dim_Q
        self.dim_K = dim_K
        self.sparser_num_heads = sparser_num_heads
        self.dim_head = dim_out // sparser_num_heads
        
        self.init_percentile = init_percentile
        
        # Base parameters
        self.W_q = nn.ModuleList([nn.Linear(self.dim_Q, self.dim_head) for _ in range(sparser_num_heads)])
        self.W_k = nn.ModuleList([nn.Linear(self.dim_K, self.dim_head) for _ in range(sparser_num_heads)])
        self.W_v = nn.ModuleList([nn.Linear(self.dim_K, self.dim_K) for _ in range(sparser_num_heads)])
        
        self.tau = nn.Parameter(torch.tensor(init_percentile / 100))
        self.lambda_sparsity = lambda_sparsity
        
        # Nuovi parametri per la regolarizzazione
        self.target_sparsity = target_sparsity
        self.sparsity_momentum = sparsity_momentum
        self.register_buffer('running_sparsity', torch.tensor(0.0))
        
        # for head in range(sparser_num_heads):
        #     nn.init.xavier_uniform_(self.W_q[head].weight)
        #     nn.init.xavier_uniform_(self.W_k[head].weight)
        #     nn.init.xavier_uniform_(self.W_v[head].weight)
        
        self.ln_q = nn.LayerNorm(self.dim_Q)
        self.ln_k = nn.LayerNorm(self.dim_K)
        self.ln_v = nn.LayerNorm(self.dim_K)
        
        self.dropout = nn.Dropout(dropout_ratio)

    def adaptive_threshold(self, attn_output, current_sparsity):
        """Adatta la threshold in base alla sparsity corrente"""
        if self.training:
            # Aggiorna la running sparsity con momentum
            self.running_sparsity = self.running_sparsity * self.sparsity_momentum + \
                                  current_sparsity * (1 - self.sparsity_momentum)
            
            # Adatta la threshold in base alla differenza tra sparsity corrente e target
            sparsity_diff = current_sparsity - self.target_sparsity
            adjustment = 0.01 * torch.sign(sparsity_diff)
            self.tau.data = torch.clamp(self.tau.data + adjustment, 0.1, 0.9)
        
        return torch.quantile(attn_output, self.tau)

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
            
            # Calcola la sparsity corrente
            current_sparsity = (torch.abs(attn_output) == 0).float().mean()
            
            # Usa la threshold adattiva
            current_thr = self.adaptive_threshold(attn_output, current_sparsity)
            mask = torch.where(torch.abs(attn_output) < current_thr, 
                             torch.zeros_like(attn_output), 
                             attn_output)
            
            # Sparsity loss con penalità per deviazione dal target
            sparsity_deviation = torch.abs(current_sparsity - self.target_sparsity)
            sparsity_loss = self.lambda_sparsity * sparsity_deviation
            total_sparsity_loss += sparsity_loss
            
            masks.append(mask)
        
        masks_ = torch.stack(masks, dim=1)
        masks_ = masks_.mean(dim=1)
        
        return masks_, total_sparsity_loss / self.sparser_num_heads
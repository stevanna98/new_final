import torch
import torch.nn as nn
import torch.nn.functional as F
import lightning.pytorch as pl
import math

class SparseMatrixAttention(pl.LightningModule):
    def __init__(self, dim_Q, dim_K, dim_out, sparser_num_heads, 
                 sparsity_target=0.1, reg_lambda=0.01):
        super(SparseMatrixAttention, self).__init__()
        self.dim_Q = dim_Q
        self.dim_K = dim_K
        self.sparser_num_heads = sparser_num_heads
        self.dim_head = dim_out // sparser_num_heads
        self.sparsity_target = sparsity_target
        self.reg_lambda = reg_lambda
        
        # Parametri base
        self.W_q = nn.ModuleList([
            nn.Linear(self.dim_Q, self.dim_head) for _ in range(sparser_num_heads)
        ])
        self.W_k = nn.ModuleList([
            nn.Linear(self.dim_K, self.dim_head) for _ in range(sparser_num_heads)
        ])
        
        # Threshold learnable per ogni testa
        self.learned_thresholds = nn.Parameter(
            torch.ones(sparser_num_heads) * 0.1
        )
        
        # Inizializzazione
        for head in range(sparser_num_heads):
            nn.init.xavier_uniform_(self.W_q[head].weight)
            nn.init.xavier_uniform_(self.W_k[head].weight)
        
        self.sparsity_loss = 0.0
        
    def sparsify_attention(self, A, head_idx, method='learned_threshold'):
        """
        Sparsifica la matrice di attenzione usando diversi metodi
        """
        if method == 'learned_threshold':
            # Usa la threshold learnable
            threshold = torch.sigmoid(self.learned_thresholds[head_idx])
            mask = A > threshold
            return A * mask.float(), mask
            
        elif method == 'topk':
            # Top-k con regolarizzazione
            k = int(A.size(-1) * self.sparsity_target)
            topk_values, _ = torch.topk(A, k, dim=-1)
            threshold = topk_values[..., -1:]
            mask = A >= threshold
            
            # Calcola la sparsity loss
            sparsity = mask.float().mean()
            self.sparsity_loss += self.reg_lambda * (sparsity - self.sparsity_target).abs()
            
            return A * mask.float(), mask
            
        elif method == 'gumbel':
            # Gumbel con regolarizzazione
            temperature = 0.1
            noise = -torch.empty_like(A).exponential_().log()
            scores = (torch.log(A + 1e-10) + noise) / temperature
            mask = F.gumbel_softmax(scores, tau=temperature, hard=True)
            
            # Calcola la sparsity loss
            sparsity = mask.float().mean()
            self.sparsity_loss += self.reg_lambda * (sparsity - self.sparsity_target).abs()
            
            return A * mask, mask

    def forward(self, Q, K, method='learned_threshold'):
        head_outputs = []
        masks = []
        
        # Resetta la sparsity loss ad ogni forward pass
        self.sparsity_loss = 0.0
        
        for head in range(self.sparser_num_heads):
            Q_ = self.W_q[head](Q)
            K_ = self.W_k[head](K)

            # Calcola i punteggi di attenzione
            A = torch.softmax(Q_.bmm(K_.transpose(1, 2)) / math.sqrt(self.dim_head), 2)
            A = torch.abs(A)
            
            # Sparsifica
            A_sparse, mask = self.sparsify_attention(A, head, method=method)
            
            head_outputs.append(mask)
            masks.append(mask)

        O = torch.stack(head_outputs, dim=1)
        O = O.mean(dim=1)

        return head_outputs[0], self.sparsity_loss / self.sparser_num_heads

import torch
import torch.nn as nn
import torch.nn.functional as F
import lightning.pytorch as pl
import math
from torch.autograd import Variable

def hard_sigmoid(x):
    return torch.min(torch.max(x, torch.zeros_like(x)), torch.ones_like(x))

class _L0Norm(pl.LightningModule):
    def __init__(self, origin, loc_mean=0, loc_sdev=0.01, beta=2/3, gamma=-0.1,
                 zeta=1.1, fix_temp=True):
        super(_L0Norm, self).__init__()
        self._origin = origin
        self._size = self._origin.weight.size()
        self.loc = nn.Parameter(torch.zeros(self._size).normal_(loc_mean, loc_sdev))
        self.temp = beta if fix_temp else nn.Parameter(torch.zeros(1).fill_(beta))
        self.register_buffer("uniform", torch.zeros(self._size))
        self.gamma = gamma
        self.zeta = zeta
        self.gamma_zeta_ratio = math.log(-gamma / zeta)

    def _get_mask(self):
        if self.training:
            self.uniform.uniform_()
            u = Variable(self.uniform)
            s = F.sigmoid((torch.log(u) - torch.log(1 - u) + self.loc) / self.temp)
            s = s * (self.zeta - self.gamma) + self.gamma
            penalty = F.sigmoid(self.loc - self.temp * self.gamma_zeta_ratio).sum()
        else:
            s = F.sigmoid(self.loc) * (self.zeta - self.gamma) + self.gamma
            penalty = 0
        return hard_sigmoid(s), penalty

class L0Linear(_L0Norm):
    def __init__(self, in_features, out_features, bias=True, **kwargs):
        super(L0Linear, self).__init__(nn.Linear(in_features, out_features, bias=bias), **kwargs)

    def forward(self, input):
        mask, penalty = self._get_mask()
        return F.linear(input, self._origin.weight * mask, self._origin.bias), penalty

class ThresholdAttention(nn.Module):
    def __init__(self, init_threshold=0.002):
        super(ThresholdAttention, self).__init__()
        self.threshold = nn.Parameter(torch.tensor(init_threshold))
        
    def forward(self, attention_weights):
        # Apply soft thresholding during training
        if self.training:
            mask = torch.sigmoid(20.0 * (attention_weights - self.threshold))
            thresholded = attention_weights * mask
        else:
            # Hard thresholding during inference
            mask = (attention_weights > self.threshold).float()
            thresholded = attention_weights * mask
            
        # Normalize the non-zero attention weights
        row_sums = thresholded.sum(dim=-1, keepdim=True)
        row_sums[row_sums == 0] = 1.0  # Avoid division by zero
        normalized = thresholded / row_sums
        
        return normalized, mask

class SparseMatrixAttention(pl.LightningModule):
    def __init__(self, dim_Q, dim_K, dim_out, sparser_num_heads, l0_weight=0.1):
        super(SparseMatrixAttention, self).__init__()
        self.dim_Q = dim_Q
        self.dim_K = dim_K
        self.sparser_num_heads = sparser_num_heads
        self.dim_head = dim_out // sparser_num_heads
        self.l0_weight = l0_weight

        self.W_q = nn.ModuleList([
            nn.Linear(self.dim_Q, self.dim_head) for _ in range(sparser_num_heads)
        ])
        self.W_k = nn.ModuleList([
            nn.Linear(self.dim_K, self.dim_head) for _ in range(sparser_num_heads)
        ])

        self.l0_norms = nn.ModuleList([
            L0Linear(dim_Q, dim_Q, bias=False) for _ in range(sparser_num_heads)
        ])
        
        # Add learnable threshold for each head
        self.thresholds = nn.ModuleList([
            ThresholdAttention() for _ in range(sparser_num_heads)
        ])

        for head in range(sparser_num_heads):
            nn.init.xavier_uniform_(self.W_q[head].weight)
            nn.init.xavier_uniform_(self.W_k[head].weight)

    def forward(self, Q, K):
        head_outputs = []
        total_penalty = 0
        sparsity_ratios = []
        
        for head in range(self.sparser_num_heads):
            Q_ = self.W_q[head](Q)
            K_ = self.W_k[head](K)

            # Compute attention scores
            A = torch.softmax(Q_.bmm(K_.transpose(1, 2)) / math.sqrt(self.dim_head), 2)
            A = torch.abs(A)

            # Apply L0 regularization
            gate, penalty = self.l0_norms[head](A)
            total_penalty += penalty

            # Apply learnable threshold
            thresholded_A, mask = self.thresholds[head](gate)
            
            # Calculate sparsity ratio for monitoring
            sparsity_ratio = (mask == 0).float().mean()
            sparsity_ratios.append(sparsity_ratio)
            
            head_outputs.append(thresholded_A)

        O = torch.stack(head_outputs, dim=1)
        O = O.mean(dim=1)

        return O, total_penalty / self.sparser_num_heads
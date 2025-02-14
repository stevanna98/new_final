import torch
import torch.nn as nn
import torch.nn.functional as F
import lightning.pytorch as pl
import math

class SparseMatrixAttention(pl.LightningModule):
    def __init__(self, dim_Q, dim_K, dim_out, sparser_num_heads):
        super(SparseMatrixAttention, self).__init__()
        self.dim_Q = dim_Q
        self.dim_K = dim_K
        self.sparser_num_heads = sparser_num_heads
        self.dim_head = dim_out // sparser_num_heads

        self.conv_q = nn.ModuleList([
            nn.Conv2d(1, self.dim_head, kernel_size=3, stride=1, padding=1) for _ in range(sparser_num_heads)
        ])
        self.conv_k = nn.ModuleList([
            nn.Conv2d(1, self.dim_head, kernel_size=3, stride=1, padding=1) for _ in range(sparser_num_heads)
        ])

        self.thr = nn.Parameter(torch.tensor(0.1, dtype=torch.float32))

        for head in range(sparser_num_heads):
            nn.init.xavier_uniform_(self.conv_q[head].weight)
            nn.init.xavier_uniform_(self.conv_k[head].weight)

    def forward(self, Q, K):
        Q = Q.unsqueeze(1)
        K = K.unsqueeze(1)
        
        head_outputs = []
        for head in range(self.sparser_num_heads):
            Q_ = self.conv_q[head](Q)
            K_ = self.conv_k[head](K)

            Q_ = Q_.mean(dim=1)
            K_ = K_.mean(dim=1)

            A = torch.softmax(Q_.bmm(K_.transpose(1, 2)) / math.sqrt(self.dim_head), 2)

            attn_out = A.bmm(Q_.transpose(1, 2)) 

            sparse_matrix = torch.where(A > self.thr, attn_out, torch.tensor(0.0, device=Q.device))  

            head_outputs.append(sparse_matrix)

        O = torch.stack(head_outputs, dim=1)
        O = O.mean(dim=1)

        l1_reg = self.lambda_l1 * (
            sum(torch.norm(conv_q.weight, p=1) for conv_q in self.conv_q) +
            sum(torch.norm(conv_k.weight, p=1) for conv_k in self.conv_k)
        )

        return O, l1_reg
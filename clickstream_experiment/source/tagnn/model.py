import math

import torch
import torch.nn.functional as F
from torch import nn
from torch.nn import Module, Parameter


class GNN(Module):
    def __init__(self, hidden_size, step=1):
        super(GNN, self).__init__()
        self.step = step
        self.hidden_size = hidden_size
        self.input_size = hidden_size * 2
        self.gate_size = 3 * hidden_size
        self.w_ih = Parameter(torch.Tensor(self.gate_size, self.input_size))
        self.w_hh = Parameter(torch.Tensor(self.gate_size, self.hidden_size))
        self.b_ih = Parameter(torch.Tensor(self.gate_size))
        self.b_hh = Parameter(torch.Tensor(self.gate_size))
        self.b_iah = Parameter(torch.Tensor(self.hidden_size))
        self.b_oah = Parameter(torch.Tensor(self.hidden_size))

        self.linear_edge_in = nn.Linear(self.hidden_size, self.hidden_size, bias=True)
        self.linear_edge_out = nn.Linear(self.hidden_size, self.hidden_size, bias=True)
        self.linear_edge_f = nn.Linear(self.hidden_size, self.hidden_size, bias=True)

    def GNNCell(self, A, hidden):
        input_in = torch.matmul(A[:, :, :A.shape[1]], self.linear_edge_in(hidden)) + self.b_iah
        input_out = torch.matmul(A[:, :, A.shape[1]: 2 * A.shape[1]], self.linear_edge_out(hidden)) + self.b_oah
        inputs = torch.cat([input_in, input_out], 2)
        gi = F.linear(inputs, self.w_ih, self.b_ih)
        gh = F.linear(hidden, self.w_hh, self.b_hh)
        i_r, i_i, i_n = gi.chunk(3, 2)
        h_r, h_i, h_n = gh.chunk(3, 2)
        resetgate = torch.sigmoid(i_r + h_r)
        inputgate = torch.sigmoid(i_i + h_i)
        newgate = torch.tanh(i_n + resetgate * h_n)
        hy = newgate + inputgate * (hidden - newgate)
        return hy

    def forward(self, A, hidden):
        for i in range(self.step):
            hidden = self.GNNCell(A, hidden)
        return hidden


class SessionGraph(Module):
    def __init__(self, opt, n_node, init_embeddings=None):
        super(SessionGraph, self).__init__()
        self.hidden_size = opt.hiddenSize
        self.n_node = n_node
        self.batch_size = opt.batchSize
        self.variant = opt.variant
        self.use_target = not opt.ignore_target
        if init_embeddings is not None:
            self.embedding = nn.Embedding.from_pretrained(torch.FloatTensor(init_embeddings))
        else:
            self.embedding = nn.Embedding(self.n_node, self.hidden_size)
        self.gnn = GNN(self.hidden_size, step=opt.step)
        self.linear_one = nn.Linear(self.hidden_size, self.hidden_size, bias=True)
        self.linear_two = nn.Linear(self.hidden_size, self.hidden_size, bias=True)
        self.linear_three = nn.Linear(self.hidden_size, 1, bias=False)
        if self.variant != 'hybrid':
            self.linear_transform = nn.Linear(self.hidden_size, self.hidden_size, bias=True)
        else:
            self.linear_transform = nn.Linear(self.hidden_size * 2, self.hidden_size, bias=True)
        self.linear_t = nn.Linear(self.hidden_size, self.hidden_size, bias=False)  #target attention
        self.loss_function = nn.CrossEntropyLoss()
        self.optimizer = torch.optim.Adam(self.parameters(), lr=opt.lr, weight_decay=opt.l2)
        self.scheduler = torch.optim.lr_scheduler.StepLR(self.optimizer, step_size=opt.lr_dc_step, gamma=opt.lr_dc)
        self.reset_parameters()

    def reset_parameters(self):
        stdv = 1.0 / math.sqrt(self.hidden_size)
        for weight in self.parameters():
            if weight.requires_grad:
                weight.data.uniform_(-stdv, stdv)

    def compute_scores(self, hidden, mask):
        # hidden - v_i^(t)
        # ht - s_local
        # a - s_global -> s_t
        # b - v_i^(0)
        # target - s_target
        ht = hidden[torch.arange(mask.shape[0]).long(), torch.sum(mask, 1) - 1] # batch_size x latent_size
        s_global = self.compute_s_global(hidden, mask, ht)
        if self.variant == 'local':
            a = self.linear_transform(ht)
        elif self.variant == 'global':
            a = self.linear_transform(s_global)
        elif self.variant == 'hybrid':
            a = self.linear_transform(torch.cat([s_global, ht], 1))
        else:
            a = torch.zeros(ht.size(), dtype=torch.float, device='cuda')

        b = self.embedding.weight[1:]  # n_nodes x latent_size
        a = a.view(ht.shape[0], 1, ht.shape[1])  # b,1,d
        if self.use_target:
            # target attention: sigmoid(hidden M b)
            # mask  # batch_size x seq_length
            hidden = hidden * mask.view(mask.shape[0], -1, 1).float()  # batch_size x seq_length x latent_size
            qt = self.linear_t(hidden)  # batch_size x seq_length x latent_size
            # beta = torch.sigmoid(b @ qt.transpose(1,2))  # batch_size x n_nodes x seq_length
            beta = F.softmax(b @ qt.transpose(1,2), -1)  # batch_size x n_nodes x seq_length
            target = beta @ hidden  # batch_size x n_nodes x latent_size
            a = a + target  # b,n,d
        scores = torch.sum(a * b, -1)  # b,n
        # scores = torch.matmul(a, b.transpose(1, 0))
        return scores

    def compute_s_global(self, hidden, mask, ht):
        q1 = self.linear_one(ht).view(ht.shape[0], 1, ht.shape[1])  # batch_size x 1 x latent_size
        q2 = self.linear_two(hidden)  # batch_size x seq_length x latent_size
        alpha = self.linear_three(torch.sigmoid(q1 + q2))  # (b,s,1)
        # alpha = torch.sigmoid(alpha) # B,S,1
        alpha = F.softmax(alpha, 1)  # B,S,1
        return torch.sum(alpha * hidden * mask.view(mask.shape[0], -1, 1).float(), 1)  # (b,d)

    def forward(self, inputs, A):
        hidden = self.embedding(inputs)
        hidden = self.gnn(A, hidden)
        return hidden

    def unfreeze_embeddings(self):
        self.embedding.requires_grad_(True)
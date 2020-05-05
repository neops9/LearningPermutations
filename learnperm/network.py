import math
import torch.nn as nn
import torch
import numpy as np


class FastText(torch.nn.Module):
    def __init__(self, embedding_table, add_unk):
        super().__init__()
        n_embs = len(embedding_table)
        if add_unk:
            n_embs += 1
        self.embs = nn.Embedding(n_embs, 300)
        self.embs.weight.requires_grad=False

        for i, v in enumerate(embedding_table):
            self.embs.weight[i, :] = torch.tensor(v)
        if add_unk:
            self.embs.weight[-1, :].zero_()

    def forward(self, sentence):
        return self.embs(sentence)


class PermutationModule(nn.Module):
    def __init__(self, args, input_dim, dropout=0.):
        super(PermutationModule, self).__init__()

        self.bigram_left_proj = nn.Linear(input_dim, args.proj_dim, bias=True)
        self.bigram_right_proj = nn.Linear(input_dim, args.proj_dim, bias=False)  # bias will be added bu the left proj
        self.bigram_activation = nn.ReLU()
        self.bigram_output_proj = nn.Linear(args.proj_dim, 1, bias=True)

        self.start_builder = nn.Sequential(
            nn.Linear(input_dim, args.proj_dim, bias=True),
            nn.ReLU(),
            nn.Linear(args.proj_dim, 1, bias=True)
        )
        self.end_builder = nn.Sequential(
            nn.Linear(input_dim, args.proj_dim, bias=True),
            nn.ReLU(),
            nn.Linear(args.proj_dim, 1, bias=True)
        )

        self.initialize_parameters()

    def initialize_parameters(self):
        with torch.no_grad():
            self.bigram_left_proj.bias.zero_()
            self.bigram_output_proj.bias.zero_()
            self.start_builder[0].bias.zero_()
            self.start_builder[2].bias.zero_()
            self.end_builder[0].bias.zero_()
            self.end_builder[2].bias.zero_()

    def forward(self, input):
        # input must be of shape (n batch, n words, emb dim)
        start = self.start_builder(input)
        end = self.end_builder(input)

        # shape: (n batch, n words, proj dim)
        left_proj = self.bigram_left_proj(input)
        right_proj = self.bigram_right_proj(input)

        # shape: (n batch, n words, n words, proj dim)
        proj = left_proj.unsqueeze(1) + right_proj.unsqueeze(2)
        bigram = self.bigram_output_proj(self.bigram_activation(proj))

        return bigram, start, end


class Network(nn.Module):
    def __init__(self, args, embeddings_table, add_unk):
        super(Network, self).__init__()
        self.feature_extractor = FastText(embeddings_table, add_unk)
        self.permutation = PermutationModule(args, 300)

    def forward(self, input):
        # input must be of shape: (batch, n words)
        feature = self.feature_extractor(input)
        ret = self.permutation(feature)
        return ret

    @staticmethod
    def add_cmd_options(cmd):
        cmd.add_argument('--proj-dim', type=int, default=128, help="Dimension of the output projection")
        cmd.add_argument('--activation', type=str, default="tanh", help="activation to use in weightning modules: tanh, relu, leaky_relu")

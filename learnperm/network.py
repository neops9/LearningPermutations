import math
import torch.nn as nn
import torch
import numpy as np

class SequenceDropout(nn.Module):
    def __init__(self, p, broadcast_batch=False, broadcast_time=False):
        super().__init__()
        self.p = p
        self.broadcast_batch = broadcast_batch
        self.broadcast_time = broadcast_time

    # (n batch, n words, n features)
    def forward(self, x):
        if self.training and self.p > 0.:
            # assume x is of shape (batch, word, feature)
            n_batch = 1 if self.broadcast_batch else x.shape[0]
            n_words = 1 if self.broadcast_time else x.shape[1]
            n_feats = x.shape[2]
            mask = torch.empty(
                (n_batch, n_words, n_feats),
                dtype=torch.float,
                device=x.device,
                requires_grad=False
            )
            mask.bernoulli_(1 - self.p)
            mask /= (1 - self.p)
            x = x * mask

        return x

class FastText(torch.nn.Module):
    def __init__(self, embedding_table, add_unk):
        super().__init__()
        n_embs = len(embedding_table)
        if add_unk:
            n_embs += 1
        self.embs = nn.Embedding(n_embs, 300)
        self.embs.weight.requires_grad=False

        with torch.no_grad():
            for i, v in enumerate(embedding_table):
                self.embs.weight[i, :] = torch.tensor(v)
            if add_unk:
                self.embs.weight[-1, :].zero_()

    def forward(self, sentence):
        return self.embs(sentence)



class PermutationModule(nn.Module):
    def __init__(self, args, input_dim, input_dropout=0., proj_dropout=0.):
        super(PermutationModule, self).__init__()

        self.input_dropout = SequenceDropout(p=input_dropout, broadcast_time=True, broadcast_batch=False)
        self.proj_dropout = SequenceDropout(p=proj_dropout, broadcast_time=True, broadcast_batch=False)

        self.bigram_left_proj = nn.Linear(input_dim, args.proj_dim, bias=True)
        self.bigram_right_proj = nn.Linear(input_dim, args.proj_dim, bias=False)  # bias will be added bu the left proj
        self.bigram_activation = nn.ReLU()
        self.bigram_output_proj = nn.Linear(args.proj_dim, 1, bias=True)

        self.left_layer_norm = nn.LayerNorm(args.proj_dim)
        self.right_layer_norm = nn.LayerNorm(args.proj_dim)

        self.start_builder = nn.Sequential(
            nn.Linear(input_dim, args.proj_dim, bias=True),
            nn.ReLU(),
            SequenceDropout(p=proj_dropout, broadcast_time=True, broadcast_batch=False),
            nn.Linear(args.proj_dim, 1, bias=True)
        )
        self.end_builder = nn.Sequential(
            nn.Linear(input_dim, args.proj_dim, bias=True),
            nn.ReLU(),
            SequenceDropout(p=proj_dropout, broadcast_time=True, broadcast_batch=False),
            nn.Linear(args.proj_dim, 1, bias=True)
        )

        self.initialize_parameters()

    def initialize_parameters(self):
        with torch.no_grad():
            self.bigram_left_proj.bias.zero_()
            self.bigram_output_proj.bias.zero_()
            self.start_builder[0].bias.zero_()
            self.start_builder[3].bias.zero_()
            self.end_builder[0].bias.zero_()
            self.end_builder[3].bias.zero_()

    def forward(self, input):
        n_batch = input.shape[0]
        n_words = input.shape[1]
        input = self.input_dropout(input)

        # input must be of shape (n batch, n words, emb dim)
        start = self.start_builder(input)
        end = self.end_builder(input)

        # shape: (n batch, n words, proj dim)
        left_proj = self.bigram_left_proj(input)
        right_proj = self.bigram_right_proj(input)

        # shape: (n batch, n words, n words, proj dim)
        proj = left_proj.unsqueeze(1) + right_proj.unsqueeze(2)
        proj = self.bigram_activation(proj)
        proj = self.proj_dropout(proj.reshape(n_batch, n_words * n_words, -1)).reshape(n_batch, n_words, n_words, -1)
        bigram = self.bigram_output_proj(proj)

        return bigram, start, end


class Network(nn.Module):
    def __init__(self, args, embeddings_table, add_unk):
        super(Network, self).__init__()
        self.feature_extractor = FastText(embeddings_table, add_unk)
        self.permutation = PermutationModule(args, 300, input_dropout=args.input_dropout, proj_dropout=args.proj_dropout)

    def forward(self, input):
        # input must be of shape: (batch, n words)
        feature = self.feature_extractor(input)
        ret = self.permutation(feature)
        return ret

    @staticmethod
    def add_cmd_options(cmd):
        cmd.add_argument('--proj-dim', type=int, default=128, help="Dimension of the output projection")
        cmd.add_argument('--activation', type=str, default="tanh", help="activation to use in weightning modules: tanh, relu, leaky_relu")
        cmd.add_argument('--input-dropout', type=float, default=0., help="Dropout for the input")
        cmd.add_argument('--proj-dropout', type=float, default=0., help="Dropout for the proj")

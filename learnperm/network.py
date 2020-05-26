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
    def __init__(self, embedding_table, add_unk, padding_idx):
        super().__init__()
        n_embs = len(embedding_table)
        if add_unk:
            n_embs += 1
        self.embs = nn.Embedding(n_embs, 300, padding_idx=padding_idx)
        self.embs.weight.requires_grad=False

        with torch.no_grad():
            for i, v in enumerate(embedding_table):
                self.embs.weight[i, :] = torch.tensor(v)
            if add_unk:
                self.embs.weight[-1, :].zero_()

    def forward(self, sentence):
        return self.embs(sentence)
        
class FeatureExtractionModule(nn.Module):
    def __init__(self, args, embeddings_table, add_unk, word_padding_idx, pos_padding_idx=0, n_tags=0):
        super(FeatureExtractionModule, self).__init__()
        self.output_dim = 0

        if args.pos_embs:
            if n_tags <= 0:
                raise RuntimeError("Number of tags is not set!")

            self.pos_embs = nn.Embedding(n_tags + 1, args.pos_embs_dim, padding_idx=pos_padding_idx).to(args.device)
            self.output_dim += args.pos_embs_dim
        else:
            self.pos_embs = None

        if args.word_embs:
            self.word_embs = FastText(embeddings_table, add_unk, word_padding_idx)
            self.output_dim += args.word_embs_dim
        else:
            self.word_embs = None

        if self.output_dim == 0:
            raise RuntimeError("No input features set!")

        self.initialize_parameters()

    def initialize_parameters(self):
        with torch.no_grad():
            if self.pos_embs is not None:
                nn.init.uniform_(self.pos_embs.weight, -0.01, 0.01)

    @staticmethod
    def add_cmd_options(cmd):
        cmd.add_argument('--pos-embs', action="store_true", help="Use POS embeddings")
        cmd.add_argument('--pos-embs-dim', type=int, default=50, help="Dimension of the POS embs")
        cmd.add_argument('--word-embs', action="store_true", help="Use word embeddings")
        cmd.add_argument('--word-embs-dim', type=int, default=300, help="Dimension of the word embs (FastText = 300)")

    def forward(self, inputs):
        repr_list = []
        #repr_list.append(self.fasttext(sentence["tokens"]).to(args.device))
        
        if self.word_embs is not None: 
            batch_words = [sentence["tokens"].to(self.word_embs.embs.weight.device) for sentence in inputs]
            padded_inputs = torch.nn.utils.rnn.pad_sequence(
                    batch_words,
                    batch_first=True,
                    padding_value=self.word_embs.embs.padding_idx
                )

            repr_list.append(self.word_embs(padded_inputs))

        if self.pos_embs is not None:
            padded_inputs = torch.nn.utils.rnn.pad_sequence(
                [sentence["tags"].to(self.pos_embs.weight.device) for sentence in inputs],
                batch_first=True,
                padding_value=self.pos_embs.padding_idx
            )
            repr_list.append(self.pos_embs(padded_inputs))

            #pos_tags = sentence["tags"].to(self.pos_embs.weight.device)
            #repr_list.append(self.pos_embs(pos_tags))

        # combine word representations
        if len(repr_list) == 1:
            token_repr = repr_list[0]
        else:
            token_repr = torch.cat(repr_list, 2)

        return token_repr

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
    def __init__(self, args, embeddings_table, add_unk, word_padding_idx, pos_padding_idx, n_tags):
        super(Network, self).__init__()
        self.feature_extractor = FeatureExtractionModule(args, embeddings_table, add_unk, word_padding_idx, pos_padding_idx, n_tags=n_tags)
        self.permutation = PermutationModule(args, self.feature_extractor.output_dim, input_dropout=args.input_dropout, proj_dropout=args.proj_dropout)

    def forward(self, input):
        # input must be of shape: (batch, n words)
        feature = self.feature_extractor(input)
        ret = self.permutation(feature)
        return ret

    @staticmethod
    def add_cmd_options(cmd):
        FeatureExtractionModule.add_cmd_options(cmd)

        cmd.add_argument('--proj-dim', type=int, default=128, help="Dimension of the output projection")
        cmd.add_argument('--activation', type=str, default="tanh", help="activation to use in weightning modules: tanh, relu, leaky_relu")
        cmd.add_argument('--input-dropout', type=float, default=0., help="Dropout for the input")
        cmd.add_argument('--proj-dropout', type=float, default=0., help="Dropout for the proj")

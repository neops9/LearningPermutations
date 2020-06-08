import math
import torch.nn as nn
import torch
import numpy as np
import learnperm.special_tokens

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


class SpecialEmbeddingsNetwork(nn.Module):
    def __init__(self, size):
        super(SpecialEmbeddingsNetwork, self).__init__()
        self.size = size

        self.dict = learnperm.special_tokens.Dict()
        self.embs = nn.Embedding(len(self.dict) + 1, self.size, padding_idx=len(self.dict)) 

        print("Special word embeddings")
        print("\tsize: %i" % size)
        print(flush=True)

    def forward(self, inputs):
        values = self.embs(inputs)

        # masks
        padding_mask = (inputs != len(self.dict)).float().unsqueeze(-1).expand((inputs.shape[0], inputs.shape[1], 300))
        values = values * padding_mask
        special_words_mask = (inputs != 0.).float().float().unsqueeze(-1).expand((inputs.shape[0], inputs.shape[1], 300))
        return values * special_words_mask

class FastText(torch.nn.Module):
    def __init__(self, args, embedding_table, add_unk, word_padding_idx):
        super().__init__()
        n_embs = len(embedding_table)

        if add_unk:
            n_embs += 1
            n_embs += len(learnperm.special_tokens.Dict())
            
        self.embs = nn.Embedding(n_embs, 300, padding_idx=word_padding_idx)
        self.embs.weight.requires_grad=False
        self.special_embs = SpecialEmbeddingsNetwork(300)
        self.n_unk = len(learnperm.special_tokens.Dict())
        self.add_context = args.add_context
        self.output_dim = args.word_embs_dim

        if self.add_context:
            self.output_dim += args.word_embs_dim * 2

        with torch.no_grad():
            for i, v in enumerate(embedding_table):
                self.embs.weight[i, :] = torch.tensor(v)
            if add_unk:
                self.embs.weight[-self.n_unk, :].zero_()

    def forward(self, inputs):
        batch_words = [sentence["tokens"].to(self.embs.weight.device) for sentence in inputs]
        batch_specials = [sentence["special_words"].to(self.special_embs.embs.weight.device) for sentence in inputs]

        if self.add_context:
            batch_words_left = []
            batch_words_right = []
            batch_specials_left = []
            batch_specials_right = []

            for sentence in inputs:
                l_t = sentence["tokens"].clone().to(self.embs.weight.device)
                r_t = sentence["tokens"].clone().to(self.embs.weight.device)
                l_s = sentence["special_words"].clone().to(self.embs.weight.device)
                r_s = sentence["special_words"].clone().to(self.embs.weight.device)
                for i in range(1, len(l_t)):
                    l_t[i] = sentence["tokens"][i-1]
                    l_s[i] = sentence["special_words"][i-1]
                batch_words_left.append(l_t)
                batch_specials_left.append(l_s)
                for i in range(len(r_t)-1):
                    r_t[i] = sentence["tokens"][i+1]
                    r_s[i] = sentence["special_words"][i+1]
                batch_words_right.append(r_t)
                batch_specials_right.append(r_s)

        padded_words = torch.nn.utils.rnn.pad_sequence(
                batch_words,
                batch_first=True,
                padding_value=self.embs.padding_idx
            )

        padded_specials = torch.nn.utils.rnn.pad_sequence(
                batch_specials,
                batch_first=True,
                padding_value=self.special_embs.embs.padding_idx
            )

        repr_list = [self.embs(padded_words) + self.special_embs(padded_specials)]

        if self.add_context:
            padded_words_left = torch.nn.utils.rnn.pad_sequence(
                    batch_words_left,
                    batch_first=True,
                    padding_value=self.embs.padding_idx
                )

            padded_specials_left = torch.nn.utils.rnn.pad_sequence(
                    batch_specials_left,
                    batch_first=True,
                    padding_value=self.special_embs.embs.padding_idx
                )

            repr_list.append(self.embs(padded_words_left) + self.special_embs(padded_specials_left))

            padded_words_right = torch.nn.utils.rnn.pad_sequence(
                    batch_words_right,
                    batch_first=True,
                    padding_value=self.embs.padding_idx
                )

            padded_specials_right = torch.nn.utils.rnn.pad_sequence(
                    batch_specials_right,
                    batch_first=True,
                    padding_value=self.special_embs.embs.padding_idx
                )

            repr_list.append(self.embs(padded_words_right) + self.special_embs(padded_specials_right))

        if len(repr_list) == 1:
            ret = repr_list[0]
        else:
            ret = torch.cat(repr_list, 2)

        return ret


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
            self.word_embs = FastText(args, embeddings_table, add_unk, word_padding_idx)
            self.output_dim += self.word_embs.output_dim
        else:
            self.word_embs = None

        if args.lstm:
            self.lstm = nn.LSTM(
                    args.word_embs_dim,
                    args.lstm_dim,
                    bidirectional=False,
                    batch_first=True
                )
            self.output_dim += args.lstm_dim
        else:
            self.lstm = None

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
        cmd.add_argument('--lstm', action="store_true", help="Use word embeddings")
        cmd.add_argument('--lstm-dim', type=int, default=200, help="Dimension of the LSTM")

    def forward(self, inputs):
        repr_list = []
        
        if self.word_embs is not None:
            ret = self.word_embs(inputs)
            repr_list.append(ret)

            if self.lstm is not None:
                lstm_ret, _ = self.lstm(ret)
                repr_list.append(lstm_ret)
                
        if self.pos_embs is not None:
            padded_inputs = torch.nn.utils.rnn.pad_sequence(
                [sentence["tags"].to(self.pos_embs.weight.device) for sentence in inputs],
                batch_first=True,
                padding_value=self.pos_embs.padding_idx
            )
            repr_list.append(self.pos_embs(padded_inputs))

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
        self.bigram_right_proj = nn.Linear(input_dim, args.proj_dim, bias=False)  # bias will be added by the left proj
        
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
        cmd.add_argument('--add-context', action="store_true", help="Add left and right context")

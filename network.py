import math
import torch.nn as nn
import torch
import numpy as np
from gensim.models.keyedvectors import KeyedVectors

class FastText(torch.nn.Module):
    def __init__(self, filename):
        super().__init__()
        self.fasttext = KeyedVectors.load_word2vec_format(filename)

    def get_word_vector(self, word):
        if not word in self.fasttext.vocab:
            return np.zeros(300)

        return self.fasttext[word]

    def get_word_vectors(self, sentence):
        return np.array([self.get_word_vector(word) for word in sentence])

    def forward(self, sentence):
        with torch.no_grad():
            features = self.get_word_vectors(sentence)

        return torch.from_numpy(features).type(torch.FloatTensor)

class PermutationModule(nn.Module):
    def __init__(self, args, input_dim, dropout=0.):
        super(PermutationModule, self).__init__()

        self.activation = torch.tanh
        self.proj_dropout = nn.Dropout(dropout)
        self.norm = nn.LayerNorm(input_dim)

        self.left_projection = torch.nn.Parameter(data=torch.Tensor(1, input_dim, args.proj_dim))
        self.right_projection = torch.nn.Parameter(data=torch.Tensor(1, input_dim, args.proj_dim))
        self.bias_projection = torch.nn.Parameter(data=torch.Tensor(1, 1, args.proj_dim))

        self.output_projection = torch.nn.Parameter(data=torch.Tensor(args.proj_dim, 1))

        self.projection_start = torch.nn.Parameter(data=torch.Tensor(1, input_dim, args.proj_dim))
        self.bias_projection_start = torch.nn.Parameter(data=torch.Tensor(1, 1, args.proj_dim))
        self.output_start_projection = torch.nn.Parameter(data=torch.Tensor(args.proj_dim, 1))

        self.projection_end = torch.nn.Parameter(data=torch.Tensor(1, input_dim, args.proj_dim))
        self.bias_projection_end = torch.nn.Parameter(data=torch.Tensor(1, 1, args.proj_dim))
        self.output_end_projection = torch.nn.Parameter(data=torch.Tensor(args.proj_dim, 1))

        self.initialize_parameters()

    def initialize_parameters(self):
        with torch.no_grad():
            # using the default xavier init function is incorrect.
            # indeed, we split the W matrix into 2 parts in order to reduce computation
            # (trick from the papger of Kiperwasser and Goldberg)
            # the pytorch function will compute the values with the fan_in / 2 instead of the real fan_in
            # torch.nn.init.xavier_uniform_(self.head_projection)
            # torch.nn.init.xavier_uniform_(self.mod_projection)
            fan_in = self.left_projection.size()[1] * 2
            fan_out = self.left_projection.size()[2]

            std = 1.0 * math.sqrt(2.0 / float(fan_in + fan_out))
            a = math.sqrt(3.0) * std  # Calculate uniform bounds from standard deviation
            torch.nn.init.uniform_(self.left_projection, -a, a)
            torch.nn.init.uniform_(self.right_projection, -a, a)

            self.bias_projection.fill_(0.0)
            
            torch.nn.init.xavier_uniform_(self.output_projection)

            torch.nn.init.xavier_uniform_(self.projection_start)
            torch.nn.init.xavier_uniform_(self.output_start_projection)
            self.bias_projection_start.fill_(0.0)

            torch.nn.init.xavier_uniform_(self.projection_end)
            torch.nn.init.xavier_uniform_(self.output_end_projection)
            self.bias_projection_end.fill_(0.0)

    def forward(self, input):
        n_words = input.size()[0]

        # Bigram
        left_proj = input.matmul(self.left_projection)
        right_proj = input.matmul(self.right_projection)

        left_proj = left_proj.view(n_words, 1, -1)
        right_proj = right_proj.view(1, n_words, -1)

        values = left_proj + right_proj
        
        values = values.view(n_words * n_words, -1)
        values = values + self.bias_projection
        values = self.activation(values)
        values = self.proj_dropout(values)

        bigram = values[0].matmul(self.output_projection)
        bigram = bigram.view(n_words, n_words)

        # Start
        start_proj = input.matmul(self.projection_start)

        start_values = start_proj + self.bias_projection_start
        start_values = self.activation(start_values)

        start = start_values[0].matmul(self.output_start_projection).view(n_words)

        # End
        end_proj = input.matmul(self.projection_end)

        end_values = end_proj + self.bias_projection_end
        end_values = self.activation(end_values)

        end = end_values[0].matmul(self.output_end_projection).view(n_words)

        return bigram, start, end
        
class Network(nn.Module):
    def __init__(self, args):
        super(Network, self).__init__()
        self.feature_extractor = FastText(args.fasttext)
        self.permutation = PermutationModule(args, 300)

    def forward(self, input):
        feature = self.feature_extractor(input)
        ret = self.permutation(feature)
        return ret

    @staticmethod
    def add_cmd_options(cmd):
        cmd.add_argument('--proj-dim', type=int, default=128, help="Dimension of the output projection")
        cmd.add_argument('--activation', type=str, default="tanh", help="activation to use in weightning modules: tanh, relu, leaky_relu")

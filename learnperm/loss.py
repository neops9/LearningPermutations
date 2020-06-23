import torch
import math
import random
import torch.nn as nn

class RandomSampler(nn.Module):
    def __init__(self, n_samples):
        super().__init__()
        self.k = n_samples
        self.gumbel = torch.distributions.Gumbel(
            torch.tensor([0.]),
            torch.tensor([1.])
        )

    def forward(self, n_words, bigram, start, end, bigram_bias=None):
        rand = self.gumbel.sample((self.k, n_words)).squeeze()
        rand_perm = rand.argsort(dim=1)
        return rand_perm


class DumbMCMC(nn.Module):
    def __init__(self, n_samples, N=10):
        super().__init__()
        self.chain_size = n_samples * N
        self.N = N
        self.gumbel = torch.distributions.Gumbel(
            torch.tensor([0.]),
            torch.tensor([1.])
        )

    def forward(self, n_words, bigram, start, end, bigram_bias=None):
        with torch.no_grad():
            rand = self.gumbel.sample((self.chain_size, n_words)).squeeze()
            rand_perm = rand.argsort(dim=1)

            # compute weights of samples
            w = bigram[rand_perm[:, :-1], rand_perm[:, 1:]]
            if bigram_bias is not None:
                w = w + bigram_bias[rand_perm[:, :-1], rand_perm[:, 1:]].unsqueeze(-1)
            w = w.sum(dim=1) + start[rand_perm[:, 0]] + end[rand_perm[:, -1]]

            chain = torch.empty((self.chain_size, n_words), dtype=torch.long, device=bigram.device)
            chain[0] = rand_perm[0]
            w_last = w[0]
            for i in range(1, self.chain_size):
                p = min(1., math.exp(w[i].item() - w_last))
                if p > random.uniform(0, 1):
                    chain[i] = rand_perm[i]
                    w_last = w[i].item()
                else:
                    chain[i] = chain[i - 1]

            return chain[torch.arange(0, self.chain_size, self.N) + (self.N - 1)]

"""
class GumbelSampler(nn.Module):
    def __init__(self, n_samples):
        super().__init__()
        self.n_samples = n_samples
        self.gumbel = torch.distributions.Gumbel(
            torch.tensor([0.]),
            torch.tensor([1.])
        )

    def forward(self, n_words, bigram, start, end):
        with torch.no_grad():
            # reshape w. respect to the number of samples
            start = start.unsqueeze(0).expand(self.k, -1)
            bigram = bigram.unsqueeze(0).expand(self.k, -1, -1)

            # add gumbel noise
            start = start + self.gumbel.sample(start.shape).squeeze(-1)
            bigram = bigram + self.gumbel.sample(bigram.shape).squeeze(-1)

            # matrix where we are going to store all permutation samples
            rand_perm = torch.empty((self.n_samples, n_words), dtype=torch.long)

            # sample first word
            pred = start.argmax(dim=1)
            rand_perm[:, 0] = pred

            # for each sample, the word selected as first word cannot be sampled anymore,
            # so we set its weight to -inf so it is never selected by argmax
            arange = torch.arange(self.n_samples)
            bigram[arange, :, pred] = float("-inf")

            for index in range(1, n_words):
                pred = bigram[arange, rand_perm[:, index-1]].argmax(dim=1)
                rand_perm[:, index] = pred
                bigram[arange, :, pred] = float("-inf")

            return rand_perm
"""

class Loss(nn.Module):
    def __init__(self, sampler):
        super().__init__()
        self.sampler = sampler

    def forward(self, bigram, start, end, bigram_bias=None):
        n_words = len(start)
        device = start.device

        samples = self.sampler(n_words, bigram, start, end, bigram_bias)
        n_samples = samples.shape[0]

        start_t = torch.zeros_like(start)
        end_t = torch.zeros_like(end)
        bigram_t = torch.zeros_like(bigram)

        start_t[0] = -1
        end_t[-1] = -1
        arange = torch.arange(n_words, device=device)
        bigram_t[arange[:-1], arange[1:]] = -1

        # we cannot batch over samples because inplace op are buffered
        # and index_add_ is non-deterministic on GPU
        for i in range(samples.shape[0]):
            start_t[samples[i, 0]] += 1 / n_samples
            end_t[samples[i, -1]] += 1 / n_samples
            bigram_t[samples[i, :-1], samples[i, 1:].reshape(-1)] += 1 / n_samples

        loss = torch.sum(start * start_t) + torch.sum(end * end_t) + torch.sum(bigram * bigram_t)
        if bigram_bias is not None:
            loss = loss + torch.sum(bigram_bias * bigram_t)
        return loss, 0


class ISLoss(nn.Module):
    def __init__(self, sampler, combine=False, reverse=False):
        super().__init__()
        self.sampler = sampler
        self.reverse = reverse
        self.combine = combine

    def forward(self, bigram, start, end, bigram_bias=None):
        n_words = len(start)
        device = start.device
        samples = self.sampler(n_words, bigram, start, end, bigram_bias)
        n_samples = samples.shape[0]

        # compute gold score, easy!
        arange = torch.arange(n_words, device=device)
        gold_score = start[0] + end[-1] + torch.sum(bigram[arange[:-1], arange[1:]])
        if bigram_bias is not None:
            gold_score = gold_score + torch.sum(bigram_bias[arange[:-1], arange[1:]])

        # compute sample scores
        # shape: (n batch, n words -1)
        w = bigram[samples[:, :-1], samples[:, 1:]]
        if bigram_bias is not None:
            w = w + bigram_bias[samples[:, :-1], samples[:, 1:]].unsqueeze(-1)

        # add start and end scores
        # shape: n batch
        w = w.sum(dim=1) + start[samples[:, 0]] + end[samples[:, -1]]
        n_worse_than_gold = sum(gold_score > w)

        if self.combine:
            raise RuntimeError("To check")
            log_Z_is = math.log(math.factorial(n_words)) - math.log(n_words) + w.logsumexp(dim=0, keepdim=False)
            log_Z_ris = math.log(math.factorial(n_words)) + math.log(n_words) - (-w).logsumexp(dim=0, keepdim=False)
            log_Z = (log_Z_is + log_Z_ris)/2
        elif self.reverse:
            raise RuntimeError("To check")
            log_Z = math.log(math.factorial(n_words)) + math.log(n_words) - (-w).logsumexp(dim=0, keepdim=False)
        else:
            log_Z = math.log(math.factorial(n_words)) - math.log(n_samples) + w.logsumexp(dim=0, keepdim=False)

        return -gold_score + log_Z, n_worse_than_gold.item()


class DifferentISLoss(nn.Module):
    def __init__(self, sampler):
        super().__init__()
        self.sampler = sampler

    def forward(self, bigram, start, end, bigram_bias=None):
        n_words = len(start)
        device = start.device

        samples = self.sampler(n_words, bigram, start, end, bigram_bias)
        n_samples = samples.shape[0]

        start_t = torch.zeros_like(start)
        end_t = torch.zeros_like(end)
        bigram_t = torch.zeros_like(bigram)

        start_t[0] = -1
        end_t[-1] = -1
        arange = torch.arange(n_words, device=device)
        bigram_t[arange[:-1], arange[1:]] = -1

        sample_ratio = start[samples[:, 0]] \
                       + start[samples[:, -1]] \
                       + bigram[samples[:, :-1], samples[:, 1:]].sum(dim=1)
        sample_ratio /= sample_ratio.sum()

        # we cannot batch over samples because inplace op are buffered
        # and index_add_ is non-deterministic on GPU
        for i in range(samples.shape[0]):
            start_t[samples[i, 0]] += sample_ratio[i]
            end_t[samples[i, -1]] += sample_ratio[i]
            bigram_t[samples[i, :-1], samples[i, 1:].reshape(-1)] += sample_ratio[i]

        loss = torch.sum(start * start_t) + torch.sum(end * end_t) + torch.sum(bigram * bigram_t)
        if bigram_bias is not None:
            loss = loss + torch.sum(bigram_bias * bigram_t)
        return loss, 0
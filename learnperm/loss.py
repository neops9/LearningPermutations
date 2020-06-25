import torch
import math
import numpy as np
import torch.nn as nn
import random

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
            gold = list(range(n_words))

            # compute weights of samples
            w = bigram[rand_perm[:, :-1], rand_perm[:, 1:]]
            if bigram_bias is not None:
                w = w + bigram_bias[rand_perm[:, :-1], rand_perm[:, 1:]].unsqueeze(-1)
            w = w.sum(dim=1) + start[rand_perm[:, 0]] + end[rand_perm[:, -1]]

            chain = torch.empty((self.chain_size, n_words), dtype=torch.long, device=bigram.device)
            chain[0] = rand_perm[0]

            for i in range(1, self.chain_size):
                if chain[0].tolist() != gold:
                    break
                chain[0] = rand_perm[i]

            w_last = w[0]
            for i in range(1, self.chain_size):
                p = min(1., math.exp(w[i].item() - w_last))
                if p > random.uniform(0, 1) and rand_perm[i].tolist() != gold:
                    chain[i] = rand_perm[i]
                    w_last = w[i].item()
                else:
                    chain[i] = chain[i - 1]

            return chain[torch.arange(0, self.chain_size, self.N) + (self.N - 1)]


class BetterMCMC(nn.Module):
    def __init__(self, n_samples, N=10, random_start=False):
        super().__init__()
        self.chain_size = n_samples * N
        self.N = N
        self.gumbel = torch.distributions.Gumbel(
            torch.tensor([0.]),
            torch.tensor([1.])
        )
        self.random_start = random_start

    def _probs(self, bigram, start, end, perm2, target):
        p = np.zeros(n_words)
        p[0] += start[target].item()
        p[-1] += end[target].item()
        for i in range(n_words):
            if i > 0:
                p[i] += bigram[perm2[i - 1], target]
            if i < n_words - 1:
                p[i] += bigram[target, perm2[i]]

        # softmax
        p = np.exp(p - np.max(p))
        p /= p.sum()

        return p

    def forward(self, n_words, bigram, start, end, bigram_bias=None):
        if bigram_bias is not None:
            raise NotImplementedError("Et non! la flemme...")

        with torch.no_grad():
            chain = np.empty((self.chain_size, n_words), dtype=np.int)
            if self.random_start:
                chain[0] = np.random.permutation(n_words)
            else:
                chain[0] = np.arange(n_words)

            # compute weights of init sample
            w_last = bigram[chain[0, :-1], chain[0, 1:]]
            w_last = w_last.sum().item() + start[chain[0, 0]].item() + end[chain[0, -1]].item()

            for i in range(1, self.chain_size):
                to_move_position = np.random.choice(n_words)
                to_move_id = chain[i - 1, to_move_position]
                perm2 = np.concatenate([chain[i - 1, :to_move_position], chain[i - 1, to_move_position + 1:]])

                p = self._probs(bigram, start, end, perm2, to_move_id)
                new_position = np.random.choice(n_words, p=p)

                sample = np.concatenate([perm2[:new_position], [to_move_id], perm2[new_position:]])
                transition_prob = p[new_position] * 1 / n_words

                perm2 = np.concatenate([sample[:new_position], sample[new_position + 1:]])
                p = probs(bigram, start, end, perm2, to_move_id)
                reverse_transition_prob = p[to_move_position] * 1 / n_words

                # compute weights of sample
                new_weight = bigram[sample[:-1], sample[1:]]
                new_weight = new_weight.sum().item() + start[sample[0]].item() + end[sample[-1]].item()

                p = min(1., math.exp(new_weight - w_last) * transition_prob / reverse_transition_prob)
                if p > random.uniform(0, 1):
                    chain[i] = sample
                    w_last = new_weight
                else:
                    chain[i] = chain[i - 1]

            return torch.from_numpy(chain[np.arange(0, self.chain_size, self.N) + (self.N - 1)]).to(bigram.device)

class BigramSampler(nn.Module):
    def __init__(self, n_samples):
        super().__init__()
        self.k = n_samples
        self.s = []

    def findPermutations(self, string, index, n):
        if len(self.s) >= self.k:
            return

        if index >= n or (index + 1) >= n:
            self.s.append(string.copy())
            return

        self.findPermutations(string, index + 1, n)

        string[index], string[index + 1] = string[index + 1], string[index]

        self.findPermutations(string, index + 2, n)

        string[index], string[index + 1] = string[index + 1], string[index]

    def forward(self, n_words, bigram, start, end, bigram_bias=None):
        i = list(range(n_words))
        self.findPermutations(i, 0, n_words)
        return torch.IntTensor(self.s)

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
        w = bigram[samples[:, :-1], samples[:, 1:]]#.unsqueeze(-1)
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
                       + end[samples[:, -1]] \
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

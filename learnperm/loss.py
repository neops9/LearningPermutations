import torch
import math
import torch.nn as nn

class RandomSampler(nn.Module):
    def __init__(self, n_samples):
        super().__init__()
        self.k = n_samples
        self.gumbel = torch.distributions.Gumbel(
            torch.tensor([0.]),
            torch.tensor([1.])
        )

    def forward(self, n_words):
        rand = self.gumbel.sample((self.k, n_words)).squeeze()
        rand_perm = rand.argsort(dim=1)
        return rand_perm

class Loss(nn.Module):
    def __init__(self, sampler):
        super().__init__()
        self.sampler = sampler

    def forward(self, bigram, start, end):
        n_words = len(start)
        device = start.device

        samples = self.sampler(n_words)
        n_samples = samples.shape[0]

        start_t = torch.zeros_like(start)
        end_t = torch.zeros_like(end)
        bigram_t = torch.zeros_like(bigram)

        start_t[0] = -1
        end_t[-1] = -1
        arange = torch.arange(n_words, device=device)
        bigram_t[arange[:-1], arange[1:]] = -1

        gold_score = - (torch.sum(start * start_t) + torch.sum(end * end_t) + torch.sum(bigram * bigram_t)).item()

        # we cannot batch over samples because inplace op are buffered
        # and index_add_ is non-deterministic on GPU
        for i in range(samples.shape[0]):
            start_t[samples[i, 0]] += 1 / n_samples
            end_t[samples[i, -1]] += 1 / n_samples
            bigram_t[samples[i, :-1], samples[i, 1:].reshape(-1)] += 1 / n_samples

        return torch.sum(start * start_t) + torch.sum(end * end_t) + torch.sum(bigram * bigram_t), gold_score


class ISLoss(nn.Module):
    def __init__(self, sampler):
        super().__init__()
        self.sampler = sampler
        self.reverse = False

    def forward(self, bigram, start, end):
        n_words = len(start)
        device = start.device
        samples = self.sampler(n_words)
        n_samples = samples.shape[0]

        # compute gold score, easy!
        arange = torch.arange(n_words, device=device)
        gold_score = start[0] + end[-1] + torch.sum(bigram[arange[:-1], arange[1:]])

        # compute sample scores
        # shape: (n batch, n words -1)
        w = bigram[samples[:, :-1], samples[:, 1:]]

        # add start and end scores
        # shape: n batch
        w = w.sum(dim=1) + start[samples[:, 0]] + end[samples[:, -1]]
        n_worse_than_gold = sum(gold_score > w)

        if self.reverse:
            log_Z = math.log(math.factorial(n_words)) + math.log(n_words) - (-w).logsumexp(dim=0, keepdim=False)
        else:
            log_Z = math.log(math.factorial(n_words)) - math.log(n_words) + w.logsumexp(dim=0, keepdim=False)

        return -gold_score + log_Z, n_worse_than_gold

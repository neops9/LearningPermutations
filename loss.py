import torch

class RandomSampler:
    def __init__(self, n_samples, device="cpu"):
        self.k = n_samples
        self.device = device

    def __call__(self, n_words):
        rand = torch.rand(self.k, n_words).to(self.device)
        rand_perm = rand.argsort(dim=1)
        return rand_perm

class Loss:
    def __init__(self, sampler):
        self.sampler = sampler

    def __call__(self, pred, gold):
        bigram, start, end = pred

        n_words = len(gold)
        samples = self.sampler(n_words)
        n_samples = samples.shape[0]

        start_t = torch.zeros_like(start)
        end_t = torch.zeros_like(end)
        bigram_t = torch.zeros_like(bigram)

        start_t[gold[0]] = -1
        end_t[gold[n_words - 1]] = -1
        bigram_t[gold[:-1], gold[1:]] = -1

        gold_score = - (torch.sum(start * start_t) + torch.sum(end * end_t) + torch.sum(bigram * bigram_t)).item()

        # we cannot batch over samples because inplace op are buffered
        # and index_add_ is non-deterministic on GPU
        for i in range(samples.shape[0]):
            start_t[samples[i, 0]] += 1 / n_samples
            end_t[samples[i, n_words - 1]] += 1 / n_samples
            bigram_t[samples[i, :-1], samples[i, 1:].reshape(-1)] += 1 / n_samples

        return torch.sum(start * start_t) + torch.sum(end * end_t) + torch.sum(bigram * bigram_t), gold_score
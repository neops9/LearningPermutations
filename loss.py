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
        k = self.sampler.k

        start_t = torch.zeros(n_words)
        start_t[gold[0]] = -1

        end_t = torch.zeros(n_words)
        end_t[gold[n_words - 1]] = -1

        bigram_t = torch.zeros((n_words, n_words))
        for i in range(n_words - 1):
            bigram_t[gold[i]][gold[i+1]] = -1

        samples = self.sampler(n_words)

        for s in samples:
            start_t[s[0]] += 1/k
            end_t[s[n_words - 1]] += 1/k

            for i in range(n_words - 1):
                bigram_t[s[i]][s[i+1]] += 1/k

        return torch.sum(start * start_t) + torch.sum(end * end_t) + torch.sum(bigram * bigram_t)
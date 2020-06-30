#cython: language_level=3
import numpy as np
import math
import random
import cython

@cython.boundscheck(False)  # Deactivate bounds checking
@cython.wraparound(False)   # Deactivate negative indexing.
def _probs(long n_words, float[:, :]& bigram, float[:]& start, float[:]& end, long[:]& perm2, int target, float[:]& p):
    p[:] = 0
    p[0] += start[target]
    p[n_words-1] += end[target]

    cdef Py_ssize_t i
    for i in range(n_words):
        if i > 0:
            p[i] += bigram[perm2[i - 1], target]
        if i < n_words - 1:
            p[i] += bigram[target, perm2[i]]

    # softmax
    p -= np.max(p)
    np.exp(p.base, p.base)
    p /= p.base.sum()
    #print("A: ", p.base)
    return p

@cython.boundscheck(False)  # Deactivate bounds checking
@cython.wraparound(False)   # Deactivate negative indexing.
def generate_chain(long n_words, long n_samples, long N, random_start, bigram, start, end):
    cdef float[:, :] bigram_view = bigram
    cdef float[:] start_view = start
    cdef float[:] end_view = end

    cdef long[:, :] chain = np.empty((n_samples, n_words), dtype=int)
    cdef long[:] mcmc_head
    if random_start:
        mcmc_head = np.random.permutation(n_words, dtype=int)
    else:
        mcmc_head = np.arange(n_words, dtype=int)

    # compute weights of init sample
    cdef float w_last = bigram_view.base[mcmc_head[0:n_words-1], mcmc_head[1:n_words]].sum()
    w_last += start_view[mcmc_head[0]] + end_view[mcmc_head[n_words-1]]

    cdef long[:] perm2 = np.empty(n_words - 1, dtype=int)
    cdef long[:] sample = np.empty(n_words, dtype=int)
    cdef float[:] p = np.empty(n_words, dtype=np.float32)
    cdef float[:] p2 = np.empty(n_words, dtype=np.float32)

    cdef Py_ssize_t i, j
    cdef long to_move_position
    cdef long to_move_id
    cdef long new_position
    for i in range(n_samples):
        for j in range(N):
            to_move_position = np.random.choice(n_words)
            to_move_id = mcmc_head[to_move_position]
            perm2[:to_move_position] = mcmc_head[:to_move_position]
            perm2[to_move_position:] = mcmc_head[to_move_position + 1:]

            # p is not updated even if it is passed by ref
            p = _probs(n_words, bigram_view, start_view, end_view, perm2, to_move_id, p)
            # choice() is slow, so we use the gumbel-max trick instead
            #new_position = np.random.choice(n_words, p=p)
            #print("B: ", p.base)
            np.log(p.base, p2.base)
            p2 += np.random.gumbel(size=n_words).astype('f4')
            new_position = p2.base.argmax()

            sample[:new_position] = perm2[:new_position]
            sample[new_position] = to_move_id
            sample[new_position+1:] = perm2[new_position:]
            transition_prob = p[new_position] * 1 / n_words
            reverse_transition_prob = p[to_move_position] * 1 / n_words

            # compute weights of sample
            new_weight = bigram[sample[:n_words-1], sample[1:]]
            new_weight = new_weight.sum().item() + start[sample[0]].item() + end[sample[n_words-1]].item()

            limit = min(1., math.exp(new_weight - w_last) * transition_prob / reverse_transition_prob)
            if limit > random.uniform(0, 1):
                mcmc_head = sample
                w_last = new_weight
            else:
                pass # keep the same
        chain[i] = mcmc_head

    return chain.base

@cython.boundscheck(False)  # Deactivate bounds checking
@cython.wraparound(False)   # Deactivate negative indexing.
def two_opt(long n_words, long n_samples, long N, bigram, start, end):
    cdef float[:, :] bigram_view = bigram
    cdef float[:] start_view = start
    cdef float[:] end_view = end

    cdef long[:, :] chain = np.empty((n_samples, n_words), dtype=int)

    # compute weights of init sample
    #cdef float w_last = bigram_view.base[mcmc_head[0:n_words-1], mcmc_head[1:n_words]].sum()
    #w_last += start_view[mcmc_head[0]] + end_view[mcmc_head[n_words-1]]

    cdef long[:] sample = np.random.shuffle(np.arange(n_words, dtype=int))

    cdef Py_ssize_t i, j, k
    cdef float min_change, change
    cdef int min_i, min_j

    for k in range(n_samples):
        sample = np.arange(n_words, dtype=int)
        np.random.shuffle(sample)
        min_change = 0

        # Find the best move
        for i in range(n_words - 2):
            for j in range(i + 2, n_words - 1):
                change = bigram_view[sample[i], sample[j]] + bigram_view[sample[i+1], sample[j+1]]
                change = - bigram_view[sample[i], sample[i+1]] - bigram_view[sample[j], sample[j+1]]
                if change < min_change:
                    min_change = change
                    min_i, min_j = i, j

        # Update tour with best move
        if min_change < 0:
            sample[min_i+1:min_j+1] = sample[min_i+1:min_j+1][::-1]

        chain[k] = sample
    return chain.base
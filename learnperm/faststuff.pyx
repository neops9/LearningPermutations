#cython: language_level=3
import numpy as np
import math
import random
import cython
cimport numpy as np
cimport cython
np.import_array()
from cython.parallel import parallel, prange
from libcpp.vector cimport vector
ctypedef np.int_t DTYPE_int
cdef extern from "math.h":
    float INFINITY
from cpython cimport array
import array

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

# Adapted from https://github.com/lucventurini/pytritex/blob/fc137ed36902e2300cf26207fed333019a30dad9/pytritex/graph_utils/k_opt_tsp.pyx

@cython.wraparound(False)
@cython.boundscheck(False)
cdef double c_sample_cost(double[:, :] bigram_view, vector[long] sample) nogil:
    cdef:
        double cost
        long shape = sample.size()
        long index = shape - 1

    cost = 0
    for index in prange(shape - 1):
        cost += bigram_view[sample[index]][sample[index + 1]]

    return cost

@cython.wraparound(False)
@cython.boundscheck(False)
cdef vector[long] _swap(long[:] sample_array, long index, long kindex) nogil:
    cdef:
        vector[long] new_sample
        long sample_it
        long internal_index
        long mirror
        long[:] to_swap

    for sample_it in prange(sample_array.shape[0]):
        new_sample.push_back(sample_array[sample_it])

    to_swap = sample_array[index:kindex + 1]
    cdef long swap_length = kindex - index + 1
    for internal_index in range(0, swap_length, 1):
        mirror = swap_length - internal_index - 1
        new_sample[index + internal_index] = to_swap[mirror]
    return new_sample

@cython.boundscheck(False)  # Deactivate bounds checking
@cython.wraparound(False)   # Deactivate negative indexing.
def local_search(long n_words, bigram, start, end):
    cdef float[:] start_view = start
    cdef float[:] end_view = end

    cdef bint improved = 1
    cdef double best_cost
    cdef double cost
    cdef long index
    cdef long kindex
    cdef long max_index
    cdef long sample_shape
    cdef double[:, :] bigram_array = bigram.astype(float)

    sample = np.arange(n_words, dtype=int)
    np.random.shuffle(sample)
    sample = np.array(sample)

    cdef long[:] sample_array = sample

    cdef long sample_it
    cdef vector[long] best_found_sample
    cdef vector[long] new_sample
    cdef long swap_length
    cdef np.ndarray[DTYPE_int, ndim=1] final_sample

    max_index = sample_array.shape[0] - 1
    for sample_it in range(sample_array.shape[0]):
        best_found_sample.push_back(sample_array[sample_it])

    best_cost = c_sample_cost(bigram_array, best_found_sample) + start_view[best_found_sample[0]] + end_view[best_found_sample[n_words-1]]
    while improved == 1:
        improved = 0
        for index in range(1, max_index):
            for kindex in range(index + 1, max_index):
                # Swap internally between index and kindex
                with nogil, parallel():
                    new_sample = _swap(sample_array, index, kindex)
                cost = c_sample_cost(bigram_array, new_sample) + start_view[new_sample[0]] + end_view[new_sample[n_words-1]]
                if cost > best_cost:
                    best_cost = cost
                    best_found_sample = new_sample
                    improved = 1
            if improved:
                break

    return np.array(best_found_sample[:])

@cython.boundscheck(False)  # Deactivate bounds checking
@cython.wraparound(False)   # Deactivate negative indexing.
def two_opt_fast(long n_words, long n_samples, long N, bigram, start, end):
    chain = np.empty((n_samples, n_words), dtype=int)

    for k in range(n_samples):
        res = local_search(n_words, bigram, start, end)
        chain[k] = res

    return chain
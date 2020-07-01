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

@cython.boundscheck(False)  # Deactivate bounds checking
@cython.wraparound(False)   # Deactivate negative indexing.
def two_opt(long n_words, long n_samples, long N, bigram, start, end):
    cdef float[:, :] bigram_view = bigram
    cdef float[:] start_view = start
    cdef float[:] end_view = end

    cdef long[:, :] chain = np.empty((n_samples, n_words), dtype=int)

    cdef long[:] sample = np.arange(n_words, dtype=int)
    np.random.shuffle(sample)

    cdef Py_ssize_t i, j, k
    cdef float change
    improved = True

    for k in range(n_samples):
        sample = np.arange(n_words, dtype=int)
        np.random.shuffle(sample)

        improved = True
        p = 0
        while improved and p < 20:
            improved = False
            p += 1
            for i in range(1, n_words - 2):
                for j in range(i + 1, n_words):
                    if j - i == 1:
                        continue

                    new_sample = sample[:] # Creates a copy of the sample
                    new_sample[i:j] = sample[j - 1:i - 1:-1]

                    change = bigram_view[sample[i-1], sample[j-1]] + bigram_view[sample[j-2], sample[i]]
                    change -= bigram_view[sample[i-1], sample[i]] + bigram_view[sample[j-2], sample[j-1]]

                    if change > 0:
                        sample = new_sample
                        improved = True    

        print(sample)
        print(type(sample))
        chain[k] = sample

    return chain.base

@cython.boundscheck(False)  # Deactivate bounds checking
@cython.wraparound(False)   # Deactivate negative indexing.
def two_opt_bis(long n_words, long n_samples, long N, bigram, start, end):

    cdef float[:, :] bigram_view = bigram
    cdef float[:] start_view = start
    cdef float[:] end_view = end

    cdef long[:, :] chain = np.empty((n_samples, n_words), dtype=int)

    cdef long[:] sample = np.arange(n_words, dtype=int)
    np.random.shuffle(sample)

    cdef float min_change, change
    cdef int i, j, min_i, min_j

    for k in range(n_samples):
        sample = np.arange(n_words, dtype=int)
        np.random.shuffle(sample)
        last_cost = bigram_view.base[sample[0:n_words-1], sample[1:n_words]].sum() + start_view[sample[0]] + end_view[sample[n_words-1]]

        min_change = 0

        improved = True
        while improved:
            improved = False
            # Find the best move
            for i in range(n_words - 2):
                for j in range(i + 2, n_words - 1):
                    change = bigram_view[sample[i], sample[j]] + bigram_view[sample[i+1], sample[j+1]]
                    change -= bigram_view[sample[i], sample[i+1]] + bigram_view[sample[j], sample[j+1]]
                    if change > min_change:
                        min_change = change
                        min_i, min_j = i, j
            # Update tour with best move
            if min_change > 0:
                improved = False
                sample[min_i+1:min_j+1] = sample[min_i+1:min_j+1][::-1]


        chain[k] = sample

    return chain.base

@cython.wraparound(False)
@cython.boundscheck(False)
cdef double c_route_cost(double[:, :] graph, vector[long] path) nogil:
    cdef:
        double cost
        long shape = path.size()
        double temp_cost
        long index = shape - 1
        long second = 0

    cost = graph[path[index], path[0]]
    for index in prange(shape - 1):
        second = index + 1
        temp_cost = graph[path[index]][path[second]]
        if temp_cost == 0:
            return 0
        else:
            cost += temp_cost

    return cost


cpdef float route_cost(np.ndarray graph, np.ndarray path):

    cdef np.ndarray[double, ndim=2, mode="c"] graph_array = graph.astype(float)
    cdef np.ndarray[long, ndim=1, mode="c"] path_array = path
    return c_route_cost(graph_array, path_array)

@cython.wraparound(False)
@cython.boundscheck(False)
cdef vector[long] _swap(long[:] route_array, long index, long kindex) nogil:
    cdef:
        vector[long] new_route
        long route_it
        long internal_index
        long mirror
        long[:] to_swap

    for route_it in prange(route_array.shape[0]):
        new_route.push_back(route_array[route_it])

    to_swap = route_array[index:kindex + 1]
    cdef long swap_length = kindex - index + 1
    for internal_index in range(0, swap_length, 1):
        mirror = swap_length - internal_index - 1
        new_route[index + internal_index] = to_swap[mirror]
    return new_route


@cython.boundscheck(False)  # Deactivate bounds checking
@cython.wraparound(False)   # Deactivate negative indexing.
def local_search(long n_words, bigram, start, end):
    """
    Approximate the optimal path of travelling salesman according to 2-opt algorithm
    Args:
        graph: 2d numpy array as graph
        route: list of nodes
    Returns:
        optimal path according to 2-opt algorithm
    Examples:
        >>> import numpy as np
        >>> graph = np.array([[  0, 300, 250, 190, 230],
        >>>                   [300,   0, 230, 330, 150],
        >>>                   [250, 230,   0, 240, 120],
        >>>                   [190, 330, 240,   0, 220],
        >>>                   [230, 150, 120, 220,   0]])
        >>> tsp_2_opt(graph)
    """
    cdef float[:] start_view = start
    cdef float[:] end_view = end

    #print(n_words)
    

    cdef bint improved = 1
    cdef double best_cost
    cdef double cost
    cdef long index
    cdef long kindex
    cdef long max_index
    cdef long route_shape
    cdef double[:, :] graph_array = bigram.astype(float)

    sample = np.arange(n_words, dtype=int)
    np.random.shuffle(sample)
    sample = np.array(sample)
    cdef long[:] route_array = sample

    cdef long route_it
    cdef vector[long] best_found_route
    cdef vector[long] new_route
    cdef long swap_length
    cdef np.ndarray[DTYPE_int, ndim=1] final_route

    max_index = route_array.shape[0] - 1
    for route_it in range(route_array.shape[0]):
        best_found_route.push_back(route_array[route_it])

    best_cost = c_route_cost(graph_array, best_found_route) + start_view[best_found_route[0]] + end_view[best_found_route[n_words-1]]
    while improved == 1:
        improved = 0
        for index in range(1, max_index):
            for kindex in range(index + 1, max_index):
                # Swap internally between index and kindex
                with nogil, parallel():
                    new_route = _swap(route_array, index, kindex)
                cost = c_route_cost(graph_array, new_route) + start_view[new_route[0]] + end_view[new_route[n_words-1]]
                if cost > best_cost:
                    best_cost = cost
                    best_found_route = new_route
                    improved = 1
            if improved:
                break

    #print(np.expand_dims(np.array(best_found_route[:]), axis=0))
    final_route = np.array(best_found_route[:])
    #print(final_route)
    #print(type(final_route))
    #chain[k] = final_route

    return final_route

@cython.boundscheck(False)  # Deactivate bounds checking
@cython.wraparound(False)   # Deactivate negative indexing.
def two_opt_fast(long n_words, long n_samples, long N, bigram, start, end):
    chain = np.empty((n_samples, n_words), dtype=int)

    for k in range(n_samples):
        res = local_search(n_words, bigram, start, end)
        #print(res)
        #print(type(res))
        chain[k] = res

    return chain
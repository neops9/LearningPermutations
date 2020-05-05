import numpy as np

def eval(preds, golds):
    total_bigram = 0
    correct_bigram = 0
    correct_start = 0
    correct_end = 0
    n_sentences = len(preds)
    for p, g in zip(preds, golds):
        p_bigram, p_start, p_end = p

        for line_p, line_g in zip(p_bigram, g):
            total_bigram += 1
            for w_p, w_g in zip(line_p, line_g):
                if w_g == 1 and w_p >= 0:
                    correct_bigram += 1
                
        if p_start[0] >= 0:
            correct_start += 1

        if p_end[0] >= 0:
            correct_end += 1

    return correct_bigram/total_bigram, correct_start/n_sentences, correct_end/n_sentences
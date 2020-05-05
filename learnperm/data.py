import os
import io
import learnperm.conll as conll
import torch

def load_embeddings(dir, langs):
    word_to_id = dict()
    id_to_word = list()
    n_embs = 0
    data = list()
    for lang in langs:
        path = os.path.join(dir, lang + ".vec")
        word_to_id[lang] = dict()
        fin = io.open(path, 'r', encoding='utf-8', newline='\n', errors='ignore')
        # ignore first list
        fin.readline().split()
        for line in fin:
            tokens = line.rstrip().split(' ')
            data.append(list(map(float, tokens[1:])))
            word_to_id[lang][tokens[0]] = n_embs
            id_to_word.append((lang, tokens[0]))
            n_embs += 1
    return data, word_to_id, id_to_word, n_embs


def read_conllu(dir, langs, section, word_to_id, unk_idx, min_length=2, device="cpu"):
    ret = list()
    for lang in langs:
        path = os.path.join(dir, lang, section + ".conllu")
        data = conll.read(path, "conllu")
        for sentence in data:
            words = [w["form"] for w in sentence["tokens"]]
            if len(words) >= min_length:
                words_tensor = torch.LongTensor([word_to_id[lang].get(w.lower(), unk_idx) for w  in words])
                words_tensor = words_tensor.to(device)
                ret.append(words_tensor)
    return ret

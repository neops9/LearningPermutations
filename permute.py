import argparse
import shutil
import torch
import sys
import io
import os
import subprocess
from learnperm import loss, network
import time
import torch.nn as nn
import learnperm.conll as conll
import numpy as np
import re
from learnperm.dict import build_tags_dictionnary
import learnperm.special_tokens
from scipy.stats import kendalltau

def read_conllu(path, word_to_id, unk_idx, tags_dict, min_length=2, device="cpu"):
    ret = list()
    data = conll.read(path, "conllu")
    for sentence in data:
        words = [w["form"] for w in sentence["tokens"]]
        tags = [w["upos"] for w in sentence["tokens"]]
        in_data = dict()
        special_words = learnperm.special_tokens.Dict(word_to_id[args.lang], unk_idx)

        if len(words) >= min_length:
            words_tensor = torch.LongTensor([word_to_id[args.lang].get(w.lower(), unk_idx) for w in words])
            words_tensor = words_tensor.to(device)
            in_data["tokens"] = words_tensor

            special_words_tensor = torch.LongTensor([special_words.to_id(w) for w in words])
            special_words_tensor = special_words_tensor.to(device)
            in_data["special_words"] = special_words_tensor

            tags_tensor = torch.LongTensor([tags_dict.word_to_id(t) for t in tags])
            tags_tensor = tags_tensor.to(device)
            in_data["tags"] = tags_tensor

            in_data["multiwords"] = sentence["multiwords"]

            ret.append(in_data)

    return ret

def load_embeddings(path, lang):
    word_to_id = dict()
    id_to_word = list()
    n_embs = 0
    data = list()
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

# Read command line
parser = argparse.ArgumentParser()
parser.add_argument('--model', type=str, required=True)
parser.add_argument('--data', type=str, required=True)
parser.add_argument('--lang', type=str, required=True)
parser.add_argument('--embeddings', type=str, required=True, help="Path to the embedding folder")
parser.add_argument('--keep-mwe', action="store_true", help="Keep multi-word expressions order")
parser.add_argument('--program', type=str, required=True)
parser.add_argument("--algo-version", type=str, default="n6")
parser.add_argument("--format", type=str, default="conllu")
parser.add_argument('--storage-device', type=str, default="cpu", help="Device where to store the data")
parser.add_argument('--device', type=str, default="cpu", help="Device to use for computation")
parser.add_argument('--verbose', action="store_true", help="")
parser.add_argument('--bias', type=float, default=0.0, help="Permutation bias")
parser.add_argument('--output', type=str, default="output", help="Output file")
parser.add_argument('--left-limit', type=str, default="-1")
parser.add_argument('--right-limit', type=str, default="-1")
parser.add_argument('--name', type=str, default="temp")
args = parser.parse_args()

print("Loading vocabulary and embeddings", file=sys.stderr, flush=True)
embeddings_table, word_to_id, id_to_word, unk_idx = load_embeddings(args.embeddings, args.lang)
checkpoint = torch.load(args.model, map_location=args.device)
checkpoint['args'].device = args.device
print("Loading test data", file=sys.stderr, flush=True)
test_data = read_conllu(args.data, word_to_id, unk_idx, checkpoint["tags_dict"], device=args.storage_device)
test_size = len(test_data)

print("Loading model", flush=True)
model = network.Network(checkpoint['args'], embeddings_table=embeddings_table, add_unk=True, word_padding_idx=unk_idx, pos_padding_idx=checkpoint["tags_dict"].pad_index, n_tags=len(checkpoint["tags_dict"]._word_to_id.keys()))
model_state = model.state_dict()
model_state.update(checkpoint['state_dict'])
model.load_state_dict(model_state)
model.to(device=args.device)
model.eval()

def fix_mwe(multiwords, bigram, c=1000):
    for mw in multiwords:
        for i in range(mw["begin"], mw["end"]):
            bigram[i-1, i] = c
    return bigram

# Write program input file
input_filename = "temp_input_" + args.name
output_filename = "temp_output_" + args.name

starts = list()
ends = list()
bigrams = list()

for i, sentence in enumerate(test_data):
    if len(sentence["tokens"]) < 2:
        continue

    n_words = len(sentence["tokens"])

    batch_bigram, batch_start, batch_end = model([sentence])
    batch_bigram = batch_bigram[0].detach().numpy()
    batch_start = batch_start[0].detach().numpy()
    batch_end = batch_end[0].detach().numpy()

    if args.keep_mwe:
        batch_bigram = fix_mwe(sentence["multiwords"], batch_bigram)

    starts.append(batch_start)
    ends.append(batch_end)
    bigrams.append(batch_bigram)

# add bias
for i in range(len(bigrams)):
    arange = np.arange(bigrams[i].shape[0])
    bigrams[i][arange[:-1], arange[1:]] += args.bias

infile = open(input_filename, 'w')

for s, e, b in zip(starts, ends, bigrams):
    infile.write(str(len(s)) + " ")

    for v in s.flatten():
        infile.write(str(v) + " ")

    for v in e.flatten():
        infile.write(str(v) + " ")

    for v in b.flatten():
        infile.write(str(v) + " ")

    infile.write("\n")

infile.close()

# Run the re-ordering algorithm
cmd = [args.program, "--input", input_filename, "--algorithm", args.algo_version, "--left-limit", args.left_limit, "--right-limit", args.right_limit, "--output", output_filename]

if args.verbose:
    cmd.append("--verbose")

proc = subprocess.check_output(cmd)
print("Re-ordering output:", proc, file=sys.stderr, flush=True)

# Regenerate conll file base on the program output
def build_conll(path, orders):
    data = conll.read(path, "conllu")

    i = 0
    for sentence in data:
        if len(sentence["tokens"]) < 2:
            continue

        order = orders[i]
        order = order.split()
        new_sentence = sentence.copy()
        i += 1

        for new_token, token in zip(new_sentence["tokens"], sentence["tokens"]):
            if token["head"] > 0:
                new_token["head"] = order.index(str(token["head"] - 1)) + 1

        new_sentence["tokens"] = [sentence["tokens"][int(i)] for i in order]
        new_sentence["multiwords"] =  []
        yield new_sentence

outfile = open(output_filename, 'r') 
new_orders = outfile.readlines()
outfile.close()

output_path = args.output
conll.write(output_path, build_conll(args.data, new_orders), format=args.format)

os.remove(input_filename)
os.remove(output_filename)
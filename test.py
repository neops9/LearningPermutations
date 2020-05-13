import argparse
import shutil
import torch
import sys
import os
import subprocess
from learnperm import loss, network
import time
import torch.nn as nn
from learnperm.data import load_embeddings, read_conllu
import learnperm.conll as conll

# Read command line
parser = argparse.ArgumentParser()
parser.add_argument('--model', type=str, required=True)
parser.add_argument('--data', type=str, required=True)
parser.add_argument('--test-langs', type=str, required=True, help="Comma separated list of languages")
parser.add_argument('--embeddings', type=str, required=True, help="Path to the embedding folder")
parser.add_argument('--program', type=str, required=True)
parser.add_argument("--algo-version", type=str, default="n6")
parser.add_argument("--format", type=str, default="conllu")
parser.add_argument('--storage-device', type=str, default="cpu", help="Device where to store the data")
parser.add_argument('--device', type=str, default="cpu", help="Device to use for computation")
parser.add_argument('--verbose', action="store_true", help="")
args = parser.parse_args()

test_langs = args.test_langs.split(",")
if len(test_langs) == 0:
    raise RuntimeError("No test languages")

all_langs = set(test_langs)
print("Test langs: ", all_langs, file=sys.stderr, flush=True)

print("Loading vocabulary and embeddings", file=sys.stderr, flush=True)
embeddings_table, word_to_id, id_to_word, unk_idx = load_embeddings(args.embeddings, all_langs)

print("Loading test data", file=sys.stderr, flush=True)
test_data = read_conllu(args.data, test_langs, "test", word_to_id, unk_idx, device=args.storage_device)
test_size = len(test_data)

print("Loading model", flush=True)
checkpoint = torch.load(args.model)
model = network.Network(checkpoint['args'], embeddings_table=embeddings_table, add_unk=True)
model_state = model.state_dict()
model_state.update(checkpoint['state_dict'])
model.load_state_dict(model_state)
model.to(device=args.device)
model.eval()

lang = test_langs[0]

# Write program input file
input_filename = "temp_input"
output_filename = "temp_output"

infile = open(input_filename, 'w')

for sentence in test_data:
    n_words = len(sentence)
    batch_bigram, batch_start, batch_end = model(sentence.unsqueeze(0))

    infile.write(str(n_words) + " ")

    for v in batch_start[0].flatten().detach().numpy():
        infile.write(str(v) + " ")

    for v in batch_end[0].flatten().detach().numpy():
        infile.write(str(v) + " ")

    for v in batch_bigram[0].flatten().detach().numpy():
        infile.write(str(v) + " ")

    infile.write("\n")

infile.close()

# Run the re-ordering algorithm
cmd = [args.program, "--input", input_filename, "--algorithm", args.algo_version, "--output", output_filename]

if args.verbose:
    cmd.append("--verbose")

proc = subprocess.Popen(cmd).wait()

# Regenerate conll file base on the program output
def build_conll(path, orders):
    data = conll.read(path, "conllu")

    for sentence, order in zip(data, orders):
        if len(sentence["tokens"]) < 2 or not (len(sentence["tokens"]) == len(order)):
            continue

        order = order.split()
        new_sentence = sentence.copy()

        for new_token, token in zip(new_sentence["tokens"], sentence["tokens"]):
            if token["head"] > 0 and str(token["head"] - 1) in order:
                    new_token["head"] = order.index(str(token["head"] - 1)) + 1

        new_sentence["tokens"] = [sentence["tokens"][int(i)] for i in order]
        yield new_sentence

outfile = open(output_filename, 'r') 
new_orders = outfile.readlines()
outfile.close()

input_path = os.path.join(args.data, lang, "test.conllu")
output_path = os.path.join(args.data, lang, "output.conllu")
conll.write(output_path, build_conll(input_path, new_orders), format=args.format)
os.remove(input_filename)
os.remove(output_filename)


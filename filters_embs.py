import sys
import conll
import io

in_embs = sys.argv[1]
out_embs = sys.argv[2]
in_conllus = sys.argv[3:]

corpus_words = set()
for path in in_conllus:
    for sentence in conll.read(path, format="conllu"):
        corpus_words.update([w["form"].lower() for w in sentence["tokens"]])


with io.open(in_embs, 'r', encoding='utf-8', newline='\n', errors='ignore') as fin:
    n, d = map(int, fin.readline().split())
    data = {}
    for line in fin:
        token = line.rstrip().split(' ')[0]
        if token in corpus_words:
            data[token] = line

with io.open(out_embs, 'w', encoding='utf-8', newline='\n', errors='ignore') as fout:
    fout.write("%i %i" % (len(data), d))
    for line in data.values():
        fout.write(line)

print("N words in input data: %i" % len(corpus_words))
print("N words mached in embeddings: %i" % len(data))

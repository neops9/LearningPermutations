mkdir embeddings

wget https://dl.fbaipublicfiles.com/fasttext/vectors-aligned/wiki.en.align.vec
python filter_embs.py wiki.en.align.vec embeddings/en_gum.vec data/en_gum/*.conllu
rm wiki.en.align.vec

wget https://dl.fbaipublicfiles.com/fasttext/vectors-aligned/wiki.fr.align.vec
python filter_embs.py wiki.fr.align.vec embeddings/fr_ftb.vec data/fr_ftb/*.conllu
rm wiki.fr.align.vec

wget https://dl.fbaipublicfiles.com/fasttext/vectors-aligned/wiki.de.align.vec
python filter_embs.py wiki.de.align.vec embeddings/de_gsd.vec data/de_gsd/*.conllu
rm wiki.de.align.vec

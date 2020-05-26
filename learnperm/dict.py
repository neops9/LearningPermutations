import torch
import learnperm.conll as conll

class Dict:
    def __init__(self, words, unk=None, boundaries=False, pad=False):
        self._boundaries = boundaries
        self._unk = unk
        self._word_to_id = dict()
        self._id_to_word = list()

        if pad:
            if "**PAD**" in words:
                raise RuntimeError("Pad is already in dict")
            self.pad_index = self._add_word("**PAD**")

        if boundaries:
            if "**BOS**" in words or "**EOS**" in words:
                raise RuntimeError("Boundaries ids are already in dict")
            self._bos = self._add_word("**BOS**")
            self._eos = self._add_word("**EOS**")

        if unk in words:
            raise RuntimeError("UNK word exists in vocabulary")

        if unk is not None:
            self.unk_index = self._add_word(unk)

        for word in words:
            self._add_word(word)

    # for internal use only!
    def _add_word(self, word):
        id = len(self._id_to_word)
        self._word_to_id[word] = id
        self._id_to_word.append(word)
        return id

    def contains(self, word):
        return word in self._word_to_id

    def word_to_id(self, word):
        if self._unk is not None:
            return self._word_to_id.get(word, self.unk_index)
        else:
            return self._word_to_id[word]

    def id_to_word(self, id):
        return self._id_to_word[id]

    def __len__(self):
        return len(self._word_to_id)

    def has_unk(self):
        return self._unk is not None

    def has_boundaries(self):
        return self._boundaries

    def bos_id(self):
        return self._bos

    def eos_id(self):
        return self._eos


def build_tags_dictionnary(path, boundaries=False, char_boundaries=False, postag="upos"):
    dict_tags = set()

    data = conll.read(path, "conllu")
    for sentence in data:
        for tokens in sentence["tokens"]:
            if postag in tokens.keys():
                dict_tags.add(tokens[postag])

    dict_tags = Dict(dict_tags, boundaries=boundaries, unk="UNK", pad=True)
    return dict_tags

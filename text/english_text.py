"""English text processing"""


_character_set = "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz!\'(),-.:;? "
_pad = "_PAD_"
_eos = "_EOS_"
_unk = "_UNK_"

_vocab = [_pad, _eos, _unk] + list(_character_set)

char2idx = {char: idx for idx, char in enumerate(_vocab)}
idx2char = {idx: char for idx, char in enumerate(_vocab)}


def load_vocab():
    """Returns the char2index mapping
    """
    return char2idx


def normalize_text(textpath):
    """Read text from file; normalize it and return it as a sequence of charaters
    """
    with open(textpath, "r") as fp:
        text = fp.readlines()
    text = " ".join(text)
    text = text.strip("\n")

    return text


def text_to_sequence(text):
    """Convert text to sequence of indices corresponding to characters in the text
    """
    sequence = [char2idx[char] if char in char2idx else char2idx["_UNK_"] for char in text]

    return sequence

"""
Prepare a counting dataset, digits in [0, 9] with ONLY monotonically increasing sequences,
with the exception of a transtion from 9-back-to-0.
"""
import os
import pickle
from typing import Literal
from copy import deepcopy
import numpy as np
from numpy.typing import NDArray



sequence_length = 1024

## construct a dataset of all possible sequences in [0, 9] w transtions back to 0 from 9 @ a given sequence length
digits = list(range(10))
def generate_dataset(sequence_length: int) -> NDArray:
    remainder = sequence_length % len(digits)
    base_seq = []
    for s in range(sequence_length // len(digits)):
        base_seq.extend(digits)
    if remainder:
        base_seq.extend(digits[:remainder])
    print(base_seq)
    # then get all sequences, 10 variations from different starting states
    data: list = [base_seq]
    for i in digits[:-1]:
        lhs = base_seq[:i+1]
        sample = deepcopy(base_seq[i+1:])
        sample.extend(lhs)
        data.append(sample)
    return np.array(data)

data = generate_dataset(sequence_length=sequence_length)
print(f"n_samples in dataset: {data.shape[0]:,}")
print(f"sequence len per sample : {data.shape[1]:,}")

# get all the unique characters that occur in this text
ValidDigits = list[Literal[0, 1, 2, 3, 4, 5, 6, 7, 8, 9]] | NDArray
vocab_size = len(digits)
print("all the unique digits:", ''.join([str(d) for d in digits]))
print(f"vocab size: {vocab_size:,}")


## match nanoGPT tokenizer handles, basically pass through fns
digit_to_token = { d:i for i,d in enumerate(digits) }
def encode(digit_seq: ValidDigits):
    return digit_seq # encoder: tokens for model = the "prompt" or digit sequence in this case
def decode(token_seq: ValidDigits):
    return token_seq # decoder: no diff here either


## create the train and test splits, 1 sample for validation...
n = len(data)
train_data = data[:int(n*0.9), :]
val_data = data[int(n*0.9):, :]


## get a count of the tokens, that is positions for which we can supervise the model
train_ids = encode(train_data)
val_ids = encode(val_data)
print(f"train has {train_ids.size:,} tokens")
print(f"val has {val_ids.size:,} tokens")


## export to bin files
train_ids = np.array(train_ids, dtype=np.uint16)
val_ids = np.array(val_ids, dtype=np.uint16)
train_ids.tofile(os.path.join(os.path.dirname(__file__), 'train.bin'))
val_ids.tofile(os.path.join(os.path.dirname(__file__), 'val.bin'))


## save the meta information as well, to help us encode/decode later
meta = {
    'vocab_size': vocab_size,
    # these two are the same for a countformer
    'itos': digit_to_token,
    'stoi': digit_to_token,
}
with open(os.path.join(os.path.dirname(__file__), 'meta.pkl'), 'wb') as f:
    pickle.dump(meta, f)

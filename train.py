import argparse
import shutil
import random
import torch
from learnperm import conll

import network
import loss
import io
import os
import torch.nn as nn
from learnperm.data import load_embeddings, read_conllu
from learnperm.batch import KMeansBatchIterator

def save_checkpoint(state, is_best, path, filename='checkpoint.pth.tar'):
    torch.save(state, path + filename)
    if is_best:
        shutil.copyfile(path + filename, path + 'model_best.pth.tar')

# Read command line
cmd = argparse.ArgumentParser()
cmd.add_argument("--data", type=str, required=True, help="Path to training data folder")
cmd.add_argument('--embeddings', type=str, required=True, help="Path to the embedding folder")
cmd.add_argument('--train-langs', type=str, required=True, help="Comma separated list of languages")
cmd.add_argument('--dev-langs', type=str, required=True, help="Comma separated list of languages")
cmd.add_argument("--model", type=str, required=True, help="Path where to store the model")
cmd.add_argument("--format", type=str, default="conllu")
cmd.add_argument("--lr", type=float, default=0.05)
cmd.add_argument("--samples", type=int, default=10000, help="Number of samples")
cmd.add_argument("--epochs", type=int, default=500, help="Number of epochs for training")
cmd.add_argument("--batch", type=int, default=1, help="Mini-batch size")
cmd.add_argument('--storage-device', type=str, default="cpu", help="Device to use for data storage")
cmd.add_argument('--device', type=str, default="cpu", help="Device to use for computation")
cmd.add_argument('--resume', type=str, default="", help="Resume training from a saved model.")
cmd.add_argument('--batch-size', type=int, default=2500, help="Maximum number of words per batch")
cmd.add_argument('--batch-clusters', type=int, default=32, help="Number of clusters to use to construct batches")
network.Network.add_cmd_options(cmd)
args = cmd.parse_args()

train_langs = args.train_langs.split(",")
dev_langs = args.dev_langs.split(",")
if len(train_langs) == 0 or len(dev_langs) == 0:
    raise RuntimeError("No train/dev languages")
all_langs = train_langs + dev_langs

print("Loading vocabulary and embeddings", flush=True)
embeddings_table, word_to_id, id_to_word, unk_idx = load_embeddings(args.embeddings, all_langs)

print("Loading train and dev data", flush=True)
train_data = read_conllu(args.data, train_langs, "train", word_to_id, unk_idx, device=args.storage_device)
dev_data = read_conllu(args.data, dev_langs, "dev", word_to_id, unk_idx, device=args.storage_device)
train_size = len(train_data)
dev_size = len(dev_data)
train_data = KMeansBatchIterator(train_data, args.batch_size, args.batch_clusters, len, shuffle=True)
dev_data = KMeansBatchIterator(dev_data, args.batch_size, args.batch_clusters, len, shuffle=False)

model = network.Network(args, embeddings_table=embeddings_table, add_unk=True)
model.to(device=args.device)

optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

#steps_per_epoch = len(list(range(0, len(train_data), 1)))
#scheduler = torch.optim.lr_scheduler.StepLR(optimizer, 10 * steps_per_epoch, gamma=0.1)

#loss_builder = torch.nn.BCEWithLogitsLoss()

sampler = loss.RandomSampler(args.samples)
loss_builder = loss.ISLoss(sampler)
loss_builder.to(args.device)

def compute_batch_loss(batch):
    batch_lengths = [len(s) for s in batch]
    batch = [s.to(args.device) for s in batch]
    batch = nn.utils.rnn.pad_sequence(batch, batch_first=True, padding_value=unk_idx)
    # bigram shape: (n batch, n words, n words)
    # start and end shape: (n batch, n words)
    batch_bigram, batch_start, batch_end = model(batch)

    # batching the loss may be way too difficult as we will have
    # some kind of special loss modules
    # lets split everything by hand
    batch_loss = list()
    for b, l in enumerate(batch_lengths):
        bigram = batch_bigram[b, :l, :l]
        start = batch_start[b, :l]
        end = batch_end[b, :l]
        loss = loss_builder(bigram, start, end)
        batch_loss.append(loss)

    return torch.sum(loss)

for epoch in range(args.epochs):
    model.train()
    train_loss = 0.
    for batch in train_data:
        optimizer.zero_grad()

        loss = compute_batch_loss(batch)
        train_loss += loss.item()

        loss.backward()
        optimizer.step()
        #scheduler.step()

    """
    save_checkpoint({
            'args': args,
            'epoch': epoch + 1,
            'state_dict': model.state_dict(),
            'best_score': best_score,
        }, False, args.model)
    """

    # evaluation
    model.eval()
    dev_loss = 0.
    with torch.no_grad():
        for batch in dev_data:
            dev_loss += compute_batch_loss(batch).item()

    # score_bigram, score_start, score_end = eval.eval(preds, golds)
    print("Epoch %i:\tTrain loss: %.4f\t\tDev loss %.4f" % (epoch, train_loss / train_size, dev_loss / dev_size))

    """
    if dev_epoch_w > best_score:
        best_score = dev_epoch_w
        best_epoch = epoch

        save_checkpoint({
            'args': args,
            'epoch': epoch + 1,
            'state_dict': model.state_dict(),
            'best_score': best_score,
        }, True, args.model)
    """
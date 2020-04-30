import argparse
import os
import shutil
import random
import torch
import numpy as np
import pydestruct.data.conll as conll
import pydestruct.algorithms.msa as msa
import pydestruct.timer as timer
import pydestruct.parser
from gensim.models.keyedvectors import KeyedVectors

import network

def save_checkpoint(state, is_best, path, filename='checkpoint.pth.tar'):
    torch.save(state, path + filename)
    if is_best:
        shutil.copyfile(path + filename, path + 'model_best.pth.tar')

def eval(preds, golds):
    total = 0
    correct = 0
    for p, g in zip(preds, golds):
        for line_p, line_g in zip(p, g):
            total += 1
            for w_p, w_g in zip(line_p, line_g):
                if w_g == 1 and w_p >= 0:
                    correct += 1

    return correct/total

# Read command line
cmd = argparse.ArgumentParser()
cmd.add_argument("--train", type=str, required=True, help="Path to training data")
cmd.add_argument("--dev", type=str, required=True, help="Path to dev data")
cmd.add_argument("--model", type=str, required=True, help="Path where to store the model")
cmd.add_argument("--format", type=str, default="conllu")
cmd.add_argument("--lr", type=float, default=0.01)
cmd.add_argument("--epochs", type=int, default=20, help="Number of epochs for training")
cmd.add_argument("--batch", type=int, default=1, help="Mini-batch size")
cmd.add_argument('--storage-device', type=str, default="cpu", help="Device where to store the data. It is useful to keep it on CPU when the dataset is large, even if computation is done on GPU")
cmd.add_argument('--device', type=str, default="cpu", help="Device to use for computation")
cmd.add_argument('--resume', type=str, default="", help="Resume training from a saved model.")
network.Network.add_cmd_options(cmd)
args = cmd.parse_args()

print("Loading train and dev data")
train_data = list(conll.read(args.train, format=args.format))
dev_data = list(conll.read(args.dev, format=args.format))

model = network.Network(args)

model.to(device=args.device)
optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

steps_per_epoch = len(list(range(0, len(train_data), 1)))
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, 10 * steps_per_epoch, gamma=0.1)

loss_builder = torch.nn.BCEWithLogitsLoss()

for epoch in range(args.epochs):
    print('Starting epoch', epoch)

    random.shuffle(train_data)
    model.train()
    epoch_loss = 0
    best_score = 0
    best_epoch = 0
    c = 0

    for sentence in train_data:
        optimizer.zero_grad()

        n_words = len(sentence["tokens"])

        if n_words < 2:
            continue

        if c % 500 == 0 and c != 0:
            print("Sentence", c)

        batch_inputs = [word["form"].lower() for word in sentence["tokens"]]

        gold = torch.zeros((n_words, n_words))
        for i in range(n_words - 1):
            gold[i][i+1] = 1
        gold = gold.reshape(-1, 1)

        out = model(batch_inputs)
        c += args.batch

        mask = torch.ones((n_words, n_words)).fill_diagonal_(0).reshape(-1, 1)
        loss_builder = torch.nn.BCEWithLogitsLoss(weight=mask)
        loss = loss_builder(out, gold)

        epoch_loss += loss.item()
        loss.backward()
        optimizer.step()
        scheduler.step()

    save_checkpoint({
            'args': args,
            'epoch': epoch + 1,
            'state_dict': model.state_dict(),
            'best_score': best_score,
        }, False, args.model)

    # dev evaluation
    model.eval()

    preds, golds = [], []

    with torch.no_grad():
        for sentence in dev_data:
            n_words = len(sentence["tokens"])
            gold = torch.zeros((n_words, n_words))
            for i in range(n_words - 1):
                gold[i][i+1] = 1
            golds.append(gold)
            
            pred = model([word["form"].lower() for word in sentence["tokens"]])
            pred = pred.view(n_words, n_words)
            preds.append(pred)

    score = eval(preds, golds)
    print("Epoch %i | Loss %f | Score %f" % (epoch, epoch_loss, score))

    if score > best_score:
        best_score = score
        best_epoch = epoch

        save_checkpoint({
            'args': args,
            'epoch': epoch + 1,
            'state_dict': model.state_dict(),
            'best_score': best_score,
        }, True, args.model)

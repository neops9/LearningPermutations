import argparse
import os
import shutil
import random
import torch
import numpy as np
import pydestruct.data.conll as conll
import pydestruct.timer as timer
import pydestruct.parser
import torch.nn.functional as F

import network
import loss
import eval

def save_checkpoint(state, is_best, path, filename='checkpoint.pth.tar'):
    torch.save(state, path + filename)
    if is_best:
        shutil.copyfile(path + filename, path + 'model_best.pth.tar')

# Read command line
cmd = argparse.ArgumentParser()
cmd.add_argument("--train", type=str, required=True, help="Path to training data")
cmd.add_argument("--dev", type=str, required=True, help="Path to dev data")
cmd.add_argument('--fasttext', type=str, required=True, help="Path to the FastText vector file")
cmd.add_argument("--model", type=str, required=True, help="Path where to store the model")
cmd.add_argument("--format", type=str, default="conllu")
cmd.add_argument("--lr", type=float, default=0.01)
cmd.add_argument("--samples", type=int, default=10, help="Number of samples")
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

#loss_builder = torch.nn.BCEWithLogitsLoss()

sampler = loss.RandomSampler(args.samples)
loss_builder = loss.Loss(sampler)

for epoch in range(args.epochs):
    print('Starting epoch', epoch)

    random.shuffle(train_data)
    model.train()
    epoch_gold_w = 0
    best_score = 0
    best_epoch = 0

    for i, sentence in enumerate(train_data):
        optimizer.zero_grad()

        n_words = len(sentence["tokens"])

        if n_words < 2:
            continue

        if i % 500 == 0 and i != 0:
            print("Sentence", i)

        batch_inputs = [word["form"].lower() for word in sentence["tokens"]]
        gold = range(n_words)

        out = model(batch_inputs)

        #mask = torch.ones((n_words, n_words)).fill_diagonal_(0).reshape(-1, 1)
        #loss = F.binary_cross_entropy_with_logits(out, gold, weight=mask)
        loss, gold_w = loss_builder(out, gold)

        epoch_gold_w += gold_w
        loss.backward()
        optimizer.step()
        scheduler.step()

    epoch_gold_w /= len(train_data)

    save_checkpoint({
            'args': args,
            'epoch': epoch + 1,
            'state_dict': model.state_dict(),
            'best_score': best_score,
        }, False, args.model)

    # evaluation
    model.eval()

    dev_epoch_loss = 0

    with torch.no_grad():
        for sentence in dev_data:
            n_words = len(sentence["tokens"])

            pred = model([word["form"].lower() for word in sentence["tokens"]])
            gold = range(n_words)

            loss, gold_w = loss_builder(pred, gold)

            dev_epoch_loss += gold_w
        
    dev_epoch_loss /= len(dev_data)

    # score_bigram, score_start, score_end = eval.eval(preds, golds)
    print("Epoch %i | Train gold weights %f | Dev gold weights %f" % (epoch, epoch_gold_w, dev_epoch_loss))

    if dev_epoch_loss > best_score:
        best_score = dev_epoch_loss
        best_epoch = epoch

        save_checkpoint({
            'args': args,
            'epoch': epoch + 1,
            'state_dict': model.state_dict(),
            'best_score': best_score,
        }, True, args.model)
import argparse
import shutil
import torch
import sys
from learnperm import loss, network
import time
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
cmd.add_argument("--model", type=str, default="", help="Path where to store the model")
cmd.add_argument("--format", type=str, default="conllu")
cmd.add_argument("--lr", type=float, default=0.005)
cmd.add_argument("--samples", type=int, default=10000, help="Number of samples")
cmd.add_argument("--epochs", type=int, default=500, help="Number of epochs for training")
cmd.add_argument('--storage-device', type=str, default="cpu", help="Device to use for data storage")
cmd.add_argument('--device', type=str, default="cpu", help="Device to use for computation")
cmd.add_argument('--resume', type=str, default="", help="Resume training from a saved model.")
cmd.add_argument('--batch-size', type=int, default=2500, help="Maximum number of words per batch")
cmd.add_argument('--batch-clusters', type=int, default=32, help="Number of clusters to use to construct batches")
network.Network.add_cmd_options(cmd)
args = cmd.parse_args()

dev_langs = args.dev_langs.split(",")
train_langs = [args.train_langs]
if len(dev_langs) == 0:
    raise RuntimeError("No dev languages")
all_langs = set(train_langs + dev_langs)
print("All langs: ", all_langs, file=sys.stderr, flush=True)
print("Train langs: ", train_langs, file=sys.stderr, flush=True)
print("Dev langs: ", dev_langs, file=sys.stderr, flush=True)

print("Loading vocabulary and embeddings", file=sys.stderr, flush=True)
embeddings_table, word_to_id, id_to_word, unk_idx = load_embeddings(args.embeddings, all_langs)

print("Loading train and dev data", file=sys.stderr, flush=True)
train_data = read_conllu(args.data, train_langs, "train", word_to_id, unk_idx, device=args.storage_device)
train_size = len(train_data)
train_data = KMeansBatchIterator(train_data, args.batch_size, args.batch_clusters, len, shuffle=True)

# for dev, we separate each language so we can report score per language
dev_datas = {
    lang: read_conllu(args.data, [lang], "dev", word_to_id, unk_idx, device=args.storage_device)
    for lang in dev_langs
}
dev_datas = {
    lang: KMeansBatchIterator(data, args.batch_size, args.batch_clusters, len, shuffle=False)
    for lang, data in dev_datas.items()
}

model = network.Network(args, embeddings_table=embeddings_table, add_unk=True)
model.to(device=args.device)

# remove the embedding table as it does not require a gradient
optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=args.lr)

#steps_per_epoch = len(list(range(0, len(train_data), 1)))
#scheduler = torch.optim.lr_scheduler.StepLR(optimizer, 10 * steps_per_epoch, gamma=0.1)

#loss_builder = torch.nn.BCEWithLogitsLoss()

sampler = loss.RandomSampler(args.samples)
loss_builder = loss.ISLoss(sampler)
loss_builder.to(args.device)

def compute_batch_loss(batch):
    batch_lengths = [len(s) for s in batch]
    batch = [s.to(args.device) for s in batch]
    # shape: (n batch, n words)
    batch = nn.utils.rnn.pad_sequence(batch, batch_first=True, padding_value=unk_idx)
    # bigram shape: (n batch, n words, n words)
    # start and end shape: (n batch, n words)
    batch_bigram, batch_start, batch_end = model(batch)

    # batching the loss may be way too difficult as we will have
    # some kind of special loss modules
    # lets split everything by hand
    batch_loss = list()
    n_worse_than_gold_total = 0.
    for b, l in enumerate(batch_lengths):
        bigram = batch_bigram[b, :l, :l]
        start = batch_start[b, :l]
        end = batch_end[b, :l]
        loss, n_worse_than_gold = loss_builder(bigram, start, end)
        batch_loss.append(loss)
        n_worse_than_gold_total += n_worse_than_gold

    return torch.sum(sum(batch_loss)), n_worse_than_gold_total

best_epoch = 0
best_score = 0

for epoch in range(args.epochs):
    epoch_start_time = time.time()
    model.train()
    train_loss = 0.
    train_n_worse = 0.
    for batch in train_data:
        optimizer.zero_grad()

        loss, n_worse = compute_batch_loss(batch)
        train_loss += loss.item()
        train_n_worse += n_worse.item()

        loss = loss / len(batch)
        loss.backward()
        optimizer.step()
        #scheduler.step()

    if not args.model == "":
        remove_list = ["feature_extractor.embs.weight"]
        state_dict = {k: v for k, v in model.state_dict().items() if not k in remove_list} 
        print(state_dict.keys())
        save_checkpoint({
                'args': args,
                'epoch': epoch + 1,
                'state_dict': state_dict,
                'best_score': best_score,
            }, False, args.model)

    # evaluation
    model.eval()
    dev_losses = {}
    dev_n_worses = {}
    for lang, data in dev_datas.items():
        dev_loss = 0.
        dev_n_worse = 0.
        dev_denum = 0.
        with torch.no_grad():
            for batch in data:
                dev_denum += len(batch)
                l, n = compute_batch_loss(batch)
                dev_loss += l.item()
                dev_n_worse += n.item()
        dev_losses[lang] = dev_loss / dev_denum
        dev_n_worses[lang] = dev_n_worse / (dev_denum * args.samples)

    print(
        "Epoch %i:"
        "\tTrain loss: %.4f"
        "\t\tTrain acc: %s"
        "\t\tDev loss: %s"
        "\t\tDev acc: %s"
        "\t\tTiming (sec): %i"
        %
        (
            epoch,
            train_loss / train_size,
            train_n_worse / (train_size * args.samples),
            ", ".join("%s=%.4f" % (lang, loss) for lang, loss in dev_losses.items()),
            ", ".join("%s=%.4f" % (lang, loss) for lang, loss in dev_n_worses.items()),
            time.time() - epoch_start_time
        ),
        file=sys.stderr,
        flush=True
    )

    if dev_n_worses[train_langs[0]] > best_score:
        best_score = dev_n_worses[train_langs[0]]
        best_epoch = epoch
     
        if not args.model == "":
            remove_list = ["feature_extractor.embs.weight"]
            state_dict = {k: v for k, v in model.state_dict().items() if not k in remove_list} 
            print(state_dict.keys())
            save_checkpoint({
                'args': args,
                'epoch': epoch + 1,
                'state_dict': state_dict,
                'best_score': best_score,
            }, True, args.model)

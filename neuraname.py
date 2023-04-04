import torch
import torch.nn.functional as F
from matplotlib import pyplot

names = open('names.txt', 'r').read().splitlines()

chars = sorted(list(set(''.join(names))))
stoi = {s: i+1 for i, s in enumerate(chars)}
stoi["."] = 0
itos = {i: s for s, i in stoi.items()}

block_size = 3

X, Y = [], []

for name in names[:5]:
    context = [0] * block_size
    print(name)

    for n in name + '.':
        ix = stoi[n]
        X.append(context)
        Y.append(ix)

        print(''.join(itos[c] for c in context) + ' --> ' + itos[ix])
        context = context[1:] + [ix]

X = torch.tensor(X)
Y = torch.tensor(Y)

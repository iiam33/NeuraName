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

g = torch.Generator().manual_seed(2147483647)
C = torch.randn((27, 2), generator=g)
# embedding
emb = C[X]

# # first approach of concatenating the embeddings
# torch.cat(torch.unbind(emb, 1), 1)

# # second approach of concatenating the embeddings
# # more efficient and will not create redundant memory
# emb.view(32, 6)

W1 = torch.randn((6, 100), generator=g)
b1 = torch.randn(100, generator=g)

# pytorch will infer the -1 should be
h = torch.tanh(emb.view(-1, 6) @ W1 + b1)

W2 = torch.randn((100, 27), generator=g)
b2 = torch.randn(27, generator=g)

parameters = [C, W1, b1, W2, b2]
# number of parameters in total
print("parameters:", sum(p.nelement() for p in parameters))

logits = h @ W2 + b2
# # softmax activation function / classification
# # this is a manual classification function
# counts = logits.exp()
# prob = counts / counts.sum(1, keepdim=True)
# loss = -prob[torch.arange(32), Y].log().mean()

# this is a built-in classification function from pytorch
# which results the same output as softmax activation function
loss = F.cross_entropy(logits, Y)

print("loss:", loss)

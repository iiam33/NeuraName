import torch
import torch.nn.functional as F

names = open('names.txt', 'r').read().splitlines()

chars = sorted(list(set(''.join(names))))
stoi = {s: i+1 for i, s in enumerate(chars)}
stoi["."] = 0

N = torch.zeros((27, 27), dtype=torch.int32)

xs, ys = [], []

for name in names:
    chars = ["."] + list(name) + ["."]
    for ch1, ch2 in zip(chars, chars[1:]):
        ix1 = stoi[ch1]
        ix2 = stoi[ch2]
        xs.append(ix1)
        ys.append(ix2)

xs = torch.tensor(xs)
ys = torch.tensor(ys)
num = xs.nelement()
print(xs)
print(ys)

g = torch.Generator().manual_seed(2147483647)
W = torch.randn((27, 27), generator=g, requires_grad=True)

for _ in range(150):
    # forward pass
    # input of the neural network: one hot encoding
    x_encode = F.one_hot(xs, num_classes=27).float()
    logits = x_encode @ W  # predict the log-counts (equivalent to w*x)

    # softmax activation function
    counts = logits.exp()  # equivalent to N
    # probabilities of the next chars
    prob = counts / counts.sum(1, keepdim=True)
    loss = -prob[torch.arange(num), ys].log().mean() + 0.01*(W**2).mean()
    print(loss.item())

    # backward pass
    W.grad = None
    loss.backward()

    # update
    W.data += -50 * W.grad

g = torch.Generator().manual_seed(2147483647)

itos = {i: s for s, i in stoi.items()}

for _ in range(10):
    ix = 0
    out = []
    while True:
        x_encode = F.one_hot(torch.tensor([ix]), num_classes=27).float()
        logits = x_encode @ W
        counts = logits.exp()
        prob = counts / counts.sum(1, keepdim=True)

        ix = torch.multinomial(prob, num_samples=1,
                               replacement=True, generator=g).item()
        out.append(itos[ix])

        if ix == 0:
            break

    print("".join(out))

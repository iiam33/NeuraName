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
# print(W)

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


# for i in range(5):
#     print(f'bigram example {i}: {itos[xs[i].item()]}{itos[ys[i].item()]} indexes({xs[i].item(), ys[i].item()})')
#     print(f'next output probability for the next char: {prob[i]}')
#     print(f'label (actual next output char): {itos[ys[i].item()]}')
#     print(f'probability assigned to the next correct character : {prob[i, ys[i].item()]}')
#     print(f'log likelihood: {prob[i, ys[i].item()].log()}')
#     print(f'negative log likelihood: {-prob[i, ys[i].item()].log()}')
#     print()

# print(prob)
# print(torch.arange(5))


# P = (N+1).float()
# P /= P.sum(1, keepdim=True)

# ix = 0
# out = []

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

        ix = torch.multinomial(prob, num_samples=1, replacement=True, generator=g).item()
        out.append(itos[ix])

        if ix == 0:
            break

    print("".join(out))

# log_likelihood = 0.0
# n = 0

# for name in names:
#     chars = ["."] + list(name) + ["."]
#     for ch1, ch2 in zip(chars, chars[1:]):
#         ix1 = stoi[ch1]
#         ix2 = stoi[ch2]
#         prob = P[ix1, ix2]
#         logprob = torch.log(prob)
#         log_likelihood += logprob
#         n += 1
#         # print(f'{ch1, ch2}: {prob:.4f} {logprob:.4f}')

# print(f'{log_likelihood=}')
# neg_log_likelihood = -log_likelihood
# print(f'{neg_log_likelihood=}')
# print(f'average of negative log_likelihood: {(neg_log_likelihood/n):.4f}')

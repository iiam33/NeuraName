import torch
import matplotlib.pyplot as plt

names = open('names.txt', 'r').read().splitlines()

bigram = {}

chars = sorted(list(set(''.join(names))))
stoi = {s: i+1 for i, s in enumerate(chars)}
stoi["."] = 0

N = torch.zeros((27, 27), dtype=torch.int32)

for name in names:
    chars = ["."] + list(name) + ["."]
    for ch1, ch2 in zip(chars, chars[1:]):
        ix1 = stoi[ch1]
        ix2 = stoi[ch2]

        N[ix1, ix2] += 1

itos = {i: s for s, i in stoi.items()}

# plt.figure(figsize=(20, 20))
# plt.imshow(N, cmap="Blues")

# for i in range(27):
#     for j in range(27):
#         chstr = itos[i]+itos[j]
#         plt.text(j, i, chstr, ha="center", va="bottom", color="grey")
#         plt.text(j, i, N[i, j].item(), ha="center", va="top", color="grey")

# plt.axis("off")
# plt.show()


# ix = torch.multinomial(p, num_samples=1, replacement=True)
# print(itos[ix.item()])

# print(N)
# plt.imshow(N)
# plt.show()

# print(sorted(bigram.items(), key=lambda b: -b[1]))

# print(N)

# 27, 27
# 27,  1

P = N.float()
P /= P.sum(1, keepdim=True)

ix = 0
out = []

g = torch.Generator().manual_seed(2147483647)

for _ in range(10):
    ix = 0
    out = []
    while True:
        p = P[ix]
        # p /= p.sum()
        ix = torch.multinomial(
            p, num_samples=1, replacement=True, generator=g).item()
        out.append(itos[ix])

        if ix == 0:
            break

    print("".join(out))

log_likelihood = 0.0
n = 0

for name in names:
# for name in ["alvzin"]:
    chars = ["."] + list(name) + ["."]
    for ch1, ch2 in zip(chars, chars[1:]):
        ix1 = stoi[ch1]
        ix2 = stoi[ch2]
        prob = P[ix1, ix2]
        logprob = torch.log(prob)
        log_likelihood += logprob
        n += 1
        # print(f'{ch1, ch2}: {prob:.4f} {logprob:.4f}')

print(f'{log_likelihood=}')
neg_log_likelihood = -log_likelihood
print(f'{neg_log_likelihood=}')
print(f'average of negative log_likelihood: {(neg_log_likelihood/n):.4f}')

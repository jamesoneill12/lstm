import torch
import torch.nn.functional as F
from torch import nn


class ANN(nn.Module):
    def __init__(self, emb_size, vocab_size,
                 hid_dim, hid_num, class_num, sent_len):
        super(ANN, self).__init__()
        self.emb = nn.Embedding(vocab_size, emb_size)
        self.fc1 = nn.Linear(sent_len * hid_dim, hid_dim)
        self.fc2 = nn.Linear(hid_dim, class_num)

    # (sent_len, batch_size, embedding_size)
    def forward(self, x):
        x = self.emb(x)
        # 40, 1000, 100
        x = x.view(x.size(1), x.size(0) * x.size(2))
        x = F.tanh(self.fc1(x))
        x = F.tanh(self.fc2(x))
        return x


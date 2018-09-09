import torch
from torch.autograd import Variable
from torch import nn
import matplotlib.pyplot as plt
from torch import optim
from model import LSTM

vocab_size = 1000
emb_size = 100
hid_dim = emb_size
hid_num = 1
class_num = 5
classes = [1, 2, 5, 8, 10, 15, 20]
batch_size = 1000
sent_len = 30
epochs = 20

x = Variable(torch.randint(0, vocab_size-1, (sent_len, batch_size)).type(torch.LongTensor))
target = Variable(torch.randint(0, class_num, (batch_size, )).type(torch.LongTensor))
model = LSTM(emb_size, vocab_size, hid_dim, hid_num, class_num).cuda()

if torch.cuda.is_available():
    x = x.cuda()
    target = target.cuda()
    model = model.cuda()


criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.01)

ls = []
for i in range(epochs):
    model.zero_grad()
    y_pred, all_y = model(x.cuda())
    loss = criterion(y_pred, target.cuda())
    loss.backward()
    optimizer.step()
    print('epoch {}, loss:{:.8f}'.format(i + 1, loss.data[0]))
    ls.append(loss.data[0])

plt.plot(ls)
plt.grid(True)
plt.show()

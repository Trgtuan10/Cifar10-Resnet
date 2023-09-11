
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader

from model import resnet50

batch_size = 50

train_loader = DataLoader(
    datasets.CIFAR10("data",train = True, download = True, transform = transforms.ToTensor()),
    batch_size = batch_size,
    shuffle = True
)

test_loader = DataLoader(
    datasets.CIFAR10("data",train = False, download = True, transform = transforms.ToTensor()),
    batch_size = 1000
)

# CHECK SIZE OF DATA
# for x,y in train_loader:
#   print({x.shape})
#   print({y.shape} , {y.dtype})
#   print(y)
#   break

model = resnet50()

# CHECK SIZE OF MODEL
# for name, p in model.named_parameters():
#   print(name, ":" , p.size())


#TRAINING MODEL
opt = optim.Adam(model.parameters(), lr = 0.0001)

model.train()

epochs = 1
for epoch in range(epochs):
  for batch, (x, y) in enumerate(train_loader):
    total_loss, acc = 0, 0
    # x, y = Variable(x), Variable(y)
    # print(x)
    # print(y)
    opt.zero_grad()
    pred = model(x)
    loss = F.cross_entropy(pred, y)
    loss.backward()
    total_loss += loss.item() #cal loss
    opt.step()
    prediction = pred.argmax(1)

    acc += (prediction == y).type(torch.float).sum().item() #cal accuracy
    # print(acc)
    # print(total_loss)

    if batch % 10 == 0:
      print('Epoch: {}\tTrain Step: {}\tLoss: {:.3f}\tAccuracy: {:.3f}'.format(epoch+1, batch,total_loss, acc/batch_size *100))
      

#TESTING MODEL
model.eval()
correct = 0
total = 0
with torch.no_grad():
  for x, y in test_loader:
    pred = model(x)
    prediction = pred.argmax(1)
    correct += (prediction == y).sum().item()
    total += y.size(0)
print('Accuracy: {:.3f}'.format(correct/total * 100))


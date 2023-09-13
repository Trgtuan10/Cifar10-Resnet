import os
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import datetime

from model import resnet50

batch_size = 50

train_loader = DataLoader(
    datasets.CIFAR10("data", train=True, download=True, transform=transforms.ToTensor()),
    batch_size=batch_size,
    shuffle=True
)

test_loader = DataLoader(
    datasets.CIFAR10("data", train=False, download=True, transform=transforms.ToTensor()),
    batch_size=1000
)

model = resnet50()

# CHECK SIZE OF MODEL
# for name, p in model.named_parameters():
#     print(name, ":", p.size())

opt = optim.Adam(model.parameters(), lr=0.0001)
loss_fn = nn.CrossEntropyLoss()
scheduler = optim.lr_scheduler.StepLR(opt, step_size=1, gamma=0.1)

# TRAINING MODEL
def train():
    model.train()
    epochs = 1
    for epoch in range(epochs):
        for batch, (x, y) in enumerate(train_loader):
            total_loss, correct = 0, 0
            opt.zero_grad()
            pred = model(x)
            loss = loss_fn(pred, y)
            loss.backward()
            total_loss += loss.item()  # Calculate loss
            opt.step()
            prediction = pred.argmax(1)
            correct += (prediction == y).sum().item()  # Calculate correct predictions
            # total += y.size(0)

            if batch % 10 == 0:
                print('Epoch: {}\tTrain Step: {}\tLoss: {:.3f}\tAccuracy: {:.3f}'.format(epoch + 1, batch,
                                                                                          total_loss,
                                                                                          correct / batch_size * 100))
                save_results('train',batch, total_loss, correct / batch_size * 100)

        scheduler.step()

# TESTING MODEL
def test():
    model.eval()

    with torch.no_grad():
        for batch, (x, y) in enumerate(test_loader):
            total_loss, correct =  0, 0
            pred = model(x)
            loss = loss_fn(pred, y)
            total_loss += loss.item()
            prediction = pred.argmax(1)
            correct += (prediction == y).sum().item()
            #total += y.size(0)

        print('Loss: {:.3f}\tAccuracy: {:.3f}'.format(total_loss, correct /batch_size * 100))
        save_results('test',batch, total_loss, correct / batch_size * 100)

# Log accuracy vs loss
def save_results(phase, batch, loss, accuracy):
    results_path = "results/"
    if not os.path.exists(results_path):
        os.makedirs(results_path)

    if phase == 'train':
        current_time = datetime.datetime.now().strftime("%Y-%m-%d %H-%M-%S")
    else :
        current_time = ""
    file_path = os.path.join(results_path, "{}.csv".format(phase))

    with open(file_path, 'a') as f:
        if batch == 0 and phase == 'train':
            f.write("Time:{},\nBatch:{},\tLoss:{},\tAcc:{}\n".format(current_time, batch, loss, accuracy)) 
        elif batch != 0 and phase == 'train':
            f.write("Batch:{},\tLoss:{},\tAcc:{}\n".format(batch, loss, accuracy))
        else:
            f.write("Loss:{},\tAcc:{}\n".format(loss, accuracy))

if __name__ == "__main__":
    train()
    test()

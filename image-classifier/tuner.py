#   this is the tuner for the classifying model

import os
import torch
from torchvision import datasets
from torch.utils.data import random_split
from torch.utils.data import DataLoader
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from sys import platform
import numpy as np
from filelock import FileLock
from functools import partial
import torchvision.transforms as transforms
from ray import tune
from ray import train as tr
from ray.train import Checkpoint
from ray.tune.schedulers import ASHAScheduler

if platform == "linux" or platform == "linux2":
    path = os.getcwd()+'/code/traindata'  # for linux
elif platform == "win32":
    path = os.getcwd()+'\\code\\traindata'  # for windows loading

if torch.cuda.is_available():
    print("CUDA available. Using GPU acceleration.")
    device = "cuda"
else:
    print("CUDA is NOT available. Using CPU for training.")
    device = "cpu"

classes = ('cherry', 'strawberry', 'tomato')

batch_size = 16


def load_data(data_dir=path):
    #   Load training images into dataset
    transform = transforms.Compose(
        [transforms.Resize((300, 300)),
         transforms.RandomHorizontalFlip(p=0.5),
         transforms.RandomVerticalFlip(0.2),
         transforms.ToTensor(),
         transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
         ])

    imgdata = datasets.ImageFolder(root=path, transform=transform)

    #   Split data into train:validation at 90/10
    train, val = random_split(imgdata, [3600, 450])

    return train, val


#   Define the Neural Network


class Net(nn.Module):
    def __init__(self, dropout=0.1):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(
            in_channels=3, out_channels=8, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(
            in_channels=8, out_channels=128, kernel_size=3, padding=1)
        self.conv3 = nn.Conv2d(
            in_channels=128, out_channels=48, kernel_size=3, padding=1)

        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)

        self.drop = nn.Dropout2d(p=dropout)

        self.fc1 = nn.Linear(65712, 48)
        self.fc2 = nn.Linear(48, 24)
        self.fc3 = nn.Linear(24, 3)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = self.pool(F.relu(self.conv3(x)))

        x = F.dropout(self.drop(x), training=self.training)

        x = x.view(-1, 65712)
        x = torch.flatten(x, 1)  # flatten all dimensions except batch
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x


def tune_cnn(config, data_dir=None):
    net = Net().to(device)

    criterion = nn.CrossEntropyLoss()
    optimiser = config["optim"]

    loaded_checkpoint = tr.get_checkpoint()

    if loaded_checkpoint:
        with loaded_checkpoint.as_directory() as loaded_checkpoint_dir:
            model_state, optimizer_state = torch.load(
                os.path.join(loaded_checkpoint_dir, "checkpoint.pt"))
        net.load_state_dict(model_state)
        optimiser.load_state_dict(optimizer_state)

    train, test = load_data(data_dir)

    train_loader = DataLoader(train, batch_size=batch_size,
                              shuffle=True, num_workers=8)
    val_loader = DataLoader(test, batch_size=batch_size,
                            shuffle=True, num_workers=8)

    for epoch in range(10):  # loop over the dataset multiple times
        running_loss = 0.0
        epoch_steps = 0
        for i, data in enumerate(train_loader, 0):
            # get the inputs; data is a list of [inputs, labels]
            inputs, labels = data
            inputs, labels = inputs.to(device), labels.to(device)

            # zero the parameter gradients
            optimiser.zero_grad()

            # forward + backward + optimize
            outputs = net(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimiser.step()

            # print statistics
            running_loss += loss.item()
            epoch_steps += 1
            if i % 2000 == 1999:  # print every 2000 mini-batches
                print(
                    "[%d, %5d] loss: %.3f"
                    % (epoch + 1, i + 1, running_loss / epoch_steps)
                )
                running_loss = 0.0

        # Validation loss
        val_loss = 0.0
        val_steps = 0
        total = 0
        correct = 0
        for i, data in enumerate(val_loader, 0):
            with torch.no_grad():
                inputs, labels = data
                inputs, labels = inputs.to(device), labels.to(device)

                outputs = net(inputs)
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()

                loss = criterion(outputs, labels)
                val_loss += loss.cpu().numpy()
                val_steps += 1

        os.makedirs("my_model", exist_ok=True)
        torch.save(
            (net.state_dict(), optimiser.state_dict()), "my_model/checkpoint.pt")
        checkpoint = Checkpoint.from_directory("my_model")
        tr.report({"loss": (val_loss / val_steps),
                     "accuracy": correct / total}, checkpoint=checkpoint)
    print("Finished Training")


def test_best_model(best_result):
    best_trained_model = Net()
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    best_trained_model.to(device)

    checkpoint_path = os.path.join(best_result.checkpoint.to_directory(), "checkpoint.pt")

    model_state, optimiser_state = torch.load(checkpoint_path)
    best_trained_model.load_state_dict(model_state)

    trainset, testset = load_data()

    testloader = torch.utils.data.DataLoader(
        testset, batch_size=batch_size, shuffle=False, num_workers=2)

    correct = 0
    total = 0
    with torch.no_grad():
        for data in testloader:
            images, labels = data
            images, labels = images.to(device), labels.to(device)
            outputs = best_trained_model(images)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    print("Best trial test set accuracy: {}".format(correct / total))


def main(num_samples=10, max_num_epochs=10, gpus_per_trial=1):
    load_data(path)
    net = Net().to(device)
    rmsprop = optim.RMSprop(net.parameters(), lr=0.007, momentum=0.7) 
    sgd = optim.SGD(net.parameters(), lr=0.007, momentum=0.7)
    adam = optim.Adam(net.parameters(), lr=0.007)
    adagrad = optim.Adagrad(net.parameters(), lr=0.007)
    adadelt = optim.Adadelta(net.parameters(), lr=0.007)
    adamw = optim.AdamW(net.parameters(), lr=0.007, weight_decay=0.01)
    config = {
        "dropout": tune.choice([0.0, 0.1, 0.2]),
        "optim": tune.choice([sgd, rmsprop])}

    scheduler = ASHAScheduler(
        max_t=max_num_epochs,
        grace_period=1,
        reduction_factor=2,
    )

    tuner = tune.Tuner(
        tune.with_resources(
            tune.with_parameters(tune_cnn),
            resources={"cpu": 4, "gpu": gpus_per_trial}
        ),
        tune_config=tune.TuneConfig(
            metric="loss",
            mode="min",
            scheduler=scheduler,
            num_samples=num_samples,
        ),
        param_space=config,
    )
    results = tuner.fit()

    best_result = results.get_best_result("loss", "min")

    print("Best trial config: {}".format(best_result.config))
    print("Best trial final validation loss: {}".format(
        best_result.metrics["loss"]))
    print("Best trial final validation accuracy: {}".format(
        best_result.metrics["accuracy"]))

    test_best_model(best_result)


if __name__ == '__main__':
    main(num_samples=10, max_num_epochs=5, gpus_per_trial=1)


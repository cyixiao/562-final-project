import torchvision
from torch.utils.tensorboard import SummaryWriter
from torch import nn
from torch.utils.data import DataLoader
from tqdm import tqdm

from model_CNN import *

# prepare dataset
train_data = torchvision.datasets.CIFAR10(root="dataset", train=True, transform=torchvision.transforms.ToTensor(),
                                          download=True)
test_data = torchvision.datasets.CIFAR10(root="dataset", train=False, transform=torchvision.transforms.ToTensor(),
                                         download=True)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# get dataset length
train_data_size = len(train_data)
test_data_size = len(test_data)
print(train_data_size)
print(test_data_size)

# gather dataset by using dataloader
train_dataloader = DataLoader(train_data, batch_size=64)
test_dataloader = DataLoader(test_data, batch_size=64)

learning_rates = [0.001, 0.01, 0.1]
weight_decays = [0.001, 0.01, 0.1]
for lr in learning_rates:
    for wd in weight_decays:
        print("================= training model (lr = {}, wd = {}) =================".format(lr, wd))
        model = CNN()
        model.to(device)

        # use cross entropy loss as our main loss function
        loss_function = nn.CrossEntropyLoss()
        loss_function.to(device)

        # Construct Optimizer, us SGD
        optimizer = torch.optim.SGD(model.parameters(), lr=lr, weight_decay=wd)

        # track training process
        writer = SummaryWriter(f"train_logs/lr_{lr}_wd_{wd}")

        # track train steps
        total_train_step = 0
        # track test steps
        total_test_step = 0
        # train rounds
        epochs = 50

        # start training for `epoch` times
        for epoch in tqdm(range(epochs), desc="Training Progress"):
            # training part
            model.train()
            total_train_loss = 0
            total_train_correct = 0
            for data in train_dataloader:
                imgs, targets = data
                imgs = imgs.to(device)
                targets = targets.to(device)
                outputs = model(imgs)
                loss = loss_function(outputs, targets)

                # optimizer
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                total_train_loss += loss.item()
                total_train_correct += (outputs.argmax(1) == targets).sum().item()

            train_loss = total_train_loss / len(train_dataloader)
            train_accuracy = total_train_correct / len(train_data)

            # test part
            model.eval()
            total_test_loss = 0
            total_test_correct = 0
            with torch.no_grad():
                for data in test_dataloader:
                    imgs, targets = data
                    imgs = imgs.to(device)
                    targets = targets.to(device)
                    outputs = model(imgs)
                    loss = loss_function(outputs, targets)

                    total_test_loss += loss.item()
                    total_test_correct += (outputs.argmax(1) == targets).sum().item()

            test_loss = total_test_loss / len(test_dataloader)
            test_accuracy = total_test_correct / len(test_data)

            writer.add_scalars("Loss", {"Training": train_loss, "Validation": test_loss}, epoch + 1)
            writer.add_scalars("Accuracy", {"Training": train_accuracy, "Validation": test_accuracy}, epoch + 1)

        writer.close()

import torch
from utils import show_image
from pathlib import Path


class train_model:
    def __init__(self, device):
        self.path = Path('./experiments/')
        self.device = device

    def train(self, model, criterion, train_loader, optimizer, epoch):
        model.train()
        for batch_idx, data in enumerate(train_loader):
            data['i1'] = data['i1'].to(self.device, dtype=torch.float)
            data['i2'] = data['i2'].to(self.device, dtype=torch.float)
            data['o1'] = data['o1'].to(self.device, dtype=torch.float)

            optimizer.zero_grad()  # making gradients 0, so that they are not accumulated over multiple batches
            output = model(data['i1'], data['i2'])
            loss = criterion(output, data['o1'])
            loss.backward()  # calculating gradients
            optimizer.step()  # updating weights

            if batch_idx % 50 == 0:
                print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                    epoch, batch_idx * len(data['i1']), len(train_loader.dataset), 100. * batch_idx / len(train_loader),
                    loss.item()))
                print('Batch ID: ', batch_idx)

                # len(dataloader.dataset) --> total number of input images
                # len(dataloader) --> total no of batches, each to specified size like 16

            if batch_idx % 500 == 0:
                show_image(output, n_row=4, title='Predicted (Training)')
                print(output)
                print(data['o1'])

            if batch_idx % 1000 == 0:
                torch.save(model.state_dict(), self.path / f"{batch_idx}.pth")

    def test(self, model, criterion, device, test_loader):
        model.eval()  # setting model eveluate mode, takes care of batch norm, dropout etc. not required in testing
        test_loss = 0
        correct = 0
        with torch.no_grad():
            for data in test_loader:
                data['i1'] = data['i1'].to(self.device, dtype=torch.float)
                data['i2'] = data['i2'].to(self.device, dtype=torch.float)
                data['o1'] = data['o1'].to(self.device, dtype=torch.float)
                output = model(data['i1'], data['i2'])

                test_loss += criterion(output, data['o1'], reduction='sum').item()
                pred = output.argmax(dim=1, keepdim=True)
                # correct += pred.eq(target.view_as(pred)).sum().item()

                show_image(output.cpu(), n_row=4, title='Predicted (validation)')
        test_loss /= len(test_loader.dataset)

    def run_model(self, model, train_loader, valid_loader, criterion,  lr=0.01, epochs=10):

        optim = torch.optim.SGD(model.parameters(), lr=lr, momentum=0.9, weight_decay=1e-5)

        for epoch in range(1, epochs+1):
            self.train(model, criterion, train_loader, optim, epoch)
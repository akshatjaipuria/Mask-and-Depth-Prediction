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
            loss = loss.view(loss.shape[0], -1).sum(1).mean()
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
                show_image(output, n_row=8, title='Predicted (Training)')
                # print(output)
                # print(data['o1'])

            if batch_idx % 1000 == 0:
                torch.save(model.state_dict(), self.path / f"{batch_idx}.pth")

    def validate(self, model, criterion, valid_loader):
        # setting model evaluate mode, takes care of batch norm, dropout etc. not required while validation
        model.eval()
        valid_loss = 0
        correct = 0
        with torch.no_grad():
            for batch_idx, data in enumerate(valid_loader):
                data['i1'] = data['i1'].to(self.device, dtype=torch.float)
                data['i2'] = data['i2'].to(self.device, dtype=torch.float)
                data['o1'] = data['o1'].to(self.device, dtype=torch.float)
                output = model(data['i1'], data['i2'])
                loss = criterion(output, data['o1'])
                valid_loss += loss.view(loss.shape[0], -1).sum(1).mean().item()
        valid_loss /= len(valid_loader)
        print("Some predicted samples:")
        show_image(output.cpu(), n_row=8, title='Predicted (validation)')
        print("Average Validation loss: ", valid_loss)

    def run_model(self, model, train_loader, valid_loader, criterion, lr=0.01, epochs=10):

        optim = torch.optim.SGD(model.parameters(), lr=lr, momentum=0.9, weight_decay=1e-5)

        for epoch in range(1, epochs + 1):
            self.train(model, criterion, train_loader, optim, epoch)
            self.validate(model, criterion, valid_loader)
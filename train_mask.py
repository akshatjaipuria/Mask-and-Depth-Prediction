import torch
from utils import show_image
from pathlib import Path


class train_model:
    def __init__(self, device, tb=None):
        self.path = Path('./experiments/')
        self.device = device
        self.globaliter = 0
        self.tb = tb

    def train(self, model, criterion, metric, train_loader, optimizer, epoch):
        model.train()
        for batch_idx, data in enumerate(train_loader):
            data['i1'] = data['i1'].to(self.device)
            data['i2'] = data['i2'].to(self.device)
            data['o1'] = data['o1'].to(self.device)

            optimizer.zero_grad()  # making gradients 0, so that they are not accumulated over multiple batches
            output = model(data['i1'], data['i2'])
            loss = criterion(output, data['o1'])
            # loss = loss.view(loss.shape[0], -1).sum(1).mean()
            loss.backward()  # calculating gradients
            optimizer.step()  # updating weights
            if self.tb:
                self.globaliter += 1
                self.tb.save_value('Train Loss', 'train_loss', self.globaliter, loss.item())

            if batch_idx % 50 == 0:
                metric_value = 0
                if metric:
                    metric_value = metric(output, data['o1']).cpu().detach().numpy()
                print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}\tMetric: {:.6f}'.format(
                    epoch, batch_idx * len(data['i1']), len(train_loader.dataset), 100. * batch_idx / len(train_loader),
                    loss.item(), metric_value))
                print('Batch ID: ', batch_idx)

                # len(dataloader.dataset) --> total number of input images
                # len(dataloader) --> total no of batches, each to specified size like 16

            if batch_idx % 500 == 0:
                show_image(data['o1'][::4].cpu(), n_row=8, title='Target (Training)')
                show_image(output[::4].cpu(), n_row=8, title='Predicted (Training)')
                # print(output)
                # print(data['o1'])

            if batch_idx % 1000 == 0:
                torch.save(model.state_dict(), self.path / f"{batch_idx}.pth")

    def validate(self, model, criterion, metric, valid_loader):
        # setting model evaluate mode, takes care of batch norm, dropout etc. not required while validation
        model.eval()
        valid_loss = 0
        correct = 0
        metric_value = 0
        with torch.no_grad():
            for batch_idx, data in enumerate(valid_loader):
                data['i1'] = data['i1'].to(self.device)
                data['i2'] = data['i2'].to(self.device)
                # data['o1'] = data['o1'].to(self.device)
                data['o1'] = data['o1'].to(self.device)

                output = model(data['i1'], data['i2'])
                loss = criterion(output, data['o1'])
                valid_loss += loss.item()  # loss.view(loss.shape[0], -1).sum(1).mean().item()
                if metric:
                    metric_value += metric(output, data['o1']).cpu().detach().numpy()
        metric_value /= len(valid_loader)
        valid_loss /= len(valid_loader)
        print("Some target vs predicted samples:")
        show_image(data['o1'][::4].cpu(), n_row=8, title='Target (validation)')
        show_image(output[::4].cpu(), n_row=8, title='Predicted (validation)')
        print("Average Validation loss: {}\t Average Metric: {}".format(valid_loss, metric_value))

    def run_model(self, model, train_loader, valid_loader, criterion, metric=None, lr=0.01, epochs=10):

        optim = torch.optim.SGD(model.parameters(), lr=lr, momentum=0.9, weight_decay=1e-5)

        for epoch in range(1, epochs + 1):
            if train_loader:
                self.train(model, criterion, metric, train_loader, optim, epoch)
            if valid_loader:
                print("Validating.....")
                self.validate(model, criterion, metric, valid_loader)
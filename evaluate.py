import torch
from utils import show_image


class train_model:
    def __init__(self, device, tb=None):
        self.device = device

    def evaluate(self, model, data_loader, criterion_mask, criterion_depth, metric_mask, metric_depth):
        # setting model evaluate mode, takes care of batch norm, dropout etc. not required while validation
        model.eval()
        valid_loss_mask = 0
        metric_value_mask = 0
        valid_loss_depth = 0
        metric_value_depth = 0
        with torch.no_grad():
            for batch_idx, data in enumerate(data_loader):
                data['i1'] = data['i1'].to(self.device)
                data['i2'] = data['i2'].to(self.device)
                data['o1'] = data['o1'].to(self.device)
                data['o2'] = data['o2'].to(self.device)

                output_mask, output_depth = model(data['i1'], data['i2'])

                loss_mask = criterion_mask(output_mask, data['o1'])
                loss_depth = criterion_depth(output_depth, data['o2'])

                valid_loss_mask += loss_mask.item()
                valid_loss_depth += loss_depth.item()

                if metric_mask:
                    metric_value_mask += metric_mask(output_mask, data['o1']).cpu().detach().numpy()
                if metric_depth:
                    metric_value_depth += metric_depth(output_depth, data['o2']).cpu().detach().numpy()

        metric_value_mask /= len(data_loader)
        valid_loss_mask /= len(data_loader)

        metric_value_depth /= len(data_loader)
        valid_loss_depth /= len(data_loader)

        print("Some target vs predicted samples:")
        show_image(data['i1'][::4].cpu(), n_row=8, title='Input (bg)', mean=[0.5039, 0.5001, 0.4849],
                   std=[0.2465, 0.2463, 0.2582])
        show_image(data['i2'][::4].cpu(), n_row=8, title='Input (fg_bg)', mean=[0.5057, 0.4966, 0.4812],
                   std=[0.2494, 0.2498, 0.2612])
        show_image(data['i2'][::4].cpu(), n_row=8, title='Input (fg_bg)')
        show_image(data['o1'][::4].cpu(), n_row=8, title='Target (Mask)')
        show_image(output_mask[::4].cpu(), n_row=8, title='Predicted (Mask)')
        print("Mask: Average Evaluation loss: {}\t Average Metric: {}".format(valid_loss_mask, metric_value_mask))
        show_image(data['o2'][::4].cpu(), n_row=8, title='Target (Depth)')
        show_image(output_depth[::4].cpu(), n_row=8, title='Predicted (Depth)')
        print("Depth: Average Evaluation loss: {}\t Average Metric: {}".format(valid_loss_depth, metric_value_depth))

    def run(self, model, data_loader, criterion_mask, criterion_depth, metric_mask=None, metric_depth=None):
        print("Evaluating.....")
        self.evaluate(model, data_loader, criterion_mask, criterion_depth, metric_mask, metric_depth)

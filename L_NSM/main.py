import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from l_nsm_net import LNSM_Net
import matplotlib.pyplot as plt

import sys
import yaml
from os.path import exists, join
from os import makedirs
import os
import argparse

parser = argparse.ArgumentParser(description='roam l_nsm')
# Config
parser.add_argument('--config', default='l_nsm.yaml')
args = parser.parse_args()
with open('configs/' + args.config) as f:
    config = yaml.load(f, Loader=yaml.FullLoader)

for key in config:
    for k, v in config[key].items():
        setattr(args, k, v)

# refactor plan: start from main and delete redundant function
torch.manual_seed(1234)
np.random.seed(1234)
print(sys.argv)
class Trainer(object):
    def __init__(self, dataloader, args, output_size, x_mean, x_std, y_mean, y_std):
        self.save_path = os.path.join(args.root_dir, args.ckpt_name)  

        self.x_mean = torch.from_numpy(x_mean).float().cuda()
        self.x_std = torch.from_numpy(x_std).float().cuda()
        self.y_mean = torch.from_numpy(y_mean).float().cuda()
        self.y_std = torch.from_numpy(y_std).float().cuda()
        self.model = LNSM_Net(output_size, args=args).cuda()
        self.model.cuda()
        self.dataloader = dataloader

        self.ckpt_name = args.ckpt_name
        self.batch_size = args.batch_size
        self.epoch = args.epochs
        self.Te = 10
        self.Tmult = 2
        self.learning_rate_ini = 0.0001
        self.weightDecay_ini = 0.0025

        self.loss_fn = torch.nn.MSELoss()

    def train(self):

        # Print Training Information
        total_batch = len(self.dataloader)
        print('Training information', flush=True)
        print('Total Batch', '->', total_batch, flush=True)
        print('--------------------', flush=True)

        # Start training
        optimizer = torch.optim.AdamW(self.model.parameters(), lr=self.learning_rate_ini,
                                      weight_decay=self.weightDecay_ini)
        iters = len(self.dataloader) * self.epoch
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=iters, eta_min=5e-6)

        train_loss = []
        for e in range(self.epoch):
            avg_cost_train = 0
            for i, (batch_X, batch_Y) in enumerate(self.dataloader):
                batch_X = batch_X.cuda()
                batch_Y = batch_Y.cuda()

                optimizer.zero_grad()
                output = self.model(batch_X)
                loss = self.loss_fn(output, batch_Y)
                loss.backward()
                optimizer.step()

                avg_cost_train += loss.item()
                scheduler.step()
            train_loss.append(avg_cost_train/i)
            print('Epoch: {} | Loss {:.4f}'.format(e,  avg_cost_train/i), flush=True)

            if e == self.epoch - 1:
                self.save_weights(self.save_path, e+1)
                self.save_onnx(e)

        # save loss as png
        print("End of training (epoch %d), saving train loss of length %d " %(e, len(train_loss)), flush=True)
        np.array(train_loss, dtype=np.float32).tofile(self.save_path + '/' + 'trainloss.bin')
        plt.plot(np.arange(len(train_loss)), train_loss)
        plt.savefig(self.save_path + '/' + 'trainloss.png')


    def update_lr_wd(self, optimizer, AP, epoch):
        clr, wdc = AP.getParameter(epoch)
        for i, param_group in enumerate(optimizer.param_groups):
            param_group['lr'] = clr
            param_group['weight_decay'] = wdc

    def save_weights(self, model_file, epoch):
        print('Saving weights to', join(model_file, '{}.bin'.format(epoch)))
        if not exists(model_file):
            makedirs(model_file)
        torch.save(self.model.state_dict(), join(model_file, '{}.bin'.format(epoch)))


    def save_onnx(self, epoch):
        """
        Exporting ONNX Model. Integrating Data Statistics into the model graph
        """

        # Dummy input
        dummy_input = torch.randn(1, self.model.dim_gating + self.model.start_gating).cuda()

        # Infusing normalization and de-normalization
        model_to_save = nn.Sequential(nn.BatchNorm1d(self.model.dim_gating + self.model.start_gating).cuda(),
                                      self.model,
                                      nn.Linear(self.y_mean.shape[0], self.y_mean.shape[0], bias=True).cuda()
                                      )

        # Using Inference mode of BatchNorm
        model_to_save.eval()

        # set weights to be statistics of X and Y
        sd = model_to_save.state_dict()
        sd['2.weight'] = torch.eye(self.y_std.shape[0]).cuda() * self.y_std
        sd['2.bias'] = self.y_mean
        sd['0.running_mean'] = self.x_mean
        sd['0.running_var'] = self.x_std ** 2

        model_to_save.load_state_dict(sd)

        output_names = ["output1"]

        torch.onnx.export(model_to_save, dummy_input, "{}/{}_e{}.onnx".format(self.save_path, self.ckpt_name, epoch), export_params=True, input_names=['inputs'],
                          verbose=True,
                          output_names=output_names)
        model_to_save.train()


def create_dataset(data_path, save_path):
    X = []
    Y = []
    with open(join(data_path, 'Input.txt'), "r") as f:
        lines = f.readlines()
        for i in range(len(lines)):
            X.append(np.array([float(x) for x in lines[i].split(' ')]))
        f.close()

    with open(join(data_path, 'Output.txt'), "r") as f:
        lines = f.readlines()
        for i in range(len(lines)):
            Y.append(np.array([float(y) for y in lines[i].split(' ')]))

        f.close()

    X = np.stack(X[:-1], 0)
    Y = np.stack(Y[:-1], 0)
    X = torch.from_numpy(X).float()
    Y = torch.from_numpy(Y).float()
    Y = Y[:X.shape[0]]
    

    assert X.shape[1] == args.start_gating + args.dim_gating

    x_mean = X.numpy().mean(0)
    x_std = X.numpy().std(0)
    x_std[x_std == 0] = 1

    y_mean = Y.numpy().mean(0)
    y_std = Y.numpy().std(0)
    y_std[y_std == 0] = 1


        
    x_mean.tofile(join(save_path, 'Xmean.bin'))
    x_std.tofile(join(save_path, 'Xstd.bin'))
    y_mean.tofile(join(save_path , 'Ymean.bin'))
    y_std.tofile(join(save_path , 'Ystd.bin'))
    X = (X - torch.from_numpy(x_mean).reshape(1, -1)) / torch.from_numpy(x_std).reshape(1, -1)
    Y = (Y - torch.from_numpy(y_mean).reshape(1, -1)) / torch.from_numpy(y_std).reshape(1, -1)

    return X, x_mean, x_std, Y, y_mean, y_std


if __name__ == '__main__':

    data_path = args.data_path
    save_path = os.path.join(args.root_dir, args.ckpt_name)
    if not exists(save_path):
        os.makedirs(save_path)
    X, x_mean, x_std, Y, y_mean, y_std = create_dataset(data_path, save_path)
    train_dataset = TensorDataset(X.float(), Y.float())
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True, drop_last=True)

    trainer = Trainer(dataloader=train_loader, args=args, output_size=Y.shape[1], x_mean=x_mean, x_std=x_std,
                      y_mean=y_mean,
                      y_std=y_std)
    trainer.train()

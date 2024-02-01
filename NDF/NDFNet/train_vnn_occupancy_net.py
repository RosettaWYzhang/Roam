
import os, os.path as osp
import configargparse
import torch
from torch.utils.data import DataLoader

import vnn_occupancy_net_pointnet_dgcnn as vnn_occupancy_network
import summaries as summaries, losses as losses, training as training, dataio


p = configargparse.ArgumentParser()
p.add('-c', '--config_filepath', required=False, is_config_file=True, help='Path to config file.')

p.add_argument('--logging_root', type=str, default=osp.join('trained_models/', 'ndf_vnn'), help='root for logging')
p.add_argument('--obj_class', type=str, required=True,
               help='bottle, mug, bowl, all')
p.add_argument('--experiment_name', type=str, required=True,
               help='Name of subdirectory in logging_root where summaries and checkpoints will be saved.')

p.add_argument('--sidelength', type=int, default=128)

# General training options
p.add_argument('--batch_size', type=int, default=16)
p.add_argument('--lr', type=float, default=1e-4, help='learning rate. default=5e-5')
p.add_argument('--num_epochs', type=int, default=150,
               help='Number of epochs to train for.')

p.add_argument('--epochs_til_ckpt', type=int, default=10,
               help='Time interval in seconds until checkpoint is saved.')
p.add_argument('--steps_til_summary', type=int, default=500,
               help='Time interval in seconds until tensorboard summary is saved.')
p.add_argument('--iters_til_ckpt', type=int, default=10000,
               help='Training steps until save checkpoint')
p.add_argument('--checkpoint_path', default=None, help='Checkpoint to trained model.')
p.add_argument('--dgcnn', action='store_true', help='If you want to use a DGCNN encoder instead of pointnet (requires more GPU memory)')
opt = p.parse_args()


train_dataset = dataio.JointOccTrainDataset(phase='train', obj_class=opt.obj_class)
val_dataset = dataio.JointOccTrainDataset(phase='val', obj_class=opt.obj_class)


train_dataloader = DataLoader(train_dataset, batch_size=opt.batch_size, shuffle=True,
                              drop_last=True, num_workers=6)
val_dataloader = DataLoader(val_dataset, batch_size=opt.batch_size, shuffle=True,
                            drop_last=True, num_workers=4)
if opt.dgcnn:
    model_type = "dgcnn"
else:
    model_type = "pointnet"
model = vnn_occupancy_network.VNNOccNet(latent_dim=256, model_type=model_type).cuda()

if opt.checkpoint_path is not None:
    model.load_state_dict(torch.load(opt.checkpoint_path))

# model_parallel = nn.DataParallel(model, device_ids=[0, 1, 2, 3])
model_parallel = model

# Define the loss
root_path = os.path.join(opt.logging_root, opt.experiment_name)

# Define the loss
summary_fn = summaries.occupancy_net
root_path = os.path.join(opt.logging_root, opt.experiment_name)
loss_fn = val_loss_fn = losses.occupancy_net

training.train(model=model_parallel, train_dataloader=train_dataloader, val_dataloader=val_dataloader, epochs=opt.num_epochs,
               lr=opt.lr, steps_til_summary=opt.steps_til_summary, epochs_til_checkpoint=opt.epochs_til_ckpt,
               model_dir=root_path, loss_fn=loss_fn, iters_til_checkpoint=opt.iters_til_ckpt, summary_fn=summary_fn,
               clip_grad=False, val_loss_fn=val_loss_fn, overwrite=True)


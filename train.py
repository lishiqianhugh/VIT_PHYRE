import os

import torch
from torch import nn
from torch.autograd import Variable
from torch.nn.functional import one_hot
# from vit_pytorch import ViT
import timm
import argparse
from time import strftime
T = strftime("%m-%d-%Y-%H-%M-%S")

import numpy as np
from dataset.phyreo import PHYREO
import logging


def arg_parse():
    parser = argparse.ArgumentParser(description='VIT_PHYRE Parameters')
    parser.add_argument('--protocal', required=True, type=str, help='within or cross', default='within')
    parser.add_argument('--fold', required=True, type=int, help='from 0 to 9', default=0)
    parser.add_argument('--batch_size', type=int, help='batch size', default=128)

    return parser.parse_args()


args = arg_parse()


# hyperparameters
model_mode = 'pretrained'  # pretrained
data_mode = 'b'  # s / b
epoch = 10
save_interval = 1
lr = 0.0001
batch_size = args.batch_size
protocal = args.protocal
fold_id = args.fold

# device
if torch.cuda.is_available():
    device = 'cuda'
else:
    device = 'cpu'

# model and optimizer
if model_mode == 'pretrained':
    print("Loading pretrained VIT model...")
    model = timm.create_model('vit_base_patch16_224', pretrained=True)
    model.head = nn.Linear(768, 2)
    model.to(device)
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=1, eta_min=1e-6)
else:
    model = ViT(
        image_size = 224,
        patch_size = 16,
        num_classes = 2,
        dim = 512,
        depth = 6,
        heads = 8,
        mlp_dim = 512
    )
    model.to(device)
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=1, eta_min=1e-6)

# loss function
# loss_fn = torch.nn.CrossEntropyLoss()
loss_fn = torch.nn.BCELoss()

exp_dir = f'exp_{protocal}_{fold_id}_{epoch}_{batch_size}_{lr}'
if not os.path.exists(exp_dir):
    os.mkdir(exp_dir)
else:
    print(f"Resuming from {exp_dir}")
    model.load_state_dict(torch.load(f'{exp_dir}/{protocal}{fold_id}_{batch_size}.pt'))

# log
LOG_FORMAT = "%(asctime)s - %(levelname)s - %(message)s"
DATE_FORMAT = "%m/%d/%Y %H:%M:%S %p"
logging.basicConfig(filename=f'{exp_dir}/exp.log', level=logging.DEBUG, format=LOG_FORMAT, datefmt=DATE_FORMAT)

save_path = f'./{exp_dir}/{protocal}{fold_id}_{batch_size}'

load_path = f'./dataset/{protocal}{fold_id}_{batch_size}.pt'


hp = {"model_mode":model_mode, "data_mode":data_mode, "epoch":epoch, "batch_size":batch_size, "lr":lr, "save": save_path, "load_data": load_path}
logging.info(hp)


print("loading train dataset...")
train_loader = torch.load(load_path)

# train
epoch_loss = []
best_loss = 100.
for i in range(epoch):
    sum_loss = []
    for batch_idx, (data, label) in enumerate(train_loader):
        # data = Variable(data, requires_grad=True)
        data = data.to(device)
        label = label.to(device)
        label_one_hot = one_hot(label.to(torch.int64), 2).float().to(device)
        # label = label.unsqueeze(1)
        opt.zero_grad()
        out = model(data).squeeze(dim=-1)
        out = nn.Softmax(1)(out)
        # print(out)
        pred = torch.argmax(out, dim=-1).float()
        acc = (pred == label).sum() / batch_size
        # print(out, label_one_hot)
        loss = loss_fn(out, label_one_hot)
        loss.backward()
        opt.step()
        scheduler.step()
        sum_loss.append(loss.cpu().detach().numpy())
        print(f'epoch {i} batch {batch_idx} acc: {acc.cpu().detach().numpy():.3f} loss: {loss:.4f}')

    mean_loss = np.mean(sum_loss)
    info = f"#######  epoch {i} : {mean_loss}  #########"
    print(info)
    logging.info(info)

    # save the improved network
    if i % save_interval == 0:
        torch.save(model.state_dict(), save_path + f'_{i+1}.pt')
        if mean_loss < best_loss:
            best_loss = mean_loss

print(f"\nloss of each epoch: {epoch_loss} \nbest loss: {best_loss}")
logging.info(f"\nloss of each epoch: {epoch_loss} \nbest loss: {best_loss}")
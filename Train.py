import os
import random
import numpy as np
from collections import OrderedDict
from tqdm import tqdm
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import optim
from torch.utils.data import DataLoader
from dataloder.data_loder import llvip
from Model.common import clamp, gradient
from Model.model import F_Net
from Model.cls_model import Illumination_classifier
import torch.backends.cudnn as cudnn

def init_seeds(seed=2024):

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    cudnn.benchmark, cudnn.deterministic = (False, True) if seed == 0 else (True, False)


def load_cls_model(model, checkpoint_path, device='cuda:0'):

    state_dict = torch.load(checkpoint_path, map_location=device)
    new_state_dict = OrderedDict()
    for k, v in state_dict.items():
        new_state_dict[k.replace('module.', '')] = v
    model.load_state_dict(new_state_dict)
    return model.eval()


class L_TV(nn.Module):

    def __init__(self, TVLoss_weight=1.0):
        super(L_TV, self).__init__()
        self.TVLoss_weight = TVLoss_weight

    def forward(self, x):
        batch_size = x.size(0)
        h_x, w_x = x.size(2), x.size(3)
        count_h = (h_x - 1) * w_x
        count_w = h_x * (w_x - 1)

        h_tv = torch.pow((x[:, :, 1:, :] - x[:, :, :h_x - 1, :]), 2).sum()
        w_tv = torch.pow((x[:, :, :, 1:] - x[:, :, :, :w_x - 1]), 2).sum()
        return (self.TVLoss_weight * 2 * (h_tv / count_h + w_tv / count_w) / batch_size).mean()

def train(model, cls_model, train_loader, optimizer, scaler, save_path, epochs=30, device='cuda:0'):

    tv_loss = L_TV()
    model.train()

    for epoch in range(epochs):
        train_tqdm = tqdm(train_loader, total=len(train_loader), ascii=True)

        for vis_rain_y, vis_gt_y, inf_image, vis_cb_image, vis_cr_image, name, vis_rain, vis_clip in train_tqdm:
            vis_rain = vis_rain.to(device)
            vis_rain_y = vis_rain_y.to(device)
            vis_gt_y = vis_gt_y.to(device)
            inf_image = inf_image.to(device)
            vis_clip = vis_clip.to(device)

            # Mixed precision
            with torch.cuda.amp.autocast():
                _, feature = cls_model(vis_clip)
                fused = model(vis_rain_y, inf_image, feature)

                loss_f = (
                    50 * F.l1_loss(gradient(fused), torch.max(gradient(vis_gt_y), gradient(inf_image))) +
                    40 * F.l1_loss(fused, torch.max(vis_gt_y, inf_image))
                )
                loss = loss_f

            optimizer.zero_grad()
            scaler.scale(loss).backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            scaler.step(optimizer)
            scaler.update()

            train_tqdm.set_postfix(epoch=epoch,
                                   loss=loss.item(),
                                   loss_f=loss_f.item())

        # Save checkpoint
        checkpoint_file = os.path.join(save_path, f"model_{epoch}.pth")
        torch.save(model.state_dict(), checkpoint_file)


# =========================== Main ===========================

if __name__ == '__main__':
    # Init Seeds
    init_seeds(2024)

    # Paths & Params
    datasets_dir = ''           # Your dataset path
    save_dir = 'runs/'
    cls_model_path = 'runs/best_cls.pth'
    batch_size = 2
    num_workers = 1
    lr = 1e-4
    num_epochs = 30
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    # Model Init
    cls_model = Illumination_classifier(input_channels=3).to(device)
    cls_model = load_cls_model(cls_model, cls_model_path, device)

    model = torch.nn.DataParallel(F_Net().to(device))
    optimizer = optim.Adam(model.parameters(), lr=lr)
    scaler = torch.cuda.amp.GradScaler()

    # Dataset & Loader
    train_ds = llvip(datasets_dir)
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True,
                               num_workers=num_workers, pin_memory=True)

    # Save Path
    os.makedirs(save_dir, exist_ok=True)

    # Start Training
    train(model, cls_model, train_loader, optimizer, scaler, save_dir, epochs=num_epochs, device=device)

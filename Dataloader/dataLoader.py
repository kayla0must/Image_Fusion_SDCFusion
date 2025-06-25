import os
import numpy as np
from PIL import Image
from torch.utils import data
from torchvision import transforms
from model.common import RGB2YCrCb

to_tensor = transforms.Compose([transforms.ToTensor()])


class llvip(data.Dataset):

    def __init__(self, data_dir, transform=to_tensor,
                 target_size=(480, 480), clip_size=(224, 224)):
        super().__init__()

        self.data_dir = data_dir
        self.transform = transform
        self.target_size = target_size
        self.clip_size = clip_size

        self.inf_path = None
        self.vis_rain = None
        self.vis_gt = None

        for sub_dir in os.listdir(data_dir):
            full_path = os.path.join(data_dir, sub_dir)
            if sub_dir == 'Inf':
                self.inf_path = full_path
            elif sub_dir == 'Vis':
                self.vis_rain = full_path
            elif sub_dir == 'gt':
                self.vis_gt = full_path

        if not all([self.inf_path, self.vis_rain, self.vis_gt]):
            raise ValueError(f"Missing required subfolders in {data_dir}. Must contain 'Inf', 'Vis', and 'gt'.")

        self.name_list = os.listdir(self.vis_gt)

    def __getitem__(self, index):
        name = self.name_list[index]

        inf_image = self._load_image(self.inf_path, name, mode='L', size=self.target_size)
        vis_rain = self._load_image(self.vis_rain, name, size=self.target_size)
        vis_gt = self._load_image(self.vis_gt, name, size=self.target_size)

        inf_image = self.transform(inf_image)
        vis_rain = self.transform(vis_rain)
        vis_gt = self.transform(vis_gt)

        vis_rain_y, vis_cb_image, vis_cr_image = RGB2YCrCb(vis_rain)
        vis_gt_y, _, _ = RGB2YCrCb(vis_gt)

        vis_clip_img = self._load_image(self.vis_rain, name, size=self.clip_size)
        vis_clip = self.transform(vis_clip_img)

        return vis_rain_y, vis_gt_y, inf_image, vis_cb_image, vis_cr_image, name, vis_rain, vis_clip

    def __len__(self):

        return len(self.name_list)

    @staticmethod
    def _load_image(folder, filename, mode='RGB', size=(480, 480)):

        img = Image.open(os.path.join(folder, filename)).convert(mode)
        return img.resize(size)


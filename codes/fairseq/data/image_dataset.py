import os
import torch
import random
random.seed(42)

class ImageDataset(torch.utils.data.Dataset):
    """
    For loading image datasets
    """
    def __init__(self, feat_path: str, mask_path: str):
        self.img_feat = torch.load(feat_path)
        self.img_feat_mask = None
        if os.path.exists(mask_path):
            self.img_feat_mask = torch.load(mask_path)

        self.size = self.img_feat.shape[0]
        self.num_lang = 3

    def __getitem__(self, idx):
        if self.img_feat_mask is None:
            return self.img_feat[idx % self.size], None
        else:
            return self.img_feat[idx % self.size], self.img_feat_mask[idx % self.size]

    # def __getitem__(self, idx):
    #     if self.img_feat_mask is None:
    #         # # tmp = self.img_feat[random.randint(0,self.size -1)]
    #         # # print(tmp.size())
    #         # return torch.zeros([197,768]), None
    #         return self.img_feat[random.randint(0,self.size -1)], None
    #     else:
    #         return self.img_feat[0], self.img_feat_mask[0]

    def __len__(self):
        return self.size * self.num_lang

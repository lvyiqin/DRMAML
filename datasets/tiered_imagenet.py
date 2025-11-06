import os
import pickle

import torch
from torch.utils.data import Dataset
import numpy as np
from PIL import Image

from .datasets import register
from .transforms import get_transform
import pdb

@register('tiered-imagenet')
class TieredImageNet(Dataset):
  def __init__(self, root_path, split='train', image_size=224,
               normalization=True, transform=None):
    super(TieredImageNet, self).__init__()
    split_dict = {'train': 'train_phase_train',        # standard train
                  'val': 'train_phase_val',            # standard val
                  'trainval': 'train_phase_trainval',  # standard train and val
                  'test': 'train_phase_test',          # standard test
                  'meta-train': 'train_phase_train',   # meta-train
                  'meta-val': 'val',                   # meta-val
                  'meta-test': 'test',                 # meta-test
                 }

    split_tag = split_dict[split]
    split_file = os.path.join(root_path, "tieredImageNet_category_split_"+split_tag + '.pt')
    # pdb.set_trace()
    assert os.path.isfile(split_file)
    with open(split_file, 'rb') as f:
      pack = torch.load(f)
    data, label = pack['data'], pack['labels']

    # pack.keys()
    # pack["catname2label"]
    # pack["label2catname"]
    # import pdb
    # pdb.set_trace()
    # torch.save(pack["catname2label"], "miniImageNet/miniImageNet_category_split_"+split_tag+'.pt')

    data = [Image.fromarray(x.numpy()) for x in data]
    label = np.array(label)
    label_key = sorted(np.unique(label))
    # pdb.set_trace()
    label_map = dict(zip(label_key, range(len(label_key))))
    new_label = np.array([label_map[x] for x in label])

    self.root_path = root_path
    self.split_tag = split_tag
    self.image_size = image_size

    self.data = data
    self.label = new_label
    self.n_classes = len(label_key)


    # pdb.set_trace()


    if normalization:
      self.norm_params = {'mean': [0.471, 0.450, 0.403],
                          'std':  [0.278, 0.268, 0.284]}
    else:
      self.norm_params = {'mean': [0., 0., 0.],
                          'std':  [1., 1., 1.]}

    self.transform = get_transform(transform, image_size, self.norm_params)

    def convert_raw(x):
      mean = torch.tensor(self.norm_params['mean']).view(3, 1, 1).type_as(x)
      std = torch.tensor(self.norm_params['std']).view(3, 1, 1).type_as(x)
      return x * std + mean
    
    self.convert_raw = convert_raw

    # pdb.set_trace()

  def __len__(self):
    return len(self.data)

  def __getitem__(self, index):
    image = self.transform(self.data[index])
    label = self.label[index]
    return image, label




@register('meta-tiered-imagenet')
class MetaTieredImageNet(TieredImageNet):
  def __init__(self, root_path, split='train', image_size=224,
               normalization=True, transform=None, val_transform=None,
               n_batch=200, n_episode=4, n_way=5, n_shot=1, n_query=15, n_pick=4):
    super(MetaTieredImageNet, self).__init__(root_path, split, image_size,
                                           normalization, transform)
    self.n_batch = n_batch
    self.n_episode = n_episode
    self.n_way = n_way
    self.n_shot = n_shot
    self.n_query = n_query
    self.split = split

    self.catlocs = tuple()
    for cat in range(self.n_classes):
      self.catlocs += (np.argwhere(self.label == cat).reshape(-1),)
    # pdb.set_trace()

    self.val_transform = get_transform(val_transform, image_size, self.norm_params)

  def __len__(self):
    return self.n_batch * self.n_episode

  def __getitem__(self, index):
    # constructing a single task
    shot, query = [], []
    cats = np.random.choice(self.n_classes, self.n_way, replace=False)
    # classnames = [self.classid_names[cat] for cat in cats]
    # print(cats)
    # pdb.set_trace()
    for c in cats:
      c_shot, c_query = [], []
      idx_list = np.random.choice(self.catlocs[c], self.n_shot + self.n_query, replace=False)
      shot_idx, query_idx = idx_list[:self.n_shot], idx_list[-self.n_query:]
      for idx in shot_idx:
        c_shot.append(self.transform(self.data[idx]))
      for idx in query_idx:
        c_query.append(self.val_transform(self.data[idx]))
      shot.append(torch.stack(c_shot))
      query.append(torch.stack(c_query))
    
    shot = torch.cat(shot, dim=0)             # [n_way * n_shot, C, H, W]
    query = torch.cat(query, dim=0)           # [n_way * n_query, C, H, W]
    cls = torch.arange(self.n_way)[:, None]   # [0, 1, 2, 3, 4]
    shot_labels = cls.repeat(1, self.n_shot).flatten()    # [n_way * n_shot]
    query_labels = cls.repeat(1, self.n_query).flatten()  # [n_way * n_query]
    # pdb.set_trace()

    if self.split == 'meta-val':
      cats = cats + 351
    if self.split == 'meta-test':
      cats = cats + 351 + 97

    return shot, query, shot_labels, query_labels, torch.Tensor(cats)
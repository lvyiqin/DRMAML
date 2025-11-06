import os

import torch
import pdb



datasets = {}

def register(name):
  def decorator(cls):
    datasets[name] = cls
    return cls
  return decorator


def make(name, dataset_root, **kwargs):
  DEFAULT_ROOT = dataset_root

  if 'meta-imagenetc' in name:
      datasetname = 'meta-imagenetc'
  else:
      datasetname = name



  if kwargs.get('root_path') is None:
    kwargs['root_path'] = os.path.join(DEFAULT_ROOT, name.replace('meta-', ''))
  # pdb.set_trace()
  dataset = datasets[datasetname](**kwargs)
  return dataset


def collate_fn(batch):
  shot, query, shot_label, query_label, cats = [], [], [], [], []
  # assert(shot==query)
  for s, q, sl, ql, cat in batch:
    shot.append(s)
    query.append(q)
    shot_label.append(sl)
    query_label.append(ql)
    cats.append(cat)
  shot = torch.stack(shot)                # [n_ep, n_way * n_shot, C, H, W]
  query = torch.stack(query)              # [n_ep, n_way * n_query, C, H, W]
  shot_label = torch.stack(shot_label)    # [n_ep, n_way * n_shot]
  query_label = torch.stack(query_label)  # [n_ep, n_way * n_query]
  cats = torch.stack(cats)    # [n_ep, n_way]
  
  return (shot, query, shot_label, query_label, cats)
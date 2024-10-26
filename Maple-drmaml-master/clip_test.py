import argparse
import random

import yaml
import torch
import torch.nn as nn
import numpy as np
from tqdm import tqdm
from torch.utils.data import DataLoader
from clip import clip
import datasets
import models
import utils
import pdb


fnames = 'LOC_synset_mapping.txt'
all_classes = {}
with open(fnames, 'r') as f:
  lines = f.readlines()
for line in lines:
  class_id  = line.strip().split(' ')[0]
  all_classes[class_id] = line[10:].strip()

# if split == 'meta-train':
#   filename = "miniImageNet_category_split_train_phase_train"
# elif split == 'meta-val':
#   filename = "miniImageNet_category_split_val"
# elif split == 'meta-test':
filename = "miniImageNet_category_split_test"

data = torch.load('../' + filename + '.pt')
label_key = sorted(np.unique(list(data.values())))
# pdb.set_trace()
# label_map = dict(zip(label_key, range(len(label_key))))
classid_names = {}
for key in data.keys():
  classid_names[data[key]] = all_classes[key]

classnames = []
for i in label_key:
  classnames.append(classid_names[i])

# pdb.set_trace()

def load_clip_to_cpu(backbone_name):
  # backbone_name = 'ViT-B/16'
  # backbone_name = "RN50"
  url = clip._MODELS[backbone_name]
  model_path = clip._download(url)

  try:
    # loading JIT archive
    model = torch.jit.load(model_path, map_location="cpu").eval()
    state_dict = None

  except RuntimeError:
    state_dict = torch.load(model_path, map_location="cpu")

  model = clip.build_model(state_dict or model.state_dict())

  return model

def main(config):
  random.seed(0)
  np.random.seed(0)
  torch.manual_seed(0)
  torch.cuda.manual_seed(0)
  # torch.backends.cudnn.deterministic = True
  # torch.backends.cudnn.benchmark = False

  ##### Dataset #####

  dataset = datasets.make(config['dataset'], **config['test'])
  utils.log('meta-test set: {} (x{}), {}'.format(
    dataset[0][0].shape, len(dataset), dataset.n_classes))
  loader = DataLoader(dataset, config['test']['n_episode'],
    collate_fn=datasets.collate_fn, num_workers=1, pin_memory=True)

  ##### RisklearnerModel #####

  clip_model = load_clip_to_cpu(config['backbone'])
  inner_args = utils.config_inner_args(config.get('inner_args'))
  clip_model.float()

  print("Building custom CLIP")
  # pdb.set_trace()
  model = models.CLIPMAML(clip_model, classnames)
  model.cuda()

  if args.efficient:
    model.go_efficient()

  if config.get('_parallel'):
    model = nn.DataParallel(model)

  utils.log('num params: {}'.format(utils.compute_n_params(model)))

  ##### Evaluation #####

  model.eval()
  aves_va = utils.AverageMeter()
  va_lst = []

  for epoch in range(1, config['epoch'] + 1):
    for data in tqdm(loader, leave=False):
      # pdb.set_trace()
      x_shot, x_query, y_shot, y_query, cats = data
      x_shot, y_shot = x_shot.cuda(), y_shot.cuda()
      x_query, y_query = x_query.cuda(), y_query.cuda()
      cats = cats.cuda()

      if inner_args['reset_classifier']:
        if config.get('_parallel'):
          model.module.reset_classifier()
        else:
          model.reset_classifier()
      # pdb.set_trace()
      logits = model(x_shot, x_query, y_shot, cats, inner_args, meta_train=False)
      logits = logits.view(-1, config['test']['n_way'])
      labels = y_query.view(-1)
      
      pred = torch.argmax(logits, dim=1)
      acc = utils.compute_acc(pred, labels)
      aves_va.update(acc, 1)
      va_lst.append(acc)

    print('test epoch {}: acc={:.2f} +- {:.2f} (%)'.format(
      epoch, aves_va.item() * 100, 
      utils.mean_confidence_interval(va_lst) * 100))


if __name__ == '__main__':
  parser = argparse.ArgumentParser()
  parser.add_argument('--config', 
                      help='configuration file')
  parser.add_argument('--gpu', 
                      help='gpu device number', 
                      type=str, default='0')
  parser.add_argument('--efficient', 
                      help='if True, enables gradient checkpointing',
                      action='store_true')
  # parser.add_argument('--backbone', default='ViT-B/16', type=str)
  args = parser.parse_args()
  config = yaml.load(open(args.config, 'r'), Loader=yaml.FullLoader)
  
  if len(args.gpu.split(',')) > 1:
    config['_parallel'] = True
    config['_gpu'] = args.gpu

  utils.set_gpu(args.gpu)
  main(config)
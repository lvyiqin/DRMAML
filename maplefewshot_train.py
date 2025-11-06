import argparse
import os
import random
from collections import OrderedDict

import pdb
import yaml
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from tqdm import tqdm
from torch.utils.data import DataLoader
from tensorboardX import SummaryWriter

from clip import clip

import datasets
import models
import utils

from optim import build_optimizer, build_lr_scheduler
from torch.optim.lr_scheduler import CosineAnnealingLR, StepLR

import drmaml


fnames = 'clip_text_feat/LOC_synset_mapping.txt'
all_classes = {}
with open(fnames, 'r') as f:
  lines = f.readlines()
for line in lines:
  class_id  = line.strip().split(' ')[0]
  all_classes[class_id] = line[10:].strip()


# ====collecting all class name in the clip_text_feat

datasetname_dict = {}

# 100 classes
datasetname_dict['meta-mini-imagenet'] = 'miniImageNet/miniImageNet'
datasetname_dict['meta-tiered-imagenet'] = 'tieredImageNet/tieredImageNet'
# 200 classes
datasetname_dict['meta-imageneta'] = 'ImageNetA/ImageNetA'
datasetname_dict['meta-imagenetr'] = 'ImageNetR/ImageNetR'

# 1000 classes
datasetname_dict['meta-imagenet'] = 'ImageNet/ImageNet'
datasetname_dict['meta-imagenetsketch'] = 'ImageNetSket/ImageNetSket'
datasetname_dict['meta-imagenetc-brightness'] = 'ImagenetC/imagenetc-brightness-ImageNetC'
datasetname_dict['meta-imagenetc-contrast'] = 'ImagenetC/imagenetc-contrast-ImageNetC'
datasetname_dict['meta-imagenetc-defocus_blur'] = 'ImagenetC/imagenetc-defocus_blur-ImageNetC'
datasetname_dict['meta-imagenetc-elastic_transform'] = 'ImagenetC/imagenetc-elastic_transform-ImageNetC'
datasetname_dict['meta-imagenetc-fog'] = 'ImagenetC/imagenetc-fog-ImageNetC'
datasetname_dict['meta-imagenetc-frost'] = 'ImagenetC/imagenetc-frost-ImageNetC'
datasetname_dict['meta-imagenetc-gaussian_noise'] = 'ImagenetC/imagenetc-gaussian_noise-ImageNetC'
datasetname_dict['meta-imagenetc-glass_blur'] = 'ImagenetC/imagenetc-glass_blur-ImageNetC'
datasetname_dict['meta-imagenetc-impulse_noise'] = 'ImagenetC/imagenetc-impulse_noise-ImageNetC'
datasetname_dict['meta-imagenetc-jpeg_compression'] = 'ImagenetC/imagenetc-jpeg_compression-ImageNetC'
datasetname_dict['meta-imagenetc-motion_blur'] = 'ImagenetC/imagenetc-motion_blur-ImageNetC'
datasetname_dict['meta-imagenetc-pixelate'] = 'ImagenetC/imagenetc-pixelate-ImageNetC'
datasetname_dict['meta-imagenetc-shot_noise'] = 'ImagenetC/imagenetc-shot_noise-ImageNetC'
datasetname_dict['meta-imagenetc-snow'] = 'ImagenetC/imagenetc-snow-ImageNetC'
datasetname_dict['meta-imagenetc-zoom_blur'] = 'ImagenetC/imagenetc-zoom_blur-ImageNetC'



# def task_descriptor_generation(cats):
#     candidate_task_num = cats.shape[0]
#     task_descriptor = mat[cats.long()]
#     task_descriptor = task_descriptor.view(candidate_task_num, -1)
#     return task_descriptor
#
def task_descriptor_generation(mat, cats):
  task_descriptor = mat[cats.long()]
  task_descriptor = task_descriptor.mean(1)
  return task_descriptor

def load_clip_to_cpu(cfg):
  backbone_name = cfg['backbone']
  url = clip._MODELS[backbone_name]
  model_path = clip._download(url)

  try:
    # loading JIT archive
    model = torch.jit.load(model_path, map_location="cpu").eval()
    state_dict = None

  except RuntimeError:
    state_dict = torch.load(model_path, map_location="cpu")
  design_details = {"trainer": 'MaPLe',
                    "vision_depth": 0,
                    "language_depth": 0, "vision_ctx": 0,
                    "language_ctx": 0,
                    "maple_length": cfg['TRAINER']['MAPLE']['N_CTX']}
  model = clip.build_model(state_dict or model.state_dict(), design_details)

  return model

#=============================================================================================================
def main(config):
    # ====collecting all class name for CLIP text encoder
    pwd = datasetname_dict[config['dataset']]

    filename = 'clip_text_feat/'+ pwd +"_category_split_train_phase_train"
    data = torch.load(filename + '.pt')
    label_key = sorted(np.unique(list(data.values())))
    classid_names = {}
    for key in data.keys():
        classid_names[data[key]] = all_classes[key]

    filename = 'clip_text_feat/'+ pwd + "_category_split_val"
    datav = torch.load(filename + '.pt')
    label_keyv = sorted(np.unique(list(datav.values())))
    for key in datav.keys():
        classid_names[datav[key]] = all_classes[key]

    filename = 'clip_text_feat/'+ pwd + "_category_split_test"
    datat = torch.load(filename + '.pt')
    label_keyt = sorted(np.unique(list(datat.values())))
    for key in datat.keys():
        classid_names[datat[key]] = all_classes[key]

    classnames = []
    for i in label_key:
        classnames.append(classid_names[i])
    for i in label_keyv:
        classnames.append(classid_names[i])
    for i in label_keyt:
        classnames.append(classid_names[i])

    # ====class_descriptor
    filename = 'clip_text_feat/' + pwd + "_category_split_train_phase_train"
    mark_to_cat = torch.load(filename + '.pt')
    mark_to_feature_all = torch.load('clip_text_feat/imagenet_1000classes_mark_2_textfeats.pt')

    n_class_train = len(mark_to_cat)
    print(pwd, n_class_train)

    mat = torch.zeros(n_class_train, 512)
    for key in mark_to_cat:
        cat = mark_to_cat[key]
        mat[cat] = mark_to_feature_all[key]
    mat = mat.cuda()

    # ====setup for experiments
    random.seed(0)
    np.random.seed(0)
    torch.manual_seed(0)
    torch.cuda.manual_seed(0)
    # torch.backends.cudnn.deterministic = True
    # torch.backends.cudnn.benchmark = False
    # pdb.set_trace()

    ckpt_name = args.name
    if ckpt_name is None:
        if config['backbone'] == "ViT-B/16":
            ckpt_name = "ViT-B-16"
        else:
            ckpt_name = config['backbone']
        ckpt_name += '_' + config['dataset'].replace('meta-', '')
        ckpt_name += '_{}_way_{}_shot_'.format(config['train']['n_way'], config['train']['n_shot'])
        ckpt_name += config['sampling_strategy']

        ckpt_name += '_balanceW={}'.format(config['balance_text_visual'])

    if args.tag is not None:
        ckpt_name += '_' + args.tag

    ckpt_path = os.path.join(args.log_date, ckpt_name)
    save_model_path = os.path.join(args.model_log, ckpt_name)
    utils.ensure_path(ckpt_path)
    utils.set_log_path(ckpt_path)

    # utils.ensure_path(save_model_path)
    utils.set_log_path2(save_model_path)

    writer = SummaryWriter(os.path.join(ckpt_path, 'tensorboard'))
    yaml.dump(config, open(os.path.join(ckpt_path, 'config.yaml'), 'w'))

    ##### Dataset #####
    # ====meta-trains
    train_set = datasets.make(config['dataset'], config["dataset_root"], **config['train'])
    utils.log2('meta-train set: {} (x{}), {}'.format(train_set[0][0].shape, len(train_set), train_set.n_classes))
    train_loader = DataLoader(train_set, config['train']['n_episode'],
                              collate_fn=datasets.collate_fn, num_workers=1, pin_memory=True)

    # ====meta-val
    eval_val = False
    if config.get('val'):
        eval_val = True
        val_set = datasets.make(config['dataset'], config["dataset_root"], **config['val'])
        utils.log('meta-val set: {} (x{}), {}'.format(val_set[0][0].shape, len(val_set), val_set.n_classes))
        val_loader = DataLoader(val_set, config['val']['n_episode'],
            collate_fn=datasets.collate_fn, num_workers=1, pin_memory=True)


    ##### Optimizer #####
    clip_model = load_clip_to_cpu(config)
    inner_args = utils.config_inner_args(config.get('inner_args'))
    clip_model.float()

    print("Building custom CLIP")
    model = models.VMAPLE(config, clip_model, classnames)
    model.float()
    model.cuda()

    name_to_update = "prompt_learner"
    for name, param in model.named_parameters():
        if name_to_update not in name:
            # Make sure that VPT prompts are updated
            if "VPT" in name: param.requires_grad_(True)
            else: param.requires_grad_(False)

    optimizer = build_optimizer(model, config['OPTIM'])
    lr_scheduler = build_lr_scheduler(optimizer, config['OPTIM'])

    start_epoch = 1
    max_va = 0.

    if args.efficient:
        model.go_efficient()

    if config.get('_parallel'):
        model = nn.DataParallel(model)

    utils.log('num params: {}'.format(utils.compute_n_params(model)))
    timer_elapsed, timer_epoch = utils.Timer(), utils.Timer()

    ##### Optimizer #####
    sampling_strategy = config["sampling_strategy"]

    ##### Training and evaluation #####
    # 'tl': meta-train loss
    # 'ta': meta-train accuracy
    # 'vl': meta-val loss
    # 'va': meta-val accuracy
    aves_keys = ['tl', 'ta', 'vl', 'va']
    trlog = dict()
    for k in aves_keys:
        trlog[k] = []

    for epoch in range(start_epoch, config['epoch'] + 1):
        timer_epoch.start()
        aves = {k: utils.AverageMeter() for k in aves_keys}

        # meta-train=========================================================
        model.train()
        writer.add_scalar('lr', optimizer.param_groups[0]['lr'], epoch)
        torch.manual_seed(epoch)

        for data in tqdm(train_loader, desc='meta-train', leave=False):
          x_shot, x_query, y_shot, y_query, cats = data
          n_pick = config['train']['n_pick']

          # ====sample n_pick from n_episode candidate tasks.
          if sampling_strategy == "random":
              x_shot, y_shot = x_shot[:n_pick].cuda(), y_shot[:n_pick].cuda()   # [n_pick, 5, 3, 224, 224]
              x_query, y_query = x_query[:n_pick].cuda(), y_query[:n_pick].cuda() # [n_pick, 75, 3, 224, 224]
              cats = cats[:n_pick].cuda()

              logits = model(x_shot, x_query, y_shot, cats, inner_args, meta_train=True)  # [n_pick, 75, 5]
              logits = logits.flatten(0, 1)
              labels = y_query.flatten()
              loss = F.cross_entropy(logits, labels)

              pred = torch.argmax(logits, dim=-1)
              acc = utils.compute_acc(pred, labels)

          elif sampling_strategy == "worst_case":
              x_shot, y_shot = x_shot.cuda(), y_shot.cuda()
              x_query, y_query = x_query.cuda(), y_query.cuda()
              cats = cats.cuda()

              logits_original = model(x_shot, x_query, y_shot, cats, inner_args, meta_train=True)
              candidate_task_num = logits_original.shape[0]
              query_num_per_task = logits_original.shape[1]
              logits = logits_original.flatten(0, 1)
              labels = y_query.flatten()
              loss_all_tasks = F.cross_entropy(logits, labels, reduction="none")

              loss_all_tasks = loss_all_tasks.view(candidate_task_num, query_num_per_task).mean(1)
              _, indices = loss_all_tasks.sort()
              worst_case_tasks_id = indices[-n_pick:]
              loss = loss_all_tasks[worst_case_tasks_id].mean()

              pred = torch.argmax(logits_original[worst_case_tasks_id].flatten(0,1), dim=-1)
              acc = utils.compute_acc(pred, y_query[worst_case_tasks_id].flatten(0,1))

          elif sampling_strategy == "drmaml":
              x_shot, y_shot = x_shot.cuda(), y_shot.cuda()
              x_query, y_query = x_query.cuda(), y_query.cuda()
              cats = cats.cuda()

              logits_original = model(x_shot, x_query, y_shot, cats, inner_args, meta_train=True)
              candidate_task_num = logits_original.shape[0]
              query_num_per_task = logits_original.shape[1]
              logits = logits_original.flatten(0, 1)
              labels = y_query.flatten()
              loss_all_tasks = F.cross_entropy(logits, labels, reduction="none")
              loss_all_tasks = loss_all_tasks.view(candidate_task_num, query_num_per_task).mean(1)
              loss, worst_case_tasks_id = drmaml.drloss(loss_all_tasks, 0.5)

              pred = torch.argmax(logits_original[worst_case_tasks_id].flatten(0,1), dim=-1)
              acc = utils.compute_acc(pred, y_query[worst_case_tasks_id].flatten(0,1))

          aves['tl'].update(loss.item(), 1)
          aves['ta'].update(acc, 1)

          optimizer.zero_grad()
          loss.backward()
          for param in optimizer.param_groups[0]['params']:
            nn.utils.clip_grad_value_(param, 10)
          optimizer.step()
          optimizer.zero_grad()


        # meta-val=========================================================
        if eval_val:
          model.eval()
          # np.random.seed(0)
          torch.manual_seed(0)

          for data in tqdm(val_loader, desc='meta-val', leave=False):
            x_shot, x_query, y_shot, y_query, cats = data
            x_shot, y_shot = x_shot.cuda(), y_shot.cuda()
            x_query, y_query = x_query.cuda(), y_query.cuda()
            cats = cats.cuda()

            if inner_args['reset_classifier']:
              if config.get('_parallel'):
                model.module.reset_classifier()
              else:
                model.reset_classifier()

            logits = model(x_shot, x_query, y_shot, cats, inner_args, meta_train=False)
            logits = logits.flatten(0, 1)
            labels = y_query.flatten()

            pred = torch.argmax(logits, dim=-1)
            acc = utils.compute_acc(pred, labels)
            loss = F.cross_entropy(logits, labels)
            aves['vl'].update(loss.item(), 1)
            aves['va'].update(acc, 1)

        if lr_scheduler is not None:
            lr_scheduler.step()


        for k, avg in aves.items():
            aves[k] = avg.item()
            trlog[k].append(aves[k])

        t_epoch = utils.time_str(timer_epoch.end())
        t_elapsed = utils.time_str(timer_elapsed.end())
        t_estimate = utils.time_str(timer_elapsed.end() / (epoch - start_epoch + 1) * (config['epoch'] - start_epoch + 1))


        # ====formats output
        log_str = 'epoch {}, meta-train {:.4f}|{:.4f}'.format(str(epoch), aves['tl'], aves['ta'])
        writer.add_scalars('loss', {'meta-train': aves['tl']}, epoch)
        writer.add_scalars('acc', {'meta-train': aves['ta']}, epoch)

        if eval_val:
            log_str += ', meta-val {:.4f}|{:.4f}'.format(aves['vl'], aves['va'])
            writer.add_scalars('loss', {'meta-val': aves['vl']}, epoch)
            writer.add_scalars('acc', {'meta-val': aves['va']}, epoch)

        log_str += ', {} {}/{}'.format(t_epoch, t_elapsed, t_estimate)
        utils.log(log_str)

        # ====saves model and meta-data
        if config.get('_parallel'):
            model_ = model.module
        else:
            model_ = model

        training = {
            'epoch': epoch,
            'max_va': max(max_va, aves['va']),
            'optimizer': config['OPTIM'],
            'optimizer_state_dict': optimizer.state_dict(),
            'lr_scheduler_state_dict': lr_scheduler.state_dict()
            if lr_scheduler is not None else None,}
        ckpt = {
            'file': __file__,
            'config': config,
            'encoder': config['backbone'],
            'encoder_state_dict': model_.encoder.state_dict(),
            'training': training,
        }
        torch.save(trlog, os.path.join(save_model_path, 'trlog.pth'))

        if aves['va'] > max_va:
            max_va = aves['va']
            torch.save(ckpt, os.path.join(save_model_path, 'max-va.pth'))
        writer.flush()

    torch.save(ckpt, os.path.join(save_model_path, 'epoch-last.pth'))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--name',
                      help='model name',
                      type=str, default=None)
    parser.add_argument('--tag',
                      help='auxiliary information',
                      type=str, default=None)
    parser.add_argument('--gpu',
                      help='gpu device number',
                      type=str, default='0')
    parser.add_argument('--log_date',
                      help='log_date',
                      type=str, default='log/test')

    parser.add_argument('--model_log',
                      help='model_log',
                      type=str, default='log/test')

    parser.add_argument('--efficient',
                      help='if True, enables gradient checkpointing',
                      action='store_true')

    parser.add_argument('--balance_text_visual', type=float, default=None)
    #-----------------dataset-----------------
    parser.add_argument('--dataset',type=str, nargs='?',default='imageneta',)
    parser.add_argument('--setting', type=str, nargs='?',default='5_way_1_shot')
    parser.add_argument('--corruption', type=str, default="gaussian_noise")

    #-----------------sampling strategy-----------------
    parser.add_argument('--sampling_strategy',type=str, nargs='?', default='drmaml')
    parser.add_argument('--dataset_root', type=str, default="/data/")
    args = parser.parse_args()
    config_file = 'configs/'+args.dataset+'/'+args.setting+'/train_maple.yaml'
    config = yaml.load(open(config_file, 'r'), Loader=yaml.FullLoader)

    #==========================================
    if args.dataset == "imageneta":
        args.balance_text_visual = args.balance_text_visual

    elif args.dataset == "imagenetsketch":
        args.balance_text_visual = args.balance_text_visual
    elif args.dataset == "imagenetc" and args.corruption == "gaussian_noise":
        args.balance_text_visual = 0.25
    elif args.dataset == "imagenetc" and args.corruption == "impulse_noise":
        args.balance_text_visual = 0.25
    elif args.dataset == "imagenetc" and args.corruption == "shot_noise":
        args.balance_text_visual = 1
    elif args.dataset == "imagenetr":
        args.balance_text_visual = 0.75
    else:
        args.balance_text_visual = args.balance_text_visual

    config["balance_text_visual"] = args.balance_text_visual
    # ==========================================

    assert args.dataset == config["dataset"]
    config['dataset'] = args.dataset
    config['setting'] = args.setting
    config["dataset_root"] = args.dataset_root

    if config['dataset'] == 'imagenetc':
        config['dataset'] = "meta-"+config['dataset'] + '-' + args.corruption
    else:
        config['dataset'] = "meta-"+config['dataset']

    config["sampling_strategy"] = args.sampling_strategy

    print(args)
    if len(args.gpu.split(',')) > 1:
        config['_parallel'] = True
        config['_gpu'] = args.gpu

    utils.set_gpu(args.gpu)
    main(config)
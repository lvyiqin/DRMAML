import argparse
import random

import os
import yaml
import torch
import torch.nn as nn
import numpy as np
from tqdm import tqdm
from torch.utils.data import DataLoader
import datasets
import models
import utils
import pdb
import drmaml


fnames = 'clip_text_feat/LOC_synset_mapping.txt'
all_classes = {}
with open(fnames, 'r') as f:
    lines = f.readlines()
for line in lines:
    class_id = line.strip().split(' ')[0]
    all_classes[class_id] = line[10:].strip()

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

def load_clip_to_cpu(cfg):
    backbone_name = cfg['backbone']
    if cfg["test_model"] == "zero-clip":
        from clip_coop import clip
        url = clip._MODELS[backbone_name]
        model_path = clip._download(url)

        print(model_path)
        try:
            # loading JIT archive
            model = torch.jit.load(model_path, map_location="cpu").eval()
            state_dict = None

        except RuntimeError:
            state_dict = torch.load(model_path, map_location="cpu")
        model = clip.build_model(state_dict or model.state_dict())

    else:
        from clip import clip
        url = clip._MODELS[backbone_name]
        model_path = clip._download(url)

        print(model_path)
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


def main(config):

    # ====collecting all class name for CLIP text encoder
    pwd = datasetname_dict[config['dataset']]


    filename = 'clip_text_feat/'+ pwd + "_category_split_train_phase_train"
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

    random.seed(0)
    np.random.seed(0)
    torch.manual_seed(0)
    torch.cuda.manual_seed(0)
    # torch.backends.cudnn.deterministic = True
    # torch.backends.cudnn.benchmark = False

    ##### Dataset #####
    dataset = datasets.make(config['dataset'], config["dataset_root"], **config['test'])
    utils.log('meta-test set: {} (x{}), {}'.format(
        dataset[0][0].shape, len(dataset), dataset.n_classes))
    loader = DataLoader(dataset, config['test']['n_episode'],
                        collate_fn=datasets.collate_fn, num_workers=1, pin_memory=True)

    ##### Model #####

    clip_model = load_clip_to_cpu(config)
    inner_args = utils.config_inner_args(config.get('inner_args'))
    clip_model.float()
    # pdb.set_trace()

    print("Building custom CLIP")

    if config["test_model"] == "zero-clip":
        model = models.CLIPMAML(config, clip_model, classnames)
        ckpt_name = args.name
    else:
        model = models.VMAPLE(config, clip_model, classnames)
        ckpt_name = args.name
        if ckpt_name is None:
            if config['backbone'] == "ViT-B/16":
                ckpt_name = "ViT-B-16"
            else:
                ckpt_name = config['backbone']
            ckpt_name += '_' + config['dataset'].replace('meta-', '')
            ckpt_name += '_{}_'.format(config['setting'])
            ckpt_name += config['sampling_strategy']
            ckpt_name += '_balanceW={}'.format(config['balance_text_visual'])

        if args.tag is not None:
            ckpt_name += '_' + args.tag
        # pdb.set_trace()
        # save_model_path = os.path.join(args.model_log, ckpt_name, "max-va.pth")
        if config['dataset']=='meta-tiered-imagenet':
            save_model_path = os.path.join(args.model_log, ckpt_name, "max-va.pth")
        else:
            save_model_path = os.path.join(args.model_log, ckpt_name, "max-va.pth")
        # ckpt_path = os.path.join(args.log_date, ckpt_name,  "max-va.pth")
        print(save_model_path)
        model.encoder.load_state_dict(torch.load(save_model_path)['encoder_state_dict'])

        # if config.get('load'):
        #     model.encoder.load_state_dict(torch.load(config['load'])['encoder_state_dict'])

    model.float()
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
    cvar_lst = []

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

        # pdb.set_trace()

        cvar = drmaml.calcu_cvar(va_lst)
        if config["test_model"] == "zero-clip":
            test_dict = {'zero-clip-acc': round(aves_va.item() * 100, 2),
                         'zero-clip-acc_ci': round(utils.mean_confidence_interval(va_lst) * 100, 2),
                         'cvar': round(cvar * 100, 2)}
            print(test_dict)

            if ckpt_name is None:
                if config['backbone'] == "ViT-B/16":
                    ckpt_name = "ViT-B-16"
                else:
                    ckpt_name = config['backbone']
                ckpt_name += '_' + config['dataset'].replace('meta-', '')
                ckpt_name += '_{}_'.format(config['setting'])

            test_path = os.path.join(args.log_date, ckpt_name+"zero_clip.txt")
            fckpt = open(test_path, "w")
            fckpt.write(str(test_dict) + "\n")
            fckpt.close()

        else:
            trlog = os.path.join(args.model_log, ckpt_name, "trlog.pth")

            test_dict = {'epoch': epoch,
                         'acc': round(aves_va.item() * 100, 2),
                         'acc_ci': round(utils.mean_confidence_interval(va_lst) * 100, 2),
                         'cvar': round(cvar * 100, 2)}
                         # "val_best_epoch": q,
                         # "val_best_acc": round(val_best_acc*100, 2)}

            print(test_dict)

            test_path = os.path.join(args.log_date, ckpt_name, "test.txt")
            fckpt = open(test_path, "w")
            fckpt.write(str(test_dict) + "\n")
            fckpt.close()

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    # parser.add_argument('--config',
    #                   help='configuration file')
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
                      type=str, default='0715')

    parser.add_argument('--model_log',
                      help='model_log',
                      type=str, default='log/0716_lambda=0.5')

    parser.add_argument('--efficient',
                      help='if True, enables gradient checkpointing',
                      action='store_true')

    parser.add_argument('--balance_text_visual', type=float, default=0.5)
    #-----------------dataset-----------------
    parser.add_argument('--dataset',type=str, nargs='?',default='imageneta',)
    parser.add_argument('--setting', type=str, nargs='?',default='5_way_1_shot')
    # if parser.dataset == 'ImagenetC':
    parser.add_argument('--corruption', type=str, default="gaussian_noise")

    #-----------------sampling strategy-----------------
    parser.add_argument('--sampling_strategy',type=str, nargs='?', default='ours')
    parser.add_argument('--test_model', type=str, default=None)  # zero-clip

    # -----------------snellius_changed_dataset_meta-----------------
    parser.add_argument('--dataset_root', type=str, default="/data/")
    args = parser.parse_args()

    # pdb.set_trace()
    # config = yaml.load(open(args.config, 'r'), Loader=yaml.FullLoader)
    config_file = 'configs/'+args.dataset+'/'+args.setting+'/test_maple.yaml'
    config = yaml.load(open(config_file, 'r'), Loader=yaml.FullLoader)

    #==========================================
    if args.dataset == "imageneta":
        args.balance_text_visual = args.balance_text_visual  #0.5
    elif args.dataset == "imagenetsketch":
        args.balance_text_visual = args.balance_text_visual  #1
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
    config["gamma_mu"] = args.gamma_mu
    config["gamma_sigma"] = args.gamma_sigma
    config["test_model"] = args.test_model
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
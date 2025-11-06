import os.path as osp
from collections import OrderedDict
import math
import pdb
import numpy as np
import torch
import torch.nn as nn
from torch.nn import functional as F
from torch.cuda.amp import GradScaler, autocast

from clip import clip
from clip.simple_tokenizer import SimpleTokenizer as _Tokenizer

from copy import deepcopy

_tokenizer = _Tokenizer()


def load_clip_to_cpu():
    backbone_name = 'ViT-B/16'
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

class TextEncoder(nn.Module):
    def __init__(self, clip_model):
        super().__init__()
        self.transformer = clip_model.transformer
        self.positional_embedding = clip_model.positional_embedding
        self.ln_final = clip_model.ln_final
        self.text_projection = clip_model.text_projection
        self.dtype = clip_model.dtype

    def forward(self, prompts, tokenized_prompts):
        x = prompts + self.positional_embedding.type(self.dtype)
        x = x.permute(1, 0, 2)  # NLD -> LND
        x = self.transformer(x)
        x = x.permute(1, 0, 2)  # LND -> NLD
        x = self.ln_final(x).type(self.dtype)

        # x.shape = [batch_size, n_ctx, transformer.width]
        # take features from the eot embedding (eot_token is the highest number in each sequence)
        x = (
                x[torch.arange(x.shape[0]), tokenized_prompts.argmax(dim=-1)]
                @ self.text_projection
        )  # ??

        return x


class CustomCLIP(nn.Module):
    def __init__(
            self,
            classnames,
            clip_model,
    ):
        super().__init__()


        self.text_encoder = TextEncoder(clip_model)
        self.dtype = clip_model.dtype

        # print('visual class-based prompt generation method is ' + vptmethod + '\n')
        # pdb.set_trace()
        self.class_prompts = torch.cat([clip.tokenize('an image of a ' + name) for name in classnames])
        # self.class_prompts = torch.cat([clip.tokenize('an image of ' + name + ' a') for name in classnames])
        self.testclasstokens = clip_model.token_embedding(self.class_prompts).type(
            self.dtype).cuda().detach()  # 7 * 77 * 512
        # pdb.set_trace()
        # if self.taskp:
        #     self.tasktokenlearner = self.prompt_learner.tasktokenlearner

        # pdb.set_trace()

    def forward(self, text):

        text_features = self.text_encoder(self.testclasstokens, self.class_prompts)
        # text_features = text_features / text_features.norm(dim=-1, keepdim=True).detach()

        num_classes = text_features.shape[0]

        return text_features



fnames = 'LOC_synset_mapping.txt'
all_classes = {}
with open(fnames, 'r') as f:
    lines = f.readlines()
for line in lines:
        class_id  = line.strip().split(' ')[0]
        all_classes[class_id] = line[10:].strip()

# clip_text_feat_root = "miniImageNet/miniImageNet"
clip_text_feat_root = "ImageNetSket/ImageNetSket"

feats = {}
for filename in ["_category_split_train_phase_train", "_category_split_val", "_category_split_test"]:
    data = torch.load(clip_text_feat_root + filename + '.pt')
    # feats = {}
    classnames = []
    classids = []
    for key in data.keys():
        classids.append(key)
        classnames.append(all_classes[key])
        # feats[all_classes[key]] = data[key]

    print(f"Loading CLIP (backbone: ViT-B/16)")
    clip_model = load_clip_to_cpu()
    clip_model.float()
    print("Building custom CLIP")
    model = CustomCLIP(classnames, clip_model)
    model.cuda()

    logits = model(0).detach().cpu()
    for i in range(len(classids)):
        feats[classids[i]] = logits[i]
    # torch.save(feats, clip_text_feat_root + filename + '_textfeats.pt')
# pdb.set_trace()
torch.save(feats, 'imagenet_1000classes_mark_2_textfeats.pt')
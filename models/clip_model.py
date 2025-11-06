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

from collections import OrderedDict

import torch
import torch.nn.functional as F
import torch.autograd as autograd
import torch.utils.checkpoint as cp

from . import encoders
from . import classifiers
from .modules import get_child_dict, Module, BatchNorm2d

# pdb.set_trace()

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
    def __init__(self, clip_model, allclasses, adapter=None):
        super().__init__()


        self.text_encoder = TextEncoder(clip_model)
        self.dtype = clip_model.dtype

        # print('visual class-based prompt generation method is ' + vptmethod + '\n')
        # pdb.set_trace()


        self.image_encoder = clip_model.visual

        self.logit_scale = clip_model.logit_scale
        # pdb.set_trace()

        self.class_prompts = torch.cat([clip.tokenize('a photo of a ' + name) for name in allclasses])
        self.testclasstokens = clip_model.token_embedding(self.class_prompts).type(self.dtype).cuda().detach()  # 7 * 77 * 512

    def forward(self, image, cats):
        # pdb.set_trace()

        text_features = self.text_encoder(self.testclasstokens[cats.long()], self.class_prompts[cats.cpu().long()])
        text_features = text_features / text_features.norm(dim=-1, keepdim=True).detach()

        image_features = self.image_encoder(image)
        image_features = image_features / image_features.norm(dim=-1, keepdim=True).detach()
        # pdb.set_trace()
        logits = self.logit_scale.exp() * (image_features @ text_features.t())

        # num_classes = text_features.shape[0]

        return logits


class CLIPMAML(Module):
  def __init__(self, cfg, clip_model, classnames, adapter=None):
    super(CLIPMAML, self).__init__()
    self.encoder = CustomCLIP(clip_model, classnames, adapter)
    # self.adapter = adapter
    # self.classifier = classifier

  def reset_classifier(self):
    self.adapter.reset_parameters()

  def _inner_forward(self, classnames, image):
    """ Forward pass for the inner loop. """
    # feat = self.encoder(x, get_child_dict(params, 'encoder'), episode)
    # logits = self.classifier(feat, get_child_dict(params, 'classifier'))
    logits = self.encoder(classnames, image)
    return logits

  def _inner_iter(self, x, y, params, mom_buffer, episode, inner_args, detach):
    """
    Performs one inner-loop iteration of MAML including the forward and
    backward passes and the parameter update.

    Args:
      x (float tensor, [n_way * n_shot, C, H, W]): per-episode support set.
      y (int tensor, [n_way * n_shot]): per-episode support set labels.
      params (dict): the model parameters BEFORE the update.
      mom_buffer (dict): the momentum buffer BEFORE the update.
      episode (int): the current episode index.
      inner_args (dict): inner-loop optimization hyperparameters.
      detach (bool): if True, detachs the graph for the current iteration.

    Returns:
      updated_params (dict): the model parameters AFTER the update.
      mom_buffer (dict): the momentum buffer AFTER the update.
    """
    with torch.enable_grad():
      # forward pass
      # logits = self._inner_forward(x, params, episode)
      logits = self._inner_forward(y, x)
      loss = F.cross_entropy(logits, y)
      # backward pass
      grads = autograd.grad(loss, params.values(),
                            create_graph=(not detach and not inner_args['first_order']),
                            only_inputs=True, allow_unused=True)
      # parameter update
      updated_params = OrderedDict()
      for (name, param), grad in zip(params.items(), grads):
        if grad is None:
          updated_param = param
        else:
          if inner_args['weight_decay'] > 0:
            grad = grad + inner_args['weight_decay'] * param
          if inner_args['momentum'] > 0:
            grad = grad + inner_args['momentum'] * mom_buffer[name]
            mom_buffer[name] = grad
          if 'encoder' in name:
            lr = inner_args['encoder_lr']
          elif 'classifier' in name:
            lr = inner_args['classifier_lr']
          else:
            raise ValueError('invalid parameter name')
          updated_param = param - lr * grad
        if detach:
          updated_param = updated_param.detach().requires_grad_(True)
        updated_params[name] = updated_param

    return updated_params, mom_buffer

  def _adapt(self, x, y, params, episode, inner_args, meta_train):
    """
    Performs inner-loop adaptation in MAML.

    Args:
      x (float tensor, [n_way * n_shot, C, H, W]): per-episode support set.
        (T: transforms, C: channels, H: height, W: width)
      y (int tensor, [n_way * n_shot]): per-episode support set labels.
      params (dict): a dictionary of parameters at meta-initialization.
      episode (int): the current episode index.
      inner_args (dict): inner-loop optimization hyperparameters.
      meta_train (bool): if True, the model is in meta-training.

    Returns:
      params (dict): model paramters AFTER inner-loop adaptation.
    """
    assert x.dim() == 4 and y.dim() == 1
    assert x.size(0) == y.size(0)

    # Initializes a dictionary of momentum buffer for gradient descent in the
    # inner loop. It has the same set of keys as the parameter dictionary.
    mom_buffer = OrderedDict()
    if inner_args['momentum'] > 0:
      for name, param in params.items():
        mom_buffer[name] = torch.zeros_like(param)
    params_keys = tuple(params.keys())
    mom_buffer_keys = tuple(mom_buffer.keys())

    for m in self.modules():
      if isinstance(m, BatchNorm2d) and m.is_episodic():
        m.reset_episodic_running_stats(episode)

    def _inner_iter_cp(episode, *state):
      """
      Performs one inner-loop iteration when checkpointing is enabled.
      The code is executed twice:
        - 1st time with torch.no_grad() for creating checkpoints.
        - 2nd time with torch.enable_grad() for computing gradients.
      """
      params = OrderedDict(zip(params_keys, state[:len(params_keys)]))
      mom_buffer = OrderedDict(
        zip(mom_buffer_keys, state[-len(mom_buffer_keys):]))

      detach = not torch.is_grad_enabled()  # detach graph in the first pass
      self.is_first_pass(detach)
      params, mom_buffer = self._inner_iter(
        x, y, params, mom_buffer, int(episode), inner_args, detach)
      state = tuple(t if t.requires_grad else t.clone().requires_grad_(True)
                    for t in tuple(params.values()) + tuple(mom_buffer.values()))
      return state

    for step in range(inner_args['n_step']):
      if self.efficient:  # checkpointing
        state = tuple(params.values()) + tuple(mom_buffer.values())
        state = cp.checkpoint(_inner_iter_cp, torch.as_tensor(episode), *state)
        params = OrderedDict(zip(params_keys, state[:len(params_keys)]))
        mom_buffer = OrderedDict(
          zip(mom_buffer_keys, state[-len(mom_buffer_keys):]))
      else:
        params, mom_buffer = self._inner_iter(
          x, y, params, mom_buffer, episode, inner_args, not meta_train)

    return params

  def forward(self, x_shot, x_query, y_shot, cats, inner_args, meta_train):
    """
    Args:
      x_shot (float tensor, [n_episode, n_way * n_shot, C, H, W]): support sets.
      x_query (float tensor, [n_episode, n_way * n_query, C, H, W]): query sets.
        (T: transforms, C: channels, H: height, W: width)
      y_shot (int tensor, [n_episode, n_way * n_shot]): support set labels.
      inner_args (dict, optional): inner-loop hyperparameters.
      meta_train (bool): if True, the model is in meta-training.

    Returns:
      logits (float tensor, [n_episode, n_way * n_shot, n_way]): predicted logits.
    """
    assert self.encoder is not None
    assert x_shot.dim() == 5 and x_query.dim() == 5
    assert x_shot.size(0) == x_query.size(0)

    # a dictionary of parameters that will be updated in the inner loop
    params = OrderedDict(self.named_parameters())
    for name in list(params.keys()):
      if not params[name].requires_grad or \
              any(s in name for s in inner_args['frozen'] + ['temp']):
        params.pop(name)

    logits = []
    for ep in range(x_shot.size(0)):
      # inner-loop training
      self.train()
      if not meta_train:
        for m in self.modules():
          if isinstance(m, BatchNorm2d) and not m.is_episodic():
            m.eval()


      # updated_params = self._adapt(
      #   x_shot[ep], y_shot[ep], params, ep, inner_args, meta_train)
      # # inner-loop validation
      # with torch.set_grad_enabled(meta_train):
      #   self.eval()
      #   logits_ep = self._inner_forward(x_query[ep], updated_params, ep)
      # logits.append(logits_ep)
      #   pdb.set_trace()
        logits_ep = self._inner_forward(x_query[ep], cats[ep])
        logits.append(logits_ep)

    self.train(meta_train)
    logits = torch.stack(logits)
    return logits
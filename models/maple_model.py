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

import os.path as osp
from collections import OrderedDict
import math
import copy
import torch
import torch.nn as nn
from torch.nn import functional as F
from torch.cuda.amp import GradScaler, autocast

# from dassl.engine import TRAINER_REGISTRY, TrainerX
# from dassl.metrics import compute_accuracy
# from dassl.utils import load_pretrained_weights, load_checkpoint
# from dassl.optim import build_optimizer, build_lr_scheduler

from clip import clip
from clip.simple_tokenizer import SimpleTokenizer as _Tokenizer

_tokenizer = _Tokenizer()


# def load_clip_to_cpu(cfg):
#     backbone_name = cfg['MODEL']['BACKBONE']['NAME']
#     url = clip._MODELS[backbone_name]
#     model_path = clip._download(url)
#
#     try:
#         # loading JIT archive
#         model = torch.jit.load(model_path, map_location="cpu").eval()
#         state_dict = None
#
#     except RuntimeError:
#         state_dict = torch.load(model_path, map_location="cpu")
#     design_details = {"trainer": 'MaPLe',
#                       "vision_depth": 0,
#                       "language_depth": 0, "vision_ctx": 0,
#                       "language_ctx": 0,
#                       "maple_length": cfg['TRAINER']['MAPLE']['N_CTX']}
#     model = clip.build_model(state_dict or model.state_dict(), design_details)
#
#     return model


class TextEncoder(nn.Module):
    def __init__(self, clip_model):
        super().__init__()
        self.transformer = clip_model.transformer
        self.positional_embedding = clip_model.positional_embedding
        self.ln_final = clip_model.ln_final
        self.text_projection = clip_model.text_projection
        self.dtype = clip_model.dtype

    def forward(self, prompts, tokenized_prompts, compound_prompts_deeper_text):
        x = prompts + self.positional_embedding.type(self.dtype)
        x = x.permute(1, 0, 2)  # NLD -> LND
        # Pass as the list, as nn.sequential cannot process multiple arguments in the forward pass
        combined = [x, compound_prompts_deeper_text, 0]  # third argument is the counter which denotes depth of prompt
        outputs = self.transformer(combined)
        x = outputs[0]  # extract the x back from here
        x = x.permute(1, 0, 2)  # LND -> NLD
        x = self.ln_final(x).type(self.dtype)

        # x.shape = [batch_size, n_ctx, transformer.width]
        # take features from the eot embedding (eot_token is the highest number in each sequence)
        x = x[torch.arange(x.shape[0]), tokenized_prompts.argmax(dim=-1)] @ self.text_projection

        return x


class MultiModalPromptLearner(nn.Module):
    def __init__(self, cfg, classnames, clip_model):
        super().__init__()
        n_cls = len(classnames)
        # pdb.set_trace()
        n_ctx = cfg['TRAINER']['MAPLE']['N_CTX']
        ctx_init = cfg['TRAINER']['MAPLE']['CTX_INIT']
        dtype = clip_model.dtype
        ctx_dim = clip_model.ln_final.weight.shape[0]
        clip_imsize = clip_model.visual.input_resolution
        cfg_imsize = 224
        # Default is 1, which is compound shallow prompting
        assert cfg['TRAINER']['MAPLE']['PROMPT_DEPTH'] >= 1, "For MaPLe, PROMPT_DEPTH should be >= 1"
        self.compound_prompts_depth = cfg['TRAINER']['MAPLE']['PROMPT_DEPTH']  # max=12, but will create 11 such shared prompts
        assert cfg_imsize == clip_imsize, f"cfg_imsize ({cfg_imsize}) must equal to clip_imsize ({clip_imsize})"

        if ctx_init and (n_ctx) <= 4:
            # use given words to initialize context vectors
            ctx_init = ctx_init.replace("_", " ")
            n_ctx = n_ctx
            prompt = clip.tokenize(ctx_init)
            with torch.no_grad():
                embedding = clip_model.token_embedding(prompt).type(dtype)
            ctx_vectors = embedding[0, 1: 1 + n_ctx, :]
            prompt_prefix = ctx_init
        else:
            # random initialization
            ctx_vectors = torch.empty(n_ctx, ctx_dim, dtype=dtype)
            nn.init.normal_(ctx_vectors, std=0.02)
            prompt_prefix = " ".join(["X"] * n_ctx)
        print('MaPLe design: Multi-modal Prompt Learning')
        print(f'Initial context: "{prompt_prefix}"')
        print(f"Number of MaPLe context words (tokens): {n_ctx}")
        # These below, related to the shallow prompts
        # Linear layer so that the tokens will project to 512 and will be initialized from 768
        self.proj = nn.Linear(ctx_dim, 768)
        self.proj.half()
        self.ctx = nn.Parameter(ctx_vectors)
        # These below parameters related to the shared prompts
        # Define the compound prompts for the deeper layers

        # Minimum can be 1, which defaults to shallow MaPLe
        # compound prompts
        self.compound_prompts_text = nn.ParameterList([nn.Parameter(torch.empty(n_ctx, 512))
                                                      for _ in range(self.compound_prompts_depth - 1)])
        for single_para in self.compound_prompts_text:
            nn.init.normal_(single_para, std=0.02)
        # Also make corresponding projection layers, for each prompt
        single_layer = nn.Linear(ctx_dim, 768)
        self.compound_prompt_projections = _get_clones(single_layer, self.compound_prompts_depth - 1)

        classnames = [name.replace("_", " ") for name in classnames]
        name_lens = [len(_tokenizer.encode(name)) for name in classnames]
        prompts = [prompt_prefix + " " + name + "." for name in classnames]

        tokenized_prompts = torch.cat([clip.tokenize(p) for p in prompts])  # (n_cls, n_tkn)
        with torch.no_grad():
            embedding = clip_model.token_embedding(tokenized_prompts).type(dtype)

        # These token vectors will be saved when in save_model(),
        # but they should be ignored in load_model() as we want to use
        # those computed using the current class names
        self.register_buffer("token_prefix", embedding[:, :1, :])  # SOS
        self.register_buffer("token_suffix", embedding[:, 1 + n_ctx:, :])  # CLS, EOS

        self.n_cls = n_cls
        self.n_ctx = n_ctx
        self.tokenized_prompts = tokenized_prompts  # torch.Tensor
        self.name_lens = name_lens

    def construct_prompts(self, ctx, prefix, suffix, label=None):
        # dim0 is either batch_size (during training) or n_cls (during testing)
        # ctx: context tokens, with shape of (dim0, n_ctx, ctx_dim)
        # prefix: the sos token, with shape of (n_cls, 1, ctx_dim)
        # suffix: remaining tokens, with shape of (n_cls, *, ctx_dim)

        if label is not None:
            prefix = prefix[label]
            suffix = suffix[label]

        prompts = torch.cat(
            [
                prefix,  # (dim0, 1, dim)
                ctx,  # (dim0, n_ctx, dim)
                suffix,  # (dim0, *, dim)
            ],
            dim=1,
        )

        return prompts

    def forward(self):
        ctx = self.ctx

        if ctx.dim() == 2:
            ctx = ctx.unsqueeze(0).expand(self.n_cls, -1, -1)

        prefix = self.token_prefix
        suffix = self.token_suffix
        prompts = self.construct_prompts(ctx, prefix, suffix)

        # Before returning, need to transform
        # prompts to 768 for the visual side
        visual_deep_prompts = []
        for index, layer in enumerate(self.compound_prompt_projections):
            visual_deep_prompts.append(layer(self.compound_prompts_text[index]))
        # Now the other way around
        # We will project the textual prompts from 512 to 768
        # pdb.set_trace()
        return prompts, self.proj(self.ctx), self.compound_prompts_text, visual_deep_prompts   # pass here original, as for visual 768 is required


class CustomCLIP(nn.Module):
    def __init__(self, cfg, clip_model, classnames, adapter=None):
        super().__init__()
        self.prompt_learner = MultiModalPromptLearner(cfg, classnames, clip_model)
        self.tokenized_prompts = self.prompt_learner.tokenized_prompts
        self.image_encoder = clip_model.visual
        self.text_encoder = TextEncoder(clip_model)
        self.logit_scale = clip_model.logit_scale
        self.dtype = clip_model.dtype

    def forward(self, image, cats):
        tokenized_prompts = self.tokenized_prompts
        logit_scale = self.logit_scale.exp()

        prompts, shared_ctx, deep_compound_prompts_text, deep_compound_prompts_vision = self.prompt_learner()
        # pdb.set_trace()
        # prompts = prompts[cats.long()]
        text_features = self.text_encoder(prompts[cats.long()], tokenized_prompts[cats.cpu().long()], deep_compound_prompts_text)
        image_features = self.image_encoder(image.type(self.dtype), shared_ctx, deep_compound_prompts_vision)

        image_features = image_features / image_features.norm(dim=-1, keepdim=True)
        text_features = text_features / text_features.norm(dim=-1, keepdim=True)
        logits = logit_scale * image_features @ text_features.t()

        return logits, image_features


#
# class CustomCLIP(nn.Module):
#     def __init__(self, clip_model, allclasses, adapter=None):
#         super().__init__()
#
#
#         self.text_encoder = TextEncoder(clip_model)
#         self.dtype = clip_model.dtype
#
#         # print('visual class-based prompt generation method is ' + vptmethod + '\n')
#         # pdb.set_trace()
#
#
#         self.image_encoder = clip_model.visual
#
#         self.logit_scale = clip_model.logit_scale
#         # pdb.set_trace()
#
#         self.class_prompts = torch.cat([clip.tokenize('a photo of a ' + name) for name in allclasses])
#         self.testclasstokens = clip_model.token_embedding(self.class_prompts).type(self.dtype).cuda().detach()  # 7 * 77 * 512
#
#     def forward(self, image, cats):
#         # pdb.set_trace()
#
#         text_features = self.text_encoder(self.testclasstokens[cats.long()], self.class_prompts[cats.cpu().long()])
#         text_features = text_features / text_features.norm(dim=-1, keepdim=True).detach()
#
#         image_features = self.image_encoder(image)
#         image_features = image_features / image_features.norm(dim=-1, keepdim=True).detach()
#         # pdb.set_trace()
#         logits = self.logit_scale.exp() * (image_features @ text_features.t())
#
#         # num_classes = text_features.shape[0]
#
#         return logits


def _get_clones(module, N):
    return nn.ModuleList([copy.deepcopy(module) for i in range(N)])


# @TRAINER_REGISTRY.register()
# class MaPLe(TrainerX):
#     def check_cfg(self, cfg):
#         assert cfg.TRAINER.MAPLE.PREC in ["fp16", "fp32", "amp"]
#
#     def build_model(self):
#         cfg = self.cfg
#         classnames = self.dm.dataset.classnames
#
#         print(f"Loading CLIP (backbone: {cfg.MODEL.BACKBONE.NAME})")
#         clip_model = load_clip_to_cpu(cfg)
#
#         if cfg.TRAINER.MAPLE.PREC == "fp32" or cfg.TRAINER.MAPLE.PREC == "amp":
#             # CLIP's default precision is fp16
#             clip_model.float()
#
#         print("Building custom CLIP")
#         self.model = CustomCLIP(cfg, classnames, clip_model)
#
#         print("Turning off gradients in both the image and the text encoder")
#         name_to_update = "prompt_learner"
#
#         for name, param in self.model.named_parameters():
#             if name_to_update not in name:
#                 # Make sure that VPT prompts are updated
#                 if "VPT" in name:
#                     param.requires_grad_(True)
#                 else:
#                     param.requires_grad_(False)
#
#         # Double check
#         enabled = set()
#         for name, param in self.model.named_parameters():
#             if param.requires_grad:
#                 enabled.add(name)
#         print(f"Parameters to be updated: {enabled}")
#
#         if cfg.MODEL.INIT_WEIGHTS:
#             load_pretrained_weights(self.model, cfg.MODEL.INIT_WEIGHTS)
#
#         self.model.to(self.device)
#         # NOTE: only give prompt_learner to the optimizer
#         self.optim = build_optimizer(self.model, cfg.OPTIM)
#         self.sched = build_lr_scheduler(self.optim, cfg.OPTIM)
#         self.register_model("MultiModalPromptLearner", self.model, self.optim, self.sched)
#
#         self.scaler = GradScaler() if cfg.TRAINER.MAPLE.PREC == "amp" else None
#
#         # Note that multi-gpu training could be slow because CLIP's size is
#         # big, which slows down the copy operation in DataParallel
#         device_count = torch.cuda.device_count()
#         if device_count > 1:
#             print(f"Multiple GPUs detected (n_gpus={device_count}), use all of them!")
#             self.model = nn.DataParallel(self.model)
#
#     def forward_backward(self, batch):
#         image, label = self.parse_batch_train(batch)
#
#         model = self.model
#         optim = self.optim
#         scaler = self.scaler
#
#         prec = self.cfg.TRAINER.MAPLE.PREC
#         if prec == "amp":
#             with autocast():
#                 loss = model(image, label)
#             optim.zero_grad()
#             scaler.scale(loss).backward()
#             scaler.step(optim)
#             scaler.update()
#         else:
#             loss = model(image, label)
#             optim.zero_grad()
#             loss.backward()
#             optim.step()
#
#         loss_summary = {"loss": loss.item()}
#
#         if (self.batch_idx + 1) == self.num_batches:
#             self.update_lr()
#
#         return loss_summary
#
#     def parse_batch_train(self, batch):
#         input = batch["img"]
#         label = batch["label"]
#         input = input.to(self.device)
#         label = label.to(self.device)
#         return input, label
#
#     def load_model(self, directory, epoch=None):
#         if not directory:
#             print("Note that load_model() is skipped as no pretrained model is given")
#             return
#
#         names = self.get_model_names()
#
#         # By default, the best model is loaded
#         model_file = "model-best.pth.tar"
#
#         if epoch is not None:
#             model_file = "model.pth.tar-" + str(epoch)
#
#         for name in names:
#             model_path = osp.join(directory, name, model_file)
#
#             if not osp.exists(model_path):
#                 raise FileNotFoundError('RisklearnerModel not found at "{}"'.format(model_path))
#
#             checkpoint = load_checkpoint(model_path)
#             state_dict = checkpoint["state_dict"]
#             epoch = checkpoint["epoch"]
#
#             # Ignore fixed token vectors
#             if "prompt_learner.token_prefix" in state_dict:
#                 del state_dict["prompt_learner.token_prefix"]
#
#             if "prompt_learner.token_suffix" in state_dict:
#                 del state_dict["prompt_learner.token_suffix"]
#
#             print("Loading weights to {} " 'from "{}" (epoch = {})'.format(name, model_path, epoch))
#             # set strict=False
#             self._models[name].load_state_dict(state_dict, strict=False)






class VMAPLE(Module):
  def __init__(self, cfg, clip_model, classnames, adapter=None):
    super(VMAPLE, self).__init__()
    self.encoder = CustomCLIP(cfg, clip_model, classnames, adapter)
    # self.adapter = adapter
    # self.classifier = classifier
    self.balance_text_visual = cfg["balance_text_visual"]

  def reset_classifier(self):
    self.adapter.reset_parameters()

  def _inner_forward(self, image, classnames):
    """ Forward pass for the inner loop. """
    # feat = self.encoder(x, get_child_dict(params, 'encoder'), episode)
    # logits = self.classifier(feat, get_child_dict(params, 'classifier'))
    logits = self.encoder(image, classnames)
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
    # pdb.set_trace()
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
      _, feat_s = self._inner_forward(x_shot[ep], cats[ep])
      # pdb.set_trace()
      logits_ep, feat_q = self._inner_forward(x_query[ep], cats[ep])
      logits_v = feat_q @ feat_s.t() * self.encoder.logit_scale.exp()
      logits.append(logits_ep + self.balance_text_visual*logits_v)
      # logits.append(logits_ep)
      # logits.append(logits_v)
      # logits.append(logits_ep + 1* logits_v)
    self.train(meta_train)
    logits = torch.stack(logits)
    return logits
# Codebase modified based on research published by:
# Hertz, Amir, et al. "Prompt-to-prompt image editing with cross attention control." 
# arXiv preprint arXiv:2208.01626 (2022).
# Copyright 2022 Google LLC -  Licensed under the Apache License, Version 2.0 

from typing import Union, Tuple, List, Callable, Dict, Optional
import torch
import torch.nn.functional as nnf
from diffusers import DiffusionPipeline
import numpy as np
from IPython.display import display
from PIL import Image
import abc

import imageio
import os

import ptp_utils
import seq_aligner

from configs.config import config


MAX_NUM_WORDS = config['max_num_words']
device = torch.device('cuda:0') if torch.cuda.is_available() else torch.device('cpu')

class LocalBlend:
    def __init__(
            self, 
            tokenizer, 
            prompts: List[str], 
            words: List[List[str]], 
            threshold: float = .3
        ):
        # Initialize alpha layers to control the attention across different words and prompts
        alpha_layers = torch.zeros(len(prompts),  1, 1, 1, 1, MAX_NUM_WORDS)

        # Populate the alpha layers with ones at positions corresponding to the words of interest in the prompts
        for i, (prompt, words_) in enumerate(zip(prompts, words)):
            if type(words_) is str:
                words_ = [words_]
            for word in words_:
                ind = ptp_utils.get_word_inds(prompt, word, tokenizer)
                alpha_layers[i, :, :, :, :, ind] = 1

        # Move the alpha layers to the appropriate device and set the threshold for the binary mask
        self.alpha_layers = alpha_layers.to(device)
        self.threshold = threshold

    def __call__(
            self, 
            x_t, 
            attention_store, 
            step
        ):
        # Initialize the kernel size and padding for max pooling
        k = 1

        # Aggregate attention masks from different layers of the U-Net (encoder and decoder parts)
        maps = attention_store["down_cross"][:2] + attention_store["up_cross"][3:6]

        # Reshape each attention mask to match the size of the number of prompts, which is represented by `self.alpha_layers`
        maps = [item.reshape(self.alpha_layers.shape[0], -1, 1, 16, 16, MAX_NUM_WORDS) for item in maps]

        maps = torch.cat(maps, dim=1)

        # Combine the attention maps with alpha layers and summarize across the last dimension (words), then average over prompts
        maps = (maps * self.alpha_layers).sum(-1).mean(1)

        # Error fixed in original code: apply max pooling to the attention maps to reduce their size and add padding
        mask = nnf.max_pool2d(maps, (k * 2 + 1, k * 2 + 1), (1, 1), padding=(k, k))

        # Resize the attention masks to match the dimensions of the image representation at time t
        mask = nnf.interpolate(mask, size=(x_t.shape[2:]))

        # Normalize the attention mask to have values between 0 and 1
        mask = mask / mask.max(2, keepdims=True)[0].max(3, keepdims=True)[0]

        mask = mask.gt(self.threshold)

        mask = (mask[:1] + mask).float()

        # Apply the mask to `x_t`, keeping the first batch's content where the mask is 0, and blending otherwise
        x_t = x_t[:1] + mask * (x_t - x_t[:1])

        return x_t


class AttentionControl(abc.ABC):

    def step_callback(self, x_t):
        return x_t

    def between_steps(self):
        return
    
    @abc.abstractmethod
    def forward(self, attn, is_cross: bool, place_in_unet: str):
        raise NotImplementedError
    # Modifies the latter half of the attention based on the current layer 
    def __call__(self, attn, is_cross: bool, place_in_unet: str):
     
        h = attn.shape[0]
        # Modifies the attention starting from the halfway point 
        attn[h // 2:] = self.forward(attn[h // 2:], is_cross, place_in_unet)
        
        self.cur_att_layer += 1

        if self.cur_att_layer == self.num_att_layers:
            self.cur_att_layer = 0
            self.cur_step += 1
            self.between_steps()

        return attn


    def reset(self):
        self.cur_step = 0
        self.cur_att_layer = 0

    # Constructor for the class that initializes the step and layer counters.
    # By default, 'num_att_layers' is set to -1 to indicate that subclasses
    # should set this with the actual number of attention layers they use.
    def __init__(self):
        self.cur_step = 0  # Current step in the attention process.
        self.num_att_layers = -1  # Total number of attention layers (to be set by subclasses).
        self.cur_att_layer = 0  # Current layer within the attention process.

class EmptyControl(AttentionControl):

    def forward (self, attn, is_cross: bool, place_in_unet: str):
        return attn


class AttentionStore(AttentionControl):

    @staticmethod
    def get_empty_store():
        return {
            "down_cross": [], 
            "mid_cross": [], 
            "up_cross": [],
            "down_self": [],  
            "mid_self": [],  
            "up_self": []
            }

    def forward(self, attn, is_cross: bool, place_in_unet: str):
        # avoid memory overhead
        if attn.shape[1] <= 16 ** 2: 
            key = f"{place_in_unet}_{'cross' if is_cross else 'self'}"
            self.step_store[key].append(attn)
        return attn

    def between_steps(self):
        if len(self.attention_store) == 0:
            self.attention_store = self.step_store
        else:
            for key in self.attention_store:
                for i in range(len(self.attention_store[key])):
                    self.attention_store[key][i] += self.step_store[key][i]
        self.step_store = self.get_empty_store()

    def get_average_attention(self):
        average_attention = {key: [item / self.cur_step for item in self.attention_store[key]] for key in self.attention_store}
        return average_attention


    def reset(self):
        super(AttentionStore, self).reset()
        self.step_store = self.get_empty_store()
        self.attention_store = {}

    def __init__(self):
        super(AttentionStore, self).__init__()
        self.step_store = self.get_empty_store()
        self.attention_store = {}


class AttentionControlEdit(AttentionStore, abc.ABC):
    
    def __init__(
            self, 
            tokenizer, 
            prompts, 
            num_steps: int,
            cross_replace_steps: Union[float, Tuple[float, float], 
            Dict[str, Tuple[float, float]]],
            self_replace_steps: Union[float, Tuple[float, float]],
            local_blend: Optional[LocalBlend]
    ):
        super(AttentionControlEdit, self).__init__()
        self.batch_size = len(prompts)
        self.cross_replace_alpha = ptp_utils.get_time_words_attention_alpha(
            prompts, 
            num_steps, 
            cross_replace_steps, 
            tokenizer
            ).to(device)
        if type(self_replace_steps) is float:
            self_replace_steps = 0, self_replace_steps
        self.num_self_replace = (
            int(num_steps * self_replace_steps[0]), 
            int(num_steps * self_replace_steps[1])
        )
        self.local_blend = local_blend

    def step_callback(self, x_t):
        if self.local_blend is not None:
            x_t = self.local_blend(x_t, self.attention_store, self.cur_step)
        return x_t

    def replace_self_attention(self, attn_base, att_replace):
        if att_replace.shape[2] <= 16 ** 2:
            return attn_base.unsqueeze(0).expand(att_replace.shape[0], *attn_base.shape)
        else:
            return att_replace

    @abc.abstractmethod
    def replace_cross_attention(self, attn_base, att_replace):
        raise NotImplementedError

    def forward(self, attn, is_cross: bool, place_in_unet: str):
        super(AttentionControlEdit, self).forward(attn, is_cross, place_in_unet)
        if is_cross or (self.num_self_replace[0] <= self.cur_step < self.num_self_replace[1]):
            h = attn.shape[0] // (self.batch_size)
            attn = attn.reshape(self.batch_size, h, *attn.shape[1:])
            attn_base, attn_repalce = attn[0], attn[1:]
            if is_cross:
                alpha_words = self.cross_replace_alpha[self.cur_step]
                attn_repalce_new = self.replace_cross_attention(attn_base, attn_repalce) * alpha_words + (1 - alpha_words) * attn_repalce
                attn[1:] = attn_repalce_new
            else:
                attn[1:] = self.replace_self_attention(attn_base, attn_repalce)
            attn = attn.reshape(self.batch_size * h, *attn.shape[2:])
        return attn


class AttentionReplace(AttentionControlEdit):

    def replace_cross_attention(self, attn_base, att_replace):
        return torch.einsum('hpw,bwn->bhpn', attn_base, self.mapper)

    def __init__(self, tokenizer, prompts, num_steps: int, cross_replace_steps: float, self_replace_steps: float,
                 local_blend: Optional[LocalBlend] = None):
        super(AttentionReplace, self).__init__(tokenizer, prompts, num_steps, cross_replace_steps, self_replace_steps, local_blend)
        self.mapper = seq_aligner.get_replacement_mapper(prompts, tokenizer).to(device)


class AttentionRefine(AttentionControlEdit):
    def __init__(
            self, 
            tokenizer, 
            prompts, 
            num_steps: int, 
            cross_replace_steps: float, 
            self_replace_steps: float,
            local_blend: Optional[LocalBlend] = None
        ):
        super(AttentionRefine, self).__init__(
            prompts, 
            num_steps, 
            cross_replace_steps, 
            self_replace_steps, 
            local_blend
        )
        self.mapper, alphas = seq_aligner.get_refinement_mapper(prompts, tokenizer)
        self.mapper, alphas = self.mapper.to(device), alphas.to(device)
        self.alphas = alphas.reshape(alphas.shape[0], 1, 1, alphas.shape[1])

    def replace_cross_attention(self, attn_base, att_replace):
        attn_base_replace = attn_base[:, :, self.mapper].permute(2, 0, 1, 3)
        attn_replace = attn_base_replace * self.alphas + att_replace * (1 - self.alphas)
        return attn_replace



class AttentionReweight(AttentionControlEdit):

    def __init__(
            self, 
            tokenizer, 
            prompts, 
            num_steps: int, 
            cross_replace_steps: float, 
            self_replace_steps: float, 
            equalizer,
            local_blend: Optional[LocalBlend] = None, 
            controller: Optional[AttentionControlEdit] = None
        ):
        super(AttentionReweight, self).__init__(
            tokenizer,
            prompts, 
            num_steps, 
            cross_replace_steps, 
            self_replace_steps, 
            local_blend
        )
        self.equalizer = equalizer.to(device)
        self.prev_controller = controller
    
    def replace_cross_attention(self, attn_base, att_replace):
        if self.prev_controller is not None:
            attn_base = self.prev_controller.replace_cross_attention(attn_base, att_replace)
        attn_replace = attn_base[None, :, :, :] * self.equalizer[:, None, None, :]
        return attn_replace


def get_equalizer(
        tokenizer, 
        text: str, 
        word_select: Union[int, Tuple[int, ...]], 
        values: Union[List[float], Tuple[float, ...]]
    ):
    if type(word_select) is int or type(word_select) is str:
        word_select = (word_select,)
    equalizer = torch.ones(len(values), 77)
    values = torch.tensor(values, dtype=torch.float32)
    for word in word_select:
        inds = ptp_utils.get_word_inds(text, word, tokenizer)
        for i in inds:
            equalizer[:, i] = values
    return equalizer


def aggregate_attention(
        prompts, 
        attention_store: AttentionStore, 
        res: int, 
        from_where: List[str], 
        is_cross: bool, 
        select: int
    ):
    out = []
    attention_maps = attention_store.get_average_attention()
    num_pixels = res ** 2
    for location in from_where:
        for item in attention_maps[f"{location}_{'cross' if is_cross else 'self'}"]:
            if item.shape[1] == num_pixels:
                cross_maps = item.reshape(len(prompts), -1, res, res, item.shape[-1])[select]
                out.append(cross_maps)
    out = torch.cat(out, dim=0)
    out = out.sum(0) / out.shape[0]
    return out.cpu()


def show_cross_attention(
        prompts, 
        tokenizer, 
        attention_store: AttentionStore, 
        res: int, 
        from_where: List[str], 
        select: int = 0
    ):
    tokens = tokenizer.encode(prompts[select])
    decoder = tokenizer.decode
    attention_maps = aggregate_attention(attention_store, res, from_where, True, select)
    images = []
    for i in range(len(tokens)):
        image = attention_maps[:, :, i]
        image = 255 * image / image.max()
        image = image.unsqueeze(-1).expand(*image.shape, 3)
        image = image.numpy().astype(np.uint8)
        image = np.array(Image.fromarray(image).resize((256, 256)))
        image = ptp_utils.text_under_image(image, decoder(int(tokens[i])))
        images.append(image)
    ptp_utils.view_images(np.stack(images, axis=0))


def show_self_attention_comp(
        attention_store: AttentionStore, 
        res: int, 
        from_where: List[str],
        max_com=10, 
        select: int = 0
    ):
    attention_maps = aggregate_attention(attention_store, res, from_where, False, select).numpy().reshape((res ** 2, res ** 2))
    u, s, vh = np.linalg.svd(attention_maps - np.mean(attention_maps, axis=1, keepdims=True))
    images = []
    for i in range(max_com):
        image = vh[i].reshape(res, res)
        image = image - image.min()
        image = 255 * image / image.max()
        image = np.repeat(np.expand_dims(image, axis=2), 3, axis=2).astype(np.uint8)
        image = Image.fromarray(image).resize((256, 256))
        image = np.array(image)
        images.append(image)
    ptp_utils.view_images(np.concatenate(images, axis=1))   

def sort_by_eq(eq):

    def inner_(images):
        swap = 0
        if eq[-1] < 1:
            for i in range(len(eq)):
                if eq[i] > 1 and eq[i + 1] < 1:
                    swap = i + 2
                    break
        else:
             for i in range(len(eq)):
                if eq[i] < 1 and eq[i + 1] > 1:
                    swap = i + 2
                    break
        print(swap)
        if swap > 0:
            images = np.concatenate([images[1:swap], images[:1], images[swap:]], axis=0)

        return images
    return inner_

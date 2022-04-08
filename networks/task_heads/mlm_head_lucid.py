import math
from functools import reduce

import torch
from torch import nn
import torch.nn.functional as F

from mask_utils_lucid import *

# main class

class MLM(nn.Module):
    def __init__(
        self,
        transformer,
        mask_prob = 0.15,
        replace_prob = 0.9,
        num_tokens = None,
        random_token_prob = 0.,
        mask_token_id = 2,
        pad_token_id = 0,
        mask_ignore_token_ids = []):
        super().__init__()

        self.transformer = transformer

        # mlm related probabilities

        self.mask_prob = mask_prob
        self.replace_prob = replace_prob

        self.num_tokens = num_tokens
        self.random_token_prob = random_token_prob

        # token ids

        self.pad_token_id = pad_token_id
        self.mask_token_id = mask_token_id
        self.mask_ignore_token_ids = set([*mask_ignore_token_ids, pad_token_id])

    def forward(self, seq, **kwargs):
        '''
        PONKU:
        mlm task head should output a tuple of: (head_out: Tensor, head_loss: Tensor)
        '''

        # do not mask [pad] tokens, or any other tokens in the tokens designated to be excluded ([cls], [sep])
        # also do not include these special tokens in the tokens chosen at random

        no_mask = mask_with_tokens(seq, self.mask_ignore_token_ids)
        mask = get_mask_subset_with_prob(~no_mask, self.mask_prob)

        # mask input with mask tokens with probability of `replace_prob` (keep tokens the same with probability 1 - replace_prob)

        masked_seq = seq.clone().detach()

        # derive labels to predict

        labels = seq.masked_fill(~mask, self.pad_token_id)

        # if random token probability > 0 for mlm

        if self.random_token_prob > 0:
            assert self.num_tokens is not None, 'num_tokens keyword must be supplied when instantiating MLM if using random token replacement'
            random_token_prob = prob_mask_like(seq, self.random_token_prob)
            random_tokens = torch.randint(0, self.num_tokens, seq.shape, device=seq.device)
            random_no_mask = mask_with_tokens(random_tokens, self.mask_ignore_token_ids)
            random_token_prob &= ~random_no_mask
            masked_seq = torch.where(random_token_prob, random_tokens, masked_seq)

            # remove tokens that were substituted randomly from being [mask]ed later
            mask = mask & ~random_token_prob

        # [mask] input

        replace_prob = prob_mask_like(seq, self.replace_prob)
        masked_seq = masked_seq.masked_fill(mask * replace_prob, self.mask_token_id)

        # get generator output and get mlm loss

        logits = self.transformer(masked_seq, **kwargs)

        mlm_loss = F.cross_entropy(
            logits.transpose(1, 2),
            labels,
            ignore_index = self.pad_token_id
        )

        return mlm_loss

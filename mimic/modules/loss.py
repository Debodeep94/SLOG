import torch
import torch.nn as nn
import torch.nn.functional as F
from .utils import *

class LanguageModelCriterion(nn.Module):
    def __init__(self):
        super(LanguageModelCriterion, self).__init__()

    def forward(self, input, target, mask):
        # truncate to the same size
        target = target[:, :input.size(1)]
        mask = mask[:, :input.size(1)]
        
        output = -input.gather(2, target.long().unsqueeze(2)).squeeze(2) * mask
        
        output = torch.sum(output) / torch.sum(mask)

        return output


def compute_loss(output, reports_ids, reports_masks):
    criterion = LanguageModelCriterion()
    loss = criterion(output, reports_ids[:, 1:], reports_masks[:, 1:]).mean()
    return loss


def surrogate_loss(outputs, chex, weights='yes'):
    device = torch.device('cuda:0')
    weights = count_weights(chex,0)
    temp1 = outputs.view(-1, 3)
    #print(temp1.size())
    #print(temp1)
    batch_targets=chex.to(torch.int64)
    temp2 = batch_targets.reshape(-1)
    if weights == 'yes':
        criterion = torch.nn.CrossEntropyLoss(weight=(weights).to(device))
    else: 
        criterion = torch.nn.CrossEntropyLoss()
    surr_loss = criterion(temp1, temp2)
    return surr_loss


class PPOCriterion(nn.Module):
    def __init__(self): # add clip param as an argument later
        super(PPOCriterion, self).__init__()
        self.clip_param = 0.1 #clip_param

    def forward(self, old_logprobs_agg, new_logprobs_agg, atarg):
        # print(f'new_logprobs_agg: {new_logprobs_agg}')
        # print(f'old_logprobs_agg: {old_logprobs_agg}')
        print(new_logprobs_agg - old_logprobs_agg)
        ratio = torch.exp(new_logprobs_agg - old_logprobs_agg)
        print(f'ratio: {ratio}')
        print(f'ratio shape: {ratio.size()}')
        surr1 = ratio * atarg
        surr2 = torch.clamp(ratio, 1 - self.clip_param, 1 + self.clip_param) * atarg
        pol_surr = -torch.min(surr1, surr2).mean()
        return pol_surr

def compute_ppo_loss(old_logprobs_agg, new_logprobs_agg, atarg):
    criterion = PPOCriterion()
    loss = criterion(old_logprobs_agg, new_logprobs_agg, atarg)
    return loss
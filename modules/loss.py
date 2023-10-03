import torch
import torch.nn as nn
import torch.nn.functional as F

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

def jsd_loss(p, q):
        m = 0.5 * (p + q)
        kl_pm = F.kl_div(p.log(), m, reduction='batchmean')
        kl_qm = F.kl_div(q.log(), m, reduction='batchmean')
        jsd = 0.5 * (kl_pm + kl_qm)
        return jsd
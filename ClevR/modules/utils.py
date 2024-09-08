import numpy as np
import cv2
import torch
import numpy as np
import re
from collections import OrderedDict

def remove_dataparallel_prefix(state_dict):
    new_state_dict = OrderedDict()
    for k, v in state_dict.items():
        if k.startswith('module.'):
            new_state_dict[k[7:]] = v  # remove the 'module.' prefix
        else:
            new_state_dict[k] = v
    return new_state_dict

def penalty_builder(penalty_config):
    if penalty_config == '':
        return lambda x, y: y
    pen_type, alpha = penalty_config.split('_')
    alpha = float(alpha)
    if pen_type == 'wu':
        return lambda x, y: length_wu(x, y, alpha)
    if pen_type == 'avg':
        return lambda x, y: length_average(x, y, alpha)


def length_wu(length, logprobs, alpha=0.):
    """
    NMT length re-ranking score from
    "Google's Neural Machine Translation System" :cite:`wu2016google`.
    """

    modifier = (((5 + length) ** alpha) /
                ((5 + 1) ** alpha))
    return logprobs / modifier


def length_average(length, logprobs, alpha=0.):
    """
    Returns the average probability of tokens in a sequence.
    """
    return logprobs / length


def split_tensors(n, x):
    if torch.is_tensor(x):
        assert x.shape[0] % n == 0
        x = x.reshape(x.shape[0] // n, n, *x.shape[1:]).unbind(1)
    elif type(x) is list or type(x) is tuple:
        x = [split_tensors(n, _) for _ in x]
    elif x is None:
        x = [None] * n
    return x


def repeat_tensors(n, x):
    """
    For a tensor of size Bx..., we repeat it n times, and make it Bnx...
    For collections, do nested repeat
    """
    if torch.is_tensor(x):
        x = x.unsqueeze(1)  # Bx1x...
        x = x.expand(-1, n, *([-1] * len(x.shape[2:])))  # Bxnx...
        x = x.reshape(x.shape[0] * n, *x.shape[2:])  # Bnx...
    elif type(x) is list or type(x) is tuple:
        x = [repeat_tensors(n, _) for _ in x]
    return x


def generate_heatmap(image, weights):
    print(image[1:])
   # cv2.imshow('image',image[1:])
    print(image)
    print(f'image shape:{image.shape}')
    image = image.transpose(1,2,0) #1,2,0 was before I edited
    height, width, _ = image.shape
    weights = weights.reshape(int(weights.shape[0] ** 0.5), int(weights.shape[0] ** 0.5))
    weights = weights - np.min(weights)
    weights = weights / np.max(weights)
    weights = cv2.resize(weights, (width, height))
    weights = np.uint8(255 * weights)
    heatmap = cv2.applyColorMap(weights, cv2.COLORMAP_JET)
    result = heatmap * 0.5 + image * 0.5
    return result

def cycle(iterable):
    #https://github.com/pytorch/pytorch/issues/23900
    iterator = iter(iterable)
    while True:
        try:
            yield next(iterator)
        except StopIteration:
            iterator = iter(iterable)

class EarlyStopping:
    def __init__(self, patience=10, delta=0):
        self.patience = patience
        self.counter = 0
        self.best_score = None
        self.delta = delta
        self.early_stop = False

    def __call__(self, val_loss):
        score = -val_loss

        if self.best_score is None:
            self.best_score = score
        elif score < self.best_score + self.delta:
            self.counter += 1
            print(f'EarlyStopping counter: {self.counter} out of {self.patience}')
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            self.counter = 0
def check_tensor_device(model):
    for name, tensor in model.named_parameters():
        print(f"Tensor: {name}, Device: {tensor.device}")

def surrogate_split(train_x, train_y, weights):
    train_x = train_x[:int(len(train_x)*0.8)]
    train_y = train_y[:int(len(train_y)*0.8)]

    val_x = train_x[int(len(train_x)*0.8):]
    val_y = train_y[int(len(train_y)*0.8):]

    weight_train= weights[:int(len(weights)*0.8)]

    return train_x, train_y, val_x, val_y, weight_train
    
def convert(val):
    if val==-1:
        return  2# This is just for matching the chexpert with ground truth
    elif val == 1:
        return val
    else:
        return 0
    
def count (tensor_data):
    counts=torch.nn.functional.one_hot (tensor_data.long()).sum (dim = 0)#.sum(dim=0)
    counts=counts.sum(dim=0)
    return counts
def count_weights(tensor_data,val):
    counts=count( tensor_data)
    #print(counts)
    weights=counts[val]/counts
    weights = torch.clamp(weights, max=15)
    return weights#/300

# now we have to calculate micro recall, macro is easy

def ce_metric(sentence_gt, sentence_pred, mode='micro'):
    plural_to_singular = {
    "spheres": "sphere",
    "cylinders": "cylinder",
    "cubes": "cube",
    "cylinders": "cylinder"}
    sentence_gt=re.sub(';','',sentence_gt)
    sentence_gt=re.sub(',','',sentence_gt)
    sentence_pred=re.sub(';','',sentence_pred)
    sentence_pred=re.sub(',','',sentence_pred)
    split_sent_gt=sentence_gt.split()
    # Convert plurals to singulars using the dictionary
    split_sent_gt = [plural_to_singular.get(word, word) for word in split_sent_gt]
    main_set_gt=[]
    temp_set_gt=[]
    for i in split_sent_gt:
        temp_set_gt.append(i)
        if i =='cube' or i=='sphere' or i=='cylinder' or i =='cubes' or i=='spheres' or i=='cylinders':
            main_set_gt.append(temp_set_gt[1:])
            temp_set_gt=[]
    split_sent_pred=sentence_pred.split()
    split_sent_pred = [plural_to_singular.get(word, word) for word in split_sent_pred]
    main_set_pred=[]
    temp_set_pred=[]
    for i in split_sent_pred:
        temp_set_pred.append(i)
        if i =='cube' or i=='sphere' or i=='cylinder' or i =='cubes' or i=='spheres' or i=='cylinders':
            main_set_pred.append(temp_set_pred[1:])
            temp_set_pred=[]
    # Convert lists to sets of tuples for easier comparison
    gt_set = set(tuple(item) for item in main_set_gt)
    preds_set = set(tuple(item) for item in main_set_pred)
    
    # True Positives (TP): Items in both gt and preds
    tp = gt_set & preds_set
    
    # False Negatives (FN): Items in gt but not in preds
    fn = gt_set - preds_set
    
    # False Positives (FP): Items in preds but not in gt
    fp = preds_set - gt_set
    
    # Convert sets back to lists if needed
    tp_list = list(tp)
    fn_list = list(fn)
    fp_list = list(fp)
    
    # # Print results
    # print("True Positives (TP):", tp_list)
    # print("False Negatives (FN):", fn_list)
    # print("False Positives (FP):", fp_list)
    
    # # Number of items
    # print("Number of TP:", len(tp_list))
    # print("Number of FN:", len(fn_list))
    # print("Number of FP:", len(fp_list))
    if mode == 'macro':
        # Recall
        precision= len(tp_list)/(len(tp_list)+len(fp_list))
        recall= len(tp_list)/(len(tp_list)+len(fn_list))
        return precision, recall
    elif mode == 'micro':
        return len(tp_list),len(fp_list), len(fn_list)
    else:
        raise ValueError ('Supports only macro amd micro as mode')
def calc_ce(gt_sent_list,pred_sent_list, mode='micro'):
    if mode == 'macro':
        count=0
        total_prec = 0
        total_rec = 0
        for i,j in zip(gt_sent_list,pred_sent_list):
            precision, recall=ce_metric(i,j,mode='macro')
            total_prec=total_prec+precision
            total_rec=total_rec+recall
            count+=1
        macro_prec=total_prec/count
        macro_rec=total_rec/count
        return macro_prec, macro_rec # macro is done
    elif mode == 'micro':
        tp_tot=0
        fp_tot=0
        fn_tot=0
        for i,j in zip(gt_sent_list,pred_sent_list):
            tp,fp, fn=ce_metric(i,j,mode='micro')
            tp_tot=tp_tot+tp
            fp_tot=fp_tot+fp
            fn_tot=fn_tot+fn
        print(tp_tot)
        print(fp_tot)
        print(fn_tot)
        micro_prec=tp_tot/(tp_tot+fp_tot)
        micro_rec=tp_tot/(tp_tot+fn_tot)
        return micro_prec, micro_rec
    else:
        raise ValueError ('Supports only macro and micro as mode')



def convert(val):
    if val==0:
        return -1
    elif val == 1:
        return val
    else:
        return 0
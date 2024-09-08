from pycocoevalcap.bleu.bleu import Bleu
from pycocoevalcap.meteor import Meteor
from pycocoevalcap.rouge import Rouge


def compute_scores(gts, res, verbose=0):
    """
    Performs the MS COCO evaluation using the Python 3 implementation (https://github.com/salaniz/pycocoevalcap)

    :param gts: Dictionary with the image ids and their gold captions,
    :param res: Dictionary with the image ids ant their generated captions
    :print: Evaluation score (the mean of the scores of all the instances) for each measure
    """
    #print('gts', gts)
    #print('Preds', res)
    # Set up scorers
    scorers = [
        (Bleu(4), ["BLEU_1", "BLEU_2", "BLEU_3", "BLEU_4"]),
        (Meteor(), "METEOR"),
        (Rouge(), "ROUGE_L")
    ]
    eval_res = {}
    eval_res_ind={} # obtain bleu scores for each report
    # Compute score for each metric
    for scorer, method in scorers:
        try:
            score, scores = scorer.compute_score(gts, res)
            #print('scores', scores)
        except TypeError:
            score, scores = scorer.compute_score(gts, res)
        if type(method) == list:
            for sc, m in zip(score, method):
                eval_res[m] = sc
            for scs, m in zip(scores, method):
                eval_res_ind[m] = scs
        else:
            eval_res[method] = score 
            eval_res_ind[m] = scores # We add the list of bleu scores for individual reports
    return eval_res, eval_res_ind

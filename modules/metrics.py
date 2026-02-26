from pycocoevalcap.bleu.bleu import Bleu
from pycocoevalcap.meteor import Meteor
from pycocoevalcap.rouge import Rouge
from f1chexbert import F1CheXbert



def compute_scores(gts, res, use_clinical=False):
    """
    Performs the MS COCO evaluation using the Python 3 implementation (https://github.com/salaniz/pycocoevalcap)

    :param gts: Dictionary with the image ids and their gold captions,
    :param res: Dictionary with the image ids ant their generated captions
    :print: Evaluation score (the mean of the scores of all the instances) for each measure
    """

    # Set up scorers
    scorers = [
        (Bleu(4), ["BLEU_1", "BLEU_2", "BLEU_3", "BLEU_4"]),
        (Meteor(), "METEOR"),
        (Rouge(), "ROUGE_L")
    ]
    eval_res = {}
    f1chexbert = F1CheXbert()

    for scorer, method in scorers:
        try:
            score, scores = scorer.compute_score(gts, res, verbose=0)
        except TypeError:
            score, scores = scorer.compute_score(gts, res)
        if type(method) == list:
            for sc, m in zip(score, method):
                eval_res[m] = sc
        else:
            eval_res[method] = score

    if use_clinical:
        f1chexbert = F1CheXbert()

        # dict -> list[str]
        gts_txt = [v[0] for v in gts.values()]
        res_txt = [v[0] for v in res.values()]

        accuracy, pe_accuracy, per_report_precision_14, per_report_recall_14, per_report_f1_14, cr, cr_5 = f1chexbert(
            hyps=res_txt,
            refs=gts_txt
        )

        # corpus-level
        eval_res["CLINICAL_ACC"] = accuracy
        eval_res["CLINICAL_precision_14"] = per_report_precision_14.mean()
        eval_res["CLINICAL_recall_14"] = per_report_recall_14.mean()
        eval_res["CLINICAL_f1_14"] = per_report_f1_14.mean()
        eval_res["CLINICAL_micro avg_precision"] = cr['micro avg']['precision']
        eval_res["CLINICAL_micro avg_recall"] = cr['micro avg']['recall']
        eval_res["CLINICAL_micro avg_f1-score"] = cr['micro avg']['f1-score']
        eval_res["CLINICAL_macro avg_precision"] = cr['macro avg']['precision']
        eval_res["CLINICAL_macro avg_recall"] = cr['macro avg']['recall']
        eval_res["CLINICAL_macro avg_f1-score"] = cr['macro avg']['f1-score']

    return eval_res


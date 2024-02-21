import logging.handlers
import argparse
import os
import sys
import torch
sys.path.append('.')
import pandas as pd
import numpy as np
from moverscore_v2 import get_idf_dict, word_mover_score
import evaluate
import os
from collections import defaultdict
from typing import List
from transformers import AutoModel

os.environ['MOVERSCORE_MODEL'] =  'xlm-roberta-large'

logger = logging.getLogger("metrics")
ch = logging.StreamHandler(sys.stdout)
ch.setLevel(logging.DEBUG)
formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
ch.setFormatter(formatter)
logger.setLevel(logging.DEBUG)
logger.addHandler(ch)
logger.propagate = False  


def sentence_score(hypothesis: str, references: List[str], trace=0):
    idf_dict_hyp = defaultdict(lambda: 1.)
    idf_dict_ref = defaultdict(lambda: 1.)

    hypothesis = [hypothesis] * len(references)

    sentence_score = 0
    scores = word_mover_score(references, hypothesis, idf_dict_ref, idf_dict_hyp, stop_words=[], n_gram=1, remove_subwords=False)

    sentence_score = np.mean(scores)

    if trace > 0:
        print(hypothesis, references, sentence_score)

    return sentence_score

def calculate_metrics(predictions, references):
    results = {}
    bertscore = evaluate.load('bertscore')
    results["BERTScore"] = bertscore.compute(predictions=predictions, references=references, lang="es", model_type="xlm-roberta-large")
    moverscore = sentence_score(predictions[0], references)
    results["MoverScore-Sentence"] = moverscore
    return results

def validate_files(pred_files, gold_files):
    if 'id' in pred_files and 'Counternarrative' in pred_files and 'id' in gold_files and 'Reference-counternarrative' in gold_files:
        return True 
    else:
        if 'id' not in pred_files or 'Counternarrative' not in pred_files:
            logger.error('Bad format for pred file {}. Cannot score.'.format(pred_file))
        else:
            logger.error('Bad format for gold file {}. Cannot score.'.format(gold_file))
        return False

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--gold_file_path",
        '-g',
        type=str,
        required=True,
        help="Paths to the file with gold annotations."
    )
    parser.add_argument(
        "--pred_file_path",
        '-p',
        type=str,
        required=True,
        help="Path to the file with predictions"
    )
    parser.add_argument(
    "--log_to_file",
    "-l",
    action='store_true',
    default=False,
    help="Set flag if you want to log the execution file. The log will be appended to <pred_file>.log"
  )
    args = parser.parse_args()
    pred_file = args.pred_file_path
    gold_file = args.gold_file_path

    logger.info("Reading gold predictions from file {}".format(args.gold_file_path))
    gold= pd.read_csv(gold_file)
    logger.info('Reading predictions file {}'.format(args.pred_file_path))
    preds= pd.read_csv(pred_file)
    
    if validate_files(preds,gold):
        logger.info('Prediction file format is correct')
        dfFinal = gold.merge(preds, on="id", how="left")
        bertScoreP, bertScoreR, bertScoreF1, moverScore = [],[],[],[]
        for i in dfFinal.index:
            dictResult = calculate_metrics(predictions = [dfFinal['Counternarrative'][i]], references=[dfFinal['Reference-counternarrative'][i]])
            bertScoreP, bertScoreR, bertScoreF1, moverScore = dictResult['BERTScore']['precision'][0],dictResult['BERTScore']['recall'][0],dictResult['BERTScore']['f1'][0], dictResult['MoverScore-Sentence']
        dfFinal['bertScoreP'],dfFinal['bertScoreR'],dfFinal['bertScoreF1'],dfFinal['moverScore'] = bertScoreP, bertScoreR, bertScoreF1, moverScore
        logger.info("BERTScore-f1={:.5f}\tBERTScore-prec={:.5f}\tBERTScore-rec={:.5f}\tMoverScore={:.5f}".format(np.mean(list(dfFinal['bertScoreP'])),np.mean(list(dfFinal['bertScoreR'])),np.mean(list(dfFinal['bertScoreF1'])),np.mean(list(dfFinal['moverScore']))))
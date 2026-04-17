import os 
import io
import subprocess
import csv
import pandas as pd
import math
import re
import time
import sys
import hashlib
import json
import numpy as np
import scipy.stats as stats
from rouge import Rouge
import bert_score.score as bertscore
import bleurt.score as bleurtscore
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"  # 0 = all logs, 1 = warnings, 2 = errors, 3 = fatal only
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"

log_file = "eval_result.txt"
log = open(log_file, "w", encoding="utf-8")

def dual_print(*args, **kwargs):
    print(*args, **kwargs)
    print(*args, **kwargs, file=log,flush=True)
#------------- ACCURACY -----------#

def compute_accuracy(reference_flags, reference_sent_id,predicted_flags,candidate_sent_id):
    match_sent ={}
    # Error Flags Accuracy (missing predictions are counted as false)
    matching_flags_nb = 0
    
    for text_id in reference_flags:
        if text_id in predicted_flags and reference_flags[text_id] == predicted_flags[text_id]:
            matching_flags_nb += 1
            
    flags_accuracy = matching_flags_nb / len(reference_flags)
    
    # Error Sentence Detection Accuracy (missing predictions are counted as false)
    matching_sentence_nb = 0
    matching_na = 0

    
    for text_id in reference_sent_id:
        if text_id in candidate_sent_id and candidate_sent_id[text_id] == reference_sent_id[text_id]:
            matching_sentence_nb += 1
            match_sent[text_id] = candidate_sent_id[text_id] 
            dual_print("text_id, flag, sent_id matched :",text_id,predicted_flags[text_id],candidate_sent_id[text_id])
    sent_accuracy = matching_sentence_nb / len(reference_sent_id)
    dual_print("deno for sent id =", len(reference_sent_id))
    return {
        #Added:
        "\nMatching_flags": matching_flags_nb,
        "\nMatching_sentence": matching_sentence_nb,
        "\nError Flags Accuracy": flags_accuracy,
        "\nError Sentence Detection Accuracy": sent_accuracy
    }

#------------- ERROR CORRECTION METRICS -------------#


def increment_counter(counters, counter_name):
    counters[counter_name] = counters[counter_name] + 1

def clip(value): # clip to a 0-1 value
    return max(0, min(1, value))

class NLGMetrics(object):

    def __init__(self, metrics = ['ROUGE']):
        self.metrics = metrics
    
    def compute(self, references, predictions, counters):
        results = {}
        assert len(predictions) == len(references), "Predictions and references do not have the same size."
        results['aggregate_subset_check'] = np.array([0 for x in range(len(predictions))])
        aggregate_components = 0

        if 'ROUGE' in self.metrics:
            rouge = Rouge() 
            rouge_scores = rouge.get_scores(predictions, references)
                            
            rouge1f_scores = []
            rouge2f_scores = []
            rougeLf_scores = []
            
            for i in range(len(references)):
                r1f = rouge_scores[i]["rouge-1"]["f"]
                r2f = rouge_scores[i]["rouge-2"]["f"]
                rlf = rouge_scores[i]["rouge-l"]["f"]
                
                rouge1f_scores.append(r1f)	
                rouge2f_scores.append(r2f)
                rougeLf_scores.append(rlf)
                
            # for checking comparison with composite
            rouge1check = np.array(rouge1f_scores).mean()
            rouge2check = np.array(rouge2f_scores).mean()
            rougeLcheck = np.array(rougeLf_scores).mean()

            results['R1F_subset_check'] = rouge1check
            results['R2F_subset_check'] = rouge2check
            results['RLF_subset_check'] = rougeLcheck
            
            ###############################
            # Composite score computation #
            ###############################
            
            """
            NLG METRIC on sentence vs. sentence cases + ones or zeros 
            when either the reference or the candidate correction is NA
            """
            
            rouge1score = np.array(rouge1f_scores).sum()
            rouge2score = np.array(rouge2f_scores).sum()
            rougeLscore = np.array(rougeLf_scores).sum()
            
            composite_score_rouge1 = (rouge1score + counters["system_provided_correct_na"]) / counters["total_texts"]
            composite_score_rouge2 = (rouge2score + counters["system_provided_correct_na"]) / counters["total_texts"]
            composite_score_rougeL = (rougeLscore + counters["system_provided_correct_na"]) / counters["total_texts"]

            results['R1FC'] = composite_score_rouge1
            results['R2FC'] = composite_score_rouge2
            results['RLFC'] = composite_score_rougeL

        if 'BERTSCORE' in self.metrics:
            bertScore_Precision, bertScore_Recall, bertScore_F1 = bertscore(predictions, references, model_type='microsoft/deberta-xlarge-mnli', lang='en', device ='cpu' , verbose=True, rescale_with_baseline=True) # roberta-large
            
            bertscores = bertScore_F1.numpy()
            ## clip scores to [0,1]
            bertscores = np.array([clip(num) for num in bertscores])

            results['BERTSCORE_subset_check'] = bertscores.mean()
            composite_score_bert = (bertscores.sum() + counters["system_provided_correct_na"]) / counters["total_texts"]
            results['BERTC'] = composite_score_bert

            # sum element-wise for later aggregate score computation
            results['aggregate_subset_check'] = results['aggregate_subset_check'] + np.array(bertscores)
            aggregate_components += 1
        

        if 'BLEURT' in self.metrics:
            bleurtscorer = bleurtscore.BleurtScorer(checkpoint="./../../bleurt_cp/BLEURT-20")
            
            bleurtscores = bleurtscorer.score(references=references, candidates=predictions, batch_size =1)
            ## clip scores to [0,1]
            bleurtscores = np.array([clip(num) for num in bleurtscores])

            results['BLEURT_subset_check'] = bleurtscores.mean()
            composite_score_bleurt = (bleurtscores.sum() + counters["system_provided_correct_na"]) / counters["total_texts"]
            results['BLEURTC'] = composite_score_bleurt

            # sum element-wise for later aggregate score computation
            results['aggregate_subset_check'] = results['aggregate_subset_check'] + np.array(bleurtscores) 
                    
            aggregate_components += 1

        if aggregate_components > 0:
            aggregate_subset_scores = results['aggregate_subset_check'] / aggregate_components
            composite_score_agg = (aggregate_subset_scores.sum() + counters["system_provided_correct_na"]) / counters["total_texts"]
            
            results['aggregate_subset_check'] = aggregate_subset_scores.mean() 
            results['AggregateC'] = composite_score_agg

        return results

def get_nlg_eval_data(reference_corrections, candidate_corrections, remove_nonprint = False):
    references = []
    predictions = []
    
    counters = {
        "total_texts": 0,
        "reference_na": 0,
        "total_system_texts": 0,
        "system_provided_na": 0,
        "system_provided_correct_na": 0,
    }
    
    for text_id in reference_corrections:
        increment_counter(counters, "total_texts")
        
        # removing non ascii chars
        reference_correction = reference_corrections[text_id]
        
        if remove_nonprint:
            reference_correction = ''.join(filter(lambda x: x in string.printable, str(reference_correction)))
            
        if reference_correction == "NA":
            increment_counter(counters, "reference_na")
            
        if text_id in candidate_corrections:
            increment_counter(counters, "total_system_texts")
            candidate = candidate_corrections[text_id]
            
            if remove_nonprint:
                candidate = ''.join(filter(lambda x: x in string.printable, candidate))
                
            if candidate == "NA":
                increment_counter(counters, "system_provided_na")
                
            # matching NA counts as 1
            if reference_correction == "NA" and candidate == "NA":
                increment_counter(counters, "system_provided_correct_na")
                continue
                
            # Run provided "NA" when a correction was required (=> 0)
            # or Run provided a correction when "NA" was required (=> 0)
            if candidate == "NA" or reference_correction == "NA":
                continue
                
            # remaining case is both reference and candidate are not "NA"
            # both are inserted/added for ROUGE/BLEURT/etc. computation
            references.append(reference_correction)
            predictions.append(candidate)
    

    
    return references, predictions, counters

if len(sys.argv) < 2:
    print("Usage: python evaluate_results.py <eval_input.json>")
    sys.exit(1)

# ---------- Load serialized data ----------
with open(sys.argv[1], "r", encoding="utf-8") as f:
#with open("eval_data.json", "r", encoding="utf-8") as f:
    data = json.load(f)

reference_corrections = data["reference_corrections"]
reference_flags     = data["reference_flags"]
reference_sent_id = data["reference_sent_id"]
candidate_corrections = data["candidate_corrections"]
candidate_flags = data["candidate_flags"]
candidate_sent_id = data["candidate_sent_id"]
model_name = data["model"]

dual_print(f"\n--- Evaluating model {model_name} ---")

# Accuracy
accuracy_results = compute_accuracy(reference_flags,reference_sent_id,candidate_flags,candidate_sent_id)
dual_print("Accuracy Results of {model_name}:\n", accuracy_results)

# Rouge
references, predictions, counters = get_nlg_eval_data(reference_corrections, candidate_corrections)
metrics = NLGMetrics(metrics = ['ROUGE', 'BERTSCORE','BLEURT'])
nlg_eval_results = metrics.compute(references, predictions, counters) 

dual_print("NLG Eval Results:\n", nlg_eval_results) 
dual_print()

# debug check
dual_print(counters)
dual_print()
dual_print(f"\n---- End of model {model_name}----n")

print(f"\n===== End  =====")

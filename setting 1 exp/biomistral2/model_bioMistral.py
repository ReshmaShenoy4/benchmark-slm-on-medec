import os 
import io
import subprocess
os.environ["CUDA_VISIBLE_DEVICES"] = "1"
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"

import csv
import pandas as pd
import math
import re
import time
import sys
import hashlib
import json

#--------------- Logging ---------------#
log_file = "Models_bioMistral.txt"
log = open(log_file, "w", encoding="utf-8")

def dual_print(*args, **kwargs):
    print(*args, **kwargs)
    print(*args, **kwargs, file=log,flush=True)

#--------------- MODEL LIST ---------------#

models = [
#          ("jsk/bio-mistral:latest", "MR_BioMistral_st2.txt", "MR_BioMistral_print.txt"),
       ("hf.co/MaziyarPanahi/BioMistral-7B-GGUF","model_response.txt","invalid_lines.txt"),
]

#--------------- TEST and EXAMPLE CSV ------------#
csv_file = "./../../../../git_dataset/MEDEC/MEDEC-MS/test_set.csv"
test_df = pd.read_csv(csv_file).dropna(how="all")
training_csv = "./../../../../git_dataset/MEDEC/MEDEC-MS/training_set.csv"
train_df = pd.read_csv(training_csv).dropna(how="all")


#------------- PROMPT GENERATION -----------#

SEED = 1648

def example_block(text_id:str,sentences:str, error_flag:str, error_id:str, corrected_sentence:str)->str:
    return f"""Question:
text_id : {text_id}
{str(sentences).strip()}
Answer: {text_id} {error_flag} {error_id} {corrected_sentence}"""

def det_index(key: str, n: int, seed: int) -> int:
    h = hashlib.sha256((str(key) + "|" + str(seed)).encode("utf-8")).digest()
    return int.from_bytes(h[:8], "big") % n


base_prompt = """<s>[INST] 
##Instructions
You are a skilled medical doctor reviewing clinical text for factual or diagnostic errors.
Each text may be correct or contain one error related to treatment, management, cause, or diagnosis.

Follow the detailed review guidelines below.

Guidelines:

Apply sound clinical reasoning to every sentence.

Focus only on errors in treatment, management, cause, or diagnosis.

If no error is present, set <error_flag> = 0, <error_sentence_id> = -1, and <corrected_sentence> = NA.

If an error exists, identify the sentence ID, provide <error_flag> = 1, <error_sentence_id> = that ID, and write one corrected version of that sentence.

Ensure your output follows exactly this schema:

text_id <error_flag> <error_sentence_id> <corrected_sentence>
For example:
text-id-1 0 -1 NA
text-id-2 1 8 "correction of sentence 8..."


Do not echo the input, give explanations, or add commentary.

Double-check that your output has four tokens separated by spaces: text_id, error_flag, error_sentence_id, and corrected_sentence / NA.

Input format:

First line: text_id

Following lines: one sentence per line in the format "<sent_id> <sentence>".

Output: one line only, matching the schema above.
"""


prompts = {}
N = len(train_df)

for _,row in test_df.iterrows():
    text_id = str(row["Text ID"])
    sentences = str(row["Sentences"]).strip()
    j1 = det_index(text_id + "_a",N,SEED)
    j2 = det_index(text_id + "_b",N,SEED)
    eg1 = train_df.iloc[j1]
    eg2 = train_df.iloc[j2]
    eg_text = ( "Here is an example:\n"
        + example_block("ms-test-0", eg1["Sentences"], eg1["Error Flag"], eg1["Error Sentence ID"], eg1["Corrected Sentence"])
        #+ "\n"
        #+ example_block("ms-test-1", eg2["Sentences"], eg2["Error Flag"], eg2["Error Sentence ID"], eg2["Corrected Sentence"])
        + "\nEnd of example.\n"
    )

    question = f"""Question:
{text_id}
{sentences}
\n"""
    full_prompt = base_prompt + "\n" + eg_text + "## User Question" + '\n' + "Please answer the following question:\n" + question + "[/INST]</s>"
    prompts[text_id] = full_prompt


##------------- SAVING PROMPTS -----------#
with open("prompts.json","w",encoding="utf-8") as f:
    json.dump(prompts, f, indent=2, ensure_ascii=False)

dual_print(f"[PromptCache] Wrote {len(prompts)} prompts to prompts.json")


#------------- PARSING -----------#

def parse_reference_file(filepath):
    reference_corrections = {}
    reference_flags = {}
    reference_sent_id = {}

    df = pd.read_csv(filepath)
    #Added below line because of extra blank rows after 597 prompts!
    df = df.dropna(subset=["Text ID"]).reset_index(drop=True) 
 
    for index, row in df.iterrows():
        text_id = row['Text ID']
        corrected_sentence = row['Corrected Sentence']
        
        if not isinstance(corrected_sentence, str):
            if math.isnan(corrected_sentence):
                corrected_sentence = "NA"
            else:
                corrected_sentence = str(corrected_sentence)
                corrected_sentence = corrected_sentence.replace("\n", " ") \
                  .replace("\r", " ").strip()
                  
        reference_corrections[text_id] = corrected_sentence
        reference_flags[text_id] = str(int(row['Error Flag']))#Added to avoid "1.0" with "1"
        reference_sent_id[text_id] = str(int(row['Error Sentence ID']))#same reason
    return reference_corrections,reference_flags,reference_sent_id


def parse_run_submission_file(filepath,print_file):
    
    file = open(filepath,"r")
    candidate_corrections = {}
    predicted_flags = {}
    candidate_sent_id = {}
    
    lines = file.readlines()
    with open(print_file,"w",encoding="utf-8") as f:    
        for line in lines:
            line = line.strip()
            
            if len(line) == 0:
                continue
                
            if not re.fullmatch(r'[a-z0-9\-]+\s[0-9]+\s\-?[0-9]+\s.+', line):
                wr = "Invalid line: " + line
                f.write(wr + "\n")
                #print("Invalid line: ", line)
                continue
                
            # replacing consecutive spaces 
            # Modified this otherwise error:bad escape (end of pattern)
            # ie ' ' was being replaced by line and if line has a \, then escape/backrefrence.
            line = re.sub(r'\s+',' ', line)
            
            # parsing
            items = line.split()
            text_id = items[0]
            error_flag = items[1]
            sentence_id = items[2]
            corrected_sentence = ' '.join(items[3:]).strip()
            
            # debug - parsing check
            # print("{} -- {} -- {} -- {}".format(text_id, error_flag, sentence_id, corrected_sentence))

            predicted_flags[text_id] = error_flag
            candidate_sent_id[text_id] = sentence_id

            # processing candidate corrections
            # removing quotes

            while corrected_sentence.startswith('"') and len(corrected_sentence) > 1:
                corrected_sentence = corrected_sentence[1:]
                
            while corrected_sentence.endswith('"') and len(corrected_sentence) > 1:
                corrected_sentence = corrected_sentence[:-1]
                       
            if error_flag == "0":
                # enforcing "NA" in predicted non-errors (used for consistent/reliable eval)
                candidate_corrections[text_id] = "NA"
            else:
                candidate_corrections[text_id] = corrected_sentence

    return candidate_corrections, predicted_flags, candidate_sent_id

#------------ TESTING MODEL -------------#
with open("prompts.json", "r", encoding="utf-8") as fcache:
    fixed_prompts = json.load(fcache)

for model,output_file,print_file in models:
    dual_print(f"\n---- Running model {model} ----")
    subprocess.run(["ollama","pull",model],check=True)
    with open(output_file,"w",encoding="utf-8") as f:
        start = time.time() 
        for i,row in test_df.iterrows():
            text_id = str(row["Text ID"])
            full_prompt = fixed_prompts[text_id]
            result = subprocess.run(["ollama","run",model,full_prompt],text=True,capture_output=True)
            model_response = result.stdout.strip()
            f.write(model_response + "\n")
        end = time.time()
        dual_print(f"Execution time: {end - start:.4f} seconds")
        dual_print(f"\n--- Evaluating model {model} ----")

    submission_file = output_file
    reference_csv_file = csv_file

    reference_corrections, reference_flags,reference_sent_id = parse_reference_file(reference_csv_file)
    candidate_corrections, candidate_flags,candidate_sent_id = parse_run_submission_file(submission_file,print_file)
    data_bundle = {
        "model": model,
        "reference_corrections": reference_corrections,
        "reference_flags": reference_flags,
        "reference_sent_id": reference_sent_id,
        "candidate_corrections": candidate_corrections,
        "candidate_flags": candidate_flags,
        "candidate_sent_id": candidate_sent_id,
    }

    with open("eval_data.json", "w", encoding="utf-8") as f:
        json.dump(data_bundle, f, ensure_ascii=False)

# ---------------- CALL EVALUATION ---------------- #
    dual_print(f"Evaluating {model} ...")
    subprocess.run(["python", "eval.py", "eval_data.json"])


import gc
gc.collect()

dual_print(f"\n===== End  =====")

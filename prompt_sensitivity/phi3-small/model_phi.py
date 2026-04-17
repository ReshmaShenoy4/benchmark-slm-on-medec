import os,sys,glob 
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
import hashlib
import json
import onnx
from onnx.external_data_helper import uses_external_data, ExternalDataInfo
import onnxruntime_genai as og
import numpy as np
import scipy.stats as stats

#--------------- Logging ---------------#
log_file = "phi_prints.txt"
log = open(log_file, "w", encoding="utf-8")

def dual_print(*args, **kwargs):
    print(*args, **kwargs)
    print(*args, **kwargs, file=log,flush=True)

#--------------- MODEL LIST ---------------#



models = [ 
        ("onnx:./phi3-small/cuda-fp16", "model_response.txt", "invalid_lines.txt"),
]

#--------------- TEST and EXAMPLE CSV ------------#
csv_file = "./../../../git_dataset/MEDEC/MEDEC-MS/test_set.csv"
test_df = pd.read_csv(csv_file).dropna(how="all")
training_csv = "./../../../git_dataset/MEDEC/MEDEC-MS/training_set.csv"
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

#base_prompt = f""" The following is a medical narative about a patient.You are a skilled medical doctor reviewing the clinical text.The text is either correct or contains one error.The text has one sentence per line.Each line starts with the sentence ID,followed by space and then the sentence to check. Check every sentence of the text.If the text is correct return text_id, followed by spaceand then 0,followed by space and then -1.If the text has a medical error related to treatment, management, cause, or diagnosis, return text_id, followed by space and then 1, followed by space and then sentence id of the sentence containing the error, followed by space, and then corrected version of the sentence.Finding and correcting the error requires medical knowledge and reasoning.
#"""



base_prompt_struct = """<|system|>
You are a skilled medical doctor reviewing clinical text for factual or diagnostic errors.The entire text is either has 0 error or 1 error.
Finding and correcting the error requires medical knowledge and reasoning.

TASK:
    To review the text and give an output

RULE:
Each text has:
- A text_id (first line)
- One sentence per line: "<sent_id> <sentence>"


OUTPUT MUST MATCH THIS SCHEMA EXACTLY
text_id <error_flag> <error_sentence_id> <corrected_sentence>

- <error_flag> = 0  → No error
- <error_flag> = 1  → A medical error exists (treatment / management / cause / diagnosis)
- <error_flag> should be either 0 or 1 only.
- <error_sentence_id> = positive ID number of the sentence with the error if error_flag=1, or -1 if no error
- <corrected_sentence> = corrected version if error_flag=1, else NA

For example:
text-id-1 0 -1 NA
text-id-2 1 8 "correction of sentence 8..."

- Use medical reasoning.
- Output only ONE line.
- Do NOT echo input or explain.
-The first four tokens must always be: text_id, error_flag( 0 or 1), error_sentence_id, corrected_sentence/NA
<|end|>
<|user|>
"""



base_prompt_text = """<|system|> You are a skilled medical doctor reviewing clinical text for factual or diagnostic errors.
Each text may be correct or contain one error related to treatment, management, cause, or diagnosis.

Follow the detailed review guidelines below.

Guidelines:

Apply sound clinical reasoning to every sentence.

Focus only on errors in treatment, management, cause, or diagnosis.

If an error exists, identify the sentence ID, provide <error_flag> = 1, <error_sentence_id> = that ID, and <corrected_sentence> = corrected version of that sentence.

If no error is present, set <error_flag> = 0, <error_sentence_id> = -1, and <corrected_sentence> = NA.

Ensure your output follows exactly this schema:

text_id <error_flag> <error_sentence_id> <corrected_sentence>
For example:
text-id-1 1 8 "correction of sentence 8..."
text-id-2 0 -1 NA


Do not echo the input, give explanations, or add commentary.

Double-check that your output has four tokens separated by spaces: text_id, error_flag, error_sentence_id, and corrected_sentence / NA.

Input format:

First line: text_id

Following lines: one sentence per line in the format "<sent_id> <sentence>".

Output: one line only, matching the schema above.
<|end|>
<|user|>
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
<|end|>\n"""
    full_prompt = base_prompt_text + "\n"  + "Please answer the following question:\n" + question + "<|assistant|>"
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


#---------------------- PHI 3 SMALL SETUP ----------------------------#
folder = sys.argv[1] if len(sys.argv) > 1 else "./../phi3-small/cuda-fp16"
onnx_files = [p for p in glob.glob(os.path.join(folder, "*.onnx"))]
assert onnx_files, f"No .onnx found in {folder}"
onnx_path = onnx_files[0]

# 1) Find the exact external-data filename the graph expects
m = onnx.load(onnx_path, load_external_data=False)
expected = None
for t in m.graph.initializer:
    if uses_external_data(t):
        info = ExternalDataInfo(t)
    if info.location:
        expected = os.path.normpath(os.path.join(os.path.dirname(onnx_path), info.location))
        break
# 2) Ensure a large *.onnx.data exists at that exact path (rename/symlink if needed)
if expected:
    if not (os.path.exists(expected) and os.path.getsize(expected) > 1_000_000_000):
    # try to locate a large data file in the same folder and link/rename it to expected
        cands = [p for p in glob.glob(os.path.join(folder, "*.onnx.data")) if os.path.getsize(p) > 1_000_000_000]
        assert cands, "No large .onnx.data found. Re-download the model folder."
        src = cands[0]
        try:
            if os.path.exists(expected): os.remove(expected)
            os.symlink(os.path.basename(src), expected)
        except Exception:
            import shutil 
            shutil.copy2(src, expected)

cfg = og.Config(folder)          # point to the FOLDER
cfg.clear_providers()
cfg.append_provider("cuda")      # or "tensorrt"
onnx_model = og.Model(cfg)
tok = og.Tokenizer(onnx_model)
param = og.GeneratorParams(onnx_model)
# param.set_search_options(...)

#------------ TESTING MODEL -------------#
with open("prompts.json", "r", encoding="utf-8") as fcache:
    fixed_prompts = json.load(fcache)

for model_name,output_file,print_file in models:
    dual_print(f"\n---- Running model {model_name} ----")
    #subprocess.run(["ollama","pull",model],check=True)
    with open(output_file,"w",encoding="utf-8") as f:
        start = time.time() 
        for i,row in test_df.iterrows():
            text_id = str(row["Text ID"])
            full_prompt = fixed_prompts[text_id]
            ids = tok.encode(full_prompt)
            gen = og.Generator(onnx_model, param)
            gen.append_tokens(ids)
            out_tokens =[]
            while not gen.is_done():
                gen.generate_next_token()
                #out = gen.get_next_tokens()[0]
                out_tokens.extend(gen.get_next_tokens())
            result = tok.decode(out_tokens).strip()
            dual_print("Sample output:", result[:200])

            #result = subprocess.run(["ollama","run",model,full_prompt],text=True,capture_output=True)
            #model_response = result.stdout.istrip()
            f.write(result + "\n")
        end = time.time()
        dual_print(f"Execution time: {end - start:.4f} seconds")
        dual_print(f"\n--- Evaluating model {model_name} ----")

    submission_file = output_file
    reference_csv_file = csv_file

    reference_corrections, reference_flags,reference_sent_id = parse_reference_file(reference_csv_file)
    candidate_corrections, candidate_flags,candidate_sent_id = parse_run_submission_file(submission_file,print_file)
    data_bundle = {
                "model": model_name,
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
    dual_print(f"Evaluating {model_name} ...")
    subprocess.run(["python", "eval.py", "eval_data.json"])

# --- After all generations and evaluations ---
try:
    del gen
    del tok
    del onnx_model
    del param
    del cfg
except Exception as e:
    dual_print(f"Cleanup warning: {e}")

import gc
gc.collect()




dual_print(f"\n===== End  =====")
























































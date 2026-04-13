import argparse
import collections
import json
import re
import string
import torch
import copy
from pathlib import Path

from nltk import sent_tokenize
import numpy as np
from rouge_score import rouge_scorer, scoring
from tqdm import tqdm
import sys
import logging
logging.basicConfig(format='%(asctime)s - %(levelname)s - %(name)s - %(message)s',
                    datefmt='%m/%d/%Y %H:%M:%S')
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from transformers import (
    AutoModelForSeq2SeqLM,
    AutoTokenizer,
    pipeline
)

from config_paths import EVALUATION_MODELS
from utils import normalize_answer, get_max_memory, remove_citations

QA_MODEL = EVALUATION_MODELS["qa"]
AUTOAIS_MODEL = EVALUATION_MODELS["autoais"]
MAUVE_MODEL = EVALUATION_MODELS["mauve"]

global autoais_model, autoais_tokenizer
autoais_model, autoais_tokenizer = None, None

def compute_f1(a_gold, a_pred):
    """Compute F1 score between two strings."""

    def _get_tokens(s):
        if not s:
            return []
        return normalize_answer(s).split()

    gold_toks = _get_tokens(a_gold)
    pred_toks = _get_tokens(a_pred)

    common = collections.Counter(gold_toks) & collections.Counter(pred_toks)
    num_same = sum(common.values())

    if len(gold_toks) == 0 or len(pred_toks) == 0:
        # If either is no-answer, then F1 is 1 if they agree, 0 otherwise
        return int(gold_toks == pred_toks)

    if num_same == 0:
        return 0

    precision = 1.0 * num_same / len(pred_toks)
    recall = 1.0 * num_same / len(gold_toks)
    f1 = (2 * precision * recall) / (precision + recall)

    return f1


def compute_exact(a_gold, a_pred):
    """Check whether two strings are equal up to normalization."""

    return int(normalize_answer(a_gold) == normalize_answer(a_pred))


def exact_presence(short_answers, context):
    """Verify if any of the answers is present in the given context.
    Args:
        short_answers: list of short answers to look for in the context
        context: a paragraph to search for short answers
    Returns:
        true if any of the short answers is present in the context
    """

    n_short_answers = [normalize_answer(sa) for sa in short_answers]
    n_context = normalize_answer(context)

    for ans in n_short_answers:
        if ans in n_context:
            return True

    return False


def compute_rouge(data):
    """Main function for rouge scoring.
    If two references are provided,
    the best score is chosen for each instance.
    Args:
        data: requires field `output` and `answer` (or `annotations` for ASQA)
        metrics: list of evaluation metrics
    Returns:
        dictionary representation of rouge scores
    """
    def _rouge_calculation(hypotheses,
                        references1,
                        references2=[],
                        metrics=['rougeLsum']):

        if references2 == []:
            references2 = references1

        scorer = rouge_scorer.RougeScorer(metrics, use_stemmer=True)
        aggregator = scoring.BootstrapAggregator()

        for i in range(len(hypotheses)):
            scores1 = scorer.score(references1[i], hypotheses[i])
            scores2 = scorer.score(references2[i], hypotheses[i])
            if scores1['rougeLsum'].fmeasure > scores2['rougeLsum'].fmeasure:
                aggregator.add_scores(scores1)
            else:
                aggregator.add_scores(scores2)

        scores = {m: [] for m in metrics}

        for m in metrics:
            fmeasure = aggregator.aggregate()[m].mid.fmeasure
            scores[m].append(fmeasure)

        for m in scores:
            scores[m] = 100 * sum(scores[m]) / len(scores[m])

        return scores

    hypotheses = {}
    references1 = {}
    references2 = {}

    for idx, item in enumerate(data):
        hypotheses[idx] = item["output"]
        if "annotations" in item and item['annotations'] is not None: # For ASQA
            references1[idx] = item["annotations"][0]["long_answer"]
            references2[idx] = item["annotations"][1]["long_answer"]
        else:
            references1[idx] = item["answer"]
            references2[idx] = item["answer"]

    h, r1, r2 = [], [], []

    for key in references1:
        h.append(hypotheses[key])
        r1.append(references1[key])

        if references2 is not None:
            r2.append(references2[key])

    h = ['\n'.join(sent_tokenize(text.lower())) for text in h]
    r1 = ['\n'.join(sent_tokenize(text.lower())) for text in r1]
    r2 = ['\n'.join(sent_tokenize(text.lower())) for text in r2]
    scores = _rouge_calculation(h, r1, r2)

    return scores['rougeLsum']


def compute_str_em(data):
    """Compute STR-EM metric (only for ASQA)
    Args:
        data: requires field `qa_pairs/short_answers` and `output`
    Returns:
        STR-EM and STR-EM-HIT ()
    """

    if 'qa_pairs' not in data[0] or data[0]['qa_pairs'] is None:
        return 0, 0

    acc = []
    hit = []

    for item in data:
        loc_acc = []
        for qa_pair in item['qa_pairs']:
            loc_acc.append(exact_presence(qa_pair['short_answers'], item["output"]))
        acc.append(np.mean(loc_acc))
        hit.append( int(np.mean(loc_acc) == 1) )

    return 100 * np.mean(acc), 100 * np.mean(hit)


def compute_len(data):
    """Compute average length of predictions."""

    res, cntr = 0, 0
    for item in data:
        res += len(item["output"].split())
        cntr += 1
    return res / cntr


def compute_qa(data):
    """Compute QA-based accuracy.
    Args:
        data: requires filed `qa_pairs/short_answers` and `output`
    Returns:
        QA metrics (QA-EM, QA-F1, QA-Hit)
    """

    if 'qa_pairs' not in data[0] or data[0]['qa_pairs'] is None:
        logger.warn("Warning: no QA pairs found in data")
        return {
            'QA-EM': 0,
            'QA-F1': 0,
            'QA-Hit': 0,
        }

    # Load model
    logger.info("Loading the RoBERTa-large SQuAD model for QA-based accuracy...")
    qa_pipeline = pipeline("question-answering", model=QA_MODEL, device=0)
    logger.info("Done")

    # Get prediction
    logger.info("Computing the QA-based accuracy...")
    em, f1, bins = [], [], []
    for item in tqdm(data):
        question = [qa_pair['question'] for qa_pair in item['qa_pairs']]
        context = item['output'] if len(item['output']) > 0 else " "
        results = qa_pipeline(question=question, context=context, handle_impossible_answer=True)
        loc_counter, loc_em, loc_f1 = 0, 0, 0

        for idx, res in enumerate(results):
            answers = item["qa_pairs"][idx]["short_answers"]
            prediction = res["answer"]

            loc_em += max([compute_exact(a, prediction) for a in answers])
            loc_f1 += max([compute_f1(a, prediction) for a in answers])
            loc_counter += 1

        em.append(loc_em / loc_counter)
        f1.append(loc_f1 / loc_counter)
        bins.append(loc_em == loc_counter)

    return {
        'QA-EM': 100 * np.mean(em),
        'QA-F1': 100 * np.mean(f1),
        'QA-Hit': 100 * np.mean(bins)
    }


def compute_mauve(data):
    """Compute Mauve score."""

    logger.info("Computing MAUVE...")
    human_data = []
    model_data = []
    for item in data:
        # Remove ending punctuations
        # Remove any new lines
        # Truncate by 100 words
        human_data.append(' '.join((item['question'] + " " + item['answer'].strip()).split()[:100]).rstrip(string.punctuation))
        model_data.append(' '.join((item['question'] + " " + item['output'].strip()).split()[:100]).rstrip(string.punctuation))
        #human_data.append(' '.join((item['question'] + " " + item['answer'].strip()).split()).rstrip(string.punctuation))
        #model_data.append(' '.join((item['question'] + " " + item['output'].strip()).split()).rstrip(string.punctuation))
    import mauve
    out = mauve.compute_mauve(
        p_text=human_data,
        q_text=model_data,
        device_id=0,
        max_text_length=512,
        verbose=True,
        batch_size=8,
        featurize_model_name=MAUVE_MODEL
    )
    return out.mauve * 100


def _run_nli_autoais(passage, claim):
    """
    Run inference for assessing AIS between a premise and hypothesis.
    Adapted from https://github.com/google-research-datasets/Attributed-QA/blob/main/evaluation.py
    """
    global autoais_model, autoais_tokenizer
    input_text = "premise: {} hypothesis: {}".format(passage, claim)
    input_ids = autoais_tokenizer(input_text, return_tensors="pt").input_ids.to(autoais_model.device)
    with torch.inference_mode():
        outputs = autoais_model.generate(input_ids, max_new_tokens=10)
    result = autoais_tokenizer.decode(outputs[0], skip_special_tokens=True)
    inference = 1 if result == "1" else 0
    return inference


def compute_claims(data):
    global autoais_model, autoais_tokenizer
    if autoais_model is None:
        logger.info("Loading AutoAIS model...")
        autoais_model = AutoModelForSeq2SeqLM.from_pretrained(AUTOAIS_MODEL, torch_dtype=torch.bfloat16, max_memory=get_max_memory(), device_map='cuda:0')
        autoais_tokenizer = AutoTokenizer.from_pretrained(AUTOAIS_MODEL, use_fast=False)

    logger.info("Computing claims...")
    scores = []
    for item in tqdm(data):
        normalized_output = remove_citations(item['output'])
        entail = 0
        claims = item["claims"]
        for claim in claims:
            entail += _run_nli_autoais(normalized_output, claim)
        scores.append(entail / len(claims))
    return 100 * np.mean(scores)


def compute_autoais(data,
                    decontext=False,
                    concat=False,
                    qampari=False,
                    at_most_citations=None,):
    """
    Compute AutoAIS score.

    Args:
        data: requires field `output` and `docs`
              - docs should be a list of items with fields `title` and `text` (or `phrase` and `sent` for QA-extracted docs)
        citation: check citations and use the corresponding references.
        decontext: decontextualize the output
    """

    global autoais_model, autoais_tokenizer
    if autoais_model is None:
        logger.info("Loading AutoAIS model...")
        autoais_model = AutoModelForSeq2SeqLM.from_pretrained(AUTOAIS_MODEL, torch_dtype=torch.bfloat16, max_memory=get_max_memory(), device_map='cuda:0')
        autoais_tokenizer = AutoTokenizer.from_pretrained(AUTOAIS_MODEL, use_fast=False)

    logger.info(f"Running AutoAIS...")

    def _format_document(doc):
        """Format document for AutoAIS."""

        if "sent" in doc:
            # QA-extracted docs
            return "Title: %s\n%s" % (doc['title'], doc['sent'])
        else:
            return "Title: %s\n%s" % (doc['title'], doc['text'])

    ais_scores = []
    ais_scores_prec = []

    sent_total = 0
    sent_mcite = 0
    sent_mcite_support = 0
    sent_mcite_overcite = 0
    autoais_log = []
    for item in tqdm(data):
        # Get sentences by using NLTK
        if qampari:
            sents = [item['question'] + " " + x.strip() for x in item['output'].rstrip().rstrip(".").rstrip(",").split(",")]
        else:
            sents = sent_tokenize(item['output'])
        if len(sents) == 0:
            continue

        target_sents = [remove_citations(sent).strip() for sent in sents]

        entail = 0
        entail_prec = 0
        total_citations = 0
        for sent_id, sent in enumerate(sents):
            target_sent = target_sents[sent_id] # Citation removed and (if opted for) decontextualized
            joint_entail = -1 # Undecided

            # Find references
            ref = [int(r[1:])-1 for r in re.findall(r"\[\d+", sent)] # In text citation id starts from 1
            logger.info(f"For `{sent}`, find citations {ref}")
            if len(ref) == 0:
                # No citations
                joint_entail = 0
            elif any([ref_id >= len(item['docs']) for ref_id in ref]):
                # Citations out of range
                joint_entail = 0
            else:
                if at_most_citations is not None:
                    ref = ref[:at_most_citations]
                total_citations += len(ref)
                joint_passage = '\n'.join([_format_document(item['docs'][psgs_id]) for psgs_id in ref])

            # If not directly rejected by citation format error, calculate the recall score
            if joint_entail == -1: 
                joint_entail = _run_nli_autoais(joint_passage, target_sent)
                autoais_log.append({
                    "question": item['question'],
                    "output": item['output'],
                    "claim": sent,
                    "passage": [joint_passage],
                    "model_type": "NLI",
                    "model_output": joint_entail,
                })

            entail += joint_entail
            if len(ref) > 1:
                sent_mcite += 1

            # calculate the precision score if applicable
            if joint_entail and len(ref) > 1:
                sent_mcite_support += 1
                # Precision check: did the model cite any unnecessary documents?
                for psgs_id in ref:
                    # condition A
                    passage = _format_document(item['docs'][psgs_id]) 
                    nli_result = _run_nli_autoais(passage, target_sent)

                    # condition B
                    if not nli_result:
                        subset_exclude = copy.deepcopy(ref)
                        subset_exclude.remove(psgs_id)
                        passage = '\n'.join([_format_document(item['docs'][pid]) for pid in subset_exclude])
                        nli_result = _run_nli_autoais(passage, target_sent)
                        if nli_result: # psgs_id is not necessary
                            flag = 0
                            sent_mcite_overcite += 1 
                        else:
                            entail_prec += 1
                    else:
                        entail_prec += 1
            else:
                entail_prec += joint_entail 

        sent_total += len(sents)
        ais_scores.append(entail / len(sents))
        ais_scores_prec.append(entail_prec / total_citations if total_citations > 0 else 0) # len(sents))

    if sent_mcite > 0 and sent_mcite_support > 0:
        print("Among all sentences, %.2f%% have multiple citations, among which %.2f%% are supported by the joint set, among which %.2f%% overcite." % (
            100 * sent_mcite / sent_total, 
            100 * sent_mcite_support / sent_mcite, 
            100 * sent_mcite_overcite / sent_mcite_support
        ))

    return {
        "citation_rec": 100 * np.mean(ais_scores),
        "citation_prec": 100 * np.mean(ais_scores_prec),
    }


def compute_qampari_f1(data, cot=False):
    prec = []
    rec = []
    rec_top5 = []
    f1 = []
    f1_top5 = []

    num_preds = []
    index = 0
    for item in data:
        index +=1
        if cot:
            if ":" in item['output']:
                o = ':'.join(item['output'].split(":")[1:]) # try to separate the COT part and the answer list part.
            else:
                o = ""
        else:
            o = item['output']
        preds = [normalize_answer(x.strip()) for x in o.rstrip().rstrip(".").rstrip(",").split(",")]

        preds = [p for p in preds if len(p) > 0] # delete empty answers
        num_preds.append(len(preds))
        answers = [[normalize_answer(x) for x in ans] for ans in item['answers']]
        flat_answers = [item for sublist in answers for item in sublist]
        
        # Initialize the count for correct predictions
        correct_count = 0

        # Calculate recall
        for answer_set in answers:
            # A flag to indicate if any of the answers is found in the predictions
            found_flag = 0
            for answer in answer_set:
                # Check if the answer is in any of the prediction sets
                if any(answer in pred_set for pred_set in preds):
                    found_flag = 1
                    break  # Once a correct answer is found, no need to check further
            correct_count += found_flag

        # Append the recall rate to the list (number of correct sets over total sets)
        recall_rate = correct_count / len(answers)
        rec.append(recall_rate)

        # Append the top-5 recall rate to the list
        # This is the minimum between 5 and the count of correct answers,
        # divided by the minimum between 5 and the total number of answers
        rec_top5.append(min(5, correct_count) / min(len(answers), 5))

        # Reset the count for the next calculation
        correct_count = 0

        # Calculate precision
        for pred_set in preds:
            found_flag = 0
            # Check if any of the predictions is in the answer sets
            for answer_set in answers:
                if any(ans in pred_set for ans in answer_set):
                    found_flag = 1
                    break  # Once a correct prediction is found, no need to check further
            correct_count += found_flag

        # Append the precision rate to the list
        # If there are predictions, calculate the ratio, otherwise, it's 0
        precision_rate = correct_count / len(preds) if preds else 0
        prec.append(precision_rate)
        if (prec[-1] + rec[-1]) == 0:
            f1.append(0)
        else:
            f1.append(2 * prec[-1] * rec[-1] / (prec[-1] + rec[-1]))
        if (prec[-1] + rec_top5[-1]) == 0:
            f1_top5.append(0) 
        else:
            f1_top5.append(2 * prec[-1] * rec_top5[-1] / (prec[-1] + rec_top5[-1]))
        

    return {
        "num_preds": np.mean(num_preds),
        "qampari_prec": 100 * np.mean(prec),
        "qampari_rec": 100 * np.mean(rec),
        "qampari_rec_top5": 100 * np.mean(rec_top5),
        "qampari_f1": 100 * np.mean(f1),
        "qampari_f1_top5": 100 * np.mean(f1_top5),
    }
        
"""         cnt = 0
        for i in range(len(answers)):
            #print(len(answers),len(flat_answers))
            flag = 0
            for j in range(len(answers[i])):
                for k in range(len(preds)):
                    if(answers[i][j] in preds[k]):
                        flag = 1
            cnt += flag
        rec.append(cnt/len(answers))  
        rec_top5.append(min(5,cnt)/min(len(answers),5)) 

        cnt = 0
        for i in range(len(preds)):
            flag = 0
            for j in range(len(answers)):
                for k in range(len(answers[j])):
                    if(answers[j][k] in preds[i]):
                        flag = 1
            cnt += flag
        prec.append(cnt/len(preds) if len(preds)>0 else 0)  
         """
            


def main():
    torch.cuda.set_device(0)
    parser = argparse.ArgumentParser()
    parser.add_argument("--f", type=str, required=True, help="Output file. Should have field `question`, `output`, (ROUGE) `answer`, \
                        (accuracy) `qa_pairs`, (AIS) `docs`")
    parser.add_argument("--output_column", type=str, default="output", help="Column name for model output in the jsonl file")
    parser.add_argument("--no_rouge", action="store_true", help="Do not evaluate ROUGE score")
    parser.add_argument("--qa", action="store_true", help="Use the QA model")
    parser.add_argument("--mauve", action="store_true", help="Use the mauve score model")
    parser.add_argument("--citations", action="store_true", help="Evaluation with citation")
    parser.add_argument("--at_most_citations", type=int, default=3, help="At most take this many documents (mostly for precision)")
    parser.add_argument("--claims_nli", action="store_true", help="Use claims for ELI5")

    # QAMPARI
    parser.add_argument("--cot", action="store_true", help="For QAMPARI, try to find colon and separate the COT and answer listing")

    args = parser.parse_args()

    #with open(args.f) as f:
    #    data_with_config = json.load(f)
    #data = data_with_config['data'] 
    data = []
    with open(args.f,"r",encoding='utf-8') as f:
        for line in f:
            json_obj = json.loads(line)
            #json_obj['output'] = json_obj['output'].split("Answer:")[1]
            
            #json_obj['output'] = json_obj['output'].replace('\n',',')
            # 使用指定的输出列名
            if args.output_column in json_obj:
                json_obj['output'] = json_obj[args.output_column]
            else:
                logger.warning(f"Column '{args.output_column}' not found in data, using default 'output' column if available")
            data.append(json_obj)

    

    if "qampari" in args.f:
        args.no_rouge = True
        args.qa = False
        args.mauve = False
        args.decontext = False
        qampari = True
    else:
        qampari = False

    # Truncate by newline and remove on the fly search result
    logger.warning("We remove all the pre/appended space/newlines and we truncate the answer by the first newline.")
    logger.warning("We replace any on the fly search result to standard bracket citation format.")
    for i in range(len(data)):
        if data[i]['output'] is None:
            data[i]['output'] = ""
        #data[i]['output'] = data[i]['output'].strip().split("\n")[0]
        data[i]['output'] = data[i]['output'].replace("<|im_end|>", "")
        data[i]['output'] = data[i]['output'].replace("\u0000", "")

    # Remove all citations for all non-AutoAIS evaluation
    normalized_data = copy.deepcopy(data)
    for i in range(len(normalized_data)):
        normalized_data[i]['output'] = remove_citations(normalized_data[i]['output'])
    #    normalized_wordlist = normalized_data[i]['output'].split()
    #    normalized_data[i]['output'] = ''
    #    for j in range(min(len(normalized_wordlist),100)):
    #        normalized_data[i]['output'] += normalized_wordlist[j] + " "  

    result = {}
    
    result['length'] = compute_len(normalized_data)
    result['str_em'], result['str_hit'] = compute_str_em(normalized_data)
    if qampari:
        result.update(compute_qampari_f1(normalized_data, cot=args.cot))
    if not args.no_rouge:
        result['rougeLsum'] = compute_rouge(normalized_data)
    if args.qa:
        result.update(compute_qa(normalized_data))
    if args.mauve:
        result['mauve'] = compute_mauve(normalized_data)
    if args.citations: 
        result.update(compute_autoais(data, qampari=qampari, at_most_citations=args.at_most_citations))
    if args.claims_nli:
        result["claims_nli"] = compute_claims(normalized_data)

    print(result)
    with open(args.f + ".score", "w") as f:
        json.dump(result, f, indent=4)


if __name__ == "__main__":
    main()

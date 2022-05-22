"""
Script to convert the retrieved HITS into an entailment dataset
USAGE:
 python convert_csqa.py input_file output_file

JSONL format of files
 1. input_file:
 {
   "id": "d3b479933e716fb388dfb297e881054c",
   "question": {
      "stem": "If a lantern is not for sale, where is it likely to be?"
      "choices": [{"label": "A", "text": "antique shop"}, {"label": "B", "text": "house"}, {"label": "C", "text": "dark place"}]
    },
     "answerKey":"B"
  }

 2. output_file:
   {
   "id": "d3b479933e716fb388dfb297e881054c",
   "question": {
      "stem": "If a lantern is not for sale, where is it likely to be?"
      "choices": [{"label": "A", "text": "antique shop"}, {"label": "B", "text": "house"}, {"label": "C", "text": "dark place"}]
    },
    "answerKey":"B",

    "statements":[
        {label:true, stem: "If a lantern is not for sale, it likely to be at house"},
        {label:false, stem: "If a lantern is not for sale, it likely to be at antique shop"},
        {label:false, stem: "If a lantern is not for sale, it likely to be at dark place"}
        ]
  }
"""

import json
import os
import pickle
import random
import re
import sys
from multiprocessing import Pool

from tqdm import tqdm
import spacy

nlp = spacy.load("en_core_web_md")
nlp.max_length = 1500000

__all__ = ['convert_to_entailment']

# String used to indicate a blank
BLANK_STR = "___"

def mp_converter(instance):
    output_dict = convert_qajson_to_entailment(instance)
    return output_dict


def convert_to_entailment(dataset_dir: str, output_file: str, SEED, K, is_distributed=False, is_test=False):
    print(f'Processing is done on server {os.environ["SERVER_NAME"]}....')
    print(f'converting {dataset_dir} to entailment dataset...')
    nrow = sum(1 for _ in open(dataset_dir, 'r'))
    mp_instances = []
    with open(dataset_dir, 'r') as ds_handle:
        for line in tqdm(ds_handle, total=nrow):
            json_line = json.loads(line)
            mp_instances.append(json_line)
            # print('here')


    if SEED!=-1 and K!=-1:
        # Choose K instances from mp_instance
        # random.seed(SEED)
        # random.shuffle(mp_instances)
        # if is_test:
        #     mp_instances = mp_instances[:int(len(mp_instances) * .2)]
        # else:
        #     mp_instances = mp_instances[:K]

        if 'train' in dataset_dir:
            SET = 'train'
        elif 'val' in dataset_dir:
            SET = 'val'

        sampled_ids = []
        with open(f'/disk1/sajad/datasets/news/xsum/json_ids/{SET}.seed{SEED}.k{K}.txt') as fR:
            for l in fR:
                sampled_ids.append(l.strip())

        new_instances = []
        for m in mp_instances:
            if m['id'] in sampled_ids:
                new_instances.append(m)

        mp_instances = new_instances


    if is_distributed:
        random.seed(88)
        random.shuffle(mp_instances)

        # 3 servers; here is the standing... (train)
        # 0: chianti
        # 1: brunello
        # 2: barolo
        # mp_instances = mp_instances[len(mp_instances)//2:]
        # mp_instances = mp_instances[:5]
        if os.environ['SERVER_NAME'] == 'chianti':
            index = 0
        elif os.environ['SERVER_NAME'] == 'brunello':
            index = 1
        elif os.environ['SERVER_NAME'] == 'barolo':
            index = 2
        # elif os.environ['SERVER_NAME'] == 'barbaresco':
        #     index = 2
        # elif os.environ['SERVER_NAME'] == 'barolo':
        #     index = 2
        # else:
        #     index = 2

        index_weights = {
            -1: 0,

            0: 0.25,
            1: 0.40,
            2: 0.35,
        }
        boundaries = {
            -1: 0,

            0: 0,
            1: 0,
            2: 0,
        }

        index_boundaries = {}
        all_len = len(mp_instances)
        for jidx, (_index, weight) in enumerate(index_weights.items()):
            if _index!= -1 and _index != 1000 and _index == index:
                # import pdb;pdb.set_trace()
                lb = sum([int(all_len*index_weights[_i]) for _i in list(index_weights.keys())[:jidx]])
                # lb = boundaries[jidx-1] if jidx != 0 else 0

                ub = lb + int(all_len * weight) if _index != list(index_weights.keys())[-1] else all_len
                # boundaries[jidx] = ub
                mp_instances = mp_instances[lb:ub]
                break

    # import pdb;pdb.set_trace()
    print('Writing instances to entailment...')
    pool = Pool(20)
    try:
        with open(output_file, 'w') as output_handle:
            for output_dict in tqdm(pool.imap_unordered(mp_converter, mp_instances), total=len(mp_instances)):
                output_handle.write(json.dumps(output_dict))
                output_handle.write("\n")
    except Exception as e:
        print('error')
        print(e)
        pass

    print(f'converted statements saved to {output_file}')
    print()


def get_sentences(text):
    doc = nlp(text)
    out_sents = []
    for sent in doc.sents:
        if len(sent) > 1:
            out_sents.append(sent.text)
    return out_sents


# Convert the QA file json to output dictionary containing premise and hypothesis
def convert_qajson_to_entailment(qa_json: dict):
    source_text = qa_json["document"]
    summary_text = qa_json["summary"]

    # now split based on sentences
    # TODO
    sentences = get_sentences(source_text)
    return {
        'statements': [s.replace('\n', '') for s in sentences],
        'highlights': summary_text,
        'id': qa_json['id']
    }

# Get a Fill-In-The-Blank (FITB) statement from the question text. E.g. "George wants to warm his
# hands quickly by rubbing them. Which skin surface will produce the most heat?" ->
# "George wants to warm his hands quickly by rubbing them. ___ skin surface will produce the most
# heat?
def get_fitb_from_question(question_text: str) -> str:
    fitb = replace_wh_word_with_blank(question_text)
    if not re.match(".*_+.*", fitb):
        # print("Can't create hypothesis from: '{}'. Appending {} !".format(question_text, BLANK_STR))
        # Strip space, period and question mark at the end of the question and add a blank
        fitb = re.sub(r"[\.\? ]*$", "", question_text.strip()) + " " + BLANK_STR
    return fitb


# Create a hypothesis statement from the the input fill-in-the-blank statement and answer choice.
def create_hypothesis(fitb: str, choice: str, ans_pos: bool) -> str:

    if ". " + BLANK_STR in fitb or fitb.startswith(BLANK_STR):
        choice = choice[0].upper() + choice[1:]
    else:
        choice = choice.lower()
    # Remove period from the answer choice, if the question doesn't end with the blank
    if not fitb.endswith(BLANK_STR):
        choice = choice.rstrip(".")
    # Some questions already have blanks indicated with 2+ underscores
    if not ans_pos:
        try:
            hypothesis = re.sub("__+", choice, fitb)
        except:
            print (choice, fitb)
        return hypothesis
    choice = choice.strip()
    m = re.search("__+", fitb)
    start = m.start()

    length = (len(choice) - 1) if fitb.endswith(BLANK_STR) and choice[-1] in ['.', '?', '!'] else len(choice)
    hypothesis = re.sub("__+", choice, fitb)

    return hypothesis, (start, start + length)


# Identify the wh-word in the question and replace with a blank
def replace_wh_word_with_blank(question_str: str):
    # if "What is the name of the government building that houses the U.S. Congress?" in question_str:
    #     print()
    question_str = question_str.replace("What's", "What is")
    question_str = question_str.replace("whats", "what")
    question_str = question_str.replace("U.S.", "US")
    wh_word_offset_matches = []
    wh_words = ["which", "what", "where", "when", "how", "who", "why"]
    for wh in wh_words:
        # Some Turk-authored SciQ questions end with wh-word
        # E.g. The passing of traits from parents to offspring is done through what?

        if wh == "who" and "people who" in question_str:
            continue

        m = re.search(wh + r"\?[^\.]*[\. ]*$", question_str.lower())
        if m:
            wh_word_offset_matches = [(wh, m.start())]
            break
        else:
            # Otherwise, find the wh-word in the last sentence
            m = re.search(wh + r"[ ,][^\.]*[\. ]*$", question_str.lower())
            if m:
                wh_word_offset_matches.append((wh, m.start()))
            # else:
            #     wh_word_offset_matches.append((wh, question_str.index(wh)))

    # If a wh-word is found
    if len(wh_word_offset_matches):
        # Pick the first wh-word as the word to be replaced with BLANK
        # E.g. Which is most likely needed when describing the change in position of an object?
        wh_word_offset_matches.sort(key=lambda x: x[1])
        wh_word_found = wh_word_offset_matches[0][0]
        wh_word_start_offset = wh_word_offset_matches[0][1]
        # Replace the last question mark with period.
        question_str = re.sub(r"\?$", ".", question_str.strip())
        # Introduce the blank in place of the wh-word
        fitb_question = (question_str[:wh_word_start_offset] + BLANK_STR +
                         question_str[wh_word_start_offset + len(wh_word_found):])
        # Drop "of the following" as it doesn't make sense in the absence of a multiple-choice
        # question. E.g. "Which of the following force ..." -> "___ force ..."
        final = fitb_question.replace(BLANK_STR + " of the following", BLANK_STR)
        final = final.replace(BLANK_STR + " of these", BLANK_STR)
        return final

    elif " them called?" in question_str:
        return question_str.replace(" them called?", " " + BLANK_STR + ".")
    elif " meaning he was not?" in question_str:
        return question_str.replace(" meaning he was not?", " he was not " + BLANK_STR + ".")
    elif " one of these?" in question_str:
        return question_str.replace(" one of these?", " " + BLANK_STR + ".")
    elif re.match(r".*[^\.\?] *$", question_str):
        # If no wh-word is found and the question ends without a period/question, introduce a
        # blank at the end. e.g. The gravitational force exerted by an object depends on its
        return question_str + " " + BLANK_STR
    else:
        # If all else fails, assume "this ?" indicates the blank. Used in Turk-authored questions
        # e.g. Virtually every task performed by living organisms requires this?
        return re.sub(r" this[ \?]", " ___ ", question_str)


# Create the output json dictionary from the input json, premise and hypothesis statement
def create_output_dict(input_json: dict, statement: str, label: bool, ans_pos: bool, pos=None) -> dict:
    if "statements" not in input_json:
        input_json["statements"] = []
    if not ans_pos:
        input_json["statements"].append({"label": label, "statement": statement})
    else:
        input_json["statements"].append({"label": label, "statement": statement, "ans_pos": pos})
    return input_json


if __name__ == "__main__":
    if len(sys.argv) < 3:
        raise ValueError("Provide at least two arguments: "
                         "json file with hits, output file name")
    convert_to_entailment(sys.argv[1], sys.argv[2])

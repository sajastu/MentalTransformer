import json
from multiprocessing import Pool

import pandas as pd
from rouge_score import rouge_scorer

from tqdm import tqdm
scorer = rouge_scorer.RougeScorer(['rouge1', 'rouge2', 'rougeL'], use_stemmer=True)

def return_with_rg(param):
    src, tldr, bart_generated, model_generated, model2_generated, brg1, brg2, brgl, mrg1, mrg2, mrgl, m2rg1, m2rg2, m2rgl = param.items()

    rg_scores = {
        'bart_rouge1': 0,
        'bart_rouge2': 0,
        'bart_rougeL': 0,
        'model_rouge1': 0,
        'model_rouge2': 0,
        'model_rougeL': 0,
        'model2_rouge1': 0,
        'model2_rouge2': 0,
        'model2_rougeL': 0,
    }

    for sys in ['bart', 'model', 'model2']:

        scores = scorer.score(tldr[1], eval(f'{sys}_generated')[1])

        for k, metric in enumerate(['rouge1', 'rouge2', 'rougeL']):
            rg_scores[f'{sys}_{metric}'] = scores[metric].fmeasure

    return param, rg_scores

def main():
    instances = {
        'src': [],
        'tldr': [],
        'bart-generated': [],
        'model-generated': [],
        'model2-generated': [],
        'bart_rouge1': [],
        'bart_rouge2': [],
        'bart_rougeL': [],
        'model_rouge1': [],
        'model_rouge2': [],
        'model_rougeL': [],
        'model2_rouge1': [],
        'model2_rouge2': [],
        'model2_rougeL': [],

    }

    instances_final = {
        'src': [],
        'tldr': [],
        'bart-generated': [],
        'model-generated': [],
        'model2-generated': [],
        'bart_rouge1': [],
        'bart_rouge2': [],
        'bart_rougeL': [],
        'model_rouge1': [],
        'model_rouge2': [],
        'model_rougeL': [],
        'model2_rouge1': [],
        'model2_rouge2': [],
        'model2_rougeL': [],
    }

    with \
            open('/disk1/sajad/datasets/medical/mental-reddit-reduced/sets/test.json') as fR1, \
            open('/disk1/sajad/saved_models/bart-large-mentsum/generated_predictions.txt') as fR2, \
            open('/disk1/sajad/saved_models/MentBart-mentsum-30kLarge/generated_predictions.txt') as fR3, \
            open('/disk1/sajad/saved_models/MentBart-mentsum/generated_predictions.txt') as fR4:
        for l1, l2, l3, l4 in zip(fR1, fR2, fR3, fR4):
            instances['src'].append( json.loads(l1.strip())['src'])
            instances['tldr'].append(json.loads(l1.strip())['tldr'])
            instances['bart-generated'].append(l2.strip())
            instances['model-generated'].append(l3.strip())
            instances['model2-generated'].append(l4.strip())
            instances['bart_rouge1'].append(0)
            instances['bart_rouge2'].append(0)
            instances['bart_rougeL'].append(0)
            instances['model_rouge1'].append(0)
            instances['model_rouge2'].append(0)
            instances['model_rougeL'].append(0)
            instances['model2_rouge1'].append(0)
            instances['model2_rouge2'].append(0)
            instances['model2_rougeL'].append(0)


    pool = Pool(15)
    instance_to_multi_process = []

    for j, ent in enumerate(instances['src']):

        instance_to_multi_process.append(
            {
                'src': ent,
            }
        )
        for other_key in [k for k in list(instances.keys()) if k!='src']:
            instance_to_multi_process[-1][other_key] = instances[other_key][j]


    # for example in instance_to_multi_process:
    #     return_with_rg(example)
    for ent, rg_scores in tqdm(pool.imap_unordered(return_with_rg, instance_to_multi_process)):

        for key, val in rg_scores.items():
            ent[key] = rg_scores[key]

        for k, v in ent.items():
            instances_final[k].append(v)


    df = pd.DataFrame(instances_final)
    df.to_csv('BART_vs_model_mentalWords.csv', index=False)


if __name__ == '__main__':
    main()
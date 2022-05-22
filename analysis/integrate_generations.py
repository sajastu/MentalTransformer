import json
from multiprocessing import Pool

import pandas as pd
from rouge_score import rouge_scorer
scorer = rouge_scorer.RougeScorer(['rouge1', 'rouge2', 'rougeL'], use_stemmer=True)

from tqdm import tqdm

def return_with_rg(param):
    id, src, tldr, model_generated, bart_generated ,mrg1, mrg2, mrgl, br1, br2, brl = param.items()

    rg_scores = {
        'model_rouge1': 0,
        'model_rouge2': 0,
        'model_rougeL': 0,
        'bart_rouge1': 0,
        'bart_rouge2': 0,
        'bart_rougeL': 0,
    }

    for sys in ['bart', 'model']:

        scores = scorer.score(tldr[1], eval(f'{sys}_generated')[1])

        for k, metric in enumerate(['rouge1', 'rouge2', 'rougeL']):
            rg_scores[f'{sys}_{metric}'] = scores[metric].fmeasure

    return param, rg_scores

def main():
    instances = {
        'id': [],
        'src': [],
        'summary': [],
        'model-generated': [],
        'bart-generated': [],
        'model_rouge1': [],
        'model_rouge2': [],
        'model_rougeL': [],
        'bart_rouge1': [],
        'bart_rouge2': [],
        'bart_rougeL': [],

    }

    instances_final = {
        'id': [],
        'src': [],
        'summary': [],
        'model-generated': [],
        'bart-generated': [],
        'model_rouge1': [],
        'model_rouge2': [],
        'model_rougeL': [],
        'bart_rouge1': [],
        'bart_rouge2': [],
        'bart_rougeL': [],
    }

    import pandas as pd
    df_ground = pd.read_parquet('/disk0/sajad/datasets/news/cnn-dm/paraq-files//test.parquet', engine='pyarrow')
    df_g_list = []
    print(f'len: {len(df_ground)}')
    for indx, row in df_ground.iterrows():
        df_g_list.append(row['id'])
    df_gg_list = []
    test_instances = {}
    with open('/disk1/sajad/datasets/news/cnn-dm/test.json') as fR:
        for l in fR:
            ent = json.loads(l)
            test_instances[ent['id']] = ent
                # import pdb;pdb.set_trace()

    for idd in df_g_list:
        df_gg_list.append(test_instances[idd])

    with open('/disk0/sajad/sci-trained-models/bart/large-cnn/generated_predictions_bart.txt') as fR1,\
            open('/disk0/sajad/sci-trained-models/grease/large-cnn/generated_predictions.txt') as fR2:
        for l1, l2,l3 in zip(df_gg_list,fR2, fR1):
            instances['id'].append(l1['id'].strip())
            instances['src'].append(l1['article'].strip())
            instances['summary'].append(l1['highlights'].strip())
            instances['model-generated'].append(l2.strip().replace('--n-- ', '\n'))
            instances['bart-generated'].append(l3.strip().replace('--n-- ', '\n'))
            instances['model_rouge1'].append(0)
            instances['model_rouge2'].append(0)
            instances['model_rougeL'].append(0)
            instances['bart_rouge1'].append(0)
            instances['bart_rouge2'].append(0)
            instances['bart_rougeL'].append(0)

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
    df.to_csv('model_cnn_grease_vs_bart.csv', index=False)


if __name__ == '__main__':
    main()
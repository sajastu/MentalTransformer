import argparse
import glob
import json
import os.path
import pickle
from multiprocessing import Pool

import numpy as np
import pandas as pd
import pyarrow.parquet as pq
from tqdm import tqdm
import pyarrow as pa

DS_NAME = 'xsum'
DS_DIR = f'/disk1/sajad/datasets/news/{DS_NAME}/'
KEYS = ('document', 'summary', 'id')

if not os.path.exists(f'{DS_DIR}/parq-files/'):
    os.makedirs(f'{DS_DIR}/parq-files/')

def _mp_connector(param):
    ent, graph = param
    paraq_dict_single = {}
    paraq_dict_single[KEYS[0]] = ent[KEYS[0]]
    paraq_dict_single[KEYS[1]] = ent[KEYS[1]]
    paraq_dict_single['subgraph'] = {'adj': graph['adj'].todense(), 'concepts': graph['sentence_mask']}
    return paraq_dict_single


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--seed', type=int, default=-1, help='enable debug mode')
    parser.add_argument('--k', type=int, default=-1, help='enable debug mode')
    parser.add_argument('--only_test', type=bool, default=False, help='enable debug mode')

    args = parser.parse_args()

    SEED = args.seed
    K = args.k

    if not args.only_test:
        for se in ['train', 'val']:
            instances = {}
            with open(f'{DS_DIR}/{se}.json') as fR:
                for l in fR:
                    instances[json.loads(l)['id']] = json.loads(l)
            graph_data = []
            for j, file in enumerate(tqdm(glob.glob(f'{DS_DIR}/graph/{se}.*.seed{SEED}.k{K}.pkl'), total=len(glob.glob(f'{DS_DIR}/graph/{se}.*.seed{SEED}.k{K}.pkl')), desc=f'{se}')):
                with open(file, mode='rb') as fR:
                    graph_data_split = pickle.load(fR)
                    graph_data.extend(graph_data_split)

            dicts = {
                KEYS[2]: [],
                KEYS[0]: [],
                KEYS[1]: [],
                'adj-row': [],
                'adj-col': [],
                'adj-data': [],
                'concepts': [],
                'sentence_mask': [],
                'shape-0': [],
                'shape-1': [],
            }
            print(f'len of graph_data {len(graph_data)}')
            for graph in tqdm(graph_data, total=len(graph_data)):
                paraq_dict = {}
                ent = instances[graph['id']]
                # mp_instances.append((ent, graph))

                paraq_dict[KEYS[0]] = ent[KEYS[0]]
                paraq_dict[KEYS[1]] = ent[KEYS[1]]
                paraq_dict[KEYS[2]] = ent[KEYS[2]]
                paraq_dict.update(
                    {'adj-row': graph['adj'].row,
                      'adj-col': graph['adj'].col,
                      'adj-data': graph['adj'].data,
                      'shape-0': graph['adj'].shape[0],
                      'shape-1': graph['adj'].shape[1],
                      'concepts': graph['concepts'],
                      'sentence_mask': graph['sentence_mask']
                    }
               )

                for k, v in paraq_dict.items():
                    dicts[k].append(v)

            del graph_data
            print('Transfer dict...')

            for k, v in dicts.items():
                dicts[k] = np.asarray(v, dtype=object)
            print(f'Creating dataframe of size {len(dicts[KEYS[0]])}...')
            df = pd.DataFrame(dicts)
            print('save to parquet...')
            df.to_parquet(f'{DS_DIR}/parq-files/{se}.seed{SEED}.k{K}.parquet')
            dicts.clear()
    else:
        for se in ['test']:
            instances = {}
            with open(f'{DS_DIR}/{se}.json') as fR:
                for l in fR:
                    instances[json.loads(l)['id']] = json.loads(l)
            graph_data = []
            for j, file in enumerate(tqdm(glob.glob(f'{DS_DIR}/graph/{se}.*.seed{SEED}.k{K}.pkl'),
                                          total=len(glob.glob(f'{DS_DIR}/graph/{se}.*.seed{SEED}.k{K}.pkl')),
                                          desc=f'{se}')):
                with open(file, mode='rb') as fR:
                    graph_data_split = pickle.load(fR)
                    graph_data.extend(graph_data_split)

            dicts = {
                KEYS[2]: [],
                KEYS[0]: [],
                KEYS[1]: [],
                'adj-row': [],
                'adj-col': [],
                'adj-data': [],
                'concepts': [],
                'sentence_mask': [],
                'shape-0': [],
                'shape-1': [],
            }
            print(f'len of graph_data {len(graph_data)}')
            for graph in tqdm(graph_data, total=len(graph_data)):
                paraq_dict = {}
                ent = instances[graph['id']]
                # mp_instances.append((ent, graph))

                paraq_dict[KEYS[0]] = ent[KEYS[0]]
                paraq_dict[KEYS[1]] = ent[KEYS[1]]
                paraq_dict[KEYS[2]] = ent[KEYS[2]]
                paraq_dict.update(
                    {'adj-row': graph['adj'].row,
                     'adj-col': graph['adj'].col,
                     'adj-data': graph['adj'].data,
                     'shape-0': graph['adj'].shape[0],
                     'shape-1': graph['adj'].shape[1],
                     'concepts': graph['concepts'],
                     'sentence_mask': graph['sentence_mask']
                     }
                )

                for k, v in paraq_dict.items():
                    dicts[k].append(v)

            del graph_data
            print('Transfer dict...')

            for k, v in dicts.items():
                dicts[k] = np.asarray(v, dtype=object)
            print(f'Creating dataframe of size {len(dicts[KEYS[0]])}...')
            df = pd.DataFrame(dicts)
            print('save to parquet...')
            df.to_parquet(f'{DS_DIR}/parq-files/{se}.seed{SEED}.k{K}.parquet')
            dicts.clear()


if __name__ == '__main__':
    main()
# import glob
# import json
# import os
#
# split_ids = {
#     'train-gao': [],
#     'train-crs': [],
#     'val-gao': [],
#     'val-crs': [],
#     'test-gao': [],
#     'test-crs': []
# }
#
# BASE_DIR='/home/sajad/packages/summarization/transformers/gov-report/'
#
# for id_file in glob.glob(BASE_DIR + '/split_ids/*.ids'):
#     # import pdb;pdb.set_trace()
#     if 'train' in id_file:
#         with open(id_file) as fr:
#             for l in fr:
#                 split_ids[f'train-{id_file.split("/")[-1].split("_")[0]}'.replace('.ids', '')].append(l.strip())
#     elif 'val' in id_file:
#         with open(id_file) as fr:
#             for l in fr:
#                 split_ids[f'val-{id_file.split("/")[-1].split("_")[0]}'.replace('.ids', '')].append(l.strip())
#     elif 'test' in id_file:
#         with open(id_file) as fr:
#             for l in fr:
#                 try:
#                     split_ids[f'test-{id_file.split("/")[-1].split("_")[0]}'.replace('.ids', '')].append(l.strip())
#                 except:
#                     import pdb;pdb.set_trace()
# wr_dir = '/disk1/sajad/gov-reports/'
# # wr_dir = 'gov-reports/'
#
# try:
#     os.makedirs(wr_dir)
# except:
#     print('exists!')
#
# split_ents = {
#     'train': [],
#     'val': [],
#     'test': []
# }
#
#
# for key, file_ids in split_ids.items():
#
#     split = key.split('-')[1]
#     se = key.split('-')[0]
#
#     for f_id in file_ids:
#         with open(BASE_DIR + f'/{split}/{f_id}.json') as fR:
#             ent = json.load(fR)
#             split_ents[se].append(ent)
#
# for key, ents in split_ents.items():
#     with open(wr_dir + f'{key}.jsonl', mode='w') as fW:
#         for ent in ents:
#             json.dump(ent, fW)
#             fW.write('\n')


####################################################################################################
####################################################################################################
####################################################################################################
import json

split_ents = {
    'train': [],
    'val': [],
    'test': [],
}

BASE_DIR='gov-report'
wr_dir = '/disk1/sajad/gov-reports/'

for key, files in split_ents.items():
    with open(f'{BASE_DIR}/{key}.source') as fS, open(f'{BASE_DIR}/{key}.target') as fT:
        for ls, lt in zip(fS, fT):
            ent= {}
            ent['source'] = ls.strip()
            ent['summary'] = lt.strip()
            split_ents[key].append(ent)

for key, ents in split_ents.items():
    with open(wr_dir + f'{key}.json', mode='w') as fW:
        for ent in ents:
            json.dump(ent, fW)
            fW.write('\n')


import torch
import networkx as nx
import itertools
import json
from tqdm import tqdm
from .conceptnet import merged_relations
import pickle
from scipy.sparse import coo_matrix
from multiprocessing import Pool
from collections import OrderedDict


from .maths import *

__all__ = ['generate_graph']


import multiprocessing
# We must import this explicitly, it is not imported by the top-level
# multiprocessing module.
import multiprocessing.pool
import time

from random import randint


class NoDaemonProcess(multiprocessing.Process):
    # make 'daemon' attribute always return False
    def _get_daemon(self):
        return False
    def _set_daemon(self, value):
        pass
    daemon = property(_get_daemon, _set_daemon)

# We sub-class multiprocessing.pool.Pool instead of multiprocessing.Pool
# because the latter is only a wrapper function, not a proper class.
class MyPool(multiprocessing.pool.Pool):
    Process = NoDaemonProcess

def sleepawhile(t):
    print("Sleeping %i seconds..." % t)
    time.sleep(t)
    return t

def work(num_procs):
    print("Creating %i (daemon) workers and jobs in child." % num_procs)
    pool = multiprocessing.Pool(num_procs)

    result = pool.map(sleepawhile,
        [randint(1, 5) for x in range(num_procs)])

    # The following is not really needed, since the (daemon) workers of the
    # child's pool are killed when the child is terminated, but it's good
    # practice to cleanup after ourselves anyway.
    pool.close()
    pool.join()
    return result


concept2id = None
id2concept = None
relation2id = None
id2relation = None

cpnet = None
cpnet_all = None
cpnet_simple = None


def load_resources(cpnet_vocab_path):
    global concept2id, id2concept, relation2id, id2relation

    with open(cpnet_vocab_path, "r", encoding="utf8") as fin:
        id2concept = [w.strip() for w in fin]
    concept2id = {w: i for i, w in enumerate(id2concept)}

    id2relation = merged_relations
    relation2id = {r: i for i, r in enumerate(id2relation)}


def load_cpnet(cpnet_graph_path):
    global cpnet, cpnet_simple
    cpnet = nx.read_gpickle(cpnet_graph_path)
    cpnet_simple = nx.Graph()
    for u, v, data in cpnet.edges(data=True):
        w = data['weight'] if 'weight' in data else 1.0
        if cpnet_simple.has_edge(u, v):
            cpnet_simple[u][v]['weight'] += w
        else:
            cpnet_simple.add_edge(u, v, weight=w)


def relational_graph_generation(qcs, acs, paths, rels):
    raise NotImplementedError()  # TODO


# plain graph generation
def plain_graph_generation(qcs, acs, paths, rels):
    global cpnet, concept2id, relation2id, id2relation, id2concept, cpnet_simple

    graph = nx.Graph()
    for p in paths:
        for c_index in range(len(p) - 1):
            h = p[c_index]
            t = p[c_index + 1]
            # TODO: the weight can computed by concept embeddings and relation embeddings of TransE
            graph.add_edge(h, t, weight=1.0)

    for qc1, qc2 in list(itertools.combinations(qcs, 2)):
        if cpnet_simple.has_edge(qc1, qc2):
            graph.add_edge(qc1, qc2, weight=1.0)

    for ac1, ac2 in list(itertools.combinations(acs, 2)):
        if cpnet_simple.has_edge(ac1, ac2):
            graph.add_edge(ac1, ac2, weight=1.0)

    if len(qcs) == 0:
        qcs.append(-1)

    if len(acs) == 0:
        acs.append(-1)

    if len(paths) == 0:
        for qc in qcs:
            for ac in acs:
                graph.add_edge(qc, ac, rel=-1, weight=0.1)

    g = nx.convert_node_labels_to_integers(graph, label_attribute='cid')  # re-index
    return nx.node_link_data(g)


def generate_adj_matrix_per_inst(nxg_str):
    global id2relation
    n_rel = len(id2relation)

    nxg = nx.node_link_graph(json.loads(nxg_str))
    n_node = len(nxg.nodes)
    cids = np.zeros(n_node, dtype=np.int32)
    for node_id, node_attr in nxg.nodes(data=True):
        cids[node_id] = node_attr['cid']

    adj = np.zeros((n_rel, n_node, n_node), dtype=np.uint8)
    for s in range(n_node):
        for t in range(n_node):
            s_c, t_c = cids[s], cids[t]
            if cpnet_all.has_edge(s_c, t_c):
                for e_attr in cpnet_all[s_c][t_c].values():
                    if e_attr['rel'] >= 0 and e_attr['rel'] < n_rel:
                        adj[e_attr['rel']][s][t] = 1
    cids += 1
    adj = coo_matrix(adj.reshape(-1, n_node))
    return (adj, cids)


# def _mp_check(param):
#     s, s_c, t, t_c, n_rel = param
#     if cpnet.has_edge(s_c, t_c):
#         for e_attr in cpnet[s_c][t_c].values():
#             if e_attr['rel'] >= 0 and e_attr['rel'] < n_rel:
#                 return (e_attr['rel'], s, t)

def concepts2adj(node_ids):


    global id2relation
    n_rel = len(id2relation)
    cids = np.array(node_ids, dtype=np.int32)
    n_node = cids.shape[0]
    adj = np.zeros((n_rel, n_node, n_node), dtype=np.uint8)


    # mp_nested = []
    for s in range(n_node):
        for t in range(n_node):
            s_c, t_c = cids[s], cids[t]
            # mp_nested.append((s, s_c, t, t_c, n_rel))

            if cpnet.has_edge(s_c, t_c):
                for e_attr in cpnet[s_c][t_c].values():
                    if e_attr['rel'] >= 0 and e_attr['rel'] < n_rel:
                        adj[e_attr['rel']][s][t] = 1

    # mypool = MyPool(5)
    # for out in tqdm(mypool.imap_unordered(_mp_check, mp_nested), total=len(mp_nested)):
    #     if out is not None:
    #         e_attr, s, t = out
    #         adj[e_attr][s][t] = 1

    # cids += 1  # note!!! index 0 is reserved for padding
    adj_main = adj
    adj = coo_matrix(adj.reshape(-1, n_node))
    return adj, cids, adj_main


def concepts_to_adj_matrices_1hop_neighbours(data):
    qc_ids, ac_ids = data
    qa_nodes = set(qc_ids) | set(ac_ids)
    extra_nodes = set()
    for u in set(qc_ids) | set(ac_ids):
        if u in cpnet.nodes:
            extra_nodes |= set(cpnet[u])
    extra_nodes = extra_nodes - qa_nodes
    schema_graph = sorted(qc_ids) + sorted(ac_ids) + sorted(extra_nodes)
    arange = np.arange(len(schema_graph))
    qmask = arange < len(qc_ids)
    amask = (arange >= len(qc_ids)) & (arange < (len(qc_ids) + len(ac_ids)))
    adj, concepts = concepts2adj(schema_graph)
    return adj, concepts, qmask, amask


def concepts_to_adj_matrices_1hop_neighbours_without_relatedto(data):
    qc_ids, ac_ids = data
    qa_nodes = set(qc_ids) | set(ac_ids)
    extra_nodes = set()
    for u in set(qc_ids) | set(ac_ids):
        if u in cpnet.nodes:
            for v in cpnet[u]:
                for data in cpnet[u][v].values():
                    if data['rel'] not in (15, 32):
                        extra_nodes.add(v)
    extra_nodes = extra_nodes - qa_nodes
    schema_graph = sorted(qc_ids) + sorted(ac_ids) + sorted(extra_nodes)
    arange = np.arange(len(schema_graph))
    qmask = arange < len(qc_ids)
    amask = (arange >= len(qc_ids)) & (arange < (len(qc_ids) + len(ac_ids)))
    adj, concepts = concepts2adj(schema_graph)
    return adj, concepts, qmask, amask


def concepts_to_adj_matrices_2hop_qa_pair(data):
    qc_ids, ac_ids = data
    qa_nodes = set(qc_ids) | set(ac_ids)
    extra_nodes = set()
    for qid in qc_ids:
        for aid in ac_ids:
            if qid != aid and qid in cpnet_simple.nodes and aid in cpnet_simple.nodes:
                extra_nodes |= set(cpnet_simple[qid]) & set(cpnet_simple[aid])
    extra_nodes = extra_nodes - qa_nodes
    schema_graph = sorted(qc_ids) + sorted(ac_ids) + sorted(extra_nodes)
    arange = np.arange(len(schema_graph))
    qmask = arange < len(qc_ids)
    amask = (arange >= len(qc_ids)) & (arange < (len(qc_ids) + len(ac_ids)))
    adj, concepts = concepts2adj(schema_graph)
    return adj, concepts, qmask, amask


def concepts_to_adj_matrices_2hop_all_pair(data):
    qc_ids, ac_ids = data
    qa_nodes = set(qc_ids) | set(ac_ids)
    extra_nodes = set()
    for qid in qa_nodes:
        for aid in qa_nodes:
            if qid != aid and qid in cpnet_simple.nodes and aid in cpnet_simple.nodes:
                extra_nodes |= set(cpnet_simple[qid]) & set(cpnet_simple[aid])
    extra_nodes = extra_nodes - qa_nodes
    schema_graph = sorted(qc_ids) + sorted(ac_ids) + sorted(extra_nodes)
    arange = np.arange(len(schema_graph))
    qmask = arange < len(qc_ids)
    amask = (arange >= len(qc_ids)) & (arange < (len(qc_ids) + len(ac_ids)))
    adj, concepts = concepts2adj(schema_graph)
    return adj, concepts, qmask, amask


def concepts_to_adj_matrices_2step_relax_all_pair(data):
    qc_ids, ac_ids = data
    qa_nodes = set(qc_ids) | set(ac_ids)
    extra_nodes = set()
    for qid in qc_ids:
        for aid in ac_ids:
            if qid != aid and qid in cpnet_simple.nodes and aid in cpnet_simple.nodes:
                extra_nodes |= set(cpnet_simple[qid]) & set(cpnet_simple[aid])
    intermediate_ids = extra_nodes - qa_nodes
    for qid in intermediate_ids:
        for aid in ac_ids:
            if qid != aid and qid in cpnet_simple.nodes and aid in cpnet_simple.nodes:
                extra_nodes |= set(cpnet_simple[qid]) & set(cpnet_simple[aid])
    for qid in qc_ids:
        for aid in intermediate_ids:
            if qid != aid and qid in cpnet_simple.nodes and aid in cpnet_simple.nodes:
                extra_nodes |= set(cpnet_simple[qid]) & set(cpnet_simple[aid])
    extra_nodes = extra_nodes - qa_nodes
    schema_graph = sorted(qc_ids) + sorted(ac_ids) + sorted(extra_nodes)
    arange = np.arange(len(schema_graph))
    qmask = arange < len(qc_ids)
    amask = (arange >= len(qc_ids)) & (arange < (len(qc_ids) + len(ac_ids)))
    adj, concepts = concepts2adj(schema_graph)
    return adj, concepts, qmask, amask


def concepts_to_adj_matrices_3hop_qa_pair(data):
    qc_ids, ac_ids = data
    qa_nodes = set(qc_ids) | set(ac_ids)
    extra_nodes = set()
    for qid in qc_ids:
        for aid in ac_ids:
            if qid != aid and qid in cpnet_simple.nodes and aid in cpnet_simple.nodes:
                for u in cpnet_simple[qid]:
                    for v in cpnet_simple[aid]:
                        if cpnet_simple.has_edge(u, v):  # ac is a 3-hop neighbour of qc
                            extra_nodes.add(u)
                            extra_nodes.add(v)
                        if u == v:  # ac is a 2-hop neighbour of qc
                            extra_nodes.add(u)
    extra_nodes = extra_nodes - qa_nodes
    schema_graph = sorted(qc_ids) + sorted(ac_ids) + sorted(extra_nodes)
    arange = np.arange(len(schema_graph))
    qmask = arange < len(qc_ids)
    amask = (arange >= len(qc_ids)) & (arange < (len(qc_ids) + len(ac_ids)))
    adj, concepts = concepts2adj(schema_graph)
    return adj, concepts, qmask, amask



######################################################################
# from transformers import LongformerForMaskedLM, LongformerTokenizer


# class LongformerForMaskedLMwithLoss(LongformerForMaskedLM):
#     #
#     def __init__(self, config):
#         super().__init__(config)
#     #
#     def forward(self, input_ids=None, attention_mask=None, token_type_ids=None, position_ids=None, head_mask=None, masked_lm_labels=None):
#         #
#         assert attention_mask is not None
#         outputs = self.roberta(input_ids, attention_mask=attention_mask, token_type_ids=token_type_ids, position_ids=position_ids, head_mask=head_mask)
#         sequence_output = outputs[0] #hidden_states of final layer (batch_size, sequence_length, hidden_size)
#         prediction_scores = self.lm_head(sequence_output)
#         outputs = (prediction_scores, sequence_output) + outputs[2:]
#         if masked_lm_labels is not None:
#             loss_fct = torch.nn.CrossEntropyLoss(reduction='none')
#             bsize, seqlen = input_ids.size()
#             masked_lm_loss = loss_fct(prediction_scores.view(-1, self.config.vocab_size), masked_lm_labels.view(-1)).view(bsize, seqlen)
#             masked_lm_loss = (masked_lm_loss * attention_mask).sum(dim=1)
#             outputs = (masked_lm_loss,) + outputs
#             # (masked_lm_loss), prediction_scores, sequence_output, (hidden_states), (attentions)
#         return outputs

# print ('loading pre-trained LM...')
# TOKENIZER = LongformerTokenizer.from_pretrained('allenai/longformer-base-4096')
# LM_MODEL = LongformerForMaskedLMwithLoss.from_pretrained('allenai/longformer-base-4096')
# LM_MODEL.cuda(); LM_MODEL.eval()
print ('loading done')

def get_LM_score(cids, source_text):
    cids = cids[:]
    cids.insert(0, -1) #QAcontext node
    sents, scores = [], []
    for cid in cids:
        if cid==-1:
            sent = source_text.lower()
        else:
            sent = '{} {}.'.format(source_text.lower(), ' '.join(id2concept[cid].split('_')))
        import pdb;pdb.set_trace()

        sent = TOKENIZER.encode(sent, add_special_tokens=True)
        sents.append(sent)

    n_cids = len(cids)
    cur_idx = 0
    batch_size = 50
    while cur_idx < n_cids:
        #Prepare batch
        input_ids = sents[cur_idx: cur_idx+batch_size]
        max_len = max([len(seq) for seq in input_ids])
        for j, seq in enumerate(input_ids):
            seq += [TOKENIZER.pad_token_id] * (max_len-len(seq))
            input_ids[j] = seq
        input_ids = torch.tensor(input_ids).cuda() #[B, seqlen]
        mask = (input_ids!=1).long() #[B, seq_len]
        #Get LM score
        with torch.no_grad():
            outputs = LM_MODEL(input_ids, attention_mask=mask, masked_lm_labels=input_ids)
            loss = outputs[0] #[B, ]
            _scores = list(-loss.detach().cpu().numpy()) #list of float
        scores += _scores
        cur_idx += batch_size
    assert len(sents) == len(scores) == len(cids)
    cid2score = OrderedDict(sorted(list(zip(cids, scores)), key=lambda x: -x[1])) #score: from high to low
    return cid2score

def concepts_to_adj_matrices_2hop_all_pair__use_LM__Part1(data):
    concept_ids, sentence = data
    # concept_nodes = set(concept_ids)
    # extra_nodes = set()
    # for cid in concept_nodes:
    #     if cid in cpnet_simple.nodes:
    #         extra_nodes |= set(cpnet_simple[cid])
    # extra_nodes = extra_nodes - concept_nodes
    return (sorted(concept_ids), sentence)

def mp_concepts_to_adj_matrices_2hop_all_pair__use_LM__Part11(data):
    concept_ids_1, concept_ids_2, sentence_1, j1, doc_id = data
    qa_nodes = set(concept_ids_1) | set(concept_ids_2)
    extra_nodes_weights = []
    seen = set()
    for qid in qa_nodes:
        for aid in qa_nodes:
            if qid != aid and qid in cpnet_simple.nodes and aid in cpnet_simple.nodes:
                cpnet_q = cpnet_simple[qid]
                cpnet_a = cpnet_simple[aid]
                # extra_nodes |= [cpnet_q[id]['weight'] + cpnet_a[id]['weight'] for id in list(set(cpnet_q) & set(cpnet_a))]
                extra_nodes_weights.extend(sorted([(cpnet_q[id]['weight'] + cpnet_a[id]['weight'], id) for id in list(set(cpnet_q) & set(cpnet_a))]))
    # sort extra_nodes_weights
    extra_nodes_weights = sorted(extra_nodes_weights, reverse=True)
    extra_nodes = [b for a, b in extra_nodes_weights
              if not (b in seen or seen.add(b))]
    nodes_to_preserve = min([10000, len(extra_nodes)])

    extra_nodes = extra_nodes[:nodes_to_preserve]

    return qa_nodes, set(extra_nodes), j1, sentence_1, doc_id


def mp_concepts_to_adj_matrices_2hop_all_pair__use_LM__Part11_debug(data):
    concept_ids_1, concept_ids_2, sentence_1, j1, doc_id = data
    qa_nodes = set(concept_ids_1) | set(concept_ids_2)
    extra_nodes_weights = []
    seen = set()
    c=0
    print(f'leeennn : {len(qa_nodes)}')
    for qid in qa_nodes:
        for aid in qa_nodes:
            if qid != aid and qid in cpnet_simple.nodes and aid in cpnet_simple.nodes:
                cpnet_q = cpnet_simple[qid]
                cpnet_a = cpnet_simple[aid]

                if cpnet.has_edge(qid, aid):
                    for data in cpnet[qid][aid].values():
                        # import pdb;pdb.set_trace()
                        if data['rel'] not in (15, 32):
                            print(data['rel'])

                            extra_nodes_weights.extend(sorted([(cpnet_q[id]['weight'] + cpnet_a[id]['weight'], id) for id in
                                                               list(set(cpnet_q) & set(cpnet_a))]))

                # extra_nodes |= [cpnet_q[id]['weight'] + cpnet_a[id]['weight'] for id in list(set(cpnet_q) & set(cpnet_a))]
                # extra_nodes_weights.extend(sorted([(cpnet_q[id]['weight'] + cpnet_a[id]['weight'], id) for id in list(set(cpnet_q) & set(cpnet_a))]))
    # sort extra_nodes_weights
    extra_nodes_weights = sorted(extra_nodes_weights, reverse=True)
    extra_nodes = [b for a, b in extra_nodes_weights
              if not (b in seen or seen.add(b))]
    nodes_to_preserve = min([50, len(extra_nodes)])

    extra_nodes = extra_nodes[:nodes_to_preserve]

    return qa_nodes, set(extra_nodes), j1, sentence_1, doc_id


def concepts_to_adj_matrices_2hop_all_pair__use_LM__Part11(sentences_unpruned_concept_ids, num_processes=20):
    all_concept_ids = set()
    all_extra_nodes = set()

    mp_list = []

    # take one set of concept_ids and compare them agains remaining concept_ids...
    # keep track of sentences as well...
    # print('Combining info from all sentences')
    doc_id = sentences_unpruned_concept_ids[0]
    sents = sentences_unpruned_concept_ids[1]

    for j1, data_1 in tqdm(enumerate(sents), total=len(sents)):
        concept_ids_1, sentence_1 = data_1
        concept_ids_rest = set()
        for j2, data_2 in enumerate(sents[:j1] + sents[j1+1:]):
            concept_ids_2, _ = data_2
            concept_ids_rest |= concept_ids_2 | concept_ids_1

        mp_list.append((concept_ids_1, concept_ids_rest, sentence_1, j1, doc_id))
        break


    # pool = Pool(20)

    qa_nodes, extra_nodes, _, _, doc_id = mp_concepts_to_adj_matrices_2hop_all_pair__use_LM__Part11_debug(mp_list[0])
    extra_nodes = extra_nodes - qa_nodes
    all_concept_ids.update(qa_nodes)
    all_sentences = ' '.join([s[1] for s in sents])
    extra_nodes = extra_nodes - qa_nodes
    all_extra_nodes.update(extra_nodes)
    all_concept_ids.update(qa_nodes)

    # for out in pool.imap_unordered(mp_concepts_to_adj_matrices_2hop_all_pair__use_LM__Part11, mp_list):
    #     qa_nodes, extra_nodes, j1, sentence_1, doc_id = out
    #     extra_nodes = extra_nodes - qa_nodes
    #     all_concept_ids.update(qa_nodes)
    #
    #     if j1 not in all_sentences.keys():
    #         all_sentences[j1] = sentence_1
    #     all_extra_nodes.update(extra_nodes)
    # all_sentences = ' '.join([s[1] for s in sorted(all_sentences.items(), key=lambda x:x[0])]).strip()


    return (doc_id, sorted(all_concept_ids), all_sentences, sorted(all_extra_nodes))

def concepts_to_adj_matrices_2hop_all_pair__use_LM__Part2(data):
    concept_ids, sentence, extra_nodes = data

    cid2score = get_LM_score(concept_ids+extra_nodes, sentence)
    return (concept_ids, sentence, extra_nodes, cid2score)

def concepts_to_adj_matrices_1hop_neighbours_without_relatedto(data):
    qc_ids, ac_ids = data
    qa_nodes = set(qc_ids) | set(ac_ids)
    extra_nodes = set()
    for u in set(qc_ids) | set(ac_ids):
        if u in cpnet.nodes:
            for v in cpnet[u]:
                for data in cpnet[u][v].values():
                    if data['rel'] not in (15, 32):
                        extra_nodes.add(v)
    extra_nodes = extra_nodes - qa_nodes
    # schema_graph = sorted(qc_ids) + sorted(ac_ids) + sorted(extra_nodes)
    # arange = np.arange(len(schema_graph))
    # qmask = arange < len(qc_ids)
    # amask = (arange >= len(qc_ids)) & (arange < (len(qc_ids) + len(ac_ids)))
    # adj, concepts = concepts2adj(schema_graph)
    # return adj, concepts, qmask, amask
    return extra_nodes


def concepts_to_adj_matrices_2hop_all_pair__use_LM__Part3(data):
    doc_id, concept_ids, sentence, extra_nodes = data

    schema_graph = concept_ids + extra_nodes
    arange = np.arange(len(schema_graph))
    sentence_mask = arange < len(concept_ids)

    # amask = (arange >= len(concept_ids)) & (arange < (len(concept_ids)))
    adj, concepts, adj_main = concepts2adj(schema_graph)

    return {'doc_id': doc_id, 'adj': adj, 'concepts': concepts, 'sentence_mask': sentence_mask, 'adj_main': adj_main, 'sentence': sentence}

################################################################################



#####################################################################################################
#                     functions below this line will be called by preprocess.py                     #
#####################################################################################################


def generate_graph(grounded_path, pruned_paths_path, cpnet_vocab_path, cpnet_graph_path, output_path):
    print(f'generating schema graphs for {grounded_path} and {pruned_paths_path}...')

    global concept2id, id2concept, relation2id, id2relation
    if any(x is None for x in [concept2id, id2concept, relation2id, id2relation]):
        load_resources(cpnet_vocab_path)

    global cpnet, cpnet_simple
    if cpnet is None or cpnet_simple is None:
        load_cpnet(cpnet_graph_path)

    nrow = sum(1 for _ in open(grounded_path, 'r'))
    with open(grounded_path, 'r') as fin_gr, \
            open(pruned_paths_path, 'r') as fin_pf, \
            open(output_path, 'w') as fout:
        for line_gr, line_pf in tqdm(zip(fin_gr, fin_pf), total=nrow):
            mcp = json.loads(line_gr)
            qa_pairs = json.loads(line_pf)

            statement_paths = []
            statement_rel_list = []
            for qas in qa_pairs:
                if qas["pf_res"] is None:
                    cur_paths = []
                    cur_rels = []
                else:
                    cur_paths = [item["path"] for item in qas["pf_res"]]
                    cur_rels = [item["rel"] for item in qas["pf_res"]]
                statement_paths.extend(cur_paths)
                statement_rel_list.extend(cur_rels)

            qcs = [concept2id[c] for c in mcp["qc"]]
            acs = [concept2id[c] for c in mcp["ac"]]

            gobj = plain_graph_generation(qcs=qcs, acs=acs,
                                          paths=statement_paths,
                                          rels=statement_rel_list)
            fout.write(json.dumps(gobj) + '\n')

    print(f'schema graphs saved to {output_path}')
    print()


def generate_adj_matrices(ori_schema_graph_path, cpnet_graph_path, cpnet_vocab_path, output_path, num_processes, num_rels=34, debug=False):
    print(f'generating adjacency matrices for {ori_schema_graph_path} and {cpnet_graph_path}...')

    global concept2id, id2concept, relation2id, id2relation
    if any(x is None for x in [concept2id, id2concept, relation2id, id2relation]):
        load_resources(cpnet_vocab_path)

    global cpnet_all
    if cpnet_all is None:
        cpnet_all = nx.read_gpickle(cpnet_graph_path)

    with open(ori_schema_graph_path, 'r') as fin:
        nxg_strs = [line for line in fin]

    if debug:
        nxgs = nxgs[:1]

    with Pool(num_processes) as p:
        res = list(tqdm(p.imap(generate_adj_matrix_per_inst, nxg_strs), total=len(nxg_strs)))

    with open(output_path, 'wb') as fout:
        pickle.dump(res, fout)

    print(f'adjacency matrices saved to {output_path}')
    print()


def generate_adj_data_from_grounded_concepts(grounded_path, cpnet_graph_path, cpnet_vocab_path, output_path, num_processes):
    """
    This function will save
        (1) adjacency matrics (each in the form of a (R*N, N) coo sparse matrix)
        (2) concepts ids
        (3) qmask that specifices whether a node is a question concept
        (4) amask that specifices whether a node is a answer concept
    to the output path in python pickle format

    grounded_path: str
    cpnet_graph_path: str
    cpnet_vocab_path: str
    output_path: str
    num_processes: int
    """
    print(f'generating adj data for {grounded_path}...')

    global concept2id, id2concept, relation2id, id2relation, cpnet_simple, cpnet
    if any(x is None for x in [concept2id, id2concept, relation2id, id2relation]):
        load_resources(cpnet_vocab_path)
    if cpnet is None or cpnet_simple is None:
        load_cpnet(cpnet_graph_path)

    qa_data = []
    with open(grounded_path, 'r', encoding='utf-8') as fin:
        for line in fin:
            dic = json.loads(line)
            q_ids = set(concept2id[c] for c in dic['qc'])
            a_ids = set(concept2id[c] for c in dic['ac'])
            q_ids = q_ids - a_ids
            qa_data.append((q_ids, a_ids))

    with Pool(num_processes) as p:
        res = list(tqdm(p.imap(concepts_to_adj_matrices_2hop_all_pair, qa_data), total=len(qa_data)))

    # res is a list of tuples, each tuple consists of four elements (adj, concepts, qmask, amask)
    with open(output_path, 'wb') as fout:
        pickle.dump(res, fout)

    print(f'adj data saved to {output_path}')
    print()

def visualize_word_graph(res3, id2concept, id2relation):
    # res3 =  {'doc_id': doc_id, 'adj': adj, 'concepts': concepts, 'sentence_mask': sentence_mask, 'adj_main': adj_main}
    adj = res3[0]['adj_main']
    cids = res3[0]['concepts']
    sentence = res3[0]['sentence']
    # get words, relations, and conections...

    relations = {}
    for relation_id in tqdm(range(adj.shape[0]), total=adj.shape[0]):
        relation_name = id2relation[relation_id]

        if relation_name == 'relatedto':
            continue

        word_list = adj[relation_id, :, :]

        for src_node in range(word_list.shape[0]):
            for tgt_node in range(word_list.shape[1]):
                if word_list[src_node, tgt_node] == 1:
                    # word match
                    if id2concept[cids[src_node]] in relations.keys():
                        relations[id2concept[cids[src_node]]].append((id2concept[cids[tgt_node]], relation_name))
                    else:
                        relations[id2concept[cids[src_node]]] = [(id2concept[cids[tgt_node]], relation_name)]

    import matplotlib.pyplot as plt
    import networkx as nx
    edges_node_names = [[s, t[0]] for s in relations.keys() for t in relations[s]]
    edges_relation_name = {(s, t[0]) : t[1] for s in relations.keys() for t in relations[s]}
    G = nx.Graph()
    G.add_edges_from(edges_node_names)
    pos = nx.spring_layout(G)
    plt.figure(3, figsize=(100, 100))

    nx.draw(
        G, pos, edge_color='black', width=1, linewidths=1,
        node_size=500, node_color='pink', alpha=0.9,font_size=5,
        labels={node: node for node in G.nodes()}
    )

    nx.draw_networkx_edge_labels(
        G, pos,
        edge_labels=edges_relation_name,font_size=5,
        font_color='red'
    )
    plt.axis('off')
    plt.show()
    plt.savefig('word_plot.pdf', format="pdf", dpi=5000)
    print(f'Sentence: \n {sentence}')




def generate_adj_data_from_grounded_concepts__use_LM(grounded_path, cpnet_graph_path, cpnet_vocab_path, output_path, num_processes):
    """
    This function will save
        (1) adjacency matrics (each in the form of a (R*N, N) coo sparse matrix)
        (2) concepts ids
        (3) qmask that specifices whether a node is a question concept
        (4) amask that specifices whether a node is a answer concept
        (5) cid2score that maps a concept id to its relevance score given the QA context
    to the output path in python pickle format

    grounded_path: str
    cpnet_graph_path: str
    cpnet_vocab_path: str
    output_path: str
    num_processes: int
    """
    print(f'generating adj data for {grounded_path}...')

    global concept2id, id2concept, relation2id, id2relation, cpnet_simple, cpnet


    if any(x is None for x in [concept2id, id2concept, relation2id, id2relation]):
        load_resources(cpnet_vocab_path)
    if cpnet is None or cpnet_simple is None:
        load_cpnet(cpnet_graph_path)

    conceptIDs_sentences_data = {}
    statement_path = grounded_path.replace('grounded', 'statement')

    with open(grounded_path, 'r', encoding='utf-8') as fin_ground, open(statement_path, 'r', encoding='utf-8') as fin_state:
        lines_ground = fin_ground.readlines()
        lines_state = fin_state.readlines()

        lines_state_searcher = {}
        for l in lines_state:
            ent = json.loads(l)
            lines_state_searcher[ent['doc_id']] = ent

        # for j, line in enumerate(lines_ground):
        j, k = 0, 0
        while j < len(lines_ground):
            line = lines_ground[j]
            dic = json.loads(line)
            concept_ids = set(concept2id[c] for c in dic['concepts'])
            statement_obj = lines_state_searcher[dic['doc_id']]
            if k < len(statement_obj['statements']):
                QAcontext = "{}".format(statement_obj['statements'][k])
            else:
                k = 0

            if dic['doc_id'] in conceptIDs_sentences_data.keys():
                conceptIDs_sentences_data[dic['doc_id']].append((concept_ids, QAcontext))
            else:
                conceptIDs_sentences_data[dic['doc_id']] = [(concept_ids, QAcontext)]

            j += 1
            k += 1
    ###############

    # res1=[]
    # with Pool(num_processes) as p:
    #     res1 = list(tqdm(p.imap(concepts_to_adj_matrices_2hop_all_pair__use_LM__Part1, conceptIDs_sentences_data), total=len(conceptIDs_sentences_data)))

    # res11 = concepts_to_adj_matrices_2hop_all_pair__use_LM__Part11(conceptIDs_sentences_data)
    # import pdb;pdb.set_trace()

    with Pool(num_processes) as p:
        res11 = list(tqdm(p.imap(concepts_to_adj_matrices_2hop_all_pair__use_LM__Part11, list(conceptIDs_sentences_data.items())), total=len(list(conceptIDs_sentences_data.items()))))
    #
    # res11 = []
    # for k, v in conceptIDs_sentences_data.items():
    #     res11.append(concepts_to_adj_matrices_2hop_all_pair__use_LM__Part11((k, v)))

    ###########

    # res2 = []
    # for j, _data in enumerate(res11):
    #     if j % 10 == 0: print (j)
    # res2.append(concepts_to_adj_matrices_2hop_all_pair__use_LM__Part2(res11))


    ############

    print('Last Part')
    with MyPool(num_processes) as p:
        res3 = list(tqdm(p.imap(concepts_to_adj_matrices_2hop_all_pair__use_LM__Part3, res11), total=len(res11)))
    # for r in res2:
    #     res3.append(concepts_to_adj_matrices_2hop_all_pair__use_LM__Part3(r))

    ############
    visualize_word_graph(res3, id2concept, id2relation)

    # res is a list of responses
    # with open(output_path, 'wb') as fout:
    #     pickle.dump(res3, fout)
    # print(f'adj data saved to {output_path}')
    # print()



#################### adj to sparse ####################

def coo_to_normalized_per_inst(data):
    adj, concepts, qm, am, max_node_num = data
    ori_adj_len = len(concepts)
    concepts = torch.tensor(concepts[:min(len(concepts), max_node_num)])
    adj_len = len(concepts)
    qm = torch.tensor(qm[:adj_len], dtype=torch.uint8)
    am = torch.tensor(am[:adj_len], dtype=torch.uint8)
    ij = adj.row
    k = adj.col
    n_node = adj.shape[1]
    n_rel = 2 * adj.shape[0] // n_node
    i, j = ij // n_node, ij % n_node
    mask = (j < max_node_num) & (k < max_node_num)
    i, j, k = i[mask], j[mask], k[mask]
    i, j, k = np.concatenate((i, i + n_rel // 2), 0), np.concatenate((j, k), 0), np.concatenate((k, j), 0)  # add inverse relations
    adj_list = []
    for r in range(n_rel):
        mask = i == r
        ones = np.ones(mask.sum(), dtype=np.float32)
        A = sparse.csr_matrix((ones, (k[mask], j[mask])), shape=(max_node_num, max_node_num))  # A is transposed by exchanging the order of j and k
        adj_list.append(normalize_sparse_adj(A, 'coo'))
    adj_list.append(sparse.identity(max_node_num, dtype=np.float32, format='coo'))
    return ori_adj_len, adj_len, concepts, adj_list, qm, am


def coo_to_normalized(adj_path, output_path, max_node_num, num_processes):
    print(f'converting {adj_path} to normalized adj')

    with open(adj_path, 'rb') as fin:
        adj_data = pickle.load(fin)
    data = [(adj, concepts, qmask, amask, max_node_num) for adj, concepts, qmask, amask in adj_data]

    ori_adj_lengths = torch.zeros((len(data),), dtype=torch.int64)
    adj_lengths = torch.zeros((len(data),), dtype=torch.int64)
    concepts_ids = torch.zeros((len(data), max_node_num), dtype=torch.int64)
    qmask = torch.zeros((len(data), max_node_num), dtype=torch.uint8)
    amask = torch.zeros((len(data), max_node_num), dtype=torch.uint8)

    adj_data = []
    with Pool(num_processes) as p:
        for i, (ori_adj_len, adj_len, concepts, adj_list, qm, am) in tqdm(enumerate(p.imap(coo_to_normalized_per_inst, data)), total=len(data)):
            ori_adj_lengths[i] = ori_adj_len
            adj_lengths[i] = adj_len
            concepts_ids[i][:adj_len] = concepts
            qmask[i][:adj_len] = qm
            amask[i][:adj_len] = am
            adj_list = [(torch.LongTensor(np.stack((adj.row, adj.col), 0)),
                         torch.FloatTensor(adj.data)) for adj in adj_list]
            adj_data.append(adj_list)

    torch.save((ori_adj_lengths, adj_lengths, concepts_ids, adj_data), output_path)

    print(f'normalized adj saved to {output_path}')
    print()

# if __name__ == '__main__':
#     generate_adj_matrices_from_grounded_concepts('./data/csqa/grounded/train.grounded.jsonl',
#                                                  './data/cpnet/conceptnet.en.pruned.graph',
#                                                  './data/cpnet/concept.txt',
#                                                  '/tmp/asdf', 40)

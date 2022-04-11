import argparse
from multiprocessing import cpu_count
from preprocess_utils.convert_csqa import convert_to_entailment
from preprocess_utils.convert_obqa import convert_to_obqa_statement
from preprocess_utils.conceptnet import extract_english, construct_graph
from preprocess_utils.grounding import create_matcher_patterns, ground
from preprocess_utils.graph import generate_adj_data_from_grounded_concepts__use_LM

input_paths = {
    'gov-reports': {
        'train': '/disk1/sajad/gov-reports/train.json',
        'val': '/disk1/sajad/gov-reports/val.json',
        'test': '/disk1/sajad/gov-reports/test.json',
    },
    'cpnet': {
        'csv': '/disk1/sajad/GreaseLM/data/cpnet/conceptnet-assertions-5.6.0.csv',
    },
}

output_paths = {
    'cpnet': {
        'csv': '/disk1/sajad/gov-reports/cpnet/conceptnet.en.csv',
        'vocab': '/disk1/sajad/gov-reports/cpnet/concept.txt',
        'patterns': '/disk1/sajad/gov-reports/cpnet/matcher_patterns.json',
        'unpruned-graph': '/disk1/sajad/gov-reports/cpnet/conceptnet.en.unpruned.graph',
        'pruned-graph': '/disk1/sajad/gov-reports/cpnet/conceptnet.en.pruned.graph',
    },
    'gov-reports': {
        'statement': {
            'train': '/disk1/sajad/gov-reports/statement/train.statement.jsonl',
            'dev': '/disk1/sajad/gov-reports/statement/dev.statement.jsonl',
            'test': '/disk1/sajad/gov-reports/statement/test.statement.jsonl',
        },
        'grounded': {
            'train': '/disk1/sajad/gov-reports/grounded/train.grounded.jsonl',
            'dev': '/disk1/sajad/gov-reports/grounded/dev.grounded.jsonl',
            'test': '/disk1/sajad/gov-reports/grounded/test.grounded.jsonl',
        },
        'graph': {
            'adj-train': '/disk1/sajad/gov-reports/graph/train.graph.adj.pk',
            'adj-dev': '/disk1/sajad/gov-reports/graph/dev.graph.adj.pk',
            'adj-test': '/disk1/sajad/gov-reports/graph/test.graph.adj.pk',
        },
    },
}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--run', default=[ 'gov-reports'], choices=['gov-reports'], nargs='+')
    parser.add_argument('-p', '--nprocs', type=int, default=cpu_count(), help='number of processes to use')
    parser.add_argument('--debug', action='store_true', help='enable debug mode')

    args = parser.parse_args()
    if args.debug:
        raise NotImplementedError()

    routines = {
        # 'common': [
        #     {'func': extract_english, 'args': (input_paths['cpnet']['csv'], output_paths['cpnet']['csv'], output_paths['cpnet']['vocab'])},
        #     {'func': construct_graph, 'args': (output_paths['cpnet']['csv'], output_paths['cpnet']['vocab'],
        #                                        output_paths['cpnet']['unpruned-graph'], False)},
        #     {'func': construct_graph, 'args': (output_paths['cpnet']['csv'], output_paths['cpnet']['vocab'],
        #                                        output_paths['cpnet']['pruned-graph'], True)},
        #     {'func': create_matcher_patterns, 'args': (output_paths['cpnet']['vocab'], output_paths['cpnet']['patterns'])},
        # ],
        'gov-reports': [
            # {'func': convert_to_entailment, 'args': (input_paths['csqa']['train'], output_paths['csqa']['statement']['train'])},
            # {'func': convert_to_entailment, 'args': (input_paths['csqa']['dev'], output_paths['csqa']['statement']['dev'])},
            {'func': convert_to_entailment, 'args': (input_paths['gov-reports']['test'], output_paths['gov-reports']['statement']['test'])},
            # {'func': ground, 'args': (output_paths['csqa']['statement']['train'], output_paths['cpnet']['vocab'],
            #                           output_paths['cpnet']['patterns'], output_paths['csqa']['grounded']['train'], args.nprocs)},
            # {'func': ground, 'args': (output_paths['csqa']['statement']['dev'], output_paths['cpnet']['vocab'],
            #                           output_paths['cpnet']['patterns'], output_paths['csqa']['grounded']['dev'], args.nprocs)},
            {'func': ground, 'args': (output_paths['gov-reports']['statement']['test'], output_paths['cpnet']['vocab'],
                                      output_paths['cpnet']['patterns'], output_paths['gov-reports']['grounded']['test'], args.nprocs)},
            # {'func': generate_adj_data_from_grounded_concepts__use_LM, 'args': (output_paths['csqa']['grounded']['train'], output_paths['cpnet']['pruned-graph'], output_paths['cpnet']['vocab'], output_paths['csqa']['graph']['adj-train'], args.nprocs)},
            # {'func': generate_adj_data_from_grounded_concepts__use_LM, 'args': (output_paths['csqa']['grounded']['dev'], output_paths['cpnet']['pruned-graph'], output_paths['cpnet']['vocab'], output_paths['csqa']['graph']['adj-dev'], args.nprocs)},

            {'func': generate_adj_data_from_grounded_concepts__use_LM, 'args': (output_paths['gov-reports']['grounded']['test'],
                                                                                output_paths['cpnet']['pruned-graph'],
                                                                                output_paths['cpnet']['vocab'],
                                                                                output_paths['gov-reports']['graph']['adj-test'], args.nprocs)},
        ],
    }

    for rt in args.run:
        for rt_dic in routines[rt]:
            rt_dic['func'](*rt_dic['args'])

    print('Successfully run {}'.format(' '.join(args.run)))


if __name__ == '__main__':
    main()

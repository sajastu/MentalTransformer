
# find word alignments between source and target
import json
import os
import pathlib
import pickle
from multiprocessing import Pool
from collections import Counter

from tqdm import tqdm
import nltk
from nltk.corpus import stopwords
import spacy

STOP_WORDS = set(
    """
a about above across after afterwards again against all almost alone along
already also although always am among amongst amount an and another any anyhow
anyone anything anyway anywhere are around as at
back be became because become becomes becoming been before beforehand behind
being below beside besides between beyond both bottom but by
call can cannot ca could
did do does doing done down due during
each eight either eleven else elsewhere empty enough even ever every
everyone everything everywhere except
few fifteen fifty first five for former formerly forty four from front full
further
get give go
had has have he hence her here hereafter hereby herein hereupon hers herself
him himself his how however hundred
i if in indeed into is it its itself
keep
last latter latterly least less
just
made make many may me meanwhile might mine more moreover most mostly move much
must my myself
name namely neither never nevertheless next nine no nobody none noone nor not
nothing now nowhere
of off often on once one only onto or other others otherwise our ours ourselves
out over own
part per perhaps please put
quite
rather re really regarding
same say see seem seemed seeming seems serious several she should show side
since six sixty so some somehow someone something sometime sometimes somewhere
still such
take ten than that the their them themselves then thence there thereafter
thereby therefore therein thereupon these they third this those though three
through throughout thru thus to together too top toward towards twelve twenty
two
under until up unless upon us used using
various very very via was we well were what whatever when whence whenever where
whereafter whereas whereby wherein whereupon wherever whether which while
whither who whoever whole whom whose why will with within without would
yet you your yours yourself yourselves
""".split()
)

nlp = spacy.load('en_core_web_lg')
stopwords_spacy = nlp.Defaults.stop_words

nltk.download('stopwords')
sws = stopwords.words('english') + ["get", "take", "n’t", '"', "ca", "nt", "'s", "got", "still", "...", ":", '"', "wo",
                                    "everything", "amp;#x200b", "_", "..", 'am', "'m", "'ve", "are", "is", "'s", ".", "n't", ",",
                                    "*", "feel", "much", "pretty", "also", "even", ")", "(", "want", "?", "/", "na", "'re", "-", "%"]

STOP_WORDS.update({"like", "going", "say", "took", "new", "tic", "lol"})



def get_tokens(txt):
    out_tokens = []
    doc = nlp(txt)
    for sent in doc.sents:
        for word in sent:
            out_tokens.append((word.text, word.lemma_))
    return out_tokens

def align_func(src_tokens, ment_main, ment_lowered):
    """
    generate new txt via <masking> the source...
    """

    old_src = [s[0] for s in src_tokens]
    found_masked = 0
    j = 0


    while j < len(src_tokens):
        tkn = src_tokens[j]

        if '<mask>' in src_tokens[j]:
            continue

        else:
            tkn_txt = tkn[0]

            ptr = 0
            while ptr < len(ment_main):
                mental_illness = ment_main[ptr].lower()
                # mental_illness_lower = ment_lowered[ptr]
                if len(mental_illness.split(' ')) == 1 and tkn_txt.lower() == mental_illness:
                    src_tokens[j] = ('<mask>', '<mask>')
                    j += 1
                    found_masked += 1
                    break

                # elif len(mental_illness_lower.split(' ')) == 1 and tkn_txt.lower() == mental_illness_lower:
                #     src_tokens[j] = ('<mask>', '<mask>')
                #     found_masked += 1
                #     j+=1
                #     break

                else:

                    if len(mental_illness.split(' ')) > 1:
                        ptr_2 = 0
                        z=0
                        while ptr_2 < len(mental_illness.split(' ')):
                            mental_illness_word = mental_illness.split(' ')[ptr_2].lower()
                            if src_tokens[j][0].lower() == mental_illness_word:
                                z+=1
                            ptr_2 += 1

                        if z == len(mental_illness.split(' ')):
                            # push j z's positions forward and change them to <mask>
                            for h in [s[0] for s in src_tokens[j: j+z]]:
                                src_tokens[h] = ('<mask>', '<mask>')
                                found_masked += 1

                            j = j+z
                            break
                        # else:
                            # j+=1
                            # break


                    # elif len(mental_illness_lower.split(' ')) > 1:
                    #     if len(mental_illness_lower.split(' ')) > 1:
                    #         ptr_2 = 0
                    #         z = 0
                    #         while ptr_2 < len(mental_illness_lower.split(' ')):
                    #             mental_illness_word = mental_illness.split(' ')[ptr_2]
                    #             if src_tokens[j][0] == mental_illness_word:
                    #                 z += 1
                    #             ptr_2 += 1
                    #
                    #         if z == len(mental_illness_lower.split(' ')):
                    #             # push j z's positions forward and change them to <mask>
                    #             for h in [s[0] for s in src_tokens[j: j+z]]:
                    #                 src_tokens[h] = ('<mask>', '<mask>')
                    #                 found_masked += 1
                    #
                    #             j = j + z
                    #             break
                            # else:
                            #     j += 1


                ptr += 1
            j += 1

    if found_masked > 2:
        return ' '.join(old_src), ' '.join([s[0] for s in src_tokens])

    else:
        return ' '.join(old_src), None

def align_func_with_tgt(src_tokens, tldr_tokens):
    """
    generate new txt via <masking> the source...
    """

    masked_tokens = []

    tldr_lemmas = [tkn[1].lower() for tkn in tldr_tokens]
    for j, tkn in enumerate(src_tokens):
        tkn_txt = tkn[0]
        tkn_lemma = tkn[1]
        if (tkn_lemma.lower() in tldr_lemmas and tkn_txt.lower() not in sws and len(tkn_txt) > 2):
            masked_tokens.append(tkn_txt.lower())

    return Counter(masked_tokens)



def _find_alignments_with_tgt(params):
    """
    find alingnments (lemmatize) except stopwords

    """

    instance = params[0]


    src_txt = instance['src']
    tldr_txt = instance['tldr']
    # post_id = instance['post_id']

    # tokenize the src and tldr txts
    src_tokens = get_tokens(src_txt)
    tldr_tokens = get_tokens(tldr_txt)
    masked_tokens_counter = align_func_with_tgt(src_tokens, tldr_tokens)



    return masked_tokens_counter

def _find_alignments(params):
    """
    find alingnments (lemmatize) except stopwords

    """

    instance_path = params[0]
    ment_main = params[1]
    ment_lowered = params[2]
    instance = json.load(open(instance_path))

    src_txt = instance['src']
    # post_id = instance['post_id']

    # tokenize the src and tldr txts
    src_tokens = get_tokens(src_txt)
    old_src, new_src = align_func(src_tokens, ment_main, ment_lowered)

    if new_src is None:
        return None, None
    else:
        return old_src.strip(), new_src.strip()




aligned_vocab = []
mental_vocab = []
mental_vocab_lowered = []
with open('mental_illnesses.txt') as fR:
    for l in fR:
        mental_vocab.append(l.strip())
        mental_vocab_lowered.append(l.strip())


    # if not os.path.exists("top_copied_terms_mental.pkl"):
    #     instances_tgt = []
    #     pool = Pool(20)
    #     for st in ['train', 'val', 'test']:
    #         with open(f'/disk1/sajad/datasets/medical/mental-reddit-final/sets/{st}.json') as fR:
    #             for l in fR:
    #                 instances_tgt.append((json.loads(l), mental_vocab, mental_vocab_lowered))
    #
    #
    #     counters = 0
    #     all_masked_tokens = Counter()
    #
    #     for masked_tokens in tqdm(pool.imap_unordered(_find_alignments_with_tgt, instances_tgt), total=len(instances_tgt)):
    #         all_masked_tokens += masked_tokens
    #
    #     # for ins_tgt in instances_tgt:
    #     #     counters+=1
    #     #     masked_tokens_counter = _find_alignments_with_tgt(ins_tgt)
    #     #     mental_vocab.extend()
    #         # if counters==20:
    #         #     break
    #
    #     all_masked_tokens = sorted(all_masked_tokens.items(), key=lambda i: i[1], reverse=True)
    #
    #     # save in pickle now...
    #     with open('top_copied_terms_mental.pkl', 'wb') as f:
    #         pickle.dump(all_masked_tokens, f)
    #
    # else:
    #     with open('top_copied_terms_mental.pkl', 'rb') as fR:
    #         all_masked_tokens = pickle.load(fR)
    #
    # all_masked_tokens_2 = [a[0] for a in all_masked_tokens][:2500]
    # for u in list(STOP_WORDS):
    #     if u in all_masked_tokens_2:
    #         all_masked_tokens_2.remove(u)
    # all_masked_tokens = all_masked_tokens_2[:2000]


    # mental_vocab += all_masked_tokens
    # mental_vocab_lowered += all_masked_tokens


# for st in ['train', 'val', 'test']:
    instances = []
    instances_tgt = []
    counter = 0
    for root, dirs, files in tqdm(os.walk("/disk1/sajad/reddit_dumps/mental-reddit/all_sets/submissions", topdown=True)):
        for name in files:
            # print(os.path.join(root, name))
            if '.json' in os.path.join(root, name):
                # with open(f'/disk1/sajad/datasets/medical/mental-reddit-final/sets/{st}.json') as fR:
                # with open(f'{os.path.join(root, name)}') as fR:
                #     for l in fR:
                instances.append((os.path.join(root, name), mental_vocab, mental_vocab_lowered))

    pool = Pool(20)

    # for ins in instances:
    #     _find_alignments(ins)


    splits = {
        'train': 200000,
        'val': 8000,
        'test': 8000,
    }
    fW_train = open(f'/disk1/sajad/datasets/medical/mental-reddit-final/sets/train-lm-large-mentalWords.json', mode='a')
    fW_val = open(f'/disk1/sajad/datasets/medical/mental-reddit-final/sets/val-lm-large-mentalWords.json', mode='a')
    fW_test = open(f'/disk1/sajad/datasets/medical/mental-reddit-final/sets/test-lm-large-mentalWords.json', mode='a')
    witten_posts = 0
    for old_src, new_src in tqdm(pool.imap_unordered(_find_alignments, instances), total=len(instances)):

        if new_src is not None:
            if splits['train'] > 0:
                fW_train.write(json.dumps(
                    {
                    'original_src': old_src,
                    'masked_src': new_src
                    }))
                witten_posts += 1
                fW_train.write('\n')
                splits['train'] -= 1
                if witten_posts % 10000 == 0:
                    print(f'train {witten_posts} instances written...')

                if witten_posts == 200000:
                    witten_posts = 0

            elif splits['val'] > 0:
                fW_val.write(json.dumps(
                    {
                        'original_src': old_src,
                        'masked_src': new_src
                    }))
                fW_val.write('\n')
                witten_posts +=1


                splits['val'] -= 1
                if witten_posts% 4000 == 0:
                    print(f'val {witten_posts} instances written...')

                if witten_posts==8000:
                    witten_posts = 0

            elif splits['test'] > 0:
                fW_test.write(json.dumps(
                    {
                        'original_src': old_src,
                        'masked_src': new_src
                    }))
                fW_test.write('\n')
                witten_posts += 1


                splits['test'] -= 1
                if witten_posts % 4000 == 0:
                    print(f'test {witten_posts} instances written...')
                if witten_posts == 8000:
                    witten_posts = 0
            else:
                break

print('done')
import copy

import torch
from torch.autograd import Variable

USE_CUDA = torch.cuda.is_available()

FloatTensor = torch.cuda.FloatTensor if USE_CUDA else torch.FloatTensor
ByteTensor = torch.cuda.ByteTensor if USE_CUDA else torch.ByteTensor
LongTensor = torch.cuda.LongTensor if USE_CUDA else torch.LongTensor


def getbatch(data, batch_size):
    sindex = 0
    eindex = batch_size
    batches = []
    while eindex < len(data):
        batch = data[sindex:eindex]
        sindex = eindex
        eindex = eindex + batch_size
        yield batch

    if sindex < len(data):
        yield data[sindex:]


flatten = lambda l: [item for sublist in l for item in sublist]


def pad_to_batch(batch, w_to_ix):  # for bAbI dataset
    fact, q, a = list(zip(*batch))
    max_fact = max([len(f) for f in fact])
    max_len = max([f.size(0) for f in flatten(fact)])
    max_q = max([qq.size(0) for qq in q])
    max_a = max([aa.size(0) for aa in a])

    facts, fact_masks, q_p, a_p = [], [], [], []
    for i in range(len(batch)):
        fact_p_t = []
        for j in range(len(fact[i])):
            if fact[i][j].size(0) < max_len:
                fact_p_t.append(torch.cat(
                    [fact[i][j], Variable(LongTensor([w_to_ix['<PAD>']] * (max_len - fact[i][j].size(0))))]).view(1,
                                                                                                                  -1))
            else:
                fact_p_t.append(fact[i][j].view(1, -1))

        while len(fact_p_t) < max_fact:
            fact_p_t.append(Variable(LongTensor([w_to_ix['<PAD>']] * max_len)).view(1, -1))

        fact_p_t = torch.cat(fact_p_t)
        facts.append(fact_p_t)
        fact_masks.append(torch.cat(
            [Variable(ByteTensor(tuple(map(lambda s: s == 0, t.data))), volatile=False) for t in fact_p_t]).view(
            fact_p_t.size(0), -1))

        if q[i].size(0) < max_q:
            q_p.append(torch.cat([q[i], Variable(LongTensor([w_to_ix['<PAD>']] * (max_q - q[i].size(0))))]).view(1, -1))
        else:
            q_p.append(q[i].view(1, -1))

        if a[i].size(0) < max_a:
            a_p.append(torch.cat([a[i], Variable(LongTensor([w_to_ix['<PAD>']] * (max_a - a[i].size(0))))]).view(1, -1))
        else:
            a_p.append(a[i].view(1, -1))

    questions = torch.cat(q_p)
    answers = torch.cat(a_p)
    question_masks = torch.cat(
        [Variable(ByteTensor(tuple(map(lambda s: s == 0, t.data))), volatile=False) for t in questions]).view(
        questions.size(0), -1)

    return facts, fact_masks, questions, question_masks, answers


def prepare_sequence(sentence, word2id):
    idxs = list(map(lambda w: word2id[w] if w in word2id.keys() else word2id["<UNK>"], sentence))
    return Variable(LongTensor(idxs))


def bAbI_data_test(data, word2ix):
    for t in data:
        for i, fact in enumerate(t[0]):
            t[0][i] = prepare_sequence(fact, word2ix).view(1, -1)

        t[1] = prepare_sequence(t[1], word2ix).view(1, -1)
        t[2] = prepare_sequence(t[2], word2ix).view(1, -1)

    return data


def bAbI_data_load(path):
    print('Load the data from %s' % path)
    try:
        data = open(path).readlines()
    except:
        print("Such a file does not exist at %s".format(path))
        return None

    data = [d[:-1] for d in data]
    data_p = []
    fact = []
    qa = []
    try:
        for d in data:
            index = d.split(' ')[0]
            if index == '1':
                fact = []
                qa = []
            if '?' in d:
                temp = d.split('\t')
                q = temp[0].strip().replace('?', '').split(' ')[1:] + ['?']
                a = temp[1].split() + ['</s>']
                stemp = copy.deepcopy(fact)
                data_p.append([stemp, q, a])
            else:
                tokens = d.replace('.', '').split(' ')[1:] + ['</s>']
                fact.append(tokens)
    except:
        print("Please check the data is right")
        return None
    print('Data Load over ,Count : %d' % len(data_p))
    return data_p


data_p = bAbI_data_load('qa5_three-arg-relations_train.txt')


def build_words_dict(data):
    print('Build the words dict now...')
    fact, q, a = list(zip(*data))  # *data把data散列， zip把data按照列组装起来，然后list

    vacab = set(flatten(flatten(fact) + list(q) + list(a)))
    word2id = {'<PAD>': 0, '<UNK>': 1, '<s>': 2, '</s>': 3}
    for w in vacab:
        if w not in word2id:
            word2id.setdefault(w, len(word2id))
    index2word = {v: k for k, v in word2id.items()}
    print('Build the words dict over.')
    return word2id, index2word

# build_words_dict(data_p)

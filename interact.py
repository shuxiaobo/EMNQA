import argparse
import random
import sys

sys.path.append('..')
import torch
from torch.autograd import Variable

from EMNQA import util
from EMNQA.model import DMN
from EMNQA.util import prepare_sequence

USE_CUDA = torch.cuda.is_available()

FloatTensor = torch.cuda.FloatTensor if USE_CUDA else torch.FloatTensor
LongTensor = torch.cuda.LongTensor if USE_CUDA else torch.LongTensor
ByteTensor = torch.cuda.ByteTensor if USE_CUDA else torch.ByteTensor
HIDDEN_SIZE = 80
BATCH_SIZE = 64
LR = 0.001
EPOCH = 50
NUM_EPISODE = 3
EARLY_STOPPING = False
DATA_WORKS = 4


def pad_fact(fact, x_to_ix):  # this is for inference

    max_x = max([s.size(1) for s in fact])
    x_p = []
    for i in range(len(fact)):
        if fact[i].size(1) < max_x:
            x_p.append(
                torch.cat([fact[i], Variable(LongTensor([x_to_ix['<PAD>']] * (max_x - fact[i].size(1)))).view(1, -1)],
                          1))
        else:
            x_p.append(fact[i])

    fact = torch.cat(x_p)
    fact_mask = torch.cat(
        [Variable(ByteTensor(tuple(map(lambda s: s == 0, t.data))), volatile=False) for t in fact]).view(fact.size(0),
                                                                                                         -1)
    return fact, fact_mask


def process(word2index):
    for t in test_data:
        for i, fact in enumerate(t[0]):
            t[0][i] = prepare_sequence(fact, word2index).view(1, -1)

        t[1] = prepare_sequence(t[1], word2index).view(1, -1)
        t[2] = prepare_sequence(t[2], word2index).view(1, -1)
    print('test data precess over')
    t = random.choice(test_data)
    fact, fact_mask = pad_fact(t[0], word2id)
    question = t[1]
    question_mask = Variable(ByteTensor([0] * t[1].size(1)), volatile=False).unsqueeze(0)
    answer = t[2].squeeze(0)

    model.zero_grad()
    pred = model([fact], [fact_mask], question, question_mask, answer.size(0), args.episode)

    print("\n\n\nFacts : ")
    print('\n'.join([' '.join(list(map(lambda x: index2word[x], f))) for f in fact.data.tolist()]))
    print("")
    print("Question : ", ' '.join(list(map(lambda x: index2word[x], question.data.tolist()[0]))))
    print("")
    print("Answer : ", ' '.join(list(map(lambda x: index2word[x], answer.data.tolist()))))
    print("Prediction : ", ' '.join(list(map(lambda x: index2word[x], pred.max(1)[1].data.tolist()))))


if __name__ == '__main__':
    args = argparse.ArgumentParser()
    args.add_argument('--episode', type=int, default=3)

    args = args.parse_args()

    m = torch.load('earlystoping-EMNQA.model', map_location=lambda storage, loc: storage)
    model = DMN(HIDDEN_SIZE, len(m['word2idx']), len(m['word2idx']), m['word2idx'])
    model.load_state_dict(state_dict=m['state_dict'])
    word2id = model.word2index
    test_data = util.bAbI_data_load('qa5_three-arg-relations_test.txt')
    index2word = {v: k for k, v in word2id.items()}
    process(word2id)

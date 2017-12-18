import torch
from torch.autograd import Variable
from DMNQA import util

USE_CUDA = torch.cuda.is_available()

FloatTensor = torch.cuda.FloatTensor if USE_CUDA else torch.FloatTensor
LongTensor = torch.cuda.LongTensor if USE_CUDA else torch.LongTensor
ByteTensor = torch.cuda.ByteTensor if USE_CUDA else torch.ByteTensor


def pad_fact(data, word2id):
    max_len = max([len(fact) for fact in data])

    facts = []
    facts_mask = []
    for f in data:
        if len(f) < max_len:
            facts.append(Variable(LongTensor(f.extend([[word2id['<PAD>']] * (max_len - len(f))]))))

        else:
            facts.append(Variable(LongTensor(f)))
        facts_mask.append(Variable(ByteTensor(tuple(map(lambda s: s == 0, f)))))

    return torch.cat(facts), torch.cat(facts_mask)

if __name__ == '__main__':
    data = util.bAbI_data_load('qa5_three-arg-relations_test.txt')
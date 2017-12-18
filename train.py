import argparse
import logging
import os
import sys

import numpy as np
import torch
from torch.autograd import Variable
from torch.optim import Adam
from torch.utils.data import DataLoader

sys.path.append('..')
from EMNQA import util
from EMNQA.data_set import QAdataset
from EMNQA.model import DMN

logger = logging.getLogger()
logger.setLevel(logging.INFO)
fmt = logging.Formatter('%(asctime)s: [ %(message)s ]', '%m/%d/%Y %I:%M:%S %p')
console = logging.StreamHandler()
console.setFormatter(fmt)
logger.addHandler(console)

HIDDEN_SIZE = 80
BATCH_SIZE = 64
LR = 0.001
EPOCH = 50
NUM_EPISODE = 3
EARLY_STOPPING = False
DATA_WORKS = 4

USE_CUDA = torch.cuda.is_available()

FloatTensor = torch.cuda.FloatTensor if USE_CUDA else torch.FloatTensor
LongTensor = torch.cuda.LongTensor if USE_CUDA else torch.LongTensor
ByteTensor = torch.cuda.ByteTensor if USE_CUDA else torch.ByteTensor


def prepare_data(filename):  # data -> dataloader
    data_p = util.bAbI_data_load(filename)

    word2idx, idx2word = util.build_words_dict(data_p)

    data_set = QAdataset(data_p, word2idx)

    train_dataloader = DataLoader(data_set,
                                  batch_size=BATCH_SIZE,
                                  # sampler=train_sampler,
                                  num_workers=DATA_WORKS,
                                  collate_fn=util.pad_to_batch,
                                  pin_memory=USE_CUDA, )

    return train_dataloader, word2idx


def seq2variable(data, word2id):  # data -> variable
    for t in data:
        for i, f in enumerate(t[0]):
            t[0][i] = util.prepare_sequence(f, word2id)
        t[1] = util.prepare_sequence(t[1], word2id)
        t[2] = util.prepare_sequence(t[2], word2id)


def train_from_scratch(filename):  # training
    train_data = util.bAbI_data_load(filename)
    test_data = util.bAbI_data_load(args_dic.test_data_file)
    word2idx, idx2word = util.build_words_dict(train_data)
    test_data = util.bAbI_data_test(test_data, word2idx)
    seq2variable(train_data, word2idx)
    print('Model init.')
    model = DMN(HIDDEN_SIZE, len(word2idx), len(word2idx), word2idx)

    if USE_CUDA:
        model = model.cuda()
    model.init_weight()
    # data_loader = prepare_data(filename)

    optimizer = Adam(model.parameters(), lr=LR)
    loss_fun = torch.nn.CrossEntropyLoss(ignore_index=0)

    EARLY_STOPPING = False

    print('Begin Training!')
    for i in range(EPOCH):
        losses = []
        if EARLY_STOPPING: break

        for j, batch in enumerate(util.getbatch(train_data, BATCH_SIZE)):
            facts, fact_masks, questions, question_masks, answers = util.pad_to_batch(batch, word2idx)

            model.zero_grad()
            pred = model(facts, fact_masks, questions, question_masks, answers.size(1), NUM_EPISODE, True)
            loss = loss_fun(pred, answers.view(-1))
            losses.append(loss.data.tolist()[0])

            loss.backward()
            optimizer.step()

            if j % 100 == 0:
                logger.info("[%d/%d] mean_loss : %0.2f" % (i, EPOCH, np.mean(losses)))
                # print("[%d/%d] mean_loss : %0.2f" % (i, EPOCH, np.mean(losses)))

                if np.mean(losses) < 0.01:
                    EARLY_STOPPING = True
                    print("Early Stopping!")
                    torch.save({'state_dict': model.state_dict(), 'word2idx': model.word2index},
                               'earlystoping-%s' % args_dic.model_file)
                    break
                losses = []
    if not EARLY_STOPPING:
        model.state_dict(destination=args_dic.model_file)
    print('Training over. To Testing...')
    evaluation(word2idx, model, test_data)
    print('OK .system finish.')


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


def evaluation(word2id, model, test_data):
    accuracy = 0
    for d in test_data:
        facts, facts_mask = pad_fact(d[0], word2id)
        question = d[1]
        question_mask = Variable(ByteTensor([0] * d[1].size(1)), volatile=False).unsqueeze(0)
        answer = d[2].squeeze(0)  # ??

        model.zero_grad()
        score = model([facts], [facts_mask], question, question_mask, num_decode=answer.size(0))

        if score.max(1)[1].data.tolist() == answer.data.tolist():
            accuracy += 1

    print(accuracy / len(test_data) * 100)


def train_from_model():
    print('Model init.')
    m = torch.load('earlystoping-EMNQA.model', map_location=lambda storage, loc: storage)
    word2idx = m['word2idx']
    model = DMN(HIDDEN_SIZE, len(word2idx), len(word2idx), word2idx)
    model.load_state_dict(state_dict=m['state_dict'])

    logger.info('Load from state dict over. Evaluation now')
    test_data = util.bAbI_data_load(args_dic.test_data_file)
    test_data = util.bAbI_data_test(test_data, word2idx)
    evaluation(word2idx, model, test_data=test_data)


if __name__ == '__main__':
    # data_file = 'qa5_three-arg-relations_train.txt'
    args = argparse.ArgumentParser()
    args.add_argument('--train-data-file', type=str, default='qa5_three-arg-relations_train.txt',
                      help='Input the train QA data')
    args.add_argument('--test-data-file', type=str, default='qa5_three-arg-relations_test.txt',
                      help='Input the test QA data')
    args.add_argument('--model-file', type=str, default='EMNQA.model',
                      help='Model file saved')

    args_dic = args.parse_args()
    data_file = args_dic.train_data_file

    logger.info('Use CUDA : %s' % USE_CUDA)
    if os.path.isfile('earlystoping-EMNQA.model'):
        logger.info("Find the model state dict . init model...")
        train_from_model(data_file)
    else:
        logger.info('No model state dict be Found .init model from scratch!')
        train_from_scratch(data_file)

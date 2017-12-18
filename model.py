import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable

USE_CUDA = torch.cuda.is_available()

FloatTensor = torch.cuda.FloatTensor if USE_CUDA else torch.FloatTensor
LongTensor = torch.cuda.LongTensor if USE_CUDA else torch.LongTensor
ByteTensor = torch.cuda.ByteTensor if USE_CUDA else torch.ByteTensor


class DMN(nn.Module):
    def __init__(self, hidden_size, input_size, output_size, word2index, dropout_p=0.1):
        super(DMN, self).__init__()

        self.hidden_size = hidden_size
        self.input_size = input_size
        self.output_size = output_size
        self.word2index = word2index

        self.embedding = nn.Embedding(input_size, hidden_size, padding_idx=0)

        self.input_gru = nn.GRU(hidden_size, hidden_size)

        self.question_gru = nn.GRU(hidden_size, hidden_size)

        self.drop_out = nn.Dropout(dropout_p)

        self.gate = nn.Sequential(
            nn.Linear(hidden_size * 4, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, 1),
            nn.Sigmoid()
        )

        self.atten_gru_cell = nn.GRUCell(hidden_size, hidden_size)

        self.memory_gru_cell = nn.GRUCell(hidden_size, hidden_size)

        self.answer_gru = nn.GRUCell(hidden_size * 2, hidden_size)

        self.fc = nn.Linear(hidden_size, output_size)

    def init_hidden(self, batch_size):
        hidden = Variable(torch.zeros(1, batch_size, self.hidden_size))
        return hidden.cuda() if torch.cuda.is_available() else hidden

    def init_weight(self):
        nn.init.xavier_uniform(self.embedding.state_dict()['weight'])

        for name, param in self.input_gru.state_dict().items():
            if 'weight' in name: nn.init.xavier_normal(param)
        for name, param in self.question_gru.state_dict().items():
            if 'weight' in name: nn.init.xavier_normal(param)
        for name, param in self.gate.state_dict().items():
            if 'weight' in name: nn.init.xavier_normal(param)
        for name, param in self.atten_gru_cell.state_dict().items():
            if 'weight' in name: nn.init.xavier_normal(param)
        for name, param in self.memory_gru_cell.state_dict().items():
            if 'weight' in name: nn.init.xavier_normal(param)
        for name, param in self.answer_gru.state_dict().items():
            if 'weight' in name: nn.init.xavier_normal(param)

        nn.init.xavier_normal(self.fc.state_dict()['weight'])
        self.fc.bias.data.fill_(0)

    def forward(self, facts, facts_mask, questions, question_masks, num_decode, episodes=3, is_training=False):
        '''
        for each batch contain number_of_facts , each facts had padded with length_of_each_fact
        here one batch equally one question

        :param facts:   B * number_of_facts * length_of_each_fact(padded) 64 * 36 * 7
        :param facts_mask:  B * number_of_facts * length_of_each_fact
        :param questions:   B * question_length
        :param question_masks:  B * question_length
        :param num_decode:
        :param episodes:
        :param is_training:
        :return:
        '''
        # Facts Module'''首先，对一个问题的所有fact，做represent，每一个事实被压缩表示成gru最后一个隐层的表示'''
        fact_represent = []
        for fact, fact_mask in zip(facts, facts_mask):  # for each batch = for each question  36 * 7
            facts_emb = self.embedding(fact)  # number_of_facts * length_of_each_fact(padded) * hidden_size

            if is_training:
                facts_emb = self.drop_out(facts_emb)

            fact_hidden = self.init_hidden(facts_emb.size(0))

            facts_emb = facts_emb.transpose(0, 1)  # length_of_each_fact(padded) * number_of_facts * hidden_size

            outputs, fact_hidden = self.input_gru(facts_emb, fact_hidden)

            f = []

            for i, v in enumerate(
                outputs.transpose(0,
                                  1)):  # here we use the lasted output to represent each fact in a word. (1 * hidden_size)
                o = v[fact_mask[i].data.tolist().count(0) - 1]
                f.append(o)

            fact_represent.append(
                torch.cat(f).view(fact.size(0), -1).unsqueeze(
                    0))  # here the dimension before cat should be 1 * num_of_facts * hidden_size

        fact_represent = torch.cat(fact_represent)  # batch * num_of_facts * hidden_size

        # Question Module
        # '''问题同fact一样表示成最后一个隐层'''
        ques_emb = self.embedding(questions)
        if is_training:
            ques_emb = self.drop_out(ques_emb)

        ques_hidden = self.init_hidden(ques_emb.size(0))
        ques_emb = ques_emb.transpose(0, 1)  # question_len * batch * hidden_size
        ques_outputs, ques_hidden = self.question_gru(ques_emb,
                                                      ques_hidden)  # the out should be (sqe_len * batch * hidden_size)

        ques_represents = []
        for ques_out, ques_mask in zip(ques_outputs.transpose(0, 1), question_masks):
            ques_len = ques_mask.data.tolist().count(0)

            ques_represents.append(ques_out[ques_len - 1].unsqueeze(0))

        ques_represents = torch.cat(ques_represents)  # batch * hidden_size

        # Episodic Memory Module
        '''周期记忆网络：1. attention.  2. memory
        对每一个问题，一次遍历它的所有相关事实，产生一个atten,具体构成
            1. 组装特征：q, f, memory。question作为初始的Memory
            2. 对每一个事实，使用双层的linear映射一个分数
            3. 把hidden和当前fact送进gru cell，使用公式计算计算新的hidden
            4. 迭代新的memory,此时memory表示的是当前episodes中各个fact对question的重要程度
        '''
        memory = ques_represents  # batch * hidden_size
        num_fact_each_batch = fact_represent.size(1)

        for i in range(episodes):
            hidden = self.init_hidden(fact_represent.size(0)).squeeze(0)

            # 对每一个事实，组一些特征，和memory还有question一起，做两层线性映射，产生ATTENTION
            # 如果一些数据集给定了问题对应的事实，那就可以进行对Attention模型进行有监督学习，可以使用cross-entropy作为目标函数。
            # Attention Mechannism Module
            for t in range(num_fact_each_batch):
                z = torch.cat(
                    [fact_represent.transpose(0, 1)[t] * ques_represents,  # batch * hidden_size
                     fact_represent.transpose(0, 1)[t] * memory,
                     torch.abs(fact_represent.transpose(0, 1)[t] - ques_represents),
                     torch.abs(fact_represent.transpose(0, 1)[t] - memory)], 1
                )  # here for each fact , dimension is : batch * (hidden * 4)

                atten_scores = self.gate(z)  # batch * 1

                # 这里的hidden相当于打包了一个问题的所有事实
                hidden = atten_scores * self.atten_gru_cell(fact_represent[:, t, :], hidden) + (
                                                                                                   1 - atten_scores) * hidden  # hidden is the fact after attened

            # Memory Update Mechanism
            # memory gru的作用是 记忆所有fact，以batch作为一个度。
            e = hidden

            memory = self.memory_gru_cell(e, memory)

        batch_size = fact_represent.size(0)

        # Answer Module
        '''回答模块：使用<s>作为初始输入，因为答案都是只有一个单词，所以gru的初始输入包含：[embed(<s>), question_represent]和上一个隐层输出
            可选择迭代几轮,每一论产生一个单词。
        '''
        answer_hidden = memory
        start_decode = Variable(LongTensor([[self.word2index['<s>']] * memory.size(0)])).transpose(0, 1)

        y_t_1 = self.embedding(start_decode).squeeze(1)

        decodes = []
        for t in range(num_decode):
            answer_hidden = self.answer_gru(torch.cat((y_t_1, ques_represents), 1),
                                            answer_hidden)  # batch * hidden_size
            decodes.append(
                F.log_softmax(self.fc(answer_hidden)))  # here should output dimension is : num_decode * output_size
            # decodes.append(
            #     F.softmax(self.fc(answer_hidden)))  # here should output dimension is : num_decode * output_size
        '''
        注意：之前使用softmax作为概率分布函数时，产生问题，在第三轮损失就不下降了。
        改为softmax之后，就没问题了。
        主要原因还是由于下溢，在第50次迭代中发现,softmax的值再e-5居多，而log后的大概还是-10到10之间
        '''
        return torch.cat(decodes, 1).view(batch_size * num_decode, -1)  # return each words probability

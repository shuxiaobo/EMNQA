from torch.utils.data import Dataset

from EMNQA import util


class QAdataset(Dataset):
    def __init__(self, data, word2id):
        self.data = data
        self.word2id = word2id

    def __getitem__(self, index):
        return util.prepare_sequence(self.data[index], self.word2id)

    def __len__(self):
        return len(self.data)

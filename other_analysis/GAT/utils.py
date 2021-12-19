import pickle

import numpy as np
import scipy
import torch
import pathlib
import dgl
import os


def parse_minibatch(adjlists_ua, user_artist_batch, device, offset=None):
    g_lists = [[], []]
    idx_batch_mapped_lists = [[], []]
    idx_node = [[], []]
    for mode, adjlists in enumerate(adjlists_ua):
        for adjlist in adjlists:
            # nodelist = [adjlist[row[mode]] for row in user_artist_batch]
            if mode == 0:
                nodelist = [adjlist[row[mode]] for row in user_artist_batch]
            else:
                nodelist = [adjlist[row[mode]] for row in user_artist_batch]
            edges = []
            nodes = set()
            for row in nodelist:
                row_parsed = np.asarray(list(map(int, row.split(' '))))
                if mode == 1:
                    row_parsed += offset
                nodes.add(row_parsed[0])
                if len(row_parsed) > 1:
                    neighbors = np.asarray(row_parsed[1:])
                else:
                    neighbors = np.asarray([row_parsed[0]])

                for dst in neighbors:
                    nodes.add(dst)
                    edges.append((row_parsed[0], dst))
            mapping = {map_from: map_to for map_to, map_from in enumerate(sorted(nodes))}
            edges = list(map(lambda tup: (mapping[tup[0]], mapping[tup[1]]), edges))
            g = dgl.DGLGraph()
            g.add_nodes(len(nodes))
            if len(edges) > 0:
                sorted_index = sorted(range(len(edges)), key=lambda i: edges[i])
                g.add_edges(*list(zip(*[(edges[i][1], edges[i][0]) for i in sorted_index])))
            g_lists[mode].append(g)
            idx_batch_mapped_lists[mode].append(np.array([mapping[row[mode]] if mode == 0 else mapping[row[mode]+offset]
                                                          for row in user_artist_batch]))
            idx_node[mode].append(torch.LongTensor(list(sorted(nodes))).to(device))

    return g_lists, idx_batch_mapped_lists, idx_node


def load_MDA_data(prefix='data/MDA'):
    infile = open(prefix + '/mirna_adjlist.txt', 'r')
    adjlist0 = [line.strip() for line in infile]
    adjlist0 = adjlist0
    infile.close()
    infile = open(prefix + '/disease_adjlist.txt', 'r')
    adjlist1 = [line.strip() for line in infile]
    adjlist1 = adjlist1
    infile.close()


    adjM = np.load(prefix + '/adjMDA.npy')
    type_mask = np.load(prefix + '/node_type_MDA.npy')
    train_val_test_pos_gene_dis = np.load(prefix + '/train_val_test_pos_gene_dis.npz')
    train_val_test_neg_gene_dis = np.load(prefix + '/train_val_test_neg_gene_dis.npz')

    return [[adjlist0], [adjlist1]], \
           adjM, type_mask, train_val_test_pos_gene_dis, train_val_test_neg_gene_dis

def load_BioNet_data(prefix='data/preprocessed'):
    infile = open(prefix + '/0/0-1-0.adjlist', 'r')
    adjlist00 = [line.strip() for line in infile]
    adjlist00 = adjlist00
    infile.close()
    infile = open(prefix + '/0/0-2-0.adjlist', 'r')
    adjlist01 = [line.strip() for line in infile]
    adjlist01 = adjlist01
    infile.close()
    infile = open(prefix + '/0/0-0.adjlist', 'r')
    adjlist02 = [line.strip() for line in infile]
    adjlist02 = adjlist02
    infile.close()
    infile = open(prefix + '/1/1-0-1.adjlist', 'r')
    adjlist10 = [line.strip() for line in infile]
    adjlist10 = adjlist10
    infile.close()
    infile = open(prefix + '/1/1-0-0-1.adjlist', 'r')
    adjlist11 = [line.strip() for line in infile]
    adjlist11 = adjlist11
    infile.close()
    infile = open(prefix + '/1/1-3-1.adjlist', 'r')
    adjlist12 = [line.strip() for line in infile]
    adjlist12 = adjlist12
    infile.close()

    # in_file = open(prefix + '/0/0-1-0_idx.pickle', 'rb')
    # idx00 = pickle.load(in_file)
    # in_file.close()
    # in_file = open(prefix + '/0/0-2-0_idx.pickle', 'rb')
    # idx01 = pickle.load(in_file)
    # in_file.close()
    # in_file = open(prefix + '/0/0-0_idx.pickle', 'rb')
    # idx02 = pickle.load(in_file)
    # in_file.close()
    # in_file = open(prefix + '/1/1-0-1_idx.pickle', 'rb')
    # idx10 = pickle.load(in_file)
    # in_file.close()
    # in_file = open(prefix + '/1/1-0-0-1_idx.pickle', 'rb')
    # idx11 = pickle.load(in_file)
    # in_file.close()
    # in_file = open(prefix + '/1/1-3-1_idx.pickle', 'rb')
    # idx12 = pickle.load(in_file)
    # in_file.close()

    adjM = scipy.sparse.load_npz(prefix + '/adjM.npz')
    type_mask = np.load(prefix + '/node_types.npy')
    train_val_test_pos_gene_dis = np.load(prefix + '/train_val_test_pos_gene_dis.npz')
    train_val_test_neg_gene_dis = np.load(prefix + '/train_val_test_neg_gene_dis.npz')

    return [[adjlist00, adjlist01, adjlist02], [adjlist10, adjlist11, adjlist12]], \
           adjM, type_mask, train_val_test_pos_gene_dis, train_val_test_neg_gene_dis
           # [[idx00, idx01, idx02], [idx10, idx11, idx12]], \


class EarlyStopping:
    """Early stops the training if validation loss doesn't improve after a given patience."""
    def __init__(self, patience, verbose=False, delta=0, save_path='checkpoint.pt'):
        """
        Args:
            patience (int): How long to wait after last time validation loss improved.
                            Default: 7
            verbose (bool): If True, prints a message for each validation loss improvement.
                            Default: False
            delta (float): Minimum change in the monitored quantity to qualify as an improvement.
                            Default: 0
        """
        self.patience = patience
        self.verbose = verbose
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.val_loss_min = np.Inf
        self.delta = delta
        self.save_path = save_path
        pathlib.Path(os.path.dirname(save_path)).mkdir(parents=True, exist_ok=True)

    def __call__(self, val_loss, model):

        score = -val_loss

        if self.best_score is None:
            self.best_score = score
            self.save_checkpoint(val_loss, model)
        elif score < self.best_score - self.delta:
            self.counter += 1
            print(f'EarlyStopping counter: {self.counter} out of {self.patience}')
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            self.save_checkpoint(val_loss, model)
            self.counter = 0

    def save_checkpoint(self, val_loss, model):
        """Saves model when validation loss decrease."""
        if self.verbose:
            print(f'Validation loss decreased ({self.val_loss_min:.6f} --> {val_loss:.6f}).  Saving model ...')
        torch.save(model.state_dict(), self.save_path)
        self.val_loss_min = val_loss

class index_generator:
    def __init__(self, batch_size, num_data=None, indices=None, shuffle=True):
        if num_data is not None:
            self.num_data = num_data
            self.indices = np.arange(num_data)
        if indices is not None:
            self.num_data = len(indices)
            self.indices = np.copy(indices)
        self.batch_size = batch_size
        self.iter_counter = 0
        self.shuffle = shuffle
        if shuffle:
            np.random.shuffle(self.indices)

    def next(self):
        if self.num_iterations_left() <= 0:
            self.reset()
        self.iter_counter += 1
        return np.copy(self.indices[(self.iter_counter - 1) * self.batch_size:self.iter_counter * self.batch_size])

    def num_iterations(self):
        return int(np.ceil(self.num_data / self.batch_size))

    def num_iterations_left(self):
        return self.num_iterations() - self.iter_counter

    def reset(self):
        if self.shuffle:
            np.random.shuffle(self.indices)
        self.iter_counter = 0
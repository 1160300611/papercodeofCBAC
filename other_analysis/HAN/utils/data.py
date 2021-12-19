import pickle
import scipy
import numpy as np


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

    return [[adjlist01, adjlist02], [adjlist10, adjlist12]], \
           adjM, type_mask, train_val_test_pos_gene_dis, train_val_test_neg_gene_dis
    # [[idx00, idx01, idx02], [idx10, idx11, idx12]],

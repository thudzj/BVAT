import numpy as np
import pickle as pkl
import networkx as nx
import scipy.sparse as sp
from scipy.sparse.linalg.eigen.arpack import eigsh
import sys
import tensorflow as tf
import os
import time
import json
from networkx.readwrite import json_graph
from sklearn.metrics import f1_score
import multiprocessing

def parse_index_file(filename):
    """Parse index file."""
    index = []
    for line in open(filename):
        index.append(int(line.strip()))
    return index


def sample_mask(idx, l):
    """Create mask."""
    mask = np.zeros(l)
    mask[idx] = 1
    return np.array(mask, dtype=np.bool)

def save_sparse_csr(filename,array):
    np.savez(filename,data = array.data ,indices=array.indices,
             indptr =array.indptr, shape=array.shape )

def load_sparse_csr(filename):
    loader = np.load(filename)
    return sp.csr_matrix((  loader['data'], loader['indices'], loader['indptr']),
                         shape = loader['shape'])

def starfind_4o_nbrs(args):
    return find_4o_nbrs(*args)

def find_4o_nbrs(adj, li):
    nbrs = []
    for i in li:
        print(i)
        tmp = adj[i]
        for ii in np.nonzero(adj[i])[1]:
            tmp += adj[ii]
            for iii in np.nonzero(adj[ii])[1]:
                tmp += adj[iii]
                tmp += adj[np.nonzero(adj[iii])[1]].sum(0)
        nbrs.append(np.nonzero(tmp)[1])
    print(li)
    return nbrs

def load_data(dataset_str, is_sparse):
    if dataset_str == "ppi":
        return load_graphsage_data('data/ppi/ppi', is_sparse)
    """Load data."""
    if dataset_str != 'nell':
        names = ['x', 'y', 'tx', 'ty', 'allx', 'ally', 'graph']
        objects = []
        for i in range(len(names)):
            with open("data/ind.{}.{}".format(dataset_str, names[i]), 'rb') as f:
                if sys.version_info > (3, 0):
                    objects.append(pkl.load(f, encoding='latin1'))
                else:
                    objects.append(pkl.load(f))

        x, y, tx, ty, allx, ally, graph = tuple(objects)

        test_idx_reorder = parse_index_file("data/ind.{}.test.index".format(dataset_str))
        test_idx_range = np.sort(test_idx_reorder)

        if dataset_str == 'citeseer':
            # Fix citeseer dataset (there are some isolated nodes in the graph)
            # Find isolated nodes, add them as zero-vecs into the right position
            test_idx_range_full = range(min(test_idx_reorder), max(test_idx_reorder)+1)
            tx_extended = sp.lil_matrix((len(test_idx_range_full), x.shape[1]))
            tx_extended[test_idx_range-min(test_idx_range), :] = tx
            tx = tx_extended
            ty_extended = np.zeros((len(test_idx_range_full), y.shape[1]))
            ty_extended[test_idx_range-min(test_idx_range), :] = ty
            ty = ty_extended

        features = sp.vstack((allx, tx)).tolil()
        features[test_idx_reorder, :] = features[test_idx_range, :]
        features = preprocess_features(features, is_sparse)
        adj = nx.adjacency_matrix(nx.from_dict_of_lists(graph))
        support = preprocess_adj(adj)

        labels = np.vstack((ally, ty))
        labels[test_idx_reorder, :] = labels[test_idx_range, :]

        idx_test = test_idx_range.tolist()
        idx_train = range(len(y))
        idx_val = range(len(y), len(y)+500)

        train_mask = sample_mask(idx_train, labels.shape[0])
        val_mask = sample_mask(idx_val, labels.shape[0])
        test_mask = sample_mask(idx_test, labels.shape[0])

        # y_train = np.zeros(labels.shape)
        # y_val = np.zeros(labels.shape)
        # y_test = np.zeros(labels.shape)
        # y_train = labels[train_mask, :]
        # y_val[val_mask, :] = labels[val_mask, :]
        # y_test[test_mask, :] = labels[test_mask, :]
    else:
        names = ['x', 'y', 'tx', 'ty', 'allx', 'ally', 'graph']
        objects = []
        for i in range(len(names)):
            with open("data/savedData/{}.{}".format(dataset_str, names[i]), 'rb') as f:
                if sys.version_info > (3, 0):
                    objects.append(pkl.load(f, encoding='latin1'))
                else:
                    objects.append(pkl.load(f))

        x, y, tx, ty, allx, ally, graph = tuple(objects)

        test_idx_reorder = parse_index_file("data/savedData/{}.test.index".format(dataset_str))
        features = allx.tolil()
        adj = nx.adjacency_matrix(nx.from_dict_of_lists(graph))
        labels = ally
        features = preprocess_features(features, is_sparse)
        support = preprocess_adj(adj)
        idx_test = test_idx_reorder
        idx_train = range(len(y))
        idx_val = range(len(y), len(y)+969)
        train_mask = sample_mask(idx_train, labels.shape[0])
        val_mask = sample_mask(idx_val, labels.shape[0])
        test_mask = sample_mask(idx_test, labels.shape[0])

    # if not os.path.isfile("data/{}.nbrs.npz".format(dataset_str)):
    #     N = adj.shape[0]
    #     pool = multiprocessing.Pool(processes=56)
    #
    #     lis = []
    #     for i in range(32):
    #         li = range(int(N/32)*i, max(int(N/32)*(i+1), N))
    #         lis.append(li)
    #     adjs = [adj] * 32
    #     results = pool.map(starfind_4o_nbrs, zip(adjs, lis))
    #
    #     pool.close()
    #     pool.join()
    #     nbrs = results[0]
    #     # cnt = 0
    #     # for i in range(32):
    #     #
    #     #     cnt += len(results[i])
    #     #     print(cnt)
    #     #     nbrs += results[i]
    #
    #     np.savez("data/{}.nbrs.npz".format(dataset_str), data = nbrs)
    # else:
    #     loader = np.load("data/{}.nbrs.npz".format(dataset_str))
    #     nbrs = loader['data']
    print(adj.shape)
    return None, support, support, features, labels, train_mask, val_mask, test_mask

def sparse_to_tuple(sparse_mx):
    """Convert sparse matrix to tuple representation."""
    def to_tuple(mx):
        if not sp.isspmatrix_coo(mx):
            mx = mx.tocoo()
        coords = np.vstack((mx.row, mx.col)).transpose()
        values = mx.data
        shape = mx.shape
        return coords, values, shape

    if isinstance(sparse_mx, list):
        for i in range(len(sparse_mx)):
            sparse_mx[i] = to_tuple(sparse_mx[i])
    else:
        sparse_mx = to_tuple(sparse_mx)

    return sparse_mx


def preprocess_features(features, sparse=True):
    """Row-normalize feature matrix and convert to tuple representation"""
    rowsum = np.array(features.sum(1))
    r_inv = np.power(rowsum, -1).flatten()
    r_inv[np.isinf(r_inv)] = 0.
    r_mat_inv = sp.diags(r_inv)
    features = r_mat_inv.dot(features)
    if sparse:
        return sparse_to_tuple(features)
    else:
        return features.toarray()


def normalize_adj(adj):
    """Symmetrically normalize adjacency matrix."""
    adj = sp.coo_matrix(adj)
    rowsum = np.array(adj.sum(1))
    d_inv_sqrt = np.power(rowsum, -0.5).flatten()
    d_inv_sqrt[np.isinf(d_inv_sqrt)] = 0.
    d_mat_inv_sqrt = sp.diags(d_inv_sqrt)
    return adj.dot(d_mat_inv_sqrt).transpose().dot(d_mat_inv_sqrt).tocoo()


def preprocess_adj(adj):
    """Preprocessing of adjacency matrix for simple GCN model and conversion to tuple representation."""
    adj_normalized = normalize_adj(adj + sp.eye(adj.shape[0]))
    return sparse_to_tuple(adj_normalized)


def construct_feed_dict(features, support, labels, labels_mask, placeholders, nbrs):
    """Construct feed dictionary."""
    feed_dict = dict()
    feed_dict.update({placeholders['labels']: labels})
    feed_dict.update({placeholders['labels_mask']: labels_mask})
    feed_dict.update({placeholders['features']: features})
    feed_dict.update({placeholders['support']: support})
    feed_dict.update({placeholders['num_features_nonzero']: features[1].shape})
    # r1 = sample_nodes(nbrs)
    # feed_dict.update({placeholders['adv_mask1']: r1})
    #feed_dict.update({placeholders['adv_mask2']: r2})
    #feed_dict.update({placeholders['norm_mtx']: r3})
    return feed_dict


def chebyshev_polynomials(adj, k):
    """Calculate Chebyshev polynomials up to order k. Return a list of sparse matrices (tuple representation)."""
    print("Calculating Chebyshev polynomials up to order {}...".format(k))

    adj_normalized = normalize_adj(adj)
    laplacian = sp.eye(adj.shape[0]) - adj_normalized
    largest_eigval, _ = eigsh(laplacian, 1, which='LM')
    scaled_laplacian = (2. / largest_eigval[0]) * laplacian - sp.eye(adj.shape[0])

    t_k = list()
    t_k.append(sp.eye(adj.shape[0]))
    t_k.append(scaled_laplacian)

    def chebyshev_recurrence(t_k_minus_one, t_k_minus_two, scaled_lap):
        s_lap = sp.csr_matrix(scaled_lap, copy=True)
        return 2 * s_lap.dot(t_k_minus_one) - t_k_minus_two

    for i in range(2, k+1):
        t_k.append(chebyshev_recurrence(t_k[-1], t_k[-2], scaled_laplacian))

    return sparse_to_tuple(t_k)


def sample_nodes(nbrs, num=100):
    N = len(nbrs)
    flag = np.zeros([N])
    output = [0] * num
    #norm_mtx = np.zeros([N, N])
    for i in range(num):
        a = np.random.randint(0, N)
        while flag[a] == 1:
            a = np.random.randint(0, N)
        output[i] = a
        # for nell to speed up
        #flag[nbrs[a]] = 1

        # tmp = np.zeros([N])
        # tmp[nbrs[a]] = 1
        #norm_mtx[nbrs[a]] = tmp
    # output_ = np.ones([N])
    # output_[output] = 0
    # output_ = np.nonzero(output_)[0]
    return sample_mask(output, N)#, norm_mtx

def kl_divergence_with_logit(q_logit, p_logit, mask=None):
    q = tf.nn.softmax(q_logit)
    if not mask is None:
        mask = tf.cast(mask, dtype=tf.float32)
        mask /= tf.reduce_mean(mask)
        qlogq = tf.reduce_mean(tf.reduce_sum(q * tf.nn.log_softmax(q_logit), 1) * mask)
        qlogp = tf.reduce_mean(tf.reduce_sum(q * tf.nn.log_softmax(p_logit), 1) * mask)
        return  - qlogp
    else:
        qlogq = tf.reduce_sum(q * tf.nn.log_softmax(q_logit), 1)
        qlogp = tf.reduce_sum(q * tf.nn.log_softmax(p_logit), 1)
        return tf.reduce_mean( - qlogp)


def entropy_y_x(logit):
    p = tf.nn.softmax(logit)
    return -tf.reduce_mean(tf.reduce_sum(p * tf.nn.log_softmax(logit), 1))

def get_normalized_vector(d, sparse=False, indices=None, dense_shape=None):
    if sparse:
        d /= (1e-12 + tf.reduce_max(tf.abs(d)))
        d2 = tf.SparseTensor(indices, tf.square(d), dense_shape)
        d = tf.SparseTensor(indices, d, dense_shape)
        d /= tf.sqrt(1e-6 + tf.sparse_reduce_sum(d2, 1, keep_dims=True))
        return d
    else:
        d /= (1e-12 + tf.reduce_max(tf.abs(d)))

        d /= tf.sqrt(1e-6 + tf.reduce_sum(tf.pow(d, 2.0), 1, keepdims=True))
        return d

def get_normalized_matrix(d, sparse=False, indices=None, dense_shape=None):
    if not sparse:
        return tf.nn.l2_normalize(d, [0,1])
    else:
        return tf.SparseTensor(indices, tf.nn.l2_normalize(d, [0]), dense_shape)

def calc_f1(y_pred, y_true, multitask):
    if multitask:
        y_pred[y_pred>0] = 1
        y_pred[y_pred<=0] = 0
    else:
        y_true = np.argmax(y_true, axis=1)
        y_pred = np.argmax(y_pred, axis=1)
    return f1_score(y_true, y_pred, average="micro"), \
           f1_score(y_true, y_pred, average="macro")

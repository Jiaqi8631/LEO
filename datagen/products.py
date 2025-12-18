import os
import numpy as np
import pandas as pd
from scipy.sparse import coo_matrix, csr_matrix
import json, torch

DOWNLOAD_URL = 'http://snap.stanford.edu/ogb/data/nodeproppred/products.zip'

CURRENT_WORK_DIR = os.getcwd()
RAW_DATA_DIR = '../datasets/data-raw'
PRODUCTS_RAW_DATA_DIR = f'{RAW_DATA_DIR}/products'
OUTPUT_DATA_DIR = '../datasets/data-output'
PRODUCTS_OUTPUT_DATA_DIR = f'{OUTPUT_DATA_DIR}/products'


def download_data():
    print('Download data...')
    if not os.path.exists(f'{RAW_DATA_DIR}/products.zip'):
        print('Start downloading...')
        assert(os.system(f'wget {DOWNLOAD_URL} -O {RAW_DATA_DIR}/products.zip') == 0)
    else:
        print('Already downloaded.')

    print('Unzip data...')
    if not os.path.exists(f'{PRODUCTS_RAW_DATA_DIR}/unzipped'):
        assert(os.system(f'cd {RAW_DATA_DIR}; unzip products.zip') == 0)
        assert(os.system(f'touch {PRODUCTS_RAW_DATA_DIR}/unzipped') == 0)
    else:
        print('Already unzipped...') 

def gen_topo_wholegraph():
    assert(os.system(f'cd {CURRENT_WORK_DIR}') == 0)
    print('Reading raw graph topo...')
    edges = pd.read_csv(f'{PRODUCTS_RAW_DATA_DIR}/raw/edge.csv.gz', compression='gzip', header=None).values.T
    num_nodes = pd.read_csv(f'{PRODUCTS_RAW_DATA_DIR}/raw/num-node-list.csv.gz', compression='gzip', header=None).values[0][0]

    src = edges[0]
    dst = edges[1]
    data = np.ones(src.shape)
    A = coo_matrix((data, (dst, src)), shape=(num_nodes, num_nodes), dtype=np.uint32)
    csr = A.tocsr()
    
    print('Generating CSR...')
    indptr = csr.indptr
    indices = csr.indices

    num_nodes = indptr.shape[0] - 1
    num_edges = indices.shape[0]
    # data = np.ones(num_edges)
   
    # csr = csr_matrix((data, indices, indptr), shape=(num_nodes, num_nodes))
    # csr += csr.transpose(copy=True)
    # indptr = csr.indptr
    # indices = csr.indices

    #A = csr_matrix((data, (dst, src)), shape=(num_nodes, num_nodes), dtype=np.uint32)
    A_T = A.transpose().tocsr()
    indptr_T = A_T.indptr
    indices_T = A_T.indices
    del src, dst, edges, A

    print('Writing CSR...')
    indptr.astype('uint32').tofile(f'{PRODUCTS_OUTPUT_DATA_DIR}/indptr.bin')
    indices.astype('uint32').tofile(f'{PRODUCTS_OUTPUT_DATA_DIR}/indices.bin')
    indptr_T.astype('uint32').tofile(f'{PRODUCTS_OUTPUT_DATA_DIR}/indptr_T.bin')
    indices_T.astype('uint32').tofile(f'{PRODUCTS_OUTPUT_DATA_DIR}/indices_T.bin')

def gen_feat_wholegraph():
    assert(os.system(f'cd {CURRENT_WORK_DIR}') == 0)
    print('Reading features...')
    feat = pd.read_csv(f'{PRODUCTS_RAW_DATA_DIR}/raw/node-feat.csv.gz', compression='gzip', header=None).values

    print('Writing features')
    feat.astype('float32').tofile(f'{PRODUCTS_OUTPUT_DATA_DIR}/feat.bin')

def gen_spilt_feat_label_wholegraph():
    assert(os.system(f'cd {CURRENT_WORK_DIR}') == 0)
    print('Reading spilt raw data and labels...')

    train_idx = pd.read_csv(f'{PRODUCTS_RAW_DATA_DIR}/split/sales_ranking/train.csv.gz', compression='gzip', header=None).values.T[0]
    valid_idx = pd.read_csv(f'{PRODUCTS_RAW_DATA_DIR}/split/sales_ranking/valid.csv.gz', compression='gzip', header=None).values.T[0]
    test_idx  = pd.read_csv(f'{PRODUCTS_RAW_DATA_DIR}/split/sales_ranking/test.csv.gz',  compression='gzip', header=None).values.T[0]

    label = pd.read_csv(f'{PRODUCTS_RAW_DATA_DIR}/raw/node-label.csv.gz', compression='gzip', header=None).values.T[0]

    print('Writing spilt raw data and labels...')

    train_idx.astype('uint32').tofile(f'{PRODUCTS_OUTPUT_DATA_DIR}/train_set.bin')
    valid_idx.astype('uint32').tofile(f'{PRODUCTS_OUTPUT_DATA_DIR}/valid_set.bin')
    test_idx.astype('uint32').tofile(f'{PRODUCTS_OUTPUT_DATA_DIR}/test_set.bin')

    label.astype('uint32').tofile(f'{PRODUCTS_OUTPUT_DATA_DIR}/label.bin')

if __name__ == '__main__':
    assert(os.system(f'mkdir -p {PRODUCTS_RAW_DATA_DIR}') == 0)
    assert(os.system(f'mkdir -p {PRODUCTS_OUTPUT_DATA_DIR}') == 0)

    download_data() 
    print(CURRENT_WORK_DIR)
    gen_topo_wholegraph()
    gen_feat_wholegraph()  
    gen_spilt_feat_label_wholegraph()


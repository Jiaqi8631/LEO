import os
import numpy as np
import pandas as pd
from scipy.sparse import coo_matrix, csr_matrix
import json, torch

DOWNLOAD_URL = 'http://snap.stanford.edu/ogb/data/nodeproppred/papers100M-bin.zip'

CURRENT_WORK_DIR = os.getcwd()
RAW_DATA_DIR = '../datasets/data-raw'
PAPERS_RAW_DATA_DIR = f'{RAW_DATA_DIR}/papers100M-bin'
OUTPUT_DATA_DIR = '../datasets/data-output'
PAPERS_OUTPUT_DATA_DIR = f'{OUTPUT_DATA_DIR}/papers100M'

def download_data():
    print('Download data...')
    if not os.path.exists(f'{RAW_DATA_DIR}/papers100M-bin.zip'):
        print('Start downloading...')
        assert(os.system(f'wget {DOWNLOAD_URL} -O {RAW_DATA_DIR}/papers100M-bin.zip') == 0)
    else:
        print('Already downloaded.') 

    print('Unzip data...')
    if not os.path.exists(f'{PAPERS_RAW_DATA_DIR}/unzipped'):
        assert(os.system(f'cd {RAW_DATA_DIR}; unzip papers100M-bin.zip') == 0)
        assert(os.system(f'touch {PAPERS_RAW_DATA_DIR}/unzipped') == 0)
    else:
        print('Already unzipped...')

def gen_topo_wholegraph():
    assert(os.system(f'cd {CURRENT_WORK_DIR}') == 0)
    print('Reading raw graph topo...')
    file = np.load(f'{PAPERS_RAW_DATA_DIR}/raw/data.npz', mmap_mode='r')
    num_nodes = file['num_nodes_list'][0]

    edge_index = file['edge_index']
    src = edge_index[0]
    dst = edge_index[1]
    data = np.zeros(src.shape) 
    A = coo_matrix((data, (dst, src)), shape=(num_nodes, num_nodes), dtype=np.uint32) 
    csr = A.tocsr()

    print('Generating CSR...')
    indptr = csr.indptr
    indices = csr.indices

    num_nodes = indptr.shape[0] - 1
    num_edges = indices.shape[0]

    print('Converting topo...')
    A_T = A.transpose().tocsr()
    indptr_T = A_T.indptr
    indices_T = A_T.indices
    del src, dst, edge_index, A

    print('Writing CSR...')
    indptr.astype('uint32').tofile(f'{PAPERS_OUTPUT_DATA_DIR}/indptr.bin')
    indices.astype('uint32').tofile(f'{PAPERS_OUTPUT_DATA_DIR}/indices.bin')
    indptr_T.astype('uint32').tofile(f'{PAPERS_OUTPUT_DATA_DIR}/indptr_T.bin')
    indices_T.astype('uint32').tofile(f'{PAPERS_OUTPUT_DATA_DIR}/indices_T.bin')      

def gen_feat_wholegraph():
    assert(os.system(f'cd {CURRENT_WORK_DIR}') == 0)
    print('Reading features...')
    file = np.load(f'{PAPERS_RAW_DATA_DIR}/raw/data.npz', mmap_mode='r')
    feat = file['node_feat']

    print('Writing featrues...')
    feat.astype('float32').tofile(f'{PAPERS_OUTPUT_DATA_DIR}/feat.bin')

def gen_spilt_feat_label_wholegraph():
    assert(os.system(f'cd {CURRENT_WORK_DIR}') == 0)
    print('Reading spilt raw data and labels...')

    train_idx = pd.read_csv(f'{PAPERS_RAW_DATA_DIR}/split/time/train.csv.gz', compression='gzip', header=None).values.T[0]
    valid_idx = pd.read_csv(f'{PAPERS_RAW_DATA_DIR}/split/time/valid.csv.gz', compression='gzip', header=None).values.T[0]
    test_idx  = pd.read_csv(f'{PAPERS_RAW_DATA_DIR}/split/time/test.csv.gz',  compression='gzip', header=None).values.T[0]

    file_label = np.load(f'{PAPERS_RAW_DATA_DIR}/raw/node-label.npz')
    label = file_label['node_label']

    print('Writing spilt raw data and labels...')

    train_idx.astype('uint32').tofile(f'{PAPERS_OUTPUT_DATA_DIR}/train_set.bin')
    valid_idx.astype('uint32').tofile(f'{PAPERS_OUTPUT_DATA_DIR}/valid_set.bin')
    test_idx.astype('uint32').tofile(f'{PAPERS_OUTPUT_DATA_DIR}/test_set.bin')

    label.astype('uint32').tofile(f'{PAPERS_OUTPUT_DATA_DIR}/label.bin')   
    

if __name__ == '__main__':
    assert(os.system(f'mkdir -p {PAPERS_RAW_DATA_DIR}') == 0)
    assert(os.system(f'mkdir -p {PAPERS_OUTPUT_DATA_DIR}') == 0)

    download_data() 
    print(CURRENT_WORK_DIR) 
    #gen_topo_wholegraph() 
    #gen_feat_wholegraph() 
    gen_spilt_feat_label_wholegraph()
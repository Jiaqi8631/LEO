
#include <iostream>
#include <stdio.h>
#include <ctime>
#include <algorithm>

#include "graph.h"
#include <vector>
// #include "utils.cuh"
// #include "neighbor_utils.cuh"
// #include "csr_formatter.h"
// #include "layer.h"



//#define validate 1 // the number (< num_GPUs) indicates the validation on which PE.
// using nidType = size_t;
// using nidType = long;
using nidType = int;

//using namespace cudl;
using namespace std;

int main(int argc, char* argv[]){

    if (argc < 4){
        printf("Usage: ./Edgelist beg_file csr_file weight_file dataset\n");
        return -1;
    }
	    
    cout << "Graph File: " << argv[1] << '\n';
    const char *beg_file = argv[1];
	const char *csr_file = argv[2];
	const char *weight_file = argv[3];
    const char *dataset = argv[4];

    string name;
    name = dataset;
    string txt = ".txt";
    name += txt;
    cout << "filename: " << name << endl;
    const char *filename = name.data();

    graph<long, long, nidType, nidType, nidType, nidType>* ginst = new graph<long, long, nidType, nidType, nidType, nidType>(beg_file, csr_file, weight_file);
    std::vector<nidType> global_row_ptr(ginst->beg_pos, ginst->beg_pos + ginst->vert_count + 1);
    std::vector<nidType> global_col_ind(ginst->csr, ginst->csr + ginst->edge_count);

    cout << "Complete loading graphs !!" << endl;
    nidType numNodes = global_row_ptr.size() - 1;
    nidType numEdges = global_col_ind.size();    

    FILE *fp;
    if((fp = fopen(filename, "wb")) == NULL){
        cout << "Can't open the file!" << endl;
        exit(0);
    }

    for (int i = 0; i < numNodes; i++){
        nidType eid_s = global_row_ptr[i];
        nidType eid_e = global_row_ptr[i + 1];

        for (int j = eid_s; j < eid_e; j++){
            nidType col_id = global_col_ind[j];
            fprintf(fp, "%d %d\n", i, col_id);
        }
    }
    fclose(fp);
    return 0;
}

# GRNMF-A Graph Regularized Generalized Matrix Factorization Model for Predicting Links in Biomedical Bipartite Networks


## Guidance

1. The following packages are needed:

   numpy, pandas, scikit-learn

2. The dataset should be placed in the './data/' containing 3 file for a single dataset:
   "datasetname_int.txt": the association matrix for two different types of bioentities A and B
   "datasetname_A_sim.txt": the similarity matrix of A
   "datasetname_B_sim.txt": the similarity matrix of B
   *Note:* all of the three file should be \t delimited text file. Please refer to the example dataset in ./data for more details.

3. To run the code, you should specify the following parameters:

   ```
   --dataset: 						specify the dataset, e.g., ic
   --method-opt (optional):		set the hyper parameters for GRGMF, use the default parameters if not specify
   
   here is the example:
   python predict.py --dataset="ic"  --method-opt="max_iter=100 lr=0.1 beta=4 lamb=0.0333 r1=0.5 r2=1 mf_dim=150 K=5"
   ```

   





--------
The code is written by:

Zichao Zhang

E-mail: gdzzc96[AT]gmail.com

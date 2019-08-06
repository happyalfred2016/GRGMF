# GRGMF-A Graph Regularized Generalized Matrix Factorization Model for Predicting Links in Biomedical Bipartite Networks

1. This code have been tested under python3.6, and the following packages are needed:

   numpy, pandas

2. The datasets should be placed in the "./data/" which includes 3 file for a single dataset:
   "datasetname_int.txt": the association matrix for two different types of bioentities A and B
   "datasetname_A_sim.txt": the similarity matrix of A
   "datasetname_B_sim.txt": the similarity matrix of B
   *Note:* All of the three files should be \t delimited pure text files. Please refer to the example dataset in "./data" for more details.

3. To run the code, you should specify the following parameters:

   ```
   --dataset: 						specify the dataset, e.g., ic
   --method-opt (optional):		set the hyper parameters for GRGMF, use the default values if not specified
   
   Here is an example:
   python predict.py --dataset="ic"  --method-opt="max_iter=100 lr=0.1 beta=4 lamb=0.0333 r1=0.5 r2=1 mf_dim=150 K=5"
   ```

4. The predicted result will be stored in "./output/".



--------
If you use this code, please cite our paper: “A Graph Regularized Generalized Matrix Factorization Model for Predicting Links in Biomedical Bipartite Networks”

The code is written by:

Zichao Zhang

E-mail: gdzzc96[AT]gmail.com

Feel free to contact the author for any questions regarding to this code.
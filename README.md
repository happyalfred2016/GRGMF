# GRGMF: A Graph Regularized Generalized Matrix Factorization Model for Predicting Links in Biomedical Bipartite Networks
This code is the implementation of GRGMF, which is both CPU and CUDA compatible(CUDA is preferred).
1. This code has been tested under:
     python=3.6
     numpy=1.16.4
     pandas=0.25.3
     pytorch=1.2.0

2. The datasets should be placed in the "./data/" which includes 3 files for one dataset:
   "datasetname_int.txt":  the biadjacency  matrix for nodes in two disjoint sets of nodes A and B
   "datasetname_A_sim.txt":  the similarity matrix of nodes in A
   "datasetname_B_sim.txt":  the similarity matrix of nodes in B
   *Note:* All of the three files should be \t delimited pure text files. Please refer to the example dataset in "./data" for more details.

3. To run the code, you should specify the following parameters:

   ```
   --dataset:                   specify the dataset, e.g., ic
   --method-opt (optional):     set the hyper parameters for GRGMF, and use the default values if not specified
   
   # Here is an example:
   python predict.py --dataset="ic"  --method-opt="max_iter=100 lr=0.1 beta=4 lamb=0.0333 r1=0.5 r2=1 K=50 k=5"
   ```

4. The predicted result will be stored in "./output/".



--------
If you use this code, please cite our paper: 
```bibtex
@article{10.1093/bioinformatics/btaa157,
    author = {Zhang, Zi-Chao and Zhang, Xiao-Fei and Wu, Min and Ou-Yang, Le and Zhao, Xing-Ming and Li, Xiao-Li},
    title = "{A Graph Regularized Generalized Matrix Factorization Model for Predicting Links in Biomedical Bipartite Networks}",
    journal = {Bioinformatics},
    year = {2020},
    month = {03},
    issn = {1367-4803},
    doi = {10.1093/bioinformatics/btaa157},
    url = {https://doi.org/10.1093/bioinformatics/btaa157},
}
```



Feel free to contact the author for any questions regarding this code(E-mail: zczhang24[at]gmail[dot]com).
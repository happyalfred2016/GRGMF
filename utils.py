import os
import numpy as np


def load_data_from_file(dataset, folder):
    def load_rating_file_as_matrix(filename):
        data = []
        stat = []
        try:
            with open(filename, "r") as f:
                line = f.readline()
                while line != None and line != "":
                    arr = np.array(line.strip().split("\t"), dtype='int')
                    stat.append(sum(arr))
                    data.append(arr)
                    line = f.readline()
        except:
            with open(filename, "r") as f:
                line = f.readline()
                while line != None and line != "":
                    arr = line.strip().split("\t")
                    try:
                        arr = arr[1:]
                        arr = np.array(arr, dtype='int')
                        stat.append(sum(arr))
                        data.append(arr)
                    except:
                        pass
                    line = f.readline()

        # Construct matrix
        mat = np.array(data)
        return mat

    def load_matrix(filename):
        data = []
        # for the situation that data files contain col/row name
        try:
            with open(filename, "r") as f:
                line = f.readline()
                while line != None and line != "":
                    arr = np.array(line.strip().split("\t"), dtype='float')
                    data.append(arr)
                    line = f.readline()
        except:
            with open(filename, "r") as f:
                line = f.readline()
                while line != None and line != "":
                    arr = line.strip().split("\t")
                    try:
                        arr = arr[1:]
                        arr = np.array(arr, dtype='float')
                        data.append(arr)
                    except:
                        pass
                    line = f.readline()
        mat = np.array(data)
        return mat

    int_array = load_rating_file_as_matrix(os.path.join(folder, dataset + "_int.txt"))

    A_sim = load_matrix(os.path.join(folder, dataset + "_A_sim.txt"))
    B_sim = load_matrix(os.path.join(folder, dataset + "_B_sim.txt"))

    intMat = np.array(int_array, dtype=np.float64)
    A_sim = np.array(A_sim, dtype=np.float64)
    B_sim = np.array(B_sim, dtype=np.float64)

    return intMat, A_sim, B_sim


def get_names(dataset, folder):
    with open(os.path.join(folder, dataset + "_int.txt"), "r") as inf:
        B = next(inf).strip("\n").split('\t')
        A = [line.strip("\n").split('\t')[0] for line in inf]
        if '' in A:
            A.remove('')
        if '' in B:
            B.remove('')
    return A, B


def WKNKN(Y, SD, ST, K, eta):
    Yd = np.zeros(Y.shape)
    Yt = np.zeros(Y.shape)
    wi = np.zeros((K,))
    wj = np.zeros((K,))
    num_drugs, num_targets = Y.shape
    for i in np.arange(num_drugs):
        dnn_i = np.argsort(SD[i,:])[::-1][1:K+1]
        Zd = np.sum(SD[i, dnn_i])
        for ii in np.arange(K):
            wi[ii] = (eta ** (ii)) * SD[i,dnn_i[ii]]
        if not np.isclose(Zd, 0.):
            Yd[i,:] = np.sum(np.multiply(wi.reshape((K,1)), Y[dnn_i,:]), axis=0) / Zd
    for j in np.arange(num_targets):
        tnn_j = np.argsort(ST[j, :])[::-1][1:K+1]
        Zt = np.sum(ST[j, tnn_j])
        for jj in np.arange(K):
            wj[jj] = (eta ** (jj)) * ST[j,tnn_j[jj]]
        if not np.isclose(Zt, 0.):
            Yt[:,j] = np.sum(np.multiply(wj.reshape((1,K)), Y[:,tnn_j]), axis=1) / Zt
    Ydt = (Yd + Yt)/2
    x, y = np.where(Ydt > Y)

    Y_tem = Y.copy()
    Y_tem[x, y] = Ydt[x, y]
    return Y_tem
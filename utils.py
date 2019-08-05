import os
import numpy as np


def load_data_from_file(dataset, folder):
    def load_rating_file_as_matrix(filename):
        '''
        Read .rating file and Return dok matrix.
        The first line of .rating file is: num_users\t num_items
        '''
        data = []
        stat = []
        try:
            with open(filename, "r") as f:
                line = f.readline()
                while line != None and line != "":
                    arr = np.array(line.split("\t"), dtype='int')
                    stat.append(sum(arr))
                    data.append(arr)
                    line = f.readline()
        except:
            with open(filename, "r") as f:
                line = f.readline()
                while line != None and line != "":
                    arr = line.split("\t")
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
        '''
        Read .rating file and Return dok matrix.
        The first line of .rating file is: num_users\t num_items
        '''
        data = []
        # for the situation that data files contain col/row name
        try:
            with open(filename, "r") as f:
                line = f.readline()
                while line != None and line != "":
                    arr = np.array(line.split("\t"), dtype='float')
                    data.append(arr)
                    line = f.readline()
        except:
            with open(filename, "r") as f:
                line = f.readline()
                while line != None and line != "":
                    arr = line.split("\t")
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

# -*- coding: utf-8 -*-
# @author: Zichao Zhang
# @date:  July 2018
import numpy as np
from numpy import dot
from time import time, asctime
import torch
from torch import mm, mul
import logging
from utils import WKNKN

datatype = torch.float32
npeps = np.finfo(float).eps
add_eps = 0.01



class GRGMF:
    def __init__(self, K=5, max_iter=100, lr=0.01, lamb=0.1, mf_dim=50, beta=0.1, r1=0.5, r2=0.5, c=5, pre_end=True,
                 cvs=None, verpose=10, resample=0, ita=0):
        '''Initialize the instance

        Args:
            K: Num of neighbors
            max_iter: Maximum num of iteration
            lr: learning rate
            lamb: trade-off parameter for norm-2 regularization for matrix U and V
            mf_dim: dimension of the subspace expanded by self-representing vectors(i.e. the dimension for MF)
            beta: trade-off parameter for norm-2 regularization for matrix U and V
            r1: trade-off parameter of graph regularization for nodes in A
            r2: trade-off parameter of graph regularization for nodes in B
            c: constant of the important level for positive sample
            cvs: cross validation setting (1, 2 or 3)
            verpose: verpose level(for standard output) TODO: remove in the future
            resample: weather to resample the positive samples or not
        '''
        if torch.cuda.is_available():
            self.DEVICE = 'cuda'
            logging.warning('CUDA is available, DEVICE=cuda')
        else:
            self.DEVICE = 'cpu'
            logging.warning('CUDA is not available, DEVICE=cpu')
        self.K = K  # number of neighbors
        self.n = -1  # data tag
        self.mf_dim = mf_dim
        self.num_factors = mf_dim
        self.max_iter = max_iter
        self.lr = lr
        self.lamb = lamb
        self.beta = beta
        self.r1 = r1
        self.r2 = r2
        self.c = c
        self.loss = [[np.inf] for i in range(50)]
        self.cvs = cvs
        self.verpose = verpose
        self.resample = resample
        self.WK = K  # let "K" in WKNKN be the same as K
        self.ita = ita
        if cvs:
            if cvs == 1:
                self.imp1 = 3.
                self.imp2 = 2.
            elif cvs == 2:
                self.imp1 = 5.
                self.imp2 = 2.
            elif cvs == 3:
                self.imp1 = 3.
                self.imp2 = 4.
        else:
            self.imp1 = 3.
            self.imp2 = 2.

    def get_nearest_neighbors(self, S, size=5):
        m, n = S.shape
        X = np.zeros((m, n))
        for i in range(m):
            ii = np.argsort(S[i, :], kind='mergesort')[::-1][:min(size, n)]
            X[i, ii] = S[i, ii]
        return X

    def fix_model(self, W, intMat, A_sim, B_sim, seed=None):
        '''Train the MF model

            Y=sigmoid(AUV^TB)
            (i.e. Y = sigmoid(Z^A U V^T Z^B))

        Args:
            W: Mask for training set
            intMat: complete interaction matrix
            A_sim: similarity matrix for nodes in A
            B_sim: similarity matrix for nodes in B
            seed: random seed to determine a random state

        Returns:
            None
        '''

        def loss_function(Y, W):
            '''
            Return the current value of loss function
            Args:
                Y: interaction matrix
                W: mask for training set

            Returns:
                current value of loss function
            '''
            temp = mm(self.ZA, mm(self.U, mm(self.V.t(), self.ZB)))  # avoiding overflow
            logexp = torch.zeros(temp.shape, dtype=datatype, device=DEVICE)
            logexp[temp > 50] = temp[temp > 50] * 1.  # np.log(np.e)
            logexp[temp <= 50] = torch.log(torch.exp(temp[temp <= 50]) + 1)

            loss = (mul(mul((1 + c * Y - Y), logexp)
                        - mul(c * Y, mm(self.ZA, mm(self.U, mm(self.V.t(), self.ZB)))), W).sum()
                    + lamb * (torch.norm(self.U, 2) ** 2 + torch.norm(self.V, 2) ** 2)
                    + beta * (torch.norm(self.ZA, 2) ** 2 + torch.norm(self.ZB, 2) ** 2)
                    + r1 * torch.trace(
                        mm(self.U.t(), mm(self.ZA.t(), mm(self.lap_mat(self.A_sim), mm(self.ZA, self.U)))))
                    + r2 * torch.trace(
                        mm(self.V.t(), mm(self.ZB, mm(self.lap_mat(self.B_sim), mm(self.ZB.t(), self.V))))))
            return loss

        # data split
        Y = intMat * W
        self.num_drugs, self.num_targets = Y.shape
        if (self.DEVICE == 'cuda') & (np.sqrt(self.num_drugs * self.num_targets) < 300):
            self.DEVICE = 'cpu'
            logging.warning('Matrix dimensions are small, DEVICE=CPU')
        DEVICE = self.DEVICE

        tt = time()
        self.n += 1
        self.A_sim = A_sim.copy()
        self.B_sim = B_sim.copy()

        # emphasize the diag of similarity matrix
        self.A_sim += self.imp1 * np.diag(np.diag(np.ones(A_sim.shape)))
        self.B_sim += self.imp2 * np.diag(np.diag(np.ones(B_sim.shape)))

        # ## sparsification
        self.A_sim = self.get_nearest_neighbors(self.A_sim, self.K + 1)
        self.B_sim = self.get_nearest_neighbors(self.B_sim, self.K + 1)

        # symmetrization
        self.A_sim = (self.A_sim + self.A_sim.T) / 2.
        self.B_sim = (self.B_sim + self.B_sim.T) / 2.

        # normalization
        self.ZA = dot(np.diag(1. / np.sum(self.A_sim + add_eps, axis=1).flatten()), self.A_sim + add_eps)
        self.ZB = dot(self.B_sim + add_eps, np.diag(1. / np.sum(self.B_sim + add_eps, axis=0).flatten()))

        # 2 initialization for U and V
        prng = np.random.RandomState(seed)
        if seed != None:
            self.U = np.sqrt(1. / float(self.num_factors)) * prng.normal(size=(self.num_drugs, self.num_factors))
            self.V = np.sqrt(1. / float(self.num_factors)) * prng.normal(size=(self.num_targets, self.num_factors))
        else:
            self.U = np.sqrt(1. / float(self.num_factors)) * np.random.normal(size=(self.num_drugs, self.num_factors))
            self.V = np.sqrt(1. / float(self.num_factors)) * np.random.normal(size=(self.num_targets, self.num_factors))
        u, s, v = np.linalg.svd(WKNKN(Y, A_sim, B_sim, self.WK, self.ita))
        self.U[:, : min(self.num_factors, min(Y.shape))] = u[:, : min(self.num_factors, min(Y.shape))]
        self.V[:, : min(self.num_factors, min(Y.shape))] = v[:, :min(self.num_factors, min(Y.shape))]
        del u, s, v

        # load variables to GPU memory
        self.U = torch.tensor(self.U, dtype=datatype, device=DEVICE)
        self.V = torch.tensor(self.V, dtype=datatype, device=DEVICE)
        self.ZA = torch.tensor(self.ZA, dtype=datatype, device=DEVICE)
        self.ZB = torch.tensor(self.ZB, dtype=datatype, device=DEVICE)
        self.A_sim = torch.tensor(self.A_sim, dtype=datatype, device=DEVICE)
        self.B_sim = torch.tensor(self.B_sim, dtype=datatype, device=DEVICE)
        Y = torch.tensor(Y, dtype=datatype, device=DEVICE)
        W_all = torch.tensor(W, dtype=datatype, device=DEVICE)
        # W_all = torch.tensor(np.ones(W.shape), dtype=datatype, device=DEVICE)
        lamb = torch.tensor(data=self.lamb, dtype=datatype, device=DEVICE)
        beta = torch.tensor(data=self.beta, dtype=datatype, device=DEVICE)
        r1 = torch.tensor(data=self.r1, dtype=datatype, device=DEVICE)
        r2 = torch.tensor(data=self.r2, dtype=datatype, device=DEVICE)
        c = torch.tensor(data=self.c, dtype=datatype, device=DEVICE)
        max_iter = torch.tensor(data=self.max_iter, dtype=torch.int, device=DEVICE)
        eps = torch.tensor(data=npeps, dtype=datatype, device=DEVICE)

        # Using adam optimizer:
        lr = self.lr
        opter = self.adam_opt
        patient = 3
        numiter = torch.tensor(data=0, dtype=torch.int, device=DEVICE)
        numiter.copy_(max_iter)
        minloss = np.inf

        # store the initial value of ZA ZB U V for later use
        init_U = torch.zeros(self.U.shape, dtype=datatype, device=DEVICE)
        init_U.copy_(self.U)
        init_V = torch.zeros(self.V.shape, dtype=datatype, device=DEVICE)
        init_V.copy_(self.V)
        init_ZA = torch.zeros(self.ZA.shape, dtype=datatype, device=DEVICE)
        init_ZA.copy_(self.ZA)
        init_ZB = torch.zeros(self.ZB.shape, dtype=datatype, device=DEVICE)
        init_ZB.copy_(self.ZB)

        # A_old refer to the value of ZA in the last iteratio
        self.A_old = torch.zeros(self.ZA.shape, dtype=datatype, device=DEVICE)
        self.A_old.copy_(self.ZA)
        self.B_old = torch.zeros(self.ZB.shape, dtype=datatype, device=DEVICE)
        self.B_old.copy_(self.ZB)
        self.U_old = torch.zeros(self.U.shape, dtype=datatype, device=DEVICE)
        self.U_old.copy_(self.U)
        self.V_old = torch.zeros(self.V.shape, dtype=datatype, device=DEVICE)
        self.V_old.copy_(self.V)

        ZA_best = torch.zeros(self.ZA.shape, dtype=datatype, device=DEVICE)
        ZB_best = torch.zeros(self.ZB.shape, dtype=datatype, device=DEVICE)
        U_best = torch.zeros(self.U.shape, dtype=datatype, device=DEVICE)
        V_best = torch.zeros(self.V.shape, dtype=datatype, device=DEVICE)

        # iteration
        while numiter > 0:
            W_ = W_all
            Y_p = mm(self.ZA, mm(self.U, mm(self.V.t(), self.ZB)))
            P = torch.sigmoid(Y_p)

            # update U,V
            t1 = time()
            # reinitialize the optimizer
            opter_U = opter(lr=lr, shape=self.U.shape, DEVICE=DEVICE)
            opter_V = opter(lr=lr, shape=self.V.shape, DEVICE=DEVICE)
            for foo in range(30):
                # compute the derivative of U and V
                deriv_U = (mm(mm(self.ZA.t(), P * W_), mm(self.ZB.t(), self.V))
                           + mm(mm((c - 1.) * self.ZA.t(), Y * P * W_), mm(self.ZB.t(), self.V))
                           - c * mm(mm(self.ZA.t(), Y * W_), mm(self.ZB.t(), self.V))
                           + 2. * lamb * self.U
                           + 2. * r1 * mm(mm(self.ZA.t(), self.lap_mat(self.A_sim)), mm(self.ZA, self.U)))
                deriv_V = (mm(self.ZB, mm((P * W_).t(), mm(self.ZA, self.U)))
                           + (c - 1.) * mm(self.ZB, mm((Y * P * W_).t(), mm(self.ZA, self.U)))
                           - c * mm(self.ZB, mm((Y * W_).t(), mm(self.ZA, self.U)))
                           + 2. * lamb * self.V
                           + 2. * r2 * mm(self.ZB, mm(self.lap_mat(self.B_sim), mm(self.ZB.t(), self.V))))

                # update using adam optimizer
                self.U += opter_U.delta(deriv_U, max_iter - numiter)
                self.V += opter_V.delta(deriv_V, max_iter - numiter)
                Y_p = mm(self.ZA, mm(self.U, mm(self.V.t(), self.ZB)))
                P = torch.sigmoid(Y_p)

                # break the loop if reach converge condition
                if ((torch.norm(self.U - self.U_old, p=1) / torch.norm(self.U_old, p=1))
                    < 0.01) and ((torch.norm(self.V - self.V_old, p=1) / torch.norm(self.V_old, p=1)) < 0.01):
                    logging.debug("UV update num: %d" % foo)
                    break
                self.U_old.copy_(self.U)
                self.V_old.copy_(self.V)

            # store matrix U, V, ZA and ZB for the currently lowest loss for later use
            self.loss[self.n].append(loss_function(Y, W_all))
            if self.loss[self.n][-1] < minloss:
                ZA_best.copy_(self.ZA)
                ZB_best.copy_(self.ZB)
                U_best.copy_(self.U)
                V_best.copy_(self.V)
                minloss = self.loss[self.n][-1]

            # if diverge reinitialize U, V, ZA, ZB and optimizer with half the present learning rate
            if self.loss[self.n][-1] > self.loss[self.n][1] * 2:
                if not patient:
                    with open('./error.log', 'a') as error:
                        err = 'Converge error\n' + str(self) + '\t' + asctime() + '\n'
                        err += 'Current lr: %f\n' % lr
                        error.write(err)
                    self.ZA = ZA_best
                    self.ZB = ZB_best
                    self.U = U_best
                    self.V = V_best
                    logging.error('Dataset[%d], U&V: Fail to reach convergence, exit with best loss=%.2f' %
                                  (self.n, min(self.loss[self.n])))
                    return
                # Reinitialization
                self.ZA.copy_(init_ZA)
                self.ZB.copy_(init_ZB)
                self.U.copy_(init_U)
                self.V.copy_(init_V)
                lr = lr * 0.5
                numiter.copy_(max_iter)
                self.loss[self.n][-1] = np.inf
                patient -= 1
                logging.warning("Dataset[%d], U&V: Diverged, attempt to retrain with half the present "
                                "learning rate, lr=%.4f/2, patient=%d-1, "
                                "best loss=%.2f" % (self.n, lr * 2, patient + 1, min(self.loss[self.n])))
                break

            # Update ZA & ZB
            t1 = time()
            for n in range(30):
                temp_p = ((mm(self.U, mm(self.V.t(), self.ZB))).t() + torch.abs(
                    mm(self.U, mm(self.V.t(), self.ZB)).t())) * 0.5
                temp_n = (torch.abs(mm(self.U, mm(self.V.t(), self.ZB)).t())
                          - mm(self.U, mm(self.V.t(), self.ZB)).t()) * 0.5
                UUT_p = (mm(self.U, self.U.t()) + torch.abs(mm(self.U, self.U.t()))) * 0.5
                UUT_n = (torch.abs(mm(self.U, self.U.t())) - mm(self.U, self.U.t())) * 0.5
                D_AP = (mm(P * W_, temp_p) + (c - 1) * mm(Y * P * W_, temp_p)
                        + c * mm(Y * W_, temp_n) + beta * (2 * self.ZA)
                        + 2. * r1 * mm(torch.diag(self.A_sim.sum(1)), mm(self.ZA, UUT_p))
                        + 2. * r1 * mm(self.A_sim, mm(self.ZA, UUT_n)))
                D_AN = (mm(P * W_, temp_n) + (c - 1) * mm(Y * P * W_, temp_n)
                        + c * mm(Y * W_, temp_p)
                        + 2. * r1 * mm(torch.diag(self.A_sim.sum(1)), mm(self.ZA, UUT_n))
                        + 2. * r1 * mm(self.A_sim, mm(self.ZA, UUT_p)))
                temp_p = ((mm(self.ZA, mm(self.U, self.V.t()))).t() + torch.abs(
                    (mm(self.ZA, mm(self.U, self.V.t()))).t())) * 0.5
                temp_n = (torch.abs(mm(self.ZA, mm(self.U, self.V.t())).t())
                          - (mm(self.ZA, mm(self.U, self.V.t()))).t()) * 0.5
                VVT_p = (mm(self.V, self.V.t()) + torch.abs(mm(self.V, self.V.t()))) * 0.5
                VVT_n = (torch.abs(mm(self.V, self.V.t())) - mm(self.V, self.V.t())) * 0.5

                D_BP = (mm(temp_p, P * W_) + (c - 1) * mm(temp_p, Y * P * W_)
                        + c * mm(temp_n, Y * W_)
                        + beta * (2 * self.ZB)
                        + 2. * r2 * mm(VVT_p, mm(self.ZB, torch.diag(self.B_sim.sum(1))))
                        + 2. * r2 * mm(VVT_n, mm(self.ZB, self.B_sim)))
                D_BN = (mm(temp_n, P * W_) + (c - 1) * mm(temp_n, Y * P * W_)
                        + c * mm(temp_p, Y * W_)
                        + 2. * r2 * mm(VVT_n, mm(self.ZB, torch.diag(self.B_sim.sum(1))))
                        + 2. * r2 * mm(VVT_p, mm(self.ZB, self.B_sim)))
                temp = (self.ZA * (1. / (D_AP + eps))).sum(1).flatten()
                D_SA = torch.diag(temp)  # refer to D superscript ZA
                E_SA = (self.ZA * D_AN * (1. / (D_AP + eps))).sum(1).reshape(self.num_drugs, 1).repeat(1,
                                                                                                      self.num_drugs)
                temp = (self.ZB * (1. / (D_BP + eps))).sum(0).flatten()
                D_SB = torch.diag(temp)  # refer to D superscript ZB
                E_SB = (self.ZB * D_BN * (1. / (D_BP + eps))).sum(0).reshape(1, self.num_targets).repeat(
                    self.num_targets, 1)

                self.ZA = self.ZA * (mm(D_SA, D_AN) + 1) * (1. / (mm(D_SA, D_AP) + E_SA + eps))
                self.ZB = self.ZB * (mm(D_BN, D_SB) + 1) * (1. / (mm(D_BP, D_SB) + E_SB + eps))
                Y_p = mm(self.ZA, mm(self.U, mm(self.V.t(), self.ZB)))
                P = torch.sigmoid(Y_p)

                # break the loop if reach converge condition
                if ((torch.norm(self.ZA - self.A_old, p=1) / torch.norm(self.A_old, p=1))
                    < 0.01) and ((torch.norm(self.ZB - self.B_old, p=1) / torch.norm(self.B_old, p=1)) < 0.01):
                    logging.debug("AB update num: %d" % n)
                    break
                self.A_old.copy_(self.ZA)
                self.B_old.copy_(self.ZB)

            # store matrix U, V, ZA and ZB for the currently lowest loss for later use
            self.loss[self.n].append(loss_function(Y, W_all))
            if self.loss[self.n][-1] < minloss:
                ZA_best.copy_(self.ZA)
                ZB_best.copy_(self.ZB)
                U_best.copy_(self.U)
                V_best.copy_(self.V)
                minloss = self.loss[self.n][-1]

            # if diverge reinitialize U, V, ZA, ZB and optimizer with half the present learning rate
            if self.loss[self.n][-1] > self.loss[self.n][1] * 2:
                if not patient:
                    with open('./error.log', 'a') as error:
                        err = 'early termination: Do not converge\n' + str(self) + '\t' + asctime() + '\n'
                        err += 'Current lr: %f\n' % lr
                        error.write(err)
                    self.ZA = ZA_best
                    self.ZB = ZB_best
                    self.U = U_best
                    self.V = V_best
                    logging.error('Dataset[%d], ZA&ZB: Fail to reach convergence, exit with best loss=%.2f' %
                                  (self.n, min(self.loss[self.n])))
                    return
                # Reinitialization
                self.ZA.copy_(init_ZA)
                self.ZB.copy_(init_ZB)
                self.U.copy_(init_U)
                self.V.copy_(init_V)
                lr = lr * 0.5
                numiter.copy_(max_iter)
                self.loss[self.n][-1] = np.inf
                patient -= 1
                logging.warning("Dataset[%d], ZA&ZB: Diverged, attempt to retrain with half the present "
                                "learning rate, lr=%.4f/2, patient=%d-1, "
                                "best loss=%.2f" % (self.n, lr * 2, patient + 1, min(self.loss[self.n])))
                break
            else:
                # training stop if reach converge condition
                delta_loss = abs(self.loss[self.n][-1] - self.loss[self.n][-2]) / abs(self.loss[self.n][-2])
                logging.info(('Delta_loss: %.4f' % delta_loss))
                if delta_loss < 1e-4:
                    numiter = torch.tensor(data=0, dtype=torch.int, device=DEVICE)
            numiter -= 1

        # retrieve the best U, V, ZA and ZB (with lowest loss)
        if self.loss[self.n][-1] > minloss:
            self.ZA = ZA_best
            self.ZB = ZB_best
            self.U = U_best
            self.V = V_best

        Y_p = mm(self.ZA, mm(self.U, mm(self.V.t(), self.ZB)))
        self.P = torch.sigmoid(Y_p).numpy()


    def predict_scores(self, test_data, N):
        '''return the predicting label of input data

        Args:
            test_data: input data

        Returns:
            predicting label
        '''

        # Evaluation
        test_data = np.array(test_data)
        Y_p = mm(self.ZA, mm(self.U, mm(self.V.t(), self.ZB)))
        P = torch.sigmoid(Y_p).cpu().numpy()
        test_data = test_data.T
        y_pre = P[test_data[0], test_data[1]]
        return y_pre

    def lap_mat(self, S):
        '''Return the Laplacian matrix of adjacent matrix S

        Args:
            S: adjacent matrix

        Returns:
            Laplacian matrix of S
        '''
        x = S.sum(1)
        L = torch.diag(x) - S  # neighborhood regularization matrix
        return L

    class adam_opt:
        def __init__(self, lr, shape, DEVICE):
            '''Adam optimizer

            Args:
                lr:
                shape:
            '''
            self.alpha = torch.tensor(data=lr, dtype=datatype, device=DEVICE)
            self.beta1 = torch.tensor(data=0.9, dtype=datatype, device=DEVICE)
            self.beta2 = torch.tensor(data=0.999, dtype=datatype, device=DEVICE)
            self.epsilon = torch.tensor(data=10E-8, dtype=datatype, device=DEVICE)
            self.eps = torch.tensor(data=npeps, dtype=datatype, device=DEVICE)
            self.t = torch.tensor(data=0, dtype=datatype, device=DEVICE)
            self.m0 = torch.zeros(shape, dtype=datatype, device=DEVICE)
            self.v0 = torch.zeros(shape, dtype=datatype, device=DEVICE)

        def delta(self, deriv, iter):
            # in case pass a matrix type grad
            self.t = (iter + 1).type(datatype)
            grad = deriv
            m_t = self.beta1 * self.m0 + (1 - self.beta1) * grad
            v_t = self.beta2 * self.v0 + (1 - self.beta2) * grad ** 2
            # In this project the number of iteration is too big so let t divided by a number
            m_cap = m_t / (1. - self.beta1 ** (self.t / 1.) + self.eps)
            v_cap = v_t / (1. - self.beta2 ** (self.t / 1.) + self.eps)
            update = - self.alpha * m_cap / (torch.sqrt(v_cap) + self.epsilon + self.eps)
            self.m0.copy_(m_t)
            self.v0.copy_(v_t)
            return update

    def __str__(self):
        return ("Model: ADPGMF, cvs: %s, K: %s, mf_dim: %s, lamb:%s, beta:%s, c:%s, resample:%d, r1:%s, r2:%s, ita:%s "
                "imp1:%s, imp2: %s, max_iter: %s, lr: %s" % (self.cvs, self.K, self.mf_dim, self.lamb, self.beta,
                                                             self.c, self.resample, self.r1, self.r2, self.ita,
                                                             self.imp1, self.imp2, self.max_iter, self.lr))

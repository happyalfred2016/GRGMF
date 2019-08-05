# -*- coding: utf-8 -*-
# @author: Zichao Zhang
# @date:  July 2018


import sys
sys.path.append('../')
import numpy as np
from numpy import multiply, array, dot
from numpy.matlib import repmat
from numpy.linalg import norm
npeps = np.finfo(float).eps
add_eps = 0.01


class GRGMF:
    def __init__(self, K=8, max_iter=500, lr=0.003, lamb=0.01, mf_dim=50, beta=0.01, r1=0.01, r2=0.01, c=5):

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

    def get_nearest_neighbors(self, S, size=5):
        m, n = S.shape
        X = np.zeros((m, n))
        for i in range(m):
            ii = np.argsort(S[i, :], kind='mergesort')[::-1][:min(size, n)]
            X[i, ii] = S[i, ii]
        return X

    def loss_function(self, Y, W):
        temp = dot(self.ZA, dot(self.U, dot(self.V.T, self.ZB)))  # avoiding overflow
        logexp = np.zeros(temp.shape)
        logexp[temp > 50] = temp[temp > 50] * np.log(np.e)
        logexp[temp <= 50] = np.log(np.exp(temp[temp <= 50]) + 1)

        loss = (multiply(multiply((1 + self.c * Y - Y), logexp)
                         - multiply(self.c * Y, dot(self.ZA, dot(self.U, dot(self.V.T, self.ZB)))), W).sum()
                + self.lamb * (norm(self.U, ord='fro') ** 2 + norm(self.V, ord='fro') ** 2)
                + self.beta * (norm(self.ZA, ord='fro') ** 2 + norm(self.ZB, ord='fro') ** 2)
                + self.r1 * np.trace(
                    dot(self.U.T, dot(self.ZA.T, dot(self.lap_mat(self.A_sim), dot(self.ZA, self.U)))))
                + self.r2 * np.trace(
                    dot(self.V.T, dot(self.ZB, dot(self.lap_mat(self.B_sim), dot(self.ZB.T, self.V))))))
        return loss


    def fix_model(self, W, intMat, A_sim, B_sim, seed=None):
        self.n += 1
        Y = intMat * W # Training part of intMat.
        W_all = W
        
        blank_row = (intMat.sum(1) < 1).sum()
        blank_col = (intMat.sum(0) < 1).sum()
        if (blank_row > 0) or (blank_col > 0):
            if blank_row > blank_col:
                imp1 = 5.
                imp2 = 2.
            else:
                imp1 = 3.
                imp2 = 4.
        else:
            imp1 = 3.
            imp2 = 2.
        self.A_sim = A_sim.copy()
        self.B_sim = B_sim.copy()
        self.num_A, self.num_B = Y.shape
        self.A_sim += imp1 * np.diag(np.diag(np.ones(A_sim.shape)))
        self.B_sim += imp2 * np.diag(np.diag(np.ones(B_sim.shape)))

        # ## sparsification
        self.A_sim = self.get_nearest_neighbors(self.A_sim, self.K + 1)
        self.B_sim = self.get_nearest_neighbors(self.B_sim, self.K + 1)
        # # symmetrization
        self.A_sim = (self.A_sim + self.A_sim.T) / 2.
        self.B_sim = (self.B_sim + self.B_sim.T) / 2.
        DL = self.lap_mat(self.A_sim)
        TL = self.lap_mat(self.B_sim)
        # normalization
        self.ZA = dot(np.diag(1. / np.sum(self.A_sim +add_eps, axis=1).flatten()), self.A_sim + add_eps)
        self.ZB = dot(self.B_sim + add_eps, np.diag(1. / np.sum(self.B_sim + add_eps, axis=0).flatten()))

        prng = np.random.RandomState(seed)
        if seed != None:
            self.U = np.sqrt(1. / float(self.num_factors)) * prng.normal(size=(self.num_A, self.num_factors))
            self.V = np.sqrt(1. / float(self.num_factors)) * prng.normal(size=(self.num_B, self.num_factors))
        else:
            self.U = np.sqrt(1. / float(self.num_factors)) * np.random.normal(size=(self.num_A, self.num_factors))
            self.V = np.sqrt(1. / float(self.num_factors)) * np.random.normal(size=(self.num_B, self.num_factors))
        self.ZA_old = self.ZA.copy()
        self.ZB_old = self.ZB.copy()
        self.U_old = self.U.copy()
        self.V_old = self.V.copy()

        # Using optimizer:
        lr = self.lr
        opter = self.adam_opt
        opter_U = opter(lr=lr, shape=self.U.shape)
        opter_V = opter(lr=lr, shape=self.V.shape)

        patient = 5
        numiter = self.max_iter
        minloss = np.inf

        init_ZA = self.ZA.copy()
        init_ZB = self.ZB.copy()
        init_U = self.U.copy()
        init_V = self.V.copy()

        while numiter:
            W_ = W_all
            Y_p = dot(self.ZA, dot(self.U, dot(self.V.T, self.ZB)))
            P = self.sigmoid(Y_p)
            for foo in range(5):
                deriv_U = (dot(self.ZA.T, dot((P+(self.c-1)*Y*P-self.c*Y)*W_, dot(self.ZB.T, self.V)))
                           + 2. * self.lamb * self.U
                           + 2. * self.r1 * dot(dot(self.ZA.T, DL), dot(self.ZA, self.U)))

                deriv_V = (dot(self.ZB, dot((P.T + (self.c - 1) * (Y * P).T - self.c * Y.T) * W_.T, dot(self.ZA, self.U)))
                           + 2. * self.lamb * self.V
                           + 2. * self.r2 * dot(self.ZB, dot(TL, dot(self.ZB.T, self.V))))

                # update using adam optimizer
                self.U += opter_U.delta(deriv_U, self.max_iter - numiter)
                self.V += opter_V.delta(deriv_V, self.max_iter - numiter)
                Y_p = dot(self.ZA, dot(self.U, dot(self.V.T, self.ZB)))
                P = self.sigmoid(Y_p)

            self.loss[self.n].append(self.loss_function(Y, W_all))
            if self.loss[self.n][-1] < minloss:
                ZA_best, ZB_best, U_best, V_best = self.ZA.copy(), self.ZB.copy(), self.U.copy(), self.V.copy()
                minloss = self.loss[self.n][-1]

            if self.loss[self.n][-1] > self.loss[self.n][1] * 10:
                if not patient:
                    self.ZA = ZA_best.copy()
                    self.ZB = ZB_best.copy()
                    self.U = U_best.copy()
                    self.V = V_best.copy()
                    return

                # Reinitialization
                self.ZA = init_ZA.copy()
                self.ZB = init_ZB.copy()
                self.U = init_U.copy()
                self.V = init_V.copy()
                lr = lr * 0.5
                opter_U = opter(lr=lr, shape=self.U.shape)
                opter_V = opter(lr=lr, shape=self.V.shape)
                numiter = self.max_iter
                self.loss[self.n][-1] = np.inf
                patient -= 1
                continue

            self.ZA_old = self.ZA.copy()
            self.ZB_old = self.ZB.copy()
            self.U_old = self.U.copy()
            self.V_old = self.V.copy()

            # Update ZA & ZB
            temp_p = ((dot(self.U, dot(self.V.T, self.ZB))).T + np.abs(dot(self.U, dot(self.V.T, self.ZB)).T)) * 0.5
            temp_n = (np.abs(dot(self.U, dot(self.V.T, self.ZB)).T) - dot(self.U, dot(self.V.T, self.ZB)).T) * 0.5
            UUT_p = (dot(self.U, self.U.T) + np.abs(dot(self.U, self.U.T))) * 0.5
            UUT_n = (np.abs(dot(self.U, self.U.T)) - dot(self.U, self.U.T)) * 0.5
            D_AP = (dot((1 + (self.c - 1) * Y) * W_ * P, temp_p) + self.c * dot(Y * W_, temp_n)
                    + self.beta * (2 * self.ZA)
                    + 2. * self.r1 * dot(np.diag(np.sum(self.A_sim, 1)), dot(self.ZA, UUT_p))
                    + 2. * self.r1 * dot(self.A_sim, dot(self.ZA, UUT_n)))
            D_AN = (dot((1 + (self.c - 1) * Y) * W_ * P, temp_n) + self.c * dot(Y * W_, temp_p)
                    + 2. * self.r1 * dot(np.diag(np.sum(self.A_sim, 1)), dot(self.ZA, UUT_n))
                    + 2. * self.r1 * dot(self.A_sim, dot(self.ZA, UUT_p)))
            temp = (np.sum(self.ZA * (1. / (D_AP + npeps)), axis=1)).flatten()
            D_SA = np.diag(temp)  # refer to D superscript A
            E_SA = repmat(np.sum(self.ZA * D_AN * (1. / (D_AP + npeps)), axis=1).reshape(self.num_A, 1), 1,
                          self.num_A)
            temp_p = ((dot(self.ZA, dot(self.U, self.V.T))).T + np.abs((dot(self.ZA, dot(self.U, self.V.T))).T)) * 0.5
            temp_n = (np.abs((dot(self.ZA, dot(self.U, self.V.T))).T) - (dot(self.ZA, dot(self.U, self.V.T))).T) * 0.5
            VVT_p = (dot(self.V, self.V.T) + np.abs(dot(self.V, self.V.T))) * 0.5
            VVT_n = (np.abs(dot(self.V, self.V.T)) - dot(self.V, self.V.T)) * 0.5
            D_BP = (dot(temp_p, (1 + (self.c - 1) * Y) * P * W_) + self.c * dot(temp_n, Y * W_)
                    + self.beta * (2 * self.ZB)
                    + 2. * self.r2 * dot(VVT_p, dot(self.ZB, np.diag(np.sum(self.B_sim, 1))))
                    + 2. * self.r2 * dot(VVT_n, dot(self.ZB, self.B_sim)))
            D_BN = (dot(temp_n, (1 + (self.c - 1) * Y) * P * W_) + self.c * dot(temp_p, Y * W_)
                    + 2. * self.r2 * dot(VVT_n, dot(self.ZB, np.diag(np.sum(self.B_sim, 1))))
                    + 2. * self.r2 * dot(VVT_p, dot(self.ZB, self.B_sim)))
            temp = (np.sum(self.ZB * (1. / (D_BP + npeps)), axis=0)).flatten()
            D_SB = np.diag(temp)  # refer to D superscript B
            E_SB = repmat(np.sum(self.ZB * D_BN * (1. / (D_BP + npeps)), axis=0).reshape(1, self.num_B),
                          self.num_B, 1)

            self.ZA = self.ZA * ((dot(D_SA, D_AN) + 1) * (1. / (dot(D_SA, D_AP) + E_SA + npeps)))
            self.ZB = self.ZB * (dot(D_BN, D_SB) + 1) * (1. / (dot(D_BP, D_SB) + E_SB + npeps))
            Y_p = dot(self.ZA, dot(self.U, dot(self.V.T, self.ZB)))
            P = self.sigmoid(Y_p)

            self.loss[self.n].append(self.loss_function(Y, W_all))
            if self.loss[self.n][-1] < minloss:
                ZA_best, ZB_best, U_best, V_best = self.ZA.copy(), self.ZB.copy(), self.U.copy(), self.V.copy()
                minloss = self.loss[self.n][-1]

            # Ealry terminated, let lr = lr*0.5 and try again.
            if self.loss[self.n][-1] > self.loss[self.n][1] * 10:
                if not patient:
                    self.ZA = ZA_best.copy()
                    self.ZB = ZB_best.copy()
                    self.U = U_best.copy()
                    self.V = V_best.copy()
                    return

                # Reinitialization
                self.ZA = init_ZA.copy()
                self.ZB = init_ZB.copy()
                self.U = init_U.copy()
                self.V = init_V.copy()

                lr = lr * 0.5
                opter_U = opter(lr=lr, shape=self.U.shape)
                opter_V = opter(lr=lr, shape=self.V.shape)
                numiter = self.max_iter
                self.loss[self.n][-1] = np.inf
                patient -= 1
                continue

            self.ZA_old = self.ZA.copy()
            self.ZB_old = self.ZB.copy()
            self.U_old = self.U.copy()
            self.V_old = self.V.copy()
            numiter -= 1

        if self.loss[self.n][-1] > minloss:
            self.ZA = ZA_best.copy()
            self.ZB = ZB_best.copy()
            self.U = U_best.copy()
            self.V = V_best.copy()
        Y_p = dot(self.ZA, dot(self.U, dot(self.V.T, self.ZB)))
        self.P = self.sigmoid(Y_p)


    def predict_scores(self, test_data, N):
        # Evaluation
        test_data = np.array(test_data)
        Y_p = dot(self.ZA, dot(self.U, dot(self.V.T, self.ZB)))
        P = self.sigmoid(Y_p)
        P = np.array(P)
        test_data = test_data.T
        y_pre = P[test_data[0], test_data[1]]
        return y_pre


    def sigmoid(self, x):
        re = np.zeros(x.shape)
        re[x >= -100] = 1. / (1 + np.exp(-x[x >= -100]))
        return re

    def lap_mat(self, S):
        x = np.sum(np.array(S), axis=1)
        L = np.diag(x) - np.array(S)  # neighborhood regularization matrix
        return L

    class adam_opt:
        def __init__(self, lr, shape):
            self.alpha = lr
            self.beta1 = 0.9
            self.beta2 = 0.999
            self.epsilon = 10E-8
            self.t = 0
            self.m0 = np.zeros(shape)
            self.v0 = np.zeros(shape)

        def delta(self, deriv, iter):
            self.t = iter + 1
            npeps = np.finfo(float).eps
            grad = array(deriv.copy())

            m_t = self.beta1 * self.m0 + (1 - self.beta1) * grad
            v_t = self.beta2 * self.v0 + (1 - self.beta2) * grad ** 2
            m_cap = m_t / (1. - self.beta1 ** (self.t / 1.) + npeps)
            v_cap = v_t / (1. - self.beta2 ** (self.t / 1.) + npeps)
            update = - self.alpha * m_cap / (np.sqrt(v_cap) + self.epsilon + npeps)
            self.m0 = m_t.copy()
            self.v0 = v_t.copy()
            return update


    def __str__(self):
        return ("Model: GRGMF, K: %s, mf_dim: %s, lamb:%s, beta:%s, c:%s, r1:%s, r2:%s, "
                "max_iter: %s, lr: %s" % (self.K, self.mf_dim, self.lamb, self.beta,
                                          self.c,  self.r1, self.r2, self.max_iter, self.lr))
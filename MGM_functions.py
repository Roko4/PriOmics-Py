# MIT License
# Copyright (c) 2023 Michael Altenbuchinger and Robin Kosch
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.

import numpy as np
import apgpy as apg
from itertools import chain

def transform_X(X, groups_X, prior_X):          #new
    p = X.shape[1]
    n = X.shape[0]

    M = np.ones((n,p))
    for i in range(len(groups_X)):
        IDs_B = groups_X[i]
        prior = prior_X[i]
        if prior=="A":
            M[:, IDs_B] = 1/len(IDs_B)*M[:, IDs_B]
        elif prior=="B":
            M[:, IDs_B] = M[:, IDs_B]
        elif prior=="C":
            M[:, IDs_B] = M[:, IDs_B]
    return M

def transform_B(B, groups_X, prior_X):          #new
    p = B.shape[1]
    for i in range(len(groups_X)):
        IDs_B = groups_X[i]
        prior = prior_X[i]
        if prior=="A" or prior=="B":
            B[np.ix_(IDs_B, IDs_B)]=0
    return B

def grad_neglogli(B, Rho, Phi, alphap, alphaq, X, D, levels, groups_X, prior_X, use_prior):
    n = X.shape[0]
    p = B.shape[0]
    q = levels.shape[0]

    if use_prior == True:       #new
        M = transform_X(X, groups_X, prior_X)       #new
    else:                   #new
        M = np.ones((n,p))#new
    X_response = X.copy()  #new
    X = M * X       #new

    levelSum = [0]
    levelSum.extend(levels)
    levelSum = np.cumsum(levelSum)
    for r in range(0, q):
        Phi[int(levelSum[r]):int(levelSum[r]+levels[r]),
            int(levelSum[r]):int(levelSum[r]+levels[r])] = 0

    Bd = np.diag(B)
    B = B - np.diag(Bd)
    B = np.triu(B)
    B = B + np.transpose(B)
    DRho = np.dot(D, Rho)
    DRho = np.dot(DRho, np.diag(np.divide(np.repeat(1., p), Bd)))

    XB = np.dot(X, B)

    XB = np.dot(XB, np.diag(np.divide(np.repeat(1., p), Bd)))
    consts = np.tile(alphap, (n, 1))

    res = consts + DRho + XB + X_response           #new X_response


    Xt = np.transpose(X)
    gradBd = np.repeat(0., p)

    for s in range(0, p):
        gradBd[s] = - n/(2.*Bd[s]) - .5*np.dot(res[:, s], res[:, s]
                                              ) + np.dot(res[:, s], XB[:, s] +
                                                         DRho[:, s])

    gradB = - np.dot(Xt, res)
    gradB = gradB - np.diag(np.diag(gradB))
    gradB = np.transpose(np.tril(gradB)) + np.triu(gradB)
    gradalphap = - np.dot(np.diag(Bd), np.sum(res, axis=0)[:, np.newaxis])
    gradalphap = gradalphap[:, 0]

    gradRho = - np.dot(np.transpose(D), res)

    RhoX = np.dot(Rho, Xt)
    Phi = Phi - np.diag(np.diag(Phi))
    Phi = np.triu(Phi)
    Phi = Phi + np.transpose(Phi)
    Phirr = np.transpose(np.tile(alphaq, (n, 1)))
    PhiD = np.dot(Phi, np.transpose(D))
    discprod = np.transpose(RhoX+Phirr+PhiD)

    for r in range(0, q):
        disctemp = discprod[:, int(levelSum[r]):int(levelSum[r]+levels[r])]
        denominator = np.logaddexp.reduce(disctemp, axis=1)
        disctemp = disctemp - denominator[:, np.newaxis]
        disctemp = np.exp(disctemp)
        temp = disctemp - D[:, int(levelSum[r]):int(levelSum[r]+levels[r])]
        discprod[:, int(levelSum[r]):int(levelSum[r]+levels[r])] = temp

    gradalphaq = np.sum(discprod, axis=0)
    gradw = np.dot(Xt, discprod)
    gradRho = gradRho+np.transpose(gradw)
    gradPhi = np.dot(np.transpose(D), discprod)

    for r in range(0, q):
        gradPhi[int(levelSum[r]):int(levelSum[r]+levels[r]),
                int(levelSum[r]):int(levelSum[r]+levels[r])] = 0

    gradPhi = np.transpose(np.tril(gradPhi))+np.triu(gradPhi)
    if use_prior == True:                       #new
        gradB = transform_B(gradB, groups_X, prior_X)   #new
    gradB.flat[::p+1] = gradBd
    gradB = gradB/n
    gradRho = gradRho/n
    gradPhi = gradPhi/n
    gradalphap = gradalphap/n
    gradalphaq = gradalphaq/n
    return gradB, gradRho, gradPhi, gradalphap, gradalphaq




def neglogli(B, Rho, Phi, alphap, alphaq, X, D, levels, groups_X, prior_X, use_prior):
    n = X.shape[0]
    p = B.shape[0]
    q = levels.shape[0]

    if use_prior == True:       #new
        M = transform_X(X, groups_X, prior_X)       #new
    else:                   #new
        M = np.ones((n,p))#new
    X_response = X.copy()  #new
    X = M * X       #new

    Bd = np.diag(B)
    B = B - np.diag(Bd)
    B = np.triu(B)
    B = B + np.transpose(B)
    DRho = np.dot(D, Rho)
    DRho = np.dot(DRho, np.diag(np.divide(np.repeat(1, p), Bd)))
    XB = np.dot(X, B)
    XB = np.dot(XB, np.diag(np.divide(np.repeat(1, p), Bd)))
    Xt = np.transpose(X)
    RhoX = np.dot(Rho, Xt)
    Phi = Phi - np.diag(np.diag(Phi))
    Phi = np.triu(Phi)
    Phi = Phi + np.transpose(Phi)
    Phirr = np.transpose(np.tile(alphaq, (n, 1)))
    PhiD = np.dot(Phi, np.transpose(D))
    levelSum = [0]
    levelSum.extend(levels)
    levelSum = np.cumsum(levelSum)
    consts = np.tile(alphap, (n, 1))
    PLcont1 = -n/2.*np.sum(np.log(-Bd)) #here, the constant term is neglected
    PLcont2 = consts + DRho + XB + X_response               #new! X to X_reponse
    PLcont2 = np.dot(PLcont2, np.diag(np.sqrt(-Bd)))
    PLcont2 = np.multiply(PLcont2, PLcont2)
    PLcont2 = 0.5*np.sum(np.sum(PLcont2, axis=0))
    temp = RhoX+Phirr+PhiD
    PLdisc = 0
    for r in range(0, q):
        temp2 = temp[int(levelSum[r]):int(levelSum[r]+levels[r]), :]
        denominator = np.sum(
            np.exp(np.dot(np.identity(int(levels[r])), temp2)), axis=0)
        numerator = np.sum(np.multiply(D[:, int(levelSum[r]):int(
            levelSum[r]+levels[r])], np.transpose(temp2)), axis=1)
        PLdisc = PLdisc-numerator+np.log(denominator)
    PLdisc = np.sum(PLdisc)
    return((PLcont1+PLcont2+PLdisc)/n)



def predict_categorical(B, Rho, Phi, alphap, alphaq, X, D, levels):
    n = X.shape[0]
    p = B.shape[0]
    q = levels.shape[0]
    X_response = X.copy()  
    Bd = np.diag(B)
    B = B - np.diag(Bd)
    B = np.triu(B)
    B = B + np.transpose(B)
    DRho = np.dot(D, Rho)
    DRho = np.dot(DRho, np.diag(np.divide(np.repeat(1, p), Bd)))
    XB = np.dot(X, B)
    XB = np.dot(XB, np.diag(np.divide(np.repeat(1, p), Bd)))
    Xt = np.transpose(X)
    RhoX = np.dot(Rho, Xt)
    Phi = Phi - np.diag(np.diag(Phi))
    Phi = np.triu(Phi)
    Phi = Phi + np.transpose(Phi)
    Phirr = np.transpose(np.tile(alphaq, (n, 1)))
    PhiD = np.dot(Phi, np.transpose(D))
    levelSum = [0]
    levelSum.extend(levels)
    levelSum = np.cumsum(levelSum)
    consts = np.tile(alphap, (n, 1))
    PLcont1 = -n/2.*np.sum(np.log(-Bd))
    PLcont2 = consts + DRho + XB + X_response            
    PLcont2 = np.dot(PLcont2, np.diag(np.sqrt(-Bd)))
    PLcont2 = np.multiply(PLcont2, PLcont2)
    PLcont2 = 0.5*np.sum(np.sum(PLcont2, axis=0))
    temp = RhoX+Phirr+PhiD
    PLdisc = 0
    for r in range(0, q):
        temp2 = temp[int(levelSum[r]):int(levelSum[r]+levels[r]), :]
        denominator = np.sum(
            np.exp(np.dot(np.identity(int(levels[r])), temp2)), axis=0)
        numerator = np.transpose(temp2)
        #numerator = np.sum(np.multiply(D[:, int(levelSum[r]):int(
        #    levelSum[r]+levels[r])], np.transpose(temp2)), axis=1)
        print(np.exp(numerator-np.log(denominator.reshape((denominator.shape[0],1)))))
        #PLdisc = PLdisc-numerator+np.log(denominator)
    #PLdisc = np.sum(PLdisc)
    #print(np.sum(PLdisc))
    #return(PLdisc)
    #return(np.exp(-PLdisc))


def neglogli_plain(B_Rho_Phi_alphap_alphaq, X, D, levels, p, q, groups_X, prior_X, use_prior):
    x2 = Inv_B_Rho_Phi_alphap_alphaq(B_Rho_Phi_alphap_alphaq, p, q)
    x1 = neglogli(x2[0], x2[1], x2[2], x2[3], x2[4], X, D, levels, groups_X, prior_X, use_prior)
    return(x1)


def B_Rho_Phi_alphap_alphaq(B, Rho, Phi, alphap, alphaq):
    p = B.shape[0]
    q = Phi.shape[0]
    sizes = np.cumsum([p*p, p*q, q*q, p, q])
    x = np.repeat(0., sizes[4])
    x[0:sizes[0]] = np.reshape(B, (1, p*p))[0, :]
    x[sizes[0]:sizes[1]] = np.reshape(Rho, (1, p*q))[0, :]
    x[sizes[1]:sizes[2]] = np.reshape(Phi, (1, q*q))[0, :]
    x[sizes[2]:sizes[3]] = alphap
    x[sizes[3]:sizes[4]] = alphaq
    return(x)


def Inv_B_Rho_Phi_alphap_alphaq(x, p, q):
    sizes = np.cumsum([p*p, p*q, q*q, p, q])
    B = np.reshape(x[0:sizes[0]], (p, p))
    Rho = np.reshape(x[sizes[0]:sizes[1]], (q, p))
    Phi = np.reshape(x[sizes[1]:sizes[2]], (q, q))
    alphap = x[sizes[2]:sizes[3]]
    alphaq = x[sizes[3]:sizes[4]]
    return(B, Rho, Phi, alphap, alphaq)


def grad_neglogli_plain(B_Rho_Phi_alphap_alphaq, X, D, levels, p, q, groups_X, prior_X, use_prior):
    x2 = Inv_B_Rho_Phi_alphap_alphaq(B_Rho_Phi_alphap_alphaq, p, q)
    x1 = grad_neglogli(x2[0], x2[1], x2[2], x2[3], x2[4], X, D, levels, groups_X, prior_X, use_prior)
    return(x1[0], x1[1], x1[2], x1[3], x1[4])


def make_starting_parameters(X, D, levels):
    p = X.shape[1]
    q = D.shape[1]
    B = -np.diag(np.repeat(1, p))
    Rho = np.zeros((q, p))
    Phi = np.zeros((q, q))
    alphap = np.zeros(p)
    alphaq = np.zeros(q)
    return(B, Rho, Phi, alphap, alphaq, p, q)


def grad_f_temp(x, X, D, levels, p, q, groups_X, prior_X, use_prior):
    x2 = grad_neglogli_plain(x, X, D, levels, p, q, groups_X, prior_X, use_prior)
    x3 = B_Rho_Phi_alphap_alphaq(x2[0], x2[1], x2[2], x2[3], x2[4])
    return(x3)


#def prox_enet(x, l_l1, l_l2, t, pen, p0, tol0):
#    prox_l1 = np.sign(x) * np.maximum(abs(x) - t * l_l1 * pen, 0)
#    return prox_l1 / (1. + t * l_l2 * pen)


# def make_penalty_factors(X, D, levels):
#     levelSum = [0]
#     levelSum.extend(levels)
#     levelSum = np.cumsum(levelSum)
#     p = X.shape[1]
#     qL = len(levels)
#     q = np.sum(levels)
#     sds = np.reshape(np.std(X, axis=0), (p, 1))
#     ps = np.reshape(np.mean(D, axis=0), (q, 1))
#     B = np.ones((p, p)) - np.diag(np.repeat(1, p))
#     B = np.multiply(B, np.dot(sds, np.transpose(sds)))
#     Rho = np.ones((q, p))
#     Phi = np.ones((q, q))
#     alphap = np.zeros(p)
#     alphaq = np.zeros(q)
#     for r in range(0, qL):
#         Phi[int(levelSum[r]), :] = np.repeat(10, Phi.shape[1])
#         Phi[:, int(levelSum[r])] = np.repeat(10, Phi.shape[1])
#         ps_t = ps[int(levelSum[r]):int(levelSum[r]+levels[r])]
#         ps[int(levelSum[r]):int(levelSum[r]+levels[r])
#            ] = np.sqrt(np.sum(ps_t*(1-ps_t)))
#         Rho[int(levelSum[r]), :] = np.repeat(10, Rho.shape[1])
#     Rho = np.multiply(Rho, np.dot(ps, np.transpose(sds)))
#     Phi = np.multiply(Phi, np.dot(ps, np.transpose(ps)))
#     return(B, Rho, Phi, alphap, alphaq, p, q)
#


def get_group_IDs_and_weights(IDs_B, IDs_rho, X, D, levels):
    p = X.shape[1]
    q = np.sum(levels)
    sds = np.ones((p,1))#np.reshape(np.std(X, axis=0), (p, 1))
    ps = np.reshape(np.mean(D, axis=0), (q, 1))
    ps = np.sqrt(ps*(1-ps))
    #B_cov = np.ones((p, p)) - np.diag(np.repeat(1, p))
    #B_cov = np.multiply(B_cov, np.dot(sds, np.transpose(sds)))
    Rho_cov = np.ones((q, p))
    Phi_cov = np.ones((q, q))
    Rho_cov = np.multiply(Rho_cov, np.dot(ps, np.transpose(sds)))
    Phi_cov = np.multiply(Phi_cov, np.dot(ps, np.transpose(ps)))


    #p = 0
    #for listElem in IDs_B:
    #   p += len(listElem)
    B = np.arange(0, p*p, 1).reshape(p, p)
    B = np.triu(B, 1)
    pairs_B = [np.ix_(i, j) for i in IDs_B for j in IDs_B]
    Blist = [[]]*len(pairs_B)
    Blist_weights = [[]]*len(pairs_B)
    j = 0
    for i in range(0, len(pairs_B)):
        B0 = B[pairs_B[i]].flatten()
        B0 = B0[B0 != 0]
        if B0.shape[0] > 0:
            Blist[j] = B0
            Blist_weights[j] = np.sqrt(B0.shape[0]) #only valid if X is standardized
            j = j + 1

    #q = 0
    #for listElem in IDs_rho:
    #    q += len(listElem)
    rho = np.arange(0, p*q, 1).reshape(q, p)
    pairs_rho = [np.ix_(i, j) for i in IDs_rho for j in IDs_B]
    rholist = [[]]*len(pairs_rho)
    rholist_weights = [[]]*len(pairs_rho)
    for i in range(0, len(pairs_rho)):
        rholist[i] = p*p + rho[pairs_rho[i]].flatten()
        rholist_weights[i] = np.sqrt(np.sum(Rho_cov[pairs_rho[i]]**2))

    phi = np.arange(0, q*q, 1).reshape(q, q)
    q0 = int(len(IDs_rho)*(len(IDs_rho)-1)/2)
    philist = [[]]*q0
    philist_weights = [[]]*q0
    k = 0
    for i in range(0, len(IDs_rho)):
        for j in range(i+1, len(IDs_rho)):
            philist[k] = p*p + q*p + phi[np.ix_(IDs_rho[i], IDs_rho[j])].flatten()
            philist_weights[k] = np.sqrt(np.sum(Phi_cov[np.ix_(IDs_rho[i], IDs_rho[j])]**2))
            k = k + 1

    return Blist + rholist + philist, Blist_weights + rholist_weights + philist_weights


def group_prox(x, t, groups, weights):
    x0 = x.copy()
    for g, w in zip(groups, weights):
        xn = np.linalg.norm(x0[g])
        r = t * np.array(w)
        if xn > r:
            x0[g] -= x0[g] * r / xn
        else:
            x0[g] = 0
    return x0

def Fit_MGM(X, D, levels, lambda_seq, iterations=1000, prior_X=None, groups_X=None, eps=1e-6):
    #X = (X - np.mean(X, axis=0)) / np.std(X, axis=0)   #add this line
    p = X.shape[1]
    q = D.shape[1]

    levelSum = [0]          #new
    levelSum.extend(levels)         #new
    levelSum = np.cumsum(levelSum)#new
    groups_Y = []#new
    for i in range(len(levels)):#new
        groups_Y = groups_Y + [list(levelSum[i] + range(levels[i]))]#new


    start = make_starting_parameters(X, D, levels)
    start = B_Rho_Phi_alphap_alphaq(
        start[0], start[1], start[2], start[3], start[4])


    if prior_X!=None:#new
        use_prior = True#new
        print('use prior')#new
    else:#new
        use_prior = False#new

    if groups_X==None:#new
        groups_X = []#new
        for i in range(p):#new
            groups_X = groups_X + [[i]]#new

    #print(groups_X)

    def grad_f(x): return grad_f_temp(x, X, D, levels, p, q, groups_X, prior_X, use_prior)   #remove x_start

    def fun(x): return neglogli_plain(x, X, D, levels, p, q, groups_X, prior_X, use_prior)   #remove x_start

    groups, weights = get_group_IDs_and_weights(groups_X, groups_Y, X, D, levels) 

    x_list = [None]*lambda_seq.shape[0]
    xtemp = start
    for i in range(0, lambda_seq.shape[0]):
        l_l1 = lambda_seq[i]
        print("lambda =", l_l1)

        #def prox_g(x, l): return prox_enet(
        #    x, l_l1, 0, t=l, pen=penalty, p0=p, tol0=eps)
        def prox_g(x, l): return group_prox(x, l*l_l1, groups, weights)

        x_fista = apg.solve(grad_f, prox_g, xtemp, eps=eps,
                            max_iters=iterations, gen_plots=False,
                            debug=False)
        x_list[i] = Inv_B_Rho_Phi_alphap_alphaq(x_fista, p, q)
        xtemp = x_fista
    loss_list = []
    k = 0
    for j in range(0, len(x_list)):
        k = k+1
        loss_vec = neglogli(
            x_list[j][0], x_list[j][1], x_list[j][2], x_list[j][3],
            x_list[j][4], X, D, levels, groups_X, prior_X, use_prior)
        loss_list = np.append(loss_list, loss_vec)
    return x_list, loss_list

def simulate_MGM(B, Rho, Phi, alphap, alphaq, levels, n_burning, n,
                blocked_continuous=None, blocked_categorical=None):

    p = B.shape[0]
    q = len(levels)

    if blocked_continuous==None:
        blocked_continuous =  [None] * p
    if blocked_categorical==None:
        blocked_categorical = [None] * q


    q0 = np.sum(levels)
    x_init = np.zeros(p)
    y_init = np.zeros(q0)
    lSum = [0]
    lSum.extend(levels)
    lSum = np.cumsum(lSum)

    for i in range(0, q):
        p_temp = [1./levels[i]]*levels[i]
        y_init[lSum[i]:lSum[i+1]] = np.random.multinomial(1, p_temp, size=1)

    X_mat = np.zeros((n_burning + n, p))
    Y_mat = np.zeros((n_burning + n, q0))
    x = x_init
    y = y_init
    for i in range(n_burning + n):
        for j in range(p):
            if blocked_continuous[j] is None:
                x_del = x
                x_del[j] = 0.
                shift = (alphap[j] + B[j, :] @ (x_del.reshape((p, 1)))
                     + (y.reshape((1, q0)) @ Rho[:, j]))/(- B[j, j])    # minus from redef. Bjj
                scale = 1./np.sqrt(- B[j, j])   # minus from redef. Bjj
                x[j] = np.random.normal(loc=shift, scale=scale, size=1)
                X_mat[i, j] = np.copy(x[j])
            else:
                X_mat[i, j] = blocked_continuous[j]
        for j in range(q):
            if blocked_categorical[j] is None:
                y_del = y
                y_del[lSum[j]:lSum[j+1]] = 0.
                t1 = Rho[lSum[j]:lSum[j+1], :]@x.reshape((p, 1))
                t2 = alphaq[lSum[j]:lSum[j+1]].reshape((levels[j],1))
                t3 = Phi[lSum[j]:lSum[j+1], :]@y_del.reshape((q0, 1))
                denom = np.sum(np.exp(t1 + t2 + t3))
                numer = np.exp(t1 + t2 + t3)
                p_temp = numer.flatten()/denom

                y[lSum[j]:lSum[j+1]] = np.random.multinomial(1, p_temp, size=1)
                Y_mat[i, lSum[j]:lSum[j+1]] = np.copy(y[lSum[j]:lSum[j+1]])
            else:
                Y_mat[i, lSum[j]:lSum[j+1]] = blocked_categorical[j]

    return(X_mat[n_burning:(n_burning + n), :],
           Y_mat[n_burning:(n_burning + n), :])


def baseline_transform_Rho(Rho, levels_Y):
    Rho0 = Rho.copy()
    q = np.sum(levels_Y)
    levels_cumsum = np.cumsum([0] + levels_Y)
    print()
    for i in range(len(levels_cumsum)-1):
        Rho_sub = Rho[levels_cumsum[i]:levels_cumsum[i+1],:]
        Rho_sub0 = Rho_sub[0,:].copy()
        for j in range(Rho_sub.shape[0]):
            Rho_sub[j,:] = Rho_sub[j,:] - Rho_sub0
        Rho0[levels_cumsum[i]:levels_cumsum[i+1],:] = Rho_sub
    return(Rho0)


def baseline_transform_Phi(Phi, levels_Y):
    Phi0 = Phi.copy()
    q = np.sum(levels_Y)
    levels_cumsum = np.cumsum([0] + levels_Y)
    for i in range(len(levels_cumsum)-1):
        for j in range(len(levels_cumsum)-1)[(i+1):]:
            Phi_sub = Phi[levels_cumsum[i]:levels_cumsum[i+1],
                          levels_cumsum[j]:levels_cumsum[j+1]]
            Phi_sub0 = Phi_sub[0,:].copy()
            for k in range(Phi_sub.shape[0]):
                Phi_sub[k,:] = Phi_sub[k,:] - Phi_sub0
            Phi_sub1 = Phi_sub[:,0].copy()
            for l in range(Phi_sub.shape[1]):
                Phi_sub[:,l] = Phi_sub[:,l] - Phi_sub1
            Phi0[levels_cumsum[i]:levels_cumsum[i+1],
                levels_cumsum[j]:levels_cumsum[j+1]] = Phi_sub
            Phi0[levels_cumsum[j]:levels_cumsum[j+1],
                levels_cumsum[i]:levels_cumsum[i+1]] = Phi_sub.T
    return(Phi0)


def Rho_Phi_flat(Rho, Phi, levels_Y):
    Phi0 = []
    Rho0 = []
    q = np.sum(levels_Y)
    levels_cumsum = np.cumsum([0] + levels_Y)
    for i in range(len(levels_cumsum)-1):
        Rho_sub = Rho[(levels_cumsum[i]+1):levels_cumsum[i+1],:].flatten().tolist()
        Rho0 = Rho0 + Rho_sub
        for j in range(len(levels_cumsum)-1)[(i+1):]:
            Phi_sub = Phi[(levels_cumsum[i]+1):levels_cumsum[i+1],
                          (levels_cumsum[j]+1):levels_cumsum[j+1]].flatten().tolist()
            Phi0 = Phi0 + Phi_sub
    return(np.array(Rho0), np.array(Phi0))


def make_simulation(p_X, levels_Y, n, etaB, etaRho, etaPhi, seed, n_burning = 1000):
    np.random.seed(seed)
    p = p_X
    q = np.sum(levels_Y)
    levels_cumsum = np.cumsum([0] + levels_Y)

    nB = int(p*(p-1)/2*etaB)
    nB_IDs = np.random.choice(range(int(p*(p-1)/2)), size=nB, replace=False)
    B = np.zeros((p,p))
    B_vec = B[np.triu_indices(p,1)]
    B_vec[nB_IDs] = np.random.uniform(-1., 1., size=len(nB_IDs))
    B[np.triu_indices(p,1)] = B_vec
    B = B + B.T


    IDs_Phi_all = []
    IDs_Phi = np.arange(q*q).reshape((q,q))
    for i in range(len(levels_cumsum)-1):
        for j in range(len(levels_cumsum)-1)[(i+1):]:
            temp_IDs = [IDs_Phi[(levels_cumsum[i]+1):levels_cumsum[i+1],
                          (levels_cumsum[j]+1):levels_cumsum[j+1]].flatten().tolist()]
            IDs_Phi_all = IDs_Phi_all + temp_IDs
    nPhi = int(len(IDs_Phi_all)*etaPhi)
    nPhi_IDs = np.random.choice(range(len(IDs_Phi_all)), size=nPhi, replace=False)
    Phi = np.zeros((q,q)).flatten()
    for i in range(len(nPhi_IDs)):
        ids = IDs_Phi_all[nPhi_IDs[i]]
        Phi[ids] = np.random.uniform(-1., 1., size=len(ids))
    Phi = Phi.reshape((q, q))
    Phi = Phi + Phi.T

    IDs_Rho_all = []
    IDs_Rho = np.arange(p*q).reshape((q, p))
    for i in range(len(levels_cumsum)-1):
        for j in range(p):
            temp_IDs = [IDs_Rho[(levels_cumsum[i]+1):levels_cumsum[i+1], j].flatten().tolist()]
            IDs_Rho_all = IDs_Rho_all + temp_IDs
    nRho = int(len(IDs_Rho_all)*etaRho)
    nRho_IDs = np.random.choice(range(len(IDs_Rho_all)), size=nRho,
                                replace=False)
    Rho = np.zeros((q, p)).flatten()
    for i in range(len(nRho_IDs)):
        ids = IDs_Rho_all[nRho_IDs[i]]
        Rho[ids] = np.random.uniform(-1., 1., size=len(ids))
    Rho = Rho.reshape((q, p))

    a = np.vstack((B, Rho))
    b = np.vstack((Rho.T , Phi))
    res = np.hstack((a, b))
    np.fill_diagonal(res, np.sum(np.abs(res), axis=0) + 0.0001)
    A = np.zeros((p+q,p+q)) +  np.sqrt(np.diag(res))
    res = res/A/A.T

    B = res[:p,:p]
    Rho = res[p:,:p]
    Phi = res[p:,p:]
    np.fill_diagonal(Phi, np.zeros(q))
    np.fill_diagonal(B, -(np.sum(np.abs(B - np.diag(np.diag(B))), axis=0) + 1))#-np.ones(p))
    def rescale(cov):
        p = cov.shape[0]
        A = np.sqrt(np.zeros((p,p))+np.diag(-cov))
        r = cov/A/A.T
        return(r)

    B = rescale(B)
    alphap = np.zeros(p)
    alphaq = np.zeros(q)

    X, Y = simulate_MGM(B, Rho, Phi, alphap, alphaq, levels_Y, n_burning, n)

    return(X, Y, B, Rho, Phi)

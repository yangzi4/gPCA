from numpy import *
from sklearn.decomposition import PCA
import scipy.special as sps
import scipy.linalg as sl

def gPCA_run(Xs__, D, normlz='SS'):
    "gPCA : statistics."
    #compute svds (K+1), errors (K+1), error2
    K = len(Xs__)
    pca = PCA(n_components=D)
    
    Xs_ = [Xs__[k] - mean(Xs__[k], axis=0) for k in range(K)]
    if normlz == 'Ns': Xs = [Xs_[k]/sqrt(shape(Xs_[k])[0]) for k in range(K)]
    if normlz == 'SS': Xs = [Xs_[k]/linalg.norm(Xs_[k]) for k in range(K)]
    Xf = vstack(Xs)
    
    hs, est_s = [], []
    for k in range(K):
        pca.fit(Xs[k])
        hs.append(pca.components_)
        est_s.append(pca.inverse_transform(pca.transform(Xs[k])))
    est_sf = vstack(est_s)
    pca.fit(Xf)
    hf = pca.components_
    est_f = pca.inverse_transform(pca.transform(Xf))
    var_expl_f, noise_var_f = pca.explained_variance_ratio_, pca.noise_variance_
    
    pca.fit(est_sf)
    h2 = pca.components_
    est_2 = pca.inverse_transform(pca.transform(est_sf))
    
    err_total = linalg.norm(Xf - est_f)**2
    errs_within = [linalg.norm(Xs[k] - est_s[k])**2 for k in range(K)]
    err_between = linalg.norm(est_sf - est_2)**2
    #print err_total, errs_within, err_between, 'err:', err_total - sum(errs_within) - err_between
    alphas = []
    for k in range(K - 1):
        for k_ in range(k + 1, K):
            alphas.append(sqrt(min(min(linalg.svd(hs[k].dot(hs[k_].T))[1]), 1.0)))
    return [[errs_within, err_between, err_total], mean(alphas), [hf, hs, h2], [var_expl_f, noise_var_f]]

def gPCA_bootstrap(Xs, D, n_boot=1000, verbose=0):
    "gPCA : bootstrap."
    K = len(Xs)
    Ns, M = [shape(Xs[k])[0] for k in range(K)], shape(Xs[0])[1]
    ratio_list, alpha_list, sigma_list = [], [], []
    for rep in range(n_boot):
        tries = 0
        while tries < 10:
            try:
                Xs_btsp = [Xs[k][random.randint(Ns[k], size=Ns[k]), :] for k in range(K)]
                res = gPCA_run(Xs_btsp, D)
                break
            except (linalg.LinAlgError):
                tries += 1
                print rep, ":", tries, D, "LinAlgError"
        ratio_list.append(sum(res[0][0])/(sum(res[0][0]) + res[0][1]))
        alpha_list.append(res[1])
        sigma_list.append(sqrt(sum(res[0][0])/res[0][1]*(1 - res[1]**2)*D*(K - 1)/K/(M - D)))
        if verbose: print rep, ratio_list[-1], alpha_list[-1], sigma_list[-1], ':', mean(alpha_list), std(alpha_list)/sqrt(len(alpha_list))
    return ratio_list, alpha_list, sigma_list

def gPCA_select(Xs, D_range=range(1, 5), vers=1, n_boot_=1000, verbose=0):
    "gPCA : rank selection."
    K = len(Xs)
    Ns, M = [shape(Xs[k])[0] for k in range(K)], shape(Xs[0])[1]
    ratios, alphas, sigmas = [], [], []
    for D_ in D_range:
        res_D = gPCA_bootstrap(Xs, D_, n_boot=n_boot_, verbose=verbose)
        ratios.append(mean(res_D[0]))
        alphas.append(mean(res_D[1]))
        sigmas.append(mean(res_D[2]))
        if verbose: print '', D_, ratios[-1], alphas[-1], sigmas[-1]
    if vers == 1: idx = nanargmin(sigmas)
    if vers == 2: idx = argmax(alphas)
    if verbose:
        print ratios
        print " " + str(alphas)
        print " " + str(sigmas)
    res = gPCA_run(Xs, D_range[idx])
    #print res[3][0]
    return ratios, alphas, sigmas, idx, res[2], res[3]

def LP_evid(X_, dim_):
    "Evidence for Laplace method."
    N, M = shape(X_)
    pca = PCA(n_components=dim_)
    pca.fit(X_)
    sigma2 = pca.noise_variance_
    
    ls = list(sl.eigh(X_.T.dot(X_))[0][::-1])
    ls_ = [ls[d] for d in range(dim_)] + [sigma2 for d in range(dim_, M)]
    l_th = (sum([log(ls[d]) for d in range(dim_)]) + (M - dim_)*log(sigma2))*N/2.0
    const = M*dim_ - dim_*(dim_ + 1)/2.0
    Az = prod([prod([N*(1.0/ls_[j] - 1.0/ls_[d])*(ls[d] - ls[j])
        for j in range(d + 1, M)]) for d in range(dim_)])
    term3 = -(const + dim_)*log(2*pi) + log(Az) + dim_*log(N)
    prob = (2**-dim_)*prod([sps.gamma((M - d)/2.0)*pi**(-(M - d)/2.0)
        for d in range(dim_)])
    #print l_th, Az, prob
    return l_th - log(prob) + term3/2.0

def BIC_evid(X_, dim_):
    "Evidence for BIC method."
    N, M = shape(X_)
    pca = PCA(n_components=dim_)
    pca.fit(X_)
    sigma2 = pca.noise_variance_
    
    ls = sl.eigh(X_.T.dot(X_), eigvals=(M - dim_, M - 1))[0]
    l_th = (sum([log(ls[d]) for d in range(dim_)]) + (M - dim_)*log(sigma2))*N/2.0
    const = M*dim_ - dim_*(dim_ + 1)/2.0
    return l_th + (const + dim_)*log(N)/2.0

def evid_select(Xs, D_range=range(1, 5), criterion=0, verbose=0):
    "Evidence-based rank selection for PCA."
    K = len(Xs)
    evids = [[] for k in range(K)]
    idcs = []
    for k in range(K):
        for D_ in D_range:
            if criterion == 'LP': evids[k].append(LP_evid(Xs[k], D_))
            elif criterion == 'BIC': evids[k].append(BIC_evid(Xs[k], D_))
            if verbose: print criterion, D_, evids[k][-1]
        if criterion in ['LP', 'BIC']: idcs.append(argmin(evids[k]))
        else: print 'invalid criterion'
    return [evids[k][idcs[k]] for k in range(K)], [D_range[idcs[k]] for k in range(K)]

def KN_sigma2(evals, dim_, N, M):
    "Estimation for KN method."
    #print evals
    s2 = mean(evals[dim_:])/(1 - dim_/float(N))
    rhos = [roots([1, -(evals[d] + s2*(1 - (M - dim_)/float(N))), evals[d]*s2])[0] for d in range(dim_)]
    diff = inf
    count = 0
    while diff > 1e-5 and count < 1e3:
        #print s2, rhos, mean(evals[dim_:]), diff, count
        new_s2 = (sum(evals[dim_:]) + sum([evals[d] - rhos[d] for d in range(dim_)]))/float(M - dim_)
        rhos = [roots([1, -(evals[d] + new_s2*(1 - (M - dim_)/float(N))), evals[d]*new_s2])[0] for d in range(dim_)]
        diff = abs(new_s2 - s2)
        s2 = new_s2
        count += 1
    return real(s2)

def KN_hypo(X_, d_, evals_):
    "Hypothesis test for KN method."
    N, M = shape(X_)
    m_Np = (sqrt(N - 0.5) + sqrt(M - d_ - 0.5))**2/float(N)
    s_Np = (sqrt(N - 0.5) + sqrt(M - d_ - 0.5))*(1/sqrt(N - 0.5) + 1/sqrt(M - d_ - 0.5))**(1/3.0)/float(N)
    s2_KN = KN_sigma2(evals_, d_, N, M)
    #print evals_[d_ - 1], s2_KN, m_Np, s_Np
    return (evals_[d_ - 1]/s2_KN - m_Np)/s_Np

def hypo_select(Xs, D_range=range(1, 5), hypo_test=0, alpha_level=0.05, verbose=0):
    "Hypothesis-based rank selection for PCA."
    K = len(Xs)
    Ns, M = [shape(Xs[k])[0] for k in range(K)], shape(Xs[0])[1]
    pvals, D_chs = [[] for k in range(K)], [D_range[0] - 1 for k in range(K)]
    for k in range(K):
        for d in D_range:
            if hypo_test == 'KG':
                evalues = list(sl.eigh(Xs[k].T.dot(Xs[k])/float(Ns[k]))[0][::-1])
                pval = evalues[d - 1]
                alpha_level = mean(evalues)
                #print evalues[:6], evalues[-6:], mean(evalues)
            if hypo_test == 'KN':
                evalues = list(sl.eigh(Xs[k].T.dot(Xs[k])/float(Ns[k]))[0][::-1])
                pval = KN_hypo(Xs[k], d, evalues)
                alpha_level = 2.422  ## s(0.005), s(0.05): 2.422, 0.979
            pvals[k].append(pval)
            if pval > alpha_level: D_chs[k] += 1
            else: break
        D_chs[k] = max([D_chs[k], D_range[0]])
    return pvals, D_chs

def calc_alpha_pval(alpha, M, D, kappa_start=10, kappa_lim=1e6, verbose=0):
    "Calculates one-sided p-value for alpha estimate (equal or larger alpha)."
    theta = arccos(alpha**2)
    if M > 50: gm_rto = (2.0/(M - 1))**(D/2.0)
    else: gm_rto = sps.gamma((M - D + 1)/2.0)/sps.gamma((M + 1)/2.0)
    term1 = sps.gamma((D + 1)/2.0)*gm_rto/sps.gamma(0.5)
    term2 = sin(theta)**(D*(M - D))
    
    kappa = kappa_start
    old_val, new_val = 0, 0
    while abs(old_val - new_val) >= 1e-3*old_val and kappa < kappa_lim:
        old_val = new_val
        new_val = ghi_m([kappa]*2, 2, [(M - D)/2.0, 0.5], [(M + 1)/2.0], D, [sin(theta)**2])[0][0]
        if verbose: print around([kappa, old_val, new_val, abs(old_val - new_val)], 3)
        kappa = int(kappa*sqrt(10))
    if kappa >= kappa_lim: print 'kappa limit reached'
    #print '', term1, term2, new_val
    return term1*term2*new_val

def calc_alpha_pval2(alpha, Xs, D, n_boot=1000, k_range='all', verbose=0):
    "Calculates one-sided p-value for alpha estimate (equal or smaller alpha)."
    K = len(Xs)
    Ns, M = [shape(Xs[k])[0] for k in range(K)], shape(Xs[0])[1]
    if k_range == 'all': k_range = range(K)
    
    ratio_list, alpha_list, sigma_list = [], [], []
    for k in k_range:
        for rep in range(n_boot):
            tries = 0
            while tries < 10:
                try:
                    Xs_btsp = split(Xs[k][random.randint(Ns[k], size=sum(Ns))],##
                        [sum(Ns[:k]) for k in range(1, K)])
                    #Xs_btsp = split(vstack(Xs)[random.randint(sum(Ns), size=sum(Ns))],##
                    #    [sum(Ns[:k]) for k in range(1, K)])
                    res = gPCA_run(Xs_btsp, D)
                    break
                except (linalg.LinAlgError):
                    tries += 1
                    print rep, ":", tries, D, "LinAlgError"
            ratio_list.append(sum(res[0][0])/(sum(res[0][0]) + res[0][1]))
            alpha_list.append(res[1])
            sigma_list.append(sqrt(sum(res[0][0])/res[0][1]*(1 - res[1]**2)*D*(K - 1)/K/(M - D)))
            if verbose: print [k, rep], ratio_list[-1], alpha_list[-1], sigma_list[-1]
    return sum(array(alpha_list) <= alpha)/float(len(alpha_list)), alpha_list

def ghi_m(max_kappa, a_const, a_vec, b_vec, dim, mat_const):
    """Evaluates Gaussian hypergeometric function with matrix arguments.
    Adapted from mghi.c (Koev and Edelman, 2006)"""
    n_mat, n_a, n_b = len(mat_const), len(a_vec), len(b_vec)
    s_ = [val for val in mat_const]
    ss_ = [1] + [0]*max_kappa[0]
    l_ = [max_kappa[1]] + [0]*dim
    z_ = [0,1] + [0]*(dim - 1)
    kt_ = [-x for x in range(dim+1)]
    
    sl, i = 0, 1
    while i > 0:
        if (sl < max_kappa[0]) and (l_[i] < l_[i-1]) and (z_[i] != 0):
            l_[i] += 1
            c = (kt_[i] + 1)/float(a_const)
            zn = dim + kt_[i] + 1
            dn = l_[i]*(kt_[i] + i + 1)
            
            for j in range(n_a): zn *= a_vec[j] + c
            for j in range(n_b): dn *= b_vec[j] + c
            
            kt_[i] += a_const
            for j in range(1, i):
                c = kt_[j] - kt_[i]
                zn *= c*(c + a_const - 1)
                dn *= (c + 1)*(c + a_const)
            
            z_[i] *= zn/float(dn)
            if i < dim:
                z_[i + 1] = z_[i]
                i += 1
            sl += 1
            ss_[sl] += z_[i]
        else:
            sl -= l_[i]
            l_[i] = 0
            kt_[i] =-i
            i -= 1
    
    for j in range(n_mat):
        s_[j] = 0
        for i in range(max_kappa[0], -1, -1): s_[j] = ss_[i] + s_[j]*mat_const[j]
    return s_, ss_

def group_data_gen(dims, D, sigma, W_=0, H_=0):
    """Data generation for matrix decomposition problems (+/-).
    dims: [N M] or [[Ns] [Ms]], etc."""
    if type(dims[0]) == int:
        if type(dims[1]) == int:
            N, M = dims
            W = random.normal(0, 1, (N, D))
            if type(W_) != int: W = W_
            H = random.normal(0, 1, (D, M))
            H = (H.T/linalg.norm(H, axis=1)).T
            if type(H_) != int: H = H_
            E = random.normal(0, sigma, (N, M))
            X = W.dot(H) + E
            return X, W, H, E
        if type(dims[1]) == list:
            N, Ms = dims
            L = len(Ms)
            Ws = [random.normal(0, 1, (N, D)) for l in range(L)]
            if type(W_) != int: Ws = [W_[l] for l in range(L)]
            Hs = [random.normal(0, 1, (D, Ms[l])) for l in range(L)]
            Hs = [(Hs[l].T/linalg.norm(Hs[l], axis=1)).T for l in range(L)]
            if type(H_) != int: Hs = [H_[l] for l in range(L)]
            Es = [random.normal(0, sigma, (N, Ms[l])) for l in range(L)]
            Xs = [Ws[l].dot(Hs[l]) + Es[l] for l in range(L)]
            return Xs, Ws, Hs, Es
    if type(dims[0]) == list:
        if type(dims[1]) == int:
            Ns, M = dims
            K = len(Ns)
            Ws = [random.normal(0, 1, (Ns[k], D)) for k in range(K)]
            if type(W_) != int: Ws = [W_[k] for k in range(K)]
            Hs = [random.normal(0, 1, (D, M)) for k in range(K)]
            Hs = [(Hs[k].T/linalg.norm(Hs[k], axis=1)).T for k in range(K)]
            if type(H_) != int: Hs = [H_[k] for k in range(K)]
            Es = [random.normal(0, sigma, (Ns[k], M)) for k in range(K)]
            Xs = [Ws[k].dot(Hs[k]) + Es[k] for k in range(K)]
            return Xs, Ws, Hs, Es
        if type(dims[1]) == list:
            Ns, Ms = dims
            K, L = len(Ns), len(Ms)
            Ws = [random.normal(0, 1, (Ns[k], D)) for k in range(K)]
            if type(W_) != int: Ws = [W_[k] for k in range(K)]
            Hs = [random.normal(0, 1, (D, Ms[l])) for l in range(L)]
            Hs = [(Hs[l].T/linalg.norm(Hs[l], axis=1)).T for l in range(L)]
            if type(H_) != int: Hs = [H_[k] for k in range(K)]
            Es = [[random.normal(0, sigma, (Ns[k], Ms[l])) for k in range(K)]
                for l in range(L)]
            Xs = [[Ws[k].dot(Hs[l]) + Es[l][k] for k in range(K)] for l in range(L)]
            return Xs, Ws, Hs, Es
    return


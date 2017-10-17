## requires 'gPCA base.py'

import os

D_range_ = range(1, 5)
Dest, Pval, save_data = 1, 1, 0

for gen_seed in range(0, 25)[:]:
    Dest_listlist = []  ## list (3 dim) of list (11 alpha) of list (6 method)
    Pval_listlist = []  ## list (3 dim) of list (11 alpha) of list (2 pvals)
    for param in [1, 2, 3, 4]:  ## M >= (K + 1)*n_vars*D_t
        grp_N, M = 13, 16
        K = 3
        D_t = param
        sigma_ = 0.1
        n_vars = 1
        dims_c = [grp_N*K, M]
        dims_d = [[grp_N]*K, M]
        
        d_weight = diag([1]*D_t)  ## dimensional weights (non-increasing)
        signal = hstack([hstack([eye(D_t)[:, d:(d+1)]]*n_vars) for d in range(D_t)])
        w_c = d_weight.dot(hstack((signal, zeros((D_t, M - D_t*n_vars))))/sqrt(n_vars))
        w_d = [d_weight.dot(hstack([zeros((D_t, D_t*n_vars))]*(k + 1) + [signal] +
            [zeros((D_t, M - D_t*n_vars*(k + 2)))])/sqrt(n_vars)) for k in range(K)]
        
        Dest_list = []
        Pval_list = []
        alpha_range = arange(0, 1.01, 0.1)
        print '___', grp_N, K, M, D_t, sigma_, n_vars, '___'
        for alpha in alpha_range:
            if (Dest or Pval) == 0: continue
            random.seed(gen_seed)
            w_cd = [alpha*w_c + sqrt(1 - alpha**2)*w_d[k] for k in range(K)]
            X_cd_ = group_data_gen(dims_d, D_t, sigma_, H_=w_cd)
            X_cd, X_cd_ne, X_cd_e = X_cd_[0], [X_cd_[1][k].dot(X_cd_[2][k]) for k in range(K)], X_cd_[3]
            
            if save_data:
                for k in range(K): savetxt('data/data(%dx%d)x%d_a%dD%ds%d_%d.txt' %
                    (grp_N, M, k, alpha*100, D_t, sigma_*100, gen_seed), X_cd[k])
            
            random.seed(1)
            res_gPCA = gPCA_select(X_cd, D_range=D_range_, vers=1)
            #res_gPCA2 = gPCA_select(X_cd, D_range=D_range_, vers=2)
            #res_LP = evid_select([vstack(X_cd)], D_range=D_range_, criterion='LP', verbose=0)
            #res_BIC = evid_select([vstack(X_cd)], D_range=D_range_, criterion='BIC', verbose=0)
            #res_KG = hypo_select([vstack(X_cd)], D_range=D_range_, hypo_test='KG', verbose=0)
            #res_KN = hypo_select([vstack(X_cd)], D_range=D_range_, hypo_test='KN', verbose=0)
            
            Dest_list.append([
                D_t == D_range_[res_gPCA[3]]#,
                #D_t == D_range_[res_gPCA2[3]],
                ##mean([D_t == res_LP[1][k] for k in range(K)]),
                #D_t == res_LP[1][0],
                ##mean([D_t == res_BIC[1][k] for k in range(K)]),
                #D_t == res_BIC[1][0],
                ##mean([D_t == res_KG[1][k] for k in range(K)]),
                #D_t == res_KG[1][0],
                ##mean([D_t == res_KN[1][k] for k in range(K)]),
                #D_t == res_KN[1][0],
                ]*6)
            if Dest: print 'Dest:', Dest_list[-1]
            Pval_list.append([
                res_gPCA[1][res_gPCA[3]],
                calc_alpha_pval(res_gPCA[1][res_gPCA[3]], M, D_range_[res_gPCA[3]]),
                calc_alpha_pval2(res_gPCA[1][res_gPCA[3]], X_cd, D_range_[res_gPCA[3]])[0]#,
                #res_gPCA2[1][res_gPCA2[3]],
                #calc_alpha_pval(res_gPCA2[1][res_gPCA2[3]], M, D_range_[res_gPCA2[3]]),
                #calc_alpha_pval2(res_gPCA2[1][res_gPCA2[3]], X_cd, D_range_[res_gPCA2[3]])[0],
            ]*2)
            if Pval: print 'Pval:', around(array(Pval_list[-1]), 3)
        Dest_listlist.append(Dest_list)
        Pval_listlist.append(Pval_list)
        
    if Dest: save('Dest(%dx%d)x%d_D%ds%d_%d' % (grp_N, M, K, D_t, sigma_*100, gen_seed),
        array(Dest_listlist))
    if Pval: save('Pval(%dx%d)x%d_D%ds%d_%d' % (grp_N, M, K, D_t, sigma_*100, gen_seed),
        array(Pval_listlist))


#os.chdir('C:/~')

def plot_Dest(p1, p2, p3, p4, svfg_=0, v_='', optn=0, lgnd=0):
    ## retrieve
    param_list = [1, 2, 3, 4]
    for param in param_list:
        grp_N, M = p1, p2
        K = p3
        D_t = param
        sigma_ = p4
    ## needs correct directory
    vals_m = zeros((len(param_list), len(alpha_range), 6))
    for g_seed in gen_seed_list:
        vals_m += load('Dest%s(%dx%d)x%d_D%ds%d_%d.npy' % (
            v_, grp_N, M, K, D_t, sigma_*100, g_seed))
    vals = vals_m/n_s  ## 3(dim) x 11(alpha) x 6(method) array
    vals_sd = zeros((len(param_list), len(alpha_range), 6))
    for g_seed in gen_seed_list:
        load_ = load('Dest%s(%dx%d)x%d_D%ds%d_%d.npy' % (
            v_, grp_N, M, K, D_t, sigma_*100, g_seed))
        vals_sd += (load_ - vals)**2
    vals2 = vals_sd/n_s
    
    #os.chdir('C:/~')
    #JIVE_vals = loadtxt('DestJIVE(%dx%d)x%d_D%ds%d_%d.txt' % (grp_N, M, K,
    #    D_t, sigma_*100, len(gen_seed_list)))
    
    for i in range(0, 6): print '\t'.join(list(around(mean(vals[:, :, i], axis=0), 2).astype(str)))
    
    ftsz1, ftsz2, ftsz3 = 16, 16, 16  ## title x legend
    plt.figure()#figsize=(9, 8)
    plt.subplot(1, 1, 1)
    plt.plot(alpha_range, mean(vals[:, :, 0], axis=0), 'cv', ms=12)  ## gPCA-L
    plt.plot(alpha_range, mean(vals[:, :, 1], axis=0), 'c^', ms=12)  ## gPCA-H
    #plt.plot(alpha_range, sum(JIVE_vals, axis=0)/125.0, 'm^', ms=12)  ## JIVE
    plt.plot(alpha_range, mean(vals[:, :, 2], axis=0), 'bs', ms=12)  ## LP
    plt.plot(alpha_range, mean(vals[:, :, 3], axis=0), 'ro', ms=12)  ## BIC
    plt.plot(alpha_range, mean(vals[:, :, 4], axis=0), 'gd', ms=12)  ## KG
    plt.plot(alpha_range, mean(vals[:, :, 5], axis=0), 'k*', ms=12)  ## NK
    plt.errorbar(alpha_range, mean(vals[:, :, 0], axis=0),
        yerr=sqrt(mean(vals2[:, :, 0], axis=0))/sqrt(n_s), color='c', ms=12)
    plt.errorbar(alpha_range, mean(vals[:, :, 1], axis=0),
        yerr=sqrt(mean(vals2[:, :, 1], axis=0))/sqrt(n_s), color='c', ms=12)
    #plt.errorbar(alpha_range, sum(JIVE_vals, axis=0)/75.0,
    #    yerr=sqrt(mean(vals2[:, :, 0], axis=0))/sqrt(n_s), color='m', ms=12)
    plt.errorbar(alpha_range, mean(vals[:, :, 2], axis=0),
        yerr=sqrt(mean(vals2[:, :, 2], axis=0))/sqrt(n_s), color='b', ms=12)
    plt.errorbar(alpha_range, mean(vals[:, :, 3], axis=0),
        yerr=sqrt(mean(vals2[:, :, 3], axis=0))/sqrt(n_s), color='r', ms=12)
    plt.errorbar(alpha_range, mean(vals[:, :, 4], axis=0),
        yerr=sqrt(mean(vals2[:, :, 4], axis=0))/sqrt(n_s), color='g', ms=12)
    plt.errorbar(alpha_range, mean(vals[:, :, 5], axis=0),
        yerr=sqrt(mean(vals2[:, :, 5], axis=0))/sqrt(n_s), color='k', ms=12)
    plt.plot(alpha_range, mean(vals[:, :, 0], axis=0), 'c--', ms=12)
    plt.plot(alpha_range, mean(vals[:, :, 1], axis=0), 'c--', ms=12)
    #plt.plot(alpha_range, sum(JIVE_vals, axis=0)/125.0, 'm--', ms=12)
    plt.plot(alpha_range, mean(vals[:, :, 2], axis=0), 'b--', ms=12)
    plt.plot(alpha_range, mean(vals[:, :, 3], axis=0), 'r--', ms=12)
    plt.plot(alpha_range, mean(vals[:, :, 4], axis=0), 'g--', ms=12)
    plt.plot(alpha_range, mean(vals[:, :, 5], axis=0), 'k--', ms=12)
    plt.xlabel(r'$\alpha$', fontsize=ftsz2)
    plt.ylim(-0.1, 1.1)
    if lgnd: plt.legend(['gPCA', 'JIVE', 'LP', 'BIC', 'KG', 'KN'], loc='upper left', fontsize=ftsz3)
    #plt.title(r'%s: dim($X_k$)$=$(%dx%d)*%d, $\sigma = $ %.2f' % (optn, grp_N, M, K, sigma_), fontsize=ftsz2)
    plt.title('%s: Proportion of correct rank selections' % optn, fontsize=ftsz1)
    if svfg_: plt.savefig('Dest%s(%dx%d)x%d_D%ds%d_all%d.png' % (v_, grp_N, M, K,
        D_t, sigma_*100, gen_seed_list[-1]), format='png', bbox_inches='tight')#dpi=150
    return

def plot_Pval(p1, p2, p3, p4, svfg_=0, v_='', optn=0, lgnd=0):
    ## retrieve
    param_list = [1, 2, 3, 4]
    for param in param_list:
        grp_N, M = p1, p2
        K = p3
        D_t = param
        sigma_ = p4
    ## needs correct directory
    vals_m = zeros((len(param_list), len(alpha_range), 6))
    for g_seed in gen_seed_list:
        vals_m += load('Pval%s(%dx%d)x%d_D%ds%d_%d.npy' % (
            v_, grp_N, M, K, D_t, sigma_*100, g_seed))
    vals = vals_m/n_s  ## 3(dim) x 11(alpha) x 6(method) array
    vals_sd = zeros((len(param_list), len(alpha_range), 6))
    for g_seed in gen_seed_list:
        load_ = load('Pval%s(%dx%d)x%d_D%ds%d_%d.npy' % (
            v_, grp_N, M, K, D_t, sigma_*100, g_seed))
        vals_sd += (load_ - vals)**2
    vals2 = sqrt(vals_sd/n_s)
    
    #os.chdir('C:/Users/ZI/Desktop/Work/Research/Code/gPCA/rank selection 3 (main)')
    #JIVE_vals = loadtxt('DestJIVE(%dx%d)x%d_D%ds%d_%d.txt' % (grp_N, M, K,
    #    D_t, sigma_*100, len(gen_seed_list)))
    
    for i in range(0, 6): print '\t'.join(list(around(mean(vals[:, :, i], axis=0), 2).astype(str)))
    
    ftsz1, ftsz2, ftsz3 = 16, 16, 16  ## title x legend
    plt.figure()#figsize=(9, 8)
    plt.subplot(1, 1, 1)
    plt.plot(alpha_range, mean(vals[:, :, 0], axis=0), 'cv', ms=12)  ## gPCA-L
    plt.plot(alpha_range, mean(vals[:, :, 1], axis=0), 'c^', ms=12)  ## gPCA-H
    #plt.plot(alpha_range, sum(JIVE_vals, axis=0)/125.0, 'm^', ms=12)  ## JIVE
    plt.plot(alpha_range, mean(vals[:, :, 2], axis=0), 'bs', ms=12)  ## LP
    plt.plot(alpha_range, mean(vals[:, :, 3], axis=0), 'ro', ms=12)  ## BIC
    plt.plot(alpha_range, mean(vals[:, :, 4], axis=0), 'gd', ms=12)  ## KG
    plt.plot(alpha_range, mean(vals[:, :, 5], axis=0), 'k*', ms=12)  ## NK
    plt.errorbar(alpha_range, mean(vals[:, :, 0], axis=0),
        yerr=sqrt(mean(vals2[:, :, 0], axis=0))/sqrt(n_s), color='c', ms=12)
    plt.errorbar(alpha_range, mean(vals[:, :, 1], axis=0),
        yerr=sqrt(mean(vals2[:, :, 1], axis=0))/sqrt(n_s), color='c', ms=12)
    #plt.errorbar(alpha_range, sum(JIVE_vals, axis=0)/75.0,
    #    yerr=sqrt(mean(vals2[:, :, 0], axis=0))/sqrt(n_s), color='m', ms=12)
    plt.errorbar(alpha_range, mean(vals[:, :, 2], axis=0),
        yerr=sqrt(mean(vals2[:, :, 2], axis=0))/sqrt(n_s), color='b', ms=12)
    plt.errorbar(alpha_range, mean(vals[:, :, 3], axis=0),
        yerr=sqrt(mean(vals2[:, :, 3], axis=0))/sqrt(n_s), color='r', ms=12)
    plt.errorbar(alpha_range, mean(vals[:, :, 4], axis=0),
        yerr=sqrt(mean(vals2[:, :, 4], axis=0))/sqrt(n_s), color='g', ms=12)
    plt.errorbar(alpha_range, mean(vals[:, :, 5], axis=0),
        yerr=sqrt(mean(vals2[:, :, 5], axis=0))/sqrt(n_s), color='k', ms=12)
    plt.plot(alpha_range, mean(vals[:, :, 0], axis=0), 'c--', ms=12)
    plt.plot(alpha_range, mean(vals[:, :, 1], axis=0), 'c--', ms=12)
    #plt.plot(alpha_range, sum(JIVE_vals, axis=0)/125.0, 'm--', ms=12)
    plt.plot(alpha_range, mean(vals[:, :, 2], axis=0), 'b--', ms=12)
    plt.plot(alpha_range, mean(vals[:, :, 3], axis=0), 'r--', ms=12)
    plt.plot(alpha_range, mean(vals[:, :, 4], axis=0), 'g--', ms=12)
    plt.plot(alpha_range, mean(vals[:, :, 5], axis=0), 'k--', ms=12)
    plt.xlabel(r'$\alpha$', fontsize=ftsz2)
    plt.ylim(-0.1, 1.1)
    if lgnd: plt.legend([r'$\alpha$', r'$p_{\alpha 1}$',  r'$p_{\alpha 2}$',
        r'$\alpha$', r'$p_{\alpha 1}$',  r'$p_{\alpha 2}$'], loc='upper left', fontsize=ftsz3)
    #plt.title(r'%s: dim($X_k$)$=$(%dx%d)*%d, $\sigma = $ %.2f' % (optn, grp_N, M, K, sigma_), fontsize=ftsz2)
    plt.title('%s: alpha, p_alpha1, p_alpha2' % optn, fontsize=ftsz1)
    if svfg_: plt.savefig('Pval%s(%dx%d)x%d_D%ds%d_all%d.png' % (v_, grp_N, M, K,
        D_t, sigma_*100, gen_seed_list[-1]), format='png', bbox_inches='tight')#dpi=150
    return

gen_seed_list = array(range(25))
n_s = float(len(gen_seed_list))
alpha_range = arange(0, 1.01, 0.1)

if 1:
    plt.close('all')
    svfg = 1
    plot_Dest(13, 16, 3, 0.1, svfg, '', '(a)', 1)
    plot_Pval(13, 16, 3, 0.1, svfg, '', '(b)', 1)
    plot_Dest(39, 16, 2, 0.1, svfg, '', '(a)', 1)
    plot_Pval(39, 16, 2, 0.1, svfg, '', '(b)', 1)



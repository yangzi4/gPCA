## requires 'gPCA base.py'

import matplotlib.pyplot as plt
from sklearn import linear_model
import csv
import os
import pandas as pd
import itertools as it

set_printoptions(suppress=True)

## ___ functions ___
def readfile(f_name):
    "Reads file and extracts data."
    f = open(f_name, 'r')
    reader = csv.reader(f, delimiter=',')
    raw = []
    for idx, row in enumerate(reader):
        if idx == 0: variables = row
        else: raw.append(row)
    f.close()
    return [array(raw), variables]

def gPCA_analyze(data_, DD_=range(1, 5), rtn='ws'):
    "Retrieves PCs from gPCA."
    otp = gPCA_select(data_, D_range=DD_)  ## r a s idx h ve/nv
    rnk_est = DD_[otp[3]]
    alpha_est = otp[1][otp[3]]
    
    #print 'rank:', rnk_est
    if rtn == 'ws':
        return [hstack([otp[4][0][i:(i+1), :].T for i in range(rnk_est)]),  ## hf
            [hstack([otp[4][1][k][i:(i+1), :].T for i in range(rnk_est)]) for k in range(len(data_))],  ## hs
            otp[5][0]]  ## var explained
    elif rtn == 'alphas':
        return [rnk_est, round(alpha_est, 3),
            calc_alpha_pval(alpha_est, shape(data_[0])[1], rnk_est),
            calc_alpha_pval2(alpha_est, data_, rnk_est)[0],
            otp[0][otp[3]]]
    return

def get_unique_levels(df_1, col_names, order_=[]):
    """Retrieves unique levels of selected factors."""
    level_list = [list(set(df_1.ix[:, col_name])) for col_name in col_names]
    print [len(list_) for list_ in level_list]
    return level_list

def gPCA_comm(df_1, fctr_idx_, specs_=[[0], [0], [0], [0], [0]], ch_=1):
    """gPCA-based commonality analysis: choose factor variable (0-4) and
    specifications (lists)."""
    K = len(iv_lvl_list[fctr_idx_])
    non_fctr_specs = [i for i in range(5) if i != fctr_idx_]
    print 'factor :', iv_col_names[fctr_idx_]
    for i in non_fctr_specs:
        print str(iv_col_names[i]), ':', [iv_lvl_list[i][lvl] for lvl in specs_[i]]
    
    ## specification variable conditions
    conds = []
    for i in range(5):
        cond_lvl = [iv_lvl_list[i][spc] for spc in specs_[i]]
        conds.append(sum([df_1.ix[:, iv_col_names[i]] == lvl for lvl in cond_lvl], axis=0))
    full_cond = prod([conds[i] for i in non_fctr_specs], axis=0)
    
    ## factor variable conditions
    fctr_conds = [df_1.ix[:, iv_col_names[fctr_idx_]] == lvl for lvl in iv_lvl_list[fctr_idx_]]
    df_2s = [df_1[(full_cond > 0) & (fctr_conds[k] > 0)] for k in range(K)]
    Ns_ = [shape(df_2s[k])[0] for k in range(K)]
    
    ## factor variable levels
    specs_rng = [specs_[i] for i in non_fctr_specs]
    n_specs = [len(specs_rng[i]) for i in range(4)]
    fctr_lvls, spc_idcs = [], [0, 0, 0, 0]
    while spc_idcs[3] < n_specs[3]:
        fctr_lvls.append([specs_rng[i][spc_idcs[i]] for i in range(4)])
        spc_idcs[0] += 1
        if spc_idcs[0] >= n_specs[0]: spc_idcs[0:2] = [0, spc_idcs[1] + 1]
        if spc_idcs[1] >= n_specs[1]: spc_idcs[1:3] = [0, spc_idcs[2] + 1]
        if spc_idcs[2] >= n_specs[2]: spc_idcs[2:4] = [0, spc_idcs[3] + 1]
    n_lvls = len(fctr_lvls)
    
    ## data matrices
    obs_lists = [list(set(df_2s[k].ix[:, 0])) for k in range(K)]
    ds_3s = [zeros((len(obs_lists[k]), n_lvls)) for k in range(K)]
    for k in range(len(df_2s)):
        for n in range(Ns_[k]):
            obs = df_2s[k].iloc[n, 0]
            obs_idx = obs_lists[k].index(obs)
            lvls = [df_2s[k].iloc[n, range(1, 6)[i]] for i in non_fctr_specs]
            fctr_lvl = [iv_lvl_list[non_fctr_specs[i]].index(lvls[i]) for i in range(4)]
            fctr_idx = fctr_lvls.index(fctr_lvl)
            #print obs, lvls, obs_idx, fctr_idx
            if obs in obs_lists[k]:
                ds_3s[k][obs_idx, fctr_idx] = df_2s[k].iloc[n, 6]
    print 'dims: ', [shape(ds_3s[k]) for k in range(K)]
    
    ## gPCA
    if ch_ == 1:
        gPCA_res = gPCA_func([log10(ds_3s[k] + 1) for k in range(K)])
        print_gPCA(gPCA_res)
        return array(gPCA_res)
    elif ch_ == 2:
        random.seed(1)
        gPCA_res2_ = gPCA_analyze([log10(ds_3s[k] + 1) for k in range(K)], rtn='ws')
        gPCA_res2 = hstack([gPCA_res2_[1][k][:, :1] for k in range(K)]).T
        print gPCA_res2
        return array(gPCA_res2)
    elif ch_ == 3:
        return [ds_3s[k] for k in range(K)]

def gPCA_func(ds_):
    "gPCA function over groups."
    K = len(ds_)
    grps = [range(K)] + [comb for comb in it.combinations(range(K), 2)]
    res = []
    for grp in grps:
        random.seed(1)
        res.append(gPCA_analyze([ds_[g] for g in grp], rtn='alphas'))
    return res

def print_gPCA(res_):
    """Reader-friendly display of gPCA analysis output."""
    print 'cm.rank a*     p(a>=a*) p(a<=a*) WSS/(WSS+BSS)'
    for row_ in res_:
        msg = str(around(array(row_), 3))
        if row_[2] <= 0.01: msg += ' *** (common)'
        elif row_[2] <= 0.05: msg += ' * (common)'
        if row_[3] <= 0.01: msg += ' *** (distinct)'
        elif row_[3] <= 0.05: msg += ' * (distinct)'
        print msg
    return


## ___ read and process ___
cellline_list = ['184B5', 'BT-20', 'BT-549', 'HCC1187', 'HCC1395',
    'HCC1806', 'HCC1937', 'HCC38', 'HCC70', 'Hs 578T',
    'MCF 10A', 'MCF 10F', 'MCF12A', 'MDA-MB-157', 'MDA-MB-231',
    'MDA-MB-436', 'MDA-MB-453', 'MDA-MB-468', 'AU565', 'BT-474',
    'HCC1419', 'HCC1569', 'HCC1954', 'HCC202', 'MDA-MB-361',
    'SK-BR-3', 'UACC-812', 'UACC-893', 'ZR-75-30', 'BT-483',
    'CAMA-1', 'HCC1428', 'HCC1500', 'MCF7', 'MDA-MB-134-VI',
    'MDA-MB-175-VII', 'MDA-MB-415', 'T47D', 'ZR-75-1']
TN_HER2_HR_list = [0]*18 + [1]*11 + [2]*10
ligand_list = ['BTC', 'EFNA1', 'EGF', 'EPR', 'FGF-1',
    'FGF-2', 'HGF', 'HRG', 'IGF-I', 'IGF-II',
    'INS', 'NGF-beta', 'PDGF-BB', 'SCF', 'VEGF165']
conc_list = ['1', '100']
timemin_list = ['10', '30', '90']
kinase_list = ['pAKT', 'pERK']
BR_subtypes = ['TN', 'H2', 'HR']

meas_data, meas_var = readfile("dataset_20140_measured.csv")  ## 1209 x 6
print shape(meas_data)
## cell line (39 3), ligand (15); concentration (2), kinase (2), time (3)

basal_data_ = [0 for i in range(39)]
for row_ in range(shape(meas_data)[0]):
    if meas_data[row_, 4] in ['1', '100']: continue
    bsl_idx = cellline_list.index(meas_data[row_, 1])
    basal_data_[bsl_idx] = [meas_data[row_, 1], meas_data[row_, 12], meas_data[row_, 13]]
basal_data = array(basal_data_)

CST_rnd = 0  ## change
data_mat = []
for row_ in range(shape(meas_data)[0]):
    if meas_data[row_, 4] not in ['1', '100']: continue
    for kns in [0, 1]:
        for tm in [0, 1, 2]:
            meas_val = meas_data[row_, 6 + tm + kns*2]
            bsl_idx = cellline_list.index(meas_data[row_, 1])
            basal_val = basal_data[bsl_idx, 1 + kns]
            data_mat.append([val for val in meas_data[row_, :6]]
                + [kns, timemin_list[tm], TN_HER2_HR_list[bsl_idx]]
                + [meas_val, basal_val, float(meas_val)/float(basal_val)])
data_arr = array(data_mat)
print shape(data_arr)
plt.figure()
plt.hist(data_arr[:, -1].astype(float), bins=20)
plt.figure()
plt.hist(log10(data_arr[:, -1].astype(float) + 1), bins=20)

col_names = array(['Cell_Line_ID', 'Cell_Line_Name', 'Protein_ID', 'Protein_Name',
    'Ligand_Conc', 'Conc_unit', 'Kinase', 'Meas_Time', 'BR_subtype',
    'Meas', 'Basal', 'Meas_Basal'])
var_select = [0, 2, 4, 6, 7, 8, 11]  ## ID IV IV IV IV IV DV
df_ = pd.DataFrame(data_arr[:, var_select], columns=col_names[var_select], dtype=float)
#print df_[:5]
id_col_names, iv_col_names = col_names[[0]], col_names[[2, 4, 6, 7, 8]]
id_lvl_list = get_unique_levels(df_, id_col_names)  ## 39
iv_lvl_list = get_unique_levels(df_, iv_col_names)  ## 15 2 2 3 3


## ___ analyze ___
## Protein_ID Ligand_Conc Kinase Meas_Time BR_subtype : 15 2 2 3 3

all_res = []
for i1 in range(2):  ## subtype x ligand : 3 by 15 (2x2x3)
    for i2 in range(2):
        for i3 in [0, 2, 1]:
            res = gPCA_comm(df_, 4, [range(15), [i1], [i2], [i3], [0, 1, 2]])
            savetxt('RTK_gPCA_res_SBTxLGD_%d-%d-%d.txt' % (i1, i2, i3), res)
            all_res.append(res)
savetxt('RTK_gPCA_res_SBTxLGD_%d-%d-%d(all).txt' % (i1, i2, i3), vstack(all_res))

all_res = []
for i2 in range(2):  ## concentration x ligand : 2 by 15 (2x3)
    for i3 in [0, 2, 1]:
        res = gPCA_comm(df_, 1, [range(15), [0, 1], [i2], [i3], [0, 1, 2]])
        savetxt('RTK_gPCA_res_CNCxLGD_%d-%d.txt' % (i2, i3), res)
        all_res.append(res)
savetxt('RTK_gPCA_res_CNCxLGD_%d-%d(all).txt' % (i2, i3), vstack(all_res))

all_res = []
for i1 in range(2):  ## kinase x ligand : 2 by 15 (2x3)
    for i3 in [0, 2, 1]:
        res = gPCA_comm(df_, 2, [range(15), [i1], [0, 1], [i3], [0, 1, 2]])
        savetxt('RTK_gPCA_res_KNSxLGD_%d-%d.txt' % (i1, i3), res)
        all_res.append(res)
savetxt('RTK_gPCA_res_KNSxLGD_%d-%d(all).txt' % (i1, i3), vstack(all_res))

all_res = []
for i1 in range(2):  ## time x ligand : 3 by 15 (2x2)
    for i2 in range(2):
        res = gPCA_comm(df_, 3, [range(15), [i1], [i2], [0, 2, 1], [0, 1, 2]])
        savetxt('RTK_gPCA_res_MTMxLGD_%d-%d.txt' % (i1, i2), res)
        all_res.append(res)
savetxt('RTK_gPCA_res_MTMxLGD_%d-%d(all).txt' % (i1, i2), vstack(all_res))

#Protein HMS LINCS ID	Protein Name
#200864	VEGF165	0
#200850	EGF	1
#200851	EPR	2
#200852	BTC	3
#200853	HRG	4
#200854	INS	5
#200855	IGF-I	6
#200856	IGF-II	7
#200857	PDGF-BB	8
#200858	HGF	9
#200859	SCF	10
#200860	FGF-1	11
#200861	FGF-2	12
#200862	NGF-beta	13
#200863	EFNA1	14

# ligand groups
ERBB_lgs = [1, 2, 3, 4]
ERBB_lgs_ = [200850, 200851, 200852, 200853]
nERBB_lgs = [0, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14]
nERBB_lgs_ = [200864, 200854, 200855, 200856, 200857,
    200858, 200859, 200860, 200861, 200862, 200863]
IGFINS_lgs = [5, 6, 7]
IGFINS_lgs_ = [200854, 200855, 200856]
n_lgs = [0, 8, 9, 10, 11, 12, 13, 14]
n_lgs_ = [200864, 200857, 200858, 200859, 200860, 200861, 200862, 200863]
lgd_grps = [ERBB_lgs, IGFINS_lgs, n_lgs]

i1, i2, i3 = 1, 0, 2  ## 100 ng/ml, AKT, 30 min
#i1, i2, i3 = 0, 1, 2  ## 1 ng/ml, ERK, 30 min
res = gPCA_comm(df_, 4, [arange(15)[ERBB_lgs], [i1], [i2], [i3], [0, 1, 2]], ch_=1)
savetxt('RTK_gPCA_res_SBTxLGD_%d-%d-%d(ERBB).txt' % (i1, i2, i3), res)
res = gPCA_comm(df_, 4, [arange(15)[nERBB_lgs], [i1], [i2], [i3], [0, 1, 2]], ch_=1)
savetxt('RTK_gPCA_res_SBTxLGD_%d-%d-%d(nERBB).txt' % (i1, i2, i3), res)
res = gPCA_comm(df_, 4, [arange(15)[IGFINS_lgs], [i1], [i2], [i3], [0, 1, 2]], ch_=1)
savetxt('RTK_gPCA_res_SBTxLGD_%d-%d-%d(IGFINS).txt' % (i1, i2, i3), res)
res = gPCA_comm(df_, 4, [arange(15)[n_lgs], [i1], [i2], [i3], [0, 1, 2]], ch_=1)
savetxt('RTK_gPCA_res_SBTxLGD_%d-%d-%d(nERBBIGFINS).txt' % (i1, i2, i3), res)

if 0:
    i1, i2, i3 = 1, 0, 2  ## 100 ng/ml, AKT, 30 min
    #i1, i2, i3 = 0, 1, 2  ## 1 ng/ml, ERK, 30 min
    res2 = gPCA_comm(df_, 4, [arange(15), [i1], [i2], [i3], [0, 1, 2]], ch_=2)
    savetxt('RTK_gPCA_res_SBTxLGD_%d-%d-%d(ws).txt' % (i1, i2, i3), res2)
    plt.figure()
    plt.pcolormesh(abs(res2))
    #plt.close('all')

i1, i2, i3 = 1, 0, 2  ## 100 ng/ml, AKT, 30 min
#i1, i2, i3 = 0, 1, 2  ## 1 ng/ml, ERK, 30 min
data_SBT_LGD1 = gPCA_comm(df_, 4, [arange(15)[ERBB_lgs], [i1], [i2], [i3], [0, 1, 2]], ch_=3)
data_SBT_LGD2 = gPCA_comm(df_, 4, [arange(15)[nERBB_lgs], [i1], [i2], [i3], [0, 1, 2]], ch_=3)
data_SBT_LGD = [data_SBT_LGD1, data_SBT_LGD2]
if 0:
    gPCA_res_ = []
    for sbt in range(3):
        res_ = gPCA_func([log10(data_SBT_LGD[k][sbt].T + 1) for k in range(2)])
        print_gPCA(res_)
        gPCA_res_.append(array(res_))
    savetxt('RTK_gPCA_res_LGDxSBT_%d-%d-%d(all).txt' % (i1, i2, i3), vstack(gPCA_res_))
if 1:
    data_SBT_LGD_f = hstack([vstack(data_SBT_LGD1), vstack(data_SBT_LGD2)])
    plt.figure()
    htmp = plt.pcolormesh(log10(data_SBT_LGD_f + 1))
    plt.xlabel('ligands')
    plt.ylabel('cell lines')
    plt.xticks([2, 4, 9.5], ['ErbB', '|', 'non-ErbB'])
    plt.yticks([9, 18, 23.5, 29, 34], ['TN', '---', 'H2', '---', 'HR'])
    plt.colorbar(htmp)
    plt.title('log10 fold change in activity - %s ng/ml, %s, %d min' %
        (conc_list[i1], kinase_list[i2], iv_lvl_list[3][i3]))
    plt.savefig('heatmap(%s).png' % kinase_list[i2], bbox_inches='tight', dpi=300)

from scipy.stats import f_oneway
from scipy.stats import ttest_ind

## 100 ng/ml, AKT, 30 min
i1, i2, i3 = 1, 0, 2
data_SBT_LGD1 = gPCA_comm(df_, 4, [arange(15)[ERBB_lgs], [i1], [i2], [i3], [0, 1, 2]], ch_=3)
data_SBT_LGD2 = gPCA_comm(df_, 4, [arange(15)[nERBB_lgs], [i1], [i2], [i3], [0, 1, 2]], ch_=3)
data_SBT_LGD = [data_SBT_LGD1, data_SBT_LGD2]
res = f_oneway(data_SBT_LGD1[1].flatten(), data_SBT_LGD2[1].flatten())
print res.statistic, res.pvalue
res = ttest_ind(data_SBT_LGD1[1].flatten(), data_SBT_LGD2[1].flatten())
print res.statistic, res.pvalue

## 1 ng/ml, ERK, 30 min
i1, i2, i3 = 0, 1, 2
data_SBT_LGD1 = gPCA_comm(df_, 4, [arange(15)[ERBB_lgs], [i1], [i2], [i3], [0, 1, 2]], ch_=3)
data_SBT_LGD2 = gPCA_comm(df_, 4, [arange(15)[nERBB_lgs], [i1], [i2], [i3], [0, 1, 2]], ch_=3)
data_SBT_LGD = [data_SBT_LGD1, data_SBT_LGD2]
data_SBT_LGD_k2 = hstack([data_SBT_LGD1[2], data_SBT_LGD2[2]])
res = f_oneway(data_SBT_LGD_k2[:, [3, 11, 12]].flatten(),
    data_SBT_LGD_k2[:, [0, 1, 2, 4, 5, 6, 7, 8, 9, 10, 13, 14]].flatten())
print res.statistic, res.pvalue
res = ttest_ind(data_SBT_LGD_k2[:, [3, 11, 12]].flatten(),
    data_SBT_LGD_k2[:, [0, 1, 2, 4, 5, 6, 7, 8, 9, 10, 13, 14]].flatten())
print res.statistic, res.pvalue

i1, i2, i3 = 1, 0, 2  ## 100 ng/ml, AKT, 30 min
#i1, i2, i3 = 0, 1, 2  ## 1 ng/ml, ERK, 30 min
data_SBT_LGD1 = gPCA_comm(df_, 4, [arange(15)[ERBB_lgs], [i1], [i2], [i3], [0, 1, 2]], ch_=3)
data_SBT_LGD2 = gPCA_comm(df_, 4, [arange(15)[nERBB_lgs], [i1], [i2], [i3], [0, 1, 2]], ch_=3)
data_SBT_LGD = [data_SBT_LGD1, data_SBT_LGD2]
print [mean(data_) for data_ in data_SBT_LGD1]
res = f_oneway(data_SBT_LGD1[0].flatten(), data_SBT_LGD1[1].flatten(),
    data_SBT_LGD1[2].flatten())
print res.statistic, res.pvalue
for grp in [[0, 1], [0, 2], [1, 2]]:
    res = f_oneway(data_SBT_LGD1[grp[0]].flatten(), data_SBT_LGD1[grp[1]].flatten())
    print res.statistic, res.pvalue
print [mean(data_) for data_ in data_SBT_LGD2]
res = f_oneway(data_SBT_LGD2[0].flatten(), data_SBT_LGD2[1].flatten(),
    data_SBT_LGD2[2].flatten())
print res.statistic, res.pvalue
for grp in [[0, 1], [0, 2], [1, 2]]:
    res = f_oneway(data_SBT_LGD2[grp[0]].flatten(), data_SBT_LGD2[grp[1]].flatten())
    print res.statistic, res.pvalue

#from statsmodels.formula.api import ols
#from statsmodels.stats.anova import anova_lm
#formula_ = ''
#model_ = ols(formula_, df_).fit()
#aov_table_ = anova_lm(model_, typ=2)

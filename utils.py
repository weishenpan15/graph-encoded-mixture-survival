from sklearn.cluster import KMeans
from sksurv.nonparametric import kaplan_meier_estimator
from lifelines.statistics import multivariate_logrank_test
import numpy as np

def cluster_logrank(x, t, e, k, seed):
    kmeans = KMeans(n_clusters=k, random_state=seed)
    kmeans.fit(x)
    c = kmeans.labels_
    tstat_list = []
    logp_list = []
    for c1 in range(kmeans.n_clusters):
        for c2 in range(c1 + 1, kmeans.n_clusters):
            # print("Log rank train {} VS {}: ".format(c1, c2))
            result = multivariate_logrank_test(t[np.where((c == c1) | (c == c2))],
                                               c[np.where((c == c1) | (c == c2))],
                                               e[np.where((c == c1) | (c == c2))])
            # print(result.test_statistic)
            # print(result.p_value)
            # print(result.print_summary())
            tstat_list.append(result.test_statistic)
            logp_list.append(-np.log(result.p_value))

    return c, tstat_list, logp_list


def eval_logrank(c, t, e):
    tstat_list = []
    logp_list = []
    
    # pairwise comparison? 
    for c1 in range(int(np.max(c) + 1)):
        for c2 in range(c1 + 1, int(np.max(c) + 1)):
            # print("Log rank train {} VS {}: ".format(c1, c2))
            result = multivariate_logrank_test(t[np.where((c == c1) | (c == c2))],
                                               c[np.where((c == c1) | (c == c2))],
                                               e[np.where((c == c1) | (c == c2))])
            # print(result.test_statistic)
            # print(result.p_value)
            # print(result.print_summary())
            tstat_list.append(result.test_statistic)
            logp_list.append(-np.log(result.p_value))

    return c, tstat_list, logp_list
import copy
import numpy as np
import networkx as nx
import utils


def main_event(data_path, eps, e_r, exp_num):
    # error vary eps
    all_deg_kl = []  # degree distribution
    all_weight_kl = []  # weight distribution
    all_cc_RMSE = []  # clustering coefficent
    all_diam_rel = []  # diameter

    for ei in range(len(eps)):
        epsilon = eps[ei]
        ##########################
        eps_edge = e_r[0] * (epsilon / w)
        eps_weight_1 = e_r[1] * (epsilon / w)
        eps_weight_2 = e_r[2] * (epsilon / w)

        # error of each experiment
        deg_kl_arr = np.zeros([exp_num])
        weight_kl_arr = np.zeros([exp_num])
        cc_RMSE_arr = np.zeros([exp_num])
        diam_rel_arr = np.zeros([exp_num])

        for exper in range(exp_num):
            print('-----------epsilon=%.1f,exper=%d/%d-------------' % (epsilon, exper + 1, exp_num))
            ##########################
            # error of each snapshot
            deg_kl_ind = np.zeros([snapshot_num])
            weight_kl_ind = np.zeros([snapshot_num])
            rel_cc = []
            syn_cc = []
            diam_rel_ind = np.zeros([snapshot_num])

            for time_index in range(snapshot_num):
                print('Dataset:%s' % (time_index))
                mat0, mid = utils.get_mat(data_path, node_num, time_index)
                mat0_index = mat0
                ########## Parameters of original graph
                mat0_graph = nx.from_numpy_array(mat0_index, create_using=nx.Graph)
                mat0_degree = np.count_nonzero(mat0_index, 1)
                mat0_deg_dist = np.bincount(np.int64(mat0_degree))
                mat0_triu = np.triu(mat0_index, 1)
                mat0_weig_seq = mat0_triu[mat0_triu != 0]
                mat0_weig_dist, bin_weigs = np.histogram(mat0_weig_seq, bins=max_h, range=(0, max_h))
                rel_cc.append(nx.transitivity(mat0_graph))
                mat0_diam = utils.cal_diam(mat0_index)

                ############### Step2: Data Range Estimation
                transform_1 = utils.transform_matrix(max_h, max_h, eps_weight_1)
                sample_user_num = len(mat0_index)
                h = np.max(mat0_index, axis=1)
                ture_hist, ture_bin_edges = np.histogram(h, bins=max_h, range=(0, max_h))
                noise_norm_h = utils.SW(h, eps_weight_1, max_h)
                # Post-processing
                ee_1 = np.exp(eps_weight_1)
                b_1 = ((eps_weight_1 * ee_1) - ee_1 + 1) / (2 * ee_1 * (ee_1 - 1 - eps_weight_1))
                norm_hist_1, bin_edges_1 = np.histogram(noise_norm_h, bins=max_h, range=(-b_1 / 2, 1 + b_1 / 2))
                pro_hist_1 = utils.EMS(max_h, norm_hist_1, transform_1)
                calib_h = []
                for i in range(sample_user_num):
                    noise_bin_index1 = utils.get_bin_index(noise_norm_h[i], -b_1 / 2, 1 + b_1 / 2, max_h)
                    post_p1 = pro_hist_1 * transform_1[noise_bin_index1] / (
                        np.sum(pro_hist_1 * transform_1[noise_bin_index1]))
                    calib_bin_index = np.argmax(post_p1)

                    calib_h.append(ture_bin_edges[calib_bin_index + 1])
                calib_h = np.array(calib_h)
                max_calib_h = np.max(calib_h)

                # Truncation
                for i in range(sample_user_num):
                    mat0_index[i][mat0_index[i] > calib_h[i]] = calib_h[i]

                ############### Step3: Aggregate Information Collection
                # degree perturbation
                dd1 = np.count_nonzero(mat0_index, 1)
                dd_noise = []
                for i in range(len(dd1)):
                    dd_noise.append(utils.geometric(dd1[i], 1, eps_edge))
                dd_calibra = utils.FO_pp_sec23(dd_noise)
                dd_calibra[dd_calibra < 0] = 0
                dd_calibra[dd_calibra >= len(dd_calibra)] = len(dd_calibra) - 1
                dd_calibra_sum = np.sum(dd_calibra)
                if dd_calibra_sum % 2 != 0:
                    dd_calibra = utils.adjust_element(dd_calibra)

                # adjacency list perturbation
                noise_norm = np.zeros([sample_user_num, sample_user_num])
                ee_2 = np.exp(eps_weight_2)
                b_2 = ((eps_weight_2 * ee_2) - ee_2 + 1) / (2 * ee_2 * (ee_2 - 1 - eps_weight_2))
                for i in range(sample_user_num):
                    noise_norm[i] = utils.SW(mat0_index[i], eps_weight_2, calib_h[i])

                cailb_mat_norm = noise_norm
                prior_mat = cailb_mat_norm.T * cailb_mat_norm
                # make sure the final adjacency matrix is symmetric
                prior_mat = np.triu(prior_mat, 1)
                prior_mat = prior_mat + np.transpose(prior_mat)

                ############### Step4: Graph Snapshot Generation
                syn_mat = np.zeros([len(mat0_index), len(mat0_index)], dtype='float32')
                dd_copy = copy.deepcopy(dd_calibra)
                dd_indices = np.where(dd_copy > 0)[0]
                dd_indices_copy = copy.deepcopy(dd_indices)
                while True:
                    if len(dd_indices) > 1 and len(dd_indices_copy) > 1:
                        smallest_dd_index = np.argmin(dd_copy[dd_indices])
                        dd_min_indices = dd_indices[smallest_dd_index]
                        extra_dd_indices = dd_indices_copy[dd_indices_copy != dd_min_indices].copy()
                        mindd_maxpriordd_index = np.where(prior_mat == np.max(prior_mat[dd_min_indices][extra_dd_indices]))
                        d1_index = mindd_maxpriordd_index[0][0]
                        d2_index = mindd_maxpriordd_index[0][1]
                        syn_mat[mindd_maxpriordd_index] = noise_norm[d1_index, d2_index]
                        dd_copy[mindd_maxpriordd_index[0][0]] = dd_copy[mindd_maxpriordd_index[0][0]] - 1
                        dd_copy[mindd_maxpriordd_index[0][1]] = dd_copy[mindd_maxpriordd_index[0][1]] - 1
                        dd_indices = np.where(dd_copy > 0)[0]
                        dd_indices_copy = copy.deepcopy(dd_indices)
                        dd_indices_copy = np.setdiff1d(dd_indices_copy, np.array([d1_index, d2_index]))
                    else:
                        break

                # Post-processing
                mat2_triu = np.triu(syn_mat, 1)
                mat2_nz_index = np.nonzero(mat2_triu)
                mat2_seq = syn_mat.ravel()[
                    np.flatnonzero(mat2_triu)]
                max_calib_h = max(int(max_calib_h), 2)
                transform_2 = utils.transform_matrix(max_calib_h, max_calib_h, eps_weight_2)
                ture_bin_edges2 = utils.divide_interval(0, 1, max_calib_h)
                norm_hist_2, bin_edges_2 = np.histogram(mat2_seq, bins=max_calib_h, range=(-b_2 / 2, 1 + b_2 / 2))
                pro_hist_2 = utils.EMS(max_calib_h, norm_hist_2, transform_2)
                if sum(pro_hist_2) == 0 or len(mat2_seq) == 0:
                    mat2_triu = np.zeros([len(mat0_index), len(mat0_index)], dtype='float32')
                else:
                    calib_mat2_seq = []
                    for i in range(len(mat2_seq)):
                        noise_bin_index2 = utils.get_bin_index(mat2_seq[i], -b_2 / 2, 1 + b_2 / 2, max_calib_h)
                        post_p2 = pro_hist_2 * transform_2[noise_bin_index2] / (
                            np.sum(pro_hist_2 * transform_2[noise_bin_index2]))
                        calib_bin_index2 = np.argmax(post_p2)
                        calib_mat2_seq.append(ture_bin_edges2[calib_bin_index2 + 1])
                    calib_mat2_seq = np.array(calib_mat2_seq)
                    calib_weig = calib_mat2_seq * calib_h[mat2_nz_index[0]]
                    calib_weig[calib_weig < 1] = 1
                    mat2_triu[mat2_nz_index] = calib_weig

                syn_mat = mat2_triu + np.transpose(mat2_triu)
                syn_graph = nx.from_numpy_array(syn_mat, create_using=nx.Graph)

                # save the graph
                #dataset_name = "EmailDept1"
                #dataset_name = "FbForum"
                #dataset_name = "tech"
                #utils.save_graph_with_params(dataset_name, epsilon, time_index, exper, syn_mat, mid, type="main_event")

                # evaluate
                syn_cc.append(nx.transitivity(syn_graph))
                syn_degree = np.count_nonzero(syn_mat, 1)
                syn_deg_dist = np.bincount(np.int64(syn_degree))
                syn_triu = np.triu(syn_mat, 1)
                syn_weig_seq = syn_triu[syn_triu != 0]
                syn_weig_dist, syn_bin_weigs = np.histogram(syn_weig_seq, bins=max_h, range=(0, max_h))
                syn_diam = utils.cal_diam(syn_mat)

                # calculate the metrics
                # degree distribution
                deg_kl = utils.cal_kl(mat0_deg_dist, syn_deg_dist)
                # weight distribution
                weight_kl = utils.cal_kl(mat0_weig_dist, syn_weig_dist)
                # diameter
                diam_rel = utils.cal_rel(mat0_diam, syn_diam)

                deg_kl_ind[time_index] = deg_kl
                weight_kl_ind[time_index] = weight_kl
                diam_rel_ind[time_index] = diam_rel

            # clustering coefficent
            cc_RMSE = utils.cal_RMSE(rel_cc, syn_cc)

            cc_RMSE_arr[exper] = cc_RMSE
            deg_kl_arr[exper] = np.mean(deg_kl_ind)
            weight_kl_arr[exper] = np.mean(weight_kl_ind)
            diam_rel_arr[exper] = np.mean(diam_rel_ind)

        all_cc_RMSE.append(np.mean(cc_RMSE_arr))
        all_deg_kl.append(np.mean(deg_kl_arr))
        all_weight_kl.append(np.mean(weight_kl_arr))
        all_diam_rel.append(np.mean(diam_rel_arr))

        print('-----------------------------')
        print('dataset:', data_path)
        print('eps=', eps)
        print('all_weight_kl=', all_weight_kl)
        print('all_deg_kl=', all_deg_kl)
        print('all_cc_RMSE=', all_cc_RMSE)
        print('all_diam_rel=', all_diam_rel)


if __name__ == '__main__':
    ######### Dataset
    # dataset: Email-Eu
    # data_path = "./data/EmailDept1_LDP/EmailDept1"
    # node_num = 319
    # snapshot_num = 173
    # max_h = 72

    # dataset: Forum
    data_path = "./data/Forum_LDP/FbForum"
    node_num = 899
    snapshot_num = 24
    max_h = 168

    # dataset: Tech-AS
    # data_path = "./data/Tech_LDP/tech"
    # node_num = 5000
    # snapshot_num = 24
    # max_h = 24

    ######### set the privacy budget, list type
    eps = [0.5, 1, 1.5, 2, 2.5]

    ######### set the ratio of the privacy budget, list type
    e_r = [1 / 3, 1 / 3, 1 / 3]

    ######### set the number of experiments
    exp_num = 10

    ######### set the sliding window size
    w = 1

    ######### run the function
    main_event(data_path, eps, e_r, exp_num)
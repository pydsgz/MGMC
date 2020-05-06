import os
import pickle
import numpy as np
import pandas as pd
import gcn_utils
# from emvert_functional_code import emvert_ml
from scipy.sparse import csgraph


def load_tadpole_data(remove_longitudinal_mri=True):
    """
    TODO
    1. LOAD longitudinal data as well
    2. Load which class labels to include
    3. Load up to which month to include
    4. When not longitudinal remove features from longitudinal MRI
    5. When longitudinal make sure to use subject IDs in the same subject
    Load tadpole dataset.

    Parameters:
    remove_longitudinal_mri: bool
        If true, will exclude columns names containing 'UCSFFSL'

    :return:
        numerical_df: pandas.DataFrame
            Features n by f (f is dimension of features)
        meta_df: pandas.DataFrame
            Meta information of shape n by f (f dimension of meta information)
        label_df: pandas.DataFrame
            Class labels of shape n x c (number of class labels)
    """
    # os.chdir('/data/Gerome/01_main/98_tf_practice/mgcnn/Notebooks/douban')

    # Load tadpole dataset
    # 822 subjects passed screening but only 819 had baseline observation.
    cross_val_df = pd.read_csv('./Data/tadpole_challenge/CROSSVAL.csv')

    # Load main data
    input_dict_df = pd.read_csv('./Data/tadpole_challenge/TADPOLE_D1_D2_Dict.csv')
    input_df = pd.read_csv('./Data/tadpole_challenge/TADPOLE_D1_D2.csv', low_memory=False)

    # Remove columns with _bl suffix
    input_df = input_df[input_df.columns[~input_df.columns.str.contains('_bl$')]]
    # Remove longitudinal features

    # Select which features to use
    d1_d2_dict_df = pd.read_excel('./Data/tadpole_challenge/d1_d2_use_columns.xls')
    use_features_df = d1_d2_dict_df[['USE_META', 'USE_FEATURE']]

    # Select all numerical features given USE_FEATURES column
    numerical_cols = d1_d2_dict_df.FLDNAME[~use_features_df.USE_FEATURE.isnull()].tolist()
    numerical_df = input_df[numerical_cols]

    meta_cols = d1_d2_dict_df.FLDNAME[~use_features_df.USE_META.isnull()].tolist()
    # Replace RAVL to RAVLT_forgetting
    meta_cols = list(pd.Series(meta_cols).replace('RAVLT', 'RAVLT_forgetting'))
    # meta_df = input_df[meta_cols]

    # Select MCI's
    # Take labels 2 and 5, MCI to MCI and MCI to AD
    mci_df = input_df[(input_df.DXCHANGE == 2) | (input_df.DXCHANGE == 5)]
    mci_df = mci_df[(mci_df.D1 == 1)]

    # Measurements intil m48
    mci_48_df = mci_df[list(mci_df.VISCODE.str.contains('^bl$|^m06$|^m12$|^m18$|^m24$|^m30$|^m36$|^m42$|^m48$'))]

    # RID of converters
    dd_mci = mci_48_df.drop_duplicates(['RID'], keep='last')
    c_mci = dd_mci.RID[list(dd_mci.DXCHANGE == 5)]
    # RID of stable
    s_mci = dd_mci.RID[list(dd_mci.DXCHANGE == 2)]

    # All basline measurements
    mci_all_baseline = mci_48_df[mci_48_df.VISCODE == 'bl']

    # Baseline of converters and stable mci
    c_mci_bl = mci_all_baseline.copy()[mci_all_baseline.RID.isin(list(c_mci))]
    s_mci_bl = mci_all_baseline.copy()[mci_all_baseline.RID.isin(list(s_mci))]
    c_mci_bl['gcn_label'] = 1
    s_mci_bl['gcn_label'] = 0
    data_x = pd.concat([c_mci_bl, s_mci_bl])
    label_df = data_x.gcn_label
    assert data_x.drop_duplicates(['RID']).shape == data_x.shape

    # Remove longitudinal MRI features
    if remove_longitudinal_mri:
        non_UCSFFSL_cols = numerical_df.columns[
            ~numerical_df.columns.str.upper().str.contains(
                'UCSFFSL')]
        numerical_df = numerical_df[non_UCSFFSL_cols]

    numerical_df = data_x[numerical_cols]
    meta_df = data_x[meta_cols]
    label_df = data_x[['gcn_label']]
    return numerical_df, meta_df, pd.DataFrame(label_df)


def quick_load_tadpole():
    """
        This is only a simple function to call preselected columns for
        GMC on non-longitudinal data AD classification.
    TODO
    Load tadpole csv

    :return:
    """
    # os.chdir('/src/keras/Gerome/01_main/98_tf_practice/mgcnn/Notebooks/douban')
    meta_data = pd.read_csv('data/tadpole/meta_data.csv')
    y = pd.read_csv('data/tadpole/label_data.csv')
    data_x = pd.read_csv('data/tadpole/feature_data.csv')
    # y = np.expand_dims(np.array(y), -1)
    print("meta data dimension is: {}".format(meta_data.shape))
    print("x data dimension is: {}".format(data_x.shape))
    print("y data dimension is: {}".format(y.shape))

    meta_data.PTGENDER = meta_data.PTGENDER.replace('Male', 0)
    meta_data.PTGENDER = meta_data.PTGENDER.replace('Female', 1)

    print('Feature shape is {}'.format(data_x.shape))
    print('Class label shape is {}'.format(y.shape))
    print('Meta information shape is {}'.format(meta_data.shape))
    # DEBUG
    data_x = df_drop(data_x, 0.01)[0]
    return data_x, meta_data, y


# def format_data(data_x, data_meta, data_y, row_g_metric, k_nn=None,
#                 col_g_metric=None, stack_y=True, row_graph_only=True,
#                 adj_parisot=False):
#     """
#     Calculate graph and matrix information.
#
#     :param data_x:
#     :param data_meta:
#     :param data_y:
#     :param k_nn:
#     :param row_g_metric:
#     :param col_g_metric:
#     :param stack_y:
#     :param row_graph_only:
#     :param adj_parisot:
#     :return:
#         TODO
#         M
#             Normalized and filled feature matrix M
#         initial_tr_mask
#             Indices of observed entries in M
#         Lrow
#             Row graph laplacian
#         Lcol
#             Column graph laplacian
#         A_matrix_row
#         A_matrix_col
#     """
#     data_x = data_x.copy()
#     data_meta = data_meta.copy()
#
#     # Format data matrix and take indices of know elements
#     M = data_x
#     observed_p_in_M = np.where(~np.isnan(np.array(M)))[0].shape[0]/np.product(
#         M.shape[:])
#     print("Percentage of observed elements {}".format(observed_p_in_M))
#
#     # Indeces of observed values in M
#     dum_idx_M = np.where(~np.isnan(np.array(M)))
#     initial_observed_mask = np.zeros_like(M)
#     initial_observed_mask[dum_idx_M] = 1
#
#     # Replace nan to 0
#     if isinstance(M, np.ndarray):
#         M = np.nan_to_num(M)
#         M = pd.DataFrame(M)
#     else:
#         M = M.replace(np.nan, 0)
#
#     # Stack labels as a last column vector of matrix M
#     if stack_y:
#         M['labels_y'] = np.array(data_y)
#     M = np.matrix(M)
#
#     # Calculate row graph
#     if k_nn is not None:
#         if row_g_metric is None:
#             graph_row_neighbors = gcn_utils.get_k_nearest_neighbors(
#                 data_meta, k_nn)
#         else:
#             graph_row_neighbors = gcn_utils.get_k_nearest_neighbors(
#                 data_meta, k_nn, metric=row_g_metric)
#         A_matrix_row = gcn_utils.get_adjacency_matrix(data_meta,
#                                                       graph_row_neighbors)
#     elif adj_parisot and k_nn is None:
#         A_matrix_row = gcn_utils.get_adjacency_parisot(data_meta)
#     else:
#         raise NotImplementedError("Wrong usage of format_data function")
#
#
#     # Calculate column graph
#     if row_graph_only == False and col_g_metric is None:
#         graph_cols_neighbors = gcn_utils.get_k_nearest_neighbors(M.T, k_nn,
#                                                                  metric=gcn_utils.corrcoef_vector)
#     elif row_graph_only:
#         graph_cols_neighbors = None
#     else:
#         graph_cols_neighbors = gcn_utils.get_k_nearest_neighbors(M.T, k_nn,
#                                                                  metric=col_g_metric)
#     if row_graph_only:
#         Lcol = None
#         A_matrix_col = None
#     else:
#         A_matrix_col = gcn_utils.get_adjacency_matrix(M.T,
#                                                       graph_cols_neighbors)
#         Lcol = csgraph.laplacian(A_matrix_col, normed=True)
#
#     # computation of the normalized laplacians
#     Lrow = csgraph.laplacian(A_matrix_row, normed=True)
#
#     return M, initial_observed_mask, Lrow, Lcol, A_matrix_row, A_matrix_col


def select_modalities(x_data, longitudinal=False, modalities=[
    'clinical_scores_df', 'mri_df', 'pet_all_df', 'dti_all_df', 'csf_all_df']):
    """
    Modality wise selection of features.

    Parameters:
        x_data: pandas dataframe
            Dataframe containing features
        longitudinal: bool
            If true, will exclude MRI longitudinal information
        modalities: list of str
            Select which modalities to include

    Returns:
    """
    data_x = x_data.copy()
    # Assert modality exists
    for i in modalities:
        modality_list = ['clinical_scores_df', 'mri_df', 'pet_df', 'dti_df',
                         'csf_df']
        assert i in modality_list, \
            '{} not in modality'.format(i)

    # Columns containing these strings "CDR|ADAS|MMSE|RAVLT|MOCA|ECOG"
    clinical_scores_df = data_x[data_x.columns[
        data_x.columns.str.upper().str.contains('CDR|ADAS|MMSE')]]

    # # col_of_interest = data_x.columns[~data_x.columns.str.upper().str.contains('^RID$|^PTID$|UID_|RUNDATE|EXAMDATE|UPDATE')]
    # # MRI
    mri_df = data_x[data_x.columns[data_x.columns.str.contains('UCSFFSX')]]
    if longitudinal:
        sl_mri_df = cx_mri = data_x[data_x.columns[data_x.columns.str.contains(
            'UCSFFSL')]]
        mri_df = pd.concat([mri_df, sl_mri_df])

    # # PET
    pet_df = data_x[data_x.columns[data_x.columns.str.contains(
        'BAIPETNMRC|UCBERKELEYAV45|UCBERKELEYAV1451')]]

    # # DTI biomarkers
    dti_df = data_x[data_x.columns[data_x.columns.str.contains(
        'DTIROI')]]

    # # CSF biomarkers
    csf_df = data_x[data_x.columns[data_x.columns.str.contains(
        'UPENNBIOM')]]

    combine_df_list = []
    for i in modalities:
        print(i)
        combine_df_list.append(eval(i))
    # combine_df_list = [eval(i) for i in modalities]
    combined_df = pd.concat(combine_df_list, axis=1)
    return combined_df


def select_p_data_df(df, p_select, random_seed):
    """
    Randomly select a given percentage of data from a dataframe

    Parameters:
        df: pd.Dataframe
        p_select: float

    Return:
        # TODO
    """
    data_arr = df.copy()
    data_arr = np.array(data_arr)
    M_num_entries = np.product(data_arr.shape[:])

    # Current index of known entries
    idx = np.where(~np.isnan(data_arr))
    idx_1 = idx[0]
    idx_2 = idx[1]
    cur_idx_len = idx_1.size

    # Current percentage of known entries
    current_p_known = cur_idx_len / M_num_entries
    # assert p_select < current_p_known, 'Percentage is larger than available ' \
    #                                    'entries'
    print('Percentage of current known entries {}'.format(current_p_known))

    # Set randomly selected entries to nan
    desired_entries = int(cur_idx_len * p_select)
    remove_entries = cur_idx_len - desired_entries

    np.random.seed(random_seed)
    idx_choice = np.random.choice(cur_idx_len, remove_entries, replace=False)
    new_idx = idx_1[idx_choice], idx_2[idx_choice]
    data_arr[new_idx] = np.nan

    # New percentage of entries
    new_p_known = np.where(~np.isnan(data_arr))[0].size / M_num_entries
    print('New percentage of known entries {}'.format(new_p_known))

    return data_arr, new_idx


def format_data_mgmc(data_x, data_meta, data_y, row_g_metric, k_nn=None,
                     stack_y=True):
    """
    Calculate graph and matrix information in multi-graph setting.
    Where age, gender, clinical scores, and risk-factors are used as
    meta-information.
    """
    data_x = data_x.copy()
    data_meta = data_meta.copy()

    # Format data matrix and take indices of known elements
    M = data_x
    observed_p_in_M = np.where(~np.isnan(np.array(M)))[0].shape[0]/np.product(
        M.shape[:])
    print("Percentage of observed elements {}".format(observed_p_in_M))

    # Indeces of observed values in M
    dum_idx_M = np.where(~np.isnan(np.array(M)))
    initial_observed_mask = np.zeros_like(M)
    initial_observed_mask[dum_idx_M] = 1

    # Replace nan to 0
    if isinstance(M, np.ndarray):
        M = np.nan_to_num(M)
        M = pd.DataFrame(M)
    else:
        M = M.replace(np.nan, 0)

    # Stack labels as a last column vector of matrix M
    if stack_y:
        M = np.matrix(M)
        data_y = np.matrix(data_y)
        M = np.hstack([M, data_y])  # one-hot encoded data can now be stacked.
    M = np.matrix(M)

    # if adj_parisot and k_nn is None:
    A_matrix_row = gcn_utils.get_multi_adjacency(data_meta)

    # Calculate col graph
    Lcol = None
    A_matrix_col = None

    # computation of the normalized laplacians
    cur_A = []
    for A in A_matrix_row:
        A += np.eye(A.shape[0])
        Lrow = csgraph.laplacian(A, normed=True)
        cur_A.append(Lrow)

    return M, initial_observed_mask, cur_A, A_matrix_row


def format_data_gcn(data_x, data_meta, data_y, dataset_name, stack_y=True,
                    multi=False):
    """
    Calculate graph and matrix information in single-graph setting.
    Where age, gender, clinical scores, and risk-factors are used as
    meta-information.
    """
    data_x = data_x.copy()
    data_meta = data_meta.copy()

    # Format data matrix and take indices of known elements
    M = data_x
    observed_p_in_M = np.where(~np.isnan(np.array(M)))[0].shape[0]/np.product(
        M.shape[:])
    print("Percentage of observed elements {}".format(observed_p_in_M))

    # Indeces of observed values in M
    dum_idx_M = np.where(~np.isnan(np.array(M)))
    initial_observed_mask = np.zeros_like(M)
    initial_observed_mask[dum_idx_M] = 1

    # Replace nan to 0
    if isinstance(M, np.ndarray):
        M = np.nan_to_num(M)
        M = pd.DataFrame(M)
    else:
        M = M.replace(np.nan, 0)

    # Stack labels as a last column vector of matrix M
    if stack_y:
        M = np.matrix(M)
        data_y = np.matrix(data_y)
        M = np.hstack([M, data_y]) # one-hot encoded data can now be stacked.
    M = np.matrix(M)

    if dataset_name == 'TADPOLE':
        A_matrix_row = gcn_utils.get_multi_adjacency(data_meta)
    elif dataset_name == 'PPMI':
        A_matrix_row = gcn_utils.get_multi_adjacency_ppmi(data_meta)
    elif dataset_name == 'EMVERT':
        A_matrix_row = gcn_utils.get_multi_adjacency_emvert(data_meta)
    elif dataset_name == 'THYROID':
        # Load existing adjacency matrix
        adjacency_matrix_path = './data/thyroid/adj_matrix.pkl'
        if os.path.exists(adjacency_matrix_path):
            with open(adjacency_matrix_path, 'rb') as f_open:
                A_matrix_row = pickle.load(f_open)
        # Else run and save
        else:
            A_matrix_row = gcn_utils.get_multi_adjacency_thyroid(data_meta)

            # save matrix
            with open(adjacency_matrix_path, 'wb') as f_open:
                pickle.dump(A_matrix_row, f_open, pickle.HIGHEST_PROTOCOL)
    elif dataset_name == 'CPET':
        # Load existing adjacency matrix
        adjacency_matrix_path = './data/cpet/adj_matrix.pkl'
        if os.path.exists(adjacency_matrix_path):
            with open(adjacency_matrix_path, 'rb') as f_open:
                A_matrix_row = pickle.load(f_open)
        # Else run and save
        else:
            A_matrix_row = gcn_utils.get_multi_adjacency_cpet(data_meta)

            # save matrix
            with open(adjacency_matrix_path, 'wb') as f_open:
                pickle.dump(A_matrix_row, f_open, pickle.HIGHEST_PROTOCOL)
    else:
        raise NotImplementedError

    if multi:
        cur_A = []
        for A in A_matrix_row:
            A += np.eye(A.shape[0])
            Lrow = csgraph.laplacian(A, normed=True)
            cur_A.append(Lrow)
    else:
        A_matrix_row = np.sum(A_matrix_row, 0)
        A_matrix_row += np.eye(A_matrix_row.shape[0])

        # computation of the normalized laplacians
        cur_A = csgraph.laplacian(A_matrix_row, normed=True)
        cur_A = [cur_A]

    return M, initial_observed_mask, cur_A, A_matrix_row


def format_data_mgmc_thyroid(data_x, data_meta, data_y, row_g_metric, k_nn=None,
                     stack_y=True):
    """
    Calculate graph and matrix information in multi-graph setting for emvert
    dataset..
    """
    data_x = data_x.copy()
    data_meta = data_meta.copy()

    # Format data matrix and take indices of known elements
    M = data_x
    observed_p_in_M = np.where(~np.isnan(np.array(M)))[0].shape[0]/np.product(
        M.shape[:])
    print("Percentage of observed elements {}".format(observed_p_in_M))

    # Indeces of observed values in M
    dum_idx_M = np.where(~np.isnan(np.array(M)))
    initial_observed_mask = np.zeros_like(M)
    initial_observed_mask[dum_idx_M] = 1

    # Replace nan to 0
    if isinstance(M, np.ndarray):
        M = np.nan_to_num(M)
        M = pd.DataFrame(M)
    else:
        M = np.array(M)
        M = np.nan_to_num(M)
        M = pd.DataFrame(M)

    # Stack labels as a last column vector of matrix M
    if stack_y:
        M['labels_y'] = np.array(data_y)
    M = np.matrix(M)

    # Load existing adjacency matrix
    adjacency_matrix_path = './data/thyroid/adj_matrix.pkl'
    if os.path.exists(adjacency_matrix_path):
        with open(adjacency_matrix_path, 'rb') as f_open:
            A_matrix_row = pickle.load(f_open)
    # Else run and save
    else:
        A_matrix_row = gcn_utils.get_multi_adjacency_thyroid(data_meta)

        # save matrix
        with open(adjacency_matrix_path, 'wb') as f_open:
            pickle.dump(A_matrix_row, f_open, pickle.HIGHEST_PROTOCOL)

    # computation of the normalized laplacians
    cur_A = []
    for A in A_matrix_row:
        Lrow = csgraph.laplacian(A, normed=True)
        cur_A.append(Lrow)

    return M, initial_observed_mask, cur_A, A_matrix_row


def format_data_mgmc_ppmi(data_x, data_meta, data_y, row_g_metric, k_nn=None,
                     stack_y=True):
    """
    Calculate graph and matrix information in multi-graph setting for emvert
    dataset..
    """
    data_x = data_x.copy()
    data_meta = data_meta.copy()

    # Format data matrix and take indices of known elements
    M = data_x
    observed_p_in_M = np.where(~np.isnan(np.array(M)))[0].shape[0]/np.product(
        M.shape[:])
    print("Percentage of observed elements {}".format(observed_p_in_M))

    # Indeces of observed values in M
    dum_idx_M = np.where(~np.isnan(np.array(M)))
    initial_observed_mask = np.zeros_like(M)
    initial_observed_mask[dum_idx_M] = 1

    # Replace nan to 0
    if isinstance(M, np.ndarray):
        M = np.nan_to_num(M)
        M = pd.DataFrame(M)
    else:
        M = M.replace(np.nan, 0)

    # Stack labels as a last column vector of matrix M
    if stack_y:
        M['labels_y'] = np.array(data_y)
    M = np.matrix(M)

    A_matrix_row = gcn_utils.get_multi_adjacency_ppmi(data_meta)

    # computation of the normalized laplacians
    cur_A = []
    for A in A_matrix_row:
        A += np.eye(A.shape[0])
        Lrow = csgraph.laplacian(A, normed=True)
        cur_A.append(Lrow)

    return M, initial_observed_mask, cur_A, A_matrix_row


def load_adni_one_baselines(remove_longitudinal_mri=True):
    """
    Load ADNI-1 baseline datasets. Dataframes will contain information from all
    229 normal subjects, 396 MCI, 188 Dementia labels.

    Parameters
    ----------
    remove_longitudinal_mri: bool
        If true, will remove columns which contains longitudinal information.

    Returns
    -------
    pd.DataFrame, pd.DataFrame, pd.DataFrame
    Dataframe containing numerical features
    Dataframe containing meta information
    Dataframe containing labels as ints
    """
    # Load main data
    input_dict_df = pd.read_csv('./data/tadpole/TADPOLE_D1_D2_Dict.csv')
    input_df = pd.read_csv('./data/tadpole/TADPOLE_D1_D2.csv', low_memory=False)

    # Remove columns with _bl suffix
    input_df = input_df[
        input_df.columns[~input_df.columns.str.contains('_bl$')]]

    # Select which features to use
    d1_d2_dict_df = pd.read_excel('./data/tadpole/d1_d2_use_columns.xls')
    use_features_df = d1_d2_dict_df[['USE_META', 'USE_FEATURE']]

    # Select all numerical features given USE_FEATURES column
    numerical_cols = d1_d2_dict_df.FLDNAME[
        ~use_features_df.USE_FEATURE.isnull()].tolist()
    numerical_df = input_df[numerical_cols]

    meta_cols = d1_d2_dict_df.FLDNAME[
        ~use_features_df.USE_META.isnull()].tolist()

    # Replace RAVL to RAVLT_forgetting
    meta_cols = list(pd.Series(meta_cols).replace('RAVLT', 'RAVLT_forgetting'))

    baseline_df = input_df[input_df.VISCODE.str.strip() == 'bl']
    baseline_df = baseline_df.drop_duplicates('RID')

    all_adni_one = baseline_df[baseline_df.COLPROT.str.strip() == 'ADNI1']
    all_adni_one = all_adni_one[all_adni_one.DX.isin(['MCI', 'NL', 'Dementia'])]
    print(all_adni_one.DX.value_counts())

    # Assign int to labels NL:0 , MCI:1, Dementia:2
    all_adni_one.DX.replace('NL', 0, inplace=True)
    all_adni_one.DX.replace('MCI', 1, inplace=True)
    all_adni_one.DX.replace('Dementia', 2, inplace=True)
    print(all_adni_one.DX.value_counts())

    numerical_df = all_adni_one[numerical_cols]

    if remove_longitudinal_mri:
        non_UCSFFSL_cols = numerical_df.columns[
            ~numerical_df.columns.str.upper().str.contains(
                'UCSFFSL')]
        numerical_df = numerical_df[non_UCSFFSL_cols]

    label_df = all_adni_one[['DX']]
    label_df.columns = ['gcn_label']
    meta_df = all_adni_one[meta_cols]

    # Replace whitespace with nan
    numerical_df.replace('^\s*$', np.nan, inplace=True, regex=True)

    assert numerical_df.shape[0] == meta_df.shape[0] and meta_df.shape[0] == \
           label_df.shape[0]


    return numerical_df, meta_df, pd.DataFrame(label_df)


def quick_load_adni_one_baseline():
    """
    Load saved ADNI1 baseline and remove columns in the feature matrix which
    contains less than 20% of entries.

    Returns
    -------
    pd.Dataframe, pd.Dataframe, pd.Dataframe
    """
    meta_data = pd.read_csv('data/tadpole/adni_one_baseline_meta_data.csv')
    y = pd.read_csv('data/tadpole/adni_one_baseline_label_data.csv')
    data_x = pd.read_csv('data/tadpole/adni_one_baseline_feature_data.csv')

    print("meta data dimension is: {}".format(meta_data.shape))
    print("x data dimension is: {}".format(data_x.shape))
    print("y data dimension is: {}".format(y.shape))

    meta_data.PTGENDER = meta_data.PTGENDER.replace('Male', 0)
    meta_data.PTGENDER = meta_data.PTGENDER.replace('Female', 1)

    print('Feature shape is {}'.format(data_x.shape))
    print('Class label shape is {}'.format(y.shape))
    print('Meta information shape is {}'.format(meta_data.shape))

    data_x.replace(-4, np.nan, inplace=True)
    data_x.replace('<200', np.nan, inplace=True)
    data_x.replace('^\s*$', np.nan, inplace=True, regex=True)
    data_x[data_x.applymap(type) == str] = np.nan

    data_x = df_drop(data_x, 0.10)[0]
    return data_x, meta_data, y


def df_drop(df, p):
    """
    Drop a column of a dataframe if elements in that column are less than
    given percentage.

    Parameters:
        df: pandas.DataFrame
            Dataframe containing all data
        p: float
            Percentage of how much a column should contain. If p is
            0.75, columns with elements greather than 75% will be included.

    Returns:
        res_df: tuple of pandas.DataFrame
            First element contains the dataframe of interest,
            Second element contains the dataframe of all columns which were
            excluded
    """
    assert p < 1.0, 'p must be less than 1.0'
    df = df.copy()
    # Percentage of elements of every column
    col_p = 1.0 - df.isnull().sum().values / df.shape[0]

    # Column names of interest
    col_of_interest_idx = col_p > p
    main_df = df[df.columns[col_of_interest_idx]]
    del_df = df[df.columns[~col_of_interest_idx]]
    res_df = main_df, del_df
    return res_df

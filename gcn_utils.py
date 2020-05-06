import os
import pandas as pd
import numpy as np
from sklearn.neighbors import NearestNeighbors
from scipy.stats import pearsonr
from scipy.spatial.distance import pdist, squareform
from joblib import Parallel, delayed
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import StratifiedKFold
import pickle
from sklearn.metrics import roc_auc_score, accuracy_score, \
    classification_report,f1_score
import matplotlib.pyplot as plt
import seaborn as sns

try:
    import tensorflow as tf
    from rpy2.robjects.packages import importr
    from rpy2.robjects.numpy2ri import numpy2ri
    stats = importr('stats')
except Exception as err:
    print(err)
    import tensorflow as tf
    # from rpy2.robjects.packages import importr
    # from rpy2.robjects.numpy2ri import numpy2ri
    # stats = importr('stats')


def get_k_nearest_neighbors(X, K, metric=None):
    """
    Get K nearest neighbors.

    Parameters:
    X: np.ndarray
        Numpy array with n by d dimension where n
        is the number of samples/nodes and d is the
        number of features
    K: int
        Number of nearest neighbors to find.
    metric

    Returns:
    res_X: np.ndarray
        Indeces of numpy array with n by K dimension. Where n are the
        number of samples and K are the number of neighbors.
    """
    K = K+1
    if metric is None:
        nbrs = NearestNeighbors(n_neighbors=K, algorithm='ball_tree').fit(X)
    else:
        nbrs = NearestNeighbors(n_neighbors=K, algorithm='ball_tree',
                                metric=metric).fit(X)
    indices = nbrs.kneighbors(X, return_distance=False)
    return indices[:,
           1:]  # exclude the first column as these are just row indices


def get_adjacency_matrix(X, indices):
    """
    Get adjacency matrix of X with size n by k (k number of 
    connections/neighbors). Output matrix will be a square 
    matrix of n by n where n is the number of nodes.

    Parameter:
    X: numpy.ndarray
        n by k matrix
    indices: numpy.ndarray

    Returns:
    res_A: np.matrix
        n by n numpy
    """
    res_A = np.zeros((X.shape[0], X.shape[0]))
    for k, v in enumerate(indices):
        np.put(res_A[k], v, 1)
    return np.matrix(res_A)


def create_labels(label, label_of_interest, label_replacement='remove_label'):
    """
    Create a list of labels given a pandas column
    and a list of desired labelling.

    Parameters:
    label: pd.Series
        list of labels
    label_of_interest: list of strings
        list containing label names to use
    label_replacement: str
        string to use as a replacement if label is not in label of interest

    Return:
        res_list: pd.Series
    """
    assert type(label) is pd.Series, 'label should be pandas Series'
    assert type(label_of_interest) is list, 'label_of_interest should be list'
    assert type(label_replacement) is str, 'label_replacement should be str'

    # Convert label and label_of_interest to lowercase
    label = label.str.lower()
    label_of_interest = [x.lower() for x in label_of_interest]

    # Change this labels to other
    labels_to_change = set(label) - set(label_of_interest)

    for i in labels_to_change:
        if type(i) is float and np.isnan(i):
            continue
        else:
            label[label == i] = label_replacement
    return label


def corrcoef_vector(v1,v2, p_val=0.05):
    """
    Calculate correlation coefficient of v1 and v2.
    
    :param v1: 
    :param v2: 
    :return: 
    """
    assert type(v1) is np.ndarray, "v1 must be numpy array"
    assert type(v2) is np.ndarray, "v2 must be numpy array"
    r, p = pearsonr(v1,v2)
    if p >= p_val:
        d = 1.0
    else:
        d = 1-r
    return d


def get_adjacency_pmsd(data_x):
    """
    Adjacency matrix similar to https://arxiv.org/abs/1703.03020
    :return:
    """
    dm = pdist(data_x, metric=node_similarity_pmsd)
    Adj = squareform(dm)
    return np.matrix(Adj)


def get_multi_adjacency(data_meta):
    """
    Get adjacency matrices given meta-information matrix and list of column
    names to include.
    """
    adj_list = []
    for col in data_meta.columns:
        if col == 'age':
            dm = pdist(data_meta[[col]], age_node_similarity)
        elif col in ['gender', 'apoe', 'clinical_scores']:
            dm = pdist(data_meta[[col]], binary_node_similarity)
        else:
            raise Exception('Meta information column name is incorrect!')
        cur_adj = squareform(dm)
        cur_adj = np.matrix(cur_adj)
        adj_list.append(cur_adj)
    return adj_list


def get_multi_adjacency_emvert(data_meta):
    """
    Get adjacency matrices given meta-information matrix and list of column
    names to include.
    """
    adj_list = []
    for col in data_meta.columns:
        if col == 'age':
            dm = pdist(data_meta[[col]], age_node_similarity_emvert)
        elif col in ['abcd2']:
            dm = pdist(data_meta[[col]], abcd2_node_similarity_emvert)
        else:
            raise Exception('Meta information column name is incorrect!')
        cur_adj = squareform(dm)
        cur_adj = np.matrix(cur_adj)
        adj_list.append(cur_adj)
    return adj_list


def get_multi_adjacency_thyroid(data_meta):
    """
    Get adjacency matrices given meta-information matrix and list of column
    names to include.
    """
    adj_list = []
    for k, col in enumerate(data_meta.columns):
        dm = pdist(data_meta.iloc[:, k:k+1], binary_node_similarity)
        cur_adj = squareform(dm)
        cur_adj = np.matrix(cur_adj)
        adj_list.append(cur_adj)
    return adj_list


def get_multi_adjacency_cpet(data_meta):
    """
    Get adjacency matrices given meta-information matrix and list of column
    names to include.
    """
    adj_list = []
    for k, col in enumerate(data_meta.columns):
        if k == 2:
            dm = pdist(data_meta.iloc[:, k:k + 1], age_node_similarity_cpet)
        else:
            dm = pdist(data_meta.iloc[:, k:k+1], binary_node_similarity)
        cur_adj = squareform(dm)
        cur_adj = np.matrix(cur_adj)
        adj_list.append(cur_adj)
    return adj_list


def age_node_similarity_cpet(u,v):
    """
    If absolute age difference is less than 6, nodes are connected.
    :return:
    """
    age_diff = 6
    ret_val = 0
    if np.abs(u - v) < age_diff:
        ret_val += 1
    return ret_val


def get_multi_adjacency_ppmi(data_meta):
    """
    Get adjacency matrices given meta-information matrix and list of column
    names to include.
    """
    adj_list = []
    for col in data_meta.columns:
        if col == 'AGE':
            dm = pdist(data_meta[[col]], age_node_similarity_ppmi)
        elif col in ['MCATOT', 'UPPDRS', 'GENDER']:
            dm = pdist(data_meta[[col]], binary_node_similarity)
        else:
            raise Exception('Meta information column name is incorrect!')
        cur_adj = squareform(dm)
        cur_adj = np.matrix(cur_adj)
        adj_list.append(cur_adj)
    return adj_list


def age_node_similarity_emvert(u,v):
    """
    If absolute age difference is less than 6, nodes are connected.
    :return:
    """
    age_diff = 6
    ret_val = 0
    if np.abs(u - v) < age_diff:
        ret_val += 1
    return ret_val


def abcd2_node_similarity_emvert(u,v):
    """
    If absolute age difference is less than 6, nodes are connected.
    :return:
    """
    _diff = 2
    ret_val = 0
    if np.abs(u - v) < _diff:
        ret_val += 1
    return ret_val


def age_node_similarity(u,v):
    """
    If age difference is less than nodes are connected.
    :return:
    """
    age_diff = 2
    ret_val = 0
    if np.abs(u - v) < age_diff:
        ret_val += 1
    return ret_val


def age_node_similarity_ppmi(u,v):
    """
    If age difference is less than nodes are connected.
    :return:
    """
    age_diff = 2
    ret_val = 0
    if np.abs(u - v) <= age_diff:
        ret_val += 1
    return ret_val


def binary_node_similarity(u,v):
    """
    Count how many binary elements are similar.
    :return:
    """
    deg_similarity = u == v
    ret_val = np.sum(deg_similarity)
    return ret_val


def node_similarity(u,v):
    """
    Node similarity similar to https://arxiv.org/abs/1703.03020.
    First value in u and v should be AGE, the next is Gender.
    :param u:
    :param v:
    :return:
    """
    # Age
    age_diff = 2
    ret_val = 0
    # for i in range (65, 105, 5):
    #     if u[0] >= i:
    #         ret_val += 1

    # Age diff of less than 2
    if np.abs(u[0]-v[0]) < age_diff:
        ret_val += 1
    #
    #  # APOE4
    if u[2] == v[2]:
        ret_val += 1

    # # GMC MICCAI 2018
    # if (u[0] - v[0]) < age_diff:
    #     ret_val += 1

    # Gender
    if u[1] == v[1]:
        ret_val += 1
    return ret_val


def node_similarity_pmsd(u,v):
    """
    Node similarity similar to https://arxiv.org/abs/1703.03020.
    First value in u and v should be AGE, the next is Gender.
    :param u:
    :param v:
    :return:
    """
    # Age
    age_diff = 2
    ret_val = 0

    # Age diff of less than 2
    # if np.abs(u[1]-v[1]) < age_diff:
    #     ret_val += 1
    # Sex
    if u[0] == v[0]:
        ret_val += 1

    return ret_val


def calc_percentage_missing(df):
    """
    Calculate the percentage of missing data in a given table.

    Parameter:
        df: pandas dataframe
            Dataframe
    Returns:
        res_percentage: float
            Percentage of missing elements in dataframe    
    """
    sum_elements = np.product(df.shape[:])
    sum_missing_elem = df.isnull().sum().sum()
    return (sum_missing_elem / sum_elements) * 100


def df_to_numeric(df, na, na_val=np.nan):
    """
    Convert dataframe columns with object dtype to numeric.

    Parameters:
        df: pandas dataframe
            Dataframe to convert
        na: list of values to replace
        na_val: float, default is np.nan
            Value of replacement
    Return:
    res_df: pandas dataframe
        (Numeric data, Dataframe with still object dtypes)

    """
    cur_df = df.copy()
    for i in na:
        if type(i) is str:
            cur_df.replace(i, na_val, regex=True, inplace=True)
        else:
            cur_df.replace(i, na_val, inplace=True)

    cur_df = cur_df[cur_df.columns[cur_df.dtypes == object]].convert_objects(
        convert_numeric=True)
    return cur_df


def load_pickle():
    """Load intermediate pickle results."""
    res_list = os.listdir('pkl_logs')
    res_list = [os.path.join('pkl_logs', x) for x in res_list]
    res_list = [x for x in res_list if '.pkl' in x]
    res_list = res_list[-1:] # last one will contain all the previous trials
    trials_data = []
    for res in res_list:
        print(res)
        with open(res, 'rb') as f:
            cur_res = pickle.load(f)
            for i in cur_res:
                print(i[1].best_trial)


def load_final_pickle():
    """ Load figs2 pickle results containing all folds."""
    # Load results
    with open('pkl_logs/event_2ZZOBYX5NWN1JUIgmc_TADPOLE_debug3_40percent_not_knn_graphall.pkl', 'rb') as f:
        cur_res = pickle.load(f)
    # Average AUC and Accuracy
    mean_auc = np.mean([x[0][0] for x in cur_res])
    mean_acc = np.mean([x[0][1] for x in cur_res])
    print("mean AUC: %0.3f" %mean_auc)
    print("mean ACC: %0.3f" %mean_acc)


def limit_gpu_usage(p=0.9):
    """
    Limit GPU usage to given percentage p

    Parameters:
        p: float
            between 0 and 1, default is 0.9

    Returns:
        None

    Example:
        import keras.backend.tensorflow_backend as KB
        KB.set_session(limit_gpu_usage())
    """
    num_threads = os.environ.get('OMP_NUM_THREADS')
    gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=p,
                                allow_growth=True)
    if num_threads:
        config = tf.ConfigProto(gpu_options=gpu_options,
                                intra_op_parallelism_threads=num_threads)
    else:
        config = tf.ConfigProto(gpu_options=gpu_options)
    return tf.Session(config=config)


def load_pkl_metrics(pkl_file):
    # Load gmc output numpy for every experiment
    # pkl_file = './gmc_logs/gmc_res_prior_graph.pkl'
    with open(pkl_file, 'rb') as f_pkle:
        gmc_output = pickle.load(f_pkle)

    return gmc_output


def pretty_metrics(gmc_output):
    """
    Given a list which contains a tuple of (ground truth, sigmoid_output,
    test_pred, test_RMSE) calculate, ROC-AUC, accuracy, f1-score. Then return
    classification and imputation metrics as one table.
    """
    # For every experiment calculate classification metrics
    pretty_output_list = []
    p_list = [0.25, 0.5, .75, 1.]
    if len(gmc_output[0][0][0].shape) > 1 and gmc_output[0][0][0].shape[-1] > 1:
        cur_data = gmc_output[0][0][0]
        num_class = cur_data.shape[-1]
    else:
        num_class = 1
    print(num_class)
    for k, experiment in enumerate(gmc_output):
        cur_auc = [roc_auc_score(x[0], x[1]) for x in
                   experiment]
        if num_class > 1:
            print('num_class here is %d' %num_class)
            cur_acc = [accuracy_score(np.argmax(x[0], 1), x[2]) for x in
                       experiment]
            cur_f1 = [f1_score(np.argmax(x[0], 1), x[2], average='weighted')
                      for x in experiment]

            # Weighted average of recall and precision
            clf_report = \
                [classification_report(np.argmax(x[0],1), x[2],
                                output_dict=True) for x in experiment]
            clf_report = [pd.DataFrame(x) for x in clf_report]

            cur_recall = [x.T for x in clf_report]
            cur_recall = [x.recall['weighted avg'] for x in cur_recall]
            cur_specificity = [x.T for x in clf_report]
            cur_specificity = [x.precision['weighted avg'] for x in
                               cur_specificity]
        else:
            cur_acc = [accuracy_score(x[0], x[2]) for x in experiment]
            cur_f1 = [f1_score(x[0], x[2]) for x in experiment]
            cur_recall = [pd.DataFrame(classification_report(x[0], x[2], \
                     output_dict=True)).T.recall[1] for x in experiment]

            cur_specificity = [pd.DataFrame(classification_report(x[0], x[2],
                          output_dict=True)).T.recall[0] for x in experiment]
        cur_rmse = [x[3] for x in experiment]


        cur_append = [cur_auc, cur_acc, cur_rmse, cur_f1, cur_recall, cur_specificity]
        cur_append = pd.DataFrame(cur_append).T
        cur_append.columns = ['AUC', 'Accuracy','RMSE', 'F-measure',
                              'Sensitivity', 'Specificity']
        cur_append['%'] = p_list[k]
        pretty_output_list.append(cur_append)
    res_df = pd.concat(pretty_output_list)
    return res_df


def pretty_table(pkl_list):
    """
    Given a list of pickle file. Create a dataframe for all results in that pickle file.
    """
    all_df = []
    for p in pkl_list:
        print(p)
        with open(p, 'rb') as pkl:
            cur_data = pickle.load(pkl)
            cur_res_df = pretty_metrics(cur_data)
            cur_res_df['model'] = p.split('/')[-1].replace('.pkl', '')
        all_df.append(cur_res_df)
    return all_df


def pretty_box_plots(all_df_as_one, order=None, color_pal=None,
                     barplot=False, plt_legend_loc=None):
    """
    Plot all results in dataframe to boxplot or barplot.

    Parameters
    ----------
    all_df_as_one: pandas.DataFrame
    order: list of str
        List of string names to use in plot legend.
    color_pal:
        Color pallate in matplotlib
    barplot: bool
        If True, will plot barplot instead of boxplot.

    Returns
    -------
    list of matplotlib figures.

    """

    list_of_fig = []
    plt.rcParams.update({'font.size': 18})
    for i in range(0, 6):
        plot_title = all_df_as_one.columns[i:i + 1][0]
        print('Working on %s' % plot_title)
        long_df = pd.melt(all_df_as_one, id_vars=['%'],
                          value_vars=list(all_df_as_one.columns[i:i+1])[0])
        long_df['model'] = list(all_df_as_one.model)

        print(long_df)
        if plot_title == 'Accuracy':
            long_df.value = long_df.value * 100

        cur_fig = plt.figure(figsize=(10, 3), dpi=300)
        plt.title(plot_title, y=1.05)

        if barplot:
            ax = sns.barplot(x="%", y="value", hue="model", data=long_df,
                         hue_order=order, palette=color_pal, ci='sd')
        else:
            ax = sns.boxplot(x="%", y="value", hue="model", data=long_df,
                             hue_order=order, palette=color_pal)
        plotter = sns.categorical._BoxPlotter(x="%", y="value", hue="model",
                                     data=long_df.copy(), order=None,
                                              hue_order=order,
                                              palette=color_pal, orient=None,
                                              width=.8, color=None,
                                              saturation=.75, dodge=True,
                                              fliersize=5, linewidth=None)
        ax.yaxis.grid()
        print(plot_title)
        if plt_legend_loc is None:
            plt.legend(loc=(1.01, 0))
        else:
            plt.legend(loc=plt_legend_loc)

        if plot_title in ['AUC', 'F-measure', 'Specificity',
                          'Sensitivity']:
            plt.ylim(None, 1)
            plt.yticks(np.arange(0, 1.01, .10))
        elif plot_title in ['Accuracy']:
            plt.ylim(None, 100)
            plt.yticks(np.arange(0, 101, 10))
        elif plot_title in ['RMSE']:
            plt.yticks(np.arange(0, 1.01, .10))
        # plt.show()
        list_of_fig.append((plotter, cur_fig))
    return list_of_fig


def get_p_value(df, best_model, compare_model, list_iter):
    """
    Check if best_model is significantly different with compare_model. Will
    compute p-values for all classification metrics.

    Parameters
    ----------
    df: pd.DataFrame
    best_model: str
        Best model's name in model column of df.
    compare_model: str
        Compare with model's name in model column of df.
    Returns
    -------
    res_: dict
        {'model_name': {'metric1':p-value, 'metric2':p-value}}
    """
    stat_metric = stats.wilcox_test
    res_dict = {}
    for i in list_iter:
        cur_dict = {}
        print(best_model, 'vs', compare_model)
        df_1 = df[(df['%'] == i) & (df.model == best_model)]
        df_2 = df[(df['%'] == i) & (df.model == compare_model)]
        for j in ['AUC', 'Accuracy', 'RMSE', 'F-measure', 'Sensitivity',
                  'Specificity']:
            cur_d1 = np.array(df_1[j])
            cur_d2 = np.array(df_2[j])
            if j == 'RMSE':
                cur_d1 = np.nan_to_num(cur_d1)
                cur_d2 = np.nan_to_num(cur_d2)
            print(i, j)
            print(cur_d1)
            print(cur_d2)

            cur_d1 = numpy2ri(cur_d1.copy())
            cur_d2 = numpy2ri(cur_d2.copy())

            res_p = stat_metric(cur_d1, cur_d2)
            res_p = np.array(res_p[2])[0]
            cur_dict[j] = res_p
        res_dict[i] = cur_dict
    res_ = dict()
    res_[compare_model] = res_dict
    return res_


def plot_asterisk(cur_bp, cur_fig, is_sig_dict, y_axis=1.03, str_text='*'):
    """
    Plot asterisk above the axis given dict of model names.

    Parameters
    ----------
    cur_bp: seaborn.categorical._Boxplotter
        Output from seaborn _Boxplotter class. Will use to get x-positions.
    cur_fig: matplotlib.figure.Figure
        Figure to annotate on. Output from seaborn plotter.
    is_sig_dict: dict
        {25: ['model_name1', 'model_name2'], 50:['model_name2']}
    y_axis: float
        Location in y dimension.
    str_text: str
        Text to annotate on figure.

    Returns
    -------
    cur_fig: matplotlib.figure.Figure
        Returns the annotated figure.
    """
    cur_ax = cur_fig.get_axes()[0]

    # For every percentage
    for i in is_sig_dict:
        cur_val = is_sig_dict[i]

        # For every model name at current percentage annotate asterisk
        for j in cur_val:
            x_pos = cur_bp.group_names.index(i) + cur_bp.hue_offsets[
                cur_bp.hue_names.index(j)]
            cur_ax.text(x_pos, y_axis, str_text)
    return cur_fig


def get_all_p_values(list_model_names, best_model_name, dframe, list_iter):
    """
    Compare if given list of models is significantly different with the
    best_model_name. Default p-value threshold is 0.05.

    Parameters
    ----------
    list_model_names: list of str
        List of model names.
    best_model_name: str
        Best model name to compare against.
    dframe: pandas.DataFrame
        Dataframe containing all information.
    Returns
    -------
    new_df: pandas.DataFrame
        Dataframe containing all p-values.
    """
    p_val_all = {}
    for i in list_model_names:
        cur_val_ = get_p_value(dframe, best_model_name, i, list_iter)
        p_val_all.update(cur_val_)
    new_df = pd.Panel(p_val_all).to_frame()
    return new_df


def get_model_names(bool_df, metric_name, p_list=[25, 50, 75, 100]):
    """
    Given dataframe of booleans, return the column name for every index where
    value is True. {25:['Mean+LR'], 50:['Mean+LR'], 100:['Mean+LR',
    'MICE+RF']}

    Parameters
    ----------
    bool_df: pandas.DataFrame
        Dataframe containing booleans. Where major index in
        classification metric name, minor index is percentage of available
        entries, and column names are model names.
    metric_name: str
        Major index name of interest.
    p_list: list of ints
        [25,50,75,100]

    Returns
    -------
    cur_dict: dict
        {25:['Mean+LR'], 50:['Mean+LR'], 100:['Mean+LR', 'MICE+RF']}
    """
    cur_dict = dict()
    for i in p_list:
        cur_df = bool_df.loc[metric_name]
        col_names = cur_df.columns
        # print(metric_name, i)
        mask = list(cur_df.loc[i])
        cur_model_names = list(col_names[mask])
        cur_dict[i] = cur_model_names
    return cur_dict


def annotate_all_figures(best_model, data_df, fig_list, model_names,
                         y_axis=1.01, rmse_mode=False):
    """
    Annotate all given figures based on p-val <= 0.05. Will add an asterisk
    above the box when there is a significant difference between comparison
    model vs best model.

    Parameters
    ----------
    best_model: str
        Model name of best model
    data_df: pandas.DataFrame
        Dataframe containing all metric results.
    fig_list: list of tuple (_Boxplotter, plt.Figure)
        Output from pretty_box_plots.
    rmse_mode: bool
        If True, will annotate RMSE figure only.

    Returns
    -------
    new_fig_list: list of plt.Figure
        Annotated list of figures.
    """
    if rmse_mode:
        iter_list = [25, 50, 75]
    else:
        iter_list = [25, 50, 75, 100]

    all_p_val_df = get_all_p_values(model_names, best_model, data_df, iter_list)
    bool_all_pval_df = all_p_val_df <= 0.05

    new_fig_list = []
    # rmse_y_min= data_df.RMSE.std()
    # Annotate each figuannotate_all_figuresre excluding figure with RMSE as title/metric_name.
    for i in fig_list:
        plt.clf()
        cur_bp = i[0]
        cur_fig = i[1]
        print(len(cur_fig.get_axes()))
        cur_ax = cur_fig.get_axes()[0]
        metric_name = cur_ax.get_title()

        if rmse_mode is False and metric_name == 'RMSE':
            pass
        else:
            is_sig_dict = get_model_names(bool_all_pval_df, metric_name,
                                          iter_list)

            if metric_name == 'Accuracy':
                new_fig = plot_asterisk(cur_bp, cur_fig, is_sig_dict,
                                        y_axis=y_axis + 100)
                new_fig_list.append(new_fig)
            else:
                new_fig = plot_asterisk(cur_bp, cur_fig, is_sig_dict,
                                        y_axis=y_axis)
                new_fig_list.append(new_fig)

    return new_fig_list



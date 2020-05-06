import os
from hyperopt import hp, Trials, STATUS_OK, STATUS_FAIL
import hyperopt
import string
import random
import pickle
from joblib import Parallel, delayed
import tensorflow as tf
import numpy as np
import time
import pandas as pd
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.metrics import roc_auc_score, accuracy_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
import data_loader as gmc_load
import models


class MultiGMCTrainer:
    """
    # Multi-graph geometric matrix completion model trainer class for
    TADPOLE dataset.
    """
    def __init__(self, rand_int, max_trials, trials_step, pickle_str,
                 autoregressive, overfit, is_training, use_knn_graph=False,
                 val_p=0.10, data_percentage=1.0, with_residual=False,
                 with_self_attention=False, is_init_labels=False,
                 separable_gmc=False, args=None):

        self.rand_int = rand_int  # Random int to use for random seed
        self.max_trials = max_trials  # Max trials before saving hyperopt
        self.trials_step = trials_step  # Number of trials to add
        self.pickle_str = pickle_str  # .pkl file filename
        self.autoregressive = autoregressive  # Autoregressive RNN model
        self.overfit = overfit  # Training error for overfit experiments.
        self.is_training = is_training  # Is it training mode
        self.use_knn_graph = use_knn_graph  # Use a KNN graph
        self.val_p = val_p  # % of validaiton set to use from train set
        self.data_percentage = data_percentage  # Percentage of known entries
        self.data_x = None
        self.data_y = None
        self.data_meta = None
        self.M = None
        self.initial_tr_mask = None
        self.Lrow = None
        self.A_matrix_row = None
        self.impute_idx = None
        self.with_residual = with_residual
        self.with_self_attention = with_self_attention
        self.is_init_labels = is_init_labels
        self.separable_gmc = separable_gmc
        self.args = args

    def load_data(self):
        """ Method to load dataset. """
        raise NotImplementedError('Please implement data loader.')

    def objective(self, params):
        """
        # Objective function to optimize hyperparameters on. Is using
        main_gmc as the main function to return 10-fold CV results evaluated
        on the validation set.
        """
        print('================================================')
        print('============Testing hyperparams {}'.format(params))
        print('================================================')

        try:
            train_res = self.main_gmc(params)
            print("Mean bce {}".format(train_res))
            return {'loss': train_res, 'status': STATUS_OK, 'space': params}
        except Exception as err:
            print(err)
            return {'loss': np.nan, 'status': STATUS_FAIL, 'space': params}

    def optimize(self):
        """
        Hyperopt optimization main function.
        """
        print('Running hyperparam search...')
        # how many additional trials to do after loading saved
        trials_step = self.trials_step

        # Use something small to not have to wait
        max_trials = self.max_trials

        try:
            # https://github.com/hyperopt/hyperopt/issues/267
            trials = pickle.load(open(self.pickle_str, "rb"))
            print("Found saved Trials! Loading...")
            # print("Rerunning from {} trials to add another one.".format(
            #     len(trials.trials)))

            # max_trials = len(trials.trials) + max_trials
            # print("Rerunning from {} trials to {} (+{}) trials".format(
            #     len(trials.trials), max_trials, trials_step))
            print("Rerunning from {} trials".format(len(trials.trials)))
        except Exception as err:
            print(err)
            trials = Trials()
            print("Starting from scratch: new trials created.")

        if self.use_knn_graph:
            if self.separable_gmc:
                space = {
                    'm_rank': hyperopt.hp.choice('m_rank_',
                                                 list(range(10, 841))),
                    'cheby_ord': hyperopt.hp.choice('cheby_ord_',
                                                    list(range(1, 20))),
                    'lr': hp.uniform('lr_', 0.00001, 0.1),
                    'k_nn': hyperopt.hp.choice('knn_', list(range(10, 200))),
                    'num_units': hyperopt.hp.choice('num_units_',
                                                    list(range(8, 512))),
                    'gamma_H': hp.uniform('gamma_H_', 0.001, 1000),
                    'gamma_W': hp.uniform('gamma_W_', 0.001, 1000),
                    'gamma': hp.uniform('gamma', 0.001, 1000),
                    'gamma_tr': hp.uniform('gamma_tr_', 0.001, 1000),
                    'gamma_bce': hp.uniform('gamma_bce_', 0.001, 1000)
                }
            else:
                space = {
                    'cheby_ord': hyperopt.hp.choice('cheby_ord_',
                                                    list(range(1, 20))),
                    'lr': hp.uniform('lr_', 0.00001, 0.1),
                    'k_nn': hyperopt.hp.choice('knn_', list(range(10, 200))),
                    'num_units': hyperopt.hp.choice('num_units_',
                                                    list(range(8, 512))),
                    'gamma': hp.uniform('gamma', 0.001, 1000),
                    'gamma_tr': hp.uniform('gamma_tr_', 0.001, 1000),
                    'gamma_bce': hp.uniform('gamma_bce_', 0.001, 1000)
                }

        else:
            if self.separable_gmc:
                space = {
                    'm_rank': hyperopt.hp.choice('m_rank_', list(range(10, 841))),
                    'cheby_ord': hyperopt.hp.choice('cheby_ord_',
                                                    list(range(1, 20))),
                    'lr': hp.uniform('lr_', 0.00001, 0.1),
                    'num_units': hyperopt.hp.choice('num_units_',
                                                    list(range(8, 512))),
                    'gamma': hp.uniform('gamma', 0.001, 1000),
                    'gamma_H': hp.uniform('gamma_H_', 0.001, 1000),
                    'gamma_W': hp.uniform('gamma_W_', 0.001, 1000),
                    'gamma_tr': hp.uniform('gamma_tr_', 0.001, 1000),
                    'gamma_bce': hp.uniform('gamma_bce_', 0.001, 1000)
                }
            else:
                space = {
                    'cheby_ord': hyperopt.hp.choice('cheby_ord_',
                                                    list(range(1, 20))),
                    'lr': hp.uniform('lr_', 0.00001, 0.1),
                    'num_units': hyperopt.hp.choice('num_units_',
                                                    list(range(8, 512))),
                    'gamma': hp.uniform('gamma', 0.001, 1000),
                    'gamma_tr': hp.uniform('gamma_tr_', 0.001, 1000),
                    'gamma_bce': hp.uniform('gamma_bce_', 0.001, 1000)
                }
        # min_obj = partial(self.objective, rand_int=self.rand_int)
        best_model = hyperopt.fmin(self.objective, space, trials=trials,
                                   algo=hyperopt.tpe.suggest,
                                   max_evals=max_trials)

        print(best_model)
        best_param = hyperopt.space_eval(space, best_model)
        print(best_param)
        with open(self.pickle_str, "wb") as f:
            pickle.dump(trials, f)

    def main_gmc(self, param):
        """
        # TODO
        """
        uniq_train_str = ''.join(random.SystemRandom().choice(
            string.ascii_uppercase + string.digits) for
                                 _ in range(15))
        uniq_train_str = uniq_train_str + '_%s_' % self.data_percentage

        # Load dataset including adjacency matrices
        self.load_data()

        # Stratified 10-fold cross-validation
        random_state = np.random.RandomState(self.rand_int)

        if self.args.cv_folds is None:
            n_splits = 10
        else:
            n_splits = self.args.cv_folds
        cv = StratifiedKFold(n_splits=n_splits, random_state=random_state)
        self.data_x = np.array(self.data_x)
        self.data_y = np.squeeze(np.array(self.data_y))
        if len(self.data_y.shape) > 1 and self.data_y.shape[-1] > 1:
            cv_y = np.argmax(self.data_y, 1)
        else:
            cv_y = self.data_y
        start_time_eval_miccai = time.time()

        # If is training will return validation bce loss. Otherwise will return
        # test set values (ground_truth, test_prediction, test_rsme)
        if self.is_training:
            result_training = \
                Parallel(n_jobs=5)(delayed(self.train_gmc) \
                                       (self.M,
                                        self.data_y,
                                        v[0], v[1],
                                        param,
                                        self.initial_tr_mask,
                                        self.Lrow,
                                        self.impute_idx,
                                        uniq_train_str=uniq_train_str,
                                        val_p=self.val_p,
                                        loop_idx=k,
                                        plot_tensorboard=False,
                                        is_training=self.is_training,
                                        autoregressive=self.autoregressive,
                                        overfit=self.overfit)
                for k, v in enumerate(cv.split(self.data_x, cv_y)))
        else:
            result_training = \
                Parallel(n_jobs=5)(delayed(self.train_gmc) \
                                       (self.M,
                                        self.data_y,
                                        v[0], v[1],
                                        param,
                                        self.initial_tr_mask,
                                        self.Lrow,
                                        self.impute_idx,
                                        uniq_train_str=uniq_train_str,
                                        val_p=self.val_p,
                                        loop_idx=k,
                                        plot_tensorboard=False,
                                        is_training=False,
                                        autoregressive=self.autoregressive,
                                        overfit=self.overfit)
                 for k, v in enumerate(cv.split(self.data_x, cv_y)))

        # During traning return validation bce. Otherwise, return ground truth
        # and pred output as well as test RMSE.
        if self.is_training:
            output = np.mean(result_training)
        else:
            output = result_training
        return output

    def train_gmc(self, M, data_y, train_index, test_index, param,
                  initial_tr_mask,
                  Lrow, impute_idx, uniq_train_str, val_p, loop_idx,
                  plot_tensorboard=False, is_training=True,
                  autoregressive=False,
                  overfit=False):
        """
        # TODO
        """
        if self.separable_gmc:
            M_rank = param['m_rank']
            gamma_H = param['gamma_H']
            gamma_W = param['gamma_W']
        cheby_order = param['cheby_ord']
        lr = param['lr']
        if 'k_nn' in param.keys():
            k_neighbors = param['k_nn']
        else:
            k_neighbors = None
        n_conv_feat = param['num_units']
        gamma = param['gamma']
        gamma_tr = param['gamma_tr']
        gamma_bce = param['gamma_bce']

        # gcn_utils.limit_gpu_usage(.25)
        # print('Training fold {}'.format(cv_iter + 1))
        data_x_shape = M.shape
        train_mask = np.zeros(data_x_shape[0], dtype=bool)

        # Validation set
        if val_p != 0 and val_p < 1:
            # Use part of the training set as validation set
            # Given val_p as the percentage to take from training set
            val_idx_len = round(len(train_index) * val_p)
            val_mask_idx = train_index[-val_idx_len:]
            train_mask_idx = train_index[:-val_idx_len]
            random_state = np.random.RandomState(0)
            train_mask_idx, val_mask_idx, _, _ = \
                train_test_split(train_index, train_index, test_size=0.10,
                                 random_state=random_state)


            # Train, test, and validation mask
            test_mask = train_mask.copy()
            test_mask[test_index] = True
            val_mask = train_mask.copy()
            train_mask[train_mask_idx] = True
            val_mask[val_mask_idx] = True

        else:
            train_mask[train_index] = True
            test_mask = ~train_mask
            val_mask = test_mask

        # Use all features X from both train and test set. Including train
        # labels.
        cur_tr_mask = train_mask.astype(int)
        cur_tr_mask = np.expand_dims(cur_tr_mask, 1)
        if len(data_y.shape) > 1 and  data_y.shape[-1] > 1:
            cur_tr_mask = np.repeat(cur_tr_mask, data_y.shape[-1], axis=1)

        if overfit:
            cur_tr_mask = np.ones_like(cur_tr_mask)
        cur_initial_tr_mask = np.hstack([initial_tr_mask, cur_tr_mask])
        cur_initial_ts_mask = test_mask.astype(int)
        cur_initial_val_mask = val_mask.astype(int)
        print(
            "Num of inner loop train rows = %d, val rows = %d, test rows = %d" %
            (train_mask.sum(), val_mask.sum(), test_mask.sum()))

        # Numpy version of SVD
        # Apply SVD initially for detecting the main components of our
        # initialization
        if self.separable_gmc:
            U, s, V = np.linalg.svd(np.multiply(M, cur_initial_tr_mask),
                                    full_matrices=0)
            print(U.shape)
            print(s.shape)
            print(V.shape)
            rank_W_H = M_rank
            print("Matrix rank used is {}".format(rank_W_H))
            partial_s = s[:rank_W_H]
            partial_S_sqrt = np.diag(np.sqrt(partial_s))
            initial_W = np.dot(U[:, :rank_W_H], partial_S_sqrt)
            initial_H = np.dot(partial_S_sqrt, V[:rank_W_H, :]).T

            print('New input matrix shapes for separable GMC')
            print(initial_W.shape)
            print(initial_H.shape)

        else:
            initial_W = np.multiply(M, cur_initial_tr_mask)

        cheby_ord = cheby_order
        ord_row = cheby_ord

        # Initialize input label column as all zeroes.
        if not self.is_init_labels:
            if self.args.adni1_baseline_only:
                num_class = self.data_y.shape[-1]
                initial_W[:, -num_class:] = 0.0
            else:
                initial_W[:, -1:] = 0.0

        # Model definition
        if self.autoregressive:
            if self.with_self_attention:
                model_ = models.MultiGMCARWithSelfAtt
            else:
                model_ = models.MultiGMCAR

        else:
            if self.with_self_attention:
                model_ = models.MultiGMCNonARWithSelfAtt
            else:
                model_ = models.MultiGMCNonAR

        if self.separable_gmc:
            learning_obj = model_(np.matrix(M), Lrow,
                                  cur_initial_tr_mask,
                                  cur_initial_ts_mask,
                                  cur_initial_val_mask,
                                  initial_W,
                                  initial_H=initial_H,
                                  labels_y=np.array(data_y),
                                  train_mask=train_mask,
                                  test_mask=test_mask,
                                  val_mask=val_mask,
                                  impute_idx=impute_idx,
                                  order_chebyshev_row=ord_row,
                                  num_iterations=10,
                                  gamma=gamma,
                                  gamma_H=gamma_H,
                                  gamma_W=gamma_W,
                                  gamma_tr=gamma_tr,
                                  gamma_bce=gamma_bce,
                                  learning_rate=lr,
                                  n_conv_feat=n_conv_feat,
                                  separable_gmc=self.separable_gmc,
                                  M_rank=M_rank, residual=self.with_residual)
        else:
            learning_obj = model_(np.matrix(M), Lrow,
                                  cur_initial_tr_mask,
                                  cur_initial_ts_mask,
                                  cur_initial_val_mask,
                                  initial_W,
                                  initial_H=None,
                                  labels_y=np.array(data_y),
                                  train_mask=train_mask,
                                  test_mask=test_mask,
                                  val_mask=val_mask,
                                  impute_idx=impute_idx,
                                  order_chebyshev_row=ord_row,
                                  num_iterations=10,
                                  gamma=gamma,
                                  gamma_tr=gamma_tr,
                                  gamma_bce=gamma_bce,
                                  learning_rate=lr,
                                  n_conv_feat=n_conv_feat,
                                  residual=self.with_residual)

        # num_iter_test = 10
        num_total_iter_training = self.args.epoch
        list_training_loss = list()
        list_validation_loss = list()
        list_training_norm_grad = list()
        list_training_times = list()

        # Tensorboard params
        if plot_tensorboard:
            if k_neighbors is None:
                TBOARD_NAME = "%s_rank%d_cheby%d_lr%0.5f" % (
                    uniq_train_str, M_rank, cheby_order, lr)
            else:
                TBOARD_NAME = "%s_rank%d_cheby%d_lr%0.5f_knneighbors%d" % (
                    uniq_train_str, M_rank, cheby_order, lr, k_neighbors)

            TBOARD_DIR = "./logs/" + TBOARD_NAME
            if not os.path.exists(TBOARD_DIR):
                os.makedirs(TBOARD_DIR)

            tb_writer = tf.summary.FileWriter(TBOARD_DIR)
            summary_tr = tf.Summary()
            summary_tr_value = summary_tr.value.add()

            summary_tr_bce = tf.Summary()
            summary_tr_bce_value = summary_tr_bce.value.add()

            summary_tr_rmse = tf.Summary()
            summary_tr_rmse_value = summary_tr_rmse.value.add()

            summary_ts = tf.Summary()
            summary_ts_bce_value = summary_ts.value.add()

            summary_ts_RMSE = tf.Summary()
            summary_ts_RMSE_value = summary_ts_RMSE.value.add()

            if val_p > 0.0 and val_p < 1.0:
                summary_val_bce = tf.Summary()
                summary_val_bce_value = summary_val_bce.value.add()

                summary_val_RMSE = tf.Summary()
                summary_val_RMSE_value = summary_ts_RMSE.value.add()

        # For early stopping
        patience = 10
        patience_cnt = 0
        num_iter = 0

        for k in range(num_iter, num_total_iter_training):
            tic = time.time()
            # Train model on train set
            # training_time = time.time() - tic
            _, current_training_loss, norm_grad, X_grad, tr_loss_bce, \
            train_RMSE = learning_obj.session.run([learning_obj.optimizer,
                                                   learning_obj.loss,
                                                   learning_obj.norm_grad,
                                                   learning_obj.var_grad,
                                                   learning_obj.tr_loss_bce,
                                                   learning_obj.train_RMSE])
            training_time = time.time() - tic

            list_training_loss.append(current_training_loss)
            list_training_norm_grad.append(norm_grad)
            list_training_times.append(training_time)
            # list_validation_loss.append(val_loss_bce)

            # Log training losses on tensorboard
            if plot_tensorboard:
                summary_tr_value.simple_value = current_training_loss
                summary_tr_value.tag = 'training_loss%s' % loop_idx
                tb_writer.add_summary(summary_tr, num_iter)

                summary_tr_bce_value.simple_value = tr_loss_bce
                summary_tr_bce_value.tag = 'training_bce%s' % loop_idx
                tb_writer.add_summary(summary_tr_bce, num_iter)

                summary_tr_rmse_value.simple_value = train_RMSE
                summary_tr_rmse_value.tag = 'tr_RMSE%s' % loop_idx
                tb_writer.add_summary(summary_tr_rmse, num_iter)

            # Validation
            val_loss_bce, val_RMSE, val_y_out = learning_obj.session.run([
                learning_obj.val_loss_bce, learning_obj.val_RMSE,
                learning_obj.val_Y_out])
            list_validation_loss.append(val_loss_bce)

            # Log validation losses on tensorboard
            if plot_tensorboard:
                summary_val_bce_value.simple_value = val_loss_bce
                summary_val_bce_value.tag = 'val_loss_bce%s' % loop_idx
                tb_writer.add_summary(summary_val_bce, num_iter)

                # Validation set RMSE
                summary_val_RMSE_value.simple_value = val_RMSE
                summary_val_RMSE_value.tag = 'val_RMSE%s' % loop_idx
                tb_writer.add_summary(summary_val_RMSE, num_iter)
            msg = "[VAL] current iter = {}, val_loss_bce = {}".format(num_iter,
                                                                      val_loss_bce)
            print(msg)

            # Early stopping
            if num_iter > 10 and list_validation_loss[-1] > np.mean(
                    list_validation_loss[-(10 + 1):-1]):
                print("Early stopping...")
                break
            # if (np.mod(num_iter, num_iter_test) == 0):
            #     # Early stopping
            #     if num_iter > 0 and list_validation_loss[num_iter - 1] - \
            #             list_validation_loss[num_iter] > MIN_DELTA:
            #         patience_cnt = 0
            #     else:
            #         patience_cnt += 1
            #
            #     if patience_cnt > patience:
            #         print("early stopping...")
            #         break

            num_iter += 1

        # Evaluate test set
        tic = time.time()
        ts_pred_error, ts_loss_bce, sigmoid_ts_y_out, ts_RMSE \
            = learning_obj.session.run(
            [learning_obj.predictions_error, learning_obj.ts_loss_bce,
             learning_obj.sigmoid_ts_Y_out, learning_obj.test_RMSE])
        test_time = time.time() - tic

        # Save test set imputations and indeces.
        # test_imputations, save_test_mask = learning_obj.session.run(
        #     [learning_obj.X, learning_obj.test_feature_mask])
        #
        # feat_map_dir = './feat_maps/'
        # feat_map_path = os.path.join(feat_map_dir, '%s_%s' % (uniq_train_str,
        #                                                       str(loop_idx)))
        # if not os.path.exists(feat_map_dir):
        #     os.makedirs(feat_map_dir)
        # np.save(feat_map_path + '_X', test_imputations)
        # np.save(feat_map_path + '_testmask', save_test_mask)

        # Output predicted labels of test set
        ts_y_pred_labels = np.zeros_like(sigmoid_ts_y_out)
        print('current prediction scores')
        print(sigmoid_ts_y_out)
        num_class = self.get_num_class()
        if num_class > 1:
            ts_y_pred_labels = np.argmax(sigmoid_ts_y_out, 1)
        else:
            ts_y_pred_labels[sigmoid_ts_y_out >= 0.5] = 1
        print('current prediction labels')
        print(ts_y_pred_labels)
        if is_training:
            out_prob = val_loss_bce
        else:
            out_prob = (data_y[test_mask], sigmoid_ts_y_out, ts_y_pred_labels,
                        ts_RMSE)
        return out_prob

    def train_dropping_entries_10cv(self, params, uniq_str,
                                    percentage_list=None):
        """
        #TODO
        """
        # percentage_list = [0.05, 0.1, 0.2, 0.3, 0.4, 1.0]
        if percentage_list is None:
            percentage_list = [.25, 0.5, 0.75, 1.0]

        res_tadpole_all = []
        for p in percentage_list:
            if self.use_knn_graph:
                cur_res = self.train_given_param(params, p)
            else:
                cur_res = self.train_given_param(params, p)
            res_tadpole_all.append(cur_res)

        # Save python object as pickle
        if self.use_knn_graph:
            output_fname = uniq_str + 'knn_age_graph.pkl'
        else:
            output_fname = uniq_str + 'prior_graph.pkl'
        with open(output_fname, 'wb') as write_pkl:
            pickle.dump(res_tadpole_all, write_pkl)

    def train_given_param(self, params, data_percentage):
        """
        # TODO
        Train model given hyperparameters
        :param param:
            Dictionary of hyperparameters.
        :return:
        """
        self.is_training = False
        self.data_percentage = data_percentage
        train_res = self.main_gmc(param=params)

        # train res (data_y[test_mask], sigmoid_ts_y_out, ts_pred_labels,
        # ts_RMSE)
        cur_auc = [roc_auc_score(x[0], x[1]) for x in train_res]
        num_class = self.get_num_class()
        if num_class > 1:
            cur_acc = [accuracy_score(np.argmax(x[0], 1), x[2]) for x in
                       train_res]
        else:
            cur_acc = [accuracy_score(x[0], x[2]) for x in train_res]
        mean_test_rmse = np.mean([x[3] for x in train_res])
        print("Mean AUC {}".format(np.mean(cur_auc)))
        print("Mean Accuracy {}".format(np.mean(cur_acc)))
        print("Mean test RMSE %s" % mean_test_rmse)

        return train_res

    def get_num_class(self):
        """ Get how many number of classes are there in the data y. Will only
        check if data has is n x class. """
        if len(self.data_y.shape) > 1 and self.data_y.shape[-1] > 1:
            self.num_class = self.data_y.shape[-1]
        else:
            self.num_class = 1
        return self.num_class


class MultiGMCTrainerPPMI(MultiGMCTrainer):
    """
    # Todo
    Multi-graph GMC trainer for PPMI dataset.
    """
    def __init__(self, rand_int, max_trials, trials_step, pickle_str,
                 autoregressive, overfit, is_training, use_knn_graph=False,
                 val_p=0.10, data_percentage=1.0, with_residual=False,
                 with_self_attention=False, is_init_labels=False,
                 separable_gmc=False, args=None):
        super().__init__(rand_int, max_trials, trials_step, pickle_str,
                         autoregressive, overfit, is_training,
                         use_knn_graph=use_knn_graph, val_p=val_p,
                         data_percentage=data_percentage,
                         with_residual=with_residual,
                         with_self_attention=with_self_attention,
                         is_init_labels=is_init_labels,
                         separable_gmc=separable_gmc, args=args)

    def load_data(self):
        """
        Data loader for PPMI dataset
        (https://www.ppmi-info.org/access-data-specimens/download-data/).
        """
        data_x_path = './data/PPMIdata/dim_reduction_output.txt'
        data_meta_path = './data/PPMIdata/non_imaging_ppmi_data.xls'
        data_x = np.genfromtxt(data_x_path, dtype='float', delimiter=',')

        # Load meta-information
        # MOCA
        data_moca = pd.read_excel(data_meta_path, 'MOCA')
        data_moca = data_moca[['PATNO', 'EVENT_ID', 'MCATOT']]
        data_moca = data_moca[(data_moca.EVENT_ID == 'SC') | (
                data_moca.EVENT_ID == 'V01')]
        data_moca.drop_duplicates('PATNO', inplace=True)

        # UPDRS
        load_data_uppdrs = pd.read_excel(data_meta_path, 'UPDRS')
        updrs_cols = load_data_uppdrs.columns[8:41]
        uppdrs_score = load_data_uppdrs[updrs_cols].sum(1)
        load_data_uppdrs = load_data_uppdrs[['PATNO', 'EVENT_ID']]
        load_data_uppdrs['uppdrs_score'] = uppdrs_score
        data_uppdrs = load_data_uppdrs[(load_data_uppdrs.EVENT_ID == 'BL') | (
                load_data_uppdrs.EVENT_ID == 'V01')]
        data_uppdrs.sort_values(['PATNO', 'uppdrs_score'], inplace=True)
        data_uppdrs.drop_duplicates(['PATNO'], inplace=True)

        # Gender
        data_gender = pd.read_excel(data_meta_path, 'Gender_and_Age')
        data_gender = data_gender[['PATNO', 'GENDER']]

        # Age
        data_age = pd.read_excel(data_meta_path, 'Gender_and_Age')
        age_ = 2018 - data_age['BIRTHDT']
        # data_acq = pd.read_csv('./data/PPMIdata/Magnetic_Resonance_Imaging.csv')
        # data_age = data_age[['PATNO', 'EVENT_ID', 'MRIDT']]
        data_age = data_age[['PATNO']]
        data_age['age'] = age_

        # From label 2 to 1 and from label 1 to 0
        # 1 Means with Parkinson's disease, 0 normal.
        data_y = pd.read_csv('./data/PPMIdata/ppmi_labels.csv', names=[
            'PATNO', 'labels'])
        data_y.labels = data_y.labels - 1

        # Merge MOCA
        new_data_meta = data_y[['PATNO']]
        new_data_meta = new_data_meta.merge(data_moca[['PATNO', 'MCATOT']],
                                            on=['PATNO'],
                                            how='left')

        # Merge UPPDRS score
        new_data_meta = new_data_meta.merge(data_uppdrs[['PATNO',
                                                        'uppdrs_score']],
                                            on='PATNO',
                                            how='left')
        # Use screening (SC) UPPDRS score of missing patient without BL/V01
        # UPPDRS score
        missing_id = list(new_data_meta.PATNO[
                              new_data_meta.uppdrs_score.isna()])
        include_uppdrs = load_data_uppdrs[load_data_uppdrs.PATNO.isin(
            missing_id)]
        # PATNO [60070, 3801, 3837, 4060, 4069, 3833]
        new_data_meta = new_data_meta.merge(include_uppdrs[['PATNO',
                                                           'uppdrs_score']],
                                            on=['PATNO'],
                                            how='left')
        new_uppdrs_score = new_data_meta.uppdrs_score_x.combine_first(
            new_data_meta.uppdrs_score_y)
        new_data_meta['uppdrs_score'] = new_uppdrs_score
        new_data_meta = new_data_meta[['PATNO', 'MCATOT', 'uppdrs_score']]

        # Merge age
        new_data_meta = new_data_meta.merge(data_age,
                                            on='PATNO',
                                            how='left')

        # Merge gender
        new_data_meta = new_data_meta.merge(data_gender,
                                            on='PATNO',
                                            how='left')

        # Remove PATNO column and rename columns
        new_data_meta.drop(columns=['PATNO'], inplace=True)
        new_data_meta.columns = ['MCATOT', 'UPPDRS', 'AGE', 'GENDER']

        # Meta informaton to use to build the graphs
        data_meta = new_data_meta

        # Drop PATNO column
        data_y.drop(columns=['PATNO'], inplace=True)

        # Permute data to test if experiment is stable under multiple experiment
        # iterations.
        if self.rand_int != 0:
            data_x = data_x.sample(frac=1.0, random_state=self.rand_int)
            data_y = data_y.sample(frac=1.0, random_state=self.rand_int)
            data_meta = data_meta.sample(frac=1.0,
                                         random_state=self.rand_int)
        # nan_idx = np.isnan(data_x)
        # normalize_pipeline = Pipeline(
        #     [("impute", SimpleImputer(strategy="mean", axis=0)),
        #      ("scale", StandardScaler())])
        # data_x = normalize_pipeline.fit_transform(data_x)
        # data_x[nan_idx] = np.nan  # nan will get converted to nan again

        if self.data_percentage < 1:
            print('Data x dimension {}'.format(data_x.shape))
            print('Will be using data_percentage {}'.format(
                self.data_percentage))
            data_x, impute_idx = gmc_load.select_p_data_df(data_x,
                                                           self.data_percentage,
                                                           random_seed=0)
            data_x = pd.DataFrame(data_x)
        elif self.data_percentage == 1:
            data_x = pd.DataFrame(data_x)
            impute_idx = None
            print('Data x dimension {}'.format(data_x.shape))
            print('Will be using data_percentage {}'.format(
                self.data_percentage))
        else:
            print('Wrong percentage value')
            raise ValueError

        # Variables for input to GMC
        M, initial_tr_mask, Lrow, A_matrix_row = gmc_load.format_data_mgmc_ppmi(
            data_x, data_meta, data_y, k_nn=None, row_g_metric=None,
            stack_y=True)
        self.data_x = data_x
        self.data_y = data_y
        self.data_meta = data_meta
        self.M = M
        self.initial_tr_mask = initial_tr_mask
        self.Lrow = Lrow
        self.A_matrix_row = A_matrix_row
        self.impute_idx = impute_idx


class MultiGMCTrainerTADPOLE(MultiGMCTrainer):
    """
    # Todo
    Multi-graph GMC trainer for TADPOLE dataset.
    """
    def __init__(self, rand_int, max_trials, trials_step, pickle_str,
                 autoregressive, overfit, is_training, use_knn_graph=False,
                 val_p=0.10, data_percentage=1.0, with_residual=False,
                 with_self_attention=False, is_init_labels=False,
                 separable_gmc=False, args=None):
        super().__init__(rand_int, max_trials, trials_step, pickle_str,
                         autoregressive, overfit, is_training,
                         use_knn_graph=use_knn_graph, val_p=val_p,
                         data_percentage=data_percentage,
                         with_residual=with_residual,
                         with_self_attention=with_self_attention,
                         is_init_labels=is_init_labels,
                         separable_gmc=separable_gmc, args=args)

    def load_data(self):
        """
        Load features, labels, and meta-information.
        """
        # Load dataset
        if self.args.adni1_baseline_only:
            data_x, data_meta, data_y = gmc_load.quick_load_adni_one_baseline()
            data_y = tf.keras.utils.to_categorical(data_y)
            data_y = pd.DataFrame(data_y)
        else:
            data_x, data_meta, data_y = gmc_load.quick_load_tadpole()

        # Permute data to test if experiment is stable under multiple experiment
        # iterations.
        if self.rand_int != 0:
            data_x = data_x.sample(frac=1.0, random_state=self.rand_int)
            data_y = data_y.sample(frac=1.0, random_state=self.rand_int)
            data_meta = data_meta.sample(frac=1.0,
                                         random_state=self.rand_int)

        # modalities = ['clinical_scores_df','mri_df', 'pet_df', 'dti_df',
        # 'csf_df']
        modalities = ['mri_df', 'pet_df', 'dti_df', 'csf_df']
        data_x = gmc_load.select_modalities(data_x, longitudinal=False,
                                            modalities=modalities)
        nan_idx = np.isnan(data_x)
        normalize_pipeline = Pipeline(
            [("impute", SimpleImputer(strategy="mean", axis=0)),
             ("scale", StandardScaler())])
        data_x = normalize_pipeline.fit_transform(data_x)
        data_x[nan_idx] = np.nan  # nan will get converted to nan again

        # data_x = emvert_ml.df_drop(data_x, 0.01)[0]
        if self.data_percentage < 1:
            print('Data x dimension {}'.format(data_x.shape))
            print('Will be using data_percentage {}'.format(
                self.data_percentage))
            data_x, impute_idx = gmc_load.select_p_data_df(data_x,
                                                           self.data_percentage,
                                                           random_seed=0)
            data_x = pd.DataFrame(data_x)
        elif self.data_percentage == 1:
            data_x = pd.DataFrame(data_x)
            impute_idx = None
            print('Data x dimension {}'.format(data_x.shape))
            print('Will be using data_percentage {}'.format(
                self.data_percentage))
        else:
            print('Wrong percentage value')
            raise ValueError
        # if loogistic_loss:
        #     data_y[data_y == 0] = -1

        data_meta = data_meta[['AGE', 'PTGENDER', 'APOE4']]
        data_meta.columns = ['age', 'gender', 'apoe']

        # Variables for input to GMC
        M, initial_tr_mask, Lrow, A_matrix_row = gmc_load.format_data_mgmc(
            data_x, data_meta, data_y, k_nn=None, row_g_metric=None,
            stack_y=True)

        self.data_x = data_x
        self.data_y = data_y
        self.data_meta = data_meta
        self.M = M
        self.initial_tr_mask = initial_tr_mask
        self.Lrow = Lrow
        self.A_matrix_row = A_matrix_row
        self.impute_idx = impute_idx

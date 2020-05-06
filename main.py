import os
import pickle
import argparse
from trainer import MultiGMCTrainerPPMI


def main(args):
    """
    Entry point to start geometric matrix completion.
    """
    ##################################
    # Argument defintions for training
    ##################################
    use_knn_graph = False
    train_hyperopt = args.train_hyperopt
    is_autoregressive = args.is_autoregressive
    model_name = args.experiment_name
    drop_list = [0.25, 0.5, 0.75, 1.0]

    if args.separable_gmc:
        model_name += 'Separable'

    if is_autoregressive:
        if args.with_self_attention:
            model_name += 'multiGMCARWithSelfAtt'
        else:
            model_name += 'multiGMCAR'
    else:
        if args.with_self_attention:
            model_name += 'multiGMCNonARWithSelfAtt'
        else:
            model_name += 'multiGMCNonAR'

    if args.with_residual:
        model_name += 'withResidual'

    if args.is_init_labels:
        model_name += 'InitLabels'
    else:
        model_name += '0.0InitLabels'

    ######################
    # Model trainer to use
    ######################
    trainer_ = MultiGMCTrainerPPMI

    # This is where hyperparameters will be saved.
    hyperopt_basedir = './hyperopt_logs/'
    hp_epochs = args.hp_epochs
    if use_knn_graph:
        raise NotImplementedError
    else:
        pickle_str = '%s_%sepochs.pkl' % (str(
            model_name), hp_epochs)
    pickle_str = os.path.join(hyperopt_basedir, pickle_str)

    ######################################
    # Run hyperopt hyperparameter training
    ######################################
    if train_hyperopt:
        if not os.path.exists(hyperopt_basedir):
            os.mkdir(hyperopt_basedir)
        print('Running Hyperopt ...')

        # Run experiments for n trial/s with random permutation of subjects.
        for trial_n in range(1):
            # cur_pickle_str = pickle_str.split('.pkl')[0] + '_exp_%s.pkl' % \
            # str(trial_n)
            replacement_str = '_exp_%s.pkl' % str(trial_n)
            cur_pickle_str = pickle_str.replace('.pkl', replacement_str)

            # Load dataset and use GMC trainer class to perform hp search.
            gmc_trainer = trainer_(rand_int=trial_n,
                                   max_trials=hp_epochs,
                                   trials_step=1,
                                   pickle_str=cur_pickle_str,
                                   autoregressive=args.is_autoregressive,
                                   overfit=args.is_overfit,
                                   is_training=True,
                                   use_knn_graph=use_knn_graph,
                                   with_residual=args.with_residual,
                                   with_self_attention=args.with_self_attention,
                                   is_init_labels=args.is_init_labels,
                                   separable_gmc=args.separable_gmc,
                                   args=args)
            gmc_trainer.optimize()
            print('=== Hyperopt training done ===')

            # Evaluate model using current best hyperparams
            gmc_basedir = './gmc_logs/'
            if not os.path.exists(gmc_basedir):
                os.mkdir(gmc_basedir)

            # Load best hyperparameters
            best_hyperparams = pickle.load(open(cur_pickle_str, "rb"))
            best_hyperparams = best_hyperparams.best_trial['result']['space']
            print('Running 10 x 10 CV ...')
            uniq_str = '%s_%sepochs' % (str(model_name), hp_epochs) + str(
                trial_n)
            uniq_str = os.path.join(gmc_basedir, uniq_str)
            trainer_.is_training = False
            gmc_trainer.train_dropping_entries_10cv(params=best_hyperparams,
                                                    uniq_str=uniq_str)

    #########################################
    # Evaluate test set given hyperparameters
    #########################################
    # Given best hyperparams train model and evaluate test set classification
    #  metrics and feature imputation.
    else:
        model_name += '_noHP'

        # GMC output
        gmc_basedir = './gmc_output/'
        if not os.path.exists(gmc_basedir):
            os.mkdir(gmc_basedir)

        # Load best hyperparameters
        for trial_n in range(1):
            cur_pickle_str = pickle_str.replace('.pkl', '_exp_%s.pkl' %
                                                trial_n)
            # Debug
            gmc_trainer = trainer_(rand_int=trial_n,
                                   max_trials=hp_epochs,
                                   trials_step=1,
                                   pickle_str=cur_pickle_str,
                                   autoregressive=args.is_autoregressive,
                                   overfit=args.is_overfit,
                                   is_training=False,
                                   use_knn_graph=use_knn_graph,
                                   with_residual=args.with_residual,
                                   with_self_attention=args.with_self_attention,
                                   is_init_labels=args.is_init_labels,
                                   separable_gmc=args.separable_gmc,
                                   args=args)

            if os.path.exists(cur_pickle_str):
                best_hyperparams = pickle.load(open(cur_pickle_str, "rb"))
                best_hyperparams = best_hyperparams.best_trial['result'][
                    'space']
            else:
                raise FileNotFoundError('Hyperopt hyperparameters file not '
                                        'found.')
            uniq_str = '%s_%sepochs' % (str(model_name), hp_epochs) + str(
                trial_n)
            uniq_str = os.path.join(gmc_basedir, uniq_str)
            gmc_trainer.train_dropping_entries_10cv(params=best_hyperparams,
                                                    uniq_str=uniq_str,
                                                    percentage_list=drop_list)


if __name__ == '__main__':
    # Command Line Parser
    parser = argparse.ArgumentParser(description='Multigraph Geometric Matrix '
                                                 'Completion')
    parser.add_argument('--train_hyperopt',
                        default=0,
                        type=int,
                        help='Find best hyperparameters.')
    parser.add_argument('--is_autoregressive',
                        default=0,
                        type=int,
                        help='If 1 will use autoregressive RGCNs, else will '
                             'use non-autoregressive MGMC.')
    parser.add_argument('--is_overfit',
                        default=0,
                        type=int,
                        help=' If 1 will overfit experiment, else will use '
                             'cross-validation.')
    parser.add_argument('--with_residual',
                        default=0,
                        type=int,
                        help='Add a residual connection for input '
                             'features to graph output by concatenatiion.')
    parser.add_argument('--with_self_attention',
                        default=1, type=int,
                        help='Use self attention to aggregate graph feature '
                             'information.')
    parser.add_argument('--experiment_name',
                        default='PPMI',
                        type=str,
                        help='Name of experiment, mainly used as unique str '
                             'for file outputs.')
    parser.add_argument('--is_init_labels',
                        default=1,
                        type=int,
                        help='If 1, will concatenate given training-set labels '
                             'as input to the model. Otherwise, will initialize'
                             'label column as all zeroes.')
    parser.add_argument('--separable_gmc',
                        default=0,
                        type=int,
                        help='If 1, will perform SVD to input features and'
                             'output W and H_transpose of size m x r and n x '
                             'r. Where r is the initial rank from hyperopt. '
                             'Otherwise, will just use the given full matrix.')
    parser.add_argument('--epoch',
                        default=500,
                        type=int,
                        help='Number of epochs to train the model.')
    parser.add_argument('--cv_folds',
                        default=10,
                        type=int,
                        help='Number of outer-loop cross-validation.')
    parser.add_argument('--hp_epochs',
                        default=120,
                        type=int,
                        help='Number hyperopt iterations.')

    parsed_args = parser.parse_args()

    ##################
    # Main entry-point
    ##################
    main(parsed_args)

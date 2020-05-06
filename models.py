import tensorflow as tf
import numpy as np
from abc import ABC


class BaseGCN(ABC):
    """
    Base class for graph convolutional neural networks.
    """
    def __init__(self):
        pass

    def frobenius_norm(self, tensor):
        square_tensor = tf.square(tensor)
        tensor_sum = tf.reduce_sum(square_tensor)
        frobenius_norm = tf.sqrt(tensor_sum)
        return frobenius_norm

    def mono_conv(self, list_lap, ord_conv, A, W, b):
        """ Calculate new node features using Chebyshev-polynomial
        approximation.  """
        feat = []
        # collect features
        for k in range(ord_conv):
            c_lap = list_lap[k]

            # dense implementation
            c_feat = tf.matmul(c_lap, A, a_is_sparse=False)
            feat.append(c_feat)

        all_feat = tf.concat(feat, 1)
        conv_feat = tf.matmul(all_feat, W) + b
        conv_feat = tf.nn.relu(conv_feat)
        return conv_feat

    def compute_cheb_polynomials(self, L, ord_cheb, list_cheb):
        """ Calculate chebyshev-polynomials. """
        for k in range(ord_cheb):
            if (k == 0):
                list_cheb.append(
                    tf.cast(tf.diag(tf.ones([tf.shape(L)[0], ])), 'float32'))
            elif (k == 1):
                list_cheb.append(tf.cast(L, 'float32'))
            else:
                list_cheb.append(
                    2 * tf.matmul(L, list_cheb[k - 1]) - list_cheb[k - 2])


class MultiGMCNonAR(BaseGCN):
    """
    Geometric matrix completion model using graph convolutional
    networks with recurrent neural networks given multiple graphs.
    """
    def rnn(self, Lr, num):
        # Loading of the Laplacians
        Lr = tf.cast(Lr, 'float32')
        labels_y = tf.cast(self.labels_y, tf.float32)

        # Compute all chebyshev polynomials a priori
        norm_Lr = Lr - tf.diag(tf.ones([Lr.shape[0], ]))
        list_row_cheb_pol = list()
        self.compute_cheb_polynomials(norm_Lr, self.ord_row,
                                      list_row_cheb_pol)

        # Definition of constant matrices
        M = tf.constant(self.M, dtype=tf.float32)
        Otraining = tf.constant(self.Otraining,
                                     dtype=tf.float32)  # training mask
        Otest = tf.constant(self.Otest, dtype=tf.float32)  # test mask
        OVal = tf.constant(self.Oval, dtype=tf.float32)  # validation

        # Definition of the weights for extracting the global features
        W_conv_W = tf.get_variable("W_conv_W" + str(num), shape=[
            self.ord_row * self.initial_W.shape[1], self.n_conv_feat],
                                        initializer=tf.contrib.layers.xavier_initializer())
        b_conv_W = tf.Variable(tf.zeros([self.n_conv_feat, ]))

        # Recurrent N parameters
        if self.residual:
            W_f_u = tf.get_variable("W_f_u" + str(num), shape=[
                self.n_conv_feat + self.initial_W.shape[1], self.n_conv_feat],
                                    initializer=tf.contrib.layers.xavier_initializer())
            W_i_u = tf.get_variable("W_i_u" + str(num), shape=[
                self.n_conv_feat + self.initial_W.shape[1], self.n_conv_feat],
                                    initializer=tf.contrib.layers.xavier_initializer())
            W_o_u = tf.get_variable("W_o_u" + str(num), shape=[
                self.n_conv_feat + self.initial_W.shape[1], self.n_conv_feat],
                                    initializer=tf.contrib.layers.xavier_initializer())
            W_c_u = tf.get_variable("W_c_u" + str(num), shape=[
                self.n_conv_feat + self.initial_W.shape[1], self.n_conv_feat],
                                    initializer=tf.contrib.layers.xavier_initializer())
        else:
            W_f_u = tf.get_variable("W_f_u" + str(num), shape=[self.n_conv_feat,
                                                         self.n_conv_feat],
                                         initializer=tf.contrib.layers.xavier_initializer())
            W_i_u = tf.get_variable("W_i_u" + str(num), shape=[self.n_conv_feat,
                                                         self.n_conv_feat],
                                         initializer=tf.contrib.layers.xavier_initializer())
            W_o_u = tf.get_variable("W_o_u" + str(num), shape=[self.n_conv_feat,
                                                         self.n_conv_feat],
                                         initializer=tf.contrib.layers.xavier_initializer())
            W_c_u = tf.get_variable("W_c_u" + str(num), shape=[self.n_conv_feat,
                                                         self.n_conv_feat],
                                         initializer=tf.contrib.layers.xavier_initializer())
        U_f_u = tf.get_variable("U_f_u" + str(num), shape=[self.n_conv_feat,
                                                     self.n_conv_feat],
                                     initializer=tf.contrib.layers.xavier_initializer())
        U_i_u = tf.get_variable("U_i_u" + str(num), shape=[self.n_conv_feat,
                                                     self.n_conv_feat],
                                     initializer=tf.contrib.layers.xavier_initializer())
        U_o_u = tf.get_variable("U_o_u" + str(num), shape=[self.n_conv_feat,
                                                     self.n_conv_feat],
                                     initializer=tf.contrib.layers.xavier_initializer())
        U_c_u = tf.get_variable("U_c_u" + str(num), shape=[self.n_conv_feat,
                                                     self.n_conv_feat],
                                     initializer=tf.contrib.layers.xavier_initializer())
        b_f_u = tf.Variable(tf.zeros([self.n_conv_feat, ]))
        b_i_u = tf.Variable(tf.zeros([self.n_conv_feat, ]))
        b_o_u = tf.Variable(tf.zeros([self.n_conv_feat, ]))
        b_c_u = tf.Variable(tf.zeros([self.n_conv_feat, ]))

        # Output parameters
        W_out_W = tf.get_variable("W_out_W" + str(num),
                                       shape=[self.n_conv_feat,
                                              self.initial_W.shape[1]],
                                  initializer=tf.contrib.layers.xavier_initializer())
        b_out_W = tf.Variable(tf.zeros([self.initial_W.shape[1],]))

        # definition of W and H
        W = tf.constant(self.initial_W.astype('float32'))
        if self.separable_gmc:
            H = tf.Variable(self.initial_H.astype('float32'))
            X = tf.matmul(W, H, transpose_b=True)

        # list_X = list()
        # list_X.append(tf.identity(self.X))

        h_u = tf.zeros([M.shape[0], self.n_conv_feat])
        c_u = tf.zeros([M.shape[0], self.n_conv_feat])

        # RNN model
        for k in range(self.num_iterations):
            # Extraction of global features vectors
            final_feat_users = self.mono_conv(list_row_cheb_pol,
                                              self.ord_row, W, W_conv_W,
                                              b_conv_W)
            if self.residual:
                final_feat_users = tf.concat([final_feat_users, W], 1)
            # RNN
            f_u = tf.sigmoid(tf.matmul(final_feat_users,W_f_u) + tf.matmul(
                h_u, U_f_u) + b_f_u)
            i_u = tf.sigmoid(tf.matmul(final_feat_users, W_i_u) + tf.matmul(
                h_u, U_i_u) + b_i_u)
            o_u = tf.sigmoid(tf.matmul(final_feat_users, W_o_u) + tf.matmul(
                h_u, U_o_u) + b_o_u)
            update_c_u = tf.sigmoid(tf.matmul(final_feat_users,W_c_u) +
                                    tf.matmul(h_u, U_c_u) + b_c_u)
            c_u = tf.multiply(f_u, c_u) + tf.multiply(i_u, update_c_u)
            h_u = tf.multiply(o_u, tf.sigmoid(c_u))

            # Compute update of matrix X
            delta_W = tf.tanh(tf.matmul(c_u, W_out_W)
                                   + b_out_W)
            if self.separable_gmc:
                X = tf.matmul(W + delta_W, H, transpose_b=True)
            else:
                X = W + delta_W
            # list_X.append(tf.identity(tf.reshape(X, [tf.shape(M)[0],
            #                                          tf.shape(M)[1]])))

        if self.separable_gmc:
            return X, W, H
        else:
            return X


    def __init__(self, M, Lr, Otraining, Otest, Oval, initial_W, initial_H,
                 labels_y, train_mask, test_mask, val_mask, impute_idx,
                 order_chebyshev_row=5, num_iterations=10, gamma=1.0,
                 gamma_H=None, gamma_W=None, gamma_tr=1.0, gamma_bce=1.0,
                 learning_rate=1e-4, n_conv_feat=32, M_rank=None,
                 idx_gpu='/gpu:0', separable_gmc=False, residual=False):
        super().__init__()
        # initial_W is the same with M if feature matrix not decomposed via SVD
        self.initial_W = initial_W  # input feature matrix
        self.initial_H = initial_H  #
        self.M = M  # input feature matrix
        self.Lr = Lr # Row graph laplacian matrix
        self.Otraining = Otraining # Training set of observed entries mask
        self.Otest = Otest # Test set mask
        self.Oval = Oval # Validation set mask
        self.labels_y = labels_y
        self.train_mask = train_mask # Target class label column train mask
        self.test_mask = test_mask
        self.val_mask = val_mask
        self.impute_idx = impute_idx # Indeces of impu
        self.ord_row = order_chebyshev_row
        self.num_iterations = num_iterations
        self.n_conv_feat = n_conv_feat
        self.separable_gmc = separable_gmc
        self.residual = residual

        with tf.Graph().as_default() as g:
            tf.logging.set_verbosity(tf.logging.ERROR)
            self.graph = g
            tf.set_random_seed(0)
            with tf.device(idx_gpu):

                # For every graph laplacian perform operation
                output_list = []
                for k, cur_lr in enumerate(Lr):
                    cur_res = self.rnn(cur_lr, k)
                    output_list.append(cur_res)

                if self.separable_gmc:
                    cur_X = tf.stack([x[0] for x in output_list], -1)
                    cur_X = tf.reduce_mean(cur_X, -1)
                else:
                    cur_X = tf.stack(output_list, -1)
                    cur_X = tf.reduce_mean(cur_X, -1)

                # Seperate features X and labels Y
                if len(self.labels_y.shape) > 1 and self.labels_y.shape[-1] \
                        > 1:
                    self.num_class = self.labels_y.shape[-1]
                else:
                    self.num_class = 1
                self.Y = cur_X[:, -self.num_class:]
                self.Y = tf.squeeze(self.Y)
                self.X = cur_X[:, :-self.num_class]

                self.labels_y = tf.cast(labels_y, tf.float32)
                self.M = tf.constant(self.M, dtype=tf.float32)

                self.Otraining = tf.constant(Otraining,
                                             dtype=tf.float32)  # training mask
                self.Otest = tf.constant(Otest, dtype=tf.float32)  # test mask
                self.OVal = tf.constant(Oval, dtype=tf.float32)  # validation

                # Classification loss
                if self.num_class == 1:
                    self.tr_loss_bce = tf.nn.sigmoid_cross_entropy_with_logits(
                        labels=tf.boolean_mask(self.labels_y, train_mask),
                        logits=tf.boolean_mask(self.Y, train_mask))
                    self.ts_loss_bce = tf.nn.sigmoid_cross_entropy_with_logits(
                        labels=tf.boolean_mask(self.labels_y, test_mask),
                        logits=tf.boolean_mask(self.Y, test_mask))
                    self.val_loss_bce = tf.nn.sigmoid_cross_entropy_with_logits(
                        labels=tf.boolean_mask(self.labels_y, val_mask),
                        logits=tf.boolean_mask(self.Y, val_mask))
                else:
                    self.tr_loss_bce = tf.nn.softmax_cross_entropy_with_logits(
                        labels=tf.boolean_mask(self.labels_y, train_mask),
                        logits=tf.boolean_mask(self.Y, train_mask))
                    self.ts_loss_bce = tf.nn.softmax_cross_entropy_with_logits(
                        labels=tf.boolean_mask(self.labels_y, test_mask),
                        logits=tf.boolean_mask(self.Y, test_mask))
                    self.val_loss_bce = tf.nn.softmax_cross_entropy_with_logits(
                        labels=tf.boolean_mask(self.labels_y, val_mask),
                        logits=tf.boolean_mask(self.Y, val_mask))

                self.tr_loss_bce = tf.reduce_mean(self.tr_loss_bce)
                self.ts_loss_bce = tf.reduce_mean(self.ts_loss_bce)
                self.val_loss_bce = tf.reduce_mean(self.val_loss_bce)

                # Validation and test set predicted probabilities
                self.val_Y_out = tf.boolean_mask(self.Y, val_mask)

                # Test set output and sigmoid output
                self.ts_y_output = tf.boolean_mask(self.Y, test_mask)
                if self.num_class == 1:
                    self.sigmoid_ts_Y_out = tf.nn.sigmoid(self.ts_y_output)
                else:
                    self.sigmoid_ts_Y_out = tf.nn.softmax(self.ts_y_output)

                ###
                # # Training set imputation loss terms with set cardinality
                ###
                frob_tensor = tf.multiply(self.Otraining[:, :-self.num_class],
                                          self.X - self.M[:, :-self.num_class])
                self.loss_frob = tf.square(self.frobenius_norm(frob_tensor))/(
                tf.reduce_sum(self.Otraining[:, :-self.num_class]))

                cur_loss_trace_row = tf.cast(0.0, 'float32')
                for cur_L in self.Lr:
                    cur_L = tf.cast(cur_L, 'float32')
                    trace_row_tensor = tf.matmul(
                        tf.matmul(self.X, cur_L, transpose_a=True), self.X)
                    cur_loss_trace_row += tf.trace(trace_row_tensor) / tf.cast(
                        tf.shape(self.X)[0] * tf.shape(self.X)[1], 'float32')
                self.loss_trace_row = cur_loss_trace_row

                # Training loss definition
                if self.separable_gmc:
                    self.frob_norm_W = tf.cast(0.0, 'float32')
                    self.frob_norm_H = tf.cast(0.0, 'float32')
                    for k,v in enumerate(output_list):
                        cur_w_matrix =  v[1]
                        cur_h_matrix = v[2]
                        self.frob_norm_W += tf.square(
                            self.frobenius_norm(cur_w_matrix)) / tf.cast(
                            tf.shape(cur_w_matrix)[0] * tf.shape(
                                cur_w_matrix)[1], 'float32')
                        self.frob_norm_H += tf.square(
                            self.frobenius_norm(cur_h_matrix)) / tf.cast(
                            tf.shape(cur_h_matrix)[0] * tf.shape(
                                cur_h_matrix)[1], 'float32')
                    self.loss = (gamma / 2) * self.loss_frob \
                                + (gamma_tr / 2) * self.loss_trace_row \
                                + (gamma_W / 2) * self.frob_norm_W \
                                + (gamma_H / 2) * self.frob_norm_H
                else:
                    self.loss = (gamma / 2) * self.loss_frob \
                                + (gamma_tr / 2) * self.loss_trace_row
                self.loss = self.loss + (gamma_bce * self.tr_loss_bce)


                # RMSE of train set rows at imputed entries without RMSE loss
                #  during training
                if impute_idx != None:
                    mat_column_len = self.Otraining[:, :-self.num_class].shape.as_list()[-1]
                    train_feature_mask = np.tile(train_mask, (mat_column_len, 1)).T
                    # train_feature_mask = tf.convert_to_tensor(train_feature_mask,
                    #                                           dtype=tf.float32)
                    train_impute_masking = np.zeros_like(train_feature_mask)
                    train_impute_masking[impute_idx] = 1
                    train_impute_masking = tf.convert_to_tensor(
                        train_impute_masking,
                        dtype=tf.float32)
                    train_feature_mask = tf.multiply(train_impute_masking,
                                                     train_feature_mask)
                    imputed_train_error = tf.multiply(train_feature_mask,
                                                      self.X - M[:, :-self.num_class])
                    self.train_RMSE = tf.sqrt(tf.reduce_mean(tf.square(
                        imputed_train_error)))

                    # RMSE of test set rows at imputed entries without RMSE loss
                    # during training
                    test_feature_mask = np.tile(test_mask, (mat_column_len, 1)).T
                    # test_feature_mask = tf.convert_to_tensor(test_feature_mask,
                    #                                     dtype=tf.float32)
                    test_impute_masking = np.zeros_like(test_feature_mask)
                    test_impute_masking[impute_idx] = 1
                    test_impute_masking = tf.convert_to_tensor(test_impute_masking,
                                                               dtype=tf.float32)
                    test_feature_mask = tf.multiply(test_impute_masking,
                                                    test_feature_mask)
                    imputed_test_error = tf.multiply(test_feature_mask,
                                                     self.X - M[:, :-self.num_class])
                    self.test_RMSE = tf.sqrt(tf.reduce_mean(tf.square(
                        imputed_test_error)))

                    # RMSE of val set rows at imputed entries without RMSE loss
                    # during training
                    val_feature_mask = np.tile(val_mask, (mat_column_len, 1)).T
                    # val_feature_mask = tf.convert_to_tensor(val_feature_mask,
                    #                                         dtype=tf.float32)
                    val_impute_masking = np.zeros_like(val_feature_mask)
                    val_impute_masking[impute_idx] = 1
                    val_impute_masking = tf.convert_to_tensor(
                        val_impute_masking,
                        dtype=tf.float32)
                    val_feature_mask = tf.multiply(val_impute_masking,
                                                     val_feature_mask)
                    imputed_val_error = tf.multiply(val_feature_mask,
                                                      self.X - M[:, :-self.num_class])
                    self.val_RMSE = tf.sqrt(tf.reduce_mean(tf.square(
                        imputed_val_error)))

                else:
                    self.train_RMSE = tf.convert_to_tensor(np.nan)
                    self.val_RMSE = tf.convert_to_tensor(np.nan)
                    self.test_RMSE = tf.convert_to_tensor(np.nan)

                self.predictions_error = self.ts_loss_bce

                # definition of the solver
                self.optimizer = tf.train.AdamOptimizer(
                    learning_rate=learning_rate).minimize(self.loss)

                self.var_grad = tf.gradients(self.loss,
                                             tf.trainable_variables())
                self.norm_grad = self.frobenius_norm(
                    tf.concat([tf.reshape(g, [-1]) for g in self.var_grad], 0))

                # Create a session for running Ops on the Graph.
                config = tf.ConfigProto(allow_soft_placement=True)
                config.gpu_options.allow_growth = True
                self.session = tf.Session(config=config)

                # Run the Op to initialize the variables.
                init = tf.initialize_all_variables()
                self.session.run(init)

                # Save checkpoints
                self.ckpt_saver = tf.train.Saver(max_to_keep=5)


class MultiGMCAR(BaseGCN):
    """
    Geometric matrix completion model using graph convolutional
    networks with recurrent neural networks given multiple graphs.
    """
    def rnn(self, Lr, num):
        # Loading of the Laplacians
        Lr = tf.cast(Lr, 'float32')
        labels_y = tf.cast(self.labels_y, tf.float32)

        # Compute all chebyshev polynomials a priori
        norm_Lr = Lr - tf.diag(tf.ones([Lr.shape[0], ]))
        list_row_cheb_pol = list()
        self.compute_cheb_polynomials(norm_Lr, self.ord_row,
                                      list_row_cheb_pol)

        # Definition of constant matrices
        M = tf.constant(self.M, dtype=tf.float32)
        Otraining = tf.constant(self.Otraining,
                                     dtype=tf.float32)  # training mask
        Otest = tf.constant(self.Otest, dtype=tf.float32)  # test mask
        OVal = tf.constant(self.Oval, dtype=tf.float32)  # validation

        # Definition of the weights for extracting the global features
        W_conv_W = tf.get_variable("W_conv_W" + str(num), shape=[
            self.ord_row * self.initial_W.shape[1], self.n_conv_feat],
                                        initializer=tf.contrib.layers.xavier_initializer())
        b_conv_W = tf.Variable(tf.zeros([self.n_conv_feat, ]))

        # Recurrent N parameters
        if self.residual:
            W_f_u = tf.get_variable("W_f_u" + str(num), shape=[
                self.n_conv_feat + self.initial_W.shape[1], self.n_conv_feat],
                                    initializer=tf.contrib.layers.xavier_initializer())
            W_i_u = tf.get_variable("W_i_u" + str(num), shape=[
                self.n_conv_feat + self.initial_W.shape[1], self.n_conv_feat],
                                    initializer=tf.contrib.layers.xavier_initializer())
            W_o_u = tf.get_variable("W_o_u" + str(num), shape=[
                self.n_conv_feat + self.initial_W.shape[1], self.n_conv_feat],
                                    initializer=tf.contrib.layers.xavier_initializer())
            W_c_u = tf.get_variable("W_c_u" + str(num), shape=[
                self.n_conv_feat + self.initial_W.shape[1], self.n_conv_feat],
                                    initializer=tf.contrib.layers.xavier_initializer())
        else:
            W_f_u = tf.get_variable("W_f_u" + str(num), shape=[self.n_conv_feat,
                                                         self.n_conv_feat],
                                         initializer=tf.contrib.layers.xavier_initializer())
            W_i_u = tf.get_variable("W_i_u" + str(num), shape=[self.n_conv_feat,
                                                         self.n_conv_feat],
                                         initializer=tf.contrib.layers.xavier_initializer())
            W_o_u = tf.get_variable("W_o_u" + str(num), shape=[self.n_conv_feat,
                                                         self.n_conv_feat],
                                         initializer=tf.contrib.layers.xavier_initializer())
            W_c_u = tf.get_variable("W_c_u" + str(num), shape=[self.n_conv_feat,
                                                         self.n_conv_feat],
                                         initializer=tf.contrib.layers.xavier_initializer())
        U_f_u = tf.get_variable("U_f_u" + str(num), shape=[self.n_conv_feat,
                                                     self.n_conv_feat],
                                     initializer=tf.contrib.layers.xavier_initializer())
        U_i_u = tf.get_variable("U_i_u" + str(num), shape=[self.n_conv_feat,
                                                     self.n_conv_feat],
                                     initializer=tf.contrib.layers.xavier_initializer())
        U_o_u = tf.get_variable("U_o_u" + str(num), shape=[self.n_conv_feat,
                                                     self.n_conv_feat],
                                     initializer=tf.contrib.layers.xavier_initializer())
        U_c_u = tf.get_variable("U_c_u" + str(num), shape=[self.n_conv_feat,
                                                     self.n_conv_feat],
                                     initializer=tf.contrib.layers.xavier_initializer())
        b_f_u = tf.Variable(tf.zeros([self.n_conv_feat, ]))
        b_i_u = tf.Variable(tf.zeros([self.n_conv_feat, ]))
        b_o_u = tf.Variable(tf.zeros([self.n_conv_feat, ]))
        b_c_u = tf.Variable(tf.zeros([self.n_conv_feat, ]))

        # Output parameters
        W_out_W = tf.get_variable("W_out_W" + str(num),
                                       shape=[self.n_conv_feat,
                                              self.initial_W.shape[1]],
                                  initializer=tf.contrib.layers.xavier_initializer())
        b_out_W = tf.Variable(tf.zeros([self.initial_W.shape[1],]))

        # definition of W and H
        W = tf.constant(self.initial_W.astype('float32'))
        if self.initial_H is not None:
            H = tf.Variable(self.initial_H.astype('float32'))
            X = tf.matmul(W, H, transpose_b=True)

        # list_X = list()
        # list_X.append(tf.identity(self.X))

        h_u = tf.zeros([M.shape[0], self.n_conv_feat])
        c_u = tf.zeros([M.shape[0], self.n_conv_feat])

        # RNN model
        for k in range(self.num_iterations):
            # Extraction of global features vectors
            final_feat_users = self.mono_conv(list_row_cheb_pol,
                                              self.ord_row, W, W_conv_W,
                                              b_conv_W)
            if self.residual:
                final_feat_users = tf.concat([final_feat_users, W], 1)
            # RNN
            f_u = tf.sigmoid(tf.matmul(final_feat_users,W_f_u) + tf.matmul(
                h_u, U_f_u) + b_f_u)
            i_u = tf.sigmoid(tf.matmul(final_feat_users, W_i_u) + tf.matmul(
                h_u, U_i_u) + b_i_u)
            o_u = tf.sigmoid(tf.matmul(final_feat_users, W_o_u) + tf.matmul(
                h_u, U_o_u) + b_o_u)
            update_c_u = tf.sigmoid(tf.matmul(final_feat_users,W_c_u) +
                                    tf.matmul(h_u, U_c_u) + b_c_u)
            c_u = tf.multiply(f_u, c_u) + tf.multiply(i_u, update_c_u)
            h_u = tf.multiply(o_u, tf.sigmoid(c_u))

            # Compute update of matrix X
            delta_W = tf.tanh(tf.matmul(c_u, W_out_W)
                                   + b_out_W)

            if self.initial_H is not None:
                W += delta_W
                X = tf.matmul(W, H, transpose_b=True)
            else:
                W += delta_W
                X = W
            # list_X.append(tf.identity(tf.reshape(X, [tf.shape(M)[0],
            #                                          tf.shape(M)[1]])))

        if self.separable_gmc:
            return X, W, H
        else:
            return X

    def __init__(self, M, Lr, Otraining, Otest, Oval, initial_W, initial_H,
                 labels_y, train_mask, test_mask, val_mask, impute_idx,
                 order_chebyshev_row=5, num_iterations=10, gamma=1.0,
                 gamma_H=None, gamma_W=None, gamma_tr=1.0, gamma_bce=1.0,
                 learning_rate=1e-4, n_conv_feat=32, M_rank=None,
                 idx_gpu='/gpu:0', separable_gmc=False, residual=False):
        super().__init__()
        # initial_W is the same with M if feature matrix not decomposed via SVD
        self.initial_W = initial_W  # input feature matrix
        self.initial_H = initial_H  #
        self.M = M  # input feature matrix
        self.Lr = Lr # Row graph laplacian matrix
        self.Otraining = Otraining # Training set of observed entries mask
        self.Otest = Otest # Test set mask
        self.Oval = Oval # Validation set mask
        self.labels_y = labels_y
        self.train_mask = train_mask # Target class label column train mask
        self.test_mask = test_mask
        self.val_mask = val_mask
        self.impute_idx = impute_idx # Indeces of impu
        self.ord_row = order_chebyshev_row
        self.num_iterations = num_iterations
        self.n_conv_feat = n_conv_feat
        self.separable_gmc = separable_gmc
        self.residual = residual

        with tf.Graph().as_default() as g:
            tf.logging.set_verbosity(tf.logging.ERROR)
            self.graph = g
            tf.set_random_seed(0)
            with tf.device(idx_gpu):

                # For every graph laplacian perform operation
                output_list = []
                for k, cur_lr in enumerate(Lr):
                    cur_res = self.rnn(cur_lr, k)
                    output_list.append(cur_res)

                if self.separable_gmc:
                    cur_X = tf.stack([x[0] for x in output_list], -1)
                    cur_X = tf.reduce_mean(cur_X, -1)
                else:
                    cur_X = tf.stack(output_list, -1)
                    cur_X = tf.reduce_mean(cur_X, -1)

                # Seperate features X and labels Y
                if len(self.labels_y.shape) > 1 and self.labels_y.shape[-1] \
                        > 1:
                    self.num_class = self.labels_y.shape[-1]
                else:
                    self.num_class = 1
                self.Y = cur_X[:, -self.num_class:]
                self.Y = tf.squeeze(self.Y)
                self.X = cur_X[:, :-self.num_class]

                self.labels_y = tf.cast(labels_y, tf.float32)
                self.M = tf.constant(self.M, dtype=tf.float32)

                self.Otraining = tf.constant(Otraining,
                                             dtype=tf.float32)  # training mask
                self.Otest = tf.constant(Otest, dtype=tf.float32)  # test mask
                self.OVal = tf.constant(Oval, dtype=tf.float32)  # validation

                # Classification loss
                if self.num_class == 1:
                    self.tr_loss_bce = tf.nn.sigmoid_cross_entropy_with_logits(
                        labels=tf.boolean_mask(self.labels_y, train_mask),
                        logits=tf.boolean_mask(self.Y, train_mask))
                    self.ts_loss_bce = tf.nn.sigmoid_cross_entropy_with_logits(
                        labels=tf.boolean_mask(self.labels_y, test_mask),
                        logits=tf.boolean_mask(self.Y, test_mask))
                    self.val_loss_bce = tf.nn.sigmoid_cross_entropy_with_logits(
                        labels=tf.boolean_mask(self.labels_y, val_mask),
                        logits=tf.boolean_mask(self.Y, val_mask))
                else:
                    self.tr_loss_bce = tf.nn.softmax_cross_entropy_with_logits(
                        labels=tf.boolean_mask(self.labels_y, train_mask),
                        logits=tf.boolean_mask(self.Y, train_mask))
                    self.ts_loss_bce = tf.nn.softmax_cross_entropy_with_logits(
                        labels=tf.boolean_mask(self.labels_y, test_mask),
                        logits=tf.boolean_mask(self.Y, test_mask))
                    self.val_loss_bce = tf.nn.softmax_cross_entropy_with_logits(
                        labels=tf.boolean_mask(self.labels_y, val_mask),
                        logits=tf.boolean_mask(self.Y, val_mask))

                self.tr_loss_bce = tf.reduce_mean(self.tr_loss_bce)
                self.ts_loss_bce = tf.reduce_mean(self.ts_loss_bce)
                self.val_loss_bce = tf.reduce_mean(self.val_loss_bce)

                # self.tr_loss_bce = tf.reduce_sum(self.tr_loss_bce) / (
                #     tf.reduce_sum(self.Otraining[:, -self.num_class:]))
                # self.ts_loss_bce = tf.reduce_sum(self.ts_loss_bce) / (
                #     tf.reduce_sum(self.Otest))
                # self.val_loss_bce = tf.reduce_sum(
                #     self.val_loss_bce) / tf.reduce_sum(self.OVal)


                # Validation and test set predicted probabilities
                self.val_Y_out = tf.boolean_mask(self.Y, val_mask)

                # Test set output and sigmoid output
                self.ts_y_output = tf.boolean_mask(self.Y, test_mask)
                if self.num_class == 1:
                    self.sigmoid_ts_Y_out = tf.nn.sigmoid(self.ts_y_output)
                else:
                    self.sigmoid_ts_Y_out = tf.nn.softmax(self.ts_y_output)

                ###
                # # Training set imputation loss terms with set cardinality
                ###
                frob_tensor = tf.multiply(self.Otraining[:, :-self.num_class],
                                          self.X - self.M[:, :-self.num_class])
                self.loss_frob = tf.square(self.frobenius_norm(frob_tensor))/(
                tf.reduce_sum(self.Otraining[:, :-self.num_class]))

                cur_loss_trace_row = 0
                for cur_L in self.Lr:
                    cur_L = tf.cast(cur_L, 'float32')
                    trace_row_tensor = tf.matmul(
                        tf.matmul(self.X, cur_L, transpose_a=True), self.X)
                    cur_loss_trace_row += tf.trace(trace_row_tensor) / tf.cast(
                        tf.shape(self.X)[0] * tf.shape(self.X)[1], 'float32')
                self.loss_trace_row = cur_loss_trace_row

                if self.separable_gmc:
                    self.frob_norm_W = tf.cast(0.0, 'float32')
                    self.frob_norm_H = tf.cast(0.0, 'float32')
                    for k, v in enumerate(output_list):
                        cur_w_matrix = v[1]
                        cur_h_matrix = v[2]
                        self.frob_norm_W += tf.square(
                            self.frobenius_norm(cur_w_matrix)) / tf.cast(
                            tf.shape(cur_w_matrix)[0] * tf.shape(
                                cur_w_matrix)[1], 'float32')
                        self.frob_norm_H += tf.square(
                            self.frobenius_norm(cur_h_matrix)) / tf.cast(
                            tf.shape(cur_h_matrix)[0] * tf.shape(
                                cur_h_matrix)[1], 'float32')
                    self.loss = (gamma / 2) * self.loss_frob \
                                + (gamma_tr / 2) * self.loss_trace_row \
                                + (gamma_W / 2) * self.frob_norm_W \
                                + (gamma_H / 2) * self.frob_norm_H
                else:
                    self.loss = (gamma / 2) * self.loss_frob \
                                + (gamma_tr / 2) * self.loss_trace_row

                self.loss = self.loss + (gamma_bce * self.tr_loss_bce)


                # RMSE of train set rows at imputed entries without RMSE loss
                #  during training
                if impute_idx != None:
                    mat_column_len = self.Otraining[:, :-self.num_class].shape.as_list()[-1]
                    train_feature_mask = np.tile(train_mask, (mat_column_len, 1)).T
                    # train_feature_mask = tf.convert_to_tensor(train_feature_mask,
                    #                                           dtype=tf.float32)
                    train_impute_masking = np.zeros_like(train_feature_mask)
                    train_impute_masking[impute_idx] = 1
                    train_impute_masking = tf.convert_to_tensor(
                        train_impute_masking,
                        dtype=tf.float32)
                    train_feature_mask = tf.multiply(train_impute_masking,
                                                     train_feature_mask)
                    imputed_train_error = tf.multiply(train_feature_mask,
                                                      self.X - M[:, :-self.num_class])
                    self.train_RMSE = tf.sqrt(tf.reduce_mean(tf.square(
                        imputed_train_error)))

                    # RMSE of test set rows at imputed entries without RMSE loss
                    # during training
                    test_feature_mask = np.tile(test_mask, (mat_column_len, 1)).T
                    # test_feature_mask = tf.convert_to_tensor(test_feature_mask,
                    #                                     dtype=tf.float32)
                    test_impute_masking = np.zeros_like(test_feature_mask)
                    test_impute_masking[impute_idx] = 1
                    test_impute_masking = tf.convert_to_tensor(test_impute_masking,
                                                               dtype=tf.float32)
                    test_feature_mask = tf.multiply(test_impute_masking,
                                                    test_feature_mask)
                    imputed_test_error = tf.multiply(test_feature_mask,
                                                     self.X - M[:, :-self.num_class])
                    self.test_RMSE = tf.sqrt(tf.reduce_mean(tf.square(
                        imputed_test_error)))

                    # RMSE of val set rows at imputed entries without RMSE loss
                    # during training
                    val_feature_mask = np.tile(val_mask, (mat_column_len, 1)).T
                    # val_feature_mask = tf.convert_to_tensor(val_feature_mask,
                    #                                         dtype=tf.float32)
                    val_impute_masking = np.zeros_like(val_feature_mask)
                    val_impute_masking[impute_idx] = 1
                    val_impute_masking = tf.convert_to_tensor(
                        val_impute_masking,
                        dtype=tf.float32)
                    val_feature_mask = tf.multiply(val_impute_masking,
                                                     val_feature_mask)
                    imputed_val_error = tf.multiply(val_feature_mask,
                                                      self.X - M[:, :-self.num_class])
                    self.val_RMSE = tf.sqrt(tf.reduce_mean(tf.square(
                        imputed_val_error)))

                else:
                    self.train_RMSE = tf.convert_to_tensor(np.nan)
                    self.val_RMSE = tf.convert_to_tensor(np.nan)
                    self.test_RMSE = tf.convert_to_tensor(np.nan)

                # test loss definition
                # self.predictions = tf.multiply(self.Otest[:,:-self.num_class], self.X - self.M[:,:-self.num_class])
                # self.predictions_error = self.frobenius_norm(self.predictions) + self.ts_loss_bce
                self.predictions_error = self.ts_loss_bce

                # definition of the solver
                self.optimizer = tf.train.AdamOptimizer(
                    learning_rate=learning_rate).minimize(self.loss)

                self.var_grad = tf.gradients(self.loss,
                                             tf.trainable_variables())
                self.norm_grad = self.frobenius_norm(
                    tf.concat([tf.reshape(g, [-1]) for g in self.var_grad], 0))

                # Create a session for running Ops on the Graph.
                config = tf.ConfigProto(allow_soft_placement=True)
                config.gpu_options.allow_growth = True
                self.session = tf.Session(config=config)

                # Run the Op to initialize the variables.
                init = tf.initialize_all_variables()
                self.session.run(init)

                # Save checkpoints
                self.ckpt_saver = tf.train.Saver(max_to_keep=5)


class MultiHeadAttention(tf.keras.layers.Layer):
    def __init__(self, d_model, num_heads):
        super(MultiHeadAttention, self).__init__()
        self.num_heads = num_heads
        self.d_model = d_model

        assert d_model % self.num_heads == 0

        self.depth = d_model // self.num_heads

        self.wq = tf.keras.layers.Dense(d_model)
        self.wk = tf.keras.layers.Dense(d_model)
        self.wv = tf.keras.layers.Dense(d_model)

        self.dense = tf.keras.layers.Dense(d_model)

    def scaled_dot_product_attention(self, q, k, v, mask):
        """Calculate the attention weights.
        q, k, v must have matching leading dimensions.
        k, v must have matching penultimate dimension, i.e.: seq_len_k = seq_len_v.
        The mask has different shapes depending on its type(padding or look ahead)
        but it must be broadcastable for addition.

        Args:
          q: query shape == (..., seq_len_q, depth)
          k: key shape == (..., seq_len_k, depth)
          v: value shape == (..., seq_len_v, depth_v)
          mask: Float tensor with shape broadcastable
                to (..., seq_len_q, seq_len_k). Defaults to None.

        Returns:
          output, attention_weights
        """

        matmul_qk = tf.matmul(q, k,
                              transpose_b=True)  # (..., seq_len_q, seq_len_k)

        # scale matmul_qk
        dk = tf.cast(tf.shape(k)[-1], tf.float32)
        scaled_attention_logits = matmul_qk / tf.math.sqrt(dk)

        # add the mask to the scaled tensor.
        if mask is not None:
            scaled_attention_logits += (mask * -1e9)

            # softmax is normalized on the last axis (seq_len_k) so that the scores
        # add up to 1.
        attention_weights = tf.nn.softmax(scaled_attention_logits,
                                          axis=-1)  # (..., seq_len_q, seq_len_k)

        output = tf.matmul(attention_weights, v)  # (..., seq_len_q, depth_v)

        return output, attention_weights


    def split_heads(self, x, batch_size):
        """Split the last dimension into (num_heads, depth).
        Transpose the result such that the shape is (batch_size, num_heads, seq_len, depth)
        """
        x = tf.reshape(x, (batch_size, -1, self.num_heads, self.depth))
        return tf.transpose(x, perm=[0, 2, 1, 3])

    def call(self, v, k, q, mask):
        batch_size = tf.shape(q)[0]

        q = self.wq(q)  # (batch_size, seq_len, d_model)
        k = self.wk(k)  # (batch_size, seq_len, d_model)
        v = self.wv(v)  # (batch_size, seq_len, d_model)

        q = self.split_heads(q,
                             batch_size)  # (batch_size, num_heads, seq_len_q, depth)
        k = self.split_heads(k,
                             batch_size)  # (batch_size, num_heads, seq_len_k, depth)
        v = self.split_heads(v,
                             batch_size)  # (batch_size, num_heads, seq_len_v, depth)

        # scaled_attention.shape == (batch_size, num_heads, seq_len_q, depth)
        # attention_weights.shape == (batch_size, num_heads, seq_len_q, seq_len_k)
        scaled_attention, attention_weights = \
            self.scaled_dot_product_attention(q, k, v, mask)

        scaled_attention = tf.transpose(scaled_attention, perm=[0, 2, 1,
                                                                3])  # (batch_size, seq_len_q, num_heads, depth)

        concat_attention = tf.reshape(scaled_attention,
                                      (batch_size, -1,
                                       self.d_model))  # (batch_size, seq_len_q, d_model)

        output = self.dense(
            concat_attention)  # (batch_size, seq_len_q, d_model)

        return output, attention_weights


class MultiGMCNonARWithSelfAtt(BaseGCN):
    """
    """
    def rnn(self, Lr, num):
        # Loading of the Laplacians
        Lr = tf.cast(Lr, 'float32')
        labels_y = tf.cast(self.labels_y, tf.float32)

        # Compute all chebyshev polynomials a priori
        norm_Lr = Lr - tf.diag(tf.ones([Lr.shape[0], ]))
        list_row_cheb_pol = list()
        self.compute_cheb_polynomials(norm_Lr, self.ord_row,
                                      list_row_cheb_pol)

        # Definition of constant matrices
        M = tf.constant(self.M, dtype=tf.float32)
        Otraining = tf.constant(self.Otraining,
                                     dtype=tf.float32)  # training mask
        Otest = tf.constant(self.Otest, dtype=tf.float32)  # test mask
        OVal = tf.constant(self.Oval, dtype=tf.float32)  # validation

        # Definition of the weights for extracting the global features
        W_conv_W = tf.get_variable("W_conv_W" + str(num), shape=[
            self.ord_row * self.initial_W.shape[1], self.n_conv_feat],
                                        initializer=tf.contrib.layers.xavier_initializer())
        b_conv_W = tf.Variable(tf.zeros([self.n_conv_feat, ]))

        # Recurrent N parameters
        if self.residual:
            W_f_u = tf.get_variable("W_f_u" + str(num), shape=[
                self.n_conv_feat + self.initial_W.shape[1], self.n_conv_feat],
                                    initializer=tf.contrib.layers.xavier_initializer())
            W_i_u = tf.get_variable("W_i_u" + str(num), shape=[
                self.n_conv_feat + self.initial_W.shape[1], self.n_conv_feat],
                                    initializer=tf.contrib.layers.xavier_initializer())
            W_o_u = tf.get_variable("W_o_u" + str(num), shape=[
                self.n_conv_feat + self.initial_W.shape[1], self.n_conv_feat],
                                    initializer=tf.contrib.layers.xavier_initializer())
            W_c_u = tf.get_variable("W_c_u" + str(num), shape=[
                self.n_conv_feat + self.initial_W.shape[1], self.n_conv_feat],
                                    initializer=tf.contrib.layers.xavier_initializer())
        else:
            W_f_u = tf.get_variable("W_f_u" + str(num), shape=[self.n_conv_feat,
                                                               self.n_conv_feat],
                                    initializer=tf.contrib.layers.xavier_initializer())
            W_i_u = tf.get_variable("W_i_u" + str(num), shape=[self.n_conv_feat,
                                                               self.n_conv_feat],
                                    initializer=tf.contrib.layers.xavier_initializer())
            W_o_u = tf.get_variable("W_o_u" + str(num), shape=[self.n_conv_feat,
                                                               self.n_conv_feat],
                                    initializer=tf.contrib.layers.xavier_initializer())
            W_c_u = tf.get_variable("W_c_u" + str(num), shape=[self.n_conv_feat,
                                                               self.n_conv_feat],
                                    initializer=tf.contrib.layers.xavier_initializer())
        U_f_u = tf.get_variable("U_f_u" + str(num), shape=[self.n_conv_feat,
                                                     self.n_conv_feat],
                                     initializer=tf.contrib.layers.xavier_initializer())
        U_i_u = tf.get_variable("U_i_u" + str(num), shape=[self.n_conv_feat,
                                                     self.n_conv_feat],
                                     initializer=tf.contrib.layers.xavier_initializer())
        U_o_u = tf.get_variable("U_o_u" + str(num), shape=[self.n_conv_feat,
                                                     self.n_conv_feat],
                                     initializer=tf.contrib.layers.xavier_initializer())
        U_c_u = tf.get_variable("U_c_u" + str(num), shape=[self.n_conv_feat,
                                                     self.n_conv_feat],
                                     initializer=tf.contrib.layers.xavier_initializer())
        b_f_u = tf.Variable(tf.zeros([self.n_conv_feat, ]))
        b_i_u = tf.Variable(tf.zeros([self.n_conv_feat, ]))
        b_o_u = tf.Variable(tf.zeros([self.n_conv_feat, ]))
        b_c_u = tf.Variable(tf.zeros([self.n_conv_feat, ]))

        # Output parameters
        W_out_W = tf.get_variable("W_out_W" + str(num),
                                       shape=[self.n_conv_feat,
                                              self.initial_W.shape[1]],
                                  initializer=tf.contrib.layers.xavier_initializer())
        b_out_W = tf.Variable(tf.zeros([self.initial_W.shape[1],]))

        # definition of W and H
        W = tf.constant(self.initial_W.astype('float32'))
        if self.initial_H is not None:
            H = tf.Variable(self.initial_H.astype('float32'))
            X = tf.matmul(W, H, transpose_b=True)

        # list_X = list()
        # list_X.append(tf.identity(self.X))

        h_u = tf.zeros([M.shape[0], self.n_conv_feat])
        c_u = tf.zeros([M.shape[0], self.n_conv_feat])

        # RNN model
        for k in range(self.num_iterations):
            # Extraction of global features vectors
            final_feat_users = self.mono_conv(list_row_cheb_pol,
                                              self.ord_row, W, W_conv_W,
                                              b_conv_W)

            # Concatenate output from GCN with input features
            if self.residual:
                final_feat_users = tf.concat([final_feat_users, W], 1)

            # RNN
            f_u = tf.sigmoid(tf.matmul(final_feat_users, W_f_u) + tf.matmul(
                h_u, U_f_u) + b_f_u)
            i_u = tf.sigmoid(tf.matmul(final_feat_users, W_i_u) + tf.matmul(
                h_u, U_i_u) + b_i_u)
            o_u = tf.sigmoid(tf.matmul(final_feat_users, W_o_u) + tf.matmul(
                h_u, U_o_u) + b_o_u)
            update_c_u = tf.sigmoid(tf.matmul(final_feat_users,W_c_u) +
                                    tf.matmul(h_u, U_c_u) + b_c_u)
            c_u = tf.multiply(f_u, c_u) + tf.multiply(i_u, update_c_u)
            h_u = tf.multiply(o_u, tf.sigmoid(c_u))

            # Compute update of matrix X
            delta_W = tf.tanh(tf.matmul(c_u, W_out_W)
                                   + b_out_W)
            if self.initial_H is not None:
                X = tf.matmul(W + delta_W, H, transpose_b=True)
            else:
                X = W + delta_W
            # list_X.append(tf.identity(tf.reshape(X, [tf.shape(M)[0],
            #                                          tf.shape(M)[1]])))
        return X


    def __init__(self, M, Lr, Otraining, Otest, Oval, initial_W, initial_H,
                 labels_y, train_mask, test_mask, val_mask, impute_idx,
                 order_chebyshev_row=5, num_iterations=10, gamma=1.0,
                 gamma_H=None, gamma_W=None, gamma_tr=1.0, gamma_bce=1.0,
                 learning_rate=1e-4, n_conv_feat=32, M_rank=None,
                 idx_gpu='/gpu:0', separable_gmc=False, residual=False):
        super().__init__()
        # initial_W is the same with M if feature matrix not decomposed via SVD
        self.initial_W = initial_W  # input feature matrix
        self.initial_H = initial_H  #
        self.M = M  # input feature matrix
        self.Lr = Lr # Row graph laplacian matrix
        self.Otraining = Otraining # Training set of observed entries mask
        self.Otest = Otest # Test set mask
        self.Oval = Oval # Validation set mask
        self.labels_y = labels_y
        self.train_mask = train_mask # Target class label column train mask
        self.test_mask = test_mask
        self.val_mask = val_mask
        self.impute_idx = impute_idx # Indeces of impu
        self.ord_row = order_chebyshev_row
        self.num_iterations = num_iterations
        self.n_conv_feat = n_conv_feat
        self.separable_gmc = separable_gmc
        self.residual = residual
        if self.separable_gmc:
            raise NotImplementedError('rnn method should return W and H. Then '
                                      'fix loss calculation.')

        with tf.Graph().as_default() as g:
            tf.logging.set_verbosity(tf.logging.ERROR)
            self.graph = g
            tf.set_random_seed(0)
            with tf.device(idx_gpu):

                # # Load list of graph laplacians
                # for i in list_lc:
                #     cur_lr = i
                #     # Perform the same operations
                #
                #     # We get n number of outputs
                #
                #     # We take the average of n number of outputs and return
                #     # single outputs

                # For every graph laplacian perform operation
                output_list = []
                for k, cur_lr in enumerate(Lr):
                    cur_res = self.rnn(cur_lr, k)
                    output_list.append(cur_res)
                cur_X = tf.stack(output_list, 1)

                # Single-head self-attention. Has many to many output and 
                # vectors averaged out.
                temp_mha = MultiHeadAttention(d_model=cur_X.shape[-1],
                                                   num_heads=1)
                out, attn = temp_mha(cur_X, k=cur_X, q=cur_X, mask=None)
                cur_X = tf.reduce_mean(out, 1)

                # Seperate features X and labels Y
                if len(self.labels_y.shape) > 1 and self.labels_y.shape[-1] \
                        > 1:
                    self.num_class = self.labels_y.shape[-1]
                else:
                    self.num_class = 1
                self.Y = cur_X[:, -self.num_class:]
                self.Y = tf.squeeze(self.Y)
                self.X = cur_X[:, :-self.num_class]

                self.labels_y = tf.cast(labels_y, tf.float32)
                self.M = tf.constant(self.M, dtype=tf.float32)

                self.Otraining = tf.constant(Otraining,
                                             dtype=tf.float32)  # training mask
                self.Otest = tf.constant(Otest, dtype=tf.float32)  # test mask
                self.OVal = tf.constant(Oval, dtype=tf.float32)  # validation

                # Classification loss
                if self.num_class == 1:
                    self.tr_loss_bce = tf.nn.sigmoid_cross_entropy_with_logits(
                        labels=tf.boolean_mask(self.labels_y, train_mask),
                        logits=tf.boolean_mask(self.Y, train_mask))
                    self.ts_loss_bce = tf.nn.sigmoid_cross_entropy_with_logits(
                        labels=tf.boolean_mask(self.labels_y, test_mask),
                        logits=tf.boolean_mask(self.Y, test_mask))
                    self.val_loss_bce = tf.nn.sigmoid_cross_entropy_with_logits(
                        labels=tf.boolean_mask(self.labels_y, val_mask),
                        logits=tf.boolean_mask(self.Y, val_mask))
                else:
                    self.tr_loss_bce = tf.nn.softmax_cross_entropy_with_logits(
                        labels=tf.boolean_mask(self.labels_y, train_mask),
                        logits=tf.boolean_mask(self.Y, train_mask))
                    self.ts_loss_bce = tf.nn.softmax_cross_entropy_with_logits(
                        labels=tf.boolean_mask(self.labels_y, test_mask),
                        logits=tf.boolean_mask(self.Y, test_mask))
                    self.val_loss_bce = tf.nn.softmax_cross_entropy_with_logits(
                        labels=tf.boolean_mask(self.labels_y, val_mask),
                        logits=tf.boolean_mask(self.Y, val_mask))

                self.tr_loss_bce = tf.reduce_mean(self.tr_loss_bce)
                self.ts_loss_bce = tf.reduce_mean(self.ts_loss_bce)
                self.val_loss_bce = tf.reduce_mean(self.val_loss_bce)

                # self.tr_loss_bce = tf.reduce_sum(self.tr_loss_bce) / (
                #     tf.reduce_sum(self.Otraining[:, -self.num_class:]))
                # self.ts_loss_bce = tf.reduce_sum(self.ts_loss_bce) / (
                #     tf.reduce_sum(self.Otest))
                # self.val_loss_bce = tf.reduce_sum(
                #     self.val_loss_bce) / tf.reduce_sum(self.OVal)


                # Validation and test set predicted probabilities
                self.val_Y_out = tf.boolean_mask(self.Y, val_mask)

                # Test set output and sigmoid output
                self.ts_y_output = tf.boolean_mask(self.Y, test_mask)
                if self.num_class == 1:
                    self.sigmoid_ts_Y_out = tf.nn.sigmoid(self.ts_y_output)
                else:
                    self.sigmoid_ts_Y_out = tf.nn.softmax(self.ts_y_output)

                ###
                # # Training set imputation loss terms with set cardinality
                ###
                frob_tensor = tf.multiply(self.Otraining[:, :-self.num_class],
                                          self.X - self.M[:, :-self.num_class])
                self.loss_frob = tf.square(self.frobenius_norm(frob_tensor))/(
                tf.reduce_sum(self.Otraining[:, :-self.num_class]))

                cur_loss_trace_row = 0
                for cur_L in self.Lr:
                    cur_L = tf.cast(cur_L, 'float32')
                    trace_row_tensor = tf.matmul(
                        tf.matmul(self.X, cur_L, transpose_a=True), self.X)
                    cur_loss_trace_row += tf.trace(trace_row_tensor) / tf.cast(
                        tf.shape(self.X)[0] * tf.shape(self.X)[1], 'float32')
                self.loss_trace_row = cur_loss_trace_row

                if self.separable_gmc:
                    self.frob_norm_H = tf.square(
                        self.frobenius_norm(self.H)) / tf.cast(
                        tf.shape(self.H)[0] * tf.shape(self.H)[1], 'float32')
                    self.frob_norm_W = tf.square(
                        self.frobenius_norm(self.W)) / tf.cast(
                        tf.shape(self.W)[0] * tf.shape(self.W)[1], 'float32')

                # Training loss definition
                if self.separable_gmc:
                    self.loss = (gamma / 2) * self.loss_frob \
                                + (gamma_tr / 2) * self.loss_trace_row \
                                + (gamma_W / 2) * self.frob_norm_W \
                                + (gamma_H / 2) * self.frob_norm_H
                else:
                    self.loss = (gamma / 2) * self.loss_frob \
                                + (gamma_tr / 2) * self.loss_trace_row
                self.loss = self.loss + (gamma_bce * self.tr_loss_bce)


                # RMSE of train set rows at imputed entries without RMSE loss
                #  during training
                if impute_idx != None:
                    mat_column_len = self.Otraining[:, :-self.num_class].shape.as_list()[-1]
                    train_feature_mask = np.tile(train_mask, (mat_column_len, 1)).T
                    # train_feature_mask = tf.convert_to_tensor(train_feature_mask,
                    #                                           dtype=tf.float32)
                    train_impute_masking = np.zeros_like(train_feature_mask)
                    train_impute_masking[impute_idx] = 1
                    train_impute_masking = tf.convert_to_tensor(
                        train_impute_masking,
                        dtype=tf.float32)
                    train_feature_mask = tf.multiply(train_impute_masking,
                                                     train_feature_mask)
                    imputed_train_error = tf.multiply(train_feature_mask,
                                                      self.X - M[:, :-self.num_class])
                    self.train_RMSE = tf.sqrt(tf.reduce_mean(tf.square(
                        imputed_train_error)))

                    # RMSE of test set rows at imputed entries without RMSE loss
                    # during training
                    test_feature_mask = np.tile(test_mask, (mat_column_len, 1)).T
                    # test_feature_mask = tf.convert_to_tensor(test_feature_mask,
                    #                                     dtype=tf.float32)
                    test_impute_masking = np.zeros_like(test_feature_mask)
                    test_impute_masking[impute_idx] = 1
                    test_impute_masking = tf.convert_to_tensor(test_impute_masking,
                                                               dtype=tf.float32)
                    test_feature_mask = tf.multiply(test_impute_masking,
                                                    test_feature_mask)
                    imputed_test_error = tf.multiply(test_feature_mask,
                                                     self.X - M[:, :-self.num_class])
                    self.test_RMSE = tf.sqrt(tf.reduce_mean(tf.square(
                        imputed_test_error)))

                    # Test feature mask for saving
                    self.test_feature_mask = test_feature_mask

                    # RMSE of val set rows at imputed entries without RMSE loss
                    # during training
                    val_feature_mask = np.tile(val_mask, (mat_column_len, 1)).T
                    # val_feature_mask = tf.convert_to_tensor(val_feature_mask,
                    #                                         dtype=tf.float32)
                    val_impute_masking = np.zeros_like(val_feature_mask)
                    val_impute_masking[impute_idx] = 1
                    val_impute_masking = tf.convert_to_tensor(
                        val_impute_masking,
                        dtype=tf.float32)
                    val_feature_mask = tf.multiply(val_impute_masking,
                                                     val_feature_mask)
                    imputed_val_error = tf.multiply(val_feature_mask,
                                                      self.X - M[:, :-self.num_class])
                    self.val_RMSE = tf.sqrt(tf.reduce_mean(tf.square(
                        imputed_val_error)))

                else:
                    self.train_RMSE = tf.convert_to_tensor(np.nan)
                    self.val_RMSE = tf.convert_to_tensor(np.nan)
                    self.test_RMSE = tf.convert_to_tensor(np.nan)

                # test loss definition
                # self.predictions = tf.multiply(self.Otest[:,:-self.num_class], self.X - self.M[:,:-self.num_class])
                # self.predictions_error = self.frobenius_norm(self.predictions) + self.ts_loss_bce
                self.predictions_error = self.ts_loss_bce

                # definition of the solver
                self.optimizer = tf.train.AdamOptimizer(
                    learning_rate=learning_rate).minimize(self.loss)

                self.var_grad = tf.gradients(self.loss,
                                             tf.trainable_variables())
                self.norm_grad = self.frobenius_norm(
                    tf.concat([tf.reshape(g, [-1]) for g in self.var_grad], 0))

                # Create a session for running Ops on the Graph.
                config = tf.ConfigProto(allow_soft_placement=True)
                config.gpu_options.allow_growth = True
                self.session = tf.Session(config=config)

                # Run the Op to initialize the variables.
                init = tf.initialize_all_variables()
                self.session.run(init)

                # Save checkpoints
                self.ckpt_saver = tf.train.Saver(max_to_keep=5)


class MultiGMCARWithSelfAtt(BaseGCN):
    """
    """
    def rnn(self, Lr, num):
        # Loading of the Laplacians
        Lr = tf.cast(Lr, 'float32')
        labels_y = tf.cast(self.labels_y, tf.float32)

        # Compute all chebyshev polynomials a priori
        norm_Lr = Lr - tf.diag(tf.ones([Lr.shape[0], ]))
        list_row_cheb_pol = list()
        self.compute_cheb_polynomials(norm_Lr, self.ord_row,
                                      list_row_cheb_pol)

        # Definition of constant matrices
        M = tf.constant(self.M, dtype=tf.float32)
        Otraining = tf.constant(self.Otraining,
                                     dtype=tf.float32)  # training mask
        Otest = tf.constant(self.Otest, dtype=tf.float32)  # test mask
        OVal = tf.constant(self.Oval, dtype=tf.float32)  # validation

        # Definition of the weights for extracting the global features
        W_conv_W = tf.get_variable("W_conv_W" + str(num), shape=[
            self.ord_row * self.initial_W.shape[1], self.n_conv_feat],
                                        initializer=tf.contrib.layers.xavier_initializer())
        b_conv_W = tf.Variable(tf.zeros([self.n_conv_feat, ]))

        # Recurrent N parameters
        if self.residual:
            W_f_u = tf.get_variable("W_f_u" + str(num), shape=[
                self.n_conv_feat + self.initial_W.shape[1], self.n_conv_feat],
                                    initializer=tf.contrib.layers.xavier_initializer())
            W_i_u = tf.get_variable("W_i_u" + str(num), shape=[
                self.n_conv_feat + self.initial_W.shape[1], self.n_conv_feat],
                                    initializer=tf.contrib.layers.xavier_initializer())
            W_o_u = tf.get_variable("W_o_u" + str(num), shape=[
                self.n_conv_feat + self.initial_W.shape[1], self.n_conv_feat],
                                    initializer=tf.contrib.layers.xavier_initializer())
            W_c_u = tf.get_variable("W_c_u" + str(num), shape=[
                self.n_conv_feat + self.initial_W.shape[1], self.n_conv_feat],
                                    initializer=tf.contrib.layers.xavier_initializer())
        else:
            W_f_u = tf.get_variable("W_f_u" + str(num), shape=[self.n_conv_feat,
                                                               self.n_conv_feat],
                                    initializer=tf.contrib.layers.xavier_initializer())
            W_i_u = tf.get_variable("W_i_u" + str(num), shape=[self.n_conv_feat,
                                                               self.n_conv_feat],
                                    initializer=tf.contrib.layers.xavier_initializer())
            W_o_u = tf.get_variable("W_o_u" + str(num), shape=[self.n_conv_feat,
                                                               self.n_conv_feat],
                                    initializer=tf.contrib.layers.xavier_initializer())
            W_c_u = tf.get_variable("W_c_u" + str(num), shape=[self.n_conv_feat,
                                                               self.n_conv_feat],
                                    initializer=tf.contrib.layers.xavier_initializer())
        U_f_u = tf.get_variable("U_f_u" + str(num), shape=[self.n_conv_feat,
                                                     self.n_conv_feat],
                                     initializer=tf.contrib.layers.xavier_initializer())
        U_i_u = tf.get_variable("U_i_u" + str(num), shape=[self.n_conv_feat,
                                                     self.n_conv_feat],
                                     initializer=tf.contrib.layers.xavier_initializer())
        U_o_u = tf.get_variable("U_o_u" + str(num), shape=[self.n_conv_feat,
                                                     self.n_conv_feat],
                                     initializer=tf.contrib.layers.xavier_initializer())
        U_c_u = tf.get_variable("U_c_u" + str(num), shape=[self.n_conv_feat,
                                                     self.n_conv_feat],
                                     initializer=tf.contrib.layers.xavier_initializer())
        b_f_u = tf.Variable(tf.zeros([self.n_conv_feat, ]))
        b_i_u = tf.Variable(tf.zeros([self.n_conv_feat, ]))
        b_o_u = tf.Variable(tf.zeros([self.n_conv_feat, ]))
        b_c_u = tf.Variable(tf.zeros([self.n_conv_feat, ]))

        # Output parameters
        W_out_W = tf.get_variable("W_out_W" + str(num),
                                       shape=[self.n_conv_feat,
                                              self.initial_W.shape[1]],
                                  initializer=tf.contrib.layers.xavier_initializer())
        b_out_W = tf.Variable(tf.zeros([self.initial_W.shape[1],]))

        # definition of W and H
        W = tf.constant(self.initial_W.astype('float32'))
        if self.initial_H is not None:
            H = tf.Variable(self.initial_H.astype('float32'))
            X = tf.matmul(W, H, transpose_b=True)

        # list_X = list()
        # list_X.append(tf.identity(self.X))

        h_u = tf.zeros([M.shape[0], self.n_conv_feat])
        c_u = tf.zeros([M.shape[0], self.n_conv_feat])

        # RNN model
        for k in range(self.num_iterations):
            # Extraction of global features vectors
            final_feat_users = self.mono_conv(list_row_cheb_pol,
                                              self.ord_row, W, W_conv_W,
                                              b_conv_W)
            if self.residual:
                final_feat_users = tf.concat([final_feat_users, W], 1)
            # RNN
            f_u = tf.sigmoid(tf.matmul(final_feat_users,W_f_u) + tf.matmul(
                h_u, U_f_u) + b_f_u)
            i_u = tf.sigmoid(tf.matmul(final_feat_users, W_i_u) + tf.matmul(
                h_u, U_i_u) + b_i_u)
            o_u = tf.sigmoid(tf.matmul(final_feat_users, W_o_u) + tf.matmul(
                h_u, U_o_u) + b_o_u)
            update_c_u = tf.sigmoid(tf.matmul(final_feat_users, W_c_u) +
                                    tf.matmul(h_u, U_c_u) + b_c_u)
            c_u = tf.multiply(f_u, c_u) + tf.multiply(i_u, update_c_u)
            h_u = tf.multiply(o_u, tf.sigmoid(c_u))

            # Compute update of matrix X
            delta_W = tf.tanh(tf.matmul(c_u, W_out_W)
                                   + b_out_W)
            if self.initial_H is not None:
                W += delta_W
                X = tf.matmul(W, H, transpose_b=True)
            else:
                W += delta_W
                X = W
            # list_X.append(tf.identity(tf.reshape(X, [tf.shape(M)[0],
            #                                          tf.shape(M)[1]])))
        return X


    def __init__(self, M, Lr, Otraining, Otest, Oval, initial_W, initial_H,
                 labels_y, train_mask, test_mask, val_mask, impute_idx,
                 order_chebyshev_row=5, num_iterations=10, gamma=1.0,
                 gamma_H=1.0, gamma_W=1.0, gamma_tr=1.0, gamma_bce=1.0,
                 learning_rate=1e-4, n_conv_feat=32, M_rank=10,
                 idx_gpu='/gpu:0', separable_gmc=False, residual=False):
        super().__init__()
        # initial_W is the same with M if feature matrix not decomposed via SVD
        self.initial_W = initial_W  # input feature matrix
        self.initial_H = initial_H  #
        self.M = M  # input feature matrix
        self.Lr = Lr # Row graph laplacian matrix
        self.Otraining = Otraining # Training set of observed entries mask
        self.Otest = Otest # Test set mask
        self.Oval = Oval # Validation set mask
        self.labels_y = labels_y
        self.train_mask = train_mask # Target class label column train mask
        self.test_mask = test_mask
        self.val_mask = val_mask
        self.impute_idx = impute_idx # Indeces of impu
        self.ord_row = order_chebyshev_row
        self.num_iterations = num_iterations
        self.n_conv_feat = n_conv_feat
        self.separable_gmc = separable_gmc
        self.residual = residual
        if self.separable_gmc:
            raise NotImplementedError('rnn method should return W and H. Then '
                                      'fix loss calculation.')

        with tf.Graph().as_default() as g:
            tf.logging.set_verbosity(tf.logging.ERROR)
            self.graph = g
            tf.set_random_seed(0)
            with tf.device(idx_gpu):
                # For every graph laplacian perform operation
                output_list = []
                for k, cur_lr in enumerate(Lr):
                    cur_res = self.rnn(cur_lr, k)
                    output_list.append(cur_res)
                cur_X = tf.stack(output_list, 1)

                # Single-head self-attention. Has many to many output and
                # vectors averaged out.
                temp_mha = MultiHeadAttention(d_model=cur_X.shape[-1],
                                                   num_heads=1)
                out, attn = temp_mha(cur_X, k=cur_X, q=cur_X, mask=None)
                cur_X = tf.reduce_mean(out, 1)

                # Seperate features X and labels Y
                if len(self.labels_y.shape) > 1 and self.labels_y.shape[-1] \
                        > 1:
                    self.num_class = self.labels_y.shape[-1]
                else:
                    self.num_class = 1
                self.Y = cur_X[:, -self.num_class:]
                self.Y = tf.squeeze(self.Y)
                self.X = cur_X[:, :-self.num_class]

                self.labels_y = tf.cast(labels_y, tf.float32)
                self.M = tf.constant(self.M, dtype=tf.float32)

                self.Otraining = tf.constant(Otraining,
                                             dtype=tf.float32)  # training mask
                self.Otest = tf.constant(Otest, dtype=tf.float32)  # test mask
                self.OVal = tf.constant(Oval, dtype=tf.float32)  # validation

                # Classification loss
                if self.num_class == 1:
                    self.tr_loss_bce = tf.nn.sigmoid_cross_entropy_with_logits(
                        labels=tf.boolean_mask(self.labels_y, train_mask),
                        logits=tf.boolean_mask(self.Y, train_mask))
                    self.ts_loss_bce = tf.nn.sigmoid_cross_entropy_with_logits(
                        labels=tf.boolean_mask(self.labels_y, test_mask),
                        logits=tf.boolean_mask(self.Y, test_mask))
                    self.val_loss_bce = tf.nn.sigmoid_cross_entropy_with_logits(
                        labels=tf.boolean_mask(self.labels_y, val_mask),
                        logits=tf.boolean_mask(self.Y, val_mask))
                else:
                    self.tr_loss_bce = tf.nn.softmax_cross_entropy_with_logits(
                        labels=tf.boolean_mask(self.labels_y, train_mask),
                        logits=tf.boolean_mask(self.Y, train_mask))
                    self.ts_loss_bce = tf.nn.softmax_cross_entropy_with_logits(
                        labels=tf.boolean_mask(self.labels_y, test_mask),
                        logits=tf.boolean_mask(self.Y, test_mask))
                    self.val_loss_bce = tf.nn.softmax_cross_entropy_with_logits(
                        labels=tf.boolean_mask(self.labels_y, val_mask),
                        logits=tf.boolean_mask(self.Y, val_mask))

                self.tr_loss_bce = tf.reduce_mean(self.tr_loss_bce)
                self.ts_loss_bce = tf.reduce_mean(self.ts_loss_bce)
                self.val_loss_bce = tf.reduce_mean(self.val_loss_bce)

                # Validation and test set predicted probabilities
                self.val_Y_out = tf.boolean_mask(self.Y, val_mask)

                # Test set output and sigmoid output
                self.ts_y_output = tf.boolean_mask(self.Y, test_mask)
                if self.num_class == 1:
                    self.sigmoid_ts_Y_out = tf.nn.sigmoid(self.ts_y_output)
                else:
                    self.sigmoid_ts_Y_out = tf.nn.softmax(self.ts_y_output)

                ###
                # # Training set imputation loss terms with set cardinality
                ###
                frob_tensor = tf.multiply(self.Otraining[:, :-self.num_class],
                                          self.X - self.M[:, :-self.num_class])
                self.loss_frob = tf.square(self.frobenius_norm(frob_tensor))/(
                tf.reduce_sum(self.Otraining[:, :-self.num_class]))

                cur_loss_trace_row = 0
                for cur_L in self.Lr:
                    cur_L = tf.cast(cur_L, 'float32')
                    trace_row_tensor = tf.matmul(
                        tf.matmul(self.X, cur_L, transpose_a=True), self.X)
                    cur_loss_trace_row += tf.trace(trace_row_tensor) / tf.cast(
                        tf.shape(self.X)[0] * tf.shape(self.X)[1], 'float32')
                self.loss_trace_row = cur_loss_trace_row

                if self.separable_gmc:
                    self.frob_norm_H = tf.square(
                        self.frobenius_norm(self.H)) / tf.cast(
                        tf.shape(self.H)[0] * tf.shape(self.H)[1], 'float32')
                    self.frob_norm_W = tf.square(
                        self.frobenius_norm(self.W)) / tf.cast(
                        tf.shape(self.W)[0] * tf.shape(self.W)[1], 'float32')

                # Training loss definition
                if self.separable_gmc:
                    self.loss = (gamma / 2) * self.loss_frob \
                                + (gamma_tr / 2) * self.loss_trace_row \
                                + (gamma_W / 2) * self.frob_norm_W \
                                + (gamma_H / 2) * self.frob_norm_H
                else:
                    self.loss = (gamma / 2) * self.loss_frob \
                                + (gamma_tr / 2) * self.loss_trace_row
                self.loss = self.loss + (gamma_bce * self.tr_loss_bce)


                # RMSE of train set rows at imputed entries without RMSE loss
                #  during training
                if impute_idx != None:
                    mat_column_len = self.Otraining[:, :-self.num_class].shape.as_list()[-1]
                    train_feature_mask = np.tile(train_mask, (mat_column_len, 1)).T
                    # train_feature_mask = tf.convert_to_tensor(train_feature_mask,
                    #                                           dtype=tf.float32)
                    train_impute_masking = np.zeros_like(train_feature_mask)
                    train_impute_masking[impute_idx] = 1
                    train_impute_masking = tf.convert_to_tensor(
                        train_impute_masking,
                        dtype=tf.float32)
                    train_feature_mask = tf.multiply(train_impute_masking,
                                                     train_feature_mask)
                    imputed_train_error = tf.multiply(train_feature_mask,
                                                      self.X - M[:, :-self.num_class])
                    self.train_RMSE = tf.sqrt(tf.reduce_mean(tf.square(
                        imputed_train_error)))

                    # RMSE of test set rows at imputed entries without RMSE loss
                    # during training
                    test_feature_mask = np.tile(test_mask, (mat_column_len, 1)).T
                    # test_feature_mask = tf.convert_to_tensor(test_feature_mask,
                    #                                     dtype=tf.float32)
                    test_impute_masking = np.zeros_like(test_feature_mask)
                    test_impute_masking[impute_idx] = 1
                    test_impute_masking = tf.convert_to_tensor(test_impute_masking,
                                                               dtype=tf.float32)
                    test_feature_mask = tf.multiply(test_impute_masking,
                                                    test_feature_mask)
                    imputed_test_error = tf.multiply(test_feature_mask,
                                                     self.X - M[:, :-self.num_class])
                    self.test_RMSE = tf.sqrt(tf.reduce_mean(tf.square(
                        imputed_test_error)))

                    # Test feature mask for saving
                    self.test_feature_mask = test_feature_mask

                    # RMSE of val set rows at imputed entries without RMSE loss
                    # during training
                    val_feature_mask = np.tile(val_mask, (mat_column_len, 1)).T
                    # val_feature_mask = tf.convert_to_tensor(val_feature_mask,
                    #                                         dtype=tf.float32)
                    val_impute_masking = np.zeros_like(val_feature_mask)
                    val_impute_masking[impute_idx] = 1
                    val_impute_masking = tf.convert_to_tensor(
                        val_impute_masking,
                        dtype=tf.float32)
                    val_feature_mask = tf.multiply(val_impute_masking,
                                                     val_feature_mask)
                    imputed_val_error = tf.multiply(val_feature_mask,
                                                      self.X - M[:, :-self.num_class])
                    self.val_RMSE = tf.sqrt(tf.reduce_mean(tf.square(
                        imputed_val_error)))

                else:
                    self.train_RMSE = tf.convert_to_tensor(np.nan)
                    self.val_RMSE = tf.convert_to_tensor(np.nan)
                    self.test_RMSE = tf.convert_to_tensor(np.nan)

                # test loss definition
                # self.predictions = tf.multiply(self.Otest[:,:-self.num_class], self.X - self.M[:,:-self.num_class])
                # self.predictions_error = self.frobenius_norm(self.predictions) + self.ts_loss_bce
                self.predictions_error = self.ts_loss_bce

                # definition of the solver
                self.optimizer = tf.train.AdamOptimizer(
                    learning_rate=learning_rate).minimize(self.loss)

                self.var_grad = tf.gradients(self.loss,
                                             tf.trainable_variables())
                self.norm_grad = self.frobenius_norm(
                    tf.concat([tf.reshape(g, [-1]) for g in self.var_grad], 0))

                # Create a session for running Ops on the Graph.
                config = tf.ConfigProto(allow_soft_placement=True)
                config.gpu_options.allow_growth = True
                self.session = tf.Session(config=config)

                # Run the Op to initialize the variables.
                init = tf.initialize_all_variables()
                self.session.run(init)

                # Save checkpoints
                self.ckpt_saver = tf.train.Saver(max_to_keep=5)
import tensorflow as tf

def cbow_forward(config, inputs, scope=None):
    with tf.variable_scope(scope or "forward"):

        JX, JQ = config.max_context_size, config.max_ques_size
        d = config.hidden_size
        x, x_len, q, q_len = [inputs[key] for key in ['x', 'x_len', 'q', 'q_len']]
        x_mask = tf.sequence_mask(x_len, JX)
        q_mask = tf.sequence_mask(q_len, JQ)

        # emb_mat = tf.get_variable('emb_mat', shape=[V, d])
        emb_mat = config.emb_mat_ph if config.serve else config.emb_mat
        emb_mat = tf.slice(emb_mat, [2, 0], [-1, -1])
        emb_mat = tf.concat([tf.get_variable('emb_mat', shape=[2, d]), emb_mat], axis=0)
        xx = tf.nn.embedding_lookup(emb_mat, x, name='xx')  # [N, JX, d]
        qq = tf.nn.embedding_lookup(emb_mat, q, name='qq')  # [N, JQ, d]

        qq_avg = tf.reduce_mean(bool_mask(qq, q_mask, expand=True), axis=1)  # [N, d]
        qq_avg_exp = tf.expand_dims(qq_avg, axis=1)  # [N, 1, d]
        qq_avg_tiled = tf.tile(qq_avg_exp, [1, JX, 1])  # [N, JX, d]

        xq = tf.concat([xx, qq_avg_tiled, xx * qq_avg_tiled], axis=2)  # [N, JX, 3d]
        xq_flat = tf.reshape(xq, [-1, 3*d])  # [N * JX, 3*d]

        # Compute logits
        with tf.variable_scope('start'):
            logits1 = exp_mask(tf.reshape(tf.layers.dense(inputs=xq_flat, units=1), [-1, JX]), x_mask)  # [N, JX]
            yp1 = tf.argmax(logits1, axis=1)  # [N]
        with tf.variable_scope('stop'):
            logits2 = exp_mask(tf.reshape(tf.layers.dense(inputs=xq_flat, units=1), [-1, JX]), x_mask)  # [N, JX]
            yp2 = tf.argmax(logits2, axis=1)  # [N]

        outputs = {'logits1': logits1, 'logits2': logits2, 'yp1': yp1, 'yp2': yp2}
        variables = {'emb_mat': emb_mat}
        return variables, outputs


def rnn_forward(config, inputs, scope=None):
    with tf.variable_scope(scope or "forward"):
        JX, JQ = config.max_context_size, config.max_ques_size
        d = config.hidden_size
        x, x_len, q, q_len = [inputs[key] for key in ['x', 'x_len', 'q', 'q_len']]
        x_mask = tf.sequence_mask(x_len, JX)
        q_mask = tf.sequence_mask(q_len, JQ)

        emb_mat = config.emb_mat_ph if config.serve else config.emb_mat
        emb_mat = tf.slice(emb_mat, [2, 0], [-1, -1])
        emb_mat = tf.concat([tf.get_variable('emb_mat', shape=[2, d]), emb_mat], axis=0) # [V, d]
        xx = tf.nn.embedding_lookup(emb_mat, x, name='xx')  # [N, JX, d]
        qq = tf.nn.embedding_lookup(emb_mat, q, name='qq')  # [N, JQ, d]

        # RNN
        with tf.variable_scope('context_rnn'):
            xx = rnn_gru(config, xx, 'context_rnn')  # [N, JX, d]
        with tf.variable_scope('question_rnn'):
            qq = rnn_gru(config, qq, 'question_rnn')  # [N, JQ, d]

        # Equation 1
        qq_avg = tf.reduce_mean(bool_mask(qq, q_mask, expand=True), axis=1)  # [N, d]
        qq_avg_exp = tf.expand_dims(qq_avg, axis=1)  # [N, 1, d]
        qq_avg_tiled = tf.tile(qq_avg_exp, [1, JX, 1])  # [N, JX, d]

        # Equation 2
        xq = tf.concat([xx, qq_avg_tiled, xx * qq_avg_tiled], axis=2)  # [N, JX, 3d]
        xq_flat = tf.reshape(xq, [-1, 3*d])  # [N * JX, 3*d]

        # Compute logits
        with tf.variable_scope('start'):
            logits1 = exp_mask(tf.reshape(tf.layers.dense(inputs=xq_flat, units=1), [-1, JX]), x_mask)  # [N, JX]
            yp1 = tf.argmax(logits1, axis=1)  # [N]
        with tf.variable_scope('stop'):
            logits2 = exp_mask(tf.reshape(tf.layers.dense(inputs=xq_flat, units=1), [-1, JX]), x_mask)  # [N, JX]
            yp2 = tf.argmax(logits2, axis=1)  # [N]

        outputs = {'logits1': logits1, 'logits2': logits2, 'yp1': yp1, 'yp2': yp2}
        variables = {'emb_mat': emb_mat}
        return variables, outputs

def attention_forward(config, inputs, scope=None):
    with tf.variable_scope(scope or "forward"):
        JX, JQ = config.max_context_size, config.max_ques_size
        d = config.hidden_size
        x, x_len, q, q_len = [inputs[key] for key in ['x', 'x_len', 'q', 'q_len']]
        x_mask = tf.sequence_mask(x_len, JX)
        q_mask = tf.sequence_mask(q_len, JQ)

        emb_mat = config.emb_mat_ph if config.serve else config.emb_mat
        emb_mat = tf.slice(emb_mat, [2, 0], [-1, -1])
        emb_mat = tf.concat([tf.get_variable('emb_mat', shape=[2, d]), emb_mat], axis=0) # [V, d]
        xx = tf.nn.embedding_lookup(emb_mat, x, name='xx')  # [N, JX, d]
        qq = tf.nn.embedding_lookup(emb_mat, q, name='qq')  # [N, JQ, d]

        # RNN
        with tf.variable_scope('context_rnn'):
            xx = rnn_gru(config, xx, 'context_rnn')  # [N, JX, d]
        with tf.variable_scope('question_rnn'):
            qq = rnn_gru(config, qq, 'question_rnn')  # [N, JQ, d]

        # Equation 10
        xx_exp = tf.expand_dims(xx, axis=2)  # [N, JX, 1, d]
        xx_tiled = tf.tile(xx_exp, [1, 1, JQ, 1])  # [N, JX, JQ, d]
        qq_exp = tf.expand_dims(qq, axis=1)  # [N, JX, 1, d]
        qq_tiled = tf.tile(qq_exp, [1, JX, 1, 1])  # [N, JX, JQ, d]
        xxqq = tf.concat([xx_tiled, qq_tiled, xx_tiled * qq_tiled], axis=2)  # [N, JX, JQ, 3d]

        xxqq_reshape = tf.reshape(xxqq, [-1, 3 * d])  # [N * JX * JQ, 3d]
        weight_p = tf.get_variable('weight_p', shape=[3 * d, 1])  # [3d, 1]
        value = tf.matmul(xxqq_reshape, weight_p)  # [N * JX * JQ, 1]
        value = tf.reshape(value, [tf.shape(xxqq)[0], JX, JQ])  # [N, JX, JQ]
        p = tf.nn.softmax(value)  # [N, JX, JQ]

        # Equation 9
        p_exp = tf.expand_dims(p, axis=3)  # [N, JX, JQ, 1]
        p_tiled = tf.tile(p_exp, [1, 1, 1, d])  # [N, JX, JQ, d]
        value = p_tiled * qq_tiled  # [N, JX, JQ, d]
        qq_hat = tf.reduce_sum(value, 2)  # [N, JX, d]

        # Equation 2
        xq = tf.concat([xx, qq_hat, xx * qq_hat], axis=2)  # [N, JX, 3d]
        xq_flat = tf.reshape(xq, [-1, 3*d])  # [N * JX, 3*d]

        # Compute logits
        with tf.variable_scope('start'):
            logits1 = exp_mask(tf.reshape(tf.layers.dense(inputs=xq_flat, units=1), [-1, JX]), x_mask)  # [N, JX]
            yp1 = tf.argmax(logits1, axis=1)  # [N]
        with tf.variable_scope('stop'):
            logits2 = exp_mask(tf.reshape(tf.layers.dense(inputs=xq_flat, units=1), [-1, JX]), x_mask)  # [N, JX]
            yp2 = tf.argmax(logits2, axis=1)  # [N]

        outputs = {'logits1': logits1, 'logits2': logits2, 'yp1': yp1, 'yp2': yp2}
        variables = {'emb_mat': emb_mat}
        return variables, outputs

    # raise NotImplementedError()


def get_loss(config, inputs, outputs, scope=None):
    with tf.name_scope(scope or "loss"):
        y1, y2 = inputs['y1'], inputs['y2']
        logits1, logits2 = outputs['logits1'], outputs['logits2']
        loss1 = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(labels=y1, logits=logits1))
        loss2 = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(labels=y2, logits=logits2))
        loss = loss1 + loss2
        acc1 = tf.reduce_mean(tf.cast(tf.equal(y1, tf.cast(tf.argmax(logits1, 1), 'int32')), 'float'))
        acc2 = tf.reduce_mean(tf.cast(tf.equal(y2, tf.cast(tf.argmax(logits2, 1), 'int32')), 'float'))
        tf.summary.scalar('loss', loss)
        tf.summary.scalar('acc1', acc1)
        tf.summary.scalar('acc2', acc2)
        return loss


def exp_mask(val, mask, expand=False):
    if expand:
        mask = tf.expand_dims(mask, -1)
    return val - (1.0 - tf.cast(mask, 'float')) * 10.0e10


def bool_mask(val, mask, expand=False):
    if expand:
        mask = tf.expand_dims(mask, -1)
    return val * tf.cast(mask, 'float')

def build_gru_cell(num_units, num_layers, batch_size, is_training, output_keep_prob):
    def build_cell(num_units, is_training, output_keep_prob):
        cell = tf.contrib.rnn.GRUCell(num_units, reuse=tf.AUTO_REUSE)
        if is_training and output_keep_prob < 1:
            cell = tf.contrib.rnn.DropoutWrapper(cell, output_keep_prob=0.9)
        return cell

    if num_layers > 1:
        with tf.name_scope('multi_cells'):
            cell = tf.nn.rnn_cell.MultiRNNCell([build_cell(num_units, is_training, output_keep_prob)] * num_layers)
    else:
        cell = build_cell(num_units, is_training, output_keep_prob)

    init_state = cell.zero_state(batch_size, tf.float32)
    return cell, init_state

def rnn_gru(config, input, scope):
    start_fw_cell, start_fw_state = build_gru_cell(config.hidden_size, 1, config.batch_size, config.is_train, config.keep_prob)
    start_bw_cell, start_bw_state = build_gru_cell(config.hidden_size, 1, config.batch_size, config.is_train, config.keep_prob)
    (outputs_fw, outputs_bw), _ = tf.nn.bidirectional_dynamic_rnn(cell_fw=start_fw_cell,
                                                                    cell_bw=start_bw_cell,
                                                                    inputs=input,
                                                                    dtype=tf.float32,
                                                                    scope=scope)
    return (outputs_fw + outputs_bw) / 2

import tensorflow as tf
from tensorflow.python.ops.rnn import dynamic_rnn as _dynamic_rnn, \
    bidirectional_dynamic_rnn as _bidirectional_dynamic_rnn

from my.tensorflow import flatten, reconstruct


def dynamic_rnn(cell, inputs, sequence_length=None, initial_state=None,
                dtype=None, parallel_iterations=None, swap_memory=False,
                time_major=False, scope=None):
    assert not time_major  # TODO : to be implemented later!
    flat_inputs = flatten(inputs, 2)  # [-1, J, d]
    flat_len = None if sequence_length is None else tf.cast(flatten(sequence_length, 0), 'int64')

    flat_outputs, final_state = _dynamic_rnn(cell, flat_inputs, sequence_length=flat_len,
                                             initial_state=initial_state, dtype=dtype,
                                             parallel_iterations=parallel_iterations, swap_memory=swap_memory,
                                             time_major=time_major, scope=scope)

    outputs = reconstruct(flat_outputs, inputs, 2)
    return outputs, final_state


def bw_dynamic_rnn(cell, inputs, sequence_length=None, initial_state=None,
                   dtype=None, parallel_iterations=None, swap_memory=False,
                   time_major=False, scope=None):
    assert not time_major  # TODO : to be implemented later!

    flat_inputs = flatten(inputs, 2)  # [-1, J, d]
    flat_len = None if sequence_length is None else tf.cast(flatten(sequence_length, 0), 'int64')

    flat_inputs = tf.reverse(flat_inputs, 1) if sequence_length is None \
        else tf.reverse_sequence(flat_inputs, sequence_length, 1)
    flat_outputs, final_state = _dynamic_rnn(cell, flat_inputs, sequence_length=flat_len,
                                             initial_state=initial_state, dtype=dtype,
                                             parallel_iterations=parallel_iterations, swap_memory=swap_memory,
                                             time_major=time_major, scope=scope)
    flat_outputs = tf.reverse(flat_outputs, 1) if sequence_length is None \
        else tf.reverse_sequence(flat_outputs, sequence_length, 1)

    outputs = reconstruct(flat_outputs, inputs, 2)
    return outputs, final_state


def bidirectional_dynamic_rnn(cell_fw, cell_bw, inputs, sequence_length=None,
                              initial_state_fw=None, initial_state_bw=None,
                              dtype=None, parallel_iterations=None,
                              swap_memory=False, time_major=False, scope=None):
    assert not time_major

    flat_inputs = flatten(inputs, 2)  # [-1, J, d]
    flat_len = None if sequence_length is None else tf.cast(flatten(sequence_length, 0), 'int64')

    (flat_fw_outputs, flat_bw_outputs), final_state = \
        _bidirectional_dynamic_rnn(cell_fw, cell_bw, flat_inputs, sequence_length=flat_len,
                                   initial_state_fw=initial_state_fw, initial_state_bw=initial_state_bw,
                                   dtype=dtype, parallel_iterations=parallel_iterations, swap_memory=swap_memory,
                                   time_major=time_major, scope=scope)

    fw_outputs = reconstruct(flat_fw_outputs, inputs, 2)
    bw_outputs = reconstruct(flat_bw_outputs, inputs, 2)
    # FIXME : final state is not reshaped!
    return (fw_outputs, bw_outputs), final_state


def bidirectional_rnn(cell_fw, cell_bw, inputs,
                      initial_state_fw=None, initial_state_bw=None,
                      dtype=None, sequence_length=None, scope=None):

    flat_inputs = flatten(inputs, 2)  # [-1, J, d]
    flat_len = None if sequence_length is None else tf.cast(flatten(sequence_length, 0), 'int64')

    (flat_fw_outputs, flat_bw_outputs), final_state = \
        tf.nn.bidirectional_dynamic_rnn(cell_fw, cell_bw, flat_inputs, sequence_length=flat_len,
                                        initial_state_fw=initial_state_fw, initial_state_bw=initial_state_bw,
                                        dtype=dtype, scope=scope)

    fw_outputs = reconstruct(flat_fw_outputs, inputs, 2)
    bw_outputs = reconstruct(flat_bw_outputs, inputs, 2)
    # FIXME : final state is not reshaped!
    return (fw_outputs, bw_outputs), final_state

def ruminating_layer(S,inputs,N,M,L,d):
    '''
    ruminate inputs over summary S.
    N - batch_size, M - sent cont,
    L- Length of sequence, d - state size
    '''
    xavier_init = tf.contrib.layers.xavier_initializer()
    zero_init = tf.constant_initializer(0)
    with tf.variable_scope("ruminating_layer"):
        W1_z = tf.get_variable('W1_z',shape=  (2*d,2*d),dtype=tf.float32,initializer=xavier_init)
        W2_z = tf.get_variable('W2_z',shape=(2*d,2*d),dtype=tf.float32,initializer=xavier_init)
        b_z = tf.get_variable('b_z',shape=(2*d,),dtype=tf.float32,initializer=zero_init)
        W1_f = tf.get_variable('W1_f',shape=(2*d,2*d),dtype=tf.float32,initializer=xavier_init)
        W2_f = tf.get_variable('W2_f',shape=(2*d,2*d),dtype=tf.float32,initializer=xavier_init)
        b_f = tf.get_variable('b_f',shape=(2*d,),dtype=tf.float32,initializer=zero_init)

        z_part1 = tf.reshape(tf.matmul(tf.reshape(S,[-1,2*d]),W1_z) ,[N,M,L,2*d],name='z_part1')
        z_part2 = tf.reshape(tf.expand_dims(tf.matmul(tf.reshape(inputs,[-1,2*d]), W2_z) + b_z, 1),[N,M,L,2*d])
        z = tf.tanh(z_part1 + z_part2 ,name='z')
        f_part1 = tf.reshape(tf.matmul(tf.reshape(S,[-1,2*d]),W1_f) ,[N,M,L,2*d],name='f_part1')
        f_part2 = tf.reshape(tf.expand_dims(tf.matmul(tf.reshape(inputs,[-1,2*d]), W2_f) + b_f, 1,name='f_part2'),[N,M,L,2*d])
        f = tf.sigmoid(f_part1 + f_part2,name='f')
        input_hat = tf.multiply(f, inputs) + tf.multiply( (1 - f),z)
    return input_hat

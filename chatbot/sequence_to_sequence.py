import numpy as np
import tensorflow as tf
from tensorflow import layers
# from tensorflow.python.util import nest
from tensorflow.python.ops import array_ops
from tensorflow.contrib import seq2seq
from tensorflow.contrib.seq2seq import BahdanauAttention
from tensorflow.contrib.seq2seq import LuongAttention
from tensorflow.contrib.seq2seq import AttentionWrapper
from tensorflow.contrib.seq2seq import BeamSearchDecoder
from tensorflow.contrib.rnn import LSTMCell
from tensorflow.contrib.rnn import GRUCell
from tensorflow.contrib.rnn import MultiRNNCell
from tensorflow.contrib.rnn import DropoutWrapper
from tensorflow.contrib.rnn import ResidualWrapper
# from tensorflow.contrib.rnn import LSTMStateTuple

from word_sequence import WordSequence
from data_utils import _get_embed_device


class SequenceToSequence(object):


    def __init__(self,
                 input_vocab_size,
                 target_vocab_size,
                 batch_size=32,
                 embedding_size=300,
                 mode='train',
                 hidden_units=256,
                 depth=1,
                 beam_width=0,
                 cell_type='lstm',
                 dropout=0.2,
                 use_dropout=False,
                 use_residual=False,
                 optimizer='adam',
                 learning_rate=1e-3,
                 min_learning_rate=1e-6,
                 decay_steps=500000,
                 max_gradient_norm=5.0,
                 max_decode_step=None,
                 attention_type='Bahdanau',
                 bidirectional=False,
                 time_major=False,
                 seed=0,
                 parallel_iterations=None,
                 share_embedding=False,
                 pretrained_embedding=False):
        """save parameters
        Args:
            input_vocab_size: input table
            target_vocab_size: output table
            batch_size: the size of batch
            embedding_size,
            mode: train or decode,
            hidden_units:
                RNN: single
                bidirectional: double
            depth: encoder & decoder's layer count.
            beam_width:
                beam_width is the hyper-parameter of beamsearch, for decoding.
                larger than 0 or equal with 0: use beamsearch，smaller than 0: not use beamsearch
            cell_type: rnn,lstm,gru
            dropout: dropout [0, 1)
            use_dropout: if use dropout
            use_residual:# if use residual
            optimizer: adam, adadelta, sgd, rmsprop, momentum
            learning_rate: 
            max_gradient_norm: max gradient
            max_decode_step:
                The maximum decoding length, which can be a very large integer, the default is None
                In the case of None, the default is 4 times the maximum length of the encoder input
            attention_type: 'Bahdanau' or 'Luong', the two attention mechanism
            bidirectional: if the encoder is Bi-directional
            time_major: Whether to use time-based batch data in the "calculation process"
            seed: Random number seed settings for some inter-layer operations
            parallel_iterations:
                Number of parallels: dynamic_rnn and dynamic_decode 
            share_embedding:
                if True，encoder and the decoder will use the same embedding
        """

        self.input_vocab_size = input_vocab_size
        self.target_vocab_size = target_vocab_size
        self.batch_size = batch_size
        self.embedding_size = embedding_size
        self.hidden_units = hidden_units
        self.depth = depth
        self.cell_type = cell_type.lower()
        self.use_dropout = use_dropout
        self.use_residual = use_residual
        self.attention_type = attention_type
        self.mode = mode
        self.optimizer = optimizer
        self.learning_rate = learning_rate
        self.min_learning_rate = min_learning_rate
        self.decay_steps = decay_steps
        self.max_gradient_norm = max_gradient_norm
        self.keep_prob = 1.0 - dropout
        self.bidirectional = bidirectional
        self.seed = seed
        self.pretrained_embedding = pretrained_embedding
        if isinstance(parallel_iterations, int):
            self.parallel_iterations = parallel_iterations
        else: # if parallel_iterations is None:
            self.parallel_iterations = batch_size
        self.time_major = time_major
        self.share_embedding = share_embedding

        self.initializer = tf.random_uniform_initializer(
            -0.05, 0.05, dtype=tf.float32
        )
        # self.initializer = None

        assert self.cell_type in ('gru', 'lstm'), \
            'cell_type should be GRU or LSTM'

        if share_embedding:
            assert input_vocab_size == target_vocab_size, \
                'if the share_embedding is true，the two vocab_size must be same'

        assert mode in ('train', 'decode'), \
            'mode must be "train" or "decode" but not "{}"'.format(mode)

        assert dropout >= 0.0 and dropout < 1.0, '0 <= dropout < 1'

        assert attention_type.lower() in ('bahdanau', 'luong'), \
            '''attention_type must be "bahdanau" or "luong" but not "{}"
            '''.format(attention_type)

        assert beam_width < target_vocab_size, \
            'beam_width {} should smaller than target vocab size {}'.format(
                beam_width, target_vocab_size
            )

        self.keep_prob_placeholder = tf.placeholder(
            tf.float32,
            shape=[],
            name='keep_prob'
        )

        self.global_step = tf.Variable(
            0, trainable=False, name='global_step'
        )

        self.use_beamsearch_decode = False
        self.beam_width = beam_width
        self.use_beamsearch_decode = True if self.beam_width > 0 else False
        self.max_decode_step = max_decode_step

        assert self.optimizer.lower() in \
            ('adadelta', 'adam', 'rmsprop', 'momentum', 'sgd'), \
            'optimizer: adadelta, adam, rmsprop, momentum, sgd'

        self.build_model()


    def build_model(self):
        """initial the model
        encoder
        decoder
        optimizer
        """
        self.init_placeholders()
        encoder_outputs, encoder_state = self.build_encoder()
        self.build_decoder(encoder_outputs, encoder_state)

        if self.mode == 'train':
            self.init_optimizer()

        self.saver = tf.train.Saver()


    def init_placeholders(self):
        """parameters for training.
        """

        self.add_loss = tf.placeholder(
            dtype=tf.float32,
            name='add_loss'
        )
        # the max length for each batch is shown as index in time_step
        self.encoder_inputs = tf.placeholder(
            dtype=tf.int32,
            shape=(self.batch_size, None),
            name='encoder_inputs'
        )

        self.encoder_inputs_length = tf.placeholder(
            dtype=tf.int32,
            shape=(self.batch_size,),
            name='encoder_inputs_length'
        )

        if self.mode == 'train':
            # train process:
            # shape=(batch_size, time_step)
            # <EOS> is included in each sentence
            self.decoder_inputs = tf.placeholder(
                dtype=tf.int32,
                shape=(self.batch_size, None),
                name='decoder_inputs'
            )

            self.rewards = tf.placeholder(
                dtype=tf.float32,
                shape=(self.batch_size, 1),
                name='rewards'
            )

            self.decoder_inputs_length = tf.placeholder(
                dtype=tf.int32,
                shape=(self.batch_size,),
                name='decoder_inputs_length'
            )

            self.decoder_start_token = tf.ones(
                shape=(self.batch_size, 1),
                dtype=tf.int32
            ) * WordSequence.START

            # the input for training the decoder is : start_token + decoder_inputs
            self.decoder_inputs_train = tf.concat([
                self.decoder_start_token,
                self.decoder_inputs
            ], axis=1)


    def build_single_cell(self, n_hidden, use_residual):
        """initial a single RNN cell
        Args:
            n_hidden
            use_residual: BOOL
        """

        if self.cell_type == 'gru':
            cell_type = GRUCell
        else:
            cell_type = LSTMCell

        cell = cell_type(n_hidden)

        if self.use_dropout:
            cell = DropoutWrapper(
                cell,
                dtype=tf.float32,
                output_keep_prob=self.keep_prob_placeholder,
                seed=self.seed
            )

        if use_residual:
            cell = ResidualWrapper(cell)

        return cell

    def build_encoder_cell(self):
        """initial a single encoder cell
        """
        return MultiRNNCell([
            self.build_single_cell(
                self.hidden_units,
                use_residual=self.use_residual
            )
            for _ in range(self.depth)
        ])


    def feed_embedding(self, sess, encoder=None, decoder=None):
        """load the well-training embedding
        """
        assert self.pretrained_embedding, \
            'U must open the pretrained_embedding to use the feed_embedding'
        assert encoder is not None or decoder is not None, \
            'encoder or decoder must be valid!'

        if encoder is not None:
            sess.run(self.encoder_embeddings_init,
                     {self.encoder_embeddings_placeholder: encoder})

        if decoder is not None:
            sess.run(self.decoder_embeddings_init,
                     {self.decoder_embeddings_placeholder: decoder})


    def build_encoder(self):
        """initial the encoder
        """
        # print("initial the encoder")
        with tf.variable_scope('encoder'):
            # build encoder_cell
            encoder_cell = self.build_encoder_cell()

            # encoder's embedding
            with tf.device(_get_embed_device(self.input_vocab_size)):

                
                if self.pretrained_embedding:

                    self.encoder_embeddings = tf.Variable(
                        tf.constant(
                            0.0,
                            shape=(self.input_vocab_size, self.embedding_size)
                        ),
                        trainable=True,
                        name='embeddings'
                    )
                    self.encoder_embeddings_placeholder = tf.placeholder(
                        tf.float32,
                        (self.input_vocab_size, self.embedding_size)
                    )
                    self.encoder_embeddings_init = \
                        self.encoder_embeddings.assign(
                            self.encoder_embeddings_placeholder)

                else:
                    self.encoder_embeddings = tf.get_variable(
                        name='embedding',
                        shape=(self.input_vocab_size, self.embedding_size),
                        initializer=self.initializer,
                        dtype=tf.float32
                    )

            # shape = (batch_size, time_step, embedding_size)
            self.encoder_inputs_embedded = tf.nn.embedding_lookup(
                params=self.encoder_embeddings,
                ids=self.encoder_inputs
            )

            if self.use_residual:
                self.encoder_inputs_embedded = \
                    layers.dense(self.encoder_inputs_embedded,
                                 self.hidden_units,
                                 use_bias=False,
                                 name='encoder_residual_projection')


            inputs = self.encoder_inputs_embedded
            if self.time_major:
                inputs = tf.transpose(inputs, (1, 0, 2))

            if not self.bidirectional:
                # RNN
                (
                    encoder_outputs,
                    encoder_state
                ) = tf.nn.dynamic_rnn(
                    cell=encoder_cell,
                    inputs=inputs,
                    sequence_length=self.encoder_inputs_length,
                    dtype=tf.float32,
                    time_major=self.time_major,
                    parallel_iterations=self.parallel_iterations,
                    swap_memory=True
                )
            else:
                # Bi-RNN
                encoder_cell_bw = self.build_encoder_cell()
                (
                    (encoder_fw_outputs, encoder_bw_outputs),
                    (encoder_fw_state, encoder_bw_state)
                ) = tf.nn.bidirectional_dynamic_rnn(
                    cell_fw=encoder_cell,
                    cell_bw=encoder_cell_bw,
                    inputs=inputs,
                    sequence_length=self.encoder_inputs_length,
                    dtype=tf.float32,
                    time_major=self.time_major,
                    parallel_iterations=self.parallel_iterations,
                    swap_memory=True
                )

                encoder_outputs = tf.concat(
                    (encoder_fw_outputs, encoder_bw_outputs), 2)

                encoder_state = []
                for i in range(self.depth):
                    encoder_state.append(encoder_fw_state[i])
                    encoder_state.append(encoder_bw_state[i])
                encoder_state = tuple(encoder_state)

            return encoder_outputs, encoder_state


    def build_decoder_cell(self, encoder_outputs, encoder_state):
        """initial decoder cell"""

        encoder_inputs_length = self.encoder_inputs_length
        batch_size = self.batch_size

        if self.bidirectional:
            encoder_state = encoder_state[-self.depth:]

        if self.time_major:
            encoder_outputs = tf.transpose(encoder_outputs, (1, 0, 2))

        if self.use_beamsearch_decode:
            encoder_outputs = seq2seq.tile_batch(
                encoder_outputs, multiplier=self.beam_width)
            encoder_state = seq2seq.tile_batch(
                encoder_state, multiplier=self.beam_width)
            encoder_inputs_length = seq2seq.tile_batch(
                self.encoder_inputs_length, multiplier=self.beam_width)
            batch_size *= self.beam_width

        # two Attention mechanism 
        if self.attention_type.lower() == 'luong':
            # 'Luong' style attention: https://arxiv.org/abs/1508.04025
            self.attention_mechanism = LuongAttention(
                num_units=self.hidden_units,
                memory=encoder_outputs,
                memory_sequence_length=encoder_inputs_length
            )
        else: # Default Bahdanau
            # 'Bahdanau' style attention: https://arxiv.org/abs/1409.0473
            self.attention_mechanism = BahdanauAttention(
                num_units=self.hidden_units,
                memory=encoder_outputs,
                memory_sequence_length=encoder_inputs_length
            )

        # Building decoder_cell
        cell = MultiRNNCell([
            self.build_single_cell(
                self.hidden_units,
                use_residual=self.use_residual
            )
            for _ in range(self.depth)
        ])

        alignment_history = (
            self.mode != 'train' and not self.use_beamsearch_decode
        )

        def cell_input_fn(inputs, attention):
            """based on the attn_input_feeding to judge if do a projection calculation before the attention action
            """
            if not self.use_residual:
                return array_ops.concat([inputs, attention], -1)

            attn_projection = layers.Dense(self.hidden_units,
                                           dtype=tf.float32,
                                           use_bias=False,
                                           name='attention_cell_input_fn')
            return attn_projection(array_ops.concat([inputs, attention], -1))

        cell = AttentionWrapper(
            cell=cell,
            attention_mechanism=self.attention_mechanism,
            attention_layer_size=self.hidden_units,
            alignment_history=alignment_history,
            cell_input_fn=cell_input_fn,
            name='Attention_Wrapper')

        # void state
        decoder_initial_state = cell.zero_state(
            batch_size, tf.float32)

        # deliver the encoder state
        decoder_initial_state = decoder_initial_state.clone(
            cell_state=encoder_state)

        return cell, decoder_initial_state


    def build_decoder(self, encoder_outputs, encoder_state):
        """initial the decoder
        """
        with tf.variable_scope('decoder') as decoder_scope:
            (
                self.decoder_cell,
                self.decoder_initial_state
            ) = self.build_decoder_cell(encoder_outputs, encoder_state)

            # the decoder's embedding
            with tf.device(_get_embed_device(self.target_vocab_size)):
                if self.share_embedding:
                    self.decoder_embeddings = self.encoder_embeddings
                elif self.pretrained_embedding:

                    self.decoder_embeddings = tf.Variable(
                        tf.constant(
                            0.0,
                            shape=(self.target_vocab_size,
                                   self.embedding_size)
                        ),
                        trainable=True,
                        name='embeddings'
                    )
                    self.decoder_embeddings_placeholder = tf.placeholder(
                        tf.float32,
                        (self.target_vocab_size, self.embedding_size)
                    )
                    self.decoder_embeddings_init = \
                        self.decoder_embeddings.assign(
                            self.decoder_embeddings_placeholder)
                else:
                    self.decoder_embeddings = tf.get_variable(
                        name='embeddings',
                        shape=(self.target_vocab_size, self.embedding_size),
                        initializer=self.initializer,
                        dtype=tf.float32
                    )

            self.decoder_output_projection = layers.Dense(
                self.target_vocab_size,
                dtype=tf.float32,
                use_bias=False,
                name='decoder_output_projection'
            )

            if self.mode == 'train':
                self.decoder_inputs_embedded = tf.nn.embedding_lookup(
                    params=self.decoder_embeddings,
                    ids=self.decoder_inputs_train
                )
                inputs = self.decoder_inputs_embedded

                if self.time_major:
                    inputs = tf.transpose(inputs, (1, 0, 2))

                training_helper = seq2seq.TrainingHelper(
                    inputs=inputs,
                    sequence_length=self.decoder_inputs_length,
                    time_major=self.time_major,
                    name='training_helper'
                )

                # output_layer should not be used in training process
                training_decoder = seq2seq.BasicDecoder(
                    cell=self.decoder_cell,
                    helper=training_helper,
                    initial_state=self.decoder_initial_state,
                )

                # Maximum decoder time_steps in current batch
                max_decoder_length = tf.reduce_max(
                    self.decoder_inputs_length
                )


                (
                    outputs,
                    self.final_state, # contain attention
                    _ # self.final_sequence_lengths
                ) = seq2seq.dynamic_decode(
                    decoder=training_decoder,
                    output_time_major=self.time_major,
                    impute_finished=True,
                    maximum_iterations=max_decoder_length,
                    parallel_iterations=self.parallel_iterations,
                    swap_memory=True,
                    scope=decoder_scope
                )

                self.decoder_logits_train = self.decoder_output_projection(
                    outputs.rnn_output
                )

                # masks: masking for valid and padded time steps,
                # [batch_size, max_time_step + 1]
                self.masks = tf.sequence_mask(
                    lengths=self.decoder_inputs_length,
                    maxlen=max_decoder_length,
                    dtype=tf.float32, name='masks'
                )

                decoder_logits_train = self.decoder_logits_train
                if self.time_major:
                    decoder_logits_train = tf.transpose(decoder_logits_train,
                                                        (1, 0, 2))

                self.decoder_pred_train = tf.argmax(
                    decoder_logits_train, axis=-1,
                    name='decoder_pred_train')

                # train_entropy = cross entropy
                self.train_entropy = \
                    tf.nn.sparse_softmax_cross_entropy_with_logits(
                        labels=self.decoder_inputs,
                        logits=decoder_logits_train)

                self.masks_rewards = self.masks * self.rewards

                self.loss_rewards = seq2seq.sequence_loss(
                    logits=decoder_logits_train,
                    targets=self.decoder_inputs,
                    weights=self.masks_rewards,
                    average_across_timesteps=True,
                    average_across_batch=True,
                )

                self.loss = seq2seq.sequence_loss(
                    logits=decoder_logits_train,
                    targets=self.decoder_inputs,
                    weights=self.masks,
                    average_across_timesteps=True,
                    average_across_batch=True,
                )

                self.loss_add = self.loss + self.add_loss

            elif self.mode == 'decode':
                # non-training process
                start_tokens = tf.tile(
                    [WordSequence.START],
                    [self.batch_size]
                )
                end_token = WordSequence.END

                def embed_and_input_proj(inputs):
                    """projection in input layer : wrapper
                    """
                    return tf.nn.embedding_lookup(
                        self.decoder_embeddings,
                        inputs
                    )

                if not self.use_beamsearch_decode:
                    # Helper to feed inputs for greedy decoding:
                    # uses the argmax of the output
                    decoding_helper = seq2seq.GreedyEmbeddingHelper(
                        start_tokens=start_tokens,
                        end_token=end_token,
                        embedding=embed_and_input_proj
                    )
                    # Basic decoder performs greedy decoding at each time step
                    # print("building greedy decoder..")
                    inference_decoder = seq2seq.BasicDecoder(
                        cell=self.decoder_cell,
                        helper=decoding_helper,
                        initial_state=self.decoder_initial_state,
                        output_layer=self.decoder_output_projection
                    )
                else:
                    # Beamsearch is used to approximately
                    # find the most likely translation
                    # print("building beamsearch decoder..")
                    inference_decoder = BeamSearchDecoder(
                        cell=self.decoder_cell,
                        embedding=embed_and_input_proj,
                        start_tokens=start_tokens,
                        end_token=end_token,
                        initial_state=self.decoder_initial_state,
                        beam_width=self.beam_width,
                        output_layer=self.decoder_output_projection,
                    )

                if self.max_decode_step is not None:
                    max_decode_step = self.max_decode_step
                else:
                    # 4 times for the output decoder based on the input length.
                    max_decode_step = tf.round(tf.reduce_max(
                        self.encoder_inputs_length) * 4)

                (
                    self.decoder_outputs_decode,
                    self.final_state,
                    _ # self.decoder_outputs_length_decode
                ) = (seq2seq.dynamic_decode(
                    decoder=inference_decoder,
                    output_time_major=self.time_major,
                    # impute_finished=True,	# error occurs
                    maximum_iterations=max_decode_step,
                    parallel_iterations=self.parallel_iterations,
                    swap_memory=True,
                    scope=decoder_scope
                ))

                if not self.use_beamsearch_decode:

                    dod = self.decoder_outputs_decode
                    self.decoder_pred_decode = dod.sample_id

                    if self.time_major:
                        self.decoder_pred_decode = tf.transpose(
                            self.decoder_pred_decode, (1, 0))

                else:
                    self.decoder_pred_decode = \
                        self.decoder_outputs_decode.predicted_ids

                    if self.time_major:
                        self.decoder_pred_decode = tf.transpose(
                            self.decoder_pred_decode, (1, 0, 2))

                    self.decoder_pred_decode = tf.transpose(
                        self.decoder_pred_decode,
                        perm=[0, 2, 1])
                    dod = self.decoder_outputs_decode
                    self.beam_prob = dod.beam_search_decoder_output.scores


    def save(self, sess, save_path='model.ckpt'):
        """save model"""
        self.saver.save(sess, save_path=save_path)


    def load(self, sess, save_path='model.ckpt'):
        """load model"""
        print('try load model from', save_path)
        self.saver.restore(sess, save_path)


    def init_optimizer(self):
        """initial the optimizer
        sgd, adadelta, adam, rmsprop, momentum
        """

        #lr descent
        learning_rate = tf.train.polynomial_decay(
            self.learning_rate,
            self.global_step,
            self.decay_steps,
            self.min_learning_rate,
            power=0.5
        )
        self.current_learning_rate = learning_rate

        # initial optimizer
        # 'adadelta', 'adam', 'rmsprop', 'momentum', 'sgd'
        trainable_params = tf.trainable_variables()
        if self.optimizer.lower() == 'adadelta':
            self.opt = tf.train.AdadeltaOptimizer(
                learning_rate=learning_rate)
        elif self.optimizer.lower() == 'adam':
            self.opt = tf.train.AdamOptimizer(
                learning_rate=learning_rate)
        elif self.optimizer.lower() == 'rmsprop':
            self.opt = tf.train.RMSPropOptimizer(
                learning_rate=learning_rate)
        elif self.optimizer.lower() == 'momentum':
            self.opt = tf.train.MomentumOptimizer(
                learning_rate=learning_rate, momentum=0.9)
        elif self.optimizer.lower() == 'sgd':
            self.opt = tf.train.GradientDescentOptimizer(
                learning_rate=learning_rate)

        gradients = tf.gradients(self.loss, trainable_params)
        # Clip gradients by a given maximum_gradient_norm
        clip_gradients, _ = tf.clip_by_global_norm(
            gradients, self.max_gradient_norm)
        # Update the model
        self.updates = self.opt.apply_gradients(
            zip(clip_gradients, trainable_params),
            global_step=self.global_step)

        # rewards for update the loss
        gradients = tf.gradients(self.loss_rewards, trainable_params)
        clip_gradients, _ = tf.clip_by_global_norm(
            gradients, self.max_gradient_norm)
        self.updates_rewards = self.opt.apply_gradients(
            zip(clip_gradients, trainable_params),
            global_step=self.global_step)

        # add the update in self.loss_add
        gradients = tf.gradients(self.loss_add, trainable_params)
        clip_gradients, _ = tf.clip_by_global_norm(
            gradients, self.max_gradient_norm)
        self.updates_add = self.opt.apply_gradients(
            zip(clip_gradients, trainable_params),
            global_step=self.global_step)


    def check_feeds(self, encoder_inputs, encoder_inputs_length,
                    decoder_inputs, decoder_inputs_length, decode):
        """check the input parameters, and the input_feed is the output.

        We need to encode the data: such as 你 好 饿 to [0, 1, 2]
        set batch_size=2，
        and the the encoder_inputs might be = [
            [0, 1, 2, 3],
            [4, 5, 6, 7]
        ]
        the meaning might be: [['我', '好', '饿', '啊'], ['你', '好', '饿', '</s>']]
        But the length is different: the first sentence is 4, but the second one is 3, because the </s>
       
        encoder_inputs_length = [4, 3] is the actual length.


        Args:
            encoder_inputs:
                An integer two-dimensional array: [batch_size, max_source_time_steps]
            encoder_inputs_length:
                An integer vector [batch_size]
            decoder_inputs:
                An integer matrix: [batch_size, max_target_time_steps]
            decoder_inputs_length:
                An integer vector [batch_size]
            decode: to flag if the (decode=False) or the (decode=True)
        Returns:
            encoder_inputs, encoder_inputs_length,
            decoder_inputs, decoder_inputs_length
        """

        input_batch_size = encoder_inputs.shape[0]
        if input_batch_size != encoder_inputs_length.shape[0]:
            raise ValueError(
                "The first dimension of encoder_inputs和encoder_inputs_length must be consistent!"
                "the batch_size, %d != %d" % (
                    input_batch_size, encoder_inputs_length.shape[0]))

        if not decode:
            target_batch_size = decoder_inputs.shape[0]
            if target_batch_size != input_batch_size:
                raise ValueError(
                    "The first dimension of encoder_inputs and decoder_inputs must be consistent!"
                    "the batch_size, %d != %d" % (
                        input_batch_size, target_batch_size))
            if target_batch_size != decoder_inputs_length.shape[0]:
                raise ValueError(
                    "The first dimension of  edeoder_inputs and decoder_inputs_length must be consistent!"
                    "the batch_size, %d != %d" % (
                        target_batch_size, decoder_inputs_length.shape[0]))

        input_feed = {}

        input_feed[self.encoder_inputs.name] = encoder_inputs
        input_feed[self.encoder_inputs_length.name] = encoder_inputs_length

        if not decode:
            input_feed[self.decoder_inputs.name] = decoder_inputs
            input_feed[self.decoder_inputs_length.name] = decoder_inputs_length

        return input_feed


    def train(self, sess, encoder_inputs, encoder_inputs_length,
              decoder_inputs, decoder_inputs_length,
              rewards=None, return_lr=False,
              loss_only=False, add_loss=None):
        """train the model"""

        # output
        input_feed = self.check_feeds(
            encoder_inputs, encoder_inputs_length,
            decoder_inputs, decoder_inputs_length,
            False
        )

        # set the dropout
        input_feed[self.keep_prob_placeholder.name] = self.keep_prob

        if loss_only:
            # output
            return sess.run(self.loss, input_feed)

        if add_loss is not None:
            input_feed[self.add_loss.name] = add_loss
            output_feed = [
                self.updates_add, self.loss_add,
                self.current_learning_rate]
            _, cost, lr = sess.run(output_feed, input_feed)

            if return_lr:
                return cost, lr

            return cost

        if rewards is not None:
            input_feed[self.rewards.name] = rewards
            output_feed = [
                self.updates_rewards, self.loss_rewards,
                self.current_learning_rate]
            _, cost, lr = sess.run(output_feed, input_feed)

            if return_lr:
                return cost, lr
            return cost

        output_feed = [
            self.updates, self.loss,
            self.current_learning_rate]
        _, cost, lr = sess.run(output_feed, input_feed)

        if return_lr:
            return cost, lr

        return cost


    def get_encoder_embedding(self, sess, encoder_inputs):
        """get the encoder_inputs from the embedding"""
        input_feed = {
            self.encoder_inputs.name: encoder_inputs
        }
        emb = sess.run(self.encoder_inputs_embedded, input_feed)
        return emb


    def entropy(self, sess, encoder_inputs, encoder_inputs_length,
                decoder_inputs, decoder_inputs_length):
        """Get entropy for a set of input and output,
        which is equal to calculate the P(target|source)
        """
        input_feed = self.check_feeds(
            encoder_inputs, encoder_inputs_length,
            decoder_inputs, decoder_inputs_length,
            False
        )
        input_feed[self.keep_prob_placeholder.name] = 1.0
        output_feed = [self.train_entropy, self.decoder_pred_train]
        entropy, logits = sess.run(output_feed, input_feed)
        return entropy, logits


    def predict(self, sess,
                encoder_inputs,
                encoder_inputs_length,
                attention=False):
        """predict the output"""

        # input
        input_feed = self.check_feeds(encoder_inputs,
                                      encoder_inputs_length, None, None, True)

        input_feed[self.keep_prob_placeholder.name] = 1.0

        # Attention output
        if attention:

            assert not self.use_beamsearch_decode, \
                'Attention cannot use BeamSearch'

            pred, atten = sess.run([
                self.decoder_pred_decode,
                self.final_state.alignment_history.stack()
            ], input_feed)

            return pred, atten

        # output with BeamSearch 
        if self.use_beamsearch_decode:
            pred, beam_prob = sess.run([
                self.decoder_pred_decode, self.beam_prob
            ], input_feed)
            beam_prob = np.mean(beam_prob, axis=1)

            pred = pred[0]
            return pred

        # (Greedy)output
        pred, = sess.run([
            self.decoder_pred_decode
        ], input_feed)

        return pred

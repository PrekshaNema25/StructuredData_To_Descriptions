
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

# We disable pylint because we need python3 compatibility.
from six.moves import xrange  # pylint: disable=redefined-builtin
from six.moves import zip     # pylint: disable=redefined-builtin

from tensorflow.python.framework import dtypes
from tensorflow.python.framework import ops
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import control_flow_ops
from tensorflow.python.ops import embedding_ops
from tensorflow.python.ops import math_ops
from tensorflow.python.ops import nn_ops
from . import rnn
#from tensorflow.python.ops import rnn
from tensorflow.python.ops import variable_scope
from tensorflow.python.util import nest
from . import rnn_cell_eval
#from tensorflow.python.ops import rnn_cell
from .basics import *
import copy

""" Vanilla-Attend-Decode model will have only document attention
(no query as an input), neither the distraction. We will build on top
of this the other models
"""


# TODO(ebrevdo): Remove once _linear is fully deprecated.
linear = rnn_cell._linear  # pylint: disable=protected-access

def _extract_argmax_and_embed(embedding, output_projection=None,
                              update_embedding=True):
  """Get a loop_function that extracts the previous symbol and embeds it.

  Args:
    embedding: embedding tensor for symbols.
    output_projection: None or a pair (W, B). If provided, each fed previous
      output will first be multiplied by W and added B.
    update_embedding: Boolean; if False, the gradients will not propagate
      through the embeddings.

  Returns:
    A loop function.
  """
  def loop_function(prev, _):
    if output_projection is not None:
      prev = nn_ops.xw_plus_b(
          prev, output_projection[0], output_projection[1])
    prev_symbol = math_ops.argmax(prev, 1)
    # Note that gradients will not propagate through the second parameter of
    # embedding_lookup.
    emb_prev = embedding_ops.embedding_lookup(embedding, prev_symbol)
    if not update_embedding:
      emb_prev = array_ops.stop_gradient(emb_prev)
    return emb_prev
  return loop_function



def sequence_loss_by_example(logits, targets, weights,
                             average_across_timesteps=True,
                             softmax_loss_function=None, name=None):
  """Weighted cross-entropy loss for a sequence of logits (per example).

  Args:
    logits: List of 2D Tensors of shape [batch_size x num_decoder_symbols].
    targets: List of 1D batch-sized int32 Tensors of the same length as logits.
    weights: List of 1D batch-sized float-Tensors of the same length as logits.
    average_across_timesteps: If set, divide the returned cost by the total
      label weight.
    softmax_loss_function: Function (inputs-batch, labels-batch) -> loss-batch
      to be used instead of the standard softmax (the default if this is None).
    name: Optional name for this operation, default: "sequence_loss_by_example".

  Returns:
    1D batch-sized float Tensor: The log-perplexity for each sequence.

  Raises:
    ValueError: If len(logits) is different from len(targets) or len(weights).
  """
  if len(targets) != len(logits) or len(weights) != len(logits):
    raise ValueError("Lengths of logits, weights, and targets must be the same "
                     "%d, %d, %d." % (len(logits), len(weights), len(targets)))
  with ops.op_scope(logits + targets + weights, name,
                    "sequence_loss_by_example"):
    log_perp_list = []
    for logit, target, weight in zip(logits, targets, weights):
      if softmax_loss_function is None:
        # TODO(irving,ebrevdo): This reshape is needed because
        # sequence_loss_by_example is called with scalars sometimes, which
        # violates our general scalar strictness policy.
        target = array_ops.reshape(target, [-1])
        crossent = nn_ops.sparse_softmax_cross_entropy_with_logits(
            logit, target)
      else:
        crossent = softmax_loss_function(logit, target)
      log_perp_list.append(crossent * weight)
    log_perps = math_ops.add_n(log_perp_list)
    if average_across_timesteps:
      total_size = math_ops.add_n(weights)
      total_size += 1e-12  # Just to avoid division by 0 for all-0 weights.
      log_perps /= total_size
  return log_perps


def sequence_loss(logits, targets, weights,
                  average_across_timesteps=True, average_across_batch=True,
                  softmax_loss_function=None, name=None):
  """Weighted cross-entropy loss for a sequence of logits, batch-collapsed.

  Args:
    logits: List of 2D Tensors of shape [batch_size x num_decoder_symbols].
    targets: List of 1D batch-sized int32 Tensors of the same length as logits.
    weights: List of 1D batch-sized float-Tensors of the same length as logits.
    average_across_timesteps: If set, divide the returned cost by the total
      label weight.
    average_across_batch: If set, divide the returned cost by the batch size.
    softmax_loss_function: Function (inputs-batch, labels-batch) -> loss-batch
      to be used instead of the standard softmax (the default if this is None).
    name: Optional name for this operation, defaults to "sequence_loss".

  Returns:
    A scalar float Tensor: The average log-perplexity per symbol (weighted).

  Raises:
    ValueError: If len(logits) is different from len(targets) or len(weights).
  """
  with ops.op_scope(logits + targets + weights, name, "sequence_loss"):
    cost = math_ops.reduce_sum(sequence_loss_by_example(
        logits, targets, weights,
        average_across_timesteps=average_across_timesteps,
        softmax_loss_function=softmax_loss_function))
    if average_across_batch:
      batch_size = array_ops.shape(targets[0])[0]
      return cost / math_ops.cast(batch_size, dtypes.float32)
    else:
      return cost


def vad_decoder(decoder_inputs,
                      initial_state,
                      distract_initial_state,
                      attention_states,
                      attention_states_fields,
                      attention_states_query,
                      cell,
                      distraction_cell,
                      output_size=None,
                      num_heads=1,
                      loop_function=None,
                      dtype=None,
                      scope=None,
                      initial_state_attention=False):
  """RNN decoder with attention for the sequence-to-sequence model.

  In this context "attention" means that, during decoding, the RNN can look up
  information in the additional tensor attention_states, and it does this by
  focusing on a few entries from the tensor. This model has proven to yield
  especially good results in a number of sequence-to-sequence tasks. This
  implementation is based on http://arxiv.org/abs/1412.7449 (see below for
  details). It is recommended for complex sequence-to-sequence tasks.

  Args:
    decoder_inputs: A list of 2D Tensors [batch_size x input_size].
    initial_state: 2D Tensor [batch_size x cell.state_size].
    attention_states: 3D Tensor [batch_size x attn_length x attn_size].
    cell: rnn_cell.RNNCell defining the cell function and size.
    output_size: Size of the output vectors; if None, we use cell.output_size.
    num_heads: Number of attention heads that read from attention_states.
    loop_function: If not None, this function will be applied to i-th output
      in order to generate i+1-th input, and decoder_inputs will be ignored,
      except for the first element ("GO" symbol). This can be used for decoding,
      but also for training to emulate http://arxiv.org/abs/1506.03099.
      Signature -- loop_function(prev, i) = next
        * prev is a 2D Tensor of shape [batch_size x output_size],
        * i is an integer, the step number (when advanced control is needed),
        * next is a 2D Tensor of shape [batch_size x input_size].
    dtype: The dtype to use for the RNN initial state (default: tf.float32).
    scope: VariableScope for the created subgraph; default: "attention_decoder".
    initial_state_attention: If False (default), initial attentions are zero.
      If True, initialize the attentions from the initial state and attention
      states -- useful when we wish to resume decoding from a previously
      stored decoder state and attention states.

  Returns:
    A tuple of the form (outputs, state), where:
      outputs: A list of the same length as decoder_inputs of 2D Tensors of
        shape [batch_size x output_size]. These represent the generated outputs.
        Output i is computed from input i (which is either the i-th element
        of decoder_inputs or loop_function(output {i-1}, i)) as follows.
        First, we run the cell on a combination of the input and previous
        attention masks:
          cell_output, new_state = cell(linear(input, prev_attn), prev_state).
        Then, we calculate new attention masks:
          new_attn = softmax(V^T * tanh(W * attention_states + U * new_state))
        and then we calculate the output:
          output = linear(cell_output, new_attn).
      state: The state of each decoder cell the final time-step.
        It is a 2D Tensor of shape [batch_size x cell.state_size].

  Raises:
    ValueError: when num_heads is not positive, there are no inputs, shapes
      of attention_states are not set, or input size cannot be inferred
      from the input.
  """
  if not decoder_inputs:
    raise ValueError("Must provide at least 1 input to attention decoder.")
  if num_heads < 1:
    raise ValueError("With less than 1 heads, use a non-attention decoder.")
  if attention_states.get_shape()[2].value is None:
    raise ValueError("Shape[2] of attention_states must be known: %s"
                     % attention_states.get_shape())
  if output_size is None:
    output_size = cell.output_size

  with variable_scope.variable_scope(
      scope or "vad_decoder", dtype=dtype) as scope:
    dtype = scope.dtype

    batch_size = array_ops.shape(decoder_inputs[0])[0]  # Needed for reshaping.
    attn_length_state = attention_states.get_shape()[1].value
    attn_length_field_state = attention_states_fields.get_shape()[1].value

    dim_1 = initial_state.get_shape()[1].value
    dim_2 = cell.output_size

    project_initial_state_W = variable_scope.get_variable("Initial_State_W", [dim_1, dim_2])
    project_initial_state_B = variable_scope.get_variable("Initial_State_Bias", [dim_2])

    distract_state = [distract_initial_state, distract_initial_state]

    print ("Preksha " + scope.name)
    if attn_length_state is None:
      attn_length_state = attention_states.shape()[1]

    attn_size_state = attention_states.get_shape()[2].value


    if attn_length_field_state is None:
      attn_length_field_state = attention_states_fields.shape()[1]


    attn_size_field_state = attention_states_fields.get_shape()[2].value

    # To calculate W1 * h_t we use a 1-by-1 convolution, need to reshape before.
    hidden_states = array_ops.reshape(
        attention_states, [-1, attn_length_state, 1, attn_size_state])

    hidden_field_states = array_ops.reshape(
      attention_states_fields, [-1, attn_length_field_state, 1, attn_size_field_state])

    hidden_features_states = []
    hidden_features_field_states = []

    v_state = []
    v_field_state = []
    attention_vec_size_state  = attn_size_state  # Size of query vectors for attention.
    attention_vec_size_field_state = attn_size_field_state

    for a in xrange(num_heads):
      k = variable_scope.get_variable("AttnW_State_%d" % a,
                                      [1, 1, attn_size_state, attention_vec_size_state])

      hidden_features_states.append(nn_ops.conv2d(hidden_states, k, [1, 1, 1, 1], "SAME"))

      v_state.append(
          variable_scope.get_variable("AttnV_State_%d" % a, [attention_vec_size_state]))


    for a in xrange(num_heads):
      k = variable_scope.get_variable("AttnW_State_F_%d" % a,
                                      [1, 1, attn_size_field_state, attention_vec_size_field_state])

      hidden_features_field_states.append(nn_ops.conv2d(hidden_field_states, k, [1, 1, 1, 1], "SAME"))

      v_field_state.append(
          variable_scope.get_variable("AttnV_State_F_%d" % a, [attention_vec_size_field_state]))


    state = math_ops.matmul(initial_state, project_initial_state_W) + project_initial_state_B

    def attention_t(query, name):
      """Put attention masks on hidden using hidden_features and query."""
      ds = []  # Results of attention reads will be stored here.
      if nest.is_sequence(query):  # If the query is a tuple, flatten it.
        query_list = nest.flatten(query)
        for q in query_list:  # Check that ndims == 2 if specified.
          ndims = q.get_shape().ndims
          if ndims:
            assert ndims == 2
        query = array_ops.concat(1, query_list)
      for a in xrange(num_heads):
        with variable_scope.variable_scope("Attention_" +  name):
          y = linear(query, attention_vec_size_state, True)
          y = array_ops.reshape(y, [-1, 1, 1, attention_vec_size_state])
          # Attention mask is a softmax of v^T * tanh(...).
          s = math_ops.reduce_sum(
              v_state[a] * math_ops.tanh(hidden_features_states[a] + y), [2, 3])
          a = nn_ops.softmax(s)
          # Now calculate the attention-weighted vector d.
          d = math_ops.reduce_sum(
              array_ops.reshape(a, [-1, attn_length_state, 1, 1]) * hidden_states,
              [1, 2])
          ds.append(array_ops.reshape(d, [-1, attn_size_state]))
      return ds, a



    def attention_f(query, name):
      """Put attention masks on hidden using hidden_features and query."""
      ds = []  # Results of attention reads will be stored here.
      if nest.is_sequence(query):  # If the query is a tuple, flatten it.
        query_list = nest.flatten(query)
        for q in query_list:  # Check that ndims == 2 if specified.
          ndims = q.get_shape().ndims
          if ndims:
            assert ndims == 2
        query = array_ops.concat(1, query_list)
      for a in xrange(num_heads):
        with variable_scope.variable_scope("Attention_f" +  name):
          y = linear(query, attention_vec_size_field_state, True)
          y = array_ops.reshape(y, [-1, 1, 1, attention_vec_size_field_state])
          # Attention mask is a softmax of v^T * tanh(...).
          s = math_ops.reduce_sum(
              v_field_state[a] * math_ops.tanh(hidden_features_field_states[a] + y), [2, 3])
          a = nn_ops.softmax(s)
          # Now calculate the attention-weighted vector d.
          d = math_ops.reduce_sum(
              array_ops.reshape(a, [-1, attn_length_field_state, 1, 1]) * hidden_field_states,
              [1, 2])
          ds.append(array_ops.reshape(d, [-1, attn_size_field_state]))
      return ds, a

    outputs = []
    prev = None
    attention_weights = []
    attention_weights_fields = []
    batch_attn_size_state = array_ops.pack([batch_size, attn_size_state])
    batch_attn_size_field_state = array_ops.pack([batch_size, attn_size_field_state])

    new_attns_state = [array_ops.zeros(batch_attn_size_state, dtype=dtype)
             for _ in xrange(num_heads)]


    attns_state_fields = [array_ops.zeros(batch_attn_size_field_state, dtype=dtype)
             for _ in xrange(num_heads)]

    for a in new_attns_state:  # Ensure the second shape of attention vectors is set.
      a.set_shape([None, attn_size_state])


    if initial_state_attention:
      new_attns_state = attention_t(initial_state, "alpha")


    for i, inp in enumerate(decoder_inputs):
      if i > 0:
        variable_scope.get_variable_scope().reuse_variables()
      # If loop_function is set, we use it instead of decoder_inputs.
      if loop_function is not None and prev is not None:
        with variable_scope.variable_scope("loop_function", reuse=True):
          inp = loop_function(prev, i)
      # Merge input and previous attentions into one vector of the right size.
      input_size = inp.get_shape().with_rank(2)[1]
      if input_size.value is None:
        raise ValueError("Could not infer input size from input: %s" % inp.name)

      print ("inp", inp.get_shape(), new_attns_state[0].get_shape())

      attn_fields = attns_state_fields[0]
      distract_output, distract_state = distraction_cell(attn_fields, distract_state)
    
      x = linear([inp] + new_attns_state + [distract_output], input_size, True)

      # Run the RNN.
      print (x.get_shape())
      cell_output, state = cell(x, state)

      print ("cell_output", cell_output.get_shape())
      # Run the attention mechanism.

      if i == 0 and initial_state_attention:
        with variable_scope.variable_scope(variable_scope.get_variable_scope(),
                                           reuse=True):
          attns_state_tokens, attn_vec_tokens = attention_t(state, "alpha")
          attns_state_fields, attn_vec_fields = attention_f(state, "beta")
      else:
        attns_state_tokens, attn_vec_tokens = attention_t(state, "alpha")
        attns_state_fields, attn_vec_fields = attention_f(state, "beta")


      # 15: Number of fields per box
      attn_vec_tokens = array_ops.reshape(attn_vec_tokens, [batch_size, 15, -1])
      #new_attn_weights = math_ops.mul(attn_vec_tokens, attn_vec_fields)

      temp = []
      for i in range(5):
	 temp.append(attn_vec_fields)

      temp = array_ops.pack(temp, axis=2)
      new_attn_weights = attn_vec_tokens * temp
      new_attn_weights = array_ops.reshape(new_attn_weights, [batch_size, -1])
      new_attn_weights_norm = math_ops.reduce_sum(new_attn_weights, reduction_indices = 1) + 1e-14
      new_attn_weights_norm = array_ops.reshape(new_attn_weights_norm, [-1,1])
      new_attn_weights = math_ops.div(new_attn_weights, new_attn_weights_norm)


      new_attns_state = math_ops.reduce_sum(
                        array_ops.reshape(new_attn_weights, [-1, attn_length_state, 1, 1]) * hidden_states, [1,2])

      new_attns_state = array_ops.reshape(new_attns_state, [-1, attn_size_state])

      new_attns_state = [new_attns_state]
      with variable_scope.variable_scope("AttnOutputProjection"):

        output = linear([cell_output] + new_attns_state + attns_state_fields, output_size, True)
        #x_shape = variable_scope.get_variable(name = 'x_shape',shape=cell_output.get_shape())

	print ("output_size", output.get_shape())
        if loop_function is not None:
          prev = output
        outputs.append(output)
	attention_weights.append(new_attn_weights)
        attention_weights_fields.append(attn_vec_fields)

        print ("Attn_vec_fields", attn_vec_fields)
        print ("len of outputs", len(outputs), len(attention_weights), len(attention_weights_fields))
  return outputs, state, attention_weights, attention_weights_fields

def vad_decoder_wrapper(decoder_inputs,
                                initial_state,
                                distract_initial_state,
                                attention_states,
                                attention_states_fields,
                                attention_states_query,
                                cell_encoder,
                                distraction_cell,
                                num_symbols,
                                embedding_size,
                                num_heads=1,
                                output_size=None,
                                output_projection=None,
                                feed_previous=False,
                                update_embedding_for_previous=True,
                                embedding_scope = None,
                                dtype=None,
                                scope=None,
                                initial_state_attention=False):
  """RNN decoder with embedding and attention and a pure-decoding option.

  Args:
    decoder_inputs: A list of 1D batch-sized int32 Tensors (decoder inputs).
    initial_state: 2D Tensor [batch_size x cell.state_size].
    attention_states: 3D Tensor [batch_size x attn_length x attn_size].
    cell: rnn_cell.RNNCell defining the cell function.
    num_symbols: Integer, how many symbols come into the embedding.
    embedding_size: Integer, the length of the embedding vector for each symbol.
    num_heads: Number of attention heads that read from attention_states.
    output_size: Size of the output vectors; if None, use output_size.
    output_projection: None or a pair (W, B) of output projection weights and
      biases; W has shape [output_size x num_symbols] and B has shape
      [num_symbols]; if provided and feed_previous=True, each fed previous
      output will first be multiplied by W and added B.
    feed_previous: Boolean; if True, only the first of decoder_inputs will be
      used (the "GO" symbol), and all other decoder inputs will be generated by:
        next = embedding_lookup(embedding, argmax(previous_output)),
      In effect, this implements a greedy decoder. It can also be used
      during training to emulate http://arxiv.org/abs/1506.03099.
      If False, decoder_inputs are used as given (the standard decoder case).
    update_embedding_for_previous: Boolean; if False and feed_previous=True,
      only the embedding for the first symbol of decoder_inputs (the "GO"
      symbol) will be updated by back propagation. Embeddings for the symbols
      generated from the decoder itself remain unchanged. This parameter has
      no effect if feed_previous=False.
    dtype: The dtype to use for the RNN initial states (default: tf.float32).
    scope: VariableScope for the created subgraph; defaults to
      "embedding_attention_decoder".
    initial_state_attention: If False (default), initial attentions are zero.
      If True, initialize the attentions from the initial state and attention
      states -- useful when we wish to resume decoding from a previously
      stored decoder state and attention states.

  Returns:
    A tuple of the form (outputs, state), where:
      outputs: A list of the same length as decoder_inputs of 2D Tensors with
        shape [batch_size x output_size] containing the generated outputs.
      state: The state of each decoder cell at the final time-step.
        It is a 2D Tensor of shape [batch_size x cell.state_size].

  Raises:
    ValueError: When output_projection has the wrong shape.
  """
  if output_size is None:
    output_size = cell_encoder.output_size
  if output_projection is not None:
    proj_biases = ops.convert_to_tensor(output_projection[1], dtype=dtype)
    proj_biases.get_shape().assert_is_compatible_with([num_symbols])


  with variable_scope.variable_scope(
    embedding_scope or "vad_decoder_wrapper", dtype=dtype,  reuse = True) as s1:

    print ("Preksha", s1.name)
    embedding = variable_scope.get_variable("embedding",
                                            [num_symbols, embedding_size])
    loop_function = _extract_argmax_and_embed(
        embedding, output_projection,
        update_embedding_for_previous) if feed_previous else None
    emb_inp = [
        embedding_ops.embedding_lookup(embedding, i) for i in decoder_inputs]

  with variable_scope.variable_scope(
    scope or "vad_decoder_wrapper", dtype =dtype) as scope:
    return vad_decoder(
        emb_inp,
        initial_state=initial_state,
        attention_states_query = attention_states_query,
        attention_states_fields = attention_states_fields,
        attention_states=attention_states,
        cell = cell_encoder,
        distract_initial_state = distract_initial_state,
        distraction_cell = distraction_cell,
        output_size=output_size,
        num_heads=num_heads,
        loop_function=loop_function,
        initial_state_attention=initial_state_attention)


def vad_seq2seq(encoder_inputs,
                                decoder_inputs,
                                query_inputs,
                                field_inputs,
                                cell_encoder_fw,
                                cell_encoder_bw,
                                distraction_cell,
                                sequence_field_length, 
                                num_encoder_symbols,
                                num_decoder_symbols,
                                embedding_size,
                                initial_embedding = None,
                                num_heads=1,
                                embedding_trainable=False,
                                output_projection=None,
                                feed_previous=False,
                                dtype=None,
                                scope=None,
                                initial_state_attention=False):
  """Embedding sequence-to-sequence model with attention.

  This model first embeds encoder_inputs by a newly created embedding (of shape
  [num_encoder_symbols x input_size]). Then it runs an RNN to encode
  embedded encoder_inputs into a state vector. It keeps the outputs of this
  RNN at every step to use for attention later. Next, it embeds decoder_inputs
  by another newly created embedding (of shape [num_decoder_symbols x
  input_size]). Then it runs attention decoder, initialized with the last
  encoder state, on embedded decoder_inputs and attending to encoder outputs.

  Args:
    encoder_inputs: A list of 1D int32 Tensors of shape [batch_size].
    decoder_inputs: A list of 1D int32 Tensors of shape [batch_size].
    cell: rnn_cell.RNNCell defining the cell function and size.
    num_encoder_symbols: Integer; number of symbols on the encoder side.
    num_decoder_symbols: Integer; number of symbols on the decoder side.
    embedding_size: Integer, the length of the embedding vector for each symbol.
    num_heads: Number of attention heads that read from attention_states.
    output_projection: None or a pair (W, B) of output projection weights and
      biases; W has shape [output_size x num_decoder_symbols] and B has
      shape [num_decoder_symbols]; if provided and feed_previous=True, each
      fed previous output will first be multiplied by W and added B.
    feed_previous: Boolean or scalar Boolean Tensor; if True, only the first
      of decoder_inputs will be used (the "GO" symbol), and all other decoder
      inputs will be taken from previous outputs (as in embedding_rnn_decoder).
      If False, decoder_inputs are used as given (the standard decoder case).
    dtype: The dtype of the initial RNN state (default: tf.float32).
    scope: VariableScope for the created subgraph; defaults to
      "embedding_attention_seq2seq".
    initial_state_attention: If False (default), initial attentions are zero.
      If True, initialize the attentions from the initial state and attention
      states.

  Returns:
    A tuple of the form (outputs, state), where:
      outputs: A list of the same length as decoder_inputs of 2D Tensors with
        shape [batch_size x num_decoder_symbols] containing the generated
        outputs.
      state: The state of each decoder cell at the final time-step.
        It is a 2D Tensor of shape [batch_size x cell.state_size].
  """
  with variable_scope.variable_scope(
      scope or "vad_seq2seq", dtype=dtype) as scope:
    dtype = scope.dtype
    # Encoder.
    """encoder_cell = rnn_cell.EmbeddingWrapper(
        cell, embedding_classes=num_encoder_symbols,
        embedding_size=embedding_size)
    """
    if initial_embedding is not None:
      embedding = variable_scope.get_variable('embedding',
            initializer=initial_embedding, trainable=embedding_trainable)

    else:
      embedding = variable_scope.get_variable('embedding', [num_encoder_symbols, embedding_size],trainable=embedding_trainable)


    embedded_inputs = embedding_ops.embedding_lookup(embedding, encoder_inputs)


    embedded_inputs = array_ops.reshape(embedded_inputs, [15, 5, -1, embedding_size])
    embedded_inputs = array_ops.unpack(embedded_inputs)

   
    embedded_fields = embedding_ops.embedding_lookup(embedding, field_inputs)

    embedded_fields = array_ops.unpack(embedded_fields)

    print ("Embedded Inputs length:", len(embedded_inputs))

    print("Shape in embedded inputs:", embedded_inputs[0].get_shape())

    print("Embedded fields:", embedded_fields[0].get_shape())


    print ("Sequence length:", len(sequence_field_length))
    initial_state_fw = None
    initial_state_bw = None
    encoder_outputs = []
    field_states = []
    with variable_scope.variable_scope("Encoder_Cell"):
      for i in range(15):

	  if (i>0):
             variable_scope.get_variable_scope().reuse_variables()

          embedded_temp_inputs = embedded_inputs[i]
          embedded_temp_inputs = array_ops.transpose(embedded_temp_inputs, perm=[1,0,2])
	  print("size of each input", embedded_temp_inputs.get_shape(), sequence_field_length[0].get_shape())
 
	  temp_outputs_list = rnn.bidirectional_dynamic_rnn(cell_encoder_fw, 
		cell_encoder_bw, embedded_temp_inputs, sequence_length=sequence_field_length[i], initial_state_fw = initial_state_fw, 
		initial_state_bw= initial_state_bw, dtype=dtype)

	  initial_state_fw = temp_outputs_list[1][0]
          initial_state_bw = temp_outputs_list[1][1]
   
          temp_outputs = array_ops.concat(2, temp_outputs_list[0])
          temp_outputs = array_ops.unpack(temp_outputs, axis=1)
	  field_states.append(array_ops.concat(1, [initial_state_fw, initial_state_bw]))
	  for j in range(5):
		encoder_outputs.append(temp_outputs[j])


    print ("Encoder outputs", len(encoder_outputs), encoder_outputs[0].get_shape())
    #with variable_scope.variable_scope("Encoder_Cell"):
    #  encoder_outputs, encoder_state_fw, encoder_state_bw = rnn.bidirectional_rnn(
    #      cell_encoder_fw, cell_encoder_bw, embedded_inputs, dtype=dtype)

    # First calculate a concatenation of encoder outputs to put attention on.

    encoder_state = array_ops.concat(1, [initial_state_fw, initial_state_bw])

    top_states_encoder = [array_ops.reshape(e, [-1, 1, 2*cell_encoder_fw.output_size])
                          for e in encoder_outputs]
    attention_states_encoder = array_ops.concat(1, top_states_encoder)




    #count = 1
    #for (i, es) in enumerate(encoder_outputs):
      # Tokens per field = 5
    #  if (count % 5 == 0):
    #    field_states.append(es)

    #  count = count + 1

    field_inputs = []

    for i in range(len(field_states)):
      field_inputs.append(array_ops.concat(1, [field_states[i], embedded_fields[i]]))
    # Decoder.

    print ("field_inputs", len(field_inputs), field_inputs[0].get_shape())
    with variable_scope.variable_scope("Field_Cell"):

      field_outputs, field_encoder_fw, field_encoder_bw = rnn.bidirectional_rnn(
        cell_encoder_fw, cell_encoder_bw, field_inputs, dtype=dtype)


    field_initial_state = array_ops.concat(1, [field_encoder_fw, field_encoder_bw])

    top_states_field_encoder = [array_ops.reshape(e, [-1, 1, 2*cell_encoder_fw.output_size])
                                for e in field_outputs]

    attention_states_fields = array_ops.concat(1, top_states_field_encoder)

    output_size = None
    if output_projection is None:
      cell_encoder_fw = rnn_cell.OutputProjectionWrapper(cell_encoder_fw, num_decoder_symbols)
      output_size = num_decoder_symbols

    if isinstance(feed_previous, bool):
      return vad_decoder_wrapper(
          decoder_inputs,
          initial_state=field_initial_state,
          attention_states=attention_states_encoder,
          attention_states_fields = attention_states_fields,
          attention_states_query = None,
          cell_encoder = cell_encoder_fw,
          num_symbols = num_decoder_symbols,
          embedding_size = embedding_size,
          distract_initial_state = field_initial_state,
          distraction_cell = distraction_cell,
          num_heads=num_heads,
          output_size=output_size,
          output_projection=output_projection,
          feed_previous=feed_previous,
          embedding_scope = scope,
          initial_state_attention=initial_state_attention)

    # If feed_previous is a Tensor, we construct 2 graphs and use cond.
    def decoder(feed_previous_bool):

      reuse = None if feed_previous_bool else True

      with variable_scope.variable_scope(
          variable_scope.get_variable_scope(), reuse=reuse) as scope:

        outputs, state, attention_weights, attention_weights_fields = vad_decoder_wrapper(
            decoder_inputs,
            initial_state=field_initial_state,
            attention_states=attention_states_encoder,
            attention_states_fields = attention_states_fields,
            attention_states_query = None,
            cell_encoder = cell_encoder_fw,
            num_symbols=num_decoder_symbols,
            embedding_size = embedding_size,
            distract_initial_state = field_initial_state,
            distraction_cell = distraction_cell,
            num_heads=num_heads,
            output_size=output_size,
            output_projection=output_projection,
            feed_previous=feed_previous_bool,
            embedding_scope = scope,
            update_embedding_for_previous=False,
            initial_state_attention=initial_state_attention)

        state_list = [state]
        if nest.is_sequence(state):
          state_list = nest.flatten(state)

	print ("attention_weights", len(attention_weights))
        return outputs + state_list + attention_weights + attention_weights_fields

    outputs_and_state = control_flow_ops.cond(feed_previous,
                                              lambda: decoder(True),
                                              lambda: decoder(False))
    outputs_len = len(decoder_inputs)  # Outputs length same as decoder inputs.
    print ("decoder input length", outputs_len)
    state_list = outputs_and_state[outputs_len:]
    attention_weights_fields = outputs_and_state[-outputs_len:]
    attention_weights = outputs_and_state[-(2*outputs_len):-outputs_len]
    state = state_list[0]
    if nest.is_sequence(encoder_state):
      state = nest.pack_sequence_as(structure=encoder_state,
                                    flat_sequence=state_list)
    return outputs_and_state[:outputs_len], state, attention_weights, attention_weights_fields


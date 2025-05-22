# This script implements a scaled dot-product self-attention mechanism
# using NengoDL to integrate TensorFlow code within a Nengo simulation.
# The goal is to demonstrate how standard, mathematically precise attention
# can be used as a component in a Nengo model.

import nengo
import nengo_dl
import numpy as np
import tensorflow as tf

# --- Example Input Data ---
# Batch size = 1, Sequence length = 3, Model dimension = 4
X = np.array([[[1, 0, 1, 0],
               [0, 1, 0, 1],
               [1, 1, 1, 1]]], dtype=np.float32)

# --- Dimension Variables ---
d_model = X.shape[-1] # Dimension of the model (input/output)
d_k = 3  # Dimension of Key/Query vectors
d_v = 3  # Dimension of Value vectors

# --- Weight Initialization ---
# Initialize weights for Q, K, V transformations
W_q_np = np.random.rand(d_model, d_k).astype(np.float32)
W_k_np = np.random.rand(d_model, d_k).astype(np.float32)
W_v_np = np.random.rand(d_model, d_v).astype(np.float32)

# Convert NumPy arrays to TensorFlow Variables
Wq_tf = tf.Variable(W_q_np, name="Wq")
Wk_tf = tf.Variable(W_k_np, name="Wk")
Wv_tf = tf.Variable(W_v_np, name="Wv")

# --- TensorFlow Function for a Single Attention Step (for TensorNode) ---
# This function implements the standard mathematical operations for self-attention
# using TensorFlow. It's designed to be wrapped by a NengoDL TensorNode.
# The tf.Variables Wq_tf, Wk_tf, Wv_tf are defined in the global scope
# and will be automatically captured by the TensorNode.
@tf.function
def attention_step(t, x): # Signature changed for TensorNode (t, x)
    """
    Calculates the attention output for a given input sequence `x` at Nengo time `t`.
    This function performs the core math of scaled dot-product attention.
    x: Input tensor (sequence_length, d_model) - batch dimension is handled by NengoDL.
    Wq_tf, Wk_tf, Wv_tf: Weight matrices for Q, K, V (captured from global scope).
    """
    # Reshape x to include a batch dimension (1) for TensorFlow's matmul operations,
    # as the original X was (batch_size, seq_len, d_model) and TensorNode
    # feeds it as (seq_len, d_model) per Nengo timestep.
    x_batched = tf.reshape(x, (1, tf.shape(x)[0], tf.shape(x)[1])) # (1, sequence_length, d_model)

    # 1. Project inputs to Query, Key, and Value spaces:
    # Q = X * Wq (Query)
    # K = X * Wk (Key)
    # V = X * Wv (Value)
    Q = tf.matmul(x_batched, Wq_tf)  # Shape: (1, sequence_length, d_k)
    K = tf.matmul(x_batched, Wk_tf)  # Shape: (1, sequence_length, d_k)
    V = tf.matmul(x_batched, Wv_tf)  # Shape: (1, sequence_length, d_v)

    # 2. Calculate Attention Scores (Scaled Dot-Product):
    # scores = (Q * K^T) / sqrt(d_k)
    # This measures the similarity between each query and all keys.
    matmul_qk = tf.matmul(Q, K, transpose_b=True)  # Shape: (1, sequence_length, sequence_length)
    dk_float = tf.cast(tf.shape(K)[-1], tf.float32) # Dimension of keys, d_k
    scaled_attention_logits = matmul_qk / tf.math.sqrt(dk_float) # Scaling prevents extreme softmax gradients

    # 3. Apply Softmax to get Attention Weights:
    # weights = softmax(scores)
    # This normalizes the scores so they sum to 1, forming attention weights.
    attention_weights = tf.nn.softmax(scaled_attention_logits, axis=-1)  # Shape: (1, sequence_length, sequence_length)

    # 4. Calculate the Output (weighted sum of Value vectors):
    # output = weights * V
    # Each output element is a weighted average of the values, based on attention.
    output = tf.matmul(attention_weights, V)  # Shape: (1, sequence_length, d_v)

    # TensorNode expects output matching its 'shape_out' (sequence_length, d_v),
    # so we remove the temporary batch dimension.
    return tf.reshape(output, (tf.shape(output)[1], tf.shape(output)[2])) # (sequence_length, d_v)

# --- Conceptual Discussion: NengoDL for Self-Attention ---
#
# Why NengoDL for Self-Attention?
# Implementing standard self-attention (as defined in models like Transformers)
# requires precise mathematical operations: matrix multiplications, scaling, and softmax.
#
# 1. Precision and Pragmatism:
#    NengoDL allows us to directly embed TensorFlow code (like `attention_step`)
#    into a Nengo model using `TensorNode`. This is pragmatic because it leverages
#    highly optimized TensorFlow operations for these calculations, ensuring mathematical
#    correctness and efficiency. This is crucial for users who want to use standard
#    attention as a functional block within a larger Nengo simulation, perhaps to
#    interface with neuromorphic components or to explore system-level interactions.
#
# 2. Challenges of Pure Nengo Implementation:
#    Building such mechanisms purely from core Nengo objects (e.g., ensembles of
#    spiking neurons, `nengo.Connection` with learned weights) is a significant
#    research challenge:
#    - Matrix Multiplication: While possible with neuron ensembles (e.g., using the
#      Product space in Nengo), achieving the precision of `tf.matmul` for large
#      matrices can be resource-intensive and may require many neurons.
#    - Softmax: Approximating a softmax function with spiking neurons is non-trivial
#      and often involves specialized network architectures or approximations.
#    - Scalability: Implementing complex operations like scaled dot-product attention
#      directly in neurons can lead to very large and complex Nengo networks.
#
# 3. This Implementation's Focus:
#    This script chooses NengoDL to provide a functionally accurate self-attention
#    module. It prioritizes getting the *exact mathematical behavior* of self-attention
#    into Nengo. This is distinct from research aimed at discovering how attention-like
#    mechanisms might arise from more biologically plausible neural dynamics.
#    Both approaches are valid, but serve different goals. Using NengoDL here is an
#    engineering choice for precision and ease of integration for this specific component.

# --- NengoDL TensorNode for Self-Attention ---
# The `nengodl.TensorNode` acts as a crucial bridge, allowing the TensorFlow-defined
# `attention_step` function to be seamlessly integrated into a Nengo network.
# - NengoDL handles the data exchange: It converts Nengo signals into TensorFlow tensors
#   for the input to `attention_step` and converts the tensor output back into Nengo signals.
# - Management of TensorFlow Variables: NengoDL automatically detects and manages any
#   `tf.Variable`s used within the TensorNode's function (like Wq_tf, Wk_tf, Wv_tf).
#   This means they can be trained using NengoDL's simulator if desired.

# Shape of the input to the TensorNode (sequence_length, d_model)
# Batching is handled by NengoDL simulator, so we define the shape for a single sequence.
sequence_length = X.shape[1]
shape_in_tensor_node = (sequence_length, d_model)

# Shape of the output from the TensorNode (sequence_length, d_v)
shape_out_tensor_node = (sequence_length, d_v)

# Create the TensorNode
# pass_input=True means the input 'x' to attention_step will be a tf.Tensor.
# If False (default), it would be a NumPy array.
attn_node = nengo_dl.TensorNode(
    attention_step,
    shape_in=shape_in_tensor_node,
    shape_out=shape_out_tensor_node,
    pass_input=True
)

# --- Constructing the Nengo Network ---
# We define a Nengo network to integrate the self-attention TensorNode.
# This network will provide input to the TensorNode and probe (record) its output.

net = nengo.Network(label="Self-Attention Network")
with net:
    # 1. Input Node (`nengo.Node`):
    # This node provides the input sequence X[0] (a single sequence from our batch X)
    # to the `attn_node`. X[0] has shape (sequence_length, d_model).
    # The lambda function outputs `input_sequence` only at the first timestep (t < dt, where dt is simulator timestep)
    # and then zeros. This is a common pattern for feeding a single, static input
    # to a TensorNode for one processing pass in a NengoDL simulation.
    input_sequence = X[0] # Example: (3, 4) -> (sequence_length, d_model)
    input_node = nengo.Node(
        output=lambda t: input_sequence if t < 0.001 else np.zeros_like(input_sequence), # Present input for one step
        label="input_sequence_node"
    )

    # 2. Connection from Input Node to TensorNode:
    # This connection feeds the output of `input_node` (our sequence X[0])
    # directly to the input of `attn_node` (the self-attention TensorNode).
    # `synapse=None` ensures the data is passed through without filtering.
    nengo.Connection(input_node, attn_node, synapse=None)

    # 3. Output Probe (`nengo.Probe`):
    # This probe records the output of the `attn_node` (the result of the
    # self-attention calculation). The probed data will allow us to inspect
    # what the attention mechanism computed.
    # The output shape will be (sequence_length, d_v).
    attention_output_probe = nengo.Probe(attn_node, label="attention_output_probe")

# --- Running the NengoDL Simulation ---
# We use the NengoDL simulator to run the network. NengoDL integrates the
# TensorFlow backend, allowing the TensorNode (and its underlying TensorFlow
# operations in `attention_step`) to execute within the Nengo simulation loop.

# The `dt` (timestep) of the simulator is set to 0.001s.
# Our `input_node` is designed to output the sequence for t < 0.001s.
# Therefore, running the simulation for a single step of 0.001s is sufficient
# to process this input.
with nengo_dl.Simulator(net, dt=0.001) as sim:
    # Run the simulation for 0.001 seconds.
    # Because `dt` is 0.001s, this corresponds to exactly one simulation step.
    # NengoDL's `unroll_simulation` parameter (default 1) means the TensorNode
    # processes its full input sequence (`shape_in`) once per simulator step.
    sim.run(0.001) # Run for one timestep

    # Retrieve the data recorded by the output probe.
    # `sim.data[probe]` contains a NumPy array of the probed values over simulation time.
    # Since we ran for one step, the data will have shape (1, sequence_length, d_v).
    output_data = sim.data[attention_output_probe]

# --- Output Verification ---
print("\n--- NengoDL Self-Attention Output ---")
# The `output_data` array has dimensions (num_timesteps, sequence_length, d_v).
# Given our single-step simulation, num_timesteps is 1.
# We print `output_data[0]` to see the attention result for our input sequence.
print("Shape of probed output data:", output_data.shape)
print("Output from attention mechanism (first/only timestep):")
if output_data.shape[0] > 0:
    print(output_data[0])
else:
    print("No output data collected.")

# The printed output represents the result of the self-attention mechanism
# applied to the input sequence X[0]. Each row in the output corresponds
# to a position in the input sequence, and the values are the weighted sum
# of the Value vectors, based on the attention scores.
# The shape should be (sequence_length, d_v), e.g., (3, 3) for our example.

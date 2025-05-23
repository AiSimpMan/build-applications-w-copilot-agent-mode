# Exploring partially spiking attention mechanisms using nengo-pytorch.
# This script will investigate how to integrate spiking neural components
# with rate-based components (e.g., from PyTorch's nn.Module)
# for attention-like computations.

import torch
import torch.nn as nn
import nengo
import nengo_pytorch
import numpy as np
import matplotlib.pyplot as plt

# --- Spiking Attention Head Module ---
class SpikingAttentionHead(nn.Module):
    """
    A single head of self-attention where the Query, Key, and Value
    transformations are followed by spiking neuron activations.
    The core attention mechanism (scaled dot-product) itself is calculated
    using standard PyTorch tensor operations.

    Args:
        d_model (int): Dimensionality of the input and output features.
        d_k (int): Dimensionality of the Key and Query vectors.
        d_v (int): Dimensionality of the Value vectors.
        spiking_neuron_type (nengo.neurons.NeuronType, optional):
            The type of Nengo neuron to use for spiking activations.
            Defaults to nengo.LIF().
    """
    def __init__(self, d_model, d_k, d_v, spiking_neuron_type=nengo.LIF()):
        super().__init__()
        self.d_k = d_k # Store d_k for the scaling factor in attention

        # Linear transformation for Query, followed by spiking activation
        self.w_q = nn.Linear(d_model, d_k, bias=False)
        # nengo_pytorch.SpikingActivation wraps Nengo neuron models for use in PyTorch.
        # It simulates the neurons for a fixed number of timesteps (default 1)
        # and returns their (potentially filtered) activity.
        self.act_q = nengo_pytorch.SpikingActivation(spiking_neuron_type)

        # Linear transformation for Key, followed by spiking activation
        self.w_k = nn.Linear(d_model, d_k, bias=False)
        self.act_k = nengo_pytorch.SpikingActivation(spiking_neuron_type)

        # Linear transformation for Value, followed by spiking activation
        self.w_v = nn.Linear(d_model, d_v, bias=False)
        self.act_v = nengo_pytorch.SpikingActivation(spiking_neuron_type)

        # Optional: A final linear projection for the output context
        # self.w_out = nn.Linear(d_v, d_model, bias=False)
        # We'll omit this for now for simplicity.

    def forward(self, x):
        """
        Forward pass of the SpikingAttentionHead.

        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, sequence_length, d_model).

        Returns:
            context (torch.Tensor): The context vector, output of the attention head,
                                   shape (batch_size, sequence_length, d_v).
            attn_weights (torch.Tensor): The attention weights,
                                       shape (batch_size, sequence_length, sequence_length).
        """
        # 1. Compute Query, Key, Value vectors with spiking activations
        # x shape: (batch_size, sequence_length, d_model)
        # q, k shapes: (batch_size, sequence_length, d_k)
        # v shape: (batch_size, sequence_length, d_v)
        q = self.act_q(self.w_q(x))
        k = self.act_k(self.w_k(x))
        v = self.act_v(self.w_v(x))

        # 2. Calculate Attention Scores (Scaled Dot-Product)
        # This part uses standard PyTorch tensor operations.
        # scores = (Q * K^T) / sqrt(d_k)
        # k.transpose(-2, -1) results in shape (batch_size, d_k, sequence_length)
        # scores shape: (batch_size, sequence_length, sequence_length)
        scores = torch.matmul(q, k.transpose(-2, -1)) / np.sqrt(self.d_k)

        # 3. Apply Softmax to get Attention Weights
        # attn_weights shape: (batch_size, sequence_length, sequence_length)
        attn_weights = torch.softmax(scores, dim=-1)

        # 4. Calculate the Output Context Vector
        # context = weights * V
        # context shape: (batch_size, sequence_length, d_v)
        context = torch.matmul(attn_weights, v)

        # Optional: Apply output projection
        # if hasattr(self, 'w_out'):
        #     context = self.w_out(context)

        return context, attn_weights

# --- Test and Verification of SpikingAttentionHead ---

# 1. Define Hyperparameters for the model
d_model = 4  # Dimensionality of input features
d_k = 3      # Dimensionality of Key and Query vectors
d_v = 3      # Dimensionality of Value vectors
print(f"Hyperparameters: d_model={d_model}, d_k={d_k}, d_v={d_v}\n")

# 2. Instantiate the SpikingAttentionHead module
# We use the default nengo.LIF() for spiking_neuron_type.
attention_head_pt = SpikingAttentionHead(d_model, d_k, d_v)
print("Instantiated SpikingAttentionHead (PyTorch module):\n", attention_head_pt, "\n")

# 3. Create an example input tensor X_torch
batch_size = 1
seq_len = 2   # Sequence length
X_torch = torch.rand(batch_size, seq_len, d_model) # Random data for input
print(f"Example Input X_torch (shape: {X_torch.shape}):\n", X_torch, "\n")

# 4. Perform an initial forward pass with the PyTorch model
# This will compute the attention context and weights using the defined layers.
# The SpikingActivation layers will simulate Nengo neurons for 1 step by default.
context_pt, attn_weights_pt = attention_head_pt(X_torch)

# 5. Print the shapes of the context_pt and attn_weights_pt tensors
print("--- Output of PyTorch SpikingAttentionHead ---")
print(f"Context vector (context_pt) shape: {context_pt.shape}")
# Expected shape: (batch_size, seq_len, d_v) -> (1, 2, 3)
print(f"Attention weights (attn_weights_pt) shape: {attn_weights_pt.shape}")
# Expected shape: (batch_size, seq_len, seq_len) -> (1, 2, 2)

# --- Dummy Training Loop (PyTorch) ---
# The purpose of this dummy training loop is to initialize the weights of the
# PyTorch module (attention_head_pt) to some non-random values.
# While not strictly necessary for nengo_pytorch.Layer conversion if pre_build=False,
# having initialized weights can sometimes be helpful if one were to inspect
# or use the PyTorch module directly, or if specific initial weight distributions
# were desired before conversion and Nengo simulation.
# This is NOT training the model to perform a meaningful attention task here,
# as the target is arbitrary.

print("\n--- Starting Dummy PyTorch Training Loop ---")

# 1. Define Dummy Target
# Create a target tensor with the same shape as the context vector output by the model.
# We use the context_pt from the previous forward pass to get the correct shape.
dummy_target = torch.rand_like(context_pt)
# Alternatively, could use torch.zeros_like(context_pt)
print(f"Dummy Target shape: {dummy_target.shape}")

# 2. Define Loss Function and Optimizer
criterion = nn.MSELoss()
optimizer = torch.optim.Adam(attention_head_pt.parameters(), lr=0.001)
print(f"Optimizer: {optimizer}")
print(f"Loss Criterion: {criterion}\n")

# 3. Training Loop
num_epochs = 20
for epoch in range(num_epochs):
    # Forward pass to get predictions
    context_pred, _ = attention_head_pt(X_torch) # X_torch is our static example input

    # Compute loss
    loss = criterion(context_pred, dummy_target)

    # Backward pass and optimize
    optimizer.zero_grad() # Clear previous gradients
    loss.backward()       # Compute gradients
    optimizer.step()      # Update weights

    if (epoch + 1) % 5 == 0:
        print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {loss.item():.4f}")

print("--- Dummy PyTorch Training Loop Complete ---")

# The weights of 'attention_head_pt' have now been updated by the optimizer.

# --- Convert PyTorch Model to Nengo Network ---
# Now we use nengo_pytorch.Converter to transform the PyTorch nn.Module
# (attention_head_pt) into a Nengo network.

print("\n--- Starting PyTorch to Nengo Conversion ---")

# 1. Instantiate the Converter
# The converter handles the translation from PyTorch layers to Nengo objects.
# It will replace PyTorch layers with equivalent Nengo networks or objects.
# For nengo_pytorch.SpikingActivation layers, it will use the specified Nengo neuron models.
converter = nengo_pytorch.Converter()

# 2. Convert the Model
# We pass the PyTorch model and example input data (X_torch).
# `input_data` helps the converter determine input shapes and batching.
# The converter builds a Nengo network that replicates the PyTorch model's structure.
# The `X_torch` has shape (batch_size, seq_len, d_model).
# The resulting Nengo network will typically expect an input of shape (seq_len, d_model)
# when batch_size is 1, as Nengo's core simulator processes one "unbatched" item at a time
# per timestep, or (batch_size, seq_len, d_model) if batch_size > 1 for the NengoDL simulator.
nengo_network = converter(attention_head_pt, input_data=X_torch)

# 3. Set Nengo Network dt (if not already set)
# The simulation timestep `dt` is crucial for Nengo simulations.
# If the converter doesn't set it, we provide a default.
if nengo_network.dt is None:
    nengo_network.dt = 0.001 # Default Nengo timestep
    print(f"Set nengo_network.dt to default: {nengo_network.dt}")


print("\nPyTorch model converted to Nengo network successfully.")
print("Nengo Network Object:\n", nengo_network) # This will be a nengo.Network object

# 4. Access Input/Output Placeholders (Probeables) in the Nengo Network
# The converter stores a mapping from the original PyTorch tensors (involved in the
# forward pass during conversion) to Nengo objects/probeables within the new network.
# We use the tensors from our PyTorch forward pass (X_torch, context_pt, attn_weights_pt)
# to find their corresponding representations in the Nengo network.

# It's important to use the *actual tensor objects* that were part of the
# PyTorch model's forward pass trace performed by the converter.
# `X_torch` was the input to the `attention_head_pt` during conversion.
# `context_pt` was the first output.
# `attn_weights_pt` was the second output.

# Find the Nengo object corresponding to the input X_torch
# This will typically be a nengo.Node that serves as the input to the converted network.
nengo_input_node_key = None
# The direct input tensor X_torch might not be directly in tensor_to_probeable if it was
# immediately processed by the first layer. We look for the output of the first layer's input.
# A common way is to find the input to the first layer of your PyTorch model.
# For `attention_head_pt.w_q` (an nn.Linear layer), its input during the trace is X_torch.
# The converter often creates probeables for inputs/outputs of converted `nengo_pytorch.Layer` objects.
# The main input to the *entire converted network* is usually `converter.inputs[original_module_input_tensor]`
if X_torch in converter.inputs:
    nengo_input_node_key = converter.inputs[X_torch]
else:
    # Fallback: try to find a key related to the first layer's input if direct X_torch fails.
    # This part can be tricky and might require inspecting converter.tensor_to_probeable
    # or understanding how Converter handles specific layer inputs.
    # For now, we'll assume converter.inputs[X_torch] works or leave it as None.
    print("Warning: Could not directly find nengo_input_node_key via converter.inputs[X_torch]. Further inspection needed.")


# Find the Nengo objects corresponding to the outputs context_pt and attn_weights_pt
# These will be nengo.Nodes or parts of Nengo objects that can be probed.
nengo_context_probe_key = converter.outputs[context_pt]
nengo_weights_probe_key = converter.outputs[attn_weights_pt]


print("\n--- Nengo Network Input/Output Keys ---")
print(f"Nengo Input Node Key (for X_torch): {nengo_input_node_key}")
print(f"Nengo Context Output Probe Key (for context_pt): {nengo_context_probe_key}")
print(f"Nengo Attention Weights Output Probe Key (for attn_weights_pt): {nengo_weights_probe_key}")

# These keys can now be used to create nengo.Probes in a simulation context.
# For example:
# with nengo_network:
#     input_probe = nengo.Probe(nengo_input_node_key) # If it's an object like a Node
#     context_probe = nengo.Probe(nengo_context_probe_key)
#     weights_probe = nengo.Probe(nengo_weights_probe_key)
#
# The actual simulation would then involve a Nengo simulator (like nengo.Simulator or nengo_dl.Simulator).

# --- Prepare Nengo Network for Simulation ---
# We now set up the necessary Nengo components (input node, connections, probes)
# within the converted `nengo_network` to run a simulation.

print("\n--- Preparing Nengo Network for Simulation ---")

# 1. Define Nengo Input Data
# We'll use the first sequence from our PyTorch batch `X_torch`.
# Nengo typically processes unbatched data per timestep, so we select X_torch[0].
# Convert to NumPy array as Nengo Nodes typically output NumPy arrays.
# X_torch shape: (batch_size, seq_len, d_model)
# nengo_input_data shape: (seq_len, d_model) e.g. (2,4)
nengo_input_data = X_torch[0].numpy()
print(f"Nengo input data shape (X_torch[0]): {nengo_input_data.shape}")

# 2. Nengo Network Context and Component Setup
# The `with nengo_network:` block ensures that all Nengo objects defined
# inside it are added to our `nengo_network`.

# Initialize dictionary to store Q, K, V probes
probes_qkv = {}

with nengo_network:
    # 3. Create Nengo Input Node
    # This node will feed the `nengo_input_data` into the network.
    # The lambda function makes it output the data constantly.
    # For a single forward pass, we might make it output only for a short duration,
    # but for simplicity in Nengo core sim, constant output is fine.
    sim_input_node = nengo.Node(
        output=lambda t: nengo_input_data,
        label="Simulation_Input_Sequence"
    )

    # 4. Connect Nengo Input Node to Model's Input Object
    # `nengo_input_node_key` is the Nengo object within the converted network
    # that corresponds to the input of the original PyTorch model (X_torch).
    # This connection feeds data from our `sim_input_node` to the model.
    if nengo_input_node_key is not None:
        nengo.Connection(sim_input_node, nengo_input_node_key, synapse=None)
        print(f"Connected sim_input_node to model input: {nengo_input_node_key}")
    else:
        print("Error: nengo_input_node_key is None. Cannot connect simulation input.")


    # 5. Probe Main Model Outputs
    # `nengo_context_probe_key` and `nengo_weights_probe_key` are the Nengo objects
    # corresponding to the outputs of the original PyTorch model.
    context_probe = nengo.Probe(nengo_context_probe_key, label="Context_Output_Probe")
    attn_weights_probe = nengo.Probe(nengo_weights_probe_key, label="Attention_Weights_Output_Probe")
    print(f"Created probe for context output: {nengo_context_probe_key}")
    print(f"Created probe for attention weights output: {nengo_weights_probe_key}")


    # 6. Probe Spiking Activation Layer Outputs (Q, K, V)
    # We iterate through the nodes in the converted network to find those
    # corresponding to the `SpikingActivation` layers (`act_q`, `act_k`, `act_v`).
    # Their labels are typically derived from the PyTorch module hierarchy.
    # We probe their `output` to get their decoded values.
    for node in nengo_network.nodes:
        if node.label == "attention_head_pt.act_q": # Default label by Converter
            probes_qkv['q'] = nengo.Probe(node.output, label=f"{node.label}_output_value_probe")
            print(f"Created probe for Q ({node.label}) output value.")
        elif node.label == "attention_head_pt.act_k":
            probes_qkv['k'] = nengo.Probe(node.output, label=f"{node.label}_output_value_probe")
            print(f"Created probe for K ({node.label}) output value.")
        elif node.label == "attention_head_pt.act_v":
            probes_qkv['v'] = nengo.Probe(node.output, label=f"{node.label}_output_value_probe")
            print(f"Created probe for V ({node.label}) output value.")

# Store Q, K, V probes for easy access (optional, already in probes_qkv)
q_probe = probes_qkv.get('q')
k_probe = probes_qkv.get('k')
v_probe = probes_qkv.get('v')

print("\nNengo network prepared for simulation with input node, connections, and probes.")

# --- Run NengoDL Simulation ---
# We now simulate the Nengo network using the NengoDL simulator.
# NengoDL can simulate Nengo networks and is particularly useful when parts
# of the network might involve TensorFlow operations (though in this converted
# network, SpikingActivation layers are Nengo objects, nengo_pytorch.Layer handles them).
# For a network converted by nengo_pytorch.Converter without further tf blocks,
# nengo.Simulator would also work. NengoDL is used here for consistency if one
# were to mix with other NengoDL features.

print("\n--- Running NengoDL Simulation ---")

# 1. NengoDL Simulator Context
# `dt` is taken from the network itself (which we ensured was set earlier).
with nengo_dl.Simulator(nengo_network, dt=nengo_network.dt) as sim:
    # 2. Run Simulation
    # We run for 0.1 seconds (100ms). Given dt=0.001s, this is 100 simulation steps.
    # The input node (`sim_input_node`) provides constant input throughout this time.
    # The spiking neurons will evolve over these steps.
    simulation_time = 0.1
    sim.run(simulation_time)
    print(f"Simulation complete ({simulation_time*1000} ms runtime).")

    # 3. Retrieve Probed Data
    # The data collected by probes is stored in `sim.data`.
    # Each entry corresponds to a probe object and contains the time-series
    # of the probed values.
    sim_data_context = sim.data[context_probe]
    sim_data_attn_weights = sim.data[attn_weights_probe]

    # Retrieve Q, K, V data if probes were successfully created
    sim_data_q = sim.data[q_probe] if q_probe is not None else None
    sim_data_k = sim.data[k_probe] if k_probe is not None else None
    sim_data_v = sim.data[v_probe] if v_probe is not None else None

    print("Probed data retrieved from simulator.")

# 4. Print Data Shapes (for verification)
# These shapes include the time dimension (number of simulation steps).
# For example, context output will be (num_steps, seq_len, d_v).
print("\n--- Nengo Simulation Output Shapes ---")
print(f"Context output shape: {sim_data_context.shape if sim_data_context is not None else 'N/A'}")
print(f"Attention weights output shape: {sim_data_attn_weights.shape if sim_data_attn_weights is not None else 'N/A'}")
if sim_data_q is not None:
    print(f"Q (act_q output) shape: {sim_data_q.shape}")
if sim_data_k is not None:
    print(f"K (act_k output) shape: {sim_data_k.shape}")
if sim_data_v is not None:
    print(f"V (act_v output) shape: {sim_data_v.shape}")

# The retrieved data (sim_data_context, sim_data_attn_weights, etc.)
# are now available in the global scope for further analysis or plotting.

# --- Plot Simulation Results ---
# We now visualize the probed data from the NengoDL simulation.

print("\n--- Plotting Simulation Results ---")

# 1. Prepare Time Axis for Plots
# simulation_time was defined before the simulation block
# dt_val should be the nengo_network.dt
dt_val = nengo_network.dt # e.g., 0.001s
time_axis = np.arange(0, simulation_time, dt_val)
# Ensure time_axis length matches the number of simulation steps
# This might be off by one if simulation_time is not a perfect multiple of dt_val,
# or due to floating point issues. It's safer to use the actual number of steps from data.
if sim_data_context is not None and len(time_axis) != sim_data_context.shape[0]:
    print(f"Adjusting time_axis length from {len(time_axis)} to {sim_data_context.shape[0]}")
    time_axis = np.linspace(0, simulation_time, sim_data_context.shape[0], endpoint=False)


# 2. Plot Q, K, V Spiking Layer Outputs (Decoded Values)
plt.figure(figsize=(12, 9)) # Adjusted figure size for 3 plots

# Plot Q
if sim_data_q is not None:
    plt.subplot(3, 1, 1)
    for i in range(sim_data_q.shape[1]): # Iterate over sequence length
        for j in range(sim_data_q.shape[2]): # Iterate over features (d_k)
            plt.plot(time_axis[:sim_data_q.shape[0]], sim_data_q[:, i, j], label=f'Q seq{i} feat{j}')
    plt.title('Spiking Q Output (Decoded Values)')
    plt.xlabel('Time (s)')
    plt.ylabel('Value')
    plt.legend(loc='best', fontsize='small')
else:
    plt.subplot(3, 1, 1)
    plt.title('Spiking Q Output (Data not available)')

# Plot K
if sim_data_k is not None:
    plt.subplot(3, 1, 2)
    for i in range(sim_data_k.shape[1]): # Iterate over sequence length
        for j in range(sim_data_k.shape[2]): # Iterate over features (d_k)
            plt.plot(time_axis[:sim_data_k.shape[0]], sim_data_k[:, i, j], label=f'K seq{i} feat{j}')
    plt.title('Spiking K Output (Decoded Values)')
    plt.xlabel('Time (s)')
    plt.ylabel('Value')
    plt.legend(loc='best', fontsize='small')
else:
    plt.subplot(3, 1, 2)
    plt.title('Spiking K Output (Data not available)')

# Plot V
if sim_data_v is not None:
    plt.subplot(3, 1, 3)
    for i in range(sim_data_v.shape[1]): # Iterate over sequence length
        for j in range(sim_data_v.shape[2]): # Iterate over features (d_v)
            plt.plot(time_axis[:sim_data_v.shape[0]], sim_data_v[:, i, j], label=f'V seq{i} feat{j}')
    plt.title('Spiking V Output (Decoded Values)')
    plt.xlabel('Time (s)')
    plt.ylabel('Value')
    plt.legend(loc='best', fontsize='small')
else:
    plt.subplot(3, 1, 3)
    plt.title('Spiking V Output (Data not available)')

plt.tight_layout() # Adjust subplots to fit in figure area.

# 3. Plot Final Context Vector Output
plt.figure(figsize=(10, 5)) # Adjusted figure size
if sim_data_context is not None:
    for i in range(sim_data_context.shape[1]): # Iterate over sequence length
        for j in range(sim_data_context.shape[2]): # Iterate over features (d_v)
            plt.plot(time_axis[:sim_data_context.shape[0]], sim_data_context[:, i, j], label=f'Context seq{i} feat{j}')
    plt.title('Final Context Vector Output Over Time')
    plt.xlabel('Time (s)')
    plt.ylabel('Value')
    plt.legend(loc='best', fontsize='small')
else:
    plt.title('Final Context Vector Output (Data not available)')
plt.tight_layout()

# 4. Plot Attention Weights (Example: for all sequence items)
# The shape of sim_data_attn_weights is (num_steps, seq_len_q, seq_len_k)
# For self-attention, seq_len_q == seq_len_k.
plt.figure(figsize=(10, 5)) # Adjusted figure size
if sim_data_attn_weights is not None:
    # sim_data_attn_weights shape is (time, seq_len, seq_len)
    seq_len_plot = sim_data_attn_weights.shape[1] # Should be same as X_torch.shape[1]
    for i in range(seq_len_plot): # Index for the query sequence item
        for j in range(seq_len_plot): # Index for the key sequence item
            plt.plot(time_axis[:sim_data_attn_weights.shape[0]], sim_data_attn_weights[:, i, j], label=f'Attn Weight [{i}][{j}] (Query {i} to Key {j})')
    plt.title('Attention Weights Over Time')
    plt.xlabel('Time (s)')
    plt.ylabel('Weight Value')
    plt.legend(loc='best', fontsize='small')
else:
    plt.title('Attention Weights (Data not available)')
plt.tight_layout()

# 5. Display all plots
# This will open windows showing the generated figures.
# In some environments (like Jupyter notebooks with %matplotlib inline),
# plots might show up automatically after each figure definition.
# plt.show() ensures they are displayed in script environments.
print("Displaying plots...")
plt.show()

print("\nEnd of script.")

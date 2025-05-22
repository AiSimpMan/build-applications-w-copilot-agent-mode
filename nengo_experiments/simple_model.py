# Import the nengo library
import nengo

# Nengo Versioning Comments:
# This script utilizes the modern Nengo Core API.
# For simple models like this one, the core API is largely consistent
# across recent Nengo versions (e.g., Nengo 2.x and Nengo 3.x/Nengo Core).
# Nengo versioning becomes more critical for:
#   - Installation procedures
#   - Specific backends (e.g., NengoLoihi for neuromorphic hardware, NengoDL for deep learning integration)
#   - Advanced features or less commonly used API components.

# Create a Nengo Network object, labeled "Simple Model"
model = nengo.Network(label="Simple Model")

with model:
    # Create input nodes
    # stim1 provides a constant value of 0.5
    stim1 = nengo.Node(output=0.5, label="stim1")
    # stim2 provides a constant value of -0.5
    stim2 = nengo.Node(output=-0.5, label="stim2")

    # Create a neural ensemble
    # 50 LIF neurons, representing a 1-dimensional space
    ensemble = nengo.Ensemble(n_neurons=50, dimensions=1, label="Ensemble")

    # Create connections
    # Connect stim1 to the ensemble
    nengo.Connection(stim1, ensemble)
    # Connect stim2 to the ensemble
    nengo.Connection(stim2, ensemble)

    # Create a probe to record the decoded output of the ensemble
    ensemble_probe = nengo.Probe(ensemble, synapse=0.01) # 0.01 synapse for smoothing

# Create the simulator
with nengo.Simulator(model) as sim:
    # Run the simulation for 1.0 second
    sim.run(1.0)
    # Retrieve the probed data
    probed_data = sim.data[ensemble_probe]

# Print the retrieved probed data
print(probed_data)

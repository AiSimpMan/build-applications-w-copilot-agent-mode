# Nengo GUI vs. Nengo Scripting

This directory contains a Python script (`simple_model.py`) that demonstrates building and simulating a basic neural model using the Nengo library. Nengo offers two primary ways to interact with its modeling capabilities: the Nengo GUI and Python scripting.

## Nengo GUI

The Nengo GUI provides a **visual and interactive environment** for constructing and simulating neural models. Users can drag and drop components, connect them graphically, and visualize simulation results in real-time. This approach is excellent for:

*   **Learning Nengo concepts:** The visual feedback makes it easier to understand how different components interact.
*   **Rapid prototyping:** Quickly sketching out model ideas and testing them.
*   **Visual exploration:** Observing neural activity and model dynamics through built-in plots and visualizations.

## Python Scripting (e.g., `simple_model.py`)

The `simple_model.py` script in this directory illustrates how the same neural model can be built **programmatically using Python**. This method involves writing code to define network components, their properties, and the connections between them. Scripting is powerful for:

*   **Complex models:** Managing large and intricate models that would be cumbersome to build graphically.
*   **Automation:** Running simulations in batches, performing parameter sweeps, and integrating Nengo models into larger experimental workflows.
*   **Version control:** Treating your model definition as code, allowing for easier tracking of changes and collaboration.
*   **Integration with other Python tools:** Seamlessly combining Nengo models with other scientific computing libraries like NumPy, SciPy, or machine learning frameworks.

## Core Concepts: Shared Foundation

It's important to understand that both the Nengo GUI and Python scripting **utilize the same core Nengo concepts**. Whether you are dragging an ensemble onto a canvas in the GUI or typing `nengo.Ensemble()` in a script, you are working with fundamental Nengo objects like:

*   **Ensembles:** Groups of neurons that represent information.
*   **Nodes:** Inputs to and outputs from the model, or internal computations.
*   **Connections:** Pathways for information flow between components.
*   **Probes:** Mechanisms for recording data from various parts of the model during simulation.

## Which Approach to Choose?

*   **Nengo GUI** is often preferred for its ease of use, immediate visual feedback, and suitability for learning and exploratory work.
*   **Python scripting** offers greater flexibility, scalability for complex models, automation capabilities, and integration possibilities, making it the choice for more advanced projects and research.

Many Nengo users start with the GUI to grasp the basics and then transition to or combine it with scripting as their needs evolve. The `simple_model.py` script serves as a basic example of the scripting approach.

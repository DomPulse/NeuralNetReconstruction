# NeuralNetReconstruction
I am attempting to recreate the underlying synaptic weights and other relevant parameters of neural networks by knowing the structure and taking measurements of simulated magnetic field data. Currently working toward adapting old code to work with biologically inspired neurons.

In a traditional feed forward network, I treated the synaptic weight as the conductance and the value of each node as a voltage. This generated a magnetic field around the network. I used this magnetic field to recreate the underlying weights in the network. This is meant to be an anolog to MEG data and synaptic weights in biological networks. Real networks a differnt in the structure of the connections and are time varying which makes the problem significantly more challenging. 

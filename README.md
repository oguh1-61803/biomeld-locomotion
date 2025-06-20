> # Locomotion biohybrid machines

This implementation utilises Neuroevolution of Augmenting Topologies (NEAT) and Hypercube-based Neuroevolution of Augmenting Topologies (HyperNEAT) to design morphologies of biohybird machines (BHMs) - sometimes called soft actuator morphologies (SAMs) -focused on locomotion tasks. In order to represent and evaluate the morphologies generated, a physics engine is used: Voxelyze, which can be found in the following GitHub repository: 

https://github.com/skriegman/reconfigurable_organisms.

Furthermore, the source code contained in this repository includes the implementation of NEAT and HyperNEAT, which are focused on replicating the behaviour of five mathematical functions: Ackeley function, Bukin function No. 6, Gramacy & Lee function, Levy function No. 13, and Rastringin function.

> **Architecture**

Since the evolutionary process implies a simulation task, the runtime takes significant time. This software has been designed to reduce the time spent finding suitable morphologies. It uses concurrency and was designed under a client-server architecture. Generally, the genetic algorithm (GA) is executed on the client side, whereas the the core of the fitness function (Voxelyze) is executed on the server side.

This software was written in Python 3.11 on the client side (and the mathematical functions code) and Python 3.10 on the server side.

**`Note`**: The client-server architecture previously described has not been implemented for the code related to the mathematical functions.

> **Repository Structure**

The source code of this repository is split into two:

* _Test functions_: Code related to replicate the behaviour of five mathematical functions.
* _Locomotion_: Code related to find optimal morphologies for BHMs.
  
Each piece of code is in a folder called "test_functions" and "locomotion", respectively. The packages used are listed in the file called "requirements.md".

> **Important Notice**

The code provided in this repository was used as part of an academic research documented in:

*

* 

Moreover, this project has received funding from the European Union’s Horizon Europe Research and Innovation programme under grant agreement No. 101070328.UWE researchers were funded by the UK Researchand Innovation grant No. 10044516.

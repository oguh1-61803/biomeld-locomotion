> # Locomotion biohybrid machines

This implementation utilises Neuroevolution of Augmenting Topologies (NEAT) and Hypercube-based Neuroevolution of Augmenting Topologies (HyperNEAT) to design morphologies of biohybird machines focused on locomotion tasks. In order to represent and evaluate the morphologies generated, a physics engine is used: Voxelyze, which can be found in the following GitHub repository: 

https://github.com/skriegman/reconfigurable_organisms.

> **Architecture**

Since the evolutionary process implies a simulation task, the runtime takes significant time. This software has been designed to reduce the time spent finding suitable morphologies. It uses concurrency and was designed under a client-server architecture. Generally, the genetic algorithm (GA) is executed on the client side, whereas the fitness function is executed on the server side.

This software was written in Python 3.11 on the client side and Python 3.10 on the server side.

> **Repository structure**

The source code of this repository is split into two:

* Client-side
* Server-side
  
Each piece of code is in a folder called "client" and "server," respectively, and contains its specific package requirements. 

> **Important Notice**

The code provided in this repository was used as part of an academic research documented in --coming soon--. 

> # **Requirements for the test functions domain**

The code related to the test function domain was written in **Python 3.11**.

> Libraries
* neat-python-2023 == 0.93
* ConfigUpdater == 3.2
  
> # Requirements for the morphology generator (locomotion) domain

# Client

The code related to the client side was written in **Python 3.11**.

> Libraries
* requests == 2.31.0
* neat-python-2023 == 0.93
* ConfigUpdater == 3.2

# Server

The code related to the server side was written in **Python 3.10**.

> Libraries
* tornado == 6.3.3 
* lxml == 4.9.3

> Other important considerations

* The executable called "voxelyze" in the path _locomotion/server/biomeld/voxelyze_ was compiled for an **ARM-based processor**, and it only works in Linux-based operative systems. If your server has an x86-based or amd64-based processor and it has a different operative systmem, you need to compile your own executable using the code contained in this repository:
   https://github.com/skriegman/reconfigurable_organisms.
  
* Lines 17-21 of the "Main.py" file, provide a suggestion regarding the server deployment.


# Paper
This repository is the implementation of the SECON 2024 paper  
**Topology Design with Resource Allocation and Entanglement Distribution for Quantum Networks**


# Warning
The code may not run as it has been modified since the submission of the paper and may still be modified for future works. At the camera ready time (Oct 23 2024), the scripts still work.


# Build
You may use requirements.txt to build your local environment, but it is not guaranteed to work as the python used by authors installs may unused packages and was used to test other projects. You may creat your own environment by installing main packages to a clean python, such as numpy, networkx, gurobipy (gurobi license required), pygeo, etc.


# Run
You can find scripts used by authors to generate the figures in the evaluation section of the paper in ./src/test/. Those scripts should give you exactly the same results as shown in the paper. To execute, run the files as python modules, such as  

python -m src.test.efficiency


# Miscellaneous
Some figures in the paper was not kept due to mistakenly re-run the scripts and early stop, but most of them should be kept as is.


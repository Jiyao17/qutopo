
# Paper
This repository is the implementation of the paper  
**Topology Design with Resource Allocation and Entanglement Distribution for Quantum Networks**


# Build
You may use requirements.txt to build your local environment, but it is not guaranteed to work as the python used by authors installs may unused packages and was used to test other projects. You may creat your own environment by installing main packages to a clean python, such as numpy, networkx, gurobipy (gurobi license required), pygeo, etc.


# Run
You can find scripts used by authors to generate the figures in the evaluation section of the paper in ./src/test/. To execute, run the files as python modules, such as  

python -m ./src/test/efficiency
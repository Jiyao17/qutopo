
# Paper
This repository is the implementation of our SECON 2024 paper  
**Topology Design with Resource Allocation and Entanglement Distribution for Quantum Networks**


# Guidelines
The code may not runnable as it has been modified since the submission of the paper and may still be modified for future works. At the camera ready time (Oct 23 2024), the scripts still work and you may refer to the version commited at that time for cnsistency with the results in our paper. If you hope to compare with our method, you may directly use the core functions (e.g., those solve given paths), rather than running the whole set of optimization.


# Build
You may use requirements.txt to build your local environment, but it is not guaranteed to work as is because some libraries requires licenses (e.g., Gurobi). 


# Run
You can find scripts used by authors to generate the figures in the evaluation section of the paper in ./src/test/. Those scripts should give you exactly the same results as shown in the paper. To execute, run the files as python modules, such as  

python -m src.test.efficiency


# Miscellaneous
Some figures in the paper was not kept due to mistakenly re-run the scripts and early stop, but most of them should be kept as is.


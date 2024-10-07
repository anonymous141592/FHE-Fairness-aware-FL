# PoPETS2024-submission99 (Issue 1).

This GitHub repository contains the implementation of a Proof-of-Concept of CKKS-based secure aggregation with three clients with group fairness measures.

this implementation is composed of two modules.
             -A Tensorflow v2 module that represents local training of the clients.
             -A Lattigo (FHE encryption library) module representing the server's homomorphic (With CKKS) fairness-aware aggregation.
     An intermediate directory "SharedFiles" serves to simulate the network, where clients write their updated models, and the server reads, encrypts then aggregates them.  

-Dependencies :
       Numpy.
       Panda.
       Sklearn
       Tensorflow.
       Lattigo homomorphic library.
       



*****************Datasets***********************
A single dataset is required to evaluate these experiments contained in datasets/ folder in the TensoflowModule : 
1) Adult-Census-Income


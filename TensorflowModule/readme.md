Federated Learning Client Simulation Module
Overview

This module simulates the local update processes of clients in a Federated Learning environment. It facilitates communication between clients and a central server, allowing for secure model aggregation via homomorphic encryption.
Communication Mechanism

Communication with the central server is implemented through read-write operations on the SharedFiles directory, which serves as a simulated communication channel.
Training Process

The training process operates in a series of rounds, as described below:

    Local Model Updates:
        At each training round (iteration i), clients perform local updates to the global model using Stochastic Gradient Descent (SGD).
        After the local update, each client writes its model parameters to the directory: ShareFiles/ClientsModels/iteration=i.

    Aggregation Process:
        The aggregation server, represented by the LattigoModule, monitors the SharedFiles directory.
        It checks the number of files in the current iteration directory against the number of registered clients to ensure all updates are received.

    Homomorphic Encryption and Aggregation:
        Once all clients' updates are submitted, the LattigoModule encrypts and aggregates the updates.
        The aggregated results are decrypted, and the global model weights are saved in the AggModel directory.

    Client Model Reconstruction:
        Clients read the aggregated model files from the AggModel directory to reconstruct the updated global model.
        The process then proceeds to the next training round.



Here's a dependencies section based on the imports you've provided:
Dependencies

To run this module, ensure you have the following packages installed:

    Python Packages:
        tensorflow (>= 2.0.0)
        scikit-learn (>= 0.24.0)
        pandas (>= 1.1.0)
        numpy (>= 1.18.0)
        flwr (>= 0.16.0)
        seaborn (>= 0.11.0)
        matplotlib (>= 3.3.0)
        imblearn (>= 0.7.0)
        scipy (>= 1.5.0)

    Standard Library:
        errno
        os
        csv
        random
        math
        time
        sys
        array
        statistics
        decimal
    
    
    pip install tensorflow scikit-learn pandas numpy flwr seaborn matplotlib imbalanced-learn scipy



Summary

This module effectively demonstrates the iterative process of client updates and aggregation in a Federated Learning setup, ensuring secure communication and model updates throughout the training rounds.



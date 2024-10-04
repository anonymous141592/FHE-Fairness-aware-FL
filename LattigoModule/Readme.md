# FHE Fair Aggregation with Lattigo v4

This project demonstrates the usage of Fully Homomorphic Encryption (FHE) for fairness-aware federated learning aggregation using the [Lattigo v4](https://github.com/tuneinsight/lattigo) library. The project implements FHE aggregation algorithms using Lattigo's CKKS and RLWE schemes.

## Requirements

- Go 1.18 or higher
- Git

## Installation

1. Initalize module : 
go mod init FHE_Fair_Aggreg

2. Install Lattigo v4:

Fetch the required packages from the Lattigo library:

go get github.com/tuneinsight/lattigo/v4
go get github.com/tuneinsight/lattigo/v4/ckks
go get github.com/tuneinsight/lattigo/v4/rlwe


3. Build the project:

To build the Go file, run:

go build FHE_Fair_Aggreg.go


4. Run the project:

To execute the program, use the following command:

./FHE_Fair_Aggreg


Alternatively, you can run it directly without building by using:


go run FHE_Fair_Aggreg.go


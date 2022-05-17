# linkage-attack-with-siamese-network
This repository contains the code the Master thesis "Tracking Individual Behavioral Patterns". The code allows to perform linkage attacks using a Siamese neural network-based approach on browsing as well as mobility data.
The neural network can be trained with contrastive or triplet loss. Further, the pair or triplet samples can be selected by using different sampling strategies. The neural networks are implemented using the library Tensorflow.

## Setup 
Requirement: 
* Python: 3.9.7
See the file *requirement.txt* for the full list.

## Installation:
```
$ git clone https://github.com/Madabaru/linkage-by-mobility-behavior.git
$ cd ..
$ cargo build --release && ./target/release/tracking-by-browsing-behavior
```
For help regarding the available parameters, simply run:
```
$ ./target/release/tracking-by-browsing-behavior --help
```

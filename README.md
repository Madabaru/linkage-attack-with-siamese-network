# linkage-attack-with-siamese-network
This repository contains the code the Master thesis "Tracking Individual Behavioral Patterns". The code allows to perform linkage attacks using a Siamese neural network-based approach on browsing as well as mobility data.
The neural network can be trained with contrastive or triplet loss. Further, the pair or triplet samples can be selected by using different sampling strategies. The neural networks are implemented using the library Tensorflow.

## Setup 
* Python: 3.9.7
* See the file `requirement.txt` for the full list of required dependencies

## Installation:
```
$ git clone https://github.com/Madabaru/linkage-by-mobility-behavior.git
$ cd ..
```
After installing the required dependencies, specifcy the path and run:
```
$ python main.py --path <...>
```

# Domain-informed Neural Networks 


[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.5771868.svg)](https://doi.org/10.5281/zenodo.5771868)

Codes for building and training the neural network model described in Domain-informed neural networks for interaction localization within astroparticle experiments.

This work proposes a novel domain-informed neural network architecture for experimental physics, using particle interaction localization with the time-projection chamber (TPC) technology for dark matter research as an example application. A key feature of the signals generated within the TPC is that they enable localization of particle interactions through a process called reconstruction (i.e., inverse-problem regression). While multilayer perceptrons (MLPs) have emerged as a leading contender for reconstruction in TPCs, such a black-box approach lacks knowledge of detector physics and this can be problematic for the extreme-precision dark matter research. It is in this regard that this paper looks anew at neural network-based interaction localization and encodes prior detector knowledge, in terms of both signal characteristics and detector geometry, into the feature encoding and the output layers of a multilayer (deep) neural network. The resulting neural network, termed Domain-informed Neural Network (DiNN), limits the receptive fields of the neurons in the initial feature encoding layers in order to account for the spatially localized nature of the signals produced within the TPC. This aspect of the DiNN, which has similarities with the emerging area of graph neural networks in that the neurons in the initial layers only connect to a handful of neurons in their succeeding layer, significantly reduces the number of parameters in the network in comparison to an MLP. In addition, in order to account for the detector geometry, the output layers of the network are modified using two geometric transformations to ensure the DiNN produces localizations within the interior of the detector. The end result is a neural network architecture that has 60\% fewer parameters than an MLP, but that still achieves similar localization performance and provides a path to future architectural developments with improved performance because of their ability to encode additional domain knowledge into the architecture

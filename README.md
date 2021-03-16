# Mouse-Models

## Linear-Classifier-PyTorch

This model takes as input all 554 neurons in every timestep in one session, and pairs with behavior data of "1" or "0" for each timestep.
It does not consider the differences betweeen trials or any information in between timesteps.

To run the model:

1. Download the "Linear-Classifier-PyTorch" folder.
2. Run "mouse-classifier.py".


## Linear-Classifier-PyTorch-Lightning

This model takes the same inputs as the Linear-Classifier-Pytorch model, but it is implemented through PyTorch Lightning. 

The code is currently congregated within *linear-classifier-pytorch-lightning.py*. 


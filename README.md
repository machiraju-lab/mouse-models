# Mouse-Models

## Linear-Classifier-PyTorch

This model takes as input all 554 neurons in every timestep in one session, and pairs with behavior data of "1" or "0" for each timestep.
It does not consider the differences betweeen trials or any information in between timesteps.

To run the model:

1. Download the "Linear-Classifier-PyTorch" folder.
2. Run "mouse-classifier.py".


## Linear-Classifier-PyTorch-Lightning

**This code does not run correctly.**

This model takes the same inputs as the Linear-Classifier-Pytorch model. 
Once the code is fixed, these inputs will be updated to reflect information across the entire trial, rather than at at each timestep. 

The code is currently congregated within *linear-classifier-pytorch-lightning.py*. 

The code executes with _loss_ = NAN, and test_acc/train_acc = 0.

![Failure](/img0.png)

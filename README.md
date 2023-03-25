��#� �C�D�C�2�0�2�3�
�
�
The files that are needed for using the neural network predictions for MPC are 

1) For the weights: all files in trained_models/IRL_based_models/    folder

2) For the function: all files in mpc_nn_files   folder

########################### For running the Safe data based and Exploratory data based neural network models ##################################
Folder: mpc_nn_files
File : mpc_combined.ipynb


from mpc_nn_model_safe import *
from mpc_nn_model_exploratory import *
import mpc_nn_model_safe as mpc_safe                   ---|
import mpc_nn_model_exploratory as mpc_exploratory     ---|  Added both files in different names

For mpc_safe        : use mpc_nn_predictor_safe neural network.
For mpc_exploratory : use mpc_nn_predictor_exploratory neural network.

eg:
1) mpc_nn_predictor_safe.nn_prediction_model() for the safe data based prediction for the control horizon. 
2) mpc_nn_predictor_exploratory.nn_prediction_model() for the exploratory data based prediction for the control horizon.

NOTE: We need to maintain two different ais_encoder_previous for each of the safe and exploratory functions.

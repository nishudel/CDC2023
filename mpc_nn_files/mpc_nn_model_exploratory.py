############################## Description ############################################################
# This function tries to predict the future trajectory of the HDV  for a horizon of timesteps  #
# We are not skipping any timesteps in this case #
# names of files-Encoder ../trained_models/IRL_based_models/IRL_explorepolicy_ais_gen1_816_pred1_24_epochs200_learning_rate0.0003_hidden_states4encoder_rnn_decoder_simple_nn_n_rollout_n_skips_MSELOSS_action_t_included.pth 
# names of files-Decoder ../trained_models/IRL_based_models/IRL_explorepolicy_ais_pred1_24_gen1_816_epochs200_learning_rate0.0003_hidden_states4encoder_rnn_decoder_simple_nn_n_rollout_n_skips_MSELOSS_action_t_included.pth 


import torch
from torch import nn


##########################################################################################
################################ AIS_gen- Encoder network ################################

class GenerateAis1(nn.Module):

    #def __init__(self,n_input,n_state,n_psi2_in=64,n_psi2_out=128):
    def __init__(self,n_input,n_state,n_psi2_in=8,n_psi2_out=16):
        super(GenerateAis1,self).__init__()
        self.PSI_layer1=nn.Linear(n_input,n_psi2_in)    
        self.PSI_layer2=nn.Linear(n_psi2_in,n_psi2_out)      
        self.PSI_layer3=nn.GRUCell(n_psi2_out,n_state)       # This is the RNN

    def forward(self,x,h):
        x=torch.relu(self.PSI_layer1(x))
        x=torch.relu(self.PSI_layer2(x))
        h=self.PSI_layer3(x,h)
        return h   

###########################################################################################
################################ AIS_pred - Decoder network ###############################


class PredictAis1(nn.Module):
    
    #def __init__(self,n_output,n_state,n_phi2_in=16,n_phi2_out=8):
    def __init__(self,n_output,n_decoder_in,n_phi2_in=6,n_phi2_out=8):
        super(PredictAis1,self).__init__()
        self.PHI_layer1=nn.Linear(n_decoder_in,n_phi2_in)  # Use RELU after
        self.PHI_layer2=nn.Linear(n_phi2_in,n_phi2_out)     # Use RELU after
        self.PHI_layer3=nn.Linear(n_phi2_out,n_output)         # mean vector of a unit-variance multivariate Gaussian distribution, samples from which are used to predict the next observation

    # x here is the hidden state and the current action that is chosen 
    # to predict the next observations for the horizon
    def forward(self,x):
        x=torch.relu(self.PHI_layer1(x))
        x=torch.relu(self.PHI_layer2(x))
        output=self.PHI_layer3(x)
        return output
    

###########################################################################################
################################ Complete neural network ##################################

class MPC_NN_Predictor(object):
    def __init__(self,n_epochs,min_timesteps,n_rollout,n_skips_per_rollout,n_test,n_input,n_output,n_state_enc,learning_rate,ais_gen_model,ais_pred_model,device):
        ## These arguments are used to create the file name to load the network weights 
        ## Some or most of them may not be of use to the actual function that is used 
        self.n_epochs=n_epochs
        self.min_timesteps=min_timesteps
        self.rollout=n_rollout
        self.n_skips=n_skips_per_rollout
        self.n_test=n_test
        self.n_input=n_input                # Dimension of input vector
        self.n_output=n_output              # Dimension of output vector
        self.n_state_enc=n_state_enc        # Hidden state size in RNN
        self.learning_rate=learning_rate                
        self.device=device                  # CUDA/CPU
        self.ais_gen_model=ais_gen_model
        self.ais_pred_model=ais_pred_model

        self.n_psi2_in=8
        self.n_psi2_out=16
        self.n_phi2_in=2
        self.n_phi2_out=4
        self.n_psi1_out=8

        
        if ais_gen_model==1:
            self.gen_model=GenerateAis1(self.n_input,self.n_state_enc).to(self.device)      
        
        if ais_pred_model==1:
            self.pred_model=PredictAis1(self.n_output,self.n_state_enc+1).to(self.device)

    def load_model_weights(self,text="(:*_*:)"):

        name_gen_path="_"
        name_pred_path="_"
        
        if self.ais_pred_model==1:
            name_pred_path="../trained_models/IRL_based_models/IRL_explorepolicy_ais_pred"+str(self.ais_pred_model)+"_"+str(self.n_phi2_in)+str(self.n_phi2_out)+"_gen"+str(self.ais_gen_model)+"_"+str(self.n_psi2_in)+str(self.n_psi2_out)+"_epochs"+str(self.n_epochs)+"_learning_rate"+str(self.learning_rate)+"_hidden_states"+str(self.n_state_enc)+text+".pth"
            name_gen_path="../trained_models/IRL_based_models/IRL_explorepolicy_ais_gen"+str(self.ais_gen_model)+"_"+str(self.n_psi2_in)+str(self.n_psi2_out)+"_pred"+str(self.ais_pred_model)+"_"+str(self.n_phi2_in)+str(self.n_phi2_out)+"_epochs"+str(self.n_epochs)+"_learning_rate"+str(self.learning_rate)+"_hidden_states"+str(self.n_state_enc)+text+".pth"

        if self.ais_pred_model==2:
            name_pred_path="../trained_models/IRL_based_models/IRL_explorepolicy_multi_rnn_ais_pred"+str(self.ais_pred_model)+"_"+str(self.n_phi2_in)+str(self.n_phi2_out)+"_gen"+str(self.ais_gen_model)+"_"+str(self.n_psi2_in)+str(self.n_psi2_out)+"_epochs"+str(self.n_epochs)+"_learning_rate"+str(self.learning_rate)+"_hidden_states"+str(self.n_state_enc)+"_"+str(self.n_state_dec)+text+".pth"
            name_gen_path="../trained_models/IRL_based_models/IRL_explorepolicy_multi_rnn_ais_gen"+str(self.ais_gen_model)+"_"+str(self.n_psi2_in)+str(self.n_psi2_out)+"_pred"+str(self.ais_pred_model)+"_"+str(self.n_phi2_in)+str(self.n_phi2_out)+"_epochs"+str(self.n_epochs)+"_learning_rate"+str(self.learning_rate)+"_hidden_states"+str(self.n_state_enc)+"_"+str(self.n_state_dec)+text+".pth"
        
        ## This part loads the model weights from the file in the location of path
        self.gen_model.load_state_dict(torch.load(name_gen_path))
        self.gen_model.eval()

        self.pred_model.load_state_dict(torch.load(name_pred_path))
        self.pred_model.eval()

        ## May use this to check if correct file is loaded
        print("names of files-Encoder",name_gen_path,"\n")
        print("names of files-Decoder",name_pred_path,"\n")

        return "loaded the weights!"
    
    ### Load model weights and call this function every timestep to get predictions for a horizon ###
    ## latest_observation,previous_action are both lists
    ## latest observation is [pos_CAV(t),pos_HDV(t),vel_CAV(t),vel_HDV(t)]
    ## previous action is [acc_CAV(t-1),acc_HDV(t-1)] - this can be set to [0,0] for the initial timestep
    ## Positon is measured from the point of conflict, so it negative
    ## till the vehicle reaches the conflict and then its positive


    def nn_prediction_model(self,latest_observation,previous_action,ais_encoder_previous=torch.tensor([0.5]),current_CAV_action=[0],time_step=0):

        ## If time_step is the initial timestep then random init ais_previous
        if time_step==0:
            ais_encoder_previous=0.5*torch.ones(self.n_state_enc,dtype=torch.float).to(self.device)

        input_list=[]
        input_list.extend(latest_observation)
        input_list.extend(previous_action)
        input_tensor=torch.tensor(input_list,dtype=torch.float).to(self.device)

        ## Passing input and the previous AIS throuth the encoder
        ais_encoder_previous=self.gen_model(input_tensor,ais_encoder_previous)

        # Adding the (action of CAV)_t as an inpug to the decoder
        decoder_input=torch.cat((ais_encoder_previous,torch.tensor([current_CAV_action],dtype=torch.float).to(self.device)),dim=0)

        ## Passing the updated AIS into the decoder
        nn_prediction=self.pred_model(decoder_input)
        
        return nn_prediction,ais_encoder_previous

#####################################################################################################
############################# Example on how to use this neural network #############################

####### Parameters #######
n_epochs=200
min_timesteps=50
n_rollout=10
n_skips_per_rollout=0
n_test=1
n_input=6
n_output=n_rollout*3 # 3 for each of the next n_rollout time steps
n_state_enc=4
learning_rate=0.0003
gen_model=1
pred_model=1
#device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
device = torch.device("cpu")

## Load the network ##
mpc_nn_predictor=MPC_NN_Predictor(n_epochs,min_timesteps,n_rollout,n_skips_per_rollout,n_test,n_input,n_output,n_state_enc,learning_rate,gen_model,pred_model,device)
mpc_nn_predictor.load_model_weights("encoder_rnn_decoder_simple_nn_n_rollout_n_skips_MSELOSS_action_t_included")

'''

## time_step=0 - beginning timestep
ais_encoder_previous=0
latest_observation= [-50.0, -61.86467554171904, 10.0, 7.39800250381899]
previous_action=[0, 0]
current_CAV_action= -5.0


nn_prediction,ais_encoder_previous=mpc_nn_predictor.nn_prediction_model(latest_observation,previous_action,ais_encoder_previous,current_CAV_action,time_step=0)


## These are the reference and the prediction for the input used in this example

# reference_tensor:  tensor([-5.3544e+01,  7.6373e+00, -1.4631e-02, -5.2017e+01,  7.6342e+00,
#         -1.5143e-02, -5.0490e+01,  7.6311e+00, -1.5625e-02, -4.8964e+01,
#          7.6279e+00, -1.6155e-02, -4.7439e+01,  7.6245e+00, -1.6715e-02,
#         -4.5914e+01,  7.6211e+00, -1.7306e-02, -4.4391e+01,  7.6175e+00,
#         -1.7929e-02, -4.2867e+01,  7.6138e+00, -1.8585e-02, -4.1345e+01,
#          7.6099e+00, -1.9274e-02, -3.9823e+01,  7.6059e+00, -1.9998e-02])
# decoder_prediction:  tensor([-54.3219,   8.9352,   0.6506, -52.5406,   9.0583,   0.5747, -50.7522,
#           9.1472,   0.4971, -48.9095,   9.2151,   0.4208, -47.0374,   9.2986,
#           0.3441, -45.1749,   9.3496,   0.3049, -43.3091,   9.3803,   0.2646,
#         -41.4378,   9.4326,   0.1963, -39.5321,   9.4583,   0.1659, -37.6572,
#           9.4721,   0.1397], grad_fn=<AddBackward0>)


print("nn_prediction",nn_prediction)



## time_step=1

latest_observation=[-48.1, -60.48507504095524, 9.0, 6.39800250381899]
previous_action=[-5.0, -5.0]
current_CAV_action= -5.0
nn_prediction,ais_encoder_previous=mpc_nn_predictor.nn_prediction_model(latest_observation,previous_action,ais_encoder_previous,current_CAV_action,time_step=1)


# ## These are the reference and the prediction for the input used in this example
# reference_tensor: [-5.2017e+01,  7.6342e+00, -1.5143e-02, -5.0490e+01,  7.6311e+00,
#         -1.5625e-02, -4.8964e+01,  7.6279e+00, -1.6155e-02, -4.7439e+01,
#          7.6245e+00, -1.6715e-02, -4.5914e+01,  7.6211e+00, -1.7306e-02,
#         -4.4391e+01,  7.6175e+00, -1.7929e-02, -4.2867e+01,  7.6138e+00,
#         -1.8585e-02, -4.1345e+01,  7.6099e+00, -1.9274e-02, -3.9823e+01,
#          7.6059e+00, -1.9998e-02, -3.8303e+01,  7.6018e+00, -2.0753e-02]

# decoder_prediction:[-51.9450,   7.1727,   0.5624, -50.5234,   7.2903,   0.5072, -49.0496,
#           7.3674,   0.4431, -47.5563,   7.4702,   0.3935, -46.0516,   7.5254,
#           0.3404, -44.5402,   7.5749,   0.3076, -43.0161,   7.6296,   0.2759,
#         -41.4827,   7.6741,   0.2346, -39.9520,   7.7092,   0.2103, -38.4100,
#           7.7451,   0.1876]


print("nn_prediction",nn_prediction)




########### To convert the tensor to a list/ array we can use ##############

nn_prediction.detach().cpu().numpy().tolist()

########## NOTE- I suggest that we do not convert the ais_encoder_previous to a list/array. ##############
'''
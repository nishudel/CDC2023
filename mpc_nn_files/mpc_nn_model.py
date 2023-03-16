############################## Description ############################################################
# This function tries to predict the future trajectory of the HDV  for a horizon of timesteps  #
# We are not skipping any timesteps in this case #
# names of files-Encoder ../trained_models/IRL_based_models/IRL_SVOpolicy_ais_gen1_816_pred1_24_epochs120_learning_rate0.0003_hidden_states4encoder_rnn_decoder_simple_nn_n_rollout_n_skips_MSELOSS.pth 
# names of files-Decoder ../trained_models/IRL_based_models/IRL_SVOpolicy_ais_pred1_24_gen1_816_epochs120_learning_rate0.0003_hidden_states4encoder_rnn_decoder_simple_nn_n_rollout_n_skips_MSELOSS.pth 


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
    def __init__(self,n_output,n_state,n_phi2_in=6,n_phi2_out=8):
        super(PredictAis1,self).__init__()
        self.PHI_layer1=nn.Linear(n_state,n_phi2_in)  
        self.PHI_layer2=nn.Linear(n_phi2_in,n_phi2_out)   
        self.PHI_layer3=nn.Linear(n_phi2_out,n_output)        

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
            self.pred_model=PredictAis1(self.n_output,self.n_state_enc).to(self.device)

    def load_model_weights(self,text="(:*_*:)"):

        name_gen_path="_"
        name_pred_path="_"
        
        if self.ais_pred_model==1:
            name_pred_path="../trained_models/IRL_based_models/IRL_SVOpolicy_ais_pred"+str(self.ais_pred_model)+"_"+str(self.n_phi2_in)+str(self.n_phi2_out)+"_gen"+str(self.ais_gen_model)+"_"+str(self.n_psi2_in)+str(self.n_psi2_out)+"_epochs"+str(self.n_epochs)+"_learning_rate"+str(self.learning_rate)+"_hidden_states"+str(self.n_state_enc)+text+".pth"
            name_gen_path="../trained_models/IRL_based_models/IRL_SVOpolicy_ais_gen"+str(self.ais_gen_model)+"_"+str(self.n_psi2_in)+str(self.n_psi2_out)+"_pred"+str(self.ais_pred_model)+"_"+str(self.n_phi2_in)+str(self.n_phi2_out)+"_epochs"+str(self.n_epochs)+"_learning_rate"+str(self.learning_rate)+"_hidden_states"+str(self.n_state_enc)+text+".pth"

        if self.ais_pred_model==2:
            name_pred_path="../trained_models/IRL_based_models/IRL_SVOpolicy_multi_rnn_ais_pred"+str(self.ais_pred_model)+"_"+str(self.n_phi2_in)+str(self.n_phi2_out)+"_gen"+str(self.ais_gen_model)+"_"+str(self.n_psi2_in)+str(self.n_psi2_out)+"_epochs"+str(self.n_epochs)+"_learning_rate"+str(self.learning_rate)+"_hidden_states"+str(self.n_state_enc)+"_"+str(self.n_state_dec)+text+".pth"
            name_gen_path="../trained_models/IRL_based_models/IRL_SVOpolicy_multi_rnn_ais_gen"+str(self.ais_gen_model)+"_"+str(self.n_psi2_in)+str(self.n_psi2_out)+"_pred"+str(self.ais_pred_model)+"_"+str(self.n_phi2_in)+str(self.n_phi2_out)+"_epochs"+str(self.n_epochs)+"_learning_rate"+str(self.learning_rate)+"_hidden_states"+str(self.n_state_enc)+"_"+str(self.n_state_dec)+text+".pth"
        
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


    def nn_prediction_model(self,latest_observation,previous_action,ais_encoder_previous=torch.tensor([0.5]),time_step=0):

        ## If time_step is the initial timestep then random init ais_previous
        if time_step==0:
            ais_encoder_previous=0.5*torch.ones(self.n_state_enc,dtype=torch.float).to(self.device)

        input_list=[]
        input_list.extend(latest_observation)
        input_list.extend(previous_action)
        input_tensor=torch.tensor(input_list,dtype=torch.float).to(self.device)

        ## Passing input and the previous AIS throuth the encoder
        ais_encoder_previous=self.gen_model(input_tensor,ais_encoder_previous)
        ## Passing the updated AIS into the decoder
        nn_prediction=self.pred_model(ais_encoder_previous)

        return nn_prediction,ais_encoder_previous

#####################################################################################################


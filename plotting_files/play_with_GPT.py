import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import json
import matplotlib.colors as mcolors

# reading the data from the file
test_key=503
name="main_files/test_results/test_results_test_index_"+str(test_key)

with open(name+'.json') as f:
    data = json.load(f)

test_results=[]

for i in range(int(len(data.keys())/2)):
    temp=[]
    for j in range(2):    
        key=str(i)+str(j)
        temp.append(data[key])
    test_results.append(temp)


x_data=[]
true_position_all=[]
for i in range(len(test_results)):
    true_position_all.append(test_results[i][0][0])
    x_data.append(i+2)



list_indices_to_plot=[0,2,10,12,20,22]
index=[0,3,6]

n_rollout=3
n_skip_per_rollout=1

x_data_all=[]
y_data_all=[]
for i in range(len(list_indices_to_plot)):
    y_data_all.append([])

for i in range(len(list_indices_to_plot)):
    for j in index:
        y_data_all[i].append(test_results[list_indices_to_plot[i]][1][j])
    #x_data_all.append(list(range(list_indices_to_plot[i],list_indices_to_plot[i]+n_rollout*(n_skip_per_rollout+1),n_skip_per_rollout+1)))
    x_data_all.append(list(range(2+list_indices_to_plot[i],2+list_indices_to_plot[i]+n_rollout*(n_skip_per_rollout+1),n_skip_per_rollout+1)))


# Create a figure and axis object
fig, ax = plt.subplots()

# using set_facecolor() method
#ax.set_facecolor("black")

for i in range(len(list_indices_to_plot)):
    line2, = ax.plot(x_data_all[i], y_data_all[i], label="prediction at t ="+str(list_indices_to_plot[i]),linewidth=3)

# Create line objects to plot the data
line1, = ax.plot(x_data, true_position_all,mcolors.CSS4_COLORS['hotpink'], label='Reference Trajectory',linewidth=1)
#line3, = ax.plot(x_data, true_position_all,mcolors.CSS4_COLORS['hotpink'], label='Reference Trajectory',linewidth=1)

# Add a legend to the plot
ax.legend()

plt.xlabel('Time in (1/10)s')
plt.ylabel('Position along the road m')
plt.title('Comparision of position of HDV from the point of conflict as origin')


# Show the plot
plt.show()


'''
# Test the network

####### Load the neural network weights from saved models #######

loaded_network=NN_structure(n_epochs,min_timesteps,n_rollout,n_skips_per_rollout,n_test,n_input,n_output,n_state_enc,learning_rate,gen_model,pred_model,device)
loaded_network.load_model_weights("encoder_rnn_decoder_simple_nn_n_rollout_3")

#test_keys=[keys_for_testing[15]]
test_key=58
test_results=loaded_network.test(test_key)
print(test_results)


import json

dict_test_results={}

for i in range(len(test_results)):
    for j in range(2):
        key=str(i)+str(j)
        dict_test_results[key]=test_results[i][j]
    
print(dict_test_results)

dict_test_results_json=json.dumps(dict_test_results)

name="test_results/test_results_test_index_"+str(test_key)
with open(name+'.json', 'w') as fp:
    fp.write(dict_test_results_json)
'''
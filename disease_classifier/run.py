# Adapted to use config.py
# Saves top N solutions

import os
import numpy as np
from matplotlib import pyplot as plt 
import kipc2_spatial_location as kirc
import importlib
import pandas as pd
neural_grid = importlib.import_module('neural_network_grid')

# Clear the terminal
os.system('cls' if os.name == 'nt' else 'clear')
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
#                Creating library for input data 
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# Load KIRC_Data.csv and extract the three mRNA columns
csv_data = pd.read_csv('KIRC_Data.csv')
c = 1
miR_200c_raw = csv_data['miR-200c'].values
miR_204_raw = csv_data['miR-204'].values
miR_887_raw = csv_data['miR-887'].values

miR_200c_log = np.log10(miR_200c_raw - np.min(miR_200c_raw) + c)
miR_204_log = np.log10(miR_204_raw - np.min(miR_204_raw) + c)
miR_887_log = np.log10(miR_887_raw - np.min(miR_887_raw) + c)

miR_200c = (miR_200c_log - np.min(miR_200c_log)) / (np.max(miR_200c_log) - np.min(miR_200c_log))
miR_204 = (miR_204_log - np.min(miR_204_log)) / (np.max(miR_204_log) - np.min(miR_204_log))
miR_887 = (miR_887_log - np.min(miR_887_log)) / (np.max(miR_887_log) - np.min(miR_887_log))

Xt = np.stack([miR_200c, miR_204, miR_887], axis=1)
ndata = Xt.shape[0]
X = Xt

# Rescale Xt from (0, 1) to (1e-8, 1e-4)
Xt_rescaled = Xt * (1e-4 - 1e-8) + 1e-8
Xt = Xt_rescaled

np.save('X_final_AHL_4.npy', Xt)
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
#                    Generating the training data
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# Set Y from the Status column: Y=0 if Healthy, else Y=1
Y = (csv_data['Status'].str.lower() != 'healthy').astype(int).values
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
#        Storing Optimised condition using Genetic Algorithm 
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# Run genetic algorithm (NOW RETURNS TOP N SOLUTIONS!)
weights, best_hidden1_topology, best_hidden2_topology, best_output_topology, act_func_params, opt_x_positions, opt_y_positions, top_solutions = kirc.run_genetic_algorithm(Xt, Y, ndata)
#print(best_hidden1_topology, best_hidden2_topology, weights, act_func_params)
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
#    Best combinations of activation function & corresponding Optimised weights
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
hidden_nodes1=best_hidden1_topology
hidden_nodes2=best_hidden2_topology
output_node=best_output_topology
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
#         Generating predicted output from the optimised weights
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
network = kirc.mlp( hidden_nodes1, hidden_nodes2, output_node, act_func_params )
opt_weights = np.array(weights)
nHN1 = len(hidden_nodes1)
nHN2 = len(hidden_nodes2)
wH1 = opt_weights[0:3*nHN1].reshape(nHN1,3) 
wH2=  opt_weights[3*nHN1:3*nHN1+nHN1*nHN2].reshape(nHN2,nHN1) 
wO = opt_weights[3*nHN1+nHN1*nHN2:]

X2 = Xt
Ytest = Y
YY = np.zeros(Ytest.shape[0])
for i in range(Ytest.shape[0]):
  #YY[i] = network.forward(X2[i,:], wH1, wH2, wO)
  YY[i] = network.forward(X2[i,:], wH1, wH2, wO)
# Binarize YY: >0.5 -> 1, else 0
YY = (YY > 0.5).astype(int)
# Count the number of KIRC and Healthy in Training data
num_ones_Ytest = np.count_nonzero(Ytest == 1)
# Count the number of KIRC and Healthy in Predicted data
num_ones_YY = np.count_nonzero(YY == 1)

# Print the counts ofKIIRC and Healthy in Training and Predicted data
print(f'Number of KIRCs in Ytest: {num_ones_Ytest}')
print(f'Number of KIRCs in YY: {num_ones_YY}')
Ytest=Ytest.reshape(-1,1)
YY=YY.reshape(-1,1)
test_data=[]
test_data=np.concatenate((X2, Ytest, YY), axis=1)
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
#                       Ploting the predicted data 
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
x =np.log10(test_data[:,0])
y = np.log10(test_data[:,1])
z= np.log10(test_data[:,2])
KIRC_train=test_data[:,3]
KIRC_test =test_data[:,4]

fig1 = plt.figure(facecolor='none') # Set the figure's facecolor to 'none'
ax = fig1.add_subplot(121, projection='3d', facecolor='none')  # Set the axes' facecolor to 'none'

# Plot the data with different colors for different Y values 
scatter = ax.scatter(x, y, z, c=KIRC_train, cmap='bwr', s=25, alpha=0.6, edgecolors='w', linewidth=0.5)
# Label the axes
ax.set_xlabel('mir-200c', color='black')
ax.set_ylabel('mir-204', color='black')
ax.set_zlabel('mir-887', color='black') 
 
# Change the color of the ticks
ax.tick_params(colors='black')
cbar = plt.colorbar(scatter)
cbar.set_ticks([0, 1])
cbar.set_ticklabels(['Healthy', 'KIRC'])
# Prevent the Z label from rotating
ax.zaxis.set_rotate_label(False)  
# Rotate the whole plot to show the Z-axis on the right side
ax.view_init(azim=-120)
# Make the grid, ticks, and labels transparent
ax.xaxis.pane.fill = False
ax.yaxis.pane.fill = False
ax.zaxis.pane.fill = False
plt.savefig("Training_data_KIRC_AHL_4.png", transparent=True, dpi=600)
#plt.show()

fig2 = plt.figure(facecolor='none')  # Set the figure's facecolor to 'none'
ax = fig2.add_subplot(122, projection='3d', facecolor='none')  # Set the axes' facecolor to 'none'

# Plot the data with different colors for different Y values 
scatter = ax.scatter(x, y, z, c=KIRC_test, cmap='bwr', s=25, alpha=0.6, edgecolors='w', linewidth=0.5)
# Label the axes
ax.set_xlabel('mir-200c', color='black')
ax.set_ylabel('mir-204', color='black')
ax.set_zlabel('mir-887', color='black') 
 
# Change the color of the ticks
ax.tick_params(colors='black')
cbar = plt.colorbar(scatter)
#cbar.set_ticks([0, 1])
#cbar.set_ticklabels(['Healthy', 'KIRC'])
# Prevent the Z label from rotating
ax.zaxis.set_rotate_label(False)  
# Rotate the whole plot to show the Z-axis on the right side
ax.view_init(azim=-120)
# Make the grid, ticks, and labels transparent
ax.xaxis.pane.fill = False
ax.yaxis.pane.fill = False
ax.zaxis.pane.fill = False
plt.savefig("Test_data_KIRC_AHL_4.png", transparent=True, dpi=600)
#plt.show()   

np.save(f'Output_data_KIRC_AHL_4.npy', test_data)
np.save(f'Optmised_weights_KIRC_AHL_4.npy', opt_weights)
np.save(f'Optmised_activation_function_KIRC_AHL_4.npy', act_func_params)
np.save(f'Optmised_hidden_layer1_KIRC_AHL_4.npy', best_hidden1_topology)
np.save(f'Optmised_hidden_layer2_KIRC_AHL_4.npy', best_hidden2_topology)
np.save(f'Optmised_output_layer_KIRC_AHL_4.npy', best_output_topology)
np.save(f'Optmised_x_positions_KIRC_AHL_4.npy', opt_x_positions)
np.save(f'Optmised_y_positions_KIRC_AHL_4.npy', opt_y_positions)


#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
print('Optimised weights:', opt_weights)
print('Optimised activation function:', act_func_params)
print('Optimised hidden layer1:', best_hidden1_topology)
print('Optimised hidden layer2:', best_hidden2_topology)
print('Optimised output layer:', best_output_topology)
print('Optimised x positions:', opt_x_positions)
print('Optimised y positions:', opt_y_positions)

#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

# Prepare HP/LP labels for hidden/output nodes
neural_grid.plot_spatial_network.hidden1_labels = [str(label) for label in best_hidden1_topology]
neural_grid.plot_spatial_network.hidden2_labels = [str(label) for label in best_hidden2_topology]
neural_grid.plot_spatial_network.output_label = str(best_output_topology[0]) if isinstance(best_output_topology, (list, np.ndarray)) else str(best_output_topology)
x_indices = np.array(opt_x_positions)
y_indices = np.array(opt_y_positions)
nHN1 = len(best_hidden1_topology)
nHN2 = len(best_hidden2_topology)
neural_grid.plot_spatial_network(x_indices, y_indices, nHN1, nHN2, title="Optimized KIRC Network (Spatial)")
plt.savefig("Optimized_KIRC_Network_AHL_4.png", bbox_inches='tight', dpi=600)

#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
def response_hp(x, act_func_params):
  KH, nH = act_func_params[0], act_func_params[1]
  return (x**nH) / (KH**nH + x**nH)

def response_lp(x, act_func_params):
  KL, nL = act_func_params[2], act_func_params[3]
  return (KL**nL) / (KL**nL + x**nL)

plt.figure(figsize=(7, 5))
x = np.sort(Xt.flatten())
y_hp = response_hp(x, act_func_params)
y_lp = response_lp(x, act_func_params)
plt.plot(x, y_hp, label='High Pass ', color='#0072B2', linewidth=3)
plt.plot(x, y_lp, label='Low Pass ', color='#D55E00', linewidth=3, linestyle='--')
plt.tick_params(axis='both', which='major', labelsize=16)
plt.xlim([np.min(x), np.max(x)])
plt.xscale('log')
plt.ylim([0, 1.05])
plt.xlabel('Input)', fontsize=18)
plt.ylabel('Output', fontsize=18)
plt.grid(True, linestyle=':', linewidth=1, alpha=0.7)
plt.legend(fontsize=18, frameon=False, loc='upper right')
plt.tight_layout()
plt.savefig("act_func_optimised_AHL_4.png", bbox_inches='tight', dpi=600)
#plt.show()



#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
#                       SAVE TOP N SOLUTIONS (NEW FEATURE!)
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
print(f"\n{'='*70}")
print(f"Saving top {len(top_solutions)} solutions...")
print(f"{'='*70}")

# Save in human-readable format
with open('disease_classifier_top_solutions.txt', 'w') as f:
    f.write("="*70 + "\n")
    f.write("TOP PERFORMING SOLUTIONS - DISEASE CLASSIFIER\n")
    f.write("="*70 + "\n\n")
    
    for sol in top_solutions:
        f.write(f"Rank {sol['rank']}:\n")
        f.write(f"  Fitness: {sol['fitness']:.6f}\n")
        f.write(f"  Hidden1 Topology: {sol['hidden1_topology']}\n")
        f.write(f"  Hidden2 Topology: {sol['hidden2_topology']}\n")
        f.write(f"  Output Topology: {sol['output_topology']}\n")
        f.write(f"  Activation Params (KH, nH, KL, nL): {sol['act_func_params']}\n")
        f.write(f"  X Positions: {sol['x_positions']}\n")
        f.write(f"  Y Positions: {sol['y_positions']}\n")
        f.write(f"  Weights (first 10): {sol['weights'][:10]}\n")
        f.write("\n" + "-"*70 + "\n\n")

# Save in numpy format
np.save('disease_classifier_top_solutions.npy', top_solutions, allow_pickle=True)

print("Top solutions saved!")
print("  - disease_classifier_top_solutions.txt (human-readable)")
print("  - disease_classifier_top_solutions.npy (numpy format)")
print(f"{'='*70}\n")

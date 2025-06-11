import os
import warnings
import json

import pandas as pd
import numpy as np
import torch
import torch.nn as nn

from sklearn.preprocessing import StandardScaler

from data_split import split_stratified_into_train_val_test
from model import SimpleNN
from learn import train, evaulate


# Define random seed
SEED = 42

# Define the base folder
# PATH = '/home/erf6575/Documents/neural_adult/'
PATH = os.getcwd() + "/"


############ DATA ############

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu') 

data = pd.read_csv(PATH+"data/adult_cleaned.csv")
# data

# Splitting 
train_set, test_set, val_set, y_train, y_test, y_val = split_stratified_into_train_val_test(data, 'income', SEED)
print("Train size: ", len(train_set))
print("Val size: ", len(val_set))
print("Test size: ", len(test_set))
print()

# Check the amount of labels we have in our response
print("Value counts of Repsonse:\n", y_train.value_counts())
print()


# Scaling and setting to device
scaler = StandardScaler()
scaler.fit(train_set)
x_train = torch.tensor(scaler.transform(train_set), dtype=torch.float32).to(device)
x_val = torch.tensor(scaler.transform(val_set), dtype=torch.float32).to(device)
x_test = torch.tensor(scaler.transform(test_set), dtype=torch.float32).to(device)

# Setting response to device
y_train = torch.tensor(y_train.values, dtype=torch.float32).to(device)
y_val = torch.tensor(y_val.values, dtype=torch.float32).to(device)
y_test = torch.tensor(y_test.values, dtype=torch.float32).to(device)


# Initiate weights with Inverse Class Frequency Method since we have unbalanced data
from sklearn.utils.class_weight import compute_class_weight
weights = compute_class_weight(class_weight="balanced", classes=np.unique(y_train.view(-1).cpu().numpy()), y=y_train.view(-1).cpu().numpy())
weights = torch.tensor(weights, dtype=torch.float32)


############ OPTIMIZE ############

from optimize import initiate, optimize
initiate(n_t=250, dev=device, train_x=x_train, train_y=y_train, val_x=x_val, val_y=y_val, wei=weights)
optimize(save_loc=f"{PATH}model_json/params.json")


############ RUN ############

# Get optimal parameters
with open(f'{PATH}model_json/params.json') as json_data:
    params = json.load(json_data)


# Instantiate the model
model = SimpleNN(input_dim=x_train.shape[1],
                    hidden_dim=params['hidden_dim'],
                    output_dim=1,
                    dropout_rate=params['dropout_rate']
                    ).to(device)

# Define the optimizer and loss function
optimizer = torch.optim.Adam(model.parameters(), lr=params['lr'], weight_decay=params['weight_decay'])
criterion = nn.BCEWithLogitsLoss(pos_weight=weights[1]) 

best_accuracy = 0.0  # Initialize to 0
best_loss = 1.0  # Initialize to 1
patience = 10  # Number of epochs with no improvement to wait before stopping
no_improvement_count = 0

print("\nTraining...")
e = 0
for epoch in range(500):
    loss = train(model, x_train, y_train, optimizer, criterion)
    probs, preds, f1, auc, precision, recall, fpr, fnr = evaulate(model, x_val, y_val)
    e = epoch
    
    if epoch % 10 == 0:
        print('Epoch: {:03d}, Loss: {:.4f}, AUC: {:.4f}'.format(epoch, loss, auc))
        
    # Check for improvement
    # rounded_accuracy = np.round(auc, decimals=4)
    # if rounded_accuracy > best_accuracy:
    #     best_accuracy = rounded_accuracy
    #     no_improvement_count = 0
    rounded_loss = np.round(loss, decimals=4)
    if rounded_loss < best_loss:
        best_loss = rounded_loss
        no_improvement_count = 0
    else:
        no_improvement_count += 1
        
    # Check if we should stop early
    if no_improvement_count >= patience:
        # print(f"No improvement for {patience} consecutive epochs. Stopping early.")
        break

# Eval section
print("\nEvaluating...")
probs, preds, f1, auc, precision, recall, fpr, fnr = evaulate(model, x_test, y_test, is_print=True)








import warnings
import json

import pandas as pd
import numpy as np
from scipy import stats
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
import seaborn as sns

import torch
import torch.nn as nn
import torch.nn.functional as F

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score, confusion_matrix

# Define random seed
SEED = 42

# Define the base folder
PATH = '/home/erf6575/Documents/neural_adult/'


############ FUNCTIONS ############

def split_stratified_into_train_val_test(df_input, stratify_colname,
                                        frac_train=0.6, frac_val=0.20, frac_test=0.20,
                                        random_state=SEED):
    
    '''
    Splits a Pandas dataframe into three subsets (train, val, and test)
    following fractional ratios provided by the user, where each subset is
    stratified by the values in a specific column (that is, each subset has
    the same relative frequency of the values in the column). It performs this
    splitting by running train_test_split() twice.

    Parameters
    ----------
    df_input : Pandas dataframe
        Input dataframe to be split.
    stratify_colname : str
        The name of the column that will be used for stratification. Usually
        this column would be for the label.
    frac_train : float
    frac_val   : float
    frac_test  : float
        The ratios with which the dataframe will be split into train, val, and
        test data. The values should be expressed as float fractions and should
        sum to 1.0.
    random_state : int, None, or RandomStateInstance
        Value to be passed to train_test_split().

    Returns
    -------
    df_train, df_val, df_test :
        Dataframes containing the three splits.
    '''

    if frac_train + frac_val + frac_test != 1.0:
        raise ValueError('fractions %f, %f, %f do not add up to 1.0' % (frac_train, frac_val, frac_test))

    if stratify_colname not in df_input.columns:
        raise ValueError('%s is not a column in the dataframe' % (stratify_colname))

    X = df_input.drop(columns=[stratify_colname]) # Contains all columns except stratified columns (response).
    y = df_input[[stratify_colname]] # Dataframe of just the column on which to stratify.

    # Split original dataframe into train and temp dataframes.
    df_train, df_temp, y_train, y_temp = train_test_split(X,
                                                        y,
                                                        stratify=y,
                                                        test_size=(1.0 - frac_train),
                                                        random_state=random_state)

    # Split the temp dataframe into val and test dataframes.
    relative_frac_test = frac_test / (frac_val + frac_test)
    df_val, df_test, y_val, y_test = train_test_split(df_temp,
                                                    y_temp,
                                                    stratify=y_temp,
                                                    test_size=relative_frac_test,
                                                    random_state=random_state)

    assert len(df_input) == len(df_train) + len(df_val) + len(df_test)

    return df_train, df_test, df_val, y_train, y_test, y_val

# Train
def train(model, x, y, optimizer, criterion):
    model.train()
    optimizer.zero_grad()
    
    # Forward pass
    logit = model(x)
    
    # with torch.no_grad():
    loss = criterion(logit, y)
    
    loss.backward()
    optimizer.step()
    
    return loss.item()

# Evaulate
def evaulate(model, x, y, is_print=False):
    
    model.eval()  # put in eval() mode
    logit = model(x)  # get output logits
    siggie = nn.Sigmoid()  # convert to probabilties 
    probs = siggie(logit)
    
    # output is the probabilities that output x is 'yes', so predictions is just rounding
    preds = torch.where(probs < 0.5, 0.0, 1.0)

    f1 = f1_score(y.cpu().numpy(), preds.cpu().detach().numpy(), average='weighted')
    auc = roc_auc_score(y.cpu().numpy(), probs.cpu().detach().numpy(), average='weighted')
    
    tn, fp, fn, tp = confusion_matrix(y.cpu().numpy(), preds.cpu().detach().numpy()).ravel()
    
    precision = tp / (tp + fp)
    recall = tp / (tp + fn)
    
    fpr = fp / (fp + tn)
    fnr = fn / (fn + tp)
    
    
    if is_print:
        print("F1:", f1)
        print("AUC:", auc)
        print(f"tn ({tn}), fp ({fp}), fn ({fn}), tp ({tp})")
        print("Precision:", precision)
        print("Recall:", recall)
        print("FPR:", fpr)
        print("FNR:", fnr)
    
    
    return probs, preds, f1, auc, precision, recall, fpr, fnr


############ MODEL ############

class SimpleNN(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, dropout_rate):
        super(SimpleNN, self).__init__()
        
        self.l1 = nn.Linear(input_dim, hidden_dim)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(p=dropout_rate)
        self.l2 = nn.Linear(hidden_dim, output_dim)
        
    def forward(self, x):
        
        out = self.l1(x)
        out = self.relu(out)
        out = self.dropout(out)
        out = self.l2(out)
        
        return out


############ DATA ############

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu') 

data = pd.read_csv(PATH+"data/adult_cleaned.csv")
# data

# Splitting 
train_set, test_set, val_set, y_train, y_test, y_val = split_stratified_into_train_val_test(data, 'income')
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

import optuna
n_t=250  # n_t = number of Optuna trials

def objective(trial):
    
    model_params = {
        'hidden_dim' : trial.suggest_int('hidden_dim', 64, 516), 
        'dropout_rate' : trial.suggest_float('dropout_rate', 0.1, 0.5)
    }
    
    optimizer_params = {
        'lr' : trial.suggest_float('lr', 0.001, 0.01),
        'weight_decay' : trial.suggest_float('weight_decay', 0.0001, 0.001)
    }
    
    
    # Instantiate the model
    model = SimpleNN(input_dim=x_train.shape[1],
                        hidden_dim=model_params['hidden_dim'],
                        output_dim=1,
                        dropout_rate=model_params['dropout_rate']
                        ).to(device)

    # Define the optimizer and loss function
    optimizer = torch.optim.Adam(model.parameters(), lr=optimizer_params['lr'], weight_decay=optimizer_params['weight_decay'])
    criterion = nn.BCEWithLogitsLoss(pos_weight=weights[1]) 

    best_accuracy = 0.0  # Initialize to 0
    best_loss = 1.0  # Initialize to 1
    patience = 10  # Number of epochs with no improvement to wait before stopping
    no_improvement_count = 0


    e = 0
    for epoch in range(500):
        loss = train(model, x_train, y_train, optimizer, criterion)
        probs, preds, f1, auc, precision, recall, fpr, fnr = evaulate(model, x_val, y_val)
        e = epoch
        
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
            break
        
        # Reporting intermediate training  results
        trial.report(loss, epoch)
        # trial.report(auc, epoch)
        
        # Prune if not good
        if trial.should_prune():
            raise optuna.exceptions.TrialPruned()
    
    return loss


# Initialize Optuna study
print("Starting Optuna...")
study = optuna.create_study(direction='minimize')  # maximize
study.optimize(objective, n_trials=n_t)

# Get best hyperparameters
best_score = study.best_value
best_params = study.best_params
print("Best loss:", best_score)
print("Best parameters:", best_params)


import json
with open(f"{PATH}model_json/params.json", "w", encoding="utf-8") as f:
    json.dump(best_params, f, ensure_ascii=False, indent=4)


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








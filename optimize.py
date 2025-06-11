import torch
import torch.nn as nn
from model import SimpleNN
from learn import train, evaulate
import optuna
import numpy as np


def initiate(n_t, dev, train_x, train_y, val_x, val_y, wei):
    global n_trials
    global device
    global x_train
    global y_train
    global x_val
    global y_val
    global weights
    
    n_trials = n_t  # n_trials = number of Optuna trials
    device = dev
    x_train = train_x
    y_train = train_y
    x_val = val_x
    y_val = val_y
    weights = wei


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


def optimize(save_loc):
    
    # Initialize Optuna study
    print("Starting Optuna...")
    study = optuna.create_study(direction='minimize')  # maximize
    study.optimize(objective, n_trials=n_trials)

    # Get best hyperparameters
    best_score = study.best_value
    best_params = study.best_params
    print("Best loss:", best_score)
    print("Best parameters:", best_params)


    import json
    with open(save_loc, "w", encoding="utf-8") as f:
        json.dump(best_params, f, ensure_ascii=False, indent=4)


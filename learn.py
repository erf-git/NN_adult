import torch
import torch.nn as nn
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score, confusion_matrix


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



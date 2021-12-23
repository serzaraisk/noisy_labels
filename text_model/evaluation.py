from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, roc_curve, precision_recall_curve, roc_auc_score, auc
import torch
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
sns.set()

def load_checkpoint(load_path, model, device):

    if load_path==None:
        return
    
    state_dict = torch.load(load_path, map_location=device)
    print(f'Model loaded from <== {load_path}')
    
    model.load_state_dict(state_dict['model_state_dict'])
    
    return state_dict['valid_loss']

def load_metrics(load_path, device):

    if load_path==None:
        return
    
    state_dict = torch.load(load_path, map_location=device)
    print(f'Model loaded from <== {load_path}')

    return state_dict['train_loss_list'], state_dict['valid_loss_list'], state_dict['global_steps_list']

def plot_metrics(destination_folder, model_name, device):
    train_loss_list, valid_loss_list, global_steps_list = load_metrics(destination_folder + '/metrics.pt', device)
    plt.plot(global_steps_list, train_loss_list, label='Train')
    plt.plot(global_steps_list, valid_loss_list, label='Valid')
    plt.xlabel('Global Steps')
    plt.ylabel('Loss')
    plt.title('Train and Val loss for model: ' + model_name)
    plt.legend()
    plt.show() 
    
    
def calculate_threshhold(model, model_name, test_loader, device, use_ground_truth):
    y_pred = []
    y_true = []

    model.eval()
    with torch.no_grad():
        for batch in test_loader:
            query, query_length = batch.query
            query = query.to(device)
            query_length = query_length.to(device)
            
            if use_ground_truth:
                answer = batch.answer.to(device)
            else:
                answer = batch.label.to(device)

            mask = query_length != 0
            query = query[mask]
            query_length = query_length[mask]
            answer = answer[mask]
            output = model(query, query_length.cpu())

            output = output.cpu()
            answer = answer.cpu()
            
            y_pred.extend(output.tolist())
            y_true.extend(answer.tolist())
            
    fpr, tpr, thresholds = roc_curve(answer, output)
    gmean = np.sqrt(tpr * (1 - fpr))
    # Find the optimal threshold
    index = np.argmax(gmean)
    thresholdOpt = round(thresholds[index], ndigits = 4)
    gmeanOpt = round(gmean[index], ndigits = 4)
    fprOpt = round(fpr[index], ndigits = 4)
    tprOpt = round(tpr[index], ndigits = 4)
    print('Best Threshold: {} with G-Mean: {}'.format(thresholdOpt, gmeanOpt))
    print('FPR: {}, TPR: {}'.format(fprOpt, tprOpt))
    print()
    return thresholdOpt
    
         
def evaluate(model, model_name, test_loader, device, version='title', threshold=0.5):
    y_pred = []
    y_true = []

    model.eval()
    with torch.no_grad():
        for batch in test_loader:
            query, query_length = batch.query
            query = query.to(device)
            query_length = query_length.to(device)
            answer = batch.answer.to(device)

            mask = query_length != 0
            query = query[mask]
            query_length = query_length[mask]
            answer = answer[mask]
            output = model(query, query_length.cpu())
            
            output = (output > threshold).int()

            output = output.cpu()
            answer = answer.cpu()
            y_pred.extend(output.tolist())
            y_true.extend(answer.tolist())
            
              
    print(f'Classification Report for model {model_name}:')
    print(classification_report(y_true, y_pred, labels=[1,0], digits=4))
    
    cm = confusion_matrix(y_true, y_pred, labels=[0,1])
    ax= plt.subplot()
    sns.heatmap(cm, annot=True, ax = ax, cmap='Blues', fmt="d")
    plt.show()

    ax.set_title('Confusion Matrix for ' + model_name)

    ax.set_xlabel('Predicted Labels')
    ax.set_ylabel('True Labels')

    ax.xaxis.set_ticklabels(['NOT_COMMERCIAL', 'COMMERCIAL'])
    ax.yaxis.set_ticklabels(['NOT_COMMERCIAL', 'COMMERCIAL'])
    
    return (roc_curve(y_true, y_pred), precision_recall_curve(y_true, y_pred), roc_auc_score(y_true, y_pred))

def get_statistics_for_one_model(folder_path, model, model_name, valid_iter, test_iter, device, use_ground_truth=False):
    load_checkpoint(folder_path + '/' + model_name + '/model.pt', model, device)
    threshold = calculate_threshhold(model,  'model_cand',  valid_iter, device, use_ground_truth=False)
    roc_auc_curve, pr_curve, roc_auc = evaluate(model,  'model_cand',  test_iter, device, threshold=threshold)
    
    area = auc(pr_curve[0], pr_curve[1])
    return {
        'roc_auc_curve': roc_auc_curve,
        'pr_curve': pr_curve,
        'roc_auc_score': roc_auc,
        'pr_area': area
        }

def print_comparison(models, figsize):

    plt.figure(figsize=figsize)

    plt.subplot(1,2,1)
    plt.title('Receiver Operating Characteristic')
    plt.plot([0, 1], [0, 1],'r--')
    plt.xlim([0, 1])
    plt.ylim([0, 1])
    plt.ylabel('True Positive Rate')
    plt.xlabel('False Positive Rate')
    for model in sorted(models):
        fpr, tpr, thresholds = models[model]['roc_auc_curve']
        roc_auc_score = models[model]['roc_auc_score']
        label = f'AUC_{model}  = {roc_auc_score:0.2f}'
        plt.plot(fpr, tpr, label = label)
    plt.legend(loc = 'lower right')   

    plt.subplot(1,2,2)
    plt.title('Precision Recall curve')
    plt.xlim([0, 1])
    plt.ylim([0, 1])
    plt.ylabel('True Positive Rate')
    plt.xlabel('False Positive Rate')
    for model in sorted(models):
        fpr, tpr, thresholds = models[model]['pr_curve']
        pr_area = models[model]['pr_area']
        label = f'PR_auc_{model}  = {pr_area:0.2f}'
        plt.plot(fpr, tpr, label = label)
    plt.legend(loc = 'lower right') 
    plt.show()
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import torch
import matplotlib.pyplot as plt
import seaborn as sns
sns.set()

def load_checkpoint(load_path, model, optimizer, device):

    if load_path==None:
        return
    
    state_dict = torch.load(load_path, map_location=device)
    print(f'Model loaded from <== {load_path}')
    
    model.load_state_dict(state_dict['model_state_dict'])
    optimizer.load_state_dict(state_dict['optimizer_state_dict'])
    
    return state_dict['valid_loss']

def load_metrics(load_path, device):

    if load_path==None:
        return
    
    state_dict = torch.load(load_path, map_location=device)
    print(f'Model loaded from <== {load_path}')
    
    return state_dict['train_loss_list'], state_dict['valid_loss_list'], state_dict['global_steps_list']

def plot_metrics(destination_folder, device):
    train_loss_list, valid_loss_list, global_steps_list = load_metrics(destination_folder + '/metrics.pt', device)
    plt.plot(global_steps_list, train_loss_list, label='Train')
    plt.plot(global_steps_list, valid_loss_list, label='Valid')
    plt.xlabel('Global Steps')
    plt.ylabel('Loss')
    plt.legend()
    plt.show() 
    
def evaluate(model, test_loader, device, version='title', threshold=0.5):
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
            y_pred.extend(output.tolist())
            y_true.extend(answer.tolist())
    
    print('Classification Report:')
    print(classification_report(y_true, y_pred, labels=[1,0], digits=4))
    
    cm = confusion_matrix(y_true, y_pred, labels=[1,0])
    ax= plt.subplot()
    sns.heatmap(cm, annot=True, ax = ax, cmap='Blues', fmt="d")

    ax.set_title('Confusion Matrix')

    ax.set_xlabel('Predicted Labels')
    ax.set_ylabel('True Labels')

    ax.xaxis.set_ticklabels(['NOT_COMMERCIAL', 'COMMERCIAL'])
    ax.yaxis.set_ticklabels(['NOT_COMMERCIAL', 'COMMERCIAL'])
    
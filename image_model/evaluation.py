from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, roc_curve, precision_recall_curve
import torch
import matplotlib.pyplot as plt
import seaborn as sns
sns.set()
from tqdm import tqdm
import torch.nn.functional as F

def load_checkpoint(load_path, model):
    model.load_state_dict(torch.load(load_path))
    model.eval()
    
def evaluate(model, model_name, test_loader, device, version='title', threshold=0.5):
    y_pred = []
    y_true = []

    model.eval()
    with torch.no_grad():
        for images,labels, index in tqdm(test_loader):
            
            images = images.to(device) 
            
            answers = index.to(device)
            
            logits = model(images)
            
            logits = F.softmax(logits,dim = 1)
            top_p,top_class = logits.topk(1,dim = 1)

            y_pred.extend(top_class.tolist())
            y_true.extend(answers.tolist())
    
    print(f'Classification Report for model {model_name}:')
    print(classification_report(y_true, y_pred, labels=[0,1,2], target_names=['Bad quality', 'OK quality', 'Good quality'], digits=4))
    
    cm = confusion_matrix(y_true, y_pred, labels=[0,1,2])
    ax= plt.subplot()
    sns.heatmap(cm, annot=True, ax = ax, cmap='Blues', fmt="d")

    ax.set_title('Confusion Matrix for ' + model_name)

    ax.set_xlabel('Predicted Labels')
    ax.set_ylabel('True Labels')

    ax.xaxis.set_ticklabels(['Bad quality', 'OK quality', 'Good quality'])
    ax.yaxis.set_ticklabels(['Bad quality', 'OK quality', 'Good quality'])
    
    #return (roc_curve(y_true, y_pred), precision_recall_curve(y_true, y_pred))
    
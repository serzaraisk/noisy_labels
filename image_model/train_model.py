from tqdm import tqdm
import shutil
import torch
import numpy as np
import torch.nn.functional as F
import os
import matplotlib.pyplot as plt
import seaborn as sns
sns.set()



def accuracy(y_pred,y_true):
    y_pred = F.softmax(y_pred,dim = 1)
    top_p,top_class = y_pred.topk(1,dim = 1)
    equals = top_class == y_true.view(*top_class.shape)
    return torch.mean(equals.type(torch.FloatTensor))

def plot_metrics(train_loss_list, valid_loss_list, global_steps_list):
    plt.plot(global_steps_list, train_loss_list, label='Train')
    plt.plot(global_steps_list, valid_loss_list, label='Valid')
    plt.xlabel('Global Steps')
    plt.ylabel('Loss')
    plt.legend()
    plt.show() 


def save_metrics(save_path, train_loss_list, valid_loss_list, global_steps_list):

    if save_path == None:
        return
    
    state_dict = {'train_loss_list': train_loss_list,
                  'valid_loss_list': valid_loss_list,
                  'global_steps_list': global_steps_list}
    
    torch.save(state_dict, save_path)
    print(f'Model saved to ==> {save_path}')
    

class ImageTrainer():
    
    def __init__(self,device, model_name, use_ground_for_training=False, use_ground_for_validate=False, criterion = None,optimizer = None,schedular = None):
        
        self.criterion = criterion
        self.optimizer = optimizer
        self.schedular = schedular
        self.device = device
        self.model_name = model_name
        self.use_ground_for_training = use_ground_for_training
        self.use_ground_for_validate = use_ground_for_validate
    
    def train_batch_loop(self,model,trainloader):
        
        train_loss = 0.0
        train_acc = 0.0
        
        for images,labels, index in tqdm(trainloader): 
            
            
            # move the data to CPU
            images = images.to(self.device)
            
            
            if self.use_ground_for_training:
                labels = index.to(self.device)
            else:
                labels = labels.to(self.device)
            logits = model(images)
            loss = self.criterion(logits,labels)
            
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
            
            train_loss += loss.item()
            train_acc += accuracy(logits,labels)
            
        return train_loss / len(trainloader), train_acc / len(trainloader) 

    
    def valid_batch_loop(self,model,validloader):
        
        valid_loss = 0.0
        valid_acc = 0.0
        
        for images,labels, index in tqdm(validloader):
            
            # move the data to CPU
            images = images.to(self.device) 
            
            if self.use_ground_for_validate:
                labels = index.to(self.device)
            else:
                labels = labels.to(self.device)
            
            logits = model(images)
            loss = self.criterion(logits,labels)
            
            valid_loss += loss.item()
            valid_acc += accuracy(logits,labels)
            
        return valid_loss / len(validloader), valid_acc / len(validloader)
            
        
    def fit(self,model, trainloader,validloader,epochs, destination_folder):
        
        valid_min_loss = np.Inf 
        
        train_loss_list = []
        valid_loss_list = []
        global_steps_list = []
        
        for i in range(epochs):
            
            model.train() # this turn on dropout
            avg_train_loss, avg_train_acc = self.train_batch_loop(model,trainloader) ###
            
            model.eval()  # this turns off the dropout lapyer and batch norm
            avg_valid_loss, avg_valid_acc = self.valid_batch_loop(model,validloader) ###
            
            if avg_valid_loss <= valid_min_loss :
                print("Valid_loss decreased {} --> {}".format(valid_min_loss,avg_valid_loss))
                if os.path.isdir(destination_folder +  '/' + self.model_name):
                    shutil.rmtree(destination_folder+  '/' + self.model_name)
                os.mkdir(destination_folder+  '/' + self.model_name)
                torch.save(model.state_dict(), destination_folder + '/' + self.model_name + '/' + self.model_name + '.pt')
                print('save model')
                valid_min_loss = avg_valid_loss
                
            print("Epoch : {} Train Loss : {:.6f} Train Acc : {:.6f}".format(i+1, avg_train_loss, avg_train_acc))
            print("Epoch : {} Valid Loss : {:.6f} Valid Acc : {:.6f}".format(i+1, avg_valid_loss, avg_valid_acc))
            
            train_loss_list.append(avg_train_loss)
            valid_loss_list.append(avg_valid_loss)
            global_steps_list.append(i)
        
        save_metrics(destination_folder + '/' + self.model_name + '/' + self.model_name + '_metrics.pt', train_loss_list, valid_loss_list, global_steps_list)
        plot_metrics(train_loss_list, valid_loss_list, global_steps_list)
import torch.nn as nn
import torch.optim as optim
import torch
import os
import shutil
from tqdm import tqdm
import numpy as np

def save_checkpoint(save_path, model, optimizer, valid_loss):

    if save_path == None:
        return
    
    state_dict = {'model_state_dict': model.state_dict(),
                  'optimizer_state_dict': optimizer.state_dict(),
                  'valid_loss': valid_loss}
    
    torch.save(state_dict, save_path)
    print(f'Model saved to ==> {save_path}')


def save_metrics(save_path, train_loss_list, valid_loss_list, global_steps_list):

    if save_path == None:
        return
    
    state_dict = {'train_loss_list': train_loss_list,
                  'valid_loss_list': valid_loss_list,
                  'global_steps_list': global_steps_list}
    
    torch.save(state_dict, save_path)
    print(f'Model saved to ==> {save_path}')

def mixup_data(x, y, query_length,device, alpha=1.0):
    '''Returns mixed inputs, pairs of targets, and lambda'''
    if alpha > 0:
        lam = np.random.beta(alpha, alpha)
    else:
        lam = 1
    
    lam = 0.5
    batch_size, col_size = x.size()

    index = torch.randperm(batch_size).numpy()   
    query = x.cpu().numpy()
    q_length = query_length.cpu().numpy()
    len_before = np.rint(q_length * alpha).astype('int')
    len_after = np.rint(q_length[index] * (1-alpha)).astype('int')
    result = np.zeros((batch_size, col_size), dtype='int')

    
    for row, len_ in enumerate(q_length):
        first_part = query[row, :len_before[row]]
        second_part = query[index[row], q_length[index[row]] - len_after[row]: q_length[index[row]]]
        zeros = col_size - first_part.shape[0] - second_part.shape[0]
        if zeros > 0:
            zeros = np.ones(zeros)
        else:
            zeros = np.ones(0)
        result[row] = np.concatenate([first_part, second_part, zeros][:col_size])

    mixed_x = torch.from_numpy(result)

    index = torch.from_numpy(index)
    if device=='cuda':
        index.cuda()
        mixed_x.cuda()
        
    y_a, y_b = y, y[index]
    return mixed_x, y_a, y_b, lam


def mixup_criterion(criterion, pred, y_a, y_b, lam):
    return lam * criterion(pred, y_a) + (1 - lam) * criterion(pred, y_b)   

# Training Function

def train(model,
          model_name,
          optimizer,
          device,
          file_path,
          train_loader,
          valid_loader,
          criterion = nn.BCELoss(),
          num_epochs = 5,
          best_valid_loss = float("Inf"),
         use_ground_for_train=False,
         use_ground_for_valid=False, 
         alpha=1):
    
    # initialize running values
    running_loss = 0.0
    valid_running_loss = 0.0
    global_step = 0
    eval_every = len(train_loader) // 2
    train_loss_list = []
    valid_loss_list = []
    global_steps_list = []

    # training loop
    model.train()
    for epoch in tqdm(range(num_epochs)):
        for batch in train_loader:
            query, query_length = batch.query
            query = query.to(device)
            query_length = query_length.to(device)
            if not use_ground_for_train:
                labels = batch.label.to(device)
            else:
                labels = batch.answer.to(device)
        
             
            # remove 0 length sentances
            mask = query_length != 0
            query = query[mask]
            query_length = query_length[mask]
            labels = labels[mask]

            inputs, targets_a, targets_b, lam = mixup_data(query, labels,query_length,
                                                       device, alpha)

            inputs = inputs.to(device)
            outputs = model(inputs, query_length.cpu())
            #loss = criterion(outputs, labels)
            loss = mixup_criterion(criterion, outputs, targets_a, targets_b, lam)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            # update running values
            running_loss += loss.item()
            global_step += 1

            # evaluation step
            if global_step % eval_every == 0:
                model.eval()
                with torch.no_grad():                    
                  # validation loop
                    for batch in valid_loader:
                        query, query_length = batch.query
                        query = query.to(device)
                        query_length = query_length.to(device)
                        if not use_ground_for_train:
                            labels = batch.label.to(device)
                        else:
                            labels = batch.answer.to(device)
                        
                        mask = query_length != 0
                        query = query[mask]
                        query_length = query_length[mask]
                        labels = labels[mask]
                        output = model(query, query_length.cpu())

                        loss = criterion(output, labels)
                        valid_running_loss += loss.item()

                # evaluation
                average_train_loss = running_loss / eval_every
                average_valid_loss = valid_running_loss / len(valid_loader)
                train_loss_list.append(average_train_loss)
                valid_loss_list.append(average_valid_loss)
                global_steps_list.append(global_step)

                # resetting running values
                running_loss = 0.0                
                valid_running_loss = 0.0
                model.train()

                # print progress
                print('Epoch [{}/{}], Step [{}/{}], Train Loss: {:.4f}, Valid Loss: {:.4f}'
                      .format(epoch+1, num_epochs, global_step, num_epochs*len(train_loader),
                              average_train_loss, average_valid_loss))
                
                # checkpoint
                if best_valid_loss > average_valid_loss:
                    best_valid_loss = average_valid_loss
                    if os.path.isdir(file_path + '/' + model_name):
                        shutil.rmtree(file_path + '/' + model_name)
                    os.mkdir(file_path + '/' + model_name)
                    save_checkpoint(file_path + '/' + model_name + '/model.pt', model, optimizer, best_valid_loss)
                    save_metrics(file_path + '/' + model_name + '/metrics.pt', train_loss_list, valid_loss_list, global_steps_list)
    
    save_metrics(file_path + '/' + model_name + '/metrics.pt', train_loss_list, valid_loss_list, global_steps_list)
    print('Finished Training!')
    
def train_outer(model, model_name, epochs=10, lr=0.001, use_ground_for_train=False, use_ground_for_valid=False):
    optimizer = optim.Adam(model.parameters(), lr=lr)
    train_model.train(model=model,
        model_name= model_name,
                optimizer=optimizer, 
                device=DEVICE, 
                file_path=DESTINATION_FOLDER,  
                train_loader=train_iter,
                valid_loader=valid_iter, 
                num_epochs=epochs,
                criterion=loss,
                 use_ground_for_train=False,
                use_ground_for_valid=False)
    plot_metrics(DESTINATION_FOLDER + '/' + model_name, model_name, DEVICE) 
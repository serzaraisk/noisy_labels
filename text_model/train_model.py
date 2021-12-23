import torch.nn as nn
import torch.optim as optim
import torch
import os
import shutil
from tqdm import tqdm

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
         use_ground_for_valid=False):
    
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
                    
            output = model(query, query_length.cpu())

            loss = criterion(output, labels)
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
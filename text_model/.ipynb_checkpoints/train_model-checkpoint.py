import torch.nn as nn
import torch.optim as optim
import torch
import os
import shutil

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
          optimizer,
          device,
          file_path,
          train_loader,
          valid_loader,
          criterion = nn.BCELoss(),
          num_epochs = 5,
          best_valid_loss = float("Inf")):
    
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
    for epoch in range(num_epochs):
        for batch in train_loader:
            query, query_length = batch.query
            query = query.to(device)
            query_length = query_length.to(device)
            labels = batch.label.to(device)
             
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
                        labels = batch.label.to(device)
                        answer = batch.answer.to(device)
                        
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
                    if os.path.isdir(file_path):
                        shutil.rmtree(file_path)
                    os.mkdir(file_path)
                    save_checkpoint(file_path + '/model.pt', model, optimizer, best_valid_loss)
                    save_metrics(file_path + '/metrics.pt', train_loss_list, valid_loss_list, global_steps_list)
    
    save_metrics(file_path + '/metrics.pt', train_loss_list, valid_loss_list, global_steps_list)
    print('Finished Training!')




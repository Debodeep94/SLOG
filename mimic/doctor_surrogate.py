import numpy as np
import torch
from torch.utils.data import TensorDataset, DataLoader, Dataset
import csv
import torch
import torch.nn as nn
import torchvision.models as models
from torchsummary import summary
import torch.optim as optim
import torch.nn.functional as F
from tqdm import tqdm
from sklearn.metrics import f1_score
import numpy as np
from torch.optim.lr_scheduler import ReduceLROnPlateau
from sklearn.model_selection import KFold
from surrogate_module.rnn_surrogate import *
from surrogate_module.surrogate_utils import *
import matplotlib.pyplot as plt

def main():

    chex = pd.read_excel('./R2Gen/data/mimic/mimic-cxr-2.0.0-chexpert.xlsx', engine='openpyxl')
    chex = chex.drop([ 'subject_id'], axis=1) #'No Finding','Support Devices',
    chexpert_cols=list(chex.columns[1:])
    print(chexpert_cols)
    chexpert_cols=list(chex.columns[1:])

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print('Currently usable device is: ', device)
    path= './R2Gen/surrogate/'
    dest = './R2Gen/surrogate/'
    # Load the image vectors and text embeddings
    text_embeddings = torch.load(path+'tensors_gt_emb_full_70.pt', map_location=torch.device('cpu'))  # shape: [num_samples, image_embedding_dim]
    image_vectors = torch.load(path+'image_vecs_70.pt', map_location=torch.device('cpu'))  # shape: [num_samples, seq_len, embedding_dim]
    outputs= torch.load(path+'surrogate_gt_labels_full_70.pt', map_location=torch.device('cpu'))
    # outputs = convert_array(outputs)
    outputs= torch.tensor(outputs)
    text_embeddings_pred = torch.load(path+'tensors_preds_emb_full_70.pt', map_location=torch.device('cpu'))  # shape: [num_samples, image_embedding_dim]
    text_embeddings = torch.stack(text_embeddings)
    text_embeddings_pred=torch.stack(text_embeddings_pred)
    image_vectors= torch.stack(image_vectors)
    # seq_lens_gt=torch.load(path+'seq_lens_gt_full_70.pt',map_location="cpu")
    # seq_lens_preds=torch.load(path+'seq_lens_pred_full_70.pt',map_location="cpu")
    # seq_lens_gt = torch.stack(seq_lens_gt)
    # seq_lens_preds = torch.stack(seq_lens_preds)
    #outputs=torch.stack(outputs)
    print(text_embeddings.size())
    print(text_embeddings_pred.size())
    print(outputs.shape)

    assert image_vectors.size(0) == text_embeddings.size(0), "Mismatch in number of samples between images and text embeddings"

    image_embedding_dim = 2048  # Based on ResNet-50 output
    text_embedding_dim = 512  # Example dimension, adjust based on your embeddings
    num_heads = 8
    hidden_dim = 512
    num_layers = 8
    num_classes = 14
    num_labels=3

    image, X_gt, X_pred, Y = image_vectors, text_embeddings, text_embeddings_pred, outputs
    
    print(X_gt.size())
    # Set random seed for reproducibility
    torch.manual_seed(42)
    total_examples=X_gt.size(0)

    #num_samples=int(total_examples*(prop/100))
    #print('picked samples: ', num_samples)
    #indices = random.sample(range(total_examples), num_samples)
    #X_gt, X_pred, Y, seq_X_gt, seq_X_pred = X_gt[indices], X_pred[indices], Y[indices], seq_X_gt[indices], seq_X_pred[indices]

    # Cross-Validate
    kf = KFold(n_splits=5, shuffle=True, random_state=42)


    torch.manual_seed(568)
    input_size = X_gt.size(-1)  # Vocabulary size
    hidden_size = 512   # Number of LSTM units
    num_layers = 8     # Number of LSTM layers
    output_size = 14     # Number of output classes
    batch_size = 512     # Batch size for training, validation, and testing
    learning_rate = 2e-6
    num_classes = output_size
    num_labels = 3
    num_heads = 8
    print('input_size: ', input_size)
    print('output_size: ', output_size)

    metrics = Metrics(chexpert_cols)
    print_freq = 5

    #dataset_size = len(train_main_dataset)
    num_folds = 5
    #fold_size = dataset_size // num_folds
    val_accs_gt=[]
    val_ind_out_gt=[]
    val_ind_out_preds=[]
    all_true=[]
    val_accs_pred=[]
    sur2_data_gt=[]
    sur2_data_pred=[]
    img_sur2 = []
    val_seq_gt = []
    val_seq_pred = []
    all_val_loss_gt = []
    all_val_loss_pred = []
    all_val_f1_gt = []
    all_val_f1_pred = []
    all_train_loss = []
    torch.cuda.empty_cache()


    fold = 0
    for train_idx, val_idx in tqdm(kf.split(X_gt)):
        fold += 1
        print(f"Fold #{fold}")
        #test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

        im_train,im_val, x_train, x_val_gt, x_val_pred = image[train_idx], image[val_idx], X_gt[train_idx], X_gt[val_idx], X_pred[val_idx]
        y_train, y_val = Y[train_idx], Y[val_idx]
        print('y_train: ', y_train)
        # seq_len_train, seq_len_val_gt, seq_len_val_preds = seq_X_gt[train_idx], seq_X_gt[val_idx], seq_X_pred[val_idx]
        weights = count_weights(y_train,0)
        print(weights)
        # PyTorch DataLoader
        train_dataset = TensorDataset(im_train, x_train, y_train)#, seq_len_train)
        train_loader = DataLoader(train_dataset, batch_size=256, shuffle=True)

        val_dataset = TensorDataset(im_val, x_val_gt, x_val_pred, y_val)#, seq_len_val_gt, seq_len_val_preds)
        val_loader = DataLoader(val_dataset, batch_size=256, shuffle=False)
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        model = VisualSurrogate(image_embedding_dim,text_embedding_dim,num_heads,
                                            hidden_dim,num_layers, num_classes, num_labels, mode='CELoss')#, num_classes, num_heads)
        model=torch.nn.DataParallel(model)
        model=model.to(device)
        sur2_data_gt.append(x_val_gt)
        sur2_data_pred.append(x_val_pred)
        img_sur2.append(im_val)
        # val_seq_gt.append(seq_len_val_gt)
        # val_seq_pred.append(seq_len_val_preds)
        # Define loss function and optimizer

        # Instantiate the custom loss function
        criterion = torch.nn.CrossEntropyLoss(weight=(weights).to(device))#, reduction='none')
        criterion2 = torch.nn.CrossEntropyLoss()
        optimizer = optim.Adam(model.parameters(), lr=learning_rate)

        # Number of training epochs
        num_epochs = 500
        val_loss_box_gt=[]
        val_loss_box_pred=[]
        train_loss_box=[]
        val_f1_gt = []
        val_f1_pred = []
        patience = 15 # Number of epochs to wait for improvement
        best_val_loss = float('inf')
        current_patience = 0
        best_f1 = 0.0
        best_recall = 0.0
        best_prec = 0.0
        # Training and validation loop
        for epoch in range(num_epochs):
            # Training phase
            model.train()
            total_loss = 0
            total_samples = 0

            for batch_images, batch_embeds, batch_targets in tqdm(train_loader):
                batch_images, batch_embeds, batch_targets = batch_images.to(device), batch_embeds.to(device), batch_targets.to(device)#, batch_seq_lens.to(device)
                
                # Modify target tensor to have the correct shape
                #batch_targets_modified = torch.zeros(batch_targets.size(0), batch_targets.size(1), 2).to(device)
                #batch_targets_modified.scatter_(2, batch_targets.unsqueeze(-1), 1)
                #print('batch_targets:', batch_targets)
                outputs = model(batch_images, batch_embeds)#, batch_seq_lens)
                #print(outputs)
                #print(outputs.size())

                #specific_output=train_y[:,3]
                # Apply sigmoid activation and reshape the outputs
                #sigmoid_outputs = torch.softmax(outputs, dim=1)#.view(batch_targets.size(0), -1, 2)
                temp1 = outputs.view(-1, num_labels)
                #print(temp1.size())
                #print(temp1)
                batch_targets=batch_targets.to(torch.int64)
                temp2 = batch_targets.view(-1)#.to(torch.float32)
                #print(temp1.size(), temp2.size())
                loss = criterion(temp1, temp2)
                
                #print(loss)
                total_loss += loss.item() * batch_embeds.size(0)
                
                total_samples += batch_targets.size(0)

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                clip_gradient(optimizer, 1.0)
            average_loss = total_loss / total_samples
            train_loss_box.append(average_loss)
            # Print training loss for the current epoch
            

            # Validation phase
            model.eval()
            total_correct = 0
            total_samples = 0

            all_acc_gt=[]
            all_acc_pred=[]
            ind_outs_gt=[]
            ind_outs_preds=[]
            true_vals=[]
            val_inputs=[]
            val_batch_seq=[]
            all_f1=[]
            all_rec=[]
            all_prec=[]
            sigmoid_matrices = []
            full_batch = []
            total_val_loss_gt = 0
            total_val_loss_pred = 0
            total_val_samples = 0
            count_iter=0
            batch_time = AverageMeter()  # forward prop. + back prop. time
            data_time = AverageMeter()  # data loading time
            losses = AverageMeter()  # loss (per word decoded)
            accs = AverageMeter()  # accuracy
            metrics_gt = Metrics(chexpert_cols)
            metrics_pred = Metrics(chexpert_cols)

            print('entering validation')
            start = time.time()
            with torch.no_grad():
                for (batch_images, batch_inputs_gt,batch_inputs_preds, batch_targets) in val_loader:
                    # store them before moving to the device
                    #print('batch inputs: ', batch_inputs)
                    #val_inputs_gt.append(batch_inputs.to('cpu'))
                    #val_batch_seq.append(batch_seq_lens.to('cpu'))

                    batch_images, batch_inputs_gt,batch_inputs_preds, batch_targets = batch_images.to(device), batch_inputs_gt.to(device),batch_inputs_preds.to(device), batch_targets.to(device)#, batch_seq_lens.to(device)
                    outputs_gt = model(batch_images, batch_inputs_gt)#, batch_seq_lens_gt)
                    outputs_pred = model(batch_images, batch_inputs_preds)#, batch_seq_lens_pred)
                    #print('output: ', outputs.size())
                    individual_outputs_gt = torch.argmax(outputs_gt, dim=2) # convert to nominals
                    individual_outputs_pred = torch.argmax(outputs_pred, dim=2)
                    #sigmoid_matrices.append(outputs)
                    #full_batch.append(batch_targets)
                    
                    val_loss_gt = criterion2(outputs_gt.view(-1, num_labels), 
                                        batch_targets.to(torch.int64).view(-1))
                    val_loss_pred = criterion2(outputs_pred.view(-1, num_labels), 
                                        batch_targets.to(torch.int64).view(-1))
                    total_val_loss_gt += val_loss_gt.item() #* batch_inputs.size(0)
                    total_val_loss_pred += val_loss_pred.item() #* batch_inputs.size(0)
                    total_predictions = outputs_gt.size(0)*outputs.size(1)
                    losses.update(val_loss_gt.item(), total_predictions)
                    total_val_samples += batch_targets.size(0)
                    correct_predictions_gt = (batch_targets == individual_outputs_gt).float() # .sum(dim=1).float() is want sum
                    # print('correct_predictions_gt: ', correct_predictions_gt)
                    # # Calculate the number of zeroes and ones in each column
                    # num_zeros = torch.sum(correct_predictions_gt == 0, dim=0)
                    # num_ones = torch.sum(correct_predictions_gt == 1, dim=0)
                    # Print the nubers
                    #print("Number of zeros in each column for correct preds of gt:", num_zeros)
                    #print("Number of ones in each column for correct preds of gt:", num_ones)
                    correct_predictions_pred = (batch_targets == individual_outputs_pred).float() # .sum(dim=1).float() is want sum
                    # Calculate the number of zeroes and ones in each column
                    # num_zeros = torch.sum(correct_predictions_pred == 0, dim=0)
                    # num_ones = torch.sum(correct_predictions_pred == 1, dim=0)
                    # Print the nubers
                    #print("Number of zeros in each column for correct preds of gt:", num_zeros)
                    #print("Number of ones in each column for correct preds of pred:", num_ones)

                    all_acc_gt.append(correct_predictions_gt.to('cpu'))
                    all_acc_pred.append(correct_predictions_pred.to('cpu'))
                    ind_outs_gt.append(individual_outputs_gt.to('cpu'))
                    ind_outs_preds.append(individual_outputs_pred.to('cpu'))
                    true_vals.append(batch_targets.to('cpu'))
                    acc = accuracy(outputs_gt.view(-1, 3).to('cpu'), batch_targets.view(-1).to('cpu'))
                    print('batch wise accuracy: ', acc)
                    accs.update(acc, total_predictions)
                    batch_time.update(time.time() - start)
                    metrics_gt.update(outputs_gt.to('cpu'), batch_targets.to('cpu'))
                    metrics_pred.update(outputs_pred.to('cpu'), batch_targets.to('cpu'))
                    
                    # Print status
                    count_iter+=1
                    start = time.time()

                    # Print status
                    if count_iter % print_freq == 0:
                        print('Validation: [{0}/{1}]\t'
                            'Batch Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                            'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                            'Top-5 Accuracy {acc.val:.3f} ({acc.avg:.3f})\t'.format(count_iter, len(val_loader), batch_time=batch_time,
                                                                                        loss=losses, acc=accs))
            
            val_loss_box_gt.append(total_val_loss_gt/len(val_loader))
            val_loss_box_pred.append(total_val_loss_pred/len(val_loader))
            print('val loss gt:', total_val_loss_gt/len(val_loader))
            print('val loss preds:', total_val_loss_pred/len(val_loader))
            metrics_dict_gt = metrics_gt.calculate_metrics()
            metrics_dict_pred = metrics_pred.calculate_metrics()
            print(
                '\n * LOSS - {loss.avg:.3f}\n'.format(
                    loss=losses))
            pos_f1_gt = metrics_dict_gt['Micro Positive F1']
            val_f1_gt.append(pos_f1_gt)
            print('positive f1 gt:',  pos_f1_gt)

            pos_f1_pred = metrics_dict_pred['Micro Positive F1']
            val_f1_pred.append(pos_f1_pred)
            print('positive f1 pred:',  pos_f1_pred)
            if pos_f1_gt > best_f1:# and recall > best_recall:
                best_f1 = pos_f1_gt
                
                #best_recall = recall
                # Save the model
                #torch.save(model.state_dict(), dest+f'best_model_surr1_fold{fold}_prop{prop}.pth')
                print('Model saved! best f1: {:.4f}'.format(best_f1))
                print(f'Best model saved to {dest}')
                current_patience = 0  # Reset patience counter
            else:
                current_patience += 1
                print('###################################################')
            if current_patience >= patience:
                print(f'Validation F1 has not improved for {patience} epochs. Stopping training.')
                # val_inputs_ = torch.concat(val_inputs, dim = 0)
                # print('val_inputs shape:', val_inputs_.size())
                # val_seq_ = torch.concat(val_batch_seq, dim = 0)
                all_acc_gt = torch.concat(all_acc_gt, dim =0)
                all_acc_pred = torch.concat(all_acc_pred, dim =0)
                ind_outs_gt = torch.concat(ind_outs_gt, dim=0)
                ind_outs_preds = torch.concat(ind_outs_preds, dim=0)
                true_vals = torch.concat(true_vals,dim=0)
                val_accs_gt.append(all_acc_gt)
                val_accs_pred.append(all_acc_pred)
                val_ind_out_gt.append(ind_outs_gt)
                val_ind_out_preds.append(ind_outs_preds)
                all_true.append(true_vals)
                #sur2_data.append(val_inputs_)
                #val_seq.append(val_seq_)
                #print(f'model saved. size of surr 2 training data {len(sur2_data)}')
                #print(f'printing surr 2 training data: {sur2_data}')
                # saving data for surrogate 2
                ##torch.save(sur2_data, path+'sur2_data.pt')
                torch.save(val_accs_gt, dest+'val_accs_gt_emb_full.pt')
                torch.save(val_accs_pred, dest+'val_accs_pred_emb_full.pt')
                torch.save(val_ind_out_gt, dest+'val_ind_out_gt_full.pt')
                torch.save(val_ind_out_preds, dest+'val_ind_out_preds_full.pt')
                torch.save(all_true, dest+'all_true_sur1_gt_full.pt')
                #val_inputs.clear()
                #all_acc.clear()
                #val_batch_seq.clear()

                break
        # storing all validation and training metrics
        all_val_loss_gt.append(val_loss_box_gt)
        all_val_loss_pred.append(val_loss_box_pred)
        all_train_loss.append(train_loss_box)
        all_val_f1_gt.append(val_f1_gt)
        all_val_f1_pred.append(val_f1_pred)

    torch.save(sur2_data_gt, dest+'sur2_data_gt_emb_full.pt')#.append(x_val_gt)
    torch.save(sur2_data_pred, dest+'sur2_data_pred_emb_full.pt')
    torch.save(val_seq_gt, dest+'val_seq_gt_emb_full.pt')
    torch.save(val_seq_pred, dest+'val_seq_pred_emb_full.pt')
    torch.save(img_sur2, path+'sur2_img_data.pt')
    # Plot the necessary curved

    # Surrogate 1
    #plt.style.use('ggplot')
    fig_sur1_loss=plt.figure(figsize=(16,9))
    plt.rcParams.update({'font.size': 30})
    # Pad lists with None for different lengths
    pad_value = None
    fold_train_losses_padded = [seq + [pad_value] * (max(map(len, all_train_loss)) - len(seq)) for seq in all_train_loss]
    fold_val_losses_padded_gt = [seq + [pad_value] * (max(map(len, all_val_loss_gt)) - len(seq)) for seq in all_val_loss_gt]
    fold_val_losses_padded_pred = [seq + [pad_value] * (max(map(len, all_val_loss_pred)) - len(seq)) for seq in all_val_loss_pred]

    # save the vectors
    torch.save(fold_train_losses_padded, dest+'fold_train_losses_padded_emb_full.pt')
    torch.save(fold_val_losses_padded_gt, dest+'fold_val_losses_padded_gt_emb_full.pt')
    torch.save(fold_val_losses_padded_pred, dest+'fold_val_losses_padded_pred_emb_full.pt')

    torch.save(all_val_f1_pred, dest+'all_val_f1_pred_full.pt')
    torch.save(all_val_f1_gt, dest+'all_val_f1_gt_full.pt')
    # Plot the training and validation losses with different tones of red and blue


    for i, (train_loss, val_loss) in enumerate(zip(fold_train_losses_padded, fold_val_losses_padded_gt)):
        #all_folds.update(f'fold_{i}':{'train_loss':train_loss, 'val_loss':val_loss})
        epochs = np.arange(1, len(train_loss) + 1)

        # Plot training loss in different tones of red
        color_train = plt.cm.Reds(0.2 + 0.2 * i)  # Adjust 0.2 and 0.2 for different tones
        plt.plot( train_loss, label=f'Fold {i + 1} - Train', color=color_train, alpha=0.7)

        # Plot validation loss in different tones of blue
        color_val = plt.cm.Blues(0.2 + 0.2 * i)  # Adjust 0.2 and 0.2 for different tones
        plt.plot( val_loss, label=f'Fold {i + 1} - Validation', color=color_val, alpha=0.7)

    plt.xlabel('x-axis')
    plt.ylabel('Loss')
    plt.title('Training and Validation Losses under 5-fold Cross-Validation')

    # Move the legend outside the plot
    plt.legend(bbox_to_anchor=(1, 1), loc='upper left')
    plt.title('loss curve for surrogate 1')
    #plt.savefig('plots/'+f'surrogate_1_loss_5_fold_full_{prop}_new.png')
    plt.close(fig_sur1_loss)
    ##torch.save(fold_train_losses_padded, path+'fold_train_losses_padded.pt')
    ##torch.save(fold_val_losses_padded, path+'fold_val_losses_padded.pt')
    #print(all_folds)

if __name__ == '__main__':
    main()
import json
import pandas as pd
import torch
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt
import torch.optim as optim
from sklearn.metrics import f1_score, recall_score, precision_score
from sklearn.utils.class_weight import compute_class_weight
import torch.nn.functional as F
import time
from sklearn.model_selection import train_test_split
from surrogate_module.surrogate_utils import *
from surrogate_module.rnn_surrogate import *
from torch.utils.data import TensorDataset, DataLoader
from torch.optim.lr_scheduler import ReduceLROnPlateau
import argparse

def parse_agrs():

    parser = argparse.ArgumentParser()
    #parser.add_argument("--proportion",type=float)
    parser.add_argument("--patience",type=int)
    args = parser.parse_args()
    return args

def main():
    args = parse_agrs()
    #prop = args.proportion

    # code for second surrogate
    torch.cuda.empty_cache()
    print(torch.cuda.memory_summary())
    base_path='/home/debodeep.banerjee/clevr/R2Gen/surrogate/'
    path=base_path#+'surrogate1/'
    dest=base_path#+'surrogate2/'
    # Load the image vectors and text embeddings
    im_data = torch.load(path+'image_vecs.pt', map_location=torch.device('cpu'))  # shape: [num_samples, seq_len, embedding_dim]
    im_data = torch.stack(im_data)
    print('len: ', len(im_data))
    outputs= torch.load(path+'surrogate_gt_labels_full_50.pt', map_location=torch.device('cpu'))
    outputs = outputs[:, 1:].astype(float)
    surr_2_y= torch.tensor(outputs)
    text_embeddings_pred = torch.load(path+'tensors_preds_emb_full_50.pt', map_location=torch.device('cpu'))  # shape: [num_samples, image_embedding_dim]
    surr_2_x=torch.stack(text_embeddings_pred)

    # seq_lengths_pred = torch.load(path+'seq_lens_pred_full_50.pt', map_location=torch.device('cpu'))  # shape: [num_samples, image_embedding_dim]
    # # seq_lengths_pred=torch.stack(seq_lengths_pred)

    # image_vectors= torch.stack(image_vectors)
    # print(f'outputs shape: {outputs.size()}')
    print(torch.cuda.memory_summary())
    print(f'sur2y: {surr_2_y.size()}')
    print(f'embedding size: {surr_2_x.size()}')
    print(f'image: {im_data.size()}')

    X = surr_2_x.detach().numpy()
    Y = surr_2_y.detach().numpy()
    im_data = im_data.detach().numpy()
    # seq_data = surr_2_seq.detach().numpy()
    

    im_train, im_test, train_x, test_x, train_y, test_y=train_test_split(im_data, X,Y,test_size=0.1, random_state=1)
    im_train,im_val, train_x, val_x, train_y, val_y = train_test_split(im_train, train_x, train_y, test_size=0.1, random_state=1) # 0.25 x 0.8 = 0.2
    #weight_train = torch.ones(len(train_x))

    im_train = torch.tensor(im_train,dtype=torch.float)
    train_x_pt = torch.tensor(train_x,dtype=torch.float) #pt: pytorch
    # seq_tr=torch.tensor(seq_tr, dtype=torch.int)

    im_val = torch.tensor(im_val, dtype=torch.float)
    val_x_pt = torch.tensor(val_x,dtype=torch.float) #pt: pytorch
    #val_x_pt = val_x_pt/torch.norm(val_x_pt, dim = 1, keepdim= True)
    # seq_val=torch.tensor(seq_val, dtype=torch.int)

    im_test = torch.tensor(im_test, dtype=torch.float)
    test_x_pt = torch.tensor(test_x,dtype=torch.float)
    #test_x_pt = test_x_pt/torch.norm(test_x_pt, dim = 1, keepdim= True)
    # seq_test=torch.tensor(seq_test, dtype=torch.int)
    # class_weights=compute_class_weight('balanced',classes=np.unique(train_y), y=train_y)
    # class_weights = torch.tensor(class_weights)
    # print('class_weights: ', class_weights)
    print(train_y)
    train_y = torch.tensor(train_y)

    val_y = torch.tensor(val_y)
    test_y = torch.tensor(test_y)

    #Save the train and validation splits
    torch.save(train_y,dest+'sec_sur_train_y.pt')
    torch.save(val_y,dest+'sec_sur_val_y.pt')
    torch.save(test_y,dest+'sec_sur_test_y.pt')

    #print('output_size: ',output_size)
    image_embedding_dim = 2048  # Based on ResNet-50 output
    text_embedding_dim = 512  # Example dimension, adjust based on your embeddings
    num_heads = 8
    hidden_dim = 512
    num_layers = 4
    num_classes = 5
    num_labels=2
    batch_size = 512
    learning_rate = 2e-4
    loss_type='BCELoss'
    weights = count_weights(train_y,1)
    print(weights)
    # Instantiate the model and move it to GPU if available
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model_2 = VisualSurrogate(image_embedding_dim, text_embedding_dim, num_heads,
                  hidden_dim, num_layers, num_classes, num_labels, mode=loss_type)#, num_classes, num_heads)\
    model_2 = torch.nn.DataParallel(model_2)
    model_2=model_2.to(device)

    # Create DataLoader for training set, validation set, and test set
    train_dataset = TensorDataset(im_train, train_x_pt, train_y)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

    val_dataset = TensorDataset(im_val, val_x_pt, val_y)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

    test_dataset = TensorDataset(im_test, test_x_pt, test_y)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    classes=['cat1', 'cat2', 'cat3', 'cat4', 'cat5']

    # Accuracy surrogate
    # Instantiate the custom loss function
    # criterion = torch.nn.CrossEntropyLoss()#, reduction='none')
    # criterion2 = torch.nn.CrossEntropyLoss()
    label_freq = torch.mean(train_y, dim=0)  # Fraction of 1s for each label
    print(label_freq)
    pos_weight = (1 - label_freq) / label_freq
    print(pos_weight) 
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight.to(device))
    criterion2 = nn.BCEWithLogitsLoss()
    optimizer = optim.Adam(model_2.parameters(), lr=learning_rate)
    scheduler = ReduceLROnPlateau(optimizer, mode='min', patience=5, factor=0.1, verbose=True, min_lr=1e-7)

    # Number of training epochs
    num_epochs = 500
    val_loss_box_sur2=[]
    train_loss_box_sur2=[]
    f1_sur2=[]
    best_val_loss = float('inf')
    current_patience = 0
    best_f1 = 0.0
    best_recall = 0.0
    best_prec = 0.0
    patience = args.patience
    # Training and validation loop
    for epoch in range(num_epochs):
        # Training phase
        model_2.train()
        total_loss = 0
        total_samples = 0

        for batch_images, batch_inputs, batch_targets in tqdm(train_loader):
            batch_images, batch_inputs, batch_targets = batch_images.to(device), batch_inputs.to(device), batch_targets.to(device)#, batch_wt_train.to(device)
            outputs = model_2(batch_images, batch_inputs)
            # print(outputs.size())
            if loss_type=='CELoss':
                temp1 = outputs.view(-1, num_labels)
                #print('temp1: ', temp1)
                batch_targets=batch_targets.to(torch.float32)
                temp2 = batch_targets.view(-1).to(torch.int64)
                #print(temp1.size(), temp2.size())
                loss = criterion(temp1, temp2)
                
                #print(loss)
                total_loss += loss.item() #* batch_inputs.size(0)
                
                total_samples += batch_targets.size(0)

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                clip_gradient(optimizer, 0.1)
            elif loss_type=='BCELoss':
                loss = criterion(outputs, batch_targets)
                #print(loss)
                total_loss += loss.item() #* batch_inputs.size(0)
                
                total_samples += batch_targets.size(0)

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                clip_gradient(optimizer, 0.5)
        average_loss = total_loss / len(train_loader)
        train_loss_box_sur2.append(average_loss)
        print('train loss: ',total_loss/(len(train_loader)))
        # Print training loss for the current epoch
        
        print('entering validation...')
        # Validation phase
        model_2.eval()
        total_correct = 0
        total_samples = 0

        all_acc=[]
        all_f1=[]
        all_rec=[]
        all_prec=[]
        sigmoid_matrices = []
        full_batch = []
        total_val_loss = 0
        total_val_samples = 0
        count_iter=0
        batch_time = AverageMeter()  # forward prop. + back prop. time
        data_time = AverageMeter()  # data loading time
        losses = AverageMeter()  # loss (per word decoded)
        accs = AverageMeter()  # accuracy
        pred_mat=[]
        start = time.time()
        with torch.no_grad():
            for batch_images, batch_inputs, batch_targets in tqdm(val_loader):
                batch_images, batch_inputs, batch_targets = batch_images.to(device), batch_inputs.to(device), batch_targets.to(device)#, batch_wt_train.to(device)
                outputs = model_2(batch_images, batch_inputs)
                #print('output: ', outputs)
                if loss_type=='CELoss':
                    individual_outputs = torch.argmax(outputs, dim=2) # convert to nominals
                    sigmoid_matrices.append(outputs)
                    full_batch.append(batch_targets)
                    pred_mat.append(individual_outputs)
                    #print(outputs)
                    #print('ground truth: ', batch_targets)
                    #print('sigmoid outputs: ', sigmoid_outputs)
                    val_loss = criterion(outputs.view(-1, num_labels), 
                                        batch_targets.to(torch.int64).view(-1))
                    
                    total_val_loss += val_loss.item() #* batch_inputs.size(0)
                elif loss_type=='BCELoss':
                    loss = criterion2(outputs, batch_targets)
                    # Apply sigmoid to get probabilities
                    probs = torch.sigmoid(outputs)
                    # Apply threshold to get binary predictions
                    individual_outputs = (probs > 0.5).int()
                    sigmoid_matrices.append(outputs)
                    full_batch.append(batch_targets)
                    pred_mat.append(individual_outputs)
                    val_loss = criterion(outputs, batch_targets)
                    
                    total_val_loss += val_loss.item() #* batch_inputs.size(0)
        final_val_loss = total_val_loss/ len(val_loader)
        val_loss_box_sur2.append(final_val_loss)
        print('final_val_loss: ', final_val_loss)

        # When the batch job is done, check for the metrics
        # As we have two labels, things are easier now. 
        all_gt = torch.stack([item for sublist in full_batch for item in sublist])
        all_preds = torch.stack([item for sublist in pred_mat for item in sublist])
        for i in range(all_gt.size(1)):
            gt = all_gt[:,i]
            pred=all_preds[:,i]
            #tn, fp, fn, tp = sklearn.metrics.confusion_matrix(gt, pred).ravel()
            tp = ((gt == 1) & (pred == 1)).sum().item()
            fp = ((gt == 0) & (pred == 1)).sum().item()
            tn = ((gt == 0) & (pred == 0)).sum().item()
            fn = ((gt == 1) & (pred == 0)).sum().item()
            print(classes[i])
            print(f'tp: {tp}, fp: {fp}, tn: {tn}, fn: {fn}')
        #print(all_preds)
        f1_micro = f1_score(all_gt.cpu(), all_preds.cpu(), average="micro")
        f1_macro = f1_score(all_gt.cpu(), all_preds.cpu(), average="macro")
        rec_micro = recall_score(all_gt.cpu(), all_preds.cpu(), average="micro")
        rec_macro = recall_score(all_gt.cpu(), all_preds.cpu(), average="macro")
        prec_micro = precision_score(all_gt.cpu(), all_preds.cpu(), average="micro")
        prec_macro = precision_score(all_gt.cpu(), all_preds.cpu(), average="macro")
        print('f1 micro: ', f1_micro)
        print('f1 macro: ', f1_macro)
        print('recall micro: ', rec_micro)
        print('recall macro: ', rec_macro)
        print('precision micro: ', prec_micro)
        print('precision macro: ', prec_macro)
        f1_sur2.append(f1_micro)
        if final_val_loss < best_val_loss and f1_micro > best_f1:
            best_val_loss = final_val_loss
            best_f1 = f1_micro
            # Save the model
            torch.save(model_2.state_dict(), dest+f'quality_checker_surrogate_long_caps_exp.pth')
            print('Model saved! best model loss: {:.4f}'.format(final_val_loss))
            print('Model saved! best model f1: {:.4f}'.format(f1_micro))
            current_patience = 0  # Reset patience counter
        else:
            current_patience += 1
            print(f'Validation loss has not improved for {current_patience} epochS.')
        if current_patience >= patience:
            print(f'Validation f1 ({f1_micro}) has not improved for {patience} epochs. STOPPING TRAINING.')
            break
        print('##########################################################################################')
        scheduler.step(final_val_loss)
    # Testing phase

    criterion = torch.nn.BCEWithLogitsLoss()
    # Instantiate the model and move it to GPU if available
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print('input size test: ', test_x_pt.size(-1))
    loaded_model = VisualSurrogate(image_embedding_dim,text_embedding_dim,num_heads,
                                            hidden_dim, num_layers, num_classes, num_labels, mode='BCELoss')
    loaded_model = torch.nn.DataParallel(loaded_model)
    loaded_model = loaded_model.to(device)
    checkpoint = torch.load(dest+f'quality_checker_surrogate_long_caps_exp.pth') #just be sure batch sizes are same. Otherwise there will be model 
    loaded_model.load_state_dict(checkpoint)

    loaded_model.eval()
    total_correct = 0
    total_samples = 0

    all_acc=[]
    all_f1=[]
    all_rec=[]
    all_prec=[]
    sigmoid_matrices = []
    full_batch = []
    final_test_loss = 0
    total_val_samples = 0
    count_iter=0
    pred_test = []
    total_test_loss=0

    start = time.time()
    with torch.no_grad():
        for batch_images, batch_inputs, batch_targets in tqdm(test_loader):
            batch_images, batch_inputs, batch_targets = batch_images.to(device), batch_inputs.to(device), batch_targets.to(device)#, batch_wt_train.to(device)
            outputs = model_2(batch_images, batch_inputs)
            
            if loss_type=='CELoss':
                    individual_outputs = torch.argmax(outputs, dim=2) # convert to nominals
                    sigmoid_matrices.append(outputs)
                    full_batch.append(batch_targets)
                    pred_mat.append(individual_outputs)
                    #print(outputs)
                    #print('ground truth: ', batch_targets)
                    #print('sigmoid outputs: ', sigmoid_outputs)
                    val_loss = criterion(outputs.view(-1, num_labels), 
                                        batch_targets.to(torch.int64).view(-1))
                    
                    total_val_loss += val_loss.item() #* batch_inputs.size(0)
            elif loss_type=='BCELoss':
                loss = criterion(outputs, batch_targets)
                # Apply sigmoid to get probabilities
                probs = torch.sigmoid(outputs)
                # Apply threshold to get binary predictions
                individual_outputs = (probs > 0.5).int()
                sigmoid_matrices.append(outputs)
                full_batch.append(batch_targets)
                pred_test.append(individual_outputs)
                val_loss = criterion(outputs, batch_targets)
                
                total_val_loss += val_loss.item() #* batch_inputs.size(0)
            #total_predictions = outputs.size(0)*outputs.size(1)
        final_test_loss = total_test_loss/ len(test_loader)    
        gt = torch.stack([item for sublist in full_batch for item in sublist])
        all_preds = torch.stack([item for sublist in pred_test for item in sublist])
        f1_micro = f1_score(gt.cpu(), all_preds.cpu(), average="micro") 
        # print classwise f1
        y_true=gt.cpu()
        y_pred=all_preds.cpu()
        #torch.save(y_true,dest+'sec_sur_gt.pt')
        #torch.save(y_pred,dest+'sec_sur_pred.pt')
        avg_f1=0
        res_mlc_f1 = {}
        for i, label in enumerate(classes):
            res_mlc_f1['f1_' + label]=round((f1_score(y_true[:,i], y_pred[:,i],labels=[1], average=None, zero_division=0)[0])*100,2)
            #print(f1)
            avg_f1 += res_mlc_f1['f1_' + label]
        res_mlc_f1['avg_f1'] = avg_f1 / len(classes)
        print('res_mlc_f1: ',res_mlc_f1)

        avg_recall=0
        res_mlc_recall = {}
        for i, label in enumerate(classes):
            res_mlc_recall['recall_' + label]=round((recall_score(y_true[:,i], y_pred[:,i],labels=[1], average=None, zero_division=0)[0])*100,2)
            #print(f1)
            avg_recall += res_mlc_recall['recall_' + label]
        res_mlc_recall['avg_recall'] = avg_recall / len(classes)
        print('res_mlc_recall: ',res_mlc_recall)

        avg_preci=0
        res_mlc_precision = {}
        for i, label in enumerate(classes):
            res_mlc_precision['precision_' + label]=round((precision_score(y_true[:,i], y_pred[:,i],labels=[1], average=None, zero_division=0)[0])*100,2)
            #print(f1)
            avg_preci += res_mlc_precision['precision_' + label]
        res_mlc_recall['avg_preci'] = avg_preci / len(classes)
        print('res_mlc_precision: ',res_mlc_precision)
        for i in range(y_true.size(1)):
            gt = y_true[:,i]
            pred=y_pred[:,i]
            #tn, fp, fn, tp = sklearn.metrics.confusion_matrix(gt, pred).ravel()
            tp = ((gt == 1) & (pred == 1)).sum().item()
            fp = ((gt == 0) & (pred == 1)).sum().item()
            tn = ((gt == 0) & (pred == 0)).sum().item()
            fn = ((gt == 1) & (pred == 0)).sum().item()
            print(classes[i])
            print(f'tp: {tp}, fp: {fp}, tn: {tn}, fn: {fn}')

        print('f_micro test: ', f1_micro)  
        print('test loss: ', final_test_loss) 
       

    # Plot the necessary curved


    # Surrogate 2
    plt.style.use('ggplot')
    fig_sur2_loss=plt.figure(figsize=(16,9))
    plt.rcParams.update({'font.size': 30})
    plt.plot(train_loss_box_sur2, label='train loss', color='red', linewidth=2.0)
    plt.plot(val_loss_box_sur2, label='validation loss', color='blue', linewidth=2.0)
    plt.xticks(fontsize=30)
    plt.yticks(fontsize=30)
    plt.legend()
    plt.title('loss curve for surrogate 2')
    plt.savefig('plots/'+f'quality_checker_loss_new.png')
    plt.close(fig_sur2_loss)

    # F1 @ surrogate2
    plt.style.use('ggplot')
    fig_sur2_f1=plt.figure(figsize=(16,9))
    plt.rcParams.update({'font.size': 30})
    plt.plot(f1_sur2, color='red', linewidth=2.0)
    plt.xticks(fontsize=30)
    plt.yticks(fontsize=30)
    #plt.legend()
    plt.title('F1 curve for surrogate 2')
    plt.savefig('plots/'+f'quality_checker_f1_new.png')
    plt.close(fig_sur2_f1)

if __name__ == '__main__':
    main()
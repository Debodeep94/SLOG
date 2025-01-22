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
    parser.add_argument("--proportion",type=float)
    parser.add_argument("--patience",type=int)
    args = parser.parse_args()
    return args

def main():
    args = parse_agrs()
    prop = args.proportion

    # code for second surrogate
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print('Currently usable device is: ', device)
    torch.cuda.empty_cache()
    print(torch.cuda.memory_summary())
    base_path='n/surrogate/'
    path=base_path#+'surrogate1/'
    text_embeddings = torch.load(path+'sur2_data_gt_emb_full.pt', map_location=torch.device('cpu'))  # shape: [num_samples, image_embedding_dim]
    image_vectors = torch.load(path+'sur2_img_data.pt', map_location=torch.device('cpu'))  # shape: [num_samples, seq_len, embedding_dim]
    outputs_preds= torch.load(path+'val_accs_pred_emb_full.pt', map_location=torch.device('cpu'))
    outputs_gt= torch.load(path+'val_accs_gt_emb_full.pt', map_location=torch.device('cpu'))
    text_embeddings_pred = torch.load(path+'sur2_data_pred_emb_full.pt', map_location=torch.device('cpu'))  # shape: [num_samples, image_embedding_dim]
    text_embeddings = torch.stack([item for sublist in text_embeddings for item in sublist])
    text_embeddings_pred=torch.stack([item for sublist in text_embeddings_pred for item in sublist])
    surr_2_x=torch.cat((text_embeddings,text_embeddings_pred), dim=0)
    image_vectors= torch.stack([item for sublist in image_vectors for item in sublist])
    print(image_vectors.size())
    image_vectors = image_vectors.repeat(2, 1, 1, 1)
    surr_2_y_gt = torch.stack([item for sublist in outputs_gt for item in sublist])
    surr_2_y_pred = torch.stack([item for sublist in outputs_preds for item in sublist])
    surr_2_y = torch.cat((surr_2_y_gt,surr_2_y_pred), dim=0)
    #outputs=torch.stack(outputs)
    print(text_embeddings.size())
    print(text_embeddings_pred.size())
    print(image_vectors.size())
    print(surr_2_y.size())
    # surr_2_y = surr_2_y.detach().numpy()
    assert image_vectors.size(0) == surr_2_x.size(0), "Mismatch in number of samples between images and text embeddings"

    print(torch.cuda.memory_summary())
    # print(f'sur2y: {surr_2_y.size()}')
    # print(f'surseq: {surr_2_seq.size()}')

    X = surr_2_x.detach().numpy()
    Y = surr_2_y.detach().numpy()
    # seq_data = surr_2_seq.detach().numpy()

    im_train, im_test, train_x, test_x, train_y, test_y=train_test_split(image_vectors, X,Y,test_size=0.1, random_state=1)
    im_train,im_val, train_x, val_x, train_y, val_y = train_test_split(im_train, train_x, train_y, test_size=0.20, random_state=1) # 0.25 x 0.8 = 0.2
    #weight_train = torch.ones(len(train_x))

    train_x_pt = torch.tensor(train_x,dtype=torch.float) #pt: pytorch
    #train_x_pt = train_x_pt/torch.norm(train_x_pt, dim = 1, keepdim= True)
    #print('train_x_surr2: ', train_x_pt)
    # seq_tr=torch.tensor(seq_tr, dtype=torch.int)
    val_x_pt = torch.tensor(val_x,dtype=torch.float) #pt: pytorch
    #val_x_pt = val_x_pt/torch.norm(val_x_pt, dim = 1, keepdim= True)
    # seq_val=torch.tensor(seq_val, dtype=torch.int)
    test_x_pt = torch.tensor(test_x,dtype=torch.float)
    #test_x_pt = test_x_pt/torch.norm(test_x_pt, dim = 1, keepdim= True)
    # seq_test=torch.tensor(seq_test, dtype=torch.int)
    train_y = torch.tensor(train_y)

    val_y = torch.tensor(val_y)
    test_y = torch.tensor(test_y)

    # Instantiate the model
    image_embedding_dim = 2048  # Based on ResNet-50 output
    text_embedding_dim = 512  # Example dimension, adjust based on your embeddings
    num_heads = 4
    hidden_dim = 256
    num_layers = 6
    num_classes = 14
    num_labels=2
    batch_size = 256
    # Create DataLoader for training set, validation set, and test set
    train_dataset = TensorDataset(im_train, train_x_pt, train_y)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

    val_dataset = TensorDataset(im_val, val_x_pt, val_y)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

    test_dataset = TensorDataset(im_test, test_x_pt, test_y)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    classes=['No Finding', 'Enlarged Cardiomediastinum', 'Cardiomegaly',
       'Lung Lesion', 'Lung Opacity', 'Edema', 'Consolidation', 'Pneumonia',
       'Atelectasis', 'Pneumothorax', 'Pleural Effusion', 'Pleural Other',
       'Fracture', 'Support Devices']
    #Save the train and validation splits
    #torch.save(train_y,dest+'sec_sur_train_y.pt')
    #torch.save(val_y,dest+'sec_sur_val_y.pt')
    #torch.save(test_y,dest+'sec_sur_test_y.pt')

    
    # # experiment with LSTM attention
    # lstm_model=LSTM_Attn(texts_pred.size(-1), hidden_dim)
    # lstm_model=lstm_model.to(device)

    #text_model = LSTM_Attn(input_size=512, hidden_size=256)
    model = VisualSurrogate(image_embedding_dim, text_embedding_dim, 
                            num_heads, hidden_dim, num_layers, num_classes, 
                            num_labels, mode='CELoss')
    #model = MultimodalModel(text_model, image_embedding_dim, hidden_dim, num_classes)
    #model = CombinedModel(text_embedding_dim, num_heads, hidden_dim, num_layers, num_classes, num_labels)
    # model = torch.nn.DataParallel(model)
    model=model.to(device)
    # Hyperparameters
    weights = count_weights(train_y,1)
    learning_rate = 2e-5
    criterion = nn.CrossEntropyLoss(weight=weights.to(device))  # Since each class label can be 0, 1, or 2
    criterion_val = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    scheduler = ReduceLROnPlateau(optimizer, mode='max', patience=10, factor=0.01, verbose=True, min_lr=1e-6)
    # Training loop
    def train(model, train_loader, criterion, optimizer):
        #for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0
        all_preds = []
        all_labels = []
        
        for images, texts, labels in tqdm(train_loader):#, desc=f'Epoch {epoch + 1}/{num_epochs}'):
            images, texts, labels = images.to(device), texts.to(device), labels.to(device)
            optimizer.zero_grad()
            
            outputs = model(images, texts)
            # batch_size, num_classes, num_labels = outputs.size()  # Should be [batch_size, 14, 3]
            # exp_out=lstm_model(texts)
            # Take the max along the last dimension to get the predicted class for each of the 14 classes
            _, preds = torch.max(outputs, dim=-1)  # preds shape: [batch_size, num_classes]

            loss = criterion(outputs.view(-1, num_labels), labels.view(-1).to(torch.int64))
            loss.backward()
            optimizer.step()
            clip_gradient(optimizer, 1)
            running_loss += loss.item()
            all_preds.append(preds.cpu().numpy())
            all_labels.append(labels.cpu().numpy())
        # Concatenate all batches
        # all_preds = np.vectorize(positive)(np.concatenate(all_preds, axis=0))
        # all_labels = np.vectorize(positive)(np.concatenate(all_labels, axis=0))
        all_preds = np.concatenate(all_preds, axis=0)
        all_labels = np.concatenate(all_labels, axis=0)
        epoch_loss = running_loss / len(train_loader)
        #print(f'Epoch [{epoch + 1}/{num_epochs}], Loss: {epoch_loss:.4f}')
        outcomes = compute_mlc_f1(all_labels, all_preds, classes)
        f1=outcomes['F1_MICRO']
        print('train f1:', f1)
        return epoch_loss, outcomes
            
            # # Validation (optional)
            # validate(model, val_loader)

    def validate(model, val_loader):
        model.eval()
        all_gt_preds = []
        all_preds_preds=[]
        all_labels = []
        val_loss = 0.0
        val_loss_on_preds=0
        with torch.no_grad():
            for images, texts, labels in val_loader:
                images, texts, labels = images.to(device), texts.to(device), labels.to(device)
                # validate with gt split
                outputs = model(images, texts)
                _, preds = torch.max(outputs, dim=-1)  # preds shape: [batch_size, num_classes]
                all_gt_preds.append(preds.cpu().numpy())
                all_labels.append(labels.cpu().numpy())
                
                labels = labels.view(-1).to(torch.int64)
                outputs = outputs.view(-1, 2)
                loss = criterion(outputs, labels)
                val_loss += loss.item()
        # Concatenate all batches
        all_gt_preds = np.concatenate(all_gt_preds, axis=0)
        all_labels = np.concatenate(all_labels, axis=0)
        final_val_loss = val_loss/ len(val_loader)
        outcomes = compute_mlc_f1(all_labels, all_gt_preds, classes)
        f1_on_gt = outcomes['F1_MICRO']
        print('val f1: ', f1_on_gt)
        
        # print('validation micro F1:',  f1['F1_MICRO'])
        # print('validation macro F1: ',  f1['F1_MACRO'])
        # if f1['F1_MICRO'] > best_f1:
        #     best_f1=f1['F1_MICRO']
        #     print('f1 scores improved, saving')
        #     torch.save(model.state_dict(), path+f'visual_surrogate.pth')
        #     print('Model saved! best f1: {:.4f}'.format(best_f1))
        #     #current_patience = 0  # Reset patience counter
        #     print('-------------------------------------------------------------------------------')
        # else:
        #     current_patience += 1
        #     #if current_patience >= patience:
        #     print(f'Validation f1 has not improved. Continue training.')
        #     print('-------------------------------------------------------------------------------')
        return final_val_loss, outcomes    

    # Assuming you have DataLoader objects `train_loader` and `val_loader` ready
    num_epochs=500
    best_val_f1=0
    current_patience=0
    patience=15
    best_val_loss = float('inf')

    for epoch in range(num_epochs):
        tr_loss, tr_f1= train(model, train_loader, criterion, optimizer)
        val_loss, val_f1_on_gt=validate(model, val_loader)
        tr_f1_mic= tr_f1['F1_MICRO']
        tr_f1_mac= tr_f1['F1_MACRO']
        val_f1_mic= val_f1_on_gt['F1_MICRO']
        val_f1_mac= val_f1_on_gt['F1_MACRO']
        # val_f1_mic_on_preds= val_f1_on_preds['F1_MICRO']
        # val_f1_mac_on_preds= val_f1_on_preds['F1_MACRO']
        print(f'Epoch [{epoch+1}/{num_epochs}],\t| Train Loss: {tr_loss},\t|\n Train F1 MICRO: {tr_f1_mic},\t| Train F1 MACRO: {tr_f1_mac}\t|\n Val Loss: {val_loss},\t|\n Val F1 MICRO: {val_f1_mic},\t| Val F1 MACRO: {val_f1_mac}')
        
        if val_f1_on_gt['F1_MICRO'] > best_val_f1:
            current_patience = 0
            best_val_f1 = val_f1_on_gt['F1_MICRO']
            best_val_loss = val_loss
            torch.save(model.state_dict(), path+'sur2_best_model_new.pt')
            print('Saved best model with F1:', best_val_f1)
        else:
            print('Validation F1 did not improve')
            current_patience = current_patience+1
        if current_patience == patience:
            print(f'Validation F1 did not improve for {patience} epochs, training aborted.')
            break
        print('===================================================================================================')


    

    # Testing phase
    all_gt_preds = []
    all_preds_preds=[]
    all_labels = []
    test_loss = 0.0
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print('input size test: ', test_x_pt.size(-1))
    loaded_model = VisualSurrogate(image_embedding_dim,text_embedding_dim,num_heads,
                                            hidden_dim, num_layers, num_classes, num_labels, mode='CELoss')
    # loaded_model = torch.nn.DataParallel(loaded_model)
    loaded_model = loaded_model.to(device)
    checkpoint = torch.load(base_path+f'sur2_best_model.pt') #just be sure batch sizes are same. Otherwise there will be model 
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
    avg_f1 = 0
    pred_test = []
    total_test_loss=0
    # loss_type=='CELoss'
    start = time.time()
    with torch.no_grad():
        for images, texts, labels in tqdm(test_loader):
            images, texts, labels = images.to(device), texts.to(device), labels.to(device)
            # validate with gt split
            outputs = loaded_model(images, texts)
            _, preds = torch.max(outputs, dim=-1)  # preds shape: [batch_size, num_classes]
            all_gt_preds.append(preds.cpu().numpy())
            all_labels.append(labels.cpu().numpy())
            
            labels = labels.view(-1).to(torch.int64)
            outputs = outputs.view(-1, 2)
            loss = criterion(outputs, labels)
            test_loss += loss.item()
        # Concatenate all batches
        all_gt_preds = np.concatenate(all_gt_preds, axis=0)
        all_labels = np.concatenate(all_labels, axis=0)
        final_test_loss = test_loss/ len(val_loader)
        outcomes = compute_mlc_f1(all_labels, all_gt_preds, classes)
        print(outcomes)
        f1_on_gt = outcomes['F1_MICRO']
        f1_on_gt_mac = outcomes['F1_MACRO']
        y_pred = all_gt_preds#.cpu()
        y_true = all_labels#.cpu()
        print('test f1: ', f1_on_gt)

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
        # for i in range(y_true.size(1)):
        #     gt = y_true[:,i]
        #     pred=y_pred[:,i]
        #     #tn, fp, fn, tp = sklearn.metrics.confusion_matrix(gt, pred).ravel()
        #     tp = ((gt == 1) & (pred == 1)).sum().item()
        #     fp = ((gt == 0) & (pred == 1)).sum().item()
        #     tn = ((gt == 0) & (pred == 0)).sum().item()
        #     fn = ((gt == 1) & (pred == 0)).sum().item()
        #     print(classes[i])
        #     print(f'tp: {tp}, fp: {fp}, tn: {tn}, fn: {fn}')

        print('f_micro test: ', f1_on_gt) 
        print('f1 macro: ', f1_on_gt_mac) 
        print('prec micro: ', round((100*outcomes['PRECISION_MICRO']),2)) 
        print('prec macro: ', round((100*outcomes['PRECISION_MACRO']),2))
        print('rec micro: ', round((100*outcomes['RECALL_MICRO']),2)) 
        print('rec macro: ', round((100*outcomes['RECALL_MACRO']),2)) 
        print('test loss: ', final_test_loss) 
    # Plot the necessary curved


    # # Surrogate 2
    # plt.style.use('ggplot')
    # fig_sur2_loss=plt.figure(figsize=(16,9))
    # plt.rcParams.update({'font.size': 30})
    # plt.plot(train_loss_box_sur2, label='train loss', color='red', linewidth=2.0)
    # plt.plot(val_loss_box_sur2, label='validation loss', color='blue', linewidth=2.0)
    # plt.xticks(fontsize=30)
    # plt.yticks(fontsize=30)
    # plt.legend()
    # plt.title('loss curve for surrogate 2')
    # plt.savefig('plots/'+f'surrogate_2_loss_prop_{prop}_new.png')
    # plt.close(fig_sur2_loss)

    # # F1 @ surrogate2
    # plt.style.use('ggplot')
    # fig_sur2_f1=plt.figure(figsize=(16,9))
    # plt.rcParams.update({'font.size': 30})
    # plt.plot(f1_sur2, color='red', linewidth=2.0)
    # plt.xticks(fontsize=30)
    # plt.yticks(fontsize=30)
    # #plt.legend()
    # plt.title('F1 curve for surrogate 2')
    # plt.savefig('plots/'+f'surrogate_2_f1_prop_{prop}_new.png')
    # plt.close(fig_sur2_f1)

if __name__ == '__main__':
    main()
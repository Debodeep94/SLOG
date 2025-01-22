import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader
from sklearn.metrics import accuracy_score, f1_score, recall_score, precision_score
from sklearn.preprocessing import MinMaxScaler
import torch.nn.functional as F
import torchvision.models as models

# Define the visual extractor
class VisualModel(nn.Module):
    def __init__(self, visual_extractor='resnet101'):
        super(VisualModel, self).__init__()
        #device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        vis_model = getattr(models, visual_extractor, )(pretrained=True)#
        modules = list(vis_model.children())[:-2]  # Remove the last two layers
        self.vis_model = nn.Sequential(*modules)
        self.avg_fnt = torch.nn.AvgPool2d(kernel_size=7, stride=1, padding=0)

    def forward(self, images):
        patch_feats = self.vis_model(images)
        avg_feats = self.avg_fnt(patch_feats).squeeze().reshape(-1, patch_feats.size(1))
        batch_size, feat_size, _, _ = patch_feats.shape
        patch_feats = patch_feats.reshape(batch_size, feat_size, -1).permute(0, 2, 1)
        return patch_feats, avg_feats

# Define the text feature extractor
class TextFeatureExtractor(nn.Module):
    def __init__(self, embedding_dim, num_heads, hidden_dim, num_layers):
        super(TextFeatureExtractor, self).__init__()
        # self.embedding = nn.Embedding(vocab_size, embedding_dim)
        encoder_layers = nn.TransformerEncoderLayer(embedding_dim, num_heads, hidden_dim)
        self.transformer_encoder = nn.TransformerEncoder(encoder_layers, num_layers)
        #self.positional_encoding = nn.Parameter(torch.zeros(1, 70, embedding_dim))  # Assume max length of 70

    def forward(self, x):
        x = x #+ self.positional_encoding[:, :x.size(1), :]
        x = x.permute(1, 0, 2)  # Transformer expects [seq_len, batch_size, embedding_dim]
        x = self.transformer_encoder(x)
        x = x.permute(1, 0, 2)  # Convert back to [batch_size, seq_len, embedding_dim]
        return x

class VisualSurrogate(nn.Module):
    def __init__(self, image_embedding_dim, text_embedding_dim, num_heads,
                  hidden_dim, num_layers, num_classes, num_labels, mode='BCELoss'):
        super(VisualSurrogate, self).__init__()
        self.image_extractor = VisualModel()
        self.text_extractor = TextFeatureExtractor(text_embedding_dim, num_heads, hidden_dim, num_layers)
        #self.text_extractor = LSTM_Attn(texts_pred.size(-1), hidden_dim//2)
        self.fc1 = nn.Linear(image_embedding_dim + text_embedding_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, num_labels*num_classes)  # Output layer for 14 classes with 3 possible values each
        self.fc_bce = nn.Linear(hidden_dim, num_classes) #only for bce
        self.bn = nn.BatchNorm1d(hidden_dim)  # Batch Normalization layer after first fully connected layer
        self.dropout = nn.Dropout(0.2)
        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()
        self.fc1 = self.fc1
        self.fc2 = self.fc2
        self.fc_bce = self.fc_bce
        self.num_labels = num_labels
        self.num_classes = num_classes
        self.mode=mode
    def forward(self, images, text_embeddings):
        # process image features acc to https://medium.com/@vinitwaingankar/rhythmicradar-classification-of-music-using-a-multimodal-approach-using-cross-modal-attention-0b52b269922b
        _, image_features = self.image_extractor(images)
        # image_features= self.relu(image_features)
        # image_features=self.relu(image_features)
        text_features = self.text_extractor(text_embeddings)
        text_features = text_features.mean(dim=1)  # Average over the sequence length
        #print('text_features_out: ',text_features.size())
        combined_features = torch.cat((image_features, text_features), dim=1)
        x = self.relu(self.fc1(combined_features))
        # x = self.fc2(x)
        if self.mode=='BCELoss':
            #x = x.view(x.size(0), self.num_classes, self.num_labels) # Reshape to [batch_size, 14, 3]
            return self.fc_bce(x)
        elif self.mode=='CELoss':
            x = self.fc2(x)
            x = x.view(x.size(0), self.num_classes, self.num_labels) # Reshape to [batch_size, 14, 3]
            return x
        else:
            raise ValueError

# A surrogate based on tanh attention

class TanhAttention(nn.Module):
    def __init__(self, hidden_size, num_labels, dropout=0.5):
        super(TanhAttention, self).__init__()
        self.attn1 = nn.Linear(hidden_size, hidden_size // 2)
        self.attn2 = nn.Linear(hidden_size // 2, 1, bias=False)
        self.dropout = nn.Dropout(p=dropout)
        self.fc = nn.Linear(hidden_size, num_labels)

    def forward(self, output):
        attn1 = torch.tanh(self.attn1(output))
        attn2 = self.attn2(attn1).squeeze(-1)  # Compute attention scores
        attn = F.softmax(attn2, dim=1)  # Apply softmax to normalize scores

        h = output.transpose(1, 2).matmul(attn.unsqueeze(2)).squeeze(2)  # Apply attention weights
        h = self.dropout(h)

        return h



class LSTM_Attn(nn.Module):
    def __init__(self, input_size, hidden_size, num_labels, num_classes=5):
        super(LSTM_Attn, self).__init__()

        self.rnn = nn.LSTM(input_size=input_size, hidden_size=hidden_size, batch_first=True, bidirectional=True)
        self.attns = nn.ModuleList([TanhAttention(hidden_size * 2, num_labels) for _ in range(num_classes)])

    def forward(self, encoded_captions):
        x = encoded_captions.to(torch.float32)
        # print(x.size())
        output, (_, _) = self.rnn(x)

        # Apply the attention mechanism to the LSTM outputs
        features = [attn(output) for attn in self.attns]
        features = torch.stack(features, dim=1)

        return features

    

class AttentionVisualSurrogate(nn.Module):
    def __init__(self, image_embedding_dim,text_embedding_dim,
                  hidden_dim, num_classes, num_labels, mode='BCELoss'):
        super(AttentionVisualSurrogate, self).__init__()
        self.image_extractor = VisualModel()
        print(image_embedding_dim)
        #self.text_extractor = TextFeatureExtractor(text_embedding_dim, num_heads, hidden_dim, num_layers)
        self.text_extractor = LSTM_Attn(text_embedding_dim, hidden_dim,num_labels)
        self.fc1 = nn.Linear(image_embedding_dim + hidden_dim * 2, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, num_classes * num_labels)  # Output layer for 14 classes with 3 possible values each
        self.fc_bce = nn.Linear(hidden_dim, num_classes) #only for bce
        self.bn = nn.BatchNorm1d(hidden_dim)  # Batch Normalization layer after first fully connected layer
        self.dropout = nn.Dropout(0.2)
        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()
        self.relu = self.relu
        self.fc1 = self.fc1
        self.fc2 = self.fc2
        self.num_labels = num_labels
        self.num_classes = num_classes
        self.mode=mode
    def forward(self, images, encoded_captions):
        # process image features acc to https://medium.com/@vinitwaingankar/rhythmicradar-classification-of-music-using-a-multimodal-approach-using-cross-modal-attention-0b52b269922b
        _, image_features = self.image_extractor(images)
        # image_features= self.relu(image_features)
        # image_features=self.relu(image_features)
        text_features = self.text_extractor(encoded_captions )
        # print(f"Text features shape: {text_features.size()}")
        text_features = text_features.mean(dim=1)  # Average over the sequence length
        image_features = image_features
        # # Print shapes for debugging
        # print(f"Image features shape: {image_features.size()}")
        # print(f"Text features shape: {text_features.size()}")

        # Concatenate features
        combined_features = torch.cat((image_features, text_features), dim=1)
        # print(f"Combined features shape: {combined_features.size()}")

        # Ensure fc1 input dimension matches the combined features dimension
        #assert combined_features.shape[1] == image_embedding_dim + 2*text_embedding_dim, "Dimension mismatch in concatenated features."
        x = self.relu(self.fc1(combined_features))
        
        if self.mode=='BCELoss':
            #x = x.view(x.size(0), self.num_classes, self.num_labels) # Reshape to [batch_size, 14, 3]
            return self.fc_bce(x)
        elif self.mode=='CELoss':
            x = self.fc2(x)
            x = x.view(x.size(0), self.num_classes, self.num_labels) # Reshape to [batch_size, 14, 3]
            return x
        else:
            raise ValueError
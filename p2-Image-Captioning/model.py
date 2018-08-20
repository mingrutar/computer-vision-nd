import torch
import torch.nn as nn
import torchvision.models as models


class EncoderCNN(nn.Module):
    def __init__(self, embed_size):
        super(EncoderCNN, self).__init__()
        resnet = models.resnet50(pretrained=True)
        for param in resnet.parameters():
            param.requires_grad_(False)
        
        modules = list(resnet.children())[:-1]
        self.resnet = nn.Sequential(*modules)
        self.embed = nn.Linear(resnet.fc.in_features, embed_size)

    def forward(self, images):
        features = self.resnet(images)
        features = features.view(features.size(0), -1)
        features = self.embed(features)
        return features
    

class DecoderRNN(nn.Module):
    def __init__(self, embed_size, hidden_size, vocab_size, num_layers=1,dropout=0.2):
        super(DecoderRNN, self).__init__()
        self.hidden_size = hidden_size
        self.word_embeddings = nn.Embedding(num_embeddings=vocab_size, embedding_dim=embed_size)
        self.lstm = nn.LSTM(embed_size, hidden_size=hidden_size, num_layers=num_layers, dropout=dropout, batch_first=True)
        self.dropout = nn.Dropout(p=dropout)
        self.fc = nn.Linear(hidden_size, vocab_size)
    
    def forward(self, features, captions):
        usq_features = features.unsqueeze(1)   # add a dim at pos 1
        embedding_vector = self.word_embeddings(captions[:, :-1])  # get rid of <end>
        lsmt_in = torch.cat((usq_features, embedding_vector), 1)
#        print("forward: features=", features.shape,"captions=",captions.shape,"usq_features=",usq_features.shape,"embedding_vector=", embedding_vector.shape, "lsmt_in=", lsmt_in.shape)
# forward: features= torch.Size([10, 256]) captions= torch.Size([10, 15]) usq_features= torch.Size([10, 1, 256]) embedding_vector= torch.Size([10, 14, 256]) lsmt_in= torch.Size([10, 15, 256])
        lsmt_out, hidden = self.lstm(lsmt_in)
        dropout_out = self.dropout(lsmt_out)
        fc_out = self.fc(dropout_out)
        return fc_out
    
    # 
    def sample(self, inputs, states=None, max_len=20):
        " accepts pre-processed image tensor (inputs) and returns predicted sentence (list of tensor ids of length max_len) "
        ret = []
        for i in range(max_len):
            output, states = self.lstm(inputs, states)
            output = self.fc(output.view(len(output), -1)) #?out = self.score(out.view(len(out), -1))
            max_idx = output.max(1)[1]
            ret.append(max_idx.item())
            inputs = self.word_embeddings(max_idx).unsqueeze(1)
        return ret

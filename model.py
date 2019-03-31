import torch
import torch.nn as nn
import torchvision.models as models


class EncoderCNN(nn.Module):
    def __init__(self, embed_size):
        super(EncoderCNN, self).__init__()
        resnet = models.resnet50(pretrained=True) #resnet50, consider resnet101 (slight gain but training time increase?)
        
        #don't backpropagate
        for param in resnet.parameters():
            param.requires_grad_(False)
        
        modules = list(resnet.children())[:-1]  #clip head for new classification features
        
        self.resnet = nn.Sequential(*modules)
        
        #bn the output and before the fc layer
        self.batchnorm_2d = nn.BatchNorm2d(resnet.fc.in_features)
        
        
        self.linear = nn.Linear(resnet.fc.in_features, embed_size)
        torch.nn.init.kaiming_normal_(self.linear.weight)
        
        #bn the final output
        self.bnFinal = nn.BatchNorm1d(embed_size)

    def forward(self, images):
        
        #cnn eval image
        features = self.resnet(images)
        
        #batchnormalize
        features = self.batchnorm_2d(features)
        
        #resize for fc
        features = features.view(features.size(0), -1)
        
        features = self.linear(features)
        
        features = self.bnFinal(features)  #normalize
        
        return features
    

class DecoderRNN(nn.Module):
    
    def __init__(self, embed_size, hidden_size, vocab_size):
        super().__init__()  #init parent class
        
        #self.device = device
        
        print("hidden_size param :",hidden_size)
        
        #create embedding tensor for vocab
        self.word_embeddings = nn.Embedding(vocab_size, embed_size)
        
        #store hidden size and init to zeroes
        self.hidden_size = hidden_size
        
        #self.hidden = self.init_hidden(num_layers=1, batch_size=10)
        self.hidden = (torch.zeros(1, 1, hidden_size),torch.zeros(1, 1, hidden_size)) 
        self.num_layers = 1
      
        #create lstm
        self.lstm = nn.LSTM(input_size=embed_size, \
                            hidden_size=hidden_size,
                            num_layers=self.num_layers, 
                            bias=True,
                            batch_first=True,  #batch size is first
                            dropout=0, #test with variations of this..paper indicates dropout
                            bidirectional=False
                           )

        # The linear layer that maps from hidden state space to vocab space
        self.linear = nn.Linear(hidden_size, vocab_size)
        
        #init
        torch.nn.init.kaiming_normal_(self.linear.weight) #linear init
        torch.nn.init.kaiming_normal_(self.word_embeddings.weight) #embedding
        
    
    def forward(self, features, captions):
        
        print("in forward - features tensor is ",type(features))
        
        captions_embedding = self.word_embeddings(captions[:,:-1])  #clip <end>
        
        #concat image features and caption embeddings
        inputs = torch.cat((features.unsqueeze(dim=1),captions_embedding), dim=1)

        #embeddings = torch.cat((features.unsqueeze(1), captions_embedding), 1)

        lstm_out, self.hidden = self.lstm(inputs) # lstm_out shape : batch_size, caption length, hidden_size

        # pass to linear layer to finalize
        outputs = self.linear(lstm_out) # outputs shape : batch_size, caption length, vocab_size

        return outputs
#########
      
    def sample(self, inputs, states=None, max_len=20):
        " accepts pre-processed image tensor (inputs) and returns predicted sentence (list of tensor ids of length max_len) "
        sentence = []
        
      
        
        for i in range(max_len):
            #pass image into lstm
            
            hiddens, states = self.lstm(inputs, states) 
            #print("hiddens shape: ",hiddens.shape)  ->  1,1,512
            
            outputs = self.linear(hiddens.squeeze(1)) 
            #print("outputs from self.linear: ",outputs.shape) -> 1,8855
            
            #get the highest/best word score
            _, predicted = outputs.max(1)  #max
            
            #add to sentence
            sentence.append(predicted.item())
            
            #prepare for next word
            inputs = self.word_embeddings(predicted)     #loop in word for next step of lstm as input
            #print("input for next layer: ",inputs.shape) -> 1,512
            inputs = inputs.unsqueeze(1)  
            #print("input after unsquezer: ",inputs.shape) -> 1,1,512
        
        return sentence  
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
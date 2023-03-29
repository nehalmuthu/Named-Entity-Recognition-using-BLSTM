import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import numpy as np
from torch.nn.utils.rnn import pad_sequence, pack_padded_sequence, pad_packed_sequence
from torch.nn.modules import padding
from torch.optim.lr_scheduler import StepLR


train_path='./data/train'
dev_path='./data/dev'
test_path='./data/test'
model_path='./model/blstm2.pt'
glove_path='./glove.6B.100d/glove.6B.100d.txt'



with open(train_path, "r") as f:
  d={}
  for line in f:
    line = line.strip()
    if len(line) != 0:
      parts = line.split(" ")
      label = parts[2]
      d[label]=1+d.get(label,0)


print('distribution of labels')
print(d)
print("\n")



#https://www.kaggle.com/code/fyycssx/first-try-lstm-with-glove-by-pytorch
embeddings_dictionary = dict()
glove_file = open(glove_path, encoding="utf8")
word2idx={}
for line in glove_file:
    records = line.split()
    word = records[0]
    vector_dimensions = np.asarray(records[1:], dtype='float32')
    embeddings_dictionary [word] = vector_dimensions
    if word not in word2idx:
      word2idx[word] = len(word2idx)
glove_file.close()


word2idx['<unk>']=len(word2idx)
word2idx['<unkcap>']=len(word2idx)
word2idx['<pad>']=len(word2idx)


class NERDataset(Dataset):
    def __init__(self, filename, word2idx):
        self.word2idx = word2idx
        self.label2idx = {}
        self.max_sent_len = 0
        self.data = []
        
        with open(filename, "r") as f:
            sentence, labels, boolean = [], [], []
            for line in f:
                line = line.strip()
                if len(line) == 0:
                    if len(sentence) > self.max_sent_len:
                        self.max_sent_len = len(sentence)
                    self.data.append((sentence, labels, boolean))
                    sentence, labels, boolean = [], [], []
                else:
                    parts = line.split(" ")
                    word = parts[1]
                    label = parts[2]
                    if word[0].isupper():
                        boolean.append(1)
                    else:
                        boolean.append(0)
                    #word=word.lower() 
                    if word.lower() not in self.word2idx:
                        word = '<unkcap>' if word[0].isupper() else '<unk>'
                    if label not in self.label2idx:
                        self.label2idx[label] = len(self.label2idx)
                    sentence.append(self.word2idx[word.lower()])
                    labels.append(self.label2idx[label])
                    
        if len(sentence) > 0:
            if len(sentence) > self.max_sent_len:
                self.max_sent_len = len(sentence)
            self.data.append((sentence, labels, boolean))

        self.word2idx['<pad>'] = len(self.word2idx)
        self.pad_idx = self.word2idx['<pad>']
        
        self.x, self.y, self.mask, self.lengths = [], [], [], []
        for sentence, labels, boolean in self.data:
            self.lengths.append(len(sentence))
            self.x.append(torch.tensor(sentence))
            self.y.append(torch.tensor(labels))
            self.mask.append(torch.tensor(boolean))
        
        self.x = pad_sequence(self.x, batch_first=True, padding_value=self.pad_idx)
        self.y = pad_sequence(self.y, batch_first=True, padding_value=self.pad_idx)
        self.mask = pad_sequence(self.mask, batch_first=True, padding_value=self.pad_idx)
                
    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        return self.x[index], self.lengths[index], self.y[index], self.mask[index]


class ValidateNERDataset(Dataset):
    def __init__(self, filename,word2idx,label2idx):
        self.data = []
        self.word2idx = word2idx
        self.label2idx = label2idx
        self.max_sent_len = 0

        
        with open(filename, "r") as f:
            sentence, labels, boolean = [], [], []
            for line in f:
                line = line.strip()
                if len(line) == 0:
                    if len(sentence) > self.max_sent_len:
                        self.max_sent_len = len(sentence)
                    self.data.append((sentence, labels, boolean))
                    sentence, labels, boolean = [], [], []
                else:
                    parts = line.split(" ")
                    word = parts[1]
                    label = parts[2]
                    if word[0].isupper():
                        boolean.append(1)
                    else:
                        boolean.append(0)
                    #word=word.lower() 
                    if word.lower() not in self.word2idx:
                        word = '<unkcap>' if word[0].isupper() else '<unk>'
                    sentence.append(self.word2idx[word.lower()])
                    labels.append(self.label2idx[label])
                    
        if len(sentence) > 0:
            if len(sentence) > self.max_sent_len:
                self.max_sent_len = len(sentence)
            self.data.append((sentence, labels, boolean))

        self.word2idx['<pad>'] = len(self.word2idx)
        self.pad_idx = self.word2idx['<pad>']
        
        self.x, self.y, self.mask, self.lengths = [], [], [], []
        for sentence, labels, boolean in self.data:
            self.lengths.append(len(sentence))
            self.x.append(torch.tensor(sentence))
            self.y.append(torch.tensor(labels))
            self.mask.append(torch.tensor(boolean))
        
        self.x = pad_sequence(self.x, batch_first=True, padding_value=self.pad_idx)
        self.y = pad_sequence(self.y, batch_first=True, padding_value=self.pad_idx)
        self.mask = pad_sequence(self.mask, batch_first=True, padding_value=self.pad_idx)
        
        
    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        return self.x[index], self.lengths[index], self.y[index], self.mask[index]


class BLSTM(nn.Module):
    def __init__(self, vocab_size, embedding_dim, hidden_dim, output_dim, dropout, label_dim,embedding_mat):
        super().__init__()
        #self.embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx=train_dataset.pad_idx)
        self.embedding = nn.Embedding(embedding_mat.shape[0], embedding_mat.shape[1])
        self.embedding.weight.data.copy_(embedding_mat)
        self.embedding.weight.requires_grad = False
        self.lstm = nn.LSTM(embedding_dim+1, hidden_dim, bidirectional=True)
        self.fc = nn.Linear(hidden_dim * 2, output_dim)
        self.dropout = nn.Dropout(dropout)
        self.act = nn.ELU()
        self.classifier = nn.Linear(output_dim,label_dim)

    def forward(self, x, x_lengths, boolean):
        #embedded = self.dropout(self.embedding(x))
        embedded = self.embedding(x)
      
        stacked_tensor = torch.cat((embedded, boolean.unsqueeze(-1)), dim=-1)

        packed_embedded = nn.utils.rnn.pack_padded_sequence(stacked_tensor, x_lengths, batch_first=True,enforce_sorted=False)
        packed_output, (hidden, cell) = self.lstm(packed_embedded)
        output, output_lengths = nn.utils.rnn.pad_packed_sequence(packed_output, batch_first=True)
        output = self.dropout(output)
        output=self.fc(output)
        output = self.act(output)
        output=self.classifier(output)
        output = output.permute(0, 2, 1)
        return output


batch_size = 32
print('reading training data')
train_dataset = NERDataset(train_path,word2idx)
# Hyperparameters
EMBEDDING_DIM = 100
HIDDEN_DIM = 256
OUTPUT_DIM = 128
OUTPUT_LABEL_DIM = len(train_dataset.label2idx)
DROPOUT = 0.33
#learning rate .1 is best
LEARNING_RATE = .2
#EPOCHS = 50
EPOCHS = 30
STEP_SIZE = 20
GAMMA = 1


vectors = list(embeddings_dictionary.values())
unk_vector = .5*np.mean(vectors, axis=0)
unk_cap_vector = np.mean(vectors, axis=0)
#not needed cos anyway i have mebedding matrix all 0 initially
pad=np.zeros(100)

print('creating glove embedding matrix')

embedding_matrix = torch.zeros((len(train_dataset.word2idx)+1, EMBEDDING_DIM))


for word,index in word2idx.items():
  if word=='<unk>':
    embedding_matrix[index]=torch.from_numpy(unk_vector)
  elif word=='<unkcap>':
    embedding_matrix[index]=torch.from_numpy(unk_cap_vector)
  elif word=='<pad>':
    embedding_matrix[index]=torch.from_numpy(pad)
  else:
    embedding_vector = embeddings_dictionary.get(word)
    embedding_matrix[index] = torch.from_numpy(embedding_vector)



device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = BLSTM(len(train_dataset.word2idx), EMBEDDING_DIM, HIDDEN_DIM, OUTPUT_DIM, DROPOUT, OUTPUT_LABEL_DIM,embedding_matrix).to(device)

train_loader = DataLoader(train_dataset, batch_size=batch_size, pin_memory=True)



optimizer = optim.SGD(model.parameters(), lr=LEARNING_RATE,momentum=0.9)
scheduler = StepLR(optimizer, step_size=STEP_SIZE, gamma=GAMMA)
freq=list(d.values())
sum_freq = sum(freq)
result = [(3.5 - f/sum_freq) for f in freq]
class_weights = torch.tensor(result).to(device)
criterion = nn.CrossEntropyLoss(ignore_index=train_dataset.pad_idx,weight=class_weights).to(device)
print('reading dev data')

dev_dataset = ValidateNERDataset(dev_path,train_dataset.word2idx,train_dataset.label2idx)
dev_loader = DataLoader(dev_dataset)


print('loading the TASK-2 model')
model.load_state_dict(torch.load(model_path,map_location=torch.device('cpu')))


#Prediction on dev data
print('predicting on dev data')

model.eval()
predicted_labels = []
true_labels = []

with torch.no_grad():
    for x, lengths, y,mask in dev_loader:
        x = x.to(device)

        y = y.to(device)

        target_packed_embedded = nn.utils.rnn.pack_padded_sequence(y, lengths, batch_first=True, enforce_sorted=False)
        target, target_lengths = nn.utils.rnn.pad_packed_sequence(target_packed_embedded, batch_first=True)
        #print('y',y)
        #print('target',target)
        output = model(x, lengths,mask.to(device))

        predicted = torch.argmax(output, dim=1)
        predicted_labels.extend(predicted.cpu().numpy().tolist())
        true_labels.extend(target.cpu().numpy().tolist())



devOutput = open("dev2.out", "w")
k=0
i=0
idx2label = {value: key for key, value in train_dataset.label2idx.items()}




with open(dev_path, 'r') as f:
  for line in f:
    line = line.strip().split(' ')
    if len(line)>1:
      idx,word,gold  = line[0], line[1],line[2]
      pred=predicted_labels[k][i]
      i=i+1
      key = idx2label[pred]
      devOutput.write(f"{idx} {word} {key}\n")
    else:
      devOutput.write(f"\n")
      k=k+1
      i=0    
f.close()
devOutput.close()



#test predictions

print('predicting on test data')

class TestNERDataset(Dataset):
    def __init__(self, filename,word2idx):
        self.data = []
        self.word2idx = word2idx
        self.max_sent_len = 0

        
        with open(filename, "r") as f:
            sentence, boolean = [], []
            for line in f:
                line = line.strip()
                if len(line) == 0:
                    if len(sentence) > self.max_sent_len:
                        self.max_sent_len = len(sentence)
                    self.data.append((sentence, boolean))
                    sentence, boolean = [], []
                else:
                    parts = line.split(" ")
                    word = parts[1]
                    if word[0].isupper():
                        boolean.append(1)
                    else:
                        boolean.append(0)
                    word=word.lower() 
                    if word not in self.word2idx:
                        word = '<unkcap>' if word[0].isupper() else '<unk>'
                    sentence.append(self.word2idx[word])
                    
        if len(sentence) > 0:
            if len(sentence) > self.max_sent_len:
                self.max_sent_len = len(sentence)
            self.data.append((sentence, boolean))

        self.pad_idx = self.word2idx['<pad>']
        
        self.x,  self.mask, self.lengths = [], [], []
        for sentence, boolean in self.data:
            self.lengths.append(len(sentence))
            self.x.append(torch.tensor(sentence))
            self.mask.append(torch.tensor(boolean))
        
        self.x = pad_sequence(self.x, batch_first=True, padding_value=self.pad_idx)
        self.mask = pad_sequence(self.mask, batch_first=True, padding_value=self.pad_idx)
        
        
    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        return self.x[index], self.lengths[index],  self.mask[index]


test_dataset = TestNERDataset(test_path,train_dataset.word2idx)
test_loader = DataLoader(test_dataset)



model.eval()
predicted_labels = []
true_labels = []

with torch.no_grad():
    for x, lengths,mask in test_loader:
        x = x.to(device)

        output = model(x, lengths,mask.to(device))

        predicted = torch.argmax(output, dim=1)
        predicted_labels.extend(predicted.cpu().numpy().tolist())


testOutput = open("test2.out", "w")
k=0
i=0



with open(test_path, 'r') as f:
  for line in f:
    line = line.strip().split(' ')
    if len(line)>1:
      idx,word  = line[0], line[1]
      pred=predicted_labels[k][i]
      i=i+1
      key = idx2label[pred]
      testOutput.write(f"{idx} {word} {key}\n")
    else:
      testOutput.write(f"\n")
      k=k+1
      i=0    
f.close()
testOutput.close()

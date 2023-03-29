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
model_path='./model/blstm1.pt'

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


class NERDataset(Dataset):
    def __init__(self, filename):
        self.data = []
        self.word2idx = {'<unk>':1,'<unkcap>':2}
        self.label2idx = {}
        self.max_sent_len = 0
        self.wordCounter={}

        with open(filename, "r") as f:
            sentence = []
            labels = []
            for line in f:
                line = line.strip()
                if len(line) != 0:
                    parts = line.split(" ")
                    word = parts[1]
                    self.wordCounter[word]=1+self.wordCounter.get(word,0)

        for word,count in self.wordCounter.items():
          if count>1 and word not in self.word2idx:
            self.word2idx[word]=len(self.word2idx)
      


        with open(filename, "r") as f:
            sentence = []
            labels = []
            for line in f:
                line = line.strip()
                if len(line) == 0:
                    if len(sentence) > self.max_sent_len:
                        self.max_sent_len = len(sentence)

                    self.data.append((sentence, labels))
                    sentence = []
                    labels = []
                else:
                    parts = line.split(" ")
                    word = parts[1]
                    label = parts[2]
                    if word not in self.word2idx:
                      if word[0].isupper():
                        word='<unkcap>'
                      else:
                        word='<unk>'
                        
                      
                    if label not in self.label2idx:
                        self.label2idx[label] = len(self.label2idx)

                    sentence.append(self.word2idx[word])
                    labels.append(self.label2idx[label])

        if len(sentence) > 0:
            if len(sentence) > self.max_sent_len:
                self.max_sent_len = len(sentence)

            self.data.append((sentence, labels))
        
    
        self.word2idx['<PAD>'] = len(self.word2idx)
        self.pad_idx = self.word2idx['<PAD>']
        
        
        # Pad sentences
        self.x = [torch.tensor(s) for s, _ in self.data]
        self.x = pad_sequence(self.x, batch_first=True, padding_value=self.pad_idx)

        # Pad labels
        self.y = [torch.tensor(l) for _, l in self.data]
        self.y = pad_sequence(self.y, batch_first=True,padding_value=self.pad_idx)

        # Calculate lengths
        self.lengths = [len(s) for s, _ in self.data]

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        return self.x[index], self.lengths[index], self.y[index]


class ValidateNERDataset(Dataset):
    def __init__(self, filename,word2idx,label2idx):
        self.data = []
        self.word2idx = word2idx
        self.label2idx = label2idx
        self.max_sent_len = 0

        with open(filename, "r") as f:
            sentence = []
            labels = []
            for line in f:
                line = line.strip()
                if len(line) == 0:
                    if len(sentence) > self.max_sent_len:
                        self.max_sent_len = len(sentence)

                    self.data.append((sentence, labels))
                    sentence = []
                    labels = []
                else:
                    parts = line.split(" ")
                    word = parts[1]
                    label = parts[2]
                    if word not in self.word2idx:
                      if word[0].isupper():
                        word = '<unkcap>'
                      else:
                        word='<unk>'

                    sentence.append(self.word2idx[word])
                    labels.append(self.label2idx[label])

        if len(sentence) > 0:
            if len(sentence) > self.max_sent_len:
                self.max_sent_len = len(sentence)

            self.data.append((sentence, labels))


        self.pad_idx = self.word2idx['<PAD>']
        
        # Pad sentences
        self.x = [torch.tensor(s) for s, _ in self.data]
        self.x = pad_sequence(self.x, batch_first=True,padding_value=self.pad_idx)

        # Pad labels
        self.y = [torch.tensor(l) for _, l in self.data]
        self.y = pad_sequence(self.y, batch_first=True,padding_value=self.pad_idx)

        # Calculate lengths
        self.lengths = [len(s) for s, _ in self.data]

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        return self.x[index], self.lengths[index], self.y[index]


class BLSTM(nn.Module):
    def __init__(self, vocab_size, embedding_dim, hidden_dim, output_dim, dropout, label_dim):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx=train_dataset.pad_idx)
        self.lstm = nn.LSTM(embedding_dim, hidden_dim, batch_first=True ,bidirectional=True)
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(hidden_dim * 2, output_dim)
        self.act = nn.ELU(alpha=0.75)
        self.classifier = nn.Linear(output_dim,label_dim)

    def forward(self, x, x_lengths):
        #embedded = self.dropout(self.embedding(x))
        embedded = self.embedding(x)
        packed_embedded = nn.utils.rnn.pack_padded_sequence(embedded, x_lengths, batch_first=True,enforce_sorted=False)
        packed_output, (hidden, cell) = self.lstm(packed_embedded)
        output, output_lengths = nn.utils.rnn.pad_packed_sequence(packed_output, batch_first=True)
        output = self.dropout(output)
        output=self.fc(output)
        output = self.act(output)
        output=self.classifier(output)
        output = output.permute(0, 2, 1)
        return output


batch_size = 8
print('reading traininid data')
train_dataset = NERDataset(train_path)
print('done')
# Hyperparameters
EMBEDDING_DIM = 100
HIDDEN_DIM = 256
OUTPUT_DIM = 128
OUTPUT_LABEL_DIM = len(train_dataset.label2idx)
DROPOUT = 0.33
#learning rate .1 is best
LEARNING_RATE = .005
EPOCHS = 50
STEP_SIZE = 10
GAMMA = 1



device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = BLSTM(len(train_dataset.word2idx), EMBEDDING_DIM, HIDDEN_DIM, OUTPUT_DIM, DROPOUT, OUTPUT_LABEL_DIM).to(device)
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, pin_memory=True)

print('reading dev data')

dev_dataset = ValidateNERDataset(dev_path, train_dataset.word2idx,train_dataset.label2idx)
print('done')

dev_loader = DataLoader(dev_dataset)


print('loading the TASK-1 model')
model.load_state_dict(torch.load(model_path,map_location=torch.device('cpu')))

#dev predictions

model.eval()
predicted_labels = []
true_labels = []
print('predicting on dev data')

with torch.no_grad():
    for x, lengths, y in dev_loader:
        x = x.to(device)

        y = y.to(device)

        target_packed_embedded = nn.utils.rnn.pack_padded_sequence(y, lengths, batch_first=True, enforce_sorted=False)
        target, target_lengths = nn.utils.rnn.pad_packed_sequence(target_packed_embedded, batch_first=True)

        output = model(x, lengths)

        predicted = torch.argmax(output, dim=1)
        predicted_labels.extend(predicted.cpu().numpy().tolist())
        true_labels.extend(target.cpu().numpy().tolist())



devOutput = open("dev1.out", "w")
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





#TEST PREDICTIONS
print('predicting on test data')

class TestNERDataset(Dataset):
    def __init__(self, filename,word2idx):
        self.data = []
        self.word2idx = word2idx
        self.max_sent_len = 0

        with open(filename, "r") as f:
            sentence = []
            for line in f:
                line = line.strip()
                if len(line) == 0:
                    if len(sentence) > self.max_sent_len:
                        self.max_sent_len = len(sentence)

                    self.data.append(sentence)
                    sentence = []
                else:
                    parts = line.split(" ")
                    word = parts[1]
                    if word not in self.word2idx:
                      if word[0].isupper():
                        word = '<unkcap>'
                      else:
                        word='<unk>'

                    sentence.append(self.word2idx[word])

        if len(sentence) > 0:
            if len(sentence) > self.max_sent_len:
                self.max_sent_len = len(sentence)

            self.data.append(sentence)


        self.pad_idx = self.word2idx['<PAD>']
        
        # Pad sentences
        self.x = [torch.tensor(s) for s in self.data]
        self.x = pad_sequence(self.x, batch_first=True,padding_value=self.pad_idx)

        # Calculate lengths
        self.lengths = [len(s) for s in self.data]

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        return self.x[index], self.lengths[index]

test_dataset = TestNERDataset(test_path,train_dataset.word2idx)
test_loader = DataLoader(test_dataset)



model.eval()
predicted_labels = []

with torch.no_grad():
    for x, lengths in test_loader:
        x = x.to(device)
      
        output = model(x, lengths)

        predicted = torch.argmax(output, dim=1)
        predicted_labels.extend(predicted.cpu().numpy().tolist())

testOutput = open("test1.out", "w")
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

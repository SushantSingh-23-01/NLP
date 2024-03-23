import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
import re
from PyPDF2 import PdfReader
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from sklearn.model_selection import train_test_split
from tqdm import trange
import matplotlib.pyplot as plt

path = '' # txt/pdf path 

def read_txt(path):
    raw_text = open(path,'r',encoding='utf-8-sig').read()
    raw_text = raw_text.lower()
    return raw_text

def read_pdf(path):
    text = ""
    pdf_reader = PdfReader(path)
    for page in pdf_reader.pages:
        text += page.extract_text()
    return text

def preprocessing(document):
    text = re.sub(r'[^A-Za-z]+',' ',document.lower())
    text = re.sub('\s+',' ',text) 
    text = ' '.join([word for word in word_tokenize(text) if not word in set(stopwords.words('english'))])
    return text
    
def create_word_index_mapping(document):
    vocab = sorted(set(word_tokenize(document)))
    vocab.insert(0,'<pad>')
    vocab.insert(1,'<unk>')
    word_index_dict = {word: i for i,word in enumerate(vocab)}
    doc_to_int = []
    for word in word_tokenize(document):
        doc_to_int.append(word_index_dict[word])
    return doc_to_int, len(vocab)      
 
def context_words(doc_to_int,context_size=2):
    context, target = [], []
    for i in range(context_size,len(doc_to_int)-context_size):
        context.append(doc_to_int[i-context_size : i] + doc_to_int[i+1:i+1+context_size])
        target.append(doc_to_int[i])
    return context, target 

def load_dataset(context,target):
    class SentDataset(Dataset):
        def __init__(self,context,target):
            self.context = context
            self.target = target
        
        def __len__(self):
            return len(self.context)
        
        def __getitem__(self,index):
            features = self.context[index]
            targets = self.target[index]
            return features, targets
        
    dataset = SentDataset(context,target)
    return dataset

def get_model(VOCAB_SIZE, EMBDED_DIM):
    class CbowModel(nn.Module):
        def __init__(self, vocab_size, embed_size):
            super(CbowModel, self).__init__()
            self.embeddings = nn.Embedding(vocab_size, embed_size)
            self.linear = nn.Linear(embed_size,vocab_size)
            
        def forward(self, inputs):
            embedding = self.embeddings(inputs)  # embeddings: batch_size x (4) x embedding size
            embedding = embedding.mean(-1)        # embeddings: batch_size x (1) x embedding size 
            embedding = embedding.squeeze(-1)     # embeddings: batch_size x embedding_size
            output = self.linear(embedding) 
            return output
    model = CbowModel(vocab_size=VOCAB_SIZE, embed_size= EMBDED_DIM)
    return model

def training_loop(model,loader):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
    num_epoch = 10
    losses, epochs = [], []
    for epoch in trange(num_epoch,desc="Progress", unit="batch/s"):
        #with tqdm(loader, unit="batch") as tepoch:
        for X_batch, y_batch in loader:
            X_batch = X_batch.to(device)
            y_batch = y_batch.to(device)
            
            optimizer.zero_grad()
            y_pred = model(X_batch)
            loss = criterion(y_pred, y_batch)
            loss.backward()
            optimizer.step()
        losses.append(float(loss))  
        epochs.append(epoch)
        #if (epoch +1) % 2 == 0:
    print(f'\nepoch : {epoch +1} | loss : {loss.item():.4f}')      
    return losses, epochs 

def plot_loss(losses,epochs):
    plt.style.use('ggplot')
    plt.title("Training Curve")
    plt.plot(epochs, losses, label="Train")
    plt.xlabel("Epoch")
    plt.ylabel("Binary Cross Entropy Loss")
    plt.show()

            
def main():
    text = read_txt(path)
    text = preprocessing(text)
    doc_to_int, vocab_size = create_word_index_mapping(text)
    context, target = context_words(doc_to_int=doc_to_int,context_size=2)
    X = torch.LongTensor(context)
    y = torch.FloatTensor(target)
    X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=0.7, shuffle=True)
    
    dataset = load_dataset(X_train,y_train)
    loader = DataLoader(dataset,shuffle=True, batch_size=2)
    
    model = get_model(VOCAB_SIZE=vocab_size,EMBDED_DIM=5)
    losses,epochs = training_loop(model=model,loader=loader)
    plot_loss(losses=losses,epochs=epochs)
    
if __name__ == '__main__':
    main()

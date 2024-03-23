import torch
import torch.nn as nn
import numpy as np
import torchtext
from torchtext.data import Field, get_tokenizer, BucketIterator,TabularDataset
from torchtext.datasets import LanguageModelingDataset

path = r'' ## Text Path

device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
tokenizer = get_tokenizer('basic_english')  
TEXT = torchtext.data.Field(lower=True, tokenize=tokenizer)
lm_data = LanguageModelingDataset(path = path,text_field= TEXT)

TEXT.build_vocab(lm_data)

class LM(nn.Module):
  
  def __init__(self, hid_size, vocab_size, n_head, n_layers, pf_size, max_len, device):
    super().__init__()

    self.device = device
    
    self.hid_size = hid_size
    self.pf_size = pf_size
    self.max_len = max_len

    self.embedding = nn.Embedding(vocab_size, hid_size)

    self.position_enc = nn.Embedding(self.max_len, self.hid_size)
    self.position_enc.weight.data = self.position_encoding_init(self.max_len, self.hid_size)
    self.scale = torch.sqrt(torch.FloatTensor([self.hid_size])).to(device)

    self.layer_norm = nn.LayerNorm(self.hid_size)
    self.decoder_layer = nn.TransformerDecoderLayer(d_model=hid_size, nhead = n_head, dim_feedforward=self.pf_size)
    self.decoder = nn.TransformerDecoder(self.decoder_layer, num_layers=n_layers, norm=self.layer_norm)
    self.fc = nn.Linear(hid_size, vocab_size)

    self._init_weights()
  
  def forward(self, x):
    sent_len, batch_size = x.shape[0], x.shape[1]
    memory_mask = self.generate_complete_mask(sent_len)
    tgt_mask = self.generate_triangular_mask(sent_len)
    memory = torch.zeros(1, batch_size, self.hid_size, device=self.device)

    temp = x
    temp = self.embedding(temp)

    pos = torch.arange(0,sent_len).unsqueeze(1).repeat(1,batch_size).to(self.device)
    temp_pos_emb = self.position_enc(pos)

    temp = temp * self.scale + temp_pos_emb
    temp = self.decoder(temp, memory, tgt_mask=tgt_mask)
    temp = self.fc(temp)
    return temp

  def _init_weights(self):
    for p in self.parameters():
      if p.dim() > 1:
        nn.init.xavier_uniform_(p)

  def append_decoder_layer(self):
    appended_mod = nn.TransformerDecoderLayer(d_model=hid_size, nhead = n_head).to(self.device)
    for p in appended_mod.parameters():
      if p.dim() > 1:
        nn.init.xavier_uniform_(p)
    model.decoder.layers.append(appended_mod)
    model.decoder.num_layers += 1

  def generate_triangular_mask(self, size):
        r"""Generate a square mask for the sequence. The masked positions are filled with float('-inf').
            Unmasked positions are filled with float(0.0).
        """
        mask = (torch.triu(torch.ones(size, size)) == 1).transpose(0, 1)
        mask = mask.float().masked_fill(mask == 0, float('-inf')).masked_fill(mask == 1, float(0.0)).to(device)
        return
        
  def generate_complete_mask(self, size):
        r"""Generate a square mask for the sequence. The masked positions are filled with float('-inf').
            Unmasked positions are filled with float(0.0).
        """
        mask = torch.empty(size, size).to(device)
        mask.fill_(float('-inf'))
        return mask

  def generate_sequence(self, src):
    #src = [sent_len]
    src = src.unsqueeze(1)
    #src = [sent_len, 1]
    generate_step = 0
    while generate_step < 20:
      out = self.forward(src)
      #out = [sent_len + 1, 1, vocab_size]
      out = torch.argmax(out[-1, :], dim=1) # [1]
      out = out.unsqueeze(0) #[1,1]
      src = torch.cat((src, out), dim=0)
      generate_step += 1
    src = src.squeeze(1)
    return src
  
  def position_encoding_init(self, n_position, d_pos_vec):
    ''' Init the sinusoid position encoding table '''

    # keep dim 0 for padding token position encoding zero vector
    position_enc = np.array([
        [pos / np.power(10000, 2*i/d_pos_vec) for i in range(d_pos_vec)]
        if pos != 0 else np.zeros(d_pos_vec) for pos in range(n_position)])

    position_enc[1:, 0::2] = np.sin(position_enc[1:, 0::2]) # dim 2i
    position_enc[1:, 1::2] = np.cos(position_enc[1:, 1::2]) # dim 2i+1
    temp = torch.from_numpy(position_enc).type(torch.FloatTensor)
    temp = temp.to(self.device)
    return temp


def load_lm_dat(path,TEXT):
    lm_data = LanguageModelingDataset(path = path,text_field= TEXT)
    examples = lm_data.examples
    return lm_data

def make_train_iter(batch_size, bptt_len,lm_data):
  train_iter = torchtext.data.BPTTIterator(
    dataset=lm_data,
    batch_size=batch_size,
    bptt_len=bptt_len, # this is where we specify the sequence length
    device=device,
    repeat=False,
    shuffle=True)
  return train_iter


def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

class NoamOpt:
    "Optim wrapper that implements rate."
    def __init__(self, model_size, factor, warmup, optimizer):
        self.optimizer = optimizer
        self._step = 0
        self.warmup = warmup
        self.factor = factor
        self.model_size = model_size
        self._rate = 0
        
    def step(self):
        "Update parameters and rate"
        self._step += 1
        rate = self.rate()
        for p in self.optimizer.param_groups:
            p['lr'] = rate
        self._rate = rate
        self.optimizer.step()
        
    def rate(self, step = None):
        "Implement `lrate` above"
        if step is None:
            step = self._step
        return self.factor * \
            (self.model_size ** (-0.5) *
            min(step ** (-0.5), step * self.warmup ** (-1.5)))
    
    def zero_grad(self):
        self.optimizer.zero_grad()
     
def train_one_epoch(model,train_iter, optimizer, criterion, clip):
  epoch_loss = 0
  model.train()
  for batch in train_iter:
    optimizer.zero_grad()
    batch_text = batch.text
    batch_target = batch.target
    result = model(batch_text)
    loss = criterion(result.view(-1, result.shape[-1]), batch_target.view(-1))
    loss.backward()
    torch.nn.utils.clip_grad_norm_(model.parameters(), clip)
    optimizer.step()
    epoch_loss += loss.item()
    return epoch_loss / len(train_iter)



def train(model, train_iter, optimizer, criterion, clip, N_EPOCH):
  for epoch in range(N_EPOCH):
    epoch_loss = train_one_epoch(model, train_iter, optimizer, criterion, clip)
    if N_EPOCH % 1 == 0:
      print("epoch is {} loss is {}".format(epoch, epoch_loss))
    
    
vocab_size = len(TEXT.vocab)
hid_size = 16
pf_size = 64
max_len = 512
n_head = 4
n_layer= 1
model = LM(hid_size, vocab_size, n_head, n_layer, pf_size, max_len, device).to(device)
     
for i in range(1, 6):
  optimizer = NoamOpt(hid_size, 1, 2000,
              torch.optim.Adam(model.parameters(), lr=0, betas=(0.9, 0.98), eps=1e-9))
  optimizer = torch.optim.Adam(model.parameters(), lr=0.01, betas=(0.9, 0.98), eps=1e-9)
  criterion = torch.nn.CrossEntropyLoss()
  train_iter = make_train_iter(4096, i,lm_data)
  train(model, train_iter, optimizer, criterion, 1, 5) 
  

source_sentence = ["the","forest"]
print(source_sentence)
model.eval()
print(' '.join(source_sentence))
print()
x = TEXT.numericalize([source_sentence]).to(device).squeeze(1)
generated_sequence =model.generate_sequence(x)
words = [TEXT.vocab.itos[word_idx] for word_idx in generated_sequence]
print(' '.join(words))

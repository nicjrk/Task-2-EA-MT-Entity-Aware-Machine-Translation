import torch
import torch.nn as nn
import torch.nn.functional as F

class Attention(nn.Module):
    def __init__(self, hidden_dim):
        super().__init__()
        # Transformă starea ascunsă concatenată într-un scor de atenție
        self.attention = nn.Linear(hidden_dim * 2, hidden_dim)
        self.v = nn.Linear(hidden_dim, 1, bias=False)
        
    def forward(self, hidden, encoder_outputs):
        # hidden = [batch_size, hidden_dim]
        # encoder_outputs = [batch_size, src_len, hidden_dim]
        
        batch_size = encoder_outputs.shape[0]
        src_len = encoder_outputs.shape[1]
        
        # Repetă hidden pentru a se potrivi cu dimensiunea encoder_outputs
        hidden = hidden.unsqueeze(1).repeat(1, src_len, 1)
        
        # Concatenează și calculează scorurile de atenție
        energy = torch.tanh(self.attention(torch.cat((hidden, encoder_outputs), dim=2)))
        attention = self.v(energy).squeeze(2)
        
        return F.softmax(attention, dim=1)

class Seq2SeqAttention(nn.Module):
    def __init__(self, input_dim, output_dim, embedding_dim, hidden_dim):
        super().__init__()
        self.encoder = nn.Embedding(input_dim, embedding_dim)
        self.decoder = nn.Embedding(output_dim, embedding_dim)
        self.rnn = nn.LSTM(embedding_dim, hidden_dim, batch_first=True)
        self.attention = Attention(hidden_dim)
        self.fc = nn.Linear(hidden_dim * 2, output_dim)  # *2 pentru concatenarea cu contextul
        
    def forward(self, source, target):
        # Encoding
        embedded_source = self.encoder(source)
        encoder_outputs, (hidden, cell) = self.rnn(embedded_source)
        
        # Decoding cu atenție
        embedded_target = self.decoder(target)
        decoder_outputs = []
        
        for t in range(embedded_target.size(1)):
            # Calculează atenția pentru pasul curent
            attn_weights = self.attention(hidden[-1], encoder_outputs)
            # Calculează vectorul de context
            context = torch.bmm(attn_weights.unsqueeze(1), encoder_outputs)
            
            # Decoder step
            decoder_input = embedded_target[:, t:t+1]
            output, (hidden, cell) = self.rnn(decoder_input, (hidden, cell))
            
            # Concatenează output cu contextul și trece prin stratul final
            output = torch.cat((output, context), dim=2)
            prediction = self.fc(output)
            decoder_outputs.append(prediction)
        
        # Concatenează toate predicțiile
        decoder_outputs = torch.cat(decoder_outputs, dim=1)
        return decoder_outputs
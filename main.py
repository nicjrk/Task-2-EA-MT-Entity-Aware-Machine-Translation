import json
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from collections import Counter
from seq2seq_attention import Seq2SeqAttention
from tqdm import tqdm, trange

# Hyperparameters
EMBEDDING_DIM = 256
HIDDEN_DIM = 512
BATCH_SIZE = 32
EPOCHS = 25
LEARNING_RATE = 0.0001

# Load training data from JSONL file
def load_training_data(file_path):
    source_sentences = []
    target_sentences = []
    with open(file_path, 'r', encoding='utf-8') as file:
        for line in file:
            sample = json.loads(line)
            source_sentences.append(sample['source'])
            target_sentences.append(sample['target'])
    return source_sentences, target_sentences

# Tokenize sentences and build vocabulary
def build_vocab(sentences):
    counter = Counter()
    for sentence in sentences:
        counter.update(sentence.split())
    vocab = {word: idx + 2 for idx, (word, _) in enumerate(counter.items())}  # Start indexing from 2
    vocab["<PAD>"] = 0
    vocab["<UNK>"] = 1
    return vocab

# Dataset class
class TranslationDataset(Dataset):
    def __init__(self, source_sentences, target_sentences, source_vocab, target_vocab):
        self.source_sentences = source_sentences
        self.target_sentences = target_sentences
        self.source_vocab = source_vocab
        self.target_vocab = target_vocab

    def __len__(self):
        return len(self.source_sentences)

    def __getitem__(self, idx):
        source = [self.source_vocab.get(word, self.source_vocab["<UNK>"]) for word in self.source_sentences[idx].split()]
        target = [self.target_vocab.get(word, self.target_vocab["<UNK>"]) for word in self.target_sentences[idx].split()]
        return torch.tensor(source), torch.tensor(target)

# Simple Seq2Seq Model
class Seq2Seq(nn.Module):
    def __init__(self, input_dim, output_dim, embedding_dim, hidden_dim):
        super(Seq2Seq, self).__init__()
        self.encoder = nn.Embedding(input_dim, embedding_dim)
        self.decoder = nn.Embedding(output_dim, embedding_dim)
        self.rnn = nn.LSTM(embedding_dim, hidden_dim, batch_first=True)
        self.fc = nn.Linear(hidden_dim, output_dim)

    def forward(self, source, target):
        embedded_source = self.encoder(source)
        _, (hidden, cell) = self.rnn(embedded_source)
        embedded_target = self.decoder(target)
        outputs, _ = self.rnn(embedded_target, (hidden, cell))
        predictions = self.fc(outputs)
        return predictions

if __name__ == "__main__":
    torch.multiprocessing.freeze_support()  # Necesară pentru Windows

    # Load dataset
    data_file_path = "./training data/train/de/train_tiny.jsonl"

    source_sentences, target_sentences = load_training_data(data_file_path)

    # Build vocabularies
    source_vocab = build_vocab(source_sentences)
    target_vocab = build_vocab(target_sentences)

    # Create dataset and data loader
    dataset = TranslationDataset(source_sentences, target_sentences, source_vocab, target_vocab)
    dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True, collate_fn=lambda x: x)

    # Initialize model, loss, and optimizer
    model = Seq2SeqAttention(len(source_vocab), len(target_vocab), EMBEDDING_DIM, HIDDEN_DIM)
    criterion = nn.CrossEntropyLoss(ignore_index=source_vocab["<PAD>"])
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)

    # Training loop
    model.train()
    for epoch in trange(EPOCHS, desc='Epochs'):
        epoch_loss = 0
        pbar = tqdm(dataloader, desc=f'Epoch {epoch + 1}/{EPOCHS}', leave=False)
        for batch in pbar:
            source_batch, target_batch = zip(*batch)
            source_batch = nn.utils.rnn.pad_sequence(source_batch, batch_first=True, padding_value=source_vocab["<PAD>"])
            target_batch = nn.utils.rnn.pad_sequence(target_batch, batch_first=True, padding_value=target_vocab["<PAD>"])
        
            optimizer.zero_grad()
            output = model(source_batch, target_batch[:, :-1])
            loss = criterion(output.reshape(-1, output.shape[-1]), target_batch[:, 1:].reshape(-1))
            loss.backward()
            optimizer.step()
        
            current_loss = loss.item()
            epoch_loss += current_loss
            pbar.set_postfix({'loss': f'{current_loss:.4f}'})
    
        avg_epoch_loss = epoch_loss / len(dataloader)
        print(f"\nEpoch {epoch + 1}/{EPOCHS}, Average Loss: {avg_epoch_loss:.4f}")

    # Save the trained model
    torch.save(model.state_dict(), "./seq2seq_translation_model.pth")
    print("Model training complete. Model saved.")

# Function for real-time translation
def translate_text(input_text):
    model.eval()  # Set model to evaluation mode
    with torch.no_grad():
        source_indices = [source_vocab.get(word, source_vocab["<UNK>"]) for word in input_text.split()]
        # Verificăm dacă secvența nu este goală
        if len(source_indices) == 0:
            return "Error: No valid words to translate."
        source_tensor = torch.tensor(source_indices, dtype=torch.long).unsqueeze(0)  # Add batch dimension
        output = model(source_tensor, source_tensor)
        predicted_indices = torch.argmax(output, dim=-1).squeeze().tolist()
        predicted_words = [list(target_vocab.keys())[list(target_vocab.values()).index(idx)] for idx in predicted_indices if idx in target_vocab.values()]
    return ' '.join(predicted_words)

# Example usage
while True:
    user_input = input("Enter text to translate (or 'exit' to quit): ")
    if user_input.lower() == 'exit':
        break
    print("Translated text:", translate_text(user_input))

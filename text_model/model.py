import torch.nn as nn
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
import gensim
import torch
import numpy as np

def create_embed_matrix(vocab, embed_model):
    vocab = vocab.stoi
    matrix_len = len(vocab)
    weights_matrix = np.zeros((matrix_len, 300))

    for word, index in vocab.items():
        if word not in ['<unk>', '<pad>']:
            weights_matrix[index] = embed_model[word] 
    return weights_matrix


def create_emb_layer(weights_matrix, non_trainable=False):
    num_embeddings, embedding_dim = weights_matrix.shape
    emb_layer = nn.Embedding(num_embeddings, embedding_dim)
    emb_layer.load_state_dict({'weight': torch.from_numpy(weights_matrix)})
    if non_trainable:
        emb_layer.weight.requires_grad = False

    return emb_layer, num_embeddings, embedding_dim
            
            
def create_embed_model(model_path):
    return gensim.models.fasttext.FastTextKeyedVectors.load(model_path)


class LSTM(nn.Module):

    def __init__(self, weights_matrix, dimension=128, num_layers=4):
        super(LSTM, self).__init__()
        
        self.embedding, num_embeddings, embedding_dim = create_emb_layer(weights_matrix, True)
        self.dimension = dimension
        self.lstm = nn.LSTM(input_size=300,
                            hidden_size=dimension,
                            num_layers=num_layers,
                            batch_first=True,
                            bidirectional=True)
        self.drop = nn.Dropout(p=0.5)

        self.fc = nn.Linear(2*dimension, 1)

    def forward(self, text, text_len):
         
        text_emb = self.embedding(text)
        
        packed_input = pack_padded_sequence(text_emb, text_len, batch_first=True, enforce_sorted=False)
        packed_output, _ = self.lstm(packed_input)
        output, _ = pad_packed_sequence(packed_output, batch_first=True)

        out_forward = output[range(len(output)), text_len - 1, :self.dimension]
        out_reverse = output[:, 0, self.dimension:]
        out_reduced = torch.cat((out_forward, out_reverse), 1)
        text_fea = self.drop(out_reduced)

        text_fea = self.fc(text_fea)
        text_fea = torch.squeeze(text_fea, 1)
        text_out = torch.sigmoid(text_fea)

        return text_out
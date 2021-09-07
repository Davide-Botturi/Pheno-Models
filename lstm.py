import torch
import torch.nn as nn

class LSTM_CLASS(nn.Module):
    def __init__(self,input_dim,output_dim,n_layers):
        super(LSTM_CLASS, self).__init__()

        self.input_dim = input_dim
        self.output_dim = output_dim
        self.n_layers = n_layers

        self.lstm = nn.LSTM(input_dim,output_dim,n_layers)

        self.first_hidden = torch.randn(self.n_layers, 1, self.output_dim)

        self.first_cn = torch.randn(self.n_layers, 1, self.output_dim)

    def forward(self,sequence):
        batch_size = sequence.shape[1]

        hidden = (self.first_hidden.repeat(1,batch_size,1),self.first_cn.repeat(1,batch_size,1))

        out,hidden = self.lstm(sequence,hidden)
        return out,hidden


if __name__ == '__main__':
    input_dim = 4
    output_dim = 2
    batch_size = 10
    n_layers = 1
    sequence_length = 100

    inputs = torch.randn(sequence_length,batch_size,input_dim)
    model = LSTM_CLASS(input_dim,output_dim,n_layers)
    out,hidden = model.forward(inputs)

    print(out.shape)
    print("Sequence,Batch,N_outputs")



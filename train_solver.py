import torch

from compute_loss import std_loss,start_end_loss
from solver import Solver
from lstm import LSTM_CLASS
'''
def forward_and_loss(solver,data,indexes):
    out, hidden = solver.model.forward(data)
    loss = 0
    for out_index in range(out.shape[-1]):
        loss += std_loss(out, indexes, out_index) + start_end_loss(out, out_index)
    return loss
'''
def train_solver(solver,data,indexes,n_epochs,eval_data = None,eval_indexes = None):
    solver.model.train()

    for epoch in range(n_epochs):
        out, hidden = solver.model.forward(data)
        solver.optimizer.zero_grad()
        loss = 0
        for out_index in range(out.shape[-1]):
            loss += std_loss(out, indexes, out_index) + start_end_loss(out,out_index)


        loss.backward()
        solver.optimizer.step()

        solver.losses.append(loss.item())







    std_loss(out, indexes, out_index,1)
    print("OUTPUT",out[:,:,0])
    return solver.losses

if __name__ == '__main__':
    input_dim = 4
    output_dim = 1
    batch_size = 3
    n_layers = 1
    sequence_length = 5

    inputs = torch.randn(sequence_length, batch_size, input_dim)
    model = LSTM_CLASS(input_dim, output_dim, n_layers)

    lr = 0.1
    solver = Solver(lr, model)


    BBCH_size = 2
    indexes = torch.randint(0, sequence_length - 1, (batch_size, BBCH_size))

    loss = train_solver(solver,inputs,indexes,100)


    print(indexes,inputs)

    import matplotlib.pyplot as plt
    plt.plot(loss)
    plt.show()
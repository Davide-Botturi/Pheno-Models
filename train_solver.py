import torch

from compute_loss import std_loss,start_end_loss
from solver import Solver
from lstm import LSTM_CLASS

def forward_and_loss(solver,data,indexes):
    out, hidden = solver.model.forward(data)
    loss = 0
    for out_index in range(out.shape[-1]):
        loss += std_loss(out, indexes, out_index) + 10* start_end_loss(out, out_index)
    return loss

def train_solver(solver,data,indexes,n_epochs,eval_data = None,eval_indexes = None):
    solver.model.train()

    for epoch in range(n_epochs):
        solver.optimizer.zero_grad()
        loss = forward_and_loss(solver,data,indexes)
        loss.backward()
        solver.optimizer.step()
        solver.losses.append(loss.item())

        if eval_data is not None:
            solver.model.eval()
            eval_loss = forward_and_loss(solver,eval_data,eval_indexes)
            solver.eval_losses.append(eval_loss.item())
            solver.model.train()

    #std_loss(out, indexes, out_index,1)
    print("OUTPUT",solver.model.forward(data)[0][:,:,0])
    print("EVAL_OUTPUT",solver.model.forward(eval_data)[0][:,:,0])
    print(solver.model.time_step_increment)

if __name__ == '__main__':
    input_dim = 3
    output_dim = 1
    batch_size = 4
    n_layers = 5
    sequence_length = 100


    inputs = torch.randn(sequence_length, batch_size, input_dim)
    eval_inputs = torch.randn(sequence_length, batch_size, input_dim)


    model = LSTM_CLASS(input_dim, output_dim, n_layers)

    lr = 0.01
    solver = Solver(lr, model)


    BBCH_size = 20

    ratio = sequence_length//(BBCH_size+1)

    indexes = torch.arange(1,BBCH_size+1) * ratio - 0*torch.randint(0, ratio, (batch_size, BBCH_size))
    eval_indexes =  torch.arange(1,BBCH_size+1) * ratio - 2*torch.randint(0, ratio, (batch_size, BBCH_size))


    n_epochs = 500

    train_solver(solver,inputs,indexes,n_epochs,eval_inputs,eval_indexes)


    print(indexes,inputs)

    import matplotlib.pyplot as plt
    plt.plot(solver.losses)
    plt.plot(solver.eval_losses)
    plt.show()
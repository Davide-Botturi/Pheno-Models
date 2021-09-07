import torch
from solver import Solver
from lstm import LSTM_CLASS
import torch.nn.functional as F
# indexes [N_batch,N_BBCH]
# out [N_sequence,N_batch,N_outputs]


# indexes[j,i] is the the day at which the j-th sample as reached the i-th phenological state

# out[indexes[i,j],Sample,out_index] is the output at the day "indexes[i,j]" of the "j-th" sample.
# Basically it's the output of the model when the j-th sample as reached the i-th phenological state.

# All the sample should have the same output for the same phenological phase.
# The error for the i-th phenological phase will be the standard deviation of all of the sample outputs of the lstm
# taken each at the day at which the i-th phenological phase has been reached.

# Out index is useful if the output has multiple dimensions. (I.e two overlapping BBCH series).


def std_loss(out, indexes, out_index,print_ = 0):

    BBCH_loss = torch.zeros(indexes.shape[0],indexes.shape[1])

    BBCH_mean = torch.zeros(indexes.shape[1]-1)
    BBCH_next_mean = torch.zeros(indexes.shape[1]-1)

    for BBCH_stage in range(indexes.shape[1]):

        for Sample in range(indexes.shape[0]):
            BBCH_loss[Sample,BBCH_stage] = out[indexes[Sample,BBCH_stage],Sample,out_index]

            if BBCH_stage != indexes.shape[1] - 1:
                BBCH_mean[BBCH_stage] += out[indexes[Sample,BBCH_stage],Sample,out_index]
            if BBCH_stage != 0:
                BBCH_next_mean[BBCH_stage-1] += out[indexes[Sample,BBCH_stage],Sample,out_index]


    BBCH_distance = (BBCH_next_mean - BBCH_mean)
    print(BBCH_distance)
    print(torch.min(BBCH_distance))

    if print_ == 1:
        print(torch.std(BBCH_loss,dim = 0).shape)
        print(torch.std(BBCH_loss,dim = 0))
        print("MATRIX OBJECTIVES",BBCH_loss)


    return torch.mean(torch.std(BBCH_loss,dim = 0))


# Loss that bounds first and last phenophase to 0 and 1 to avoid that the model maps everything to the same number.

def start_end_loss(out,out_index):
    return torch.mean(torch.abs(out[0,:,out_index])) + torch.mean(torch.abs(out[-1,:,out_index]-1))

def labeled_loss(out,label):
    return torch.nn.MSELoss()(out,label)


if __name__ == '__main__':
    input_dim = 4
    output_dim = 2
    batch_size = 2
    n_layers = 1
    sequence_length = 100

    inputs = torch.randn(sequence_length,batch_size,input_dim)
    model = LSTM_CLASS(input_dim,output_dim,n_layers)
    out,hidden = model.forward(inputs)


    lr = 0.01
    solver = Solver(lr,model)



    BBCH_size = 10
    indexes = torch.randint(0, sequence_length - 1, (batch_size, BBCH_size))
    stages_std = std_loss(out, indexes, 0)
    print(stages_std)





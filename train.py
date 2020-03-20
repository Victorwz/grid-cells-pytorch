# Copyright 2018 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================

"""Supervised training for the Grid cell network.

-------------

Adapted for pytorch, Lucas Pompe, 2019
"""
import numpy as np
import utils
import model_lstm as model
import ensembles
from dataloading import Dataset
from model_utils import get_latest_model_file, get_model_epoch

import torch
from torch.utils import data
from torch import nn

from rate_coding import PadCoder
import scores

"""
Args:
N_PC: 256 target place cells
N_HDC: 12 target head direction cells
nh_lstm: 128 units
nh_bottleneck: 256 hidden units in linear bottleneck

"""

N_EPOCHS = 1000
STEPS_PER_EPOCH = 100
ENV_SIZE = 2.2
BATCH_SIZE = 100
GRAD_CLIPPING = 1e-5
SEED = 9101
N_PC = [256]
N_HDC = [12]
BOTTLENECK_DROPOUT = 0.5
WEIGHT_DECAY = 1e-5
LR = 1e-5
MOMENTUM = 0.9
TIME = 50
PAUSE_TIME = None
SAVE_LOC = 'experiments/'

# Added by Weizhi, parameters for scorer
NH_LSTM = 128
NH_BOTTLENECK = 256
saver_results_directory = 'results/'


torch.manual_seed(SEED)
np.random.seed(SEED)


# CUDA for PyTorch
# https://stanford.edu/~shervine/blog/pytorch-how-to-generate-data-parallel
use_cuda = torch.cuda.is_available()
device = torch.device("cuda:0" if use_cuda else "cpu")
print("USING DEVICE:", device)
# Parameters
data_params = {'batch_size': BATCH_SIZE,
          'shuffle': True,
          'num_workers': 6, # num cpus,
          }

test_params = {'batch_size': BATCH_SIZE,
          'shuffle': True,
          'num_workers': 2, # num cpus,
          }

dataset = Dataset(batch_size=data_params['batch_size'])
data_generator = data.DataLoader(dataset, **data_params)
test_generator = data.DataLoader(dataset, **test_params)

# Create the ensembles that provide targets during training

place_cell_ensembles = utils.get_place_cell_ensembles(
        env_size=ENV_SIZE,
        neurons_seed=SEED,
        targets_type='softmax',
        lstm_init_type='softmax',
        n_pc=N_PC,
        pc_scale=[0.01])

head_direction_ensembles = utils.get_head_direction_ensembles(
        neurons_seed=SEED,
        targets_type='softmax',
        lstm_init_type='softmax',
        n_hdc=N_HDC,
        hdc_concentration=[20.])

target_ensembles = place_cell_ensembles + head_direction_ensembles





model = model.GridTorch(target_ensembles, (BATCH_SIZE, 100, 3)).cuda()
params = model.parameters()



# Code here is for load checkpoints

saved_model_file = get_latest_model_file(SAVE_LOC)
start_epoch = 0

if saved_model_file:
    state_dict = torch.load(saved_model_file)
    if use_cuda:
        for k, v in state_dict.items():
            state_dict[k] = v.cuda()
    model.load_state_dict(state_dict)
    start_epoch = get_model_epoch(saved_model_file)
    print("RESTORING MODEL AT:", saved_model_file)
    print("STARTING AT EPOCH:", start_epoch)



#loss ops:
logsoftmax = nn.LogSoftmax(dim=-1)
def cross_entropy(pred, soft_targets):
    return torch.sum(- soft_targets * logsoftmax(pred), -1)

#Optimisation opts
optimiser = torch.optim.RMSprop(params,
                                    lr=LR,
                                    momentum=MOMENTUM,
                                    alpha=0.9,
                                    eps=1e-10)


to_cuda = lambda x:x.cuda()

# This function is corresponding to the encode_targers in deepmind implementation
# The purpose may be to encode the target inputs including the coordinates of target and ensembles of place and head cell

def encode_inputs(X, y, place_cell_ensembles, head_direction_ensembles, cuda=True, coder=None):

    # X should be a tuple for initial inputs
    # y is the target position and head direction
    # X and y is not what we think as coordinates

    """
    
    initial position: B * 2
    initial head direction: B * 1
    egocentric velocity: B * T * 3
    target position: B * T * 2
    target head direction: B * T * 1
    
    """
    init_pos , init_hd, ego_vel = X
    target_pos, target_hd = y


    initial_conds = utils.encode_initial_conditions(init_pos ,
                                                    init_hd,
                                                    place_cell_ensembles,
                                                    head_direction_ensembles)

    ensembles_targets = utils.encode_targets(target_pos,
                                            target_hd,
                                            place_cell_ensembles,
                                            head_direction_ensembles)
    inputs = ego_vel

    if cuda:
        init_pos = init_pos.cuda()
        init_hd = init_hd.cuda()
        inputs = inputs.cuda()
        target_pos = target_pos.cuda()
        target_hd = target_hd.cuda()
        initial_conds = tuple(map(to_cuda, initial_conds))

    """
    
    target position: B * T * 2
    After coder: target_pos = B * 4T * 2
    Increase 3 to the second dimension in two ways
    i.e. a = [[ 1.2259,  0.8159,  1.9066]]
    If target, 
    a = [[ 1.2259,  0.8159,  1.9066],
         [ 1.2259,  0.8159,  1.9066],
         [ 1.2259,  0.8159,  1.9066],
         [ 1.2259,  0.8159,  1.9066]]
    If with a value template, 
    a = [[ 1.2259,  0.8159,  1.9066],
         [ 0.0000,  1.0000,  0.0000],
         [ 0.0000,  1.0000,  0.0000],
         [ 0.0000,  1.0000,  0.0000]] 
    
    """
    if coder:
        inputs = coder(inputs, value=torch.Tensor([0., 1., 0.]))
        target_pos = coder(target_pos, target=True)
        target_hd = coder(target_hd, target=True)

    inputs = inputs.transpose(1,0)
    return init_pos, init_hd, inputs, target_pos, target_hd, initial_conds, ensembles_targets

# The outputs is a tuple, then the decoding part just change the dimension of each element in tuple
def decode_outputs(outs, ensembles_targets, cuda=True, coder=None):
        if cuda:
            pc_targets = ensembles_targets[0].cuda()
            hd_targets = ensembles_targets[1].cuda()
        else:
            pc_targets = ensembles_targets[0]
            hd_targets = ensembles_targets[1]

        logits_hd, logits_pc, bottleneck_acts, lstm_states, _ = outs
        pc_targets, hd_targets = (pc_targets.transpose(1,0),
                                    hd_targets.transpose(1,0))

        # 256 place cells and 12 head direction cells
        logits_pc = logits_pc.view(-1, N_PC[0])
        logits_hd = logits_hd.view(-1, N_HDC[0])

        if coder:
            pc_targets, hd_targets = coder(pc_targets, target=True), coder(hd_targets, target=True)

        pc_targets = pc_targets.contiguous().view(-1, N_PC[0])
        hd_targets = hd_targets.contiguous().view(-1, N_HDC[0])

        return bottleneck_acts, logits_pc, logits_hd, pc_targets, hd_targets

# function for computing loss, one parameter bottleneck_acts not used
def get_loss(logits_pc, logits_hd, pc_targets, hd_targets, bottleneck_acts):
    pc_loss = cross_entropy(logits_pc,pc_targets)
    hd_loss = cross_entropy(logits_hd, hd_targets)

    return torch.mean(pc_loss + hd_loss) 

# coder
coder = PadCoder(3)


if __name__ == '__main__':
    torch.save(target_ensembles, SAVE_LOC + 'target_ensembles.pt')
    torch.save(model.state_dict(), SAVE_LOC + 'model_epoch_0.pt')

    # Store the grid scores
    grid_scores = dict()
    grid_scores['btln_60'] = np.zeros((NH_BOTTLENECK,))
    grid_scores['btln_90'] = np.zeros((NH_BOTTLENECK,))
    grid_scores['btln_60_separation'] = np.zeros((NH_BOTTLENECK,))
    grid_scores['btln_90_separation'] = np.zeros((NH_BOTTLENECK,))
    grid_scores['lstm_60'] = np.zeros((NH_LSTM,))
    grid_scores['lstm_90'] = np.zeros((NH_LSTM,))
    
    # Create scorer
    starts = [0.2] * 10
    ends = np.linspace(0.4, 1.0, num=10)
    masks_parameters = zip(starts, ends.tolist())
    latest_epoch_scorer = scores.GridScorer(20, ((-1.1, 1.1), (-1.1, 1.1)),
                                          masks_parameters)

    for e in range(start_epoch, N_EPOCHS):


        model.train()
        step = 0
        losses = []
        for X, y in data_generator:
            #Avoid gradient accumulation
            optimiser.zero_grad()

            (init_pos,
            init_hd,
            inputs,
            target_pos,
            target_hd,
            initial_conds,
            ensembles_targets) = encode_inputs(X, y, place_cell_ensembles, head_direction_ensembles, coder=coder)



            outs  = model.forward(inputs, initial_conds)

            bottleneck_acts, logits_pc, logits_hd, pc_targets, hd_targets = decode_outputs(outs, ensembles_targets, coder=coder)
            loss = get_loss(logits_pc, logits_hd, pc_targets, hd_targets, bottleneck_acts)

            loss += model.l2_loss * WEIGHT_DECAY
            loss.backward()
            nn.utils.clip_grad_value_(params, GRAD_CLIPPING)
            optimiser.step()
            # just collect the losses for 10 epoches and then output the mean of them
            losses.append(loss.clone().item())
            if step > STEPS_PER_EPOCH:
                break

            step += 1
        print("EPOCH", e, 'LOSS :', torch.mean(torch.Tensor(losses)))

        #evaluation routine
        if e % 10 == 0 and e > 0:
            state_dict = model.state_dict()
            for k, v in state_dict.items():
                state_dict[k] = v.cpu()
            torch.save(state_dict, SAVE_LOC + 'model_epoch_{}.pt'.format(e))
            with torch.no_grad():
                model.eval()
                for data in test_generator:
                    test_X, test_y = data

                    (init_pos,
                    init_hd,
                    inputs,
                    target_pos,
                    target_hd,
                    initial_conds,
                    ensembles_targets) = encode_inputs(test_X, test_y, place_cell_ensembles, head_direction_ensembles, coder=coder)


                    outs  = model.forward(inputs, initial_conds)

                    bottleneck_acts, logits_pc, logits_hd, pc_targets, hd_targets = decode_outputs(outs, ensembles_targets, coder=coder)

                    loss = get_loss(logits_pc, logits_hd, pc_targets, hd_targets, bottleneck_acts)

                    print("LOSS:", loss)

                    acts = bottleneck_acts.transpose(1,0).detach().numpy()
                    pos_xy = target_pos.detach().numpy()
                    print(target_pos.shape, acts.shape)
                    print(bottleneck_acts.shape, pos_xy.shape)
                    assert(pos_xy.shape == (BATCH_SIZE, STEPS_PER_EPOCH, 2))
                    assert(acts.shape == (BATCH_SIZE, STEPS_PER_EPOCH, NH_BOTTLENECK))
                    
                    """
                    
                    targets_pos: B * T * 2
                    bottleneck_acts: B * T * nh_bottleneck

                    """
                    # Store at the end of validation, added by Weizhi
                    filename = 'rates_and_sac_latest_hd_{}.pdf'.format(e)
                    grid_scores['btln_60'], grid_scores['btln_90'], grid_scores['btln_60_separation'], grid_scores['btln_90_separation'] \
                    = utils.get_scores_and_plot(latest_epoch_scorer, pos_xy, acts, saver_results_directory, filename)


                    break

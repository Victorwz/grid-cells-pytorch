"""
Pytorch implementation of the supervised part of:

"Vector-based navigation using grid-like representations in artificial agents" (Banino et al., 2018)

Lucas Pompe, 2019
"""
import torch
from torch import nn
import numpy as np

def truncated_normal_(tensor, mean=0, std=1):
    # https://discuss.pytorch.org/t/implementing-truncated-normal-initializer/4778/15
    size = tensor.shape
    tmp = tensor.new_empty(size + (4,)).normal_()
    valid = (tmp < 2) & (tmp > -2)
    ind = valid.max(-1, keepdim=True)[1]
    tensor.data.copy_(tmp.gather(-1, ind).squeeze(-1))
    tensor.data.mul_(std).add_(mean)
    return tensor

def init_trunc_normal(t, size):
    std = 1. / np.sqrt(size)
    return truncated_normal_(t, 0, std)

def rearrange_tf_weights(weights):
    i, j, f, o = weights.chunk(4, 0)
    return torch.cat((i, f, j, o))

def load_tf_param(loc):
    return nn.Parameter(torch.Tensor(np.load(loc).T))

class GridTorch(nn.Module):

    def __init__(self,
               target_ensembles,
               input_size,
               init_conds_size=268,
               nh_lstm=128,
               nh_bottleneck=256,
               n_pcs = 256,
               n_hdcs = 12,
               dropoutrates_bottleneck=0.5,
               bottleneck_weight_decay=1e-5,
               bottleneck_has_bias=False,
               init_weight_disp=0.0,
               tf_weights_loc=None):

        super(GridTorch, self).__init__()
        self.target_ensembles = target_ensembles

        self.rnn = TFLSTMCell(n_inputs=3,
                            n_units=nh_lstm,
                            #batch_first=True
                            )

        self.bottleneck = nn.Linear(nh_lstm, nh_bottleneck,
                                                bias=bottleneck_has_bias)
        self.pc_logits = nn.Linear(nh_bottleneck, target_ensembles[0].n_cells)
        self.hd_logits = nn.Linear(nh_bottleneck, target_ensembles[1].n_cells)

        self.state_embed = nn.Linear(init_conds_size, nh_lstm)
        self.cell_embed = nn.Linear(init_conds_size, nh_lstm)

        self.dropout = nn.Dropout(dropoutrates_bottleneck)


        with torch.no_grad():
            self.state_embed.weight = init_trunc_normal(self.state_embed.weight, 128)
            self.cell_embed.weight = init_trunc_normal(self.cell_embed.weight, 128)
            self.bottleneck.weight = init_trunc_normal(self.bottleneck.weight, 256)
            self.pc_logits.weight = init_trunc_normal(self.pc_logits.weight, 256)
            self.hd_logits.weight = init_trunc_normal(self.hd_logits.weight, 12)

            nn.init.zeros_(self.state_embed.bias)
            nn.init.zeros_(self.cell_embed.bias)
            nn.init.zeros_(self.pc_logits.bias)
            nn.init.zeros_(self.hd_logits.bias)

        if tf_weights_loc:
            self.init_tf_weights(tf_weights_loc)

    @property
    def l2_loss(self,):
        return (self.bottleneck.weight.norm(2) +
                    self.pc_logits.weight.norm(2) +
                    self.hd_logits.weight.norm(2))


    def init_tf_weights(self, loc):

        self.pc_logits.bias = load_tf_param(loc + 'grid_cells_core_pc_logits_b:0.npy')
        self.pc_logits.weight = load_tf_param(loc + 'grid_cells_core_pc_logits_w:0.npy')

        self.hd_logits.bias = load_tf_param(loc + 'grid_cells_core_pc_logits_1_b:0.npy')
        self.hd_logits.weight = load_tf_param(loc + 'grid_cells_core_pc_logits_1_w:0.npy')

        self.bottleneck.weight = load_tf_param(loc + 'grid_cells_core_bottleneck_w:0.npy')

        self.state_embed.bias = load_tf_param(loc + "grid_cell_supervised_state_init_b:0.npy")
        self.state_embed.weight = load_tf_param(loc + "grid_cell_supervised_state_init_w:0.npy")

        self.cell_embed.bias = load_tf_param(loc + "grid_cell_supervised_cell_init_b:0.npy")
        self.cell_embed.weight = load_tf_param(loc + "grid_cell_supervised_cell_init_w:0.npy")

        lstm_ws = load_tf_param(loc + "grid_cells_core_lstm_w_gates:0.npy")
        lstm_bs = load_tf_param(loc + "grid_cells_core_lstm_b_gates:0.npy")

        self.rnn.weight = nn.Parameter(lstm_ws.transpose(1, 0))
        self.rnn.bias = nn.Parameter(lstm_bs)



    def forward(self, x, initial_conds):
        init = torch.cat(initial_conds, dim=1)

        init_state = self.state_embed(init)
        init_cell = self.cell_embed(init)

        h_t, c_t = init_state , init_cell
        logits_hd = []
        logits_pc = []
        bottleneck_acts = []
        rnn_states = []
        cell_states = []
        for t in x: # get rnn output predictions
            h_t, c_t = self.rnn(t, (h_t, c_t))


            bottleneck_activations = self.dropout(self.bottleneck(h_t))

            pc_preds = self.pc_logits(bottleneck_activations)
            hd_preds = self.hd_logits(bottleneck_activations)

            logits_hd += [hd_preds]
            logits_pc += [pc_preds]
            bottleneck_acts += [bottleneck_activations]
            rnn_states += [h_t]
            cell_states += [c_t]

        final_state = h_t
        outs = (torch.stack(logits_hd),
                torch.stack(logits_pc),
                torch.stack(bottleneck_acts),
                torch.stack(rnn_states),
                torch.stack(cell_states))
        return outs



class TFLSTMCell(nn.Module):

    def __init__(self, n_inputs=3, n_units=128):
        super(TFLSTMCell, self).__init__()
        self.n_units = n_units
        self.n_inputs = n_inputs

        with torch.no_grad():
            wi = wj = wf = wo = torch.Tensor(n_units + n_inputs, n_units)
            ws = [wi, wj ,wf, wo]
            for i, w in enumerate(ws):
                ws[i] = nn.init.kaiming_uniform_(w)





            self.weight = nn.Parameter(torch.cat(ws, dim=1), requires_grad=True)
            self.bias = nn.Parameter(torch.zeros((n_units * 4,)), requires_grad=True)




    def forward(self, x_t, state):
        """
        forward pass of the mode
        """
        h_tm1, c_tm1 = state
        args = torch.cat((x_t,h_tm1), dim=1)

        out = torch.matmul(args, self.weight) + self.bias

        i, j, f, o = torch.split(out, self.n_units, 1)

        g = torch.tanh(j)
        sigmoid_f = nn.functional.sigmoid(f + 1.)
        c_t = torch.mul(c_tm1, sigmoid_f) + torch.mul(nn.functional.sigmoid(i) ,g)
        h_t = torch.mul(torch.tanh(c_t),nn.functional.sigmoid(o))

        return h_t, c_t

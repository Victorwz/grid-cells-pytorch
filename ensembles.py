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

"""Ensembles of place and head direction cells.

These classes provide the targets for the training of grid-cell networks.


-----

Adapted for PyTorch by Lucas Pompe, 2019
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import torch


def one_hot(batch,depth):
    emb = nn.Embedding(depth, depth)
    emb.weight.data = torch.eye(depth)
    return emb(batch)


def one_hot_max(x, axis=-1):
  """Compute one-hot vectors setting to one the index with the maximum value."""
  print(x.shape)
  _, idx = torch.max(x, axis)
  depth = x.shape[-1]
  return one_hot(idx, depth)


def softmax(x, axis=-1):
  """Compute softmax values for each sets of scores in x."""
  return torch.nn.functional.softmax(x, dim=axis)


def softmax_sample(x):
  """Sample the categorical distribution from logits and sample it."""
  dist = torch.distributions.OneHotCategorical(logits=x)
  return dist.sample()


class CellEnsemble(object):
  """Abstract parent class for place and head direction cell ensembles."""

  def __init__(self, n_cells, soft_targets, soft_init):
    self.n_cells = n_cells
    if soft_targets not in ["softmax", "voronoi", "sample", "normalized"]:
      raise ValueError
    else:
      self.soft_targets = soft_targets
    # Provide initialization of LSTM in the same way as targets if not specified
    # i.e one-hot if targets are Voronoi
    if soft_init is None:
      self.soft_init = soft_targets
    else:
      if soft_init not in [
          "softmax", "voronoi", "sample", "normalized", "zeros"
      ]:
        raise ValueError
      else:
        self.soft_init = soft_init

  def get_targets(self, x):
    """Type of target."""

    if self.soft_targets == "normalized":
      targets = torch.exp(self.unnor_logpdf(x))
    elif self.soft_targets == "softmax":
      lp = self.log_posterior(x)
      targets = softmax(lp)
    elif self.soft_targets == "sample":
      lp = self.log_posterior(x)
      targets = softmax_sample(lp)
    elif self.soft_targets == "voronoi":
      lp = self.log_posterior(x)
      targets = one_hot_max(lp)
    return targets

  def get_init(self, x):
    """Type of initialisation."""

    if self.soft_init == "normalized":
      init = torch.exp(self.unnor_logpdf(x))
    elif self.soft_init == "softmax":
      lp = self.log_posterior(x)
      init = softmax(lp)
    elif self.soft_init == "sample":
      lp = self.log_posterior(x)
      init = softmax_sample(lp)
    elif self.soft_init == "voronoi":
      lp = self.log_posterior(x)
      init = one_hot_max(lp)
    elif self.soft_init == "zeros":
      init = torch.zeros_like(self.unnor_logpdf(x))

      print(init)
    return init

  def loss(self, predictions, targets):
    """Loss."""
    crit = torch.nn.BCEWithLogitsLoss()
    if self.soft_targets == "normalized":
      smoothing = 1e-2
      loss = crit((1. - smoothing) * targets + smoothing * 0.5,
            predictions)
      loss = torch.mean(loss, dim=-1)
    else:
      loss = crit(
          targets,
          predictions)
    return loss

  def log_posterior(self, x):
    logp = self.unnor_logpdf(x)
    log_posteriors = logp - torch.logsumexp(logp, dim=2, keepdim=True)
    return log_posteriors


class PlaceCellEnsemble(CellEnsemble):
  """Calculates the dist over place cells given an absolute position."""

  def __init__(self, n_cells, stdev=0.35, pos_min=-5, pos_max=5, seed=None,
               soft_targets=None, soft_init=None):
    super(PlaceCellEnsemble, self).__init__(n_cells, soft_targets, soft_init)
    # Create a random MoG with fixed cov over the position (Nx2)
    rs = torch.manual_seed(seed)
    uni = torch.distributions.Uniform(pos_min, pos_max)

    self.means = uni.sample((self.n_cells, 2))
    self.variances = (torch.ones_like(self.means) * stdev**2)

  def unnor_logpdf(self, trajs):
    # Output the probability of each component at each point (BxTxN)
    diff = trajs[:, :, None, :] - self.means[None, None, ...]
    unnor_logp = -0.5 * torch.sum((diff**2)/ self.variances, dim=-1)
    return unnor_logp


class HeadDirectionCellEnsemble(CellEnsemble):
  """Calculates the dist over HD cells given an absolute angle."""

  def __init__(self, n_cells, concentration=20, seed=None,
               soft_targets=None, soft_init=None):
    super(HeadDirectionCellEnsemble, self).__init__(n_cells,
                                                    soft_targets,
                                                    soft_init)
    # Create a random Von Mises with fixed cov over the position
    rs = torch.manual_seed(seed)
    uni = torch.distributions.Uniform(-np.pi, np.pi)
    self.means = uni.sample((n_cells,))
    self.kappa = torch.ones_like(self.means) * concentration

  def unnor_logpdf(self, x):
    return self.kappa * torch.cos(x - self.means[None, None, :])

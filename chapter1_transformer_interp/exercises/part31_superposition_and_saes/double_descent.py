#%%
import sys
from dataclasses import dataclass
import math
from pathlib import Path
from typing import Callable, Literal
import einops
import numpy as np
import torch as t
from jaxtyping import Float
from torch import Tensor, nn
from torch.distributions.categorical import Categorical
from torch.nn import functional as F
from tqdm.auto import tqdm

# Make sure exercises are in the path
chapter = r"chapter1_transformer_interp"
exercises_dir = Path(f"{os.getcwd().split(chapter)[0]}/{chapter}/exercises").resolve()
section_dir = exercises_dir / "part31_superposition_and_saes"
if str(exercises_dir) not in sys.path: sys.path.append(str(exercises_dir))

import part31_superposition_and_saes.utils as utils
import part31_superposition_and_saes.tests as tests
from plotly_utils import line, imshow

device = t.device(
    "mps" if t.backends.mps.is_available() else "cuda" if t.cuda.is_available() else "cpu"
)

MAIN = __name__ == "__main__"


# %%
NUM_WARMUP_STEPS = 2500
NUM_BATCH_UPDATES = 50_000

WEIGHT_DECAY = 1e-2
LEARNING_RATE = 1e-3

BATCH_SIZES = [3, 5, 6, 8, 10, 15, 30, 50, 100, 200, 500, 1000, 2000]

N_FEATURES = 1000
N_INSTANCES = 5
N_HIDDEN = 2
SPARSITY = 0.99
FEATURE_PROBABILITY = 1 - SPARSITY

#%%
def linear_warmup(step, steps):
    return step / steps

def cosine_decay(step, num_warmup_steps, num_training_steps):
    0.5 * (1 + math.cos(math.pi * (step - num_warmup_steps) / (num_training_steps - num_warmup_steps)))


def get_lr(step, num_batch_updates):
    if step < NUM_WARMUP_STEPS:
        print(linear_warmup(step, NUM_WARMUP_STEPS))
        return linear_warmup(step, NUM_WARMUP_STEPS)
    else:
        return cosine_decay(step, NUM_WARMUP_STEPS, num_batch_updates)

#%%
@dataclass
class Config:
    # We optimize n_inst models in a single training loop to let us sweep over sparsity or importance
    # curves efficiently. You should treat the number of instances `n_inst` like a batch dimension, 
    # but one which is built into our training setup. Ignore the latter 3 arguments for now, they'll
    # return in later exercises.
    n_inst: int
    n_features: int = 5
    d_hidden: int = 2
    n_correlated_pairs: int = 0
    n_anticorrelated_pairs: int = 0
    feat_mag_distn: Literal["unif", "jump"] = "unif"


class Model(nn.Module):
    W: Float[Tensor, "inst d_hidden feats"]
    b_final: Float[Tensor, "inst feats"]

    # Our linear map (for a single instance) is x -> ReLU(W.T @ W @ x + b_final)

    def __init__(
        self,
        cfg: Config,
        feature_probability: float | Tensor = 0.01,
        importance: float | Tensor = 1.0,
        device=device,
    ):
        super(Model, self).__init__()
        self.cfg = cfg

        if isinstance(feature_probability, float):
            feature_probability = t.tensor(feature_probability)
        self.feature_probability = feature_probability.to(device).broadcast_to(
            (cfg.n_inst, cfg.n_features)
        )
        if isinstance(importance, float):
            importance = t.tensor(importance)
        self.importance = importance.to(device).broadcast_to((cfg.n_inst, cfg.n_features))

        self.W = nn.Parameter(
            nn.init.xavier_normal_(t.empty((cfg.n_inst, cfg.d_hidden, cfg.n_features)))
        )
        self.b_final = nn.Parameter(t.zeros((cfg.n_inst, cfg.n_features)))
        self.to(device)

    @classmethod
    def dimensionality(
        cls, data: Float[Tensor, "... batch d_hidden"]
    ) -> Float[Tensor, "... batch"]:
        """
        Calculates dimensionalities of data. Assumes data is of shape ... batch d_hidden, i.e. if it's 2D then
        it's a batch of vectors of length `d_hidden` and we return the dimensionality as a 1D tensor of length
        `batch`. If it has more dimensions at the start, we assume this means separate calculations for each
        of these dimensions (i.e. they are independent batches of vectors).
        """
        # Compute the norms of each vector (this will be the numerator)
        squared_norms = einops.reduce(data.pow(2), "... batch d_hidden -> ... batch", "sum")
        # Compute the denominator (i.e. get the dotproduct then sum over j)
        data_normed = data / data.norm(dim=-1, keepdim=True)
        interference = einops.einsum(
            data_normed, data, "... batch_i d_hidden, ... batch_j d_hidden -> ... batch_i batch_j"
        )
        polysemanticity = einops.reduce(
            interference.pow(2), "... batch_i batch_j -> ... batch_i", "sum"
        )
        assert squared_norms.shape == polysemanticity.shape

        return squared_norms / polysemanticity


    def forward(
        self,
        features: Float[Tensor, "... inst feats"],
    ) -> Float[Tensor, "... inst feats"]:
        # YOUR CODE HERE
        w_transpose = einops.rearrange(self.W, "n_ist d_hidden n_features -> n_ist n_features d_hidden")
        hidden = einops.einsum(self.W, features, "n_ist d_hidden n_features, ... n_ist n_features -> ... n_ist d_hidden")
        out = F.relu(einops.einsum(w_transpose, hidden, "n_ist n_features d_hidden, ... n_ist d_hidden -> ... n_ist n_features") + self.b_final)
        return out
        # return F.relu(einops.einsum(w_transpose, self.W, features, "n_ist n_features d_hidden, n_ist d_hidden n_features, ... inst feats -> ... inst feats") + self.b_final)


    def generate_batch(self, batch_size) -> Float[Tensor, "batch inst feats"]:
        """
        Generates a batch of data.
        """
        ...

        out = t.rand(size=(batch_size, *self.feature_probability.shape)).to(device)

        is_present = t.bernoulli(einops.repeat(self.feature_probability, "inst feats -> batch inst feats", batch=batch_size)).bool()

        out[~is_present] = 0

        out_norm = out.norm(dim=-1, keepdim=True)

        out_norm = t.where(out_norm.abs() < 1e-6, t.ones_like(out_norm), out_norm)

        normalized_out = out/out_norm

        return normalized_out
        


    def calculate_loss(
        self,
        out: Float[Tensor, "batch inst feats"],
        batch: Float[Tensor, "batch inst feats"],
    ) -> Float[Tensor, ""]:
        """
        Calculates the loss for a given batch (as a scalar tensor), using this loss described in the
        Toy Models of Superposition paper:

            https://transformer-circuits.pub/2022/toy_model/index.html#demonstrating-setup-loss

        Note, `self.importance` is guaranteed to broadcast with the shape of `out` and `batch`.
        """
        # You'll fill this in later
        loss = einops.einsum(self.importance, (batch - out)**2, "inst feats, batch inst feats -> ")
        batch_size = out.shape[0]
        return loss/(batch_size*self.cfg.n_features)


    def optimize(
        self,
        batch_size: int = 1024,
        steps: int = NUM_BATCH_UPDATES,
        log_freq: int = 50,
        lr: float = 1e-3,
        lr_scale: Callable[[int, int], float] = get_lr,
    ):
        """
        Optimizes the model using the given hyperparameters.
        """
        optimizer = t.optim.AdamW(list(self.parameters()), lr=lr, weight_decay=WEIGHT_DECAY)

        progress_bar = tqdm(range(steps))

        for step in progress_bar:
            # Update learning rate
            print(lr_scale(step, steps))
            step_lr = lr * lr_scale(step, steps)
            for group in optimizer.param_groups:
                group["lr"] = step_lr

            # Optimize
            optimizer.zero_grad()
            batch = self.generate_batch(batch_size)
            out = self(batch)
            loss = self.calculate_loss(out, batch)
            loss.backward()
            optimizer.step()

            # Display progress bar
            if step % log_freq == 0 or (step + 1 == steps):
                progress_bar.set_postfix(loss=loss.item() / self.cfg.n_inst, lr=step_lr)

        with t.inference_mode():
            out = self.forward(batch)
            loss_per_inst = self.calculate_loss(out, batch, per_inst=True)
            best_inst = loss_per_inst.argmin()
            print(f"Best instance = #{best_inst}, with loss {loss_per_inst[best_inst].item():.4e}")

        return batch[:, best_inst], self.W[best_inst].detach()

# %%
tests.test_model(Model)
tests.test_generate_batch(Model)
tests.test_calculate_loss(Model)

# %%
features_list = []
hidden_representations_list = []

for batch_size in tqdm(BATCH_SIZES):
    # Define our model
    cfg = Config(n_features=N_FEATURES, n_inst=N_INSTANCES, d_hidden=N_HIDDEN)
    model = Model(cfg, feature_probability=FEATURE_PROBABILITY).to(device)

    # Optimize, and return the best batch & weight matrix
    batch_inst, W_inst = model.optimize(batch_size=batch_size, steps=15_000)

    # Calculate the hidden feature representations, and add both this and weight matrix to our lists of data
    with t.inference_mode():
        hidden = einops.einsum(
            batch_inst, W_inst, "batch features, hidden features -> hidden batch"
        )
    features_list.append(W_inst.cpu())
    hidden_representations_list.append(hidden.cpu())
# %%

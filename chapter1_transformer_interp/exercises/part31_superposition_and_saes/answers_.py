#%%
import sys
from dataclasses import dataclass
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
t.manual_seed(2)

W = t.randn(2, 5)
W_normed = W / W.norm(dim=0, keepdim=True)

imshow(W_normed.T @ W_normed, title="Cosine similarities of each pair of 2D feature embeddings", width=600)
# %%
utils.plot_features_in_2d(W_normed)
# %%
def linear_lr(step, steps):
    return (1 - (step / steps))

def constant_lr(*_):
    return 1.0

def cosine_decay_lr(step, steps):
    return np.cos(0.5 * np.pi * step / (steps - 1))


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

        return out
        


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
        steps: int = 10_000,
        log_freq: int = 50,
        lr: float = 1e-3,
        lr_scale: Callable[[int, int], float] = constant_lr,
    ):
        """
        Optimizes the model using the given hyperparameters.
        """
        optimizer = t.optim.Adam(list(self.parameters()), lr=lr)

        progress_bar = tqdm(range(steps))

        for step in progress_bar:
            # Update learning rate
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


tests.test_model(Model)
# %%
tests.test_generate_batch(Model)
# %%
tests.test_calculate_loss(Model)
# %%
cfg = Config(n_inst=8, n_features=5, d_hidden=2)

# importance varies within features for each instance
importance = (0.9 ** t.arange(cfg.n_features))

# sparsity is the same for all features in a given instance, but varies over instances
feature_probability = (50 ** -t.linspace(0, 1, cfg.n_inst))

line(importance, width=600, height=400, title="Importance of each feature (same over all instances)", labels={"y": "Feature importance", "x": "Feature"})
line(feature_probability, width=600, height=400, title="Feature probability (varied over instances)", labels={"y": "Probability", "x": "Instance"})
# %%
model = Model(
    cfg=cfg,
    device=device,
    importance=importance[None, :],
    feature_probability=feature_probability[:, None],
)
model.optimize(steps=10_000)

utils.plot_features_in_2d(
    model.W,
    colors=model.importance,
    title=f"Superposition: {cfg.n_features} features represented in 2D space",
    subplot_titles=[f"1 - S = {i:.3f}" for i in feature_probability.squeeze()],
)
# %%
with t.inference_mode():
    batch = model.generate_batch(250)
    h = einops.einsum(
        batch, model.W, "batch inst feats, inst hidden feats -> inst hidden batch"
    )

utils.plot_features_in_2d(h, title="Hidden state representation of a random batch of data")
# %%
cfg = Config(n_inst=10, n_features=100, d_hidden=20)

importance = 100 ** -t.linspace(0, 1, cfg.n_features)
feature_probability = 20 ** -t.linspace(0, 1, cfg.n_inst)

line(importance, width=600, height=400, title="Importance of each feature (same over all instances)", labels={"y": "Feature importance", "x": "Feature"})
line(feature_probability, width=600, height=400, title="Feature probability (varied over instances)", labels={"y": "Probability", "x": "Instance"})

model = Model(
    cfg=cfg,
    device=device,
    importance=importance[None, :],
    feature_probability=feature_probability[:, None],
)
model.optimize(steps=10_000)
# %%
utils.plot_features_in_Nd(
    model.W,
    height=800,
    width=1600,
    title="ReLU output model: n_features = 80, d_hidden = 20, I<sub>i</sub> = 0.9<sup>i</sup>",
    subplot_titles=[f"Feature prob = {i:.3f}" for i in feature_probability],
)
# %%

class Model(nn.Module):
    ...
    def generate_correlated_features(
        self, batch_size: int, n_correlated_pairs: int
    ) -> Float[Tensor, "batch inst 2*n_correlated_pairs"]:
        """
        Generates a batch of correlated features. For each pair `batch[i, j, [2k, 2k+1]]`, one of
        them is non-zero if and only if the other is non-zero.
        """
        assert t.all((self.feature_probability == self.feature_probability[:, [0]]))
        p = self.feature_probability[:, [0]]  # shape (n_inst, 1)

        # YOUR CODE HERE!
        out = t.rand((batch_size, p.shape[0], 2*n_correlated_pairs)).to(device)

        p = einops.repeat(p, "n_inst 1 ->  batch n_correlated_pairs n_inst", batch=batch_size, n_correlated_pairs=n_correlated_pairs)

        is_present = t.bernoulli(p).bool()

        is_present = einops.repeat(is_present, "batch n_inst n_correlated_pairs ->  batch n_inst (n_correlated_pairs 2)")

        out[~is_present] = 0

        return out

    def generate_anticorrelated_features(
        self, batch_size: int, n_anticorrelated_pairs: int
    ) -> Float[Tensor, "batch inst 2*n_anticorrelated_pairs"]:
        """
        Generates a batch of anti-correlated features. For each pair `batch[i, j, [2k, 2k+1]]`, each
        of them can only be non-zero if the other one is zero.
        """
        assert t.all((self.feature_probability == self.feature_probability[:, [0]]))
        p = self.feature_probability[:, [0]]  # shape (n_inst, 1)

        assert p.max().item() <= 0.5, "For anticorrelated features, must have 2p < 1"

        p_ = einops.repeat(p, "n_inst 1 ->  batch n_inst pairs", batch=batch_size, pairs=n_anticorrelated_pairs).to(device)

        pair_is_present = t.bernoulli(p_).to(device)

        pair_probs = (t.rand(pair_is_present.shape) > 0.5).to(device)
        X = pair_is_present + (pair_is_present * pair_probs)

        left_pairs = t.zeros_like(X).to(device)
        right_pairs = t.zeros_like(X).to(device)

        left_pairs[X==1] = 1
        right_pairs[X==2] = 1

        stacked = t.stack([left_pairs, right_pairs], dim=-1)
    
        is_present = einops.rearrange(stacked, '... n (interleave 2) -> ... (n interleave)', interleave=2)

        out = t.rand((batch_size, p.shape[0], 2*n_anticorrelated_pairs)).to(device)
        out[~is_present] = 0

        return out


    def generate_uncorrelated_features(self, batch_size: int, n_uncorrelated: int) -> Tensor:
        """
        Generates a batch of uncorrelated features.
        """
        if n_uncorrelated == self.cfg.n_features:
            p = self.feature_probability
        else:
            assert t.all((self.feature_probability == self.feature_probability[:, [0]]))
            p = self.feature_probability[:, [0]]  # shape (n_inst, 1)

        # YOUR CODE HERE!
        p = einops.repeat(p, "n_inst 1 ->  batch n_inst n_uncorrelated", batch=batch_size, n_uncorrelated=n_uncorrelated)

        out = t.rand(size=(batch_size, p.shape[0])).to(device)

        is_present = t.bernoulli(p).bool()

        out[~is_present] = 0

        return out


    def generate_batch(self, batch_size) -> Float[Tensor, "batch inst feats"]:
        """
        Generates a batch of data, with optional correlated & anticorrelated features.
        """
        n_corr_pairs = self.cfg.n_correlated_pairs
        n_anti_pairs = self.cfg.n_anticorrelated_pairs
        n_uncorr = self.cfg.n_features - 2 * n_corr_pairs - 2 * n_anti_pairs

        data = []
        if n_corr_pairs > 0:
            data.append(self.generate_correlated_features(batch_size, n_corr_pairs))
        if n_anti_pairs > 0:
            data.append(self.generate_anticorrelated_features(batch_size, n_anti_pairs))
        if n_uncorr > 0:
            data.append(self.generate_uncorrelated_features(batch_size, n_uncorr))
        batch = t.cat(data, dim=-1)
        return batch

#%%
cfg = Config(n_inst=30, n_features=4, d_hidden=2, n_correlated_pairs=1, n_anticorrelated_pairs=1)

feature_probability = 10 ** -t.linspace(0.5, 1, cfg.n_inst).to(device)

model = Model(cfg=cfg, device=device, feature_probability=feature_probability[:, None])

# Generate a batch of 4 features: first 2 are correlated, second 2 are anticorrelated
batch = model.generate_batch(batch_size=100_000)
corr0, corr1, anticorr0, anticorr1 = batch.unbind(dim=-1)

assert ((corr0 != 0) == (corr1 != 0)).all(), "Correlated features should be active together"
assert (
    ((corr0 != 0).float().mean(0) - feature_probability).abs().mean() < 0.002
), "Each correlated feature should be active with probability `feature_probability`"

assert (
    (anticorr0 != 0) & (anticorr1 != 0)
).int().sum().item() == 0, "Anticorrelated features should never be active together"
assert (
    ((anticorr0 != 0).float().mean(0) - feature_probability).abs().mean() < 0.002
), "Each anticorrelated feature should be active with probability `feature_probability`"
# %%

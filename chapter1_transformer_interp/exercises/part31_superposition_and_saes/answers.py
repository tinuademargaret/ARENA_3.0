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

        p = einops.repeat(p, "n_inst 1 ->  batch n_inst n_correlated_pairs", batch=batch_size, n_correlated_pairs=n_correlated_pairs)

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

        pair_is_present = t.bernoulli(2*p_).bool().to(device)

        left_probs = (t.rand(pair_is_present.shape) > 0.5).to(device)

        left_pairs = t.zeros_like(pair_is_present).to(device)
        right_pairs = t.zeros_like(pair_is_present).to(device)

        left_pairs[pair_is_present & left_probs] = 1
        right_pairs[pair_is_present & ~left_probs] = 1

        stacked = t.stack([left_pairs, right_pairs], dim=-1)
    
        is_present = einops.rearrange(stacked, '... n interleave -> ... (n interleave)').bool()

        out = t.rand((batch_size, p.shape[0], 2*n_anticorrelated_pairs)).to(device)
        out[~is_present] = 0

        return out


    def generate_uncorrelated_features(self, batch_size: int, n_uncorrelated: int) -> Float[Tensor, "batch inst n_uncorrelated"]:
        """
        Generates a batch of uncorrelated features.
        """
        if n_uncorrelated == self.cfg.n_features:
            p = self.feature_probability
            p = einops.repeat(p, "n_inst n_feats -> batch n_inst n_feats", batch=batch_size)
        else:
            assert t.all((self.feature_probability == self.feature_probability[:, [0]]))
            p = self.feature_probability[:, [0]]  # shape (n_inst, 1)
            p = einops.repeat(p, "n_inst 1 -> batch n_inst n_uncorrelated", batch=batch_size, n_uncorrelated=n_uncorrelated)

        n_inst = p.shape[1]
        out = t.rand(size=(batch_size, n_inst, n_uncorrelated)).to(device)

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

print("All tests passed!")
# %%
# Generate a batch of 4 features: first 2 are correlated, second 2 are anticorrelated
batch = model.generate_batch(batch_size=1)
correlated_feature_batch, anticorrelated_feature_batch = batch.split(2, dim=-1)

# Plot correlated features
utils.plot_correlated_features(
    correlated_feature_batch, title="Correlated feature pairs: should always co-occur"
)
utils.plot_correlated_features(
    anticorrelated_feature_batch, title="Anti-correlated feature pairs: should never co-occur"
)
# %%
cfg = Config(n_inst=5, n_features=6, d_hidden=2, n_correlated_pairs=3)

# All same importance, very low feature probabilities (ranging from 5% down to 0.25%)
importance = t.ones(cfg.n_features, dtype=t.float, device=device)
feature_probability = 400 ** -t.linspace(0.5, 1, cfg.n_inst)

model = Model(
    cfg=cfg,
    device=device,
    importance=importance[None, :],
    feature_probability=feature_probability[:, None],
)
model.optimize(steps=10_000)

utils.plot_features_in_2d(
    model.W,
    colors=["blue"] * 2 + ["limegreen"] * 2 + ["red"] * 2,
    title="Correlated feature sets are represented in local orthogonal bases",
    subplot_titles=[f"1 - S = {i:.3f}" for i in feature_probability],
)
# %%

class NeuronModel(Model):
    def forward(
        self,
        features: Float[Tensor, "... instances features"]
    ) -> Float[Tensor, "... instances features"]:
        w_transpose = einops.rearrange(self.W, "n_ist d_hidden n_features -> n_ist n_features d_hidden")
        hidden = F.relu(einops.einsum(self.W, features, "n_ist d_hidden n_features, ... n_ist n_features -> ... n_ist d_hidden"))
        out = F.relu(einops.einsum(w_transpose, hidden, "n_ist n_features d_hidden, ... n_ist d_hidden -> ... n_ist n_features") + self.b_final)
        return out


tests.test_neuron_model(NeuronModel)
# %%

cfg = Config(n_inst=7, n_features=10, d_hidden=5)

importance = 0.75 ** t.arange(1, 1 + cfg.n_features)
feature_probability = t.tensor([0.75, 0.35, 0.15, 0.1, 0.06, 0.02, 0.01])

model = NeuronModel(
    cfg=cfg,
    device=device,
    importance=importance[None, :],
    feature_probability=feature_probability[:, None],
)
model.optimize(steps=10_000)

utils.plot_features_in_Nd(
    model.W,
    height=600,
    width=1000,
    title=f"Neuron model: {cfg.n_features=}, {cfg.d_hidden=}, I<sub>i</sub> = 0.75<sup>i</sup>",
    subplot_titles=[f"1 - S = {i:.2f}" for i in feature_probability.squeeze()],
    neuron_plot=True,
)
# %%

class NeuronComputationModel(Model):
    W1: Float[Tensor, "inst d_hidden feats"]
    W2: Float[Tensor, "inst feats d_hidden"]
    b_final: Float[Tensor, "inst feats"]

    def __init__(
        self,
        cfg: Config,
        feature_probability: float | Tensor = 1.0,
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

        self.W1 = nn.Parameter(nn.init.kaiming_uniform_(t.empty((cfg.n_inst, cfg.d_hidden, cfg.n_features))))
        self.W2 = nn.Parameter(nn.init.kaiming_uniform_(t.empty((cfg.n_inst, cfg.n_features, cfg.d_hidden))))
        self.b_final = nn.Parameter(t.zeros((cfg.n_inst, cfg.n_features)))
        self.to(device)


    def forward(self, features: Float[Tensor, "... inst feats"]) -> Float[Tensor, "... inst feats"]:
        h = F.relu(einops.einsum(self.W1, features, "inst d_hidden features, ... inst features -> ... inst d_hidden"))
        y_prime = F.relu(einops.einsum(self.W2, h, "inst features d_hidden, ... inst d_hidden -> ... inst features") + self.b_final)
        return y_prime


    def generate_batch(self, batch_size) -> Tensor:
        out = (t.rand(size=(batch_size, *self.feature_probability.shape)).to(device) * 2) - 1

        is_present = t.bernoulli(einops.repeat(self.feature_probability, "inst feats -> batch inst feats", batch=batch_size)).bool()

        out[~is_present] = 0

        return out


    def calculate_loss(
        self,
        out: Float[Tensor, "batch instances features"],
        batch: Float[Tensor, "batch instances features"],
    ) -> Float[Tensor, ""]:
        loss = einops.einsum(self.importance, (t.abs(batch) - out)**2, "inst feats, batch inst feats -> ")
        batch_size = out.shape[0]
        return loss/(batch_size*self.cfg.n_features)


tests.test_neuron_computation_model(NeuronComputationModel)
# %%

cfg = Config(n_inst=7, n_features=100, d_hidden=40)

importance = 0.8 ** t.arange(1, 1 + cfg.n_features)
feature_probability = t.tensor([1.0, 0.3, 0.1, 0.03, 0.01, 0.003, 0.001])

model = NeuronComputationModel(
    cfg=cfg,
    device=device,
    importance=importance[None, :],
    feature_probability=feature_probability[:, None],
)
model.optimize(steps=10_000)


utils.plot_features_in_Nd(
    model.W1,
    height=800,
    width=1400,
    neuron_plot=True,
    subplot_titles=[f"1 - S = {i:.3f}<br>" for i in feature_probability.squeeze()],
    title=f"Neuron computation model: {cfg.n_features=}, {cfg.d_hidden=}, I<sub>i</sub> = 0.75<sup>i</sup>",
)

# %%

cfg = Config(n_inst=5, n_features=10, d_hidden=10)

importance = 0.8 ** t.arange(1, 1 + cfg.n_features)
feature_probability = 0.5

model = NeuronComputationModel(
    cfg=cfg,
    device=device,
    importance=importance[None, :],
    feature_probability=feature_probability,
)
model.optimize(steps=10_000)

utils.plot_features_in_Nd_discrete(
    W1=model.W1,
    W2=model.W2,
    title="Neuron computation model (colored discretely, by feature)",
    legend_names=[
        f"I<sub>{i}</sub> = {importance.squeeze()[i]:.3f}" for i in range(cfg.n_features)
    ],
)
# %%

@dataclass
class SAEConfig:
    n_inst: int
    d_in: int
    d_sae: int
    l1_coeff: float = 0.2
    weight_normalize_eps: float = 1e-8
    tied_weights: bool = False
    architecture: Literal["standard", "gated"] = "standard"


class SAE(nn.Module):
    W_enc: Float[Tensor, "inst d_in d_sae"]
    _W_dec: Float[Tensor, "inst d_sae d_in"] | None
    b_enc: Float[Tensor, "inst d_sae"]
    b_dec: Float[Tensor, "inst d_in"]

    def __init__(self, cfg: SAEConfig, model: Model) -> None:
        super(SAE, self).__init__()

        assert cfg.d_in == model.cfg.d_hidden, "Model's hidden dim doesn't match SAE input dim"
        self.cfg = cfg
        self.model = model.requires_grad_(False)

        self.b_enc = nn.Parameter(t.empty(self.cfg.n_inst, self.cfg.d_sae))
        self.W_enc = nn.Parameter(t.empty(self.cfg.n_inst, self.cfg.d_in, self.cfg.d_sae))
        self.b_dec = nn.Parameter(t.empty(self.cfg.n_inst, self.cfg.d_in))

        nn.init.kaiming_uniform_(self.b_enc)
        nn.init.kaiming_uniform_(self.W_enc)
        nn.init.kaiming_uniform_(self.b_dec)

        if self.cfg.tied_weights:
            self._W_dec = None
        else:
            self._W_dec = nn.Parameter(t.empty(self.cfg.n_inst, self.cfg.d_sae, self.cfg.d_in))
            nn.init.kaiming_uniform_(self._W_dec)



        self.to(device)

    @property
    def W_dec(self) -> Float[Tensor, "inst d_sae d_in"]:
        return self._W_dec if self._W_dec is not None else self.W_enc.transpose(-1, -2)

    @property
    def W_dec_normalized(self) -> Float[Tensor, "inst d_sae d_in"]:
        """Returns decoder weights, normalized over the autoencoder input dimension."""
        return self.W_dec / (self.cfg.d_in + self.cfg.weight_normalize_eps)

    def generate_batch(self, batch_size: int) -> Float[Tensor, "batch inst d_in"]:
        """
        Generates a batch of hidden activations from our model.
        """
        batch = self.model.generate_batch(batch_size)
        return einops.einsum(self.model.W, batch, "inst hidden feats, batch inst feats -> batch inst hidden")


    def forward(
        self, h: Float[Tensor, "batch inst d_in"]
    ) -> tuple[
        dict[str, Float[Tensor, "batch inst"]],
        Float[Tensor, ""],
        Float[Tensor, "batch inst d_sae"],
        Float[Tensor, "batch inst d_in"],
    ]:
        """
        Forward pass on the autoencoder.

        Args:
            h: hidden layer activations of model

        Returns:
            loss_dict: dict of different loss function term values, for every (batch elem, instance)
            loss: scalar total loss (summed over instances & averaged over batch dim)
            acts: autoencoder feature activations
            h_reconstructed: reconstructed autoencoder input
        """
        sae_latents = einops.einsum(self.W_enc, h - self.b_dec, "inst d_in d_sae, batch inst d_in -> batch inst d_sae") + self.b_enc
        z = F.relu(sae_latents)
        h_prime = einops.einsum(self.W_dec, z, "inst d_sae d_in, batch inst d_sae -> batch inst d_in") + self.b_dec
        mse = ((h - h_prime) ** 2).mean(dim=-1)
        l1 = t.abs(z).sum(dim=-1)

        loss = (mse + self.cfg.l1_coeff * l1).sum(dim=-1).mean()

        return {"L_reconstruction": mse, "L_sparsity": l1}, loss, z, h_prime


    def optimize(
        self,
        batch_size: int = 1024,
        steps: int = 10_000,
        log_freq: int = 50,
        lr: float = 1e-3,
        lr_scale: Callable[[int, int], float] = constant_lr,
        resample_method: Literal["simple", "advanced", None] = None,
        resample_freq: int = 2500,
        resample_window: int = 500,
        resample_scale: float = 0.5,
    ) -> dict[str, list]:
        """
        Optimizes the autoencoder using the given hyperparameters.

        Args:
            model:              we reconstruct features from model's hidden activations
            batch_size:         size of batches we pass through model & train autoencoder on
            steps:              number of optimization steps
            log_freq:           number of optimization steps between logging
            lr:                 learning rate
            lr_scale:           learning rate scaling function
            resample_method:    method for resampling dead latents
            resample_freq:      number of optimization steps between resampling dead latents
            resample_window:    number of steps needed for us to classify a neuron as dead
            resample_scale:     scale factor for resampled neurons

        Returns:
            data_log:               dictionary containing data we'll use for visualization
        """
        assert resample_window <= resample_freq

        optimizer = t.optim.Adam(list(self.parameters()), lr=lr, betas=(0.0, 0.999))
        frac_active_list = []
        progress_bar = tqdm(range(steps))

        # Create lists to store data we'll eventually be plotting
        data_log = {"steps": [], "W_enc": [], "W_dec": [], "frac_active": []}

        for step in progress_bar:
            # Resample dead latents
            if (resample_method is not None) and ((step + 1) % resample_freq == 0):
                frac_active_in_window = t.stack(frac_active_list[-resample_window:], dim=0)
                if resample_method == "simple":
                    self.resample_simple(frac_active_in_window, resample_scale)
                elif resample_method == "advanced":
                    self.resample_advanced(frac_active_in_window, resample_scale, batch_size)

            # Update learning rate
            step_lr = lr * lr_scale(step, steps)
            for group in optimizer.param_groups:
                group["lr"] = step_lr

            # Get a batch of hidden activations from the model
            with t.inference_mode():
                h = self.generate_batch(batch_size)

            # Optimize
            loss_dict, loss, acts, _ = self.forward(h)
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()

            # Normalize decoder weights by modifying them inplace (if not using tied weights)
            if not self.cfg.tied_weights:
                self.W_dec.data = self.W_dec_normalized

            # Calculate the mean sparsities over batch dim for each feature
            frac_active = (acts.abs() > 1e-8).float().mean(0)
            frac_active_list.append(frac_active)

            # Display progress bar, and append new values for plotting
            if step % log_freq == 0 or (step + 1 == steps):
                progress_bar.set_postfix(
                    lr=step_lr,
                    frac_active=frac_active.mean().item(),
                    **{k: v.mean(0).sum().item() for k, v in loss_dict.items()},  # type: ignore
                )
                data_log["W_enc"].append(self.W_enc.detach().cpu().clone())
                data_log["W_dec"].append(self.W_dec.detach().cpu().clone())
                data_log["frac_active"].append(frac_active.detach().cpu().clone())
                data_log["steps"].append(step)

        return data_log

    @t.no_grad()
    def resample_simple(
        self,
        frac_active_in_window: Float[Tensor, "window inst d_sae"],
        resample_scale: float,
    ) -> None:
        """
        Resamples dead latents, by modifying the model's weights and biases inplace.

        Resampling method is:
            - For each dead neuron, generate a random vector of size (d_in,), and normalize these vectors
            - Set new values of W_dec and W_enc to be these normalized vectors, at each dead neuron
            - Set b_enc to be zero, at each dead neuron
        """
        dead_latents_mask = (frac_active_in_window < 1e-8).all(dim=0)  # [instances d_sae]
        n_dead = int(dead_latents_mask.int().sum().item())

        # Get our random replacement values of shape [n_dead d_in], and scale them
        replacement_values = t.randn((n_dead, self.cfg.d_in), device=self.W_enc.device)
        replacement_values_normed = replacement_values / (
            replacement_values.norm(dim=-1, keepdim=True) + self.cfg.weight_normalize_eps
        )

        # Change the corresponding values in W_enc, W_dec, and b_enc
        self.W_enc.data.transpose(-1, -2)[dead_latents_mask] = resample_scale * replacement_values_normed
        self.W_dec.data[dead_latents_mask] = replacement_values_normed
        self.b_enc.data[dead_latents_mask] = 0.0

    @t.no_grad()
    def resample_advanced(
        self,
        frac_active_in_window: Float[Tensor, "window inst d_sae"],
        resample_scale: float,
        batch_size: int,
    ) -> None:
        """
        Resamples latents that have been dead for 'dead_feature_window' steps, according to `frac_active`.

        Resampling method is:
            - Compute the L2 reconstruction loss produced from the hidden state vectors `h`
            - Randomly choose values of `h` with probability proportional to their reconstruction loss
            - Set new values of W_dec and W_enc to be these (centered and normalized) vectors, at each dead neuron
            - Set b_enc to be zero, at each dead neuron
        """
        h = self.generate_batch(batch_size)
        l2_loss = self.forward(h)[0]["L_reconstruction"]

        for instance in range(self.cfg.n_inst):
            # Find the dead latents in this instance. If all latents are alive, continue
            is_dead = (frac_active_in_window[:, instance] < 1e-8).all(dim=0)
            dead_latents = t.nonzero(is_dead).squeeze(-1)
            n_dead = dead_latents.numel()
            if n_dead == 0:
                continue  # If we have no dead features, then we don't need to resample

            # Compute L2 loss for each element in the batch
            l2_loss_instance = l2_loss[:, instance]  # [batch_size]
            if l2_loss_instance.max() < 1e-6:
                continue  # If we have zero reconstruction loss, we don't need to resample

            # Draw `d_sae` samples from [0, 1, ..., batch_size-1], with probabilities proportional to l2_loss
            distn = Categorical(probs=l2_loss_instance.pow(2) / l2_loss_instance.pow(2).sum())
            replacement_indices = distn.sample((n_dead,))  # type: ignore

            # Index into the batch of hidden activations to get our replacement values
            replacement_values = (h - self.b_dec)[replacement_indices, instance]  # [n_dead d_in]
            replacement_values_normalized = replacement_values / (
                replacement_values.norm(dim=-1, keepdim=True) + self.cfg.weight_normalize_eps
            )

            # Get the norm of alive neurons (or 1.0 if there are no alive neurons)
            W_enc_norm_alive_mean = (
                self.W_enc[instance, :, ~is_dead].norm(dim=0).mean().item()
                if (~is_dead).any()
                else 1.0
            )

            # Lastly, set the new weights & biases (W_dec is normalized, W_enc needs specific scaling, b_enc is zero)
            self.W_dec.data[instance, dead_latents, :] = replacement_values_normalized
            self.W_enc.data[instance, :, dead_latents] = (
                replacement_values_normalized.T * W_enc_norm_alive_mean * resample_scale
            )
            self.b_enc.data[instance, dead_latents] = 0.0







#%%
tests.test_sae_init(SAE)
tests.test_sae_W_dec_normalized(SAE)
tests.test_sae_generate_batch(SAE)
tests.test_sae_forward(SAE)
#%%
d_hidden = d_in = 2
n_features = d_sae = 5
n_inst = 8

cfg = Config(n_inst=n_inst, n_features=n_features, d_hidden=d_hidden)

model = Model(cfg=cfg, device=device)
model.optimize(steps=10_000)

sae = SAE(cfg=SAEConfig(n_inst=n_inst, d_in=d_in, d_sae=d_sae), model=model)

h = sae.generate_batch(500)
utils.plot_features_in_2d(model.W, title="Base model")
utils.plot_features_in_2d(
    einops.rearrange(h, "batch inst d_in -> inst d_in batch"),
    title="Hidden state representation of a random batch of data",
)
# %%
data_log = sae.optimize(steps=25_000)

utils.animate_features_in_2d(
    {
        "Encoder weights": t.stack(data_log["W_enc"]),
        "Decoder weights": t.stack(data_log["W_dec"]).transpose(-1, -2),
    },
    steps=data_log["steps"],
    filename="animation-training.html",
    title="SAE on toy model",
)
# %%
utils.frac_active_line_plot(
    frac_active=t.stack(data_log["frac_active"]),
    title="Probability of sae features being active during training",
    avg_window=10,
)
# %%
tests.test_resample_simple(SAE)

# %%
sae = SAE(cfg=SAEConfig(n_inst=n_inst, d_in=d_in, d_sae=d_sae), model=model)

data_log = sae.optimize(steps=25_000, resample_method="simple")

utils.animate_features_in_2d(
    {
        "Encoder weights": t.stack(data_log["W_enc"]),
        "Decoder weights": t.stack(data_log["W_dec"]).transpose(-1, -2),
    },
    steps=data_log["steps"],
    filename="animation-resampling.html",
    title="SAE on toy model with simple resampling",
)
# %%
with t.inference_mode():
    h_r = sae(h)[-1]

utils.animate_features_in_2d(
    {
        "h": einops.rearrange(h, "batch inst d_in -> inst d_in batch"),
        "h<sub>r</sub>": einops.rearrange(h_r, "batch inst d_in -> inst d_in batch"),
    },
    filename="animation-reconstructions.html",
    title="Hidden state vs reconstructions",
)
# %%
tests.test_resample_advanced(SAE)
# %%
# Train with overcomplete basis
d_sae = 10
sae = SAE(cfg=SAEConfig(n_inst=n_inst, d_in=d_in, d_sae=d_sae), model=model)

data_log = sae.optimize(steps=25_000, resample_method="simple")

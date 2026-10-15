from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import nn
from torch.nn import functional as F


@dataclass(frozen=True)
class RSSMConfig:
    observation_dim: int
    action_dim: int
    embed_dim: int = 64
    deter_dim: int = 128
    stoch_dim: int = 32
    hidden_dim: int = 128
    classes: int = 16
    unimix: float = 0.01
    physical_context_dim: int = 0
    physical_context_embed_dim: int = 64
    building_patch_size: int = 0
    building_patch_embed_dim: int = 64


class LowDimEncoder(nn.Module):
    def __init__(self, observation_dim: int, embed_dim: int, hidden_dim: int) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(observation_dim, hidden_dim),
            nn.ELU(),
            nn.Linear(hidden_dim, embed_dim),
            nn.ELU(),
        )

    def forward(self, observations: torch.Tensor) -> torch.Tensor:
        return self.net(observations.float())


class LocalPatchEncoder(nn.Module):
    """CNN encoder for local building-height patches centered at the UAV."""

    def __init__(self, embed_dim: int) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(1, 16, kernel_size=3, padding=1),
            nn.ELU(),
            nn.Conv2d(16, 32, kernel_size=3, padding=1),
            nn.ELU(),
            nn.AdaptiveAvgPool2d((4, 4)),
            nn.Flatten(),
            nn.Linear(32 * 4 * 4, embed_dim),
            nn.ELU(),
        )

    def forward(self, patches: torch.Tensor) -> torch.Tensor:
        return self.net(patches.float())


class RSSMState(dict):
    """Dictionary state container used by the categorical RSSM."""


class CategoricalRSSM(nn.Module):
    """DreamerV3-style categorical RSSM with unimix and straight-through samples."""

    def __init__(self, config: RSSMConfig) -> None:
        super().__init__()
        self.config = config
        self.action_dim = int(config.action_dim)
        self.stoch_dim = int(config.stoch_dim)
        self.classes = int(config.classes)
        self.deter_dim = int(config.deter_dim)
        self.unimix = float(config.unimix)
        self.feat_dim = self.stoch_dim * self.classes + self.deter_dim
        self.context_dim = int(config.physical_context_embed_dim) if int(config.physical_context_dim) > 0 else 0

        if self.classes <= 1:
            raise ValueError("categorical RSSM requires RSSMConfig.classes > 1")
        if not 0.0 <= self.unimix < 1.0:
            raise ValueError("categorical RSSM requires 0 <= RSSMConfig.unimix < 1")

        self.img_in = nn.Sequential(
            nn.Linear(self.stoch_dim * self.classes + self.action_dim + self.context_dim, config.hidden_dim),
            nn.ELU(),
        )
        self.gru = nn.GRUCell(config.hidden_dim, self.deter_dim)
        self.prior_net = nn.Sequential(
            nn.Linear(self.deter_dim, config.hidden_dim),
            nn.ELU(),
            nn.Linear(config.hidden_dim, self.stoch_dim * self.classes),
        )
        self.post_net = nn.Sequential(
            nn.Linear(self.deter_dim + config.embed_dim, config.hidden_dim),
            nn.ELU(),
            nn.Linear(config.hidden_dim, self.stoch_dim * self.classes),
        )

    def initial(self, batch_size: int, device) -> RSSMState:
        return RSSMState(
            logit=torch.zeros(batch_size, self.stoch_dim, self.classes, device=device),
            stoch=torch.zeros(batch_size, self.stoch_dim, self.classes, device=device),
            deter=torch.zeros(batch_size, self.deter_dim, device=device),
        )

    def get_feat(self, state: dict[str, torch.Tensor]) -> torch.Tensor:
        stoch = state["stoch"].reshape(*state["stoch"].shape[:-2], self.stoch_dim * self.classes)
        return torch.cat([stoch, state["deter"]], dim=-1)

    def img_step(
        self,
        prev_state: dict[str, torch.Tensor],
        action: torch.Tensor,
        context: torch.Tensor | None = None,
        sample: bool = True,
        *,
        generator: torch.Generator | None = None,
    ) -> RSSMState:
        action_one_hot = F.one_hot(action.long(), num_classes=self.action_dim).float()
        stoch = prev_state["stoch"].reshape(prev_state["stoch"].shape[0], self.stoch_dim * self.classes)
        if self.context_dim > 0:
            if context is None:
                context = torch.zeros(action_one_hot.shape[0], self.context_dim, device=action_one_hot.device)
            x = torch.cat([stoch, action_one_hot, context.float()], dim=-1)
        else:
            x = torch.cat([stoch, action_one_hot], dim=-1)
        x = self.img_in(x)
        deter = self.gru(x, prev_state["deter"])
        logit = self.prior_net(deter).reshape(-1, self.stoch_dim, self.classes)
        stoch = self._sample(logit, sample=sample, generator=generator)
        return RSSMState(logit=logit, stoch=stoch, deter=deter)

    def obs_step(
        self,
        prev_state: dict[str, torch.Tensor],
        action: torch.Tensor,
        embed: torch.Tensor,
        context: torch.Tensor | None = None,
        sample: bool = True,
        *,
        generator: torch.Generator | None = None,
    ) -> tuple[RSSMState, RSSMState]:
        prior = self.img_step(
            prev_state,
            action,
            context=context,
            sample=sample,
            generator=generator,
        )
        x = torch.cat([prior["deter"], embed], dim=-1)
        logit = self.post_net(x).reshape(-1, self.stoch_dim, self.classes)
        stoch = self._sample(logit, sample=sample, generator=generator)
        post = RSSMState(logit=logit, stoch=stoch, deter=prior["deter"])
        return post, prior

    def observe(
        self,
        embeds: torch.Tensor,
        actions: torch.Tensor,
        contexts: torch.Tensor | None = None,
        state: dict[str, torch.Tensor] | None = None,
        resets: torch.Tensor | None = None,
        sample: bool = True,
        *,
        generator: torch.Generator | None = None,
    ) -> tuple[dict[str, torch.Tensor], dict[str, torch.Tensor]]:
        batch_size, horizon = actions.shape[:2]
        if state is None:
            state = self.initial(batch_size, actions.device)
        if resets is not None and resets.shape != actions.shape:
            raise ValueError("RSSM resets must have the same [batch, time] shape as actions")
        posts: dict[str, list[torch.Tensor]] = {key: [] for key in ("logit", "stoch", "deter")}
        priors: dict[str, list[torch.Tensor]] = {key: [] for key in ("logit", "stoch", "deter")}
        for t in range(horizon):
            if resets is not None:
                reset_t = resets[:, t].to(dtype=torch.bool)
                initial = self.initial(batch_size, actions.device)
                state = {
                    key: torch.where(
                        reset_t.reshape(batch_size, *([1] * (value.ndim - 1))),
                        initial[key],
                        value,
                    )
                    for key, value in state.items()
                }
            context_t = contexts[:, t] if contexts is not None else None
            post, prior = self.obs_step(
                state,
                actions[:, t],
                embeds[:, t],
                context=context_t,
                sample=sample,
                generator=generator,
            )
            for key in posts:
                posts[key].append(post[key])
                priors[key].append(prior[key])
            state = post
        return (
            {key: torch.stack(value, dim=1) for key, value in posts.items()},
            {key: torch.stack(value, dim=1) for key, value in priors.items()},
        )

    def imagine(
        self,
        actions: torch.Tensor,
        state: dict[str, torch.Tensor],
        contexts: torch.Tensor | None = None,
        sample: bool = False,
        *,
        generator: torch.Generator | None = None,
    ) -> dict[str, torch.Tensor]:
        priors: dict[str, list[torch.Tensor]] = {key: [] for key in ("logit", "stoch", "deter")}
        current = state
        for t in range(actions.shape[1]):
            context_t = contexts[:, t] if contexts is not None else None
            current = self.img_step(
                current,
                actions[:, t],
                context=context_t,
                sample=sample,
                generator=generator,
            )
            for key in priors:
                priors[key].append(current[key])
        return {key: torch.stack(value, dim=1) for key, value in priors.items()}

    def _sample(
        self,
        logit: torch.Tensor,
        sample: bool,
        *,
        generator: torch.Generator | None = None,
    ) -> torch.Tensor:
        probs = self._mixed_probs(logit)
        if sample:
            flat_probs = probs.reshape(-1, self.classes)
            index = torch.multinomial(
                flat_probs,
                num_samples=1,
                generator=generator,
            ).squeeze(-1)
            one_hot = F.one_hot(index, num_classes=self.classes).float().reshape_as(probs)
        else:
            index = probs.argmax(dim=-1)
            one_hot = F.one_hot(index, num_classes=self.classes).float()
        return one_hot.detach() + probs - probs.detach()

    def _mixed_probs(self, logit: torch.Tensor) -> torch.Tensor:
        probs = torch.softmax(logit, dim=-1)
        if self.unimix > 0.0:
            uniform = torch.full_like(probs, 1.0 / self.classes)
            probs = (1.0 - self.unimix) * probs + self.unimix * uniform
        return probs.clamp_min(1.0e-8)

    def kl(self, post: dict[str, torch.Tensor], prior: dict[str, torch.Tensor]) -> torch.Tensor:
        post_probs = self._mixed_probs(post["logit"])
        prior_probs = self._mixed_probs(prior["logit"])
        kl = post_probs * (post_probs.log() - prior_probs.log())
        return kl.sum(dim=(-1, -2))


class RSSMWorldModel(nn.Module):
    """Physics-conditioned DreamerV3-style categorical world model."""

    def __init__(self, config: RSSMConfig) -> None:
        super().__init__()
        self.config = config
        self.encoder = LowDimEncoder(config.observation_dim, config.embed_dim, config.hidden_dim)
        self.context_embed_dim = int(config.physical_context_embed_dim) if int(config.physical_context_dim) > 0 else 0
        self.physical_context_encoder = (
            nn.Sequential(
                nn.Linear(config.physical_context_dim, config.hidden_dim),
                nn.ELU(),
                nn.Linear(config.hidden_dim, self.context_embed_dim),
                nn.ELU(),
            )
            if self.context_embed_dim > 0
            else None
        )
        self.building_patch_encoder = (
            LocalPatchEncoder(config.building_patch_embed_dim) if config.building_patch_size > 0 else None
        )
        if self.physical_context_encoder is not None and self.building_patch_encoder is not None:
            self.context_fusion = nn.Sequential(
                nn.Linear(self.context_embed_dim + config.building_patch_embed_dim, self.context_embed_dim),
                nn.ELU(),
            )
        else:
            self.context_fusion = None
        self.rssm = CategoricalRSSM(config)
        feat_dim = int(self.rssm.feat_dim)
        self.feat_dim = feat_dim
        head_input_dim = feat_dim + self.context_embed_dim
        self.decoder = nn.Sequential(
            nn.Linear(head_input_dim, config.hidden_dim),
            nn.ELU(),
            nn.Linear(config.hidden_dim, config.observation_dim),
        )
        self.reward_head = nn.Sequential(
            nn.Linear(head_input_dim, config.hidden_dim),
            nn.ELU(),
            nn.Linear(config.hidden_dim, 1),
        )
        self.outage_head = nn.Sequential(
            nn.Linear(head_input_dim, config.hidden_dim),
            nn.ELU(),
            nn.Linear(config.hidden_dim, 1),
        )
        self.high_outage_head = nn.Sequential(
            nn.Linear(head_input_dim, config.hidden_dim),
            nn.ELU(),
            nn.Linear(config.hidden_dim, 1),
        )
        self.done_head = nn.Sequential(
            nn.Linear(head_input_dim, config.hidden_dim),
            nn.ELU(),
            nn.Linear(config.hidden_dim, 1),
        )

    def encode_observations(self, observations: torch.Tensor) -> torch.Tensor:
        batch_shape = observations.shape[:-1]
        flat = observations.reshape(-1, observations.shape[-1])
        embed = self.encoder(flat)
        return embed.reshape(*batch_shape, -1)

    def encode_physical_contexts(
        self,
        physical_contexts: torch.Tensor | None = None,
        building_patches: torch.Tensor | None = None,
    ) -> torch.Tensor | None:
        if self.context_embed_dim <= 0:
            return None
        if physical_contexts is None:
            raise ValueError("physics-conditioned RSSM requires physical_contexts")
        batch_shape = physical_contexts.shape[:-1]
        flat_context = physical_contexts.reshape(-1, physical_contexts.shape[-1]).float()
        assert self.physical_context_encoder is not None
        context_embed = self.physical_context_encoder(flat_context)
        if self.building_patch_encoder is not None:
            if building_patches is None:
                raise ValueError("building_patch_size > 0 requires building_patches")
            flat_patches = building_patches.reshape(
                -1,
                building_patches.shape[-3],
                building_patches.shape[-2],
                building_patches.shape[-1],
            )
            building_embed = self.building_patch_encoder(flat_patches)
            if self.context_fusion is not None:
                context_embed = self.context_fusion(torch.cat([context_embed, building_embed], dim=-1))
        return context_embed.reshape(*batch_shape, self.context_embed_dim)

    def heads(self, state: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
        feat = self.rssm.get_feat(state)
        return self.heads_from_feat(feat)

    def heads_from_feat(self, feat: torch.Tensor, context: torch.Tensor | None = None) -> dict[str, torch.Tensor]:
        if self.context_embed_dim > 0:
            if context is None:
                context = torch.zeros(*feat.shape[:-1], self.context_embed_dim, dtype=feat.dtype, device=feat.device)
            feat = torch.cat([feat, context], dim=-1)
        result = {
            "observations": self.decoder(feat),
            "rewards": self.reward_head(feat).squeeze(-1),
            "outages": torch.sigmoid(self.outage_head(feat).squeeze(-1)),
            "high_outage_logits": self.high_outage_head(feat).squeeze(-1),
            "done_logits": self.done_head(feat).squeeze(-1),
        }
        return result

    def observe_sequence(
        self,
        observations: torch.Tensor,
        actions: torch.Tensor,
        physical_contexts: torch.Tensor | None = None,
        building_patches: torch.Tensor | None = None,
        resets: torch.Tensor | None = None,
        sample: bool = True,
    ) -> tuple[
        dict[str, torch.Tensor],
        dict[str, torch.Tensor],
        dict[str, torch.Tensor],
        dict[str, torch.Tensor],
    ]:
        embeds = self.encode_observations(observations[:, 1:])
        context_embeds = self.encode_physical_contexts(physical_contexts, building_patches)
        post, prior = self.rssm.observe(
            embeds,
            actions,
            contexts=context_embeds,
            resets=resets,
            sample=sample,
        )
        post_pred = self.heads_from_feat(self.rssm.get_feat(post), context_embeds)
        prior_pred = self.heads_from_feat(self.rssm.get_feat(prior), context_embeds)
        return post, prior, post_pred, prior_pred

    @torch.no_grad()
    def posterior_from_context(
        self,
        observations: torch.Tensor,
        actions: torch.Tensor,
        physical_contexts: torch.Tensor | None = None,
        building_patches: torch.Tensor | None = None,
        sample: bool = False,
    ) -> dict[str, torch.Tensor]:
        embeds = self.encode_observations(observations[:, 1:])
        context_embeds = self.encode_physical_contexts(physical_contexts, building_patches)
        post, _ = self.rssm.observe(embeds, actions, contexts=context_embeds, sample=sample)
        return {key: value[:, -1] for key, value in post.items()}

    @torch.no_grad()
    def imagine_from_state(
        self,
        state: dict[str, torch.Tensor],
        actions: torch.Tensor,
        physical_contexts: torch.Tensor | None = None,
        building_patches: torch.Tensor | None = None,
        sample: bool = False,
    ) -> tuple[dict[str, torch.Tensor], dict[str, torch.Tensor]]:
        context_embeds = self.encode_physical_contexts(physical_contexts, building_patches)
        prior = self.rssm.imagine(actions, state, contexts=context_embeds, sample=sample)
        pred = self.heads_from_feat(self.rssm.get_feat(prior), context_embeds)
        return prior, pred


def rssm_kl(
    rssm: CategoricalRSSM,
    post: dict[str, torch.Tensor],
    prior: dict[str, torch.Tensor],
) -> torch.Tensor:
    return rssm.kl(post, prior)

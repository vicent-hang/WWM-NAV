# Method Overview

WWM-NAV is the pure navigation layer of a wireless world-model UAV framework. The policy does not receive the full outage map. It observes only the current navigation state and the outage measured at visited locations, while the environment keeps the complete radio map for simulation and evaluation.

```text
Radio map + start/goal + physical context
        |
        v
GridNavigationEnv executes one of four moves
        |
        v
Visited-position outage and reward are appended to replay
        |
        v
RSSM learns latent dynamics and prediction heads
        |
        v
Actor-Critic trains inside RSSM imagination
        |
        v
Actor-only execution with posterior RSSM feedback
```

The RSSM predicts observation, reward, outage probability, high-outage probability, and done probability. Actor-Critic training uses the learned reward head and done head to compute lambda returns over imagined rollouts. The outage head is supervised and logged for diagnostics; in the default research-one contract, risk enters the Actor objective through the environment reward learned by the reward head.

The SNARM-style prior is a distance-to-goal navigation prior. It does not contain outage information and should not be described as a radio-risk oracle.

## Main Metrics

| Metric | Meaning |
|---|---|
| success | Whether the UAV reached the goal tolerance before timeout/out-of-bounds termination. |
| episode_steps | Number of executed environment steps. |
| mean_outage | Average outage probability along executed steps. |
| max_outage | Maximum executed-step outage probability. |
| high_outage_count | Number of executed steps with outage above the configured threshold. |
| episode_return | Environment reward accumulated along the executed trajectory. |

Report task-level metrics rather than only world-model prediction losses.

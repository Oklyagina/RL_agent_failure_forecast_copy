from grid2op.Action import PlayableAction, PowerlineSetAction
from grid2op.Reward import AlarmReward
from grid2op.Rules import DefaultRules
from grid2op.Chronics import Multifolder
from grid2op.Chronics import GridStateFromFileWithForecasts
from grid2op.Backend import PandaPowerBackend
from grid2op.Opponent import GeometricOpponent, BaseActionBudget
from grid2op.operator_attention import LinearAttentionBudget

lines_attacked = ["62_58_180", "62_63_160", "48_50_136", "48_53_141", "41_48_131", "39_41_121",
                  "43_44_125", "44_45_126", "34_35_110", "54_58_154"]

opponent_attack_cooldown = 12  # 1 hour, 1 hour being 12 time steps
opponent_attack_duration = 96  # 8 hours at maximum
opponent_budget_per_ts = 0.17  # opponent_attack_duration / opponent_attack_cooldown + epsilon
opponent_init_budget = 144.  # no need to attack straightfully, it can attack starting at midday the first day

# Thermal limits (in Amps), one value per line, in `env.name_line` order.
# ai4realnet_small is a physical subgrid of ai4realnet_large: every line here also
# exists in ai4realnet_large/config.py's `th_lim`, under the same name (e.g. "58_60_156").
# These values MUST stay equal to the corresponding entries there - do not tune them
# independently, or line loading (rho) will no longer be comparable between the two
# environments.
th_lim = [
    83.0, 353.0, 541.0, 316.0, 1033.0, 379.0, 316.0, 313.0, 371.0, 301.0,
    346.0, 449.0, 571.0, 169.0, 273.0, 88.0, 113.0, 446.0, 589.0, 589.0,
    279.0, 256.0, 157.0, 195.0, 221.0, 119.0, 256.9, 326.0, 376.6, 179.5,
    927.9, 223.0, 90.0, 119.0, 75.0, 79.0, 317.9, 236.0, 249.0, 118.0,
    693.0, 671.0, 453.0, 318.5, 427.2, 689.0, 701.0, 721.0, 616.0, 616.0,
    108.7, 340.2, 223.0, 384.0, 409.0, 661.0, 689.0, 397.0, 1019.0,
]

config = {
    "backend": PandaPowerBackend,
    "action_class": PlayableAction,
    "observation_class": None,
    "reward_class": AlarmReward,
    "gamerules_class": DefaultRules,
    "chronics_class": Multifolder,
    "grid_value_class": GridStateFromFileWithForecasts,
    "volagecontroler_class": None,
    "names_chronics_to_grid": None,
    "thermal_limits": th_lim,
    "opponent_attack_cooldown": opponent_attack_cooldown,
    "opponent_attack_duration": opponent_attack_duration,
    "opponent_budget_per_ts": opponent_budget_per_ts,
    "opponent_init_budget": opponent_init_budget,
    "opponent_action_class": PowerlineSetAction,
    "opponent_class": GeometricOpponent,
    "opponent_budget_class": BaseActionBudget,
    'kwargs_opponent': {"lines_attacked": lines_attacked,
                        "attack_every_xxx_hour": 24,
                        "average_attack_duration_hour": 4,
                        "minimum_attack_duration_hour": 1},
    "has_attention_budget": True,
    "attention_budget_class": LinearAttentionBudget,
    "kwargs_attention_budget": {"max_budget": 3.,
                                "budget_per_ts": 1. / (12. * 16),
                                "alarm_cost": 1.,
                                "init_budget": 2.}
}
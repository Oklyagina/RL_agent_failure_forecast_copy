"""InteractiveAI recommendation formatting helpers.

This module vendors the Parade/ExpertAgent-style formatting used by the
platform so the API can compute ``efficiency_of_the_reco`` in-container.
"""
from __future__ import annotations

from typing import Any, Optional

import numpy as np


def _action_payload(act: Any) -> Any:
    if hasattr(act, "to_json"):
        return act.to_json()
    if hasattr(act, "as_serializable_dict"):
        return act.as_serializable_dict()
    return str(act)


def _is_do_nothing(act: Any, env: Optional[Any]) -> bool:
    if env is None or not hasattr(env, "action_space"):
        return False
    try:
        return act == env.action_space({})
    except Exception:
        return False


def get_parade_info(act: Any, obs: Any, env: Optional[Any] = None) -> dict:
    """Compile one recommendation in InteractiveAI frontend format."""
    kpis = {}
    title = []
    description = []
    impact = act.impact_on_objects()

    if getattr(act, "_modif_redispatch", False):
        kpis["type_of_the_reco"] = "Redispatch"
        title.append("Injection recommendation: production source redispatch")
        cpt = 0
        for gen_idx in range(act.n_gen):
            if act._redispatch[gen_idx] != 0.0:
                gen_name = act.name_gen[gen_idx]
                r_amount = act._redispatch[gen_idx]
                if cpt > 0:
                    description.append(", ")
                cpt = 1
                description.append(f'"{gen_name}" de {r_amount:.2f} MW')

    if getattr(act, "_modif_storage", False):
        kpis["type_of_the_reco"] = "Storage"
        title.append("Storage recommendation")
        cpt = 0
        for stor_idx in range(act.n_storage):
            amount_ = act._storage_power[stor_idx]
            if np.isfinite(amount_) and amount_ != 0.0:
                name_ = act.name_storage[stor_idx]
                if cpt > 0:
                    description.append(", ")
                cpt = 1
                description.append(
                    f'Ask unit "{name_}" to '
                    f'{"charge" if amount_ > 0.0 else "discharge"} '
                    f'{abs(amount_):.2f} MW (setpoint: {amount_:.2f} MW)'
                )

    if getattr(act, "_modif_curtailment", False):
        kpis["type_of_the_reco"] = "Injection"
        title.append("Injection recommendation")
        cpt = 0
        for gen_idx in range(act.n_gen):
            amount_ = act._curtail[gen_idx]
            if np.isfinite(amount_) and amount_ != -1.0:
                name_ = act.name_gen[gen_idx]
                if cpt > 0:
                    description.append(", ")
                cpt = 1
                description.append(
                    f'Limit unit "{name_}" to '
                    f'{100.0 * amount_:.1f}% of its maximum capacity '
                    f'(setpoint: {amount_:.3f})'
                )

    force_line_impact = impact["force_line"]
    if force_line_impact["changed"]:
        kpis["type_of_the_reco"] = "Topological"
        title.append("Topological recommendation: connection/disconnection of line")
        reconnections = force_line_impact["reconnections"]
        if reconnections["count"] > 0:
            description.append(
                f"Reconnection of {reconnections['count']} lines "
                f"({reconnections['powerlines']})"
            )

        disconnections = force_line_impact["disconnections"]
        if disconnections["count"] > 0:
            description.append(
                f"Disconnection of {disconnections['count']} lines "
                f"({disconnections['powerlines']})"
            )

    switch_line_impact = impact["switch_line"]
    if switch_line_impact["changed"]:
        kpis["type_of_the_reco"] = "Topological"
        title.append("Topological: change a line state")
        description.append(
            f"Change the state of {switch_line_impact['count']} lines "
            f"({switch_line_impact['powerlines']})"
        )

    bus_switch_impact = impact["topology"]["bus_switch"]
    if len(bus_switch_impact) > 0:
        substation = (
            bus_switch_impact.get("substation")
            if hasattr(bus_switch_impact, "get")
            else bus_switch_impact[0]["substation"]
        )
        kpis["type_of_the_reco"] = "Topological"
        title.append(
            "Topological recommendation: Schematic acquisition at substation "
            + str(substation)
        )
        description.append("Busbar change:")
        for switch in bus_switch_impact:
            description.append(
                f"\t \t - Switch bus of {switch['object_type']} id "
                f"{switch['object_id']} [at station {switch['substation']}]"
            )

    assigned_bus_impact = impact["topology"]["assigned_bus"]
    disconnect_bus_impact = impact["topology"]["disconnect_bus"]
    if len(assigned_bus_impact) > 0 or len(disconnect_bus_impact) > 0:
        substation = (
            assigned_bus_impact[0]["substation"]
            if assigned_bus_impact
            else disconnect_bus_impact[0]["substation"]
        )
        kpis["type_of_the_reco"] = "Topological"
        title.append(
            "Topological recommendation: Schematic acquisition at substation "
            + str(substation)
        )
        if assigned_bus_impact:
            description.append("")
        cpt = 0
        for assigned in assigned_bus_impact:
            if cpt > 0:
                description.append(", ")
            cpt = 1
            description.append(
                f" Assign bus {assigned['bus']} to "
                f"{assigned['object_type']} id {assigned['object_id']}"
            )
        if disconnect_bus_impact:
            description.append("")
        cpt = 0
        for disconnected in disconnect_bus_impact:
            if cpt > 0:
                description.append(", ")
            cpt = 1
            description.append(
                f"Disconnect {disconnected['object_type']} with id "
                f"{disconnected['object_id']} [at the substation level "
                f"{disconnected['substation']}]"
            )

    if not title and _is_do_nothing(act, env):
        kpis["type_of_the_reco"] = "Do nothing"
        title.append("Poursuivre")
        description.append("Continuation of the scenario without operator action")

    title_text = "".join(title)
    description_text = "".join(description)

    if title_text:
        obs_simulate, _, _, _ = obs.simulate(act, time_step=1)
        kpis["efficiency_of_the_reco"] = float(np.float32(obs_simulate.rho.max()))

    return {
        "title": title_text,
        "description": description_text,
        "use_case": "PowerGrid",
        "agent_type": 2,
        "actions": [_action_payload(act)],
        "kpis": kpis,
    }

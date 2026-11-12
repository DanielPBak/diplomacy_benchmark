"""A text map of the standard board for the system prompt, built from the engine's own map
data (the same adjacency the adjudicator uses), so it always matches the rules enforced."""
from __future__ import annotations

from functools import lru_cache

from diplomacy import Game

KIND = {"LAND": "land", "WATER": "sea", "COAST": "coast", "PORT": "coast"}


@lru_cache(maxsize=1)
def map_reference() -> str:
    m = Game().map
    names = {}
    for full, abbr in m.loc_name.items():
        names.setdefault(abbr.upper(), full.title())
    home_of = {c: p for p, homes in m.homes.items() for c in homes}
    provinces = sorted({loc.split("/")[0].upper() for loc in m.locs})
    targets = sorted({loc.upper() for loc in m.locs})

    lines = []
    for prov in provinces:
        kind = m.loc_type.get(prov) or m.loc_type.get(prov.lower())
        if kind == "SHUT":
            lines.append(f"{prov} {names.get(prov, '')} (impassable)")
            continue
        tags = [KIND.get(kind, kind.lower())]
        if prov in m.scs:
            tags.append(f"{home_of[prov]} home SC" if prov in home_of else "SC")
        parts = []
        if kind != "WATER":
            army = sorted({t.split("/")[0] for t in targets if t.split("/")[0] != prov and m.abuts("A", prov, "-", t)})
            parts.append("army: " + ", ".join(army))
        coasts = sorted(loc.upper() for loc in m.locs if loc.upper().startswith(prov + "/"))
        for loc in (coasts or ([prov] if kind in ("WATER", "COAST", "PORT") else [])):
            fleet = sorted({t for t in targets if t.split("/")[0] != prov and m.abuts("F", loc, "-", t)})
            label = "fleet" if loc == prov else f"fleet from {loc}"
            if fleet:
                parts.append(f"{label}: " + ", ".join(fleet))
        lines.append(f"{prov} {names.get(prov, '')} ({'; '.join(tags)}) - " + "; ".join(parts))

    homes = "; ".join(f"{p}: {', '.join(sorted(h))}" for p, h in sorted(m.homes.items()))
    return ("MAP (standard board, from the game engine; this is the complete map). Armies move between land and coast "
            "provinces; fleets move between sea and coast provinces along coastlines. Each line lists where a unit "
            "there can move in one step; an army can also cross sea provinces when fleets there convoy it (see "
            "ORDER FORMAT). Builds happen only in a power's own unoccupied home supply centers.\n"
            f"SUPPLY CENTERS (all {len(m.scs)}; 18 wins): {', '.join(sorted(m.scs))}\n"
            f"HOME CENTERS: {homes}\n" + "\n".join(lines))

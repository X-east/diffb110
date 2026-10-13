#!/usr/bin/env python3
"""Run a more relaxed RND-EPI ChEMBL similarity screen."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import screen_chembl_rnd_epi as base


@dataclass(frozen=True)
class RelaxedScreenGroup:
    """One relaxed candidate-pool selection rule."""

    group_id: str
    name: str
    purpose: str

    def match(self, row: dict[str, object]) -> bool:
        """Return whether one molecule matches this relaxed group."""
        ecfp4 = float(row["ecfp4_max"])
        ecfp6 = float(row["ecfp6_max"])
        fcfp4 = float(row["fcfp4_max"])
        maccs = float(row["maccs_max"])

        if self.group_id == "R01":
            return bool(row["is_seed_exact"])
        if self.group_id == "R02":
            return ecfp4 >= 0.55 or ecfp6 >= 0.50
        if self.group_id == "R03":
            return fcfp4 >= 0.55
        if self.group_id == "R04":
            return maccs >= 0.75
        if self.group_id == "R05":
            return max(ecfp4, ecfp6, fcfp4) >= 0.35
        if self.group_id == "R06":
            return max(ecfp4, ecfp6, fcfp4) >= 0.30
        if self.group_id == "R07":
            return max(ecfp4, ecfp6, fcfp4) >= 0.25
        if self.group_id == "R08":
            return fcfp4 >= 0.25 or maccs >= 0.60
        if self.group_id == "R09":
            return fcfp4 >= 0.30 and ecfp4 < 0.30
        if self.group_id == "R10":
            return max(ecfp4, ecfp6, fcfp4) >= 0.20 or maccs >= 0.55
        raise ValueError(f"Unknown group: {self.group_id}")


RELAXED_GROUPS: list[RelaxedScreenGroup] = [
    RelaxedScreenGroup("R01", "exact_seed", "ChEMBL entry exactly matches one of 53 RND-EPI seeds"),
    RelaxedScreenGroup("R02", "structural_055", "ECFP4 >= 0.55 or ECFP6 >= 0.50 to any seed"),
    RelaxedScreenGroup("R03", "functional_055", "FCFP4 >= 0.55 to any seed"),
    RelaxedScreenGroup("R04", "maccs_075", "MACCS >= 0.75 to any seed"),
    RelaxedScreenGroup("R05", "union_035", "max(ECFP4, ECFP6, FCFP4) >= 0.35 to any seed"),
    RelaxedScreenGroup("R06", "union_030", "max(ECFP4, ECFP6, FCFP4) >= 0.30 to any seed"),
    RelaxedScreenGroup("R07", "union_025", "max(ECFP4, ECFP6, FCFP4) >= 0.25 to any seed"),
    RelaxedScreenGroup("R08", "pharmacophore_or_maccs", "FCFP4 >= 0.25 or MACCS >= 0.60 to any seed"),
    RelaxedScreenGroup("R09", "relaxed_scaffold_hopping", "FCFP4 >= 0.30 and ECFP4 < 0.30 to any seed"),
    RelaxedScreenGroup("R10", "ultra_broad_prepool", "max(ECFP4, ECFP6, FCFP4) >= 0.20 or MACCS >= 0.55"),
]


def main() -> None:
    """Run the relaxed screen into a separate output directory."""
    base.OUTPUT_DIR = (
        base.PROJECT_ROOT / "analysis" / "rnd_epi_chembl_similarity_relaxed_20260618"
    )
    base.OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    base.GROUPS = RELAXED_GROUPS
    base.main()


if __name__ == "__main__":
    main()

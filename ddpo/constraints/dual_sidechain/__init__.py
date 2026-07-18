"""Dual-sidechain substitution audit utilities."""

from ddpo.constraints.dual_sidechain.audit import (
    DualSidechainAuditResult,
    SidechainBranchFacts,
    audit_dual_sidechain,
    build_dual_substitution_spec_from_phenolic_sites,
    extract_sidechain_branch,
    find_phenolic_oh_sites,
    summarize_audit_results,
)
from ddpo.constraints.dual_sidechain.mapping import (
    CoreMappingQuery,
    CoreMappingResult,
    build_core_mapping_query,
    map_core_to_target,
)
from ddpo.constraints.dual_sidechain.spec import DualSubstitutionSpec, PhenolicOHSite
from ddpo.constraints.dual_sidechain.substitution import (
    DualRSubstitutionResult,
    RGroupTemplate,
    available_r_group_templates,
    get_r_group_template,
    substitute_dual_r_groups,
)
from ddpo.constraints.dual_sidechain.copy_sidechain import (
    DEFAULT_CLASH_FRACTION,
    SidechainCopyResult,
    copy_sidechain_branch_to_anchor,
    intramolecular_clash_check,
)

__all__ = [
    "CoreMappingQuery",
    "CoreMappingResult",
    "DEFAULT_CLASH_FRACTION",
    "DualRSubstitutionResult",
    "DualSidechainAuditResult",
    "DualSubstitutionSpec",
    "PhenolicOHSite",
    "RGroupTemplate",
    "SidechainBranchFacts",
    "SidechainCopyResult",
    "audit_dual_sidechain",
    "available_r_group_templates",
    "build_core_mapping_query",
    "build_dual_substitution_spec_from_phenolic_sites",
    "copy_sidechain_branch_to_anchor",
    "extract_sidechain_branch",
    "find_phenolic_oh_sites",
    "get_r_group_template",
    "intramolecular_clash_check",
    "map_core_to_target",
    "summarize_audit_results",
    "substitute_dual_r_groups",
]

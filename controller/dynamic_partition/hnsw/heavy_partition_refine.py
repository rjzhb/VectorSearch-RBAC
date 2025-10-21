"""Targeted refinement utilities for oversized dynamic partitions.

The functions in this module inspect a single oversized partition produced by
AnonySys dynamic partitioning and recursively split it using role predicates
only. This is useful as a follow-up rebalancing step when the storage budget
prevents the main partitioner from evenly distributing the load up-front.
"""

from __future__ import annotations

import argparse
import logging
import math
import sys
from collections import defaultdict
from dataclasses import dataclass, field
from itertools import combinations
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Set, Tuple

repo_root = Path(__file__).resolve().parents[4]
repo_root_str = str(repo_root)
if repo_root_str not in sys.path:
    sys.path.append(repo_root_str)

from services.config import get_db_connection  # pylint: disable=wrong-import-position


@dataclass
class RoleDocument:
    block_id: int
    doc_id: int
    accessible_roles: Set[str]


@dataclass
class RoleTreeNode:
    depth: int
    documents: List[RoleDocument]
    split_role: Optional[str] = None
    left_child: Optional["RoleTreeNode"] = None
    right_child: Optional["RoleTreeNode"] = None
    required_roles: Set[str] = field(default_factory=set)

    def is_leaf(self) -> bool:
        return self.split_role is None


def _fetch_partition_documents(table_name: str) -> List[RoleDocument]:
    conn = get_db_connection()
    cur = conn.cursor()
    try:
        cur.execute(
            f"""
            SELECT p.block_id,
                   p.document_id,
                   array_agg(DISTINCT pa.role_id) AS roles
            FROM {table_name} AS p
            JOIN PermissionAssignment pa ON p.document_id = pa.document_id
            GROUP BY p.block_id, p.document_id
            ORDER BY p.document_id, p.block_id;
            """
        )
        documents: List[RoleDocument] = []
        for block_id, document_id, roles in cur.fetchall():
            documents.append(
                RoleDocument(
                    block_id=int(block_id),
                    doc_id=int(document_id),
                    accessible_roles={str(role) for role in roles or []},
                )
            )
    finally:
        cur.close()
        conn.close()
    return documents


def _partition_by_role(
    documents: Sequence[RoleDocument],
    role_value: str,
) -> Tuple[List[RoleDocument], List[RoleDocument]]:
    left_docs: List[RoleDocument] = []
    right_docs: List[RoleDocument] = []
    for doc in documents:
        if role_value in doc.accessible_roles:
            left_docs.append(doc)
        else:
            right_docs.append(doc)
    return left_docs, right_docs


def _find_best_role_split(
    documents: Sequence[RoleDocument],
    candidate_roles: Iterable[str],
    min_leaf_size: Optional[int],
) -> Tuple[Optional[str], Optional[List[RoleDocument]], Optional[List[RoleDocument]]]:
    best_role: Optional[str] = None
    best_left: Optional[List[RoleDocument]] = None
    best_right: Optional[List[RoleDocument]] = None
    best_cost = None

    for role_value in candidate_roles:
        left_docs, right_docs = _partition_by_role(documents, role_value)
        if not left_docs or not right_docs:
            continue
        cost = math.log(len(left_docs)) + math.log(len(right_docs))
        if best_cost is None or cost < best_cost:
            best_cost = cost
            best_role = role_value
            best_left = left_docs
            best_right = right_docs

    return best_role, best_left, best_right


def _build_role_tree(
    documents: Sequence[RoleDocument],
    available_roles: Set[str],
    min_leaf_size: Optional[int],
    depth: int = 0,
    required_roles: Optional[Set[str]] = None,
) -> RoleTreeNode:
    req_roles = set(required_roles) if required_roles else set()

    if len(documents) <= 1 or not available_roles:
        return RoleTreeNode(depth=depth, documents=list(documents), required_roles=req_roles)

    best_role, left_docs, right_docs = _find_best_role_split(documents, available_roles, min_leaf_size)
    if best_role is None or left_docs is None or right_docs is None:
        return RoleTreeNode(depth=depth, documents=list(documents), required_roles=req_roles)

    remaining_roles = set(available_roles)
    remaining_roles.discard(best_role)

    left_required = set(req_roles)
    left_required.add(best_role)

    left_child = _build_role_tree(left_docs, remaining_roles, min_leaf_size, depth + 1, left_required)
    right_child = _build_role_tree(right_docs, remaining_roles, min_leaf_size, depth + 1, req_roles)

    return RoleTreeNode(
        depth=depth,
        documents=list(documents),
        split_role=best_role,
        left_child=left_child,
        right_child=right_child,
        required_roles=req_roles,
    )


def _collect_leaves(root: RoleTreeNode) -> List[RoleTreeNode]:
    leaves: List[RoleTreeNode] = []

    stack = [root]
    while stack:
        node = stack.pop()
        if node.is_leaf():
            leaves.append(node)
        else:
            if node.right_child is not None:
                stack.append(node.right_child)
            if node.left_child is not None:
                stack.append(node.left_child)
    return leaves


def inspect_partition(table_name: str) -> None:
    """CLI helper to print the role-split tree for a materialized partition."""
    documents = _fetch_partition_documents(table_name)
    total_blocks = len(documents)
    logger = logging.getLogger(__name__)

    if total_blocks == 0:
        logger.info("Partition %s is empty; skipping", table_name)
        return

    role_values = {role for doc in documents for role in doc.accessible_roles}
    if not role_values:
        logger.info("Partition %s has no role information; skipping", table_name)
        return

    logger.info(
        "Building role-only refinement for %s (blocks=%s, unique_roles=%s)",
        table_name,
        total_blocks,
        len(role_values),
    )

    tree_root = _build_role_tree(documents, role_values, min_leaf_size=None)
    leaves = _collect_leaves(tree_root)

    logger.info("%s split into %s role-derived partitions", table_name, len(leaves))
    for idx, leaf in enumerate(sorted(leaves, key=lambda n: len(n.documents), reverse=True)):
        logger.info(
            "  leaf %s size=%s required_roles=%s",
            idx,
            len(leaf.documents),
            sorted(leaf.required_roles) or ["-"],
        )


def rebalance_heavy_partition(
    partition_assignment: Dict[int, Set[int]],
    comb_role_trackers: Dict[Tuple[int, ...], Dict[int, Set[int]]],
    document_index_to_roles: Dict[int, Set[int]],
    logger: Optional[logging.Logger] = None,
    target_partitions: Optional[Set[int]] = None,
    min_leaf_size: Optional[int] = None,
) -> Tuple[Dict[int, Set[int]], Dict[Tuple[int, ...], Dict[int, Set[int]]]]:
    """Split oversized partitions using role predicates and update trackers accordingly.

    Args:
        partition_assignment: current mapping of partition -> document indices.
        comb_role_trackers: role-combination tracker.
        document_index_to_roles: precomputed doc->roles mapping.
        logger: optional logger.
        target_partitions: subset of partition ids to refine (others are copied through).
        min_leaf_size: minimum document count required per refined leaf (None disables the bound).
    """
    log = logger or logging.getLogger(__name__)
    if log.level == logging.NOTSET:
        log.setLevel(logging.INFO)

    next_pid = max(partition_assignment.keys()) + 1 if partition_assignment else 0
    updated_assignment: Dict[int, Set[int]] = {}
    partition_leaf_info: Dict[int, List[Tuple[int, Set[int], Set[int], Dict[int, Set[int]]]]] = {}
    partition_required_docs: Dict[int, Dict[int, Set[int]]] = {}

    for pid in sorted(partition_assignment.keys()):
        doc_indices = partition_assignment[pid]
        allowed_roles_int: Set[int] = set()
        for comb, partition_roles in comb_role_trackers.items():
            if pid in partition_roles:
                allowed_roles_int.update(partition_roles[pid])

        role_required_docs: Dict[int, Set[int]] = defaultdict(set)
        for idx in doc_indices:
            for role in document_index_to_roles.get(idx, set()):
                if allowed_roles_int and role not in allowed_roles_int:
                    continue
                role_required_docs[role].add(idx)
        partition_required_docs[pid] = {role: set(doc_ids) for role, doc_ids in role_required_docs.items()}

        if target_partitions is not None and pid not in target_partitions:
            doc_set = set(doc_indices)
            updated_assignment[pid] = doc_set
            roles_present_int: Set[int] = set()
            role_doc_map: Dict[int, Set[int]] = defaultdict(set)
            for idx in doc_indices:
                roles = document_index_to_roles.get(idx, set())
                roles_present_int.update(roles)
                for role in roles:
                    if allowed_roles_int and role not in allowed_roles_int:
                        continue
                    role_doc_map[role].add(idx)
            partition_leaf_info[pid] = [
                (pid, doc_set, roles_present_int, {role: set(doc_ids) for role, doc_ids in role_doc_map.items()})
            ]
            log.debug(
                "Partition %s skipped refinement (docs=%s, roles=%s)",
                pid,
                len(doc_set),
                sorted(roles_present_int) or ["-"],
            )
            continue

        documents: List[RoleDocument] = []
        allowed_roles_str: Optional[Set[str]] = None
        if allowed_roles_int:
            allowed_roles_str = {str(role) for role in allowed_roles_int}

        doc_role_union: Set[str] = set()
        for idx in doc_indices:
            raw_roles = document_index_to_roles.get(idx, set())
            role_strings = {str(role) for role in raw_roles}
            documents.append(
                RoleDocument(
                    block_id=int(idx),
                    doc_id=int(idx),
                    accessible_roles=role_strings,
                )
            )
            doc_role_union.update(role_strings)

        if allowed_roles_str:
            available_roles = {role for role in allowed_roles_str if role in doc_role_union}
            if not available_roles:
                available_roles = set(doc_role_union)
        else:
            available_roles = set(doc_role_union)

        if len(documents) <= 1 or (min_leaf_size is not None and len(documents) < min_leaf_size) or not available_roles:
            updated_assignment[pid] = set(doc_indices)
            roles_present_int: Set[int] = set()
            role_doc_map: Dict[int, Set[int]] = defaultdict(set)
            for idx in doc_indices:
                roles = document_index_to_roles.get(idx, set())
                roles_present_int.update(roles)
                for role in roles:
                    if allowed_roles_int and role not in allowed_roles_int:
                        continue
                    role_doc_map[role].add(idx)
            if allowed_roles_int:
                roles_present_int &= allowed_roles_int
            partition_leaf_info[pid] = [
                (
                    pid,
                    set(doc_indices),
                    roles_present_int,
                    {role: set(doc_ids) for role, doc_ids in role_doc_map.items()},
                )
            ]
            log.debug(
                "Partition %s retained (docs=%s, roles=%s)",
                pid,
                len(doc_indices),
                sorted(roles_present_int) or ["-"],
            )
            continue

        tree_root = _build_role_tree(documents, set(available_roles), min_leaf_size)
        leaves = _collect_leaves(tree_root)

        leaf_infos: List[Tuple[int, Set[int], Set[int], Dict[int, Set[int]]]] = []
        for leaf in leaves:
            doc_set = {int(doc.block_id) for doc in leaf.documents}
            roles_present: Set[int] = set()
            role_doc_map: Dict[int, Set[int]] = defaultdict(set)
            for doc_idx in doc_set:
                roles_here = document_index_to_roles.get(doc_idx, set())
                roles_present.update(roles_here)
                for role in roles_here:
                    if allowed_roles_int and role not in allowed_roles_int:
                        continue
                    role_doc_map[role].add(doc_idx)
            if allowed_roles_int:
                roles_present &= allowed_roles_int
            leaf_infos.append(
                (
                    -1,
                    doc_set,
                    roles_present,
                    {role: set(doc_ids) for role, doc_ids in role_doc_map.items()},
                )
            )

        if len(leaf_infos) == 1:
            updated_assignment[pid] = leaf_infos[0][1]
            partition_leaf_info[pid] = [
                (pid, leaf_infos[0][1], leaf_infos[0][2], leaf_infos[0][3])
            ]
            log.debug(
                "Partition %s retained single leaf (docs=%s, roles=%s)",
                pid,
                len(leaf_infos[0][1]),
                sorted(leaf_infos[0][2]) or ["-"],
            )
            continue

        log.info("Refined partition %s into %s role subsets", pid, len(leaf_infos))

        updated_assignment[pid] = leaf_infos[0][1]
        assigned_infos: List[Tuple[int, Set[int], Set[int], Dict[int, Set[int]]]] = [
            (pid, leaf_infos[0][1], leaf_infos[0][2], leaf_infos[0][3])
        ]
        log.debug(
            "  leaf pid=%s docs=%s roles=%s",
            pid,
            len(leaf_infos[0][1]),
            sorted(leaf_infos[0][2]) or ["-"],
        )

        for doc_set, roles_present, role_doc_map in [info[1:] for info in leaf_infos[1:]]:
            new_pid = next_pid
            next_pid += 1
            updated_assignment[new_pid] = doc_set
            assigned_infos.append((new_pid, doc_set, roles_present, role_doc_map))
            log.info(
                "  leaf pid=%s docs=%s roles=%s",
                new_pid,
                len(doc_set),
                sorted(roles_present) or ["-"],
            )

        partition_leaf_info[pid] = assigned_infos

    refined_comb_trackers: Dict[Tuple[int, ...], Dict[int, Set[int]]] = defaultdict(lambda: defaultdict(set))
    for comb, partition_roles in comb_role_trackers.items():
        for pid, roles in partition_roles.items():
            if pid not in partition_leaf_info:
                refined_comb_trackers[comb][pid].update(roles)
                continue
            leaf_info = partition_leaf_info[pid]
            sorted_leaf_info = sorted(
                leaf_info,
                key=lambda item: (len(item[1]), len(item[2])),
            )
            required_docs = partition_required_docs.get(pid, {})
            candidate_indices = list(range(len(sorted_leaf_info)))
            best_subset: Optional[List[int]] = None
            best_cost: Optional[Tuple[int, int]] = None

            for r in range(1, len(candidate_indices) + 1):
                for subset in combinations(candidate_indices, r):
                    covers_all = True
                    for role in roles:
                        needed = required_docs.get(role, set())
                        covered = set()
                        for idx in subset:
                            role_doc_map = sorted_leaf_info[idx][3]
                            covered.update(role_doc_map.get(role, set()))
                        if needed and not needed.issubset(covered):
                            covers_all = False
                            break
                    if not covers_all:
                        continue
                    total_docs = sum(len(sorted_leaf_info[idx][1]) for idx in subset)
                    cost = (len(subset), total_docs)
                    if best_cost is None or cost < best_cost:
                        best_cost = cost
                        best_subset = list(subset)
                if best_subset is not None:
                    break

            if best_subset is None:
                best_subset = [0] if sorted_leaf_info else []

            for idx in best_subset:
                new_pid, _doc_set, _roles_present, role_doc_map = sorted_leaf_info[idx]
                for role in roles:
                    if role_doc_map.get(role):
                        refined_comb_trackers[comb][new_pid].add(role)

            for role in roles:
                role_assigned = any(
                    role in refined_comb_trackers[comb][sorted_leaf_info[idx][0]]
                    for idx in best_subset
                )
                if not role_assigned and sorted_leaf_info:
                    fallback_pid = sorted_leaf_info[best_subset[0]][0]
                    refined_comb_trackers[comb][fallback_pid].add(role)

    used_partitions: Set[int] = set()
    for partition_roles in refined_comb_trackers.values():
        for pid, roles in list(partition_roles.items()):
            if not roles:
                partition_roles.pop(pid)
                continue
            used_partitions.add(pid)

    filtered_assignment = {
        pid: docs for pid, docs in updated_assignment.items() if pid in used_partitions or target_partitions is None
    }

    return filtered_assignment, refined_comb_trackers


def remap_comb_role_trackers(
    comb_role_trackers: Dict[Tuple[int, ...], Dict[int, Set[int]]],
    partition_mapping: Dict[int, int],
) -> Dict[Tuple[int, ...], Dict[int, Set[int]]]:
    remapped: Dict[Tuple[int, ...], Dict[int, Set[int]]] = defaultdict(lambda: defaultdict(set))
    for comb, partitions in comb_role_trackers.items():
        for pid, roles in partitions.items():
            new_pid = partition_mapping.get(pid, pid)
            remapped[comb][new_pid].update(roles)
    return remapped


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Recursively split AnonySys partitions using role predicates only and report the structure.",
    )
    parser.add_argument(
        "--partition-prefix",
        type=str,
        default="documentblocks_partition",
        help="Prefix of AnonySys materialized partition tables.",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Only process the first N partitions (after sorting by name).",
    )
    parser.add_argument(
        "--log-level",
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"],
        help="Logging verbosity level.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    logging.basicConfig(
        level=getattr(logging, args.log_level.upper(), logging.INFO),
        format="%(asctime)s [%(levelname)s] %(message)s",
    )

    conn = get_db_connection()
    cur = conn.cursor()
    try:
        cur.execute(
            """
            SELECT tablename
            FROM pg_tables
            WHERE tablename LIKE %s
            ORDER BY tablename;
            """,
            (f"{args.partition_prefix}_%",),
        )
        tables = [row[0] for row in cur.fetchall()]
    finally:
        cur.close()
        conn.close()

    if args.limit is not None:
        tables = tables[: args.limit]

    logging.info("Discovered %s partition tables with prefix '%s'", len(tables), args.partition_prefix)

    for table_name in tables:
        inspect_partition(table_name)


if __name__ == "__main__":
    main()

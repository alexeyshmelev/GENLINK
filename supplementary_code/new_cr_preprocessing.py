# Self-contained cell (FULLY UPDATED):
# Builds & saves:
#  (0) NON-SHARED NODES REPORT: nodes NOT shared across all 3 files
#      columns: node_id, <graph_filename>, <pred_filename>, <tree_filename> with "yes"/"no"
#  (A) SEMI-UNLABELED: edges touching at least one main node (NO unlabeled–unlabeled edges)
#      columns: node_id1,node_id2,label_id1,label_id2,label1_*,label2_*,ibd_sum,ibd_n
#  (B) LABELED-ONLY: only main nodes
#      columns: node_id1,node_id2,label_id1,label_id2,ibd_sum,ibd_n
#  (C) ALL-NODES (NEW): keeps ALL vertices after PRIMARY filtration (shared-nodes + CR>80),
#      i.e. includes ALL unlabeled vertices (all not-main vertices) and allows unlabeled–unlabeled edges
#      columns: node_id1,node_id2,label_id1,label_id2,label1_*,label2_*,ibd_sum,ibd_n
#
# Filtration rules (applied to A, B, C):
#  - Primary filtration: keep only nodes shared across graph/pred/tree
#  - CR > 80
#  - Sibling filtering (ibd_sum > 300):
#      * if exactly one endpoint is unlabeled -> drop ONLY unlabeled node
#      * else -> drop BOTH nodes
#    then assert no remaining ibd_sum > 300 in each dataset
#
# Logs class balance for A, B, C using EXACT label_id values stored in each dataset (node counted once).

import os
import numpy as np
import pandas as pd

# --------------------
# Paths
# --------------------
GRAPH_PATH = "/disk/10tb/home/shmelev/New_CR_2025/df_anonymized.txt"
PRED_PATH  = "/disk/10tb/home/shmelev/New_CR_2025/samples_2347658_pivot_ancestry_anonymized.csv"
TREE_PATH  = "/disk/10tb/home/shmelev/New_CR_2025/trees_no_dups.csv"

OUT_SEMI_PATH      = "/disk/10tb/home/shmelev/New_CR_2025/CR_gt80_semi_unlabeled_all_masks_2_classes.csv"
OUT_LABELED_PATH   = "/disk/10tb/home/shmelev/New_CR_2025/CR_gt80_labeled_only_direct_labels_2_classes.csv"
OUT_ALLNODES_PATH  = "/disk/10tb/home/shmelev/New_CR_2025/CR_gt80_all_nodes_including_unlabeled_2_classes.csv"
OUT_NONSHARED_PATH = "/disk/10tb/home/shmelev/New_CR_2025/non_shared_nodes_report.csv"

# --------------------
# Constants
# --------------------
TARGET_GROUPS = ["Northen Russians", "Southern Russians"] # ["Northen Russians", "Southern Russians", "Belarusians", "Ukranians"]
LABEL1_COLS = [f"label1_{g}" for g in TARGET_GROUPS]
LABEL2_COLS = [f"label2_{g}" for g in TARGET_GROUPS]
UNLABELED_NAME = "Unlabeled"
SIB_THR = 300.0

GRAPH_FN = os.path.basename(GRAPH_PATH)
PRED_FN  = os.path.basename(PRED_PATH)
TREE_FN  = os.path.basename(TREE_PATH)

# --------------------
# Helpers
# --------------------
def node_class_balance_from_label_cols(df_edges: pd.DataFrame, col1="label_id1", col2="label_id2") -> pd.Series:
    """Counts each node once using the EXACT label_id values stored in the dataset."""
    if df_edges.empty:
        return pd.Series(dtype="int64")
    node_class = pd.concat([
        df_edges[["node_id1", col1]].rename(columns={"node_id1": "node_id", col1: "class"}),
        df_edges[["node_id2", col2]].rename(columns={"node_id2": "node_id", col2: "class"}),
    ], ignore_index=True).drop_duplicates(subset="node_id")
    return node_class["class"].value_counts()

def sibling_filter_drop_unlabeled_only_when_mixed(
    df: pd.DataFrame,
    thr: float = 300.0,
    unlabeled_value: str = "Unlabeled",
    label_col1: str = "label_id1",
    label_col2: str = "label_id2",
) -> pd.DataFrame:
    """
    For sib edges (ibd_sum > thr):
      - if exactly one endpoint is unlabeled -> remove ONLY that unlabeled node
      - else -> remove BOTH nodes
    Then drop all edges incident to removed nodes.
    Assert no ibd_sum > thr remains.
    """
    if df.empty:
        return df

    sib = df["ibd_sum"] > thr
    if not sib.any():
        return df

    sib_df = df.loc[sib, ["node_id1", "node_id2", label_col1, label_col2]]

    is_u1 = sib_df[label_col1].eq(unlabeled_value)
    is_u2 = sib_df[label_col2].eq(unlabeled_value)
    mixed = is_u1 ^ is_u2  # exactly one unlabeled

    unlabeled_to_remove = pd.concat([
        sib_df.loc[mixed & is_u1, "node_id1"],
        sib_df.loc[mixed & is_u2, "node_id2"],
    ], ignore_index=True)

    both_to_remove = pd.concat([
        sib_df.loc[~mixed, "node_id1"],
        sib_df.loc[~mixed, "node_id2"],
    ], ignore_index=True)

    nodes_to_remove = pd.Index(pd.unique(pd.concat([unlabeled_to_remove, both_to_remove], ignore_index=True)))

    df = df[~df["node_id1"].isin(nodes_to_remove) & ~df["node_id2"].isin(nodes_to_remove)].copy()

    assert not (df["ibd_sum"] > thr).any(), f"Sibling edges still present: {(df['ibd_sum'] > thr).sum()} rows > {thr}"
    return df

def add_masks_and_labels(
    df_edges: pd.DataFrame,
    tree_cr: pd.DataFrame,
    main_nodes: pd.Index,
    main_majority_group: pd.Series,
    unlabeled_name: str = "Unlabeled",
) -> pd.DataFrame:
    """
    Adds label1_*/label2_* masks + label_id1/label_id2 (strings).
    main nodes get one of the 4 TARGET_GROUPS; others -> Unlabeled.
    """
    if df_edges.empty:
        # ensure schema exists even if empty
        out = df_edges.copy()
        out["label_id1"] = pd.Series(dtype="string")
        out["label_id2"] = pd.Series(dtype="string")
        for c in LABEL1_COLS + LABEL2_COLS:
            out[c] = pd.Series(dtype="int32")
        return out

    included_nodes = pd.Index(
        pd.concat([df_edges["node_id1"], df_edges["node_id2"]], ignore_index=True).dropna().unique(),
        dtype="string",
    )

    # counts table for masks
    tree_for_counts = tree_cr[tree_cr["group"].isin(TARGET_GROUPS)].copy()
    counts_tbl = (
        tree_for_counts
        .groupby(["node_tubeid", "group"])
        .size()
        .unstack(fill_value=0)
        .reindex(included_nodes, fill_value=0)
        .reindex(columns=TARGET_GROUPS, fill_value=0)
    )

    u = df_edges["node_id1"].astype("string")
    v = df_edges["node_id2"].astype("string")

    u_counts = counts_tbl.loc[u].to_numpy(dtype=np.int32)
    v_counts = counts_tbl.loc[v].to_numpy(dtype=np.int32)

    out = df_edges.copy()

    for j, g in enumerate(TARGET_GROUPS):
        out[f"label1_{g}"] = u_counts[:, j]
        out[f"label2_{g}"] = v_counts[:, j]

    # label_id*: exact values stored in dataset
    node_to_label = pd.Series(unlabeled_name, index=included_nodes, dtype="string")
    node_to_label.loc[main_nodes] = main_majority_group.reindex(main_nodes).astype("string")

    out["label_id1"] = node_to_label.reindex(u).to_numpy()
    out["label_id2"] = node_to_label.reindex(v).to_numpy()

    out = out[["node_id1", "node_id2", "label_id1", "label_id2", *LABEL1_COLS, *LABEL2_COLS, "ibd_sum", "ibd_n"]].copy()
    return out

# --------------------
# Load
# --------------------
print("\n[Stage A] Loading files...")
pd_graph = pd.read_csv(GRAPH_PATH, dtype={"node_id1": "string", "node_id2": "string"})
predicted_ancestry = pd.read_csv(PRED_PATH, dtype={"node_tubeid": "string"})
ancestry_information = pd.read_csv(TREE_PATH, dtype={"node_tubeid": "string"})

required_graph_cols = {"node_id1", "node_id2", "ibd_sum", "ibd_n"}
required_pred_cols  = {"node_tubeid", "Central-Russia"}
required_tree_cols  = {"node_tubeid", "group"}
missing = (required_graph_cols - set(pd_graph.columns)) | (required_pred_cols - set(predicted_ancestry.columns)) | (required_tree_cols - set(ancestry_information.columns))
if missing:
    raise ValueError(f"Missing required columns: {sorted(missing)}")

print("pd_graph:", pd_graph.shape, "| predicted_ancestry:", predicted_ancestry.shape, "| ancestry_information:", ancestry_information.shape)

# --------------------
# Stage 0.1) Non-shared nodes report (nodes NOT shared across all three files)
# --------------------
print("\n[Stage 0.1] Build non-shared nodes report...")

graph_nodes = pd.Index(
    pd.concat([pd_graph["node_id1"], pd_graph["node_id2"]], ignore_index=True).dropna().unique(),
    dtype="string",
)
pred_nodes  = pd.Index(predicted_ancestry["node_tubeid"].dropna().unique(), dtype="string")
tree_nodes  = pd.Index(ancestry_information["node_tubeid"].dropna().unique(), dtype="string")

common_nodes = graph_nodes.intersection(pred_nodes).intersection(tree_nodes)
all_nodes_union = graph_nodes.union(pred_nodes).union(tree_nodes)

nonshared_nodes = all_nodes_union.difference(common_nodes)
print("nonshared_nodes:", len(nonshared_nodes))

nonshared_df = pd.DataFrame({
    "node_id": nonshared_nodes.astype("string"),
    GRAPH_FN: np.where(nonshared_nodes.isin(graph_nodes), "yes", "no"),
    PRED_FN:  np.where(nonshared_nodes.isin(pred_nodes),  "yes", "no"),
    TREE_FN:  np.where(nonshared_nodes.isin(tree_nodes),  "yes", "no"),
})
nonshared_df.to_csv(OUT_NONSHARED_PATH, index=False)
print("Saved non-shared report:", OUT_NONSHARED_PATH)


# --------------------
# Stage 0) Primary filtration: keep only nodes common to ALL three sources
# --------------------
print("\n[Stage 0] Primary filtration: keep only shared nodes in graph/pred/tree...")

pd_graph_common = pd_graph[pd_graph["node_id1"].isin(common_nodes) & pd_graph["node_id2"].isin(common_nodes)].copy()
pred_common = predicted_ancestry[predicted_ancestry["node_tubeid"].isin(common_nodes)].copy()
tree_common = ancestry_information[ancestry_information["node_tubeid"].isin(common_nodes)].copy()
print("pd_graph_common:", pd_graph_common.shape, "| pred_common:", pred_common.shape, "| tree_common:", tree_common.shape)

# --------------------
# Stage 1) CR > 80
# --------------------
print("\n[Stage 1] Filter Central-Russia > 80...")
cr_nodes = pd.Index(pred_common.loc[pred_common["Central-Russia"] > 80.0, "node_tubeid"].unique(), dtype="string")
print("cr_nodes:", len(cr_nodes))

pd_graph_cr = pd_graph_common[pd_graph_common["node_id1"].isin(cr_nodes) & pd_graph_common["node_id2"].isin(cr_nodes)].copy()
tree_cr = tree_common[tree_common["node_tubeid"].isin(cr_nodes)].copy()
print("pd_graph_cr:", pd_graph_cr.shape, "| tree_cr:", tree_cr.shape)

tree_cr["group"] = tree_cr["group"].astype("string")

# --------------------
# Stage 2) Main nodes + majority class
# --------------------
print("\n[Stage 2] Select main nodes + majority class (fast groupby)...")

cnt = (
    tree_cr
    .groupby(["node_tubeid", "group"])
    .size()
    .unstack(fill_value=0)
)

total_anc = cnt.sum(axis=1)
distinct_groups = (cnt > 0).sum(axis=1)
maj_group = cnt.idxmax(axis=1)
maj_cnt = cnt.max(axis=1)

main_mask = (total_anc == 4) & (distinct_groups <= 2) & (maj_cnt >= 3) & (maj_group != "Unknown")
main_majority_group = maj_group[main_mask].astype("string")

# keep ONLY nodes whose majority is one of the 4 target classes
main_majority_group = main_majority_group[main_majority_group.isin(TARGET_GROUPS)]
main_nodes = pd.Index(main_majority_group.index, dtype="string")

print("Main nodes:", len(main_nodes))

# --------------------
# Stage 3A) SEMI dataset edges (touches main only)
# --------------------
print("\n[Stage 3A] Build SEMI dataset edge set (touches main only; no unlabeled-unlabeled edges)...")

pd_graph_semi_base = pd_graph_cr[
    pd_graph_cr["node_id1"].isin(main_nodes) | pd_graph_cr["node_id2"].isin(main_nodes)
].copy()

# Safety check: every edge must touch a main node
touches_main_mask = pd_graph_semi_base["node_id1"].isin(main_nodes) | pd_graph_semi_base["node_id2"].isin(main_nodes)
if not bool(touches_main_mask.all()):
    bad = pd_graph_semi_base.loc[~touches_main_mask, ["node_id1", "node_id2"]].head(10)
    raise RuntimeError(f"Found edges not touching main nodes (showing up to 10):\n{bad}")

print("pd_graph_semi_base:", pd_graph_semi_base.shape)

# --------------------
# Stage 3B) LABELED-ONLY dataset edges (both endpoints main)
# --------------------
print("\n[Stage 3B] Build LABELED-ONLY dataset edge set (both endpoints main)...")

pd_graph_labeled = pd_graph_cr[
    pd_graph_cr["node_id1"].isin(main_nodes) & pd_graph_cr["node_id2"].isin(main_nodes)
].copy()

pd_graph_labeled["label_id1"] = main_majority_group.reindex(pd_graph_labeled["node_id1"].astype("string")).to_numpy()
pd_graph_labeled["label_id2"] = main_majority_group.reindex(pd_graph_labeled["node_id2"].astype("string")).to_numpy()
pd_graph_labeled = pd_graph_labeled[["node_id1", "node_id2", "label_id1", "label_id2", "ibd_sum", "ibd_n"]].copy()

print("pd_graph_labeled:", pd_graph_labeled.shape)

# --------------------
# Stage 3C) ALL-NODES dataset edges (NEW): keep ALL nodes/edges after primary filtration + CR>80
# --------------------
print("\n[Stage 3C] Build ALL-NODES dataset edge set (includes all unlabeled vertices; allows unlabeled-unlabeled edges)...")

pd_graph_all_base = pd_graph_cr.copy()
print("pd_graph_all_base:", pd_graph_all_base.shape)

# --------------------
# Stage 4) Add masks + label_id* to SEMI and ALL-NODES datasets
# --------------------
print("\n[Stage 4] Add masks + label_id* to SEMI and ALL-NODES datasets...")

pd_graph_semi = add_masks_and_labels(pd_graph_semi_base, tree_cr, main_nodes, main_majority_group, unlabeled_name=UNLABELED_NAME)
pd_graph_all  = add_masks_and_labels(pd_graph_all_base,  tree_cr, main_nodes, main_majority_group, unlabeled_name=UNLABELED_NAME)

print("pd_graph_semi:", pd_graph_semi.shape)
print("pd_graph_all :", pd_graph_all.shape)

# --------------------
# Stage 5) Sibling filtering for A, B, C + assert
# --------------------
print("\n[Stage 5] Sibling filtering (ibd_sum > 300) for SEMI, LABELED, ALL-NODES + assert...")

pd_graph_semi = sibling_filter_drop_unlabeled_only_when_mixed(
    pd_graph_semi, thr=SIB_THR, unlabeled_value=UNLABELED_NAME, label_col1="label_id1", label_col2="label_id2"
)
pd_graph_labeled = sibling_filter_drop_unlabeled_only_when_mixed(
    pd_graph_labeled, thr=SIB_THR, unlabeled_value=UNLABELED_NAME, label_col1="label_id1", label_col2="label_id2"
)
pd_graph_all = sibling_filter_drop_unlabeled_only_when_mixed(
    pd_graph_all, thr=SIB_THR, unlabeled_value=UNLABELED_NAME, label_col1="label_id1", label_col2="label_id2"
)

print("After sibling filter | semi:", pd_graph_semi.shape, "| labeled:", pd_graph_labeled.shape, "| all:", pd_graph_all.shape)

# --------------------
# Stage 6) Class balance for A, B, C (exact label_id values; node counted once)
# --------------------
print("\n[Stage 6] Class balance (exact label_id values; each node counted once)")

semi_balance   = node_class_balance_from_label_cols(pd_graph_semi,   "label_id1", "label_id2")
labeled_balance= node_class_balance_from_label_cols(pd_graph_labeled,"label_id1", "label_id2")
all_balance    = node_class_balance_from_label_cols(pd_graph_all,    "label_id1", "label_id2")

print("\n[Semi-unlabeled] class balance (includes 'Unlabeled'):")
print(semi_balance)

print("\n[Labeled-only] class balance:")
print(labeled_balance)

print("\n[All-nodes] class balance (includes 'Unlabeled'):")
print(all_balance)

# --------------------
# Stage 7) Save all outputs
# --------------------
print("\n[Stage 7] Saving datasets...")
pd_graph_semi.to_csv(OUT_SEMI_PATH, index=False)
pd_graph_labeled.to_csv(OUT_LABELED_PATH, index=False)
pd_graph_all.to_csv(OUT_ALLNODES_PATH, index=False)

print("Saved non-shared report:", OUT_NONSHARED_PATH, "| rows:", len(nonshared_df))
print("Saved semi:", OUT_SEMI_PATH, "| edges:", len(pd_graph_semi))
print("Saved labeled:", OUT_LABELED_PATH, "| edges:", len(pd_graph_labeled))
print("Saved all-nodes:", OUT_ALLNODES_PATH, "| edges:", len(pd_graph_all))

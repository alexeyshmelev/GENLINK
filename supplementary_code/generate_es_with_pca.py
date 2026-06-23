# Self-contained dataset builder for latest GENLINK format with PCA.
#
# Output without PCA:
# node_id1,node_id2,label_id1,label_id2,ibd_sum,ibd_n
#
# Output with PCA:
# node_id1,node_id2,label_id1,label_id2,ibd_sum,ibd_n,
# node1_PC1,...,node1_PC20,node2_PC1,...,node2_PC20
#
# Logic:
# - primary filtration: keep only nodes shared across graph / prediction / tree files
# - keep only nodes with Central-Russia > 80
# - keep all CR-CR edges, including masked-masked edges
# - label only nodes that pass the 3-ancestor rule
# - all other included CR > 80 nodes become "masked"
# - sibling filtering:
#     if ibd_sum > 300 and exactly one endpoint is masked -> remove only masked node
#     otherwise remove both endpoints
# - optional PCA:
#     if a node has PCA coordinates, write them
#     if a node has no PCA coordinates, fill PC1...PC20 with zeros
# - final global edge-weight threshold:
#     if MIN_EDGE_IBD_SUM_THRESHOLD is not None, drop edges with ibd_sum < threshold
# - final stats are printed at the end, including min/max edge weight

import os
import numpy as np
import pandas as pd
from tqdm.auto import tqdm

# --------------------
# Paths
# --------------------
GRAPH_PATH = "/disk/10tb/home/shmelev/New_CR_2025/df_anonymized.txt"
PRED_PATH  = "/disk/10tb/home/shmelev/New_CR_2025/samples_2347658_pivot_ancestry_anonymized.csv"
TREE_PATH  = "/disk/10tb/home/shmelev/New_CR_2025/trees_no_dups.csv"

# Optional PCA file.
# Set to None if you do not want PCA columns.
PCA_PATH = "/disk/10tb/home/shmelev/New_CR_2025/anonymized_id_PC1-PC20_chiptype.tsv"
# Example:
# PCA_PATH = "/disk/10tb/home/shmelev/New_CR_2025/pca_coordinates.csv"

OUT_PATH = "/disk/10tb/home/shmelev/New_CR_2025/CR_gt80_all_unlabeled_genlink_with_pca.csv"
OUT_NONSHARED_PATH = "/disk/10tb/home/shmelev/New_CR_2025/non_shared_nodes_report.csv"

# --------------------
# Settings
# --------------------
CR_THRESHOLD = 80.0
SIBLING_IBD_SUM_THRESHOLD = 300.0

# Global final edge filter.
# If None, no final minimum edge-weight filter is applied.
# If set to a number, edges with ibd_sum < MIN_EDGE_IBD_SUM_THRESHOLD are dropped.
MIN_EDGE_IBD_SUM_THRESHOLD = 12.0
# Example:
# MIN_EDGE_IBD_SUM_THRESHOLD = 8.0

TARGET_GROUPS = [
    "Northen Russians",
    "Southern Russians",
    "Belarusians",
    "Ukranians",
]

MASK_LABEL = "masked"

PCA_ID_COL = "anonymized_id"
PCA_DIM = 20
PCA_COLS = [f"PC{i}" for i in range(1, PCA_DIM + 1)]


def normalize_string_id(series):
    return series.astype("string").str.strip()


def get_unique_edge_nodes(df_edges):
    if df_edges.empty:
        return pd.Index([], dtype="string")

    return pd.Index(
        pd.concat(
            [df_edges["node_id1"], df_edges["node_id2"]],
            ignore_index=True,
        ).dropna().unique(),
        dtype="string",
    )


def get_edge_weight_min_max(df_edges):
    if df_edges.empty:
        return np.nan, np.nan

    return float(df_edges["ibd_sum"].min()), float(df_edges["ibd_sum"].max())


def print_stats_table(title, df):
    print(f"\n{title}")
    if df.empty:
        print("Empty")
    else:
        print(df.to_string(index=False))


def sibling_filter_drop_masked_only_when_mixed(
    df_edges,
    threshold=300.0,
    mask_label="masked",
):
    """
    For edges with ibd_sum > threshold:
      - if exactly one endpoint is masked -> remove only the masked endpoint
      - otherwise -> remove both endpoints

    After choosing nodes to remove, drop all edges incident to those nodes.
    """
    if df_edges.empty:
        return df_edges.copy(), pd.Index([], dtype="string")

    sibling_mask = df_edges["ibd_sum"] > threshold

    if not sibling_mask.any():
        return df_edges.copy(), pd.Index([], dtype="string")

    sibling_edges = df_edges.loc[
        sibling_mask,
        ["node_id1", "node_id2", "label_id1", "label_id2", "ibd_sum"],
    ].copy()

    node1_is_masked = sibling_edges["label_id1"].eq(mask_label)
    node2_is_masked = sibling_edges["label_id2"].eq(mask_label)

    mixed_labeled_masked = node1_is_masked ^ node2_is_masked

    masked_nodes_to_remove = pd.concat(
        [
            sibling_edges.loc[mixed_labeled_masked & node1_is_masked, "node_id1"],
            sibling_edges.loc[mixed_labeled_masked & node2_is_masked, "node_id2"],
        ],
        ignore_index=True,
    )

    both_endpoint_nodes_to_remove = pd.concat(
        [
            sibling_edges.loc[~mixed_labeled_masked, "node_id1"],
            sibling_edges.loc[~mixed_labeled_masked, "node_id2"],
        ],
        ignore_index=True,
    )

    nodes_to_remove = pd.Index(
        pd.unique(
            pd.concat(
                [masked_nodes_to_remove, both_endpoint_nodes_to_remove],
                ignore_index=True,
            )
        ),
        dtype="string",
    )

    filtered_edges = df_edges[
        ~df_edges["node_id1"].isin(nodes_to_remove)
        & ~df_edges["node_id2"].isin(nodes_to_remove)
    ].copy()

    remaining_sibling_edges = int((filtered_edges["ibd_sum"] > threshold).sum())
    if remaining_sibling_edges != 0:
        raise RuntimeError(
            f"Sibling filtering failed: {remaining_sibling_edges} edges still have "
            f"ibd_sum > {threshold}"
        )

    return filtered_edges, nodes_to_remove


# --------------------
# Stage A: Load
# --------------------
print("\n[Stage A] Loading files...")

pd_graph = pd.read_csv(
    GRAPH_PATH,
    dtype={"node_id1": "string", "node_id2": "string"},
)

predicted_ancestry = pd.read_csv(
    PRED_PATH,
    dtype={"node_tubeid": "string"},
)

ancestry_information = pd.read_csv(
    TREE_PATH,
    dtype={"node_tubeid": "string"},
)

pd_graph["node_id1"] = normalize_string_id(pd_graph["node_id1"])
pd_graph["node_id2"] = normalize_string_id(pd_graph["node_id2"])
predicted_ancestry["node_tubeid"] = normalize_string_id(predicted_ancestry["node_tubeid"])
ancestry_information["node_tubeid"] = normalize_string_id(ancestry_information["node_tubeid"])

if "group" in ancestry_information.columns:
    ancestry_information["group"] = ancestry_information["group"].astype("string").str.strip()

required_graph_cols = {"node_id1", "node_id2", "ibd_sum", "ibd_n"}
required_pred_cols = {"node_tubeid", "Central-Russia"}
required_tree_cols = {"node_tubeid", "group"}

missing = (
    (required_graph_cols - set(pd_graph.columns))
    | (required_pred_cols - set(predicted_ancestry.columns))
    | (required_tree_cols - set(ancestry_information.columns))
)

if missing:
    raise ValueError(f"Missing required columns: {sorted(missing)}")

pd_graph["ibd_sum"] = pd.to_numeric(pd_graph["ibd_sum"], errors="raise")
pd_graph["ibd_n"] = pd.to_numeric(pd_graph["ibd_n"], errors="raise")

predicted_ancestry["Central-Russia"] = pd.to_numeric(
    predicted_ancestry["Central-Russia"],
    errors="coerce",
)

print(
    "pd_graph:", pd_graph.shape,
    "| predicted_ancestry:", predicted_ancestry.shape,
    "| ancestry_information:", ancestry_information.shape,
)

# --------------------
# Stage 0.1: Non-shared nodes report
# --------------------
print("\n[Stage 0.1] Build non-shared nodes report...")

graph_nodes = pd.Index(
    pd.concat([pd_graph["node_id1"], pd_graph["node_id2"]], ignore_index=True)
    .dropna()
    .unique(),
    dtype="string",
)

pred_nodes = pd.Index(
    predicted_ancestry["node_tubeid"].dropna().unique(),
    dtype="string",
)

tree_nodes = pd.Index(
    ancestry_information["node_tubeid"].dropna().unique(),
    dtype="string",
)

common_nodes = graph_nodes.intersection(pred_nodes).intersection(tree_nodes)
all_nodes_union = graph_nodes.union(pred_nodes).union(tree_nodes)
nonshared_nodes = all_nodes_union.difference(common_nodes)

graph_filename = os.path.basename(GRAPH_PATH)
pred_filename = os.path.basename(PRED_PATH)
tree_filename = os.path.basename(TREE_PATH)

nonshared_df = pd.DataFrame(
    {
        "node_id": nonshared_nodes.astype("string"),
        graph_filename: np.where(nonshared_nodes.isin(graph_nodes), "yes", "no"),
        pred_filename: np.where(nonshared_nodes.isin(pred_nodes), "yes", "no"),
        tree_filename: np.where(nonshared_nodes.isin(tree_nodes), "yes", "no"),
    }
)

nonshared_df.to_csv(OUT_NONSHARED_PATH, index=False)

print("Common nodes:", len(common_nodes))
print("Non-shared nodes:", len(nonshared_nodes))
print("Saved non-shared report:", OUT_NONSHARED_PATH)

# --------------------
# Stage 0: Primary filtration
# --------------------
print("\n[Stage 0] Primary filtration: keep only shared graph / pred / tree nodes...")

pd_graph_common = pd_graph[
    pd_graph["node_id1"].isin(common_nodes)
    & pd_graph["node_id2"].isin(common_nodes)
].copy()

pred_common = predicted_ancestry[
    predicted_ancestry["node_tubeid"].isin(common_nodes)
].copy()

tree_common = ancestry_information[
    ancestry_information["node_tubeid"].isin(common_nodes)
].copy()

print(
    "pd_graph_common:", pd_graph_common.shape,
    "| pred_common:", pred_common.shape,
    "| tree_common:", tree_common.shape,
)

# --------------------
# Stage 1: Central-Russia > 80
# --------------------
print(f"\n[Stage 1] Filter Central-Russia > {CR_THRESHOLD}...")

cr_nodes = pd.Index(
    pred_common.loc[
        pred_common["Central-Russia"] > CR_THRESHOLD,
        "node_tubeid",
    ].dropna().unique(),
    dtype="string",
)

pd_graph_cr = pd_graph_common[
    pd_graph_common["node_id1"].isin(cr_nodes)
    & pd_graph_common["node_id2"].isin(cr_nodes)
].copy()

tree_cr = tree_common[
    tree_common["node_tubeid"].isin(cr_nodes)
].copy()

print("cr_nodes:", len(cr_nodes))
print("pd_graph_cr:", pd_graph_cr.shape, "| tree_cr:", tree_cr.shape)

# --------------------
# Stage 2: Select labeled nodes by 3-ancestor rule
# --------------------
print("\n[Stage 2] Select labeled nodes by 3-ancestor rule...")

counts_by_group = (
    tree_cr
    .groupby(["node_tubeid", "group"])
    .size()
    .unstack(fill_value=0)
)

total_ancestors = counts_by_group.sum(axis=1)
distinct_groups = (counts_by_group > 0).sum(axis=1)
majority_group = counts_by_group.idxmax(axis=1)
majority_count = counts_by_group.max(axis=1)

main_mask = (
    (total_ancestors == 4)
    & (distinct_groups <= 2)
    & (majority_count >= 3)
    & (majority_group != "Unknown")
    & (majority_group.isin(TARGET_GROUPS))
)

main_majority_group = majority_group[main_mask].astype("string")
main_nodes = pd.Index(main_majority_group.index, dtype="string")

print("Labeled nodes selected before edge filtering:", len(main_nodes))
print("Selected labeled class balance:")
print(main_majority_group.value_counts())

# --------------------
# Stage 3: Final graph = all CR-CR edges
# --------------------
print("\n[Stage 3] Build final all-nodes CR > 80 graph...")

pd_graph_final = pd_graph_cr[
    ["node_id1", "node_id2", "ibd_sum", "ibd_n"]
].copy()

included_nodes_before_sibling_filter = get_unique_edge_nodes(pd_graph_final)
edge_min_before_sibling, edge_max_before_sibling = get_edge_weight_min_max(pd_graph_final)

print("Final CR-CR edges before sibling filter:", len(pd_graph_final))
print("Included nodes before sibling filter:", len(included_nodes_before_sibling_filter))
print("Min ibd_sum before sibling filter:", edge_min_before_sibling)
print("Max ibd_sum before sibling filter:", edge_max_before_sibling)
print(
    "CR > 80 nodes not written because they have no CR-CR edge:",
    len(cr_nodes.difference(included_nodes_before_sibling_filter)),
)

# --------------------
# Stage 4: Add GENLINK labels
# --------------------
print("\n[Stage 4] Add label_id1 and label_id2...")

node_label = {
    node: MASK_LABEL
    for node in included_nodes_before_sibling_filter
}

for node, label in main_majority_group.items():
    if node in node_label:
        node_label[node] = str(label)

pd_graph_final["label_id1"] = pd_graph_final["node_id1"].map(node_label)
pd_graph_final["label_id2"] = pd_graph_final["node_id2"].map(node_label)

if pd_graph_final["label_id1"].isna().any() or pd_graph_final["label_id2"].isna().any():
    bad = pd_graph_final[
        pd_graph_final["label_id1"].isna()
        | pd_graph_final["label_id2"].isna()
    ][["node_id1", "node_id2", "label_id1", "label_id2"]].head(10)

    raise RuntimeError(f"Some endpoint labels were not assigned:\n{bad}")

pd_graph_final = pd_graph_final[
    ["node_id1", "node_id2", "label_id1", "label_id2", "ibd_sum", "ibd_n"]
].copy()

print("Label balance before sibling filter:")
node_labels_before = pd.concat(
    [
        pd_graph_final[["node_id1", "label_id1"]].rename(
            columns={"node_id1": "node", "label_id1": "label"}
        ),
        pd_graph_final[["node_id2", "label_id2"]].rename(
            columns={"node_id2": "node", "label_id2": "label"}
        ),
    ],
    ignore_index=True,
).drop_duplicates(subset="node")

print(node_labels_before["label"].value_counts())

# --------------------
# Stage 5: Sibling filtering
# --------------------
print(f"\n[Stage 5] Sibling filtering: ibd_sum > {SIBLING_IBD_SUM_THRESHOLD}...")

edges_before_sibling_filter = len(pd_graph_final)
nodes_before_sibling_filter = get_unique_edge_nodes(pd_graph_final)

pd_graph_final, sibling_removed_nodes = sibling_filter_drop_masked_only_when_mixed(
    pd_graph_final,
    threshold=SIBLING_IBD_SUM_THRESHOLD,
    mask_label=MASK_LABEL,
)

edges_after_sibling_filter = len(pd_graph_final)
nodes_after_sibling_filter = get_unique_edge_nodes(pd_graph_final)
edge_min_after_sibling, edge_max_after_sibling = get_edge_weight_min_max(pd_graph_final)

print("Edges before sibling filter:", edges_before_sibling_filter)
print("Edges after sibling filter:", edges_after_sibling_filter)
print("Removed edges:", edges_before_sibling_filter - edges_after_sibling_filter)
print("Nodes before sibling filter:", len(nodes_before_sibling_filter))
print("Nodes after sibling filter:", len(nodes_after_sibling_filter))
print("Removed nodes by sibling filter:", len(sibling_removed_nodes))
print("Min ibd_sum after sibling filter:", edge_min_after_sibling)
print("Max ibd_sum after sibling filter:", edge_max_after_sibling)

if len(sibling_removed_nodes) > 0:
    print("First sibling-removed nodes:", sibling_removed_nodes[:10].tolist())

# --------------------
# Stage 6: Optional PCA coordinates
# --------------------
pca_nodes = pd.Index([], dtype="string")
pca_was_used = PCA_PATH is not None

if PCA_PATH is not None:
    print("\n[Stage 6] Add PCA coordinates...")

    pca_df = pd.read_csv(
        PCA_PATH,
        dtype={PCA_ID_COL: "string"},
        sep=None,
        engine="python",
    )

    if PCA_ID_COL not in pca_df.columns:
        raise ValueError(f"PCA file must contain ID column: {PCA_ID_COL}")

    missing_pca_cols = [c for c in PCA_COLS if c not in pca_df.columns]
    if missing_pca_cols:
        raise ValueError(f"PCA file is missing columns: {missing_pca_cols}")

    pca_df[PCA_ID_COL] = normalize_string_id(pca_df[PCA_ID_COL])

    pca_df = pca_df[[PCA_ID_COL, *PCA_COLS]].copy()
    pca_df = pca_df.drop_duplicates(subset=PCA_ID_COL, keep="first")

    for c in PCA_COLS:
        pca_df[c] = pd.to_numeric(pca_df[c], errors="raise").astype(np.float32)

    pca_nodes = pd.Index(
        pca_df[PCA_ID_COL].dropna().unique(),
        dtype="string",
    )

    current_nodes_before_final_edge_filter = get_unique_edge_nodes(pd_graph_final)
    missing_pca_nodes_before_final_edge_filter = current_nodes_before_final_edge_filter.difference(pca_nodes)

    print("Nodes before final edge-weight threshold:", len(current_nodes_before_final_edge_filter))
    print("Nodes with PCA before final edge-weight threshold:", len(current_nodes_before_final_edge_filter.intersection(pca_nodes)))
    print("Nodes without PCA before final edge-weight threshold:", len(missing_pca_nodes_before_final_edge_filter))

    if len(missing_pca_nodes_before_final_edge_filter) > 0:
        print(
            "First nodes without PCA before final edge-weight threshold:",
            missing_pca_nodes_before_final_edge_filter[:10].tolist(),
        )

    node1_pca_cols = [f"node1_PC{i}" for i in range(1, PCA_DIM + 1)]
    node2_pca_cols = [f"node2_PC{i}" for i in range(1, PCA_DIM + 1)]

    pca_node1 = pca_df.rename(
        columns={
            PCA_ID_COL: "node_id1",
            **{f"PC{i}": f"node1_PC{i}" for i in range(1, PCA_DIM + 1)},
        }
    )

    pca_node2 = pca_df.rename(
        columns={
            PCA_ID_COL: "node_id2",
            **{f"PC{i}": f"node2_PC{i}" for i in range(1, PCA_DIM + 1)},
        }
    )

    pd_graph_final = pd_graph_final.merge(
        pca_node1,
        on="node_id1",
        how="left",
    )

    pd_graph_final = pd_graph_final.merge(
        pca_node2,
        on="node_id2",
        how="left",
    )

    for c in node1_pca_cols + node2_pca_cols:
        pd_graph_final[c] = pd_graph_final[c].fillna(0.0).astype(np.float32)

    pd_graph_final = pd_graph_final[
        [
            "node_id1",
            "node_id2",
            "label_id1",
            "label_id2",
            "ibd_sum",
            "ibd_n",
            *node1_pca_cols,
            *node2_pca_cols,
        ]
    ].copy()

    print("Added PCA columns:", len(node1_pca_cols) + len(node2_pca_cols))

else:
    print("\n[Stage 6] PCA_PATH is None, saving without PCA columns...")

# --------------------
# Stage 7: Final global edge-weight threshold before saving
# --------------------
print("\n[Stage 7] Final global edge-weight threshold before saving...")

edges_before_min_edge_filter = len(pd_graph_final)
nodes_before_min_edge_filter = get_unique_edge_nodes(pd_graph_final)
edge_min_before_min_edge_filter, edge_max_before_min_edge_filter = get_edge_weight_min_max(pd_graph_final)

if MIN_EDGE_IBD_SUM_THRESHOLD is not None:
    print(f"Applying final edge filter: keep ibd_sum >= {MIN_EDGE_IBD_SUM_THRESHOLD}")

    pd_graph_final = pd_graph_final[
        pd_graph_final["ibd_sum"] >= MIN_EDGE_IBD_SUM_THRESHOLD
    ].copy()
else:
    print("MIN_EDGE_IBD_SUM_THRESHOLD is None, no final minimum edge-weight filter applied.")

edges_after_min_edge_filter = len(pd_graph_final)
nodes_after_min_edge_filter = get_unique_edge_nodes(pd_graph_final)
edge_min_final, edge_max_final = get_edge_weight_min_max(pd_graph_final)

threshold_removed_edges = edges_before_min_edge_filter - edges_after_min_edge_filter
threshold_removed_nodes = nodes_before_min_edge_filter.difference(nodes_after_min_edge_filter)

print("Edges before final edge-weight threshold:", edges_before_min_edge_filter)
print("Edges after final edge-weight threshold:", edges_after_min_edge_filter)
print("Removed edges by final edge-weight threshold:", threshold_removed_edges)
print("Nodes before final edge-weight threshold:", len(nodes_before_min_edge_filter))
print("Nodes after final edge-weight threshold:", len(nodes_after_min_edge_filter))
print("Removed nodes by final edge-weight threshold:", len(threshold_removed_nodes))
print("Min ibd_sum before final edge-weight threshold:", edge_min_before_min_edge_filter)
print("Max ibd_sum before final edge-weight threshold:", edge_max_before_min_edge_filter)
print("Final min ibd_sum:", edge_min_final)
print("Final max ibd_sum:", edge_max_final)

if len(threshold_removed_nodes) > 0:
    print("First nodes removed by final edge-weight threshold:", threshold_removed_nodes[:10].tolist())

included_nodes = nodes_after_min_edge_filter

node_label_after_filter = {
    node: node_label[node]
    for node in included_nodes
}

node_summary = pd.DataFrame({"node": included_nodes.astype("string")})
node_summary["label"] = node_summary["node"].map(node_label_after_filter)
node_summary["node_type"] = np.where(
    node_summary["label"] == MASK_LABEL,
    "masked",
    "labeled",
)

if pca_was_used:
    node_summary["has_pca"] = node_summary["node"].isin(pca_nodes)
else:
    node_summary["has_pca"] = False

# --------------------
# Stage 8: Save
# --------------------
print("\n[Stage 8] Saving...")

pd_graph_final.to_csv(OUT_PATH, index=False)

print("Saved:", OUT_PATH)

# --------------------
# Stage 9: Final stats
# --------------------
print("\n[Stage 9] Final dataset stats...")

total_nodes = len(node_summary)
labeled_nodes = int((node_summary["node_type"] == "labeled").sum())
masked_nodes = int((node_summary["node_type"] == "masked").sum())

nodes_with_pca = int(node_summary["has_pca"].sum())
nodes_without_pca = int((~node_summary["has_pca"]).sum())

overall_stats = pd.DataFrame(
    [
        {
            "metric": "final_edges",
            "value": int(pd_graph_final.shape[0]),
            "proportion": np.nan,
        },
        {
            "metric": "final_columns",
            "value": int(pd_graph_final.shape[1]),
            "proportion": np.nan,
        },
        {
            "metric": "final_min_ibd_sum",
            "value": edge_min_final,
            "proportion": np.nan,
        },
        {
            "metric": "final_max_ibd_sum",
            "value": edge_max_final,
            "proportion": np.nan,
        },
        {
            "metric": "total_nodes",
            "value": total_nodes,
            "proportion": 1.0,
        },
        {
            "metric": "labeled_nodes",
            "value": labeled_nodes,
            "proportion": labeled_nodes / total_nodes if total_nodes else 0.0,
        },
        {
            "metric": "masked_nodes",
            "value": masked_nodes,
            "proportion": masked_nodes / total_nodes if total_nodes else 0.0,
        },
        {
            "metric": "nodes_with_pca",
            "value": nodes_with_pca,
            "proportion": nodes_with_pca / total_nodes if total_nodes else 0.0,
        },
        {
            "metric": "nodes_without_pca",
            "value": nodes_without_pca,
            "proportion": nodes_without_pca / total_nodes if total_nodes else 0.0,
        },
        {
            "metric": "sibling_removed_nodes",
            "value": int(len(sibling_removed_nodes)),
            "proportion": len(sibling_removed_nodes) / len(nodes_before_sibling_filter)
            if len(nodes_before_sibling_filter)
            else 0.0,
        },
        {
            "metric": "final_edge_threshold",
            "value": MIN_EDGE_IBD_SUM_THRESHOLD if MIN_EDGE_IBD_SUM_THRESHOLD is not None else np.nan,
            "proportion": np.nan,
        },
        {
            "metric": "final_edge_threshold_removed_edges",
            "value": int(threshold_removed_edges),
            "proportion": threshold_removed_edges / edges_before_min_edge_filter
            if edges_before_min_edge_filter
            else 0.0,
        },
        {
            "metric": "final_edge_threshold_removed_nodes",
            "value": int(len(threshold_removed_nodes)),
            "proportion": len(threshold_removed_nodes) / len(nodes_before_min_edge_filter)
            if len(nodes_before_min_edge_filter)
            else 0.0,
        },
        {
            "metric": "cr_nodes_without_cr_edges_not_written",
            "value": int(len(cr_nodes.difference(included_nodes_before_sibling_filter))),
            "proportion": len(cr_nodes.difference(included_nodes_before_sibling_filter)) / len(cr_nodes)
            if len(cr_nodes)
            else 0.0,
        },
    ]
)

print_stats_table("Overall stats", overall_stats)

stats_by_node_type = (
    node_summary
    .groupby("node_type", dropna=False)
    .agg(
        total_nodes=("node", "size"),
        nodes_with_pca=("has_pca", "sum"),
    )
    .reset_index()
)

stats_by_node_type["nodes_without_pca"] = (
    stats_by_node_type["total_nodes"] - stats_by_node_type["nodes_with_pca"]
)

stats_by_node_type["category_proportion_of_all_nodes"] = (
    stats_by_node_type["total_nodes"] / total_nodes if total_nodes else 0.0
)

stats_by_node_type["with_pca_proportion_inside_category"] = np.where(
    stats_by_node_type["total_nodes"] > 0,
    stats_by_node_type["nodes_with_pca"] / stats_by_node_type["total_nodes"],
    0.0,
)

stats_by_node_type["without_pca_proportion_inside_category"] = np.where(
    stats_by_node_type["total_nodes"] > 0,
    stats_by_node_type["nodes_without_pca"] / stats_by_node_type["total_nodes"],
    0.0,
)

print_stats_table("PCA stats by node type", stats_by_node_type)

label_order = TARGET_GROUPS + [MASK_LABEL]

stats_by_label = (
    node_summary
    .groupby("label", dropna=False)
    .agg(
        total_nodes=("node", "size"),
        nodes_with_pca=("has_pca", "sum"),
    )
)

stats_by_label = stats_by_label.reindex(label_order, fill_value=0).reset_index()

stats_by_label["nodes_without_pca"] = (
    stats_by_label["total_nodes"] - stats_by_label["nodes_with_pca"]
)

stats_by_label["category_proportion_of_all_nodes"] = np.where(
    total_nodes > 0,
    stats_by_label["total_nodes"] / total_nodes,
    0.0,
)

stats_by_label["with_pca_proportion_inside_category"] = np.where(
    stats_by_label["total_nodes"] > 0,
    stats_by_label["nodes_with_pca"] / stats_by_label["total_nodes"],
    0.0,
)

stats_by_label["without_pca_proportion_inside_category"] = np.where(
    stats_by_label["total_nodes"] > 0,
    stats_by_label["nodes_without_pca"] / stats_by_label["total_nodes"],
    0.0,
)

print_stats_table("PCA stats by label category", stats_by_label)

print("\nFinal preview:")
pd_graph_final.head()
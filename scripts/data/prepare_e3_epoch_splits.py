"""按三分类仓库 E3 规则生成全量 MIT-BIH 的样本级五折索引。

仅生成划分文件；不复制波形，也不对 ECG 做预处理。使用三分类原始标签分层，
可保证折成员与 E3 一致；下游训练再把标签折叠成二分类。
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
from sklearn.model_selection import StratifiedKFold

ROOT = Path(__file__).resolve().parents[2]
DEFAULT_MASTER = ROOT.parent / "dataset" / "mitbih_master" / "mitbih_master_30s.npz"
DEFAULT_OUTPUT = ROOT / "protocols" / "e3_full_epoch5fold_20260913"
THREE_CLASS = {
    "Normal": 0,
    "H": 1,
    "HA": 1,
    "X": 2,
    "OA": 2,
    "CA": 2,
    "CAA": 2,
}
SEED = 20260913
EXPECTED_ROWS = 10197


def sha256(path: Path) -> str:
    """记录输入 master 的内容哈希，避免以后用到不同版本的数据。"""

    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def build_roles(raw_labels: np.ndarray) -> dict[int, dict[str, np.ndarray]]:
    """复现 E3 的分层折编号与测试/下一折验证/其余三折训练规则。"""

    labels = np.asarray(raw_labels).astype(str)
    unknown = sorted(set(labels.tolist()) - set(THREE_CLASS))
    if unknown:
        raise ValueError(f"master 含未知原始标签: {unknown}")
    strata = np.asarray([THREE_CLASS[label] for label in labels], dtype=np.int64)
    rows = np.arange(len(labels), dtype=np.int64)
    fold_ids = np.empty(len(labels), dtype=np.int64)
    splitter = StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED)
    for fold, (_, test_indices) in enumerate(splitter.split(rows, strata)):
        fold_ids[test_indices] = fold

    roles_by_fold: dict[int, dict[str, np.ndarray]] = {}
    for fold in range(5):
        next_fold = (fold + 1) % 5
        roles_by_fold[fold] = {
            "train": rows[(fold_ids != fold) & (fold_ids != next_fold)],
            "val": rows[fold_ids == next_fold],
            "test": rows[fold_ids == fold],
        }
    return roles_by_fold


def verify_reference(
    roles_by_fold: dict[int, dict[str, np.ndarray]], reference_path: Path
) -> str:
    """若提供三分类仓库冻结索引，逐折逐角色核对是否完全相同。"""

    with np.load(reference_path, allow_pickle=False) as reference:
        for fold, roles in roles_by_fold.items():
            for role, indices in roles.items():
                key = f"e3_run{fold + 1}_{role}"
                if key not in reference or not np.array_equal(indices, reference[key]):
                    raise ValueError(f"生成的 {key} 与三分类仓库冻结索引不一致")
    return sha256(reference_path)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--master", type=Path, default=DEFAULT_MASTER)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--reference-splits", type=Path)
    args = parser.parse_args()

    master_path = args.master.resolve()
    output_dir = args.output_dir.resolve()
    with np.load(master_path, allow_pickle=False) as master:
        labels = master["raw_label"].astype(str)
        subjects = master["subject_id"].astype(str)
    if len(labels) != EXPECTED_ROWS or len(subjects) != EXPECTED_ROWS:
        raise ValueError(f"E3 要求 {EXPECTED_ROWS} 行 master 数据")

    roles_by_fold = build_roles(labels)
    reference_sha = (
        verify_reference(roles_by_fold, args.reference_splits.resolve())
        if args.reference_splits is not None
        else None
    )
    all_test = np.concatenate([roles["test"] for roles in roles_by_fold.values()])
    if not np.array_equal(np.sort(all_test), np.arange(EXPECTED_ROWS)):
        raise ValueError("五个测试折未恰好覆盖全部样本")

    output_dir.mkdir(parents=True, exist_ok=True)
    for fold, roles in roles_by_fold.items():
        payload = {
            "name": f"fold_{fold}",
            "protocol": "E3 full 10197 epoch-level stratified five-fold",
            "split_unit": "epoch",
            "seed": SEED,
            "stratification": "Normal / Hypopnea / Apnea from seven raw labels",
            "fold": fold,
        }
        for role, indices in roles.items():
            payload[f"{role}_indices"] = indices.tolist()
            payload[f"{role}_subjects"] = sorted(set(subjects[indices].tolist()))
            payload[f"{role}_size"] = len(indices)
            payload[f"{role}_binary_counts"] = [
                int(np.sum(labels[indices] == "Normal")),
                int(np.sum(labels[indices] != "Normal")),
            ]
        (output_dir / f"fold_{fold}.json").write_text(
            json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8"
        )

    manifest = {
        "protocol": "E3 full 10197 epoch-level stratified five-fold",
        "split_unit": "epoch",
        "seed": SEED,
        "rows": EXPECTED_ROWS,
        "stratification": "Normal / Hypopnea / Apnea from seven raw labels",
        "roles": "fold i test, fold (i+1)%5 validation, other folds train",
        "subject_independent": False,
        "master_sha256": sha256(master_path),
        "reference_splits_sha256": reference_sha,
    }
    (output_dir / "manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(f"已生成五折样本级划分: {output_dir}")
    print(f"与参考索引逐项一致: {reference_sha is not None}")


if __name__ == "__main__":
    main()

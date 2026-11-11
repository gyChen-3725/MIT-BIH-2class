"""确认两路输入组合时保留样本对应关系和各自的时间序列。"""

import numpy as np

from scripts.clustering.cluster_daeac_features import prepare_cluster_inputs


def test_three_cluster_inputs_share_the_same_training_rows() -> None:
    raw = np.zeros((4, 2, 256), dtype=np.float32)
    raw[:, 0, :] = np.asarray([0.5, 0.7, 0.9, 1.1])[:, None]
    raw[:, 1, :] = np.asarray([-2.0, -1.0, 1.0, 2.0])[:, None]
    inputs, scaler = prepare_cluster_inputs(raw)

    assert inputs["rri"].shape == (4, 256)
    assert inputs["amplitude"].shape == (4, 256)
    assert inputs["both"].shape == (4, 512)
    np.testing.assert_array_equal(inputs["both"][:, :256], inputs["rri"])
    np.testing.assert_array_equal(inputs["both"][:, 256:], inputs["amplitude"])
    assert abs(inputs["rri"].mean()) < 1e-6
    assert abs(inputs["amplitude"].mean()) < 1e-6
    assert len(scaler["mean"]) == len(scaler["std"]) == 2

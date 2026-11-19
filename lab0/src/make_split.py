import hashlib
import json
from pathlib import Path
import numpy as np
import sklearn
from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split


def main():
    Path("data").mkdir(exist_ok=True)
    X, y = load_digits(return_X_y=True)
    idx = np.arange(len(y))

    rest, test = train_test_split(idx, test_size=0.2, stratify=y, random_state=0)
    train, val = train_test_split(rest, test_size=0.25, stratify=y[rest], random_state=0)

    # 校验无交集、总数为1797、每部分含全10类
    assert len(set(train) & set(val)) == 0
    assert len(set(train) & set(test)) == 0
    assert len(set(val) & set(test)) == 0
    assert len(train) + len(val) + len(test) == 1797
    for part in (train, val, test):
        assert len(np.unique(y[part])) == 10

    split_data = {
        "split_seed": 0,
        "train": sorted(train.tolist()),
        "val": sorted(val.tolist()),
        "test": sorted(test.tolist()),
    }
    split_path = Path("data/split.json")
    split_bytes = json.dumps(split_data, indent=2).encode("utf-8")
    split_path.write_bytes(split_bytes)

    sha256_hash = hashlib.sha256(split_bytes).hexdigest()
    manifest = {
        "source": f"scikit-learn {sklearn.__version__} load_digits",
        "split_rule": "stratified train/val/test = 60%/20%/20% (1077/360/360) with split_seed=0",
        "preprocessing": "features divided by 16.0",
        "sizes": {"train": len(train), "val": len(val), "test": len(test)},
        "split_sha256": sha256_hash,
    }
    Path("data/manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(f"Split saved. SHA-256: {sha256_hash}")


if __name__ == "__main__":
    main()
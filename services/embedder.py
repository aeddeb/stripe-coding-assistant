"""Local ONNX sentence embedder — all-MiniLM-L6-v2, CPU, no API costs.

Used at both index time (embedding chunks) and query time (embedding the
user's question), so it lives in ``services`` rather than ``ingestion``.

The model reads at most ~256 tokens (~1,200 characters); longer input is
truncated, which is why the chunker caps chunk size (see
``ingestion/chunking.py``).

Get the model files (~90 MB, one time)::

    uv run python -m services.embedder
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import onnxruntime as ort
from tokenizers import Tokenizer

MODEL_REPO = "Xenova/all-MiniLM-L6-v2"
MODEL_DIR = Path(__file__).resolve().parent.parent / "data" / "models" / MODEL_REPO
EMBEDDING_DIM = 384


class Embedder:
    def __init__(self, path: Path = MODEL_DIR):
        path = Path(path)
        if not ((path / "tokenizer.json").exists() and (path / "model.onnx").exists()):
            # Model files are not in the repo (data/ is gitignored). Docker
            # bakes them at build time; other hosts (e.g. Streamlit Cloud)
            # download them on first use.
            ensure_model(dest=path)
        self.tokenizer = Tokenizer.from_file(str(path / "tokenizer.json"))
        self.session = ort.InferenceSession(
            str(path / "model.onnx"), providers=["CPUExecutionProvider"]
        )
        self.input_names = {inp.name for inp in self.session.get_inputs()}

    def encode(self, text: str, normalize: bool = True) -> np.ndarray:
        return self.encode_batch([text], normalize=normalize)[0]

    def encode_batch(self, texts: list[str], normalize: bool = True) -> np.ndarray:
        # Truncate to the model's positional limit; without this, inputs
        # longer than 512 tokens fail at inference time.
        self.tokenizer.enable_truncation(max_length=512)
        self.tokenizer.enable_padding()
        encoded = self.tokenizer.encode_batch(texts)
        feed = {}
        if "input_ids" in self.input_names:
            feed["input_ids"] = np.array([e.ids for e in encoded], dtype=np.int64)
        if "attention_mask" in self.input_names:
            feed["attention_mask"] = np.array(
                [e.attention_mask for e in encoded], dtype=np.int64
            )
        if "token_type_ids" in self.input_names:
            feed["token_type_ids"] = np.array(
                [e.type_ids for e in encoded], dtype=np.int64
            )
        hidden = self.session.run(None, feed)[0]
        # Mean pooling over real (non-padding) tokens.
        mask = feed["attention_mask"][..., None]
        pooled = (hidden * mask).sum(axis=1) / mask.sum(axis=1)
        if normalize:
            pooled = pooled / np.linalg.norm(pooled, axis=1, keepdims=True)
        return pooled


def ensure_model(repo: str = MODEL_REPO, dest: Path = MODEL_DIR) -> Path:
    """Download tokenizer + ONNX weights from the Hugging Face Hub if they
    are not already on disk."""
    dest = Path(dest)
    if (dest / "model.onnx").exists() and (dest / "tokenizer.json").exists():
        return dest

    import shutil

    from huggingface_hub import hf_hub_download, list_repo_files

    dest.mkdir(parents=True, exist_ok=True)
    files = list_repo_files(repo_id=repo)
    onnx_file = next(
        (c for c in ("onnx/model.onnx", "model.onnx") if c in files), None
    )
    if not onnx_file:
        raise FileNotFoundError(f"No ONNX model found in {repo}")
    for remote, local in [("tokenizer.json", "tokenizer.json"), (onnx_file, "model.onnx")]:
        src = hf_hub_download(repo_id=repo, filename=remote)
        shutil.copy(src, dest / local)
    return dest


if __name__ == "__main__":
    path = ensure_model()
    dim = Embedder(path).encode("hello world").shape[0]
    print(f"Model ready at {path} (dim={dim})")

"""Track the same sentences in a fixed 2D PCA space during training."""
import argparse
import csv
import hashlib
import warnings
import io
import json
from pathlib import Path

import numpy as np
from sklearn.decomposition import PCA
import torch
from transformers import TrainerCallback

LABEL_NAMES = ["Negative", "Neutral", "Positive"]
COLORS = ["#db746b", "#9c8acf", "#319c8b"]


def normalize_embeddings(values):
    values = np.asarray(values, dtype=float)
    if values.ndim != 2 or min(values.shape) < 2 or not np.isfinite(values).all():
        raise ValueError("Embeddings must be a finite matrix with at least two rows and two dimensions.")
    return values / np.maximum(np.linalg.norm(values, axis=1, keepdims=True), 1e-12)


class FixedProjection:
    """Fit once on the first snapshot; never rotate the axes between steps."""
    def __init__(self, initial):
        initial = normalize_embeddings(initial)
        if np.allclose(initial, initial[0]):
            raise ValueError("Initial embeddings are identical; PCA needs variation between sentences.")
        self.pca = PCA(n_components=2, svd_solver="full").fit(initial)
        self.shape = initial.shape

    def transform(self, values):
        values = normalize_embeddings(values)
        if values.shape != self.shape:
            raise ValueError("Every snapshot must contain the same sentences in the same order and dimension.")
        return self.pca.transform(values)


def load_plot_sentences(path, max_samples=300, seed=42):
    """Read an external text,label CSV and select a fixed sample once."""
    if max_samples < 2:
        raise ValueError("embedding_plot_max_samples must be at least 2.")
    with Path(path).open(encoding="utf-8-sig", newline="") as stream:
        reader = csv.DictReader(stream)
        if not {"text", "label"}.issubset(reader.fieldnames or []):
            raise ValueError("Embedding plot CSV requires text,label columns (0=negative, 1=neutral, 2=positive).")
        rows = list(reader)
    if len(rows) < 2:
        raise ValueError("Embedding plot CSV needs at least two sentences.")
    for row in rows:
        if not row["text"].strip() or row["label"] not in {"0", "1", "2"}:
            raise ValueError("Each plot row requires nonempty text and a label of 0, 1 or 2.")
    if len(rows) > max_samples:
        chosen = np.sort(np.random.default_rng(seed).choice(len(rows), max_samples, replace=False))
        rows = [rows[i] for i in chosen]
    return [row["text"] for row in rows], np.array([int(row["label"]) for row in rows])


def write_projection_report(output, frames, labels, variance, *, demo=False):
    """Write an offline step slider and a standalone SVG of the latest frame."""
    from matplotlib.figure import Figure
    from matplotlib.backends.backend_svg import FigureCanvasSVG

    labels = np.asarray(labels)
    if labels.ndim != 1 or len(labels) < 2 or not np.isin(labels, [0, 1, 2]).all():
        raise ValueError("Labels must be a one-dimensional array of 0, 1 or 2.")
    if not frames:
        raise ValueError("At least one projection frame is required.")
    coordinates = [np.asarray(frame["coordinates"]) for frame in frames]
    if any(values.shape != (len(labels), 2) or not np.isfinite(values).all() for values in coordinates):
        raise ValueError("Each frame must have finite coordinates of shape (number of labels, 2).")
    all_points = np.concatenate(coordinates)
    low, high = all_points.min(axis=0), all_points.max(axis=0)
    margin = np.maximum((high - low) * 0.08, 0.03)
    slides = []
    for frame, points in zip(frames, coordinates):
        figure = Figure(figsize=(8, 5.4), layout="constrained")
        axis = figure.subplots()
        for label, name, color in zip(range(3), LABEL_NAMES, COLORS):
            selected = points[labels == label]
            axis.scatter(selected[:, 0], selected[:, 1], s=20, c=color, alpha=.72, linewidths=0, label=name)
        axis.set(xlim=(low[0] - margin[0], high[0] + margin[0]),
                 ylim=(low[1] - margin[1], high[1] + margin[1]),
                 xlabel=f"PC1 (initial variance {variance[0]:.1%})",
                 ylabel=f"PC2 (initial variance {variance[1]:.1%})",
                 title=f"{'Synthetic demo | ' if demo else ''}Step {int(frame['step'])} | fixed initial PCA")
        axis.set_aspect("equal", adjustable="box")
        axis.spines[["top", "right"]].set_visible(False)
        axis.grid(alpha=.15)
        axis.legend(loc="upper right", frameon=False)
        stream = io.StringIO()
        FigureCanvasSVG(figure).print_svg(stream, metadata={"Date": None})
        svg = stream.getvalue()
        svg = svg[svg.index("<svg"):]
        slides.append({"step": int(frame["step"]), "svg": svg})
    payload = json.dumps({"demo": bool(demo), "count": len(labels), "frames": slides}, ensure_ascii=False).replace("<", "\\u003c")
    template = Path(__file__).with_name("report_template.html").read_text(encoding="utf-8")
    destination = Path(output)
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(template.replace("__REPORT_DATA__", payload), encoding="utf-8")
    destination.with_suffix(".svg").write_text(slides[-1]["svg"], encoding="utf-8")
    return destination


class EmbeddingPlotCallback(TrainerCallback):
    """Optional, single-process snapshots on a fixed external sentence set."""
    def __init__(self, tokenizer, csv_file, output, every_steps=250, max_samples=300,
                 max_frames=30, batch_size=32, max_length=64, seed=42, resume_checkpoint=None):
        if every_steps < 1 or max_frames < 2 or batch_size < 1:
            raise ValueError("Plot interval/batch size must be positive; max_frames must be at least 2.")
        self.texts, self.labels = load_plot_sentences(csv_file, max_samples, seed)
        self.tokenizer, self.output = tokenizer, Path(output)
        self.every_steps, self.max_frames = every_steps, max_frames
        self.batch_size, self.max_length = batch_size, max_length
        self.frames, self.projection = [], None
        self.resume_checkpoint = resume_checkpoint
        identity = {"texts": self.texts, "labels": self.labels.tolist(), "max_length": max_length,
                    "vocab": tokenizer.get_vocab() if hasattr(tokenizer, "get_vocab") else None}
        self.fingerprint = hashlib.sha256(json.dumps(identity, sort_keys=True, ensure_ascii=False).encode()).hexdigest()

    def save_state(self, checkpoint):
        path = Path(checkpoint) / "embedding-plot.npz"
        path.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(path, fingerprint=self.fingerprint,
                            shape=self.projection.shape,
                            components=self.projection.pca.components_,
                            mean=self.projection.pca.mean_,
                            variance=self.projection.pca.explained_variance_ratio_,
                            steps=[frame["step"] for frame in self.frames],
                            coordinates=np.stack([frame["coordinates"] for frame in self.frames]))

    def restore_state(self, checkpoint, step):
        with np.load(Path(checkpoint) / "embedding-plot.npz", allow_pickle=False) as saved:
            if saved["fingerprint"].item() != self.fingerprint:
                raise ValueError("Plot sentences, labels, tokenizer or max_length changed since the checkpoint.")
            shape = tuple(int(value) for value in saved["shape"])
            components, mean, variance = saved["components"], saved["mean"], saved["variance"]
            coordinates, steps = saved["coordinates"], saved["steps"]
            if (len(shape) != 2 or shape[0] != len(self.labels) or shape[1] < 2
                    or components.shape != (2, shape[1]) or mean.shape != (shape[1],)
                    or variance.shape != (2,) or steps.ndim != 1 or len(steps) < 1
                    or coordinates.shape != (len(steps), len(self.labels), 2)
                    or not all(np.isfinite(value).all() for value in (components, mean, variance, coordinates, steps))
                    or (np.diff(steps) <= 0).any() or (steps < 0).any() or (steps > step).any()):
                raise ValueError("Invalid embedding plot state in checkpoint.")
            projection = FixedProjection.__new__(FixedProjection)
            projection.shape = shape
            projection.pca = PCA(n_components=2, svd_solver="full")
            projection.pca.components_ = components.copy()
            projection.pca.mean_ = mean.copy()
            projection.pca.explained_variance_ratio_ = variance.copy()
            projection.pca.n_features_in_ = shape[1]
            self.projection = projection
            self.frames = [{"step": int(value), "coordinates": points.copy()}
                           for value, points in zip(steps, coordinates)]
            if len(self.frames) > self.max_frames:
                self.frames = [self.frames[0], *self.frames[-(self.max_frames - 1):]]

    def capture(self, model, step):
        if self.frames and self.frames[-1]["step"] == step:
            return
        was_training = model.training
        device = next(model.parameters()).device
        devices = [device.index] if device.type == "cuda" else []
        try:
            # Snapshotting must not consume the training RNG or leave dropout disabled.
            model.eval()
            batches = []
            with torch.random.fork_rng(devices=devices), torch.inference_mode():
                for start in range(0, len(self.texts), self.batch_size):
                    batch = self.tokenizer(self.texts[start:start + self.batch_size], padding=True,
                                           truncation=True, max_length=self.max_length, return_tensors="pt")
                    batch = {key: value.to(device) for key, value in batch.items()}
                    vectors = model(**batch, sent_emb=True, return_dict=True).pooler_output
                    batches.append(vectors.float().cpu().numpy())
            values = np.concatenate(batches)
            if self.projection is None:
                self.projection = FixedProjection(values)
            self.frames.append({"step": int(step), "coordinates": self.projection.transform(values)})
            if len(self.frames) > self.max_frames:
                # Retain the initial basis and the most recent observations.
                self.frames = [self.frames[0], *self.frames[-(self.max_frames - 1):]]
            write_projection_report(self.output, self.frames, self.labels,
                                    self.projection.pca.explained_variance_ratio_)
        finally:
            model.train(was_training)

    def on_train_begin(self, args, state, control, model=None, **kwargs):
        if args.world_size != 1 or args.fsdp or args.deepspeed:
            raise ValueError("Embedding snapshots currently support single-process training only.")
        if state.global_step > 0:
            checkpoint = self.resume_checkpoint
            if checkpoint is True:
                from transformers.trainer_utils import get_last_checkpoint
                checkpoint = get_last_checkpoint(args.output_dir)
            if checkpoint and (Path(checkpoint) / "embedding-plot.npz").is_file():
                self.restore_state(checkpoint, state.global_step)
                write_projection_report(self.output, self.frames, self.labels,
                                        self.projection.pca.explained_variance_ratio_)
            else:
                warnings.warn("Checkpoint has no plot state; starting a new PCA baseline in a separate report.")
                self.output = self.output.with_name(f"embeddings-from-step-{state.global_step}.html")
        self.capture(model, state.global_step)

    def on_save(self, args, state, control, **kwargs):
        if self.projection is not None:
            self.save_state(Path(args.output_dir) / f"checkpoint-{state.global_step}")

    def on_step_end(self, args, state, control, model=None, **kwargs):
        if state.global_step % self.every_steps == 0:
            self.capture(model, state.global_step)

    def on_train_end(self, args, state, control, model=None, **kwargs):
        self.capture(model, state.global_step)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--demo", action="store_true", required=True, help="Generate synthetic snapshots, not model results.")
    parser.add_argument("--output", type=Path, default=Path("outputs/embedding-demo.html"))
    args = parser.parse_args()
    rng = np.random.default_rng(42)
    labels = np.repeat(np.arange(3), 60)
    initial = rng.normal(size=(180, 16))
    initial[:, :3] += np.eye(3)[labels] * .8
    projection = FixedProjection(initial)
    frames = []
    for step, separation in [(0, 0), (250, .7), (500, 1.6)]:
        values = initial.copy()
        values[:, :3] += np.eye(3)[labels] * separation
        frames.append({"step": step, "coordinates": projection.transform(values)})
    print(write_projection_report(args.output, frames, labels, projection.pca.explained_variance_ratio_, demo=True))


if __name__ == "__main__":
    main()

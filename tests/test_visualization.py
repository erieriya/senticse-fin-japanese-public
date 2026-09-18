import json
import re
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from senticse.visualization import FixedProjection, EmbeddingPlotCallback, load_plot_sentences, write_projection_report


def test_fixed_projection_does_not_refit():
    rng = np.random.default_rng(42)
    initial = rng.normal(size=(20, 8))
    projection = FixedProjection(initial)
    components = projection.pca.components_.copy()
    before = projection.transform(initial)
    changed = initial.copy(); changed[:, 0] += .5
    after = projection.transform(changed)
    np.testing.assert_array_equal(components, projection.pca.components_)
    np.testing.assert_allclose(projection.transform(initial), before)
    assert not np.allclose(before, after)
    with pytest.raises(ValueError, match="same sentences"):
        projection.transform(changed[:10])


def test_svg_and_offline_slider(tmp_path):
    labels = [0, 1, 2]
    frames = [{"step": 0, "coordinates": np.array([[0, 1], [1, 0], [-1, 0]])},
              {"step": 10, "coordinates": np.array([[0, 2], [2, 0], [-2, 0]])}]
    path = write_projection_report(tmp_path / "plot.html", frames, labels, [.6, .3], demo=True)
    html = path.read_text()
    payload = json.loads(re.search(r'<script id="report-data" type="application/json">(.*?)</script>', html, re.S).group(1))
    assert payload["demo"]
    assert [frame["step"] for frame in payload["frames"]] == [0, 10]
    assert all("<svg" in frame["svg"] for frame in payload["frames"])
    assert path.with_suffix(".svg").is_file()
    assert '<script src=' not in html and 'fetch(' not in html
    assert "__REPORT_DATA__" not in html


def test_csv_fixed_sample(tmp_path):
    path = tmp_path / "sentences.csv"
    path.write_text("text,label\n" + "\n".join(f"synthetic-{i},{i % 3}" for i in range(20)))
    texts, labels = load_plot_sentences(path, max_samples=5)
    assert len(texts) == len(labels) == 5
    assert texts == load_plot_sentences(path, max_samples=5)[0]
    path.write_text("text,label\nhello,9\nworld,0\n")
    with pytest.raises(ValueError, match="label"):
        load_plot_sentences(path)


@pytest.mark.parametrize("values", [np.ones((4, 3)), [[float('nan'), 1], [1, 2]], [[1, 2]]])
def test_invalid_embeddings(values):
    with pytest.raises(ValueError):
        FixedProjection(values)


def test_callback_preserves_rng_and_mode_and_limits_frames(tmp_path):
    class Tokenizer:
        def __call__(self, texts, **kwargs):
            return {"input_ids": torch.tensor([[int(text[-1])] for text in texts])}
    class Model(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.embeddings = torch.nn.Embedding(4, 3)
        def forward(self, input_ids, **kwargs):
            # Force random consumption to verify snapshot RNG isolation.
            torch.rand(1)
            return SimpleNamespace(pooler_output=self.embeddings(input_ids[:, 0]))
    path = tmp_path / "sentences.csv"
    path.write_text("text,label\nPRIVATE_SENTENCE_0,0\nPRIVATE_SENTENCE_1,1\nPRIVATE_SENTENCE_2,2\nPRIVATE_SENTENCE_3,1\n")
    callback = EmbeddingPlotCallback(Tokenizer(), path, tmp_path / "frames.html", max_frames=2)
    model = Model().train()
    before = torch.get_rng_state().clone()
    callback.capture(model, 0)
    torch.testing.assert_close(torch.get_rng_state(), before)
    assert model.training
    callback.capture(model, 1); callback.capture(model, 2); callback.capture(model, 2)
    assert [frame["step"] for frame in callback.frames] == [0, 2]
    callback.save_state(tmp_path / "checkpoint-2")
    restored = EmbeddingPlotCallback(Tokenizer(), path, tmp_path / "restored.html", max_frames=2)
    restored.restore_state(tmp_path / "checkpoint-2", 2)
    np.testing.assert_array_equal(restored.projection.pca.components_, callback.projection.pca.components_)
    np.testing.assert_array_equal(restored.frames[-1]["coordinates"], callback.frames[-1]["coordinates"])
    vectors = model.embeddings.weight.detach().numpy()
    np.testing.assert_allclose(restored.projection.transform(vectors), callback.projection.transform(vectors))
    changed = EmbeddingPlotCallback(Tokenizer(), path, tmp_path / "changed.html", max_length=20)
    with pytest.raises(ValueError, match="changed"):
        changed.restore_state(tmp_path / "checkpoint-2", 2)
    assert "PRIVATE_SENTENCE" not in (tmp_path / "frames.html").read_text().split('id="report-data"')[1].split('</script>')[0]

from pathlib import Path
import pytest
from scripts.export_public import export, check, public_files


def test_export_excludes_data_and_history(tmp_path):
    root = tmp_path / "private"; root.mkdir()
    (root / "public-files.txt").write_text("train.py\npublic-files.txt\n")
    (root / "train.py").write_text("print('synthetic')\n")
    (root / "data.csv").write_text("PRIVATE")
    (root / ".git").mkdir()
    target = tmp_path / "public"
    assert export(target, root) == 2
    assert check(target, root) == 2
    assert not (target / "data.csv").exists()
    assert not (target / ".git").exists()
    with pytest.raises(ValueError, match="must not exist"):
        export(target, root)
    (target / "leak.csv").write_text("PRIVATE")
    with pytest.raises(ValueError, match="Unexpected"):
        check(target, root)


@pytest.mark.parametrize("entry", ["../secret.py", "/tmp/secret.py", "data/file.py", "weights.npy", ".git/config"])
def test_manifest_rejects_private_paths(tmp_path, entry):
    (tmp_path / "public-files.txt").write_text(entry)
    with pytest.raises(ValueError):
        public_files(tmp_path)


def test_manifest_rejects_symlinks(tmp_path):
    (tmp_path / "public-files.txt").write_text("public.py")
    (tmp_path / "secret.py").write_text("PRIVATE")
    (tmp_path / "public.py").symlink_to(tmp_path / "secret.py")
    with pytest.raises(ValueError, match="symlinks"):
        public_files(tmp_path)

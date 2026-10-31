from cxrbench.data import find_images


def test_find_images_uses_class_folders(tmp_path):
    folders = {
        "Normal": "a.png",
        "Lung Opacity": "b.jpg",
        "Viral Pneumonia": "c.png",
    }
    for folder, filename in folders.items():
        directory = tmp_path / folder
        directory.mkdir()
        (directory / filename).write_bytes(b"placeholder")
    df = find_images(tmp_path)
    assert len(df) == 3
    assert sorted(df["y"].tolist()) == [0, 1, 2]

import shutil

from PIL import Image

from scripts.prepare_official_plantdoc import prepare


def make_image(path, color) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.new("RGB", (128, 128), color).save(path)


def test_preserves_official_test_and_filters_to_tomato(tmp_path) -> None:
    source = tmp_path / "official"
    classes = ["Tomato leaf", "Tomato leaf late blight", "Apple leaf"]
    for split in ["train", "test"]:
        for class_index, label in enumerate(classes):
            count = 8 if split == "train" else 3
            for image_index in range(count):
                make_image(
                    source / split / label / f"{image_index}.jpg",
                    (
                        20 + class_index * 60,
                        30 + image_index * 5,
                        40 + (0 if split == "train" else 80),
                    ),
                )

    for image_index in range(5):
        make_image(
            source / "train" / "Tomato train only disease" / f"{image_index}.jpg",
            (200, 20 + image_index * 5, 30),
        )

    duplicate_source = source / "test" / "Tomato leaf" / "0.jpg"
    shutil.copy2(duplicate_source, source / "train" / "Tomato leaf" / "duplicate.jpg")

    output = tmp_path / "processed"
    summary = prepare(source, output, label_regex="tomato", val_ratio=0.25, seed=42)

    assert summary["official_test_preserved"] is True
    assert summary["class_count"] == 2
    assert summary["dropped_train_only_labels"] == ["Tomato train only disease"]
    assert summary["removed_exact_train_test_duplicates"] == 1
    assert not (output / "train" / "Apple leaf").exists()
    assert len(list((output / "test" / "Tomato leaf").glob("*.jpg"))) == 3
    assert (output / "manifest.csv").exists()
    assert (output / "preparation_summary.json").exists()


def test_accepts_single_wrapper_directory(tmp_path) -> None:
    source = tmp_path / "download"
    wrapped = source / "PlantDoc-Dataset-master"
    for split in ["train", "test"]:
        count = 4 if split == "train" else 2
        for image_index in range(count):
            make_image(
                wrapped / split / "Tomato leaf" / f"{image_index}.jpg",
                (20 + image_index * 10, 40, 60 + (80 if split == "test" else 0)),
            )

    output = tmp_path / "processed"
    summary = prepare(source, output, label_regex="tomato", val_ratio=0.25, seed=42)

    assert summary["class_count"] == 1
    assert len(list((output / "test" / "Tomato leaf").glob("*.jpg"))) == 2


def test_quarantines_identical_pixels_with_conflicting_labels(tmp_path) -> None:
    source = tmp_path / "official"
    labels = ["Tomato mold leaf", "Tomato leaf yellow virus"]
    for split in ["train", "test"]:
        count = 6 if split == "train" else 3
        for label_index, label in enumerate(labels):
            for image_index in range(count):
                make_image(
                    source / split / label / f"{image_index}.jpg",
                    (
                        20 + label_index * 100,
                        30 + image_index * 7,
                        40 + (80 if split == "test" else 0),
                    ),
                )

    conflict_source = source / "train" / labels[0] / "0.jpg"
    shutil.copy2(conflict_source, source / "train" / labels[1] / "conflict.jpg")

    output = tmp_path / "processed"
    summary = prepare(source, output, label_regex="tomato", val_ratio=0.25, seed=42)

    assert summary["conflicting_duplicate_groups_removed"] == 1
    assert summary["conflicting_duplicate_images_removed"] == {
        "train": 2,
        "test": 0,
    }
    conflict_digest = summary["conflicting_duplicate_details"][0]["digest"]
    manifest_text = (output / "manifest.csv").read_text(encoding="utf-8")
    assert conflict_digest not in manifest_text

import re
import shutil
from pathlib import Path


INPUT_ROOT = Path("dataset/25045993_He")
OUTPUT_ROOT = Path("dataset")
IMAGE_EXTS = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff", ".webp"}


def is_gesture_dir_name(name: str) -> bool:
    return re.match(r"^G\d{2}", name) is not None


def gesture_code(name: str) -> str:
    match = re.match(r"^(G\d{2})", name)
    if match is None:
        raise ValueError(f"Invalid gesture directory name: {name}")
    return match.group(1)


def find_payload_root(input_root: Path) -> Path:
    input_root = input_root.resolve()
    children = [p for p in input_root.iterdir() if p.is_dir()]
    if len(children) == 1:
        child = children[0]
        if any(p.is_dir() and is_gesture_dir_name(p.name) for p in child.iterdir()):
            return child
    return input_root


def is_flat_training_style(root: Path) -> bool:
    root = root.resolve()
    return all((root / name).is_dir() for name in ("annotation", "depth", "rgb"))


def is_nested_clip_style(root: Path) -> bool:
    root = root.resolve()
    gesture_dirs = [p for p in root.iterdir() if p.is_dir() and is_gesture_dir_name(p.name)]
    if not gesture_dirs:
        return False

    for gesture_dir in gesture_dirs:
        clip_dirs = [p for p in gesture_dir.iterdir() if p.is_dir() and p.name.startswith("clip")]
        if not clip_dirs:
            return False
        sample_clip = clip_dirs[0]
        if not all((sample_clip / name).is_dir() for name in ("annotation", "depth", "rgb")):
            return False
    return True


def image_stem_map(folder: Path) -> dict[str, Path]:
    if not folder.is_dir():
        return {}
    return {
        path.stem: path
        for path in folder.iterdir()
        if path.is_file() and path.suffix.lower() in IMAGE_EXTS
    }


def ensure_output_dirs(output_root: Path, gesture_codes: list[str]) -> None:
    output_root.mkdir(parents=True, exist_ok=True)
    for modality in ("annotation", "depth", "rgb"):
        modality_root = output_root / modality
        if modality_root.exists():
            shutil.rmtree(modality_root)
        for code in gesture_codes:
            (modality_root / code).mkdir(parents=True, exist_ok=True)


def copy_nested_to_flat(source_root: Path, output_root: Path) -> None:
    source_root = source_root.resolve()
    output_root = output_root.resolve()

    gesture_dirs = sorted(
        [p.resolve() for p in source_root.iterdir() if p.is_dir() and is_gesture_dir_name(p.name)],
        key=lambda p: p.name,
    )
    codes = sorted({gesture_code(p.name) for p in gesture_dirs})
    ensure_output_dirs(output_root, codes)

    counters = {code: 1 for code in codes}
    total_copied = 0

    for gesture_dir in gesture_dirs:
        if not gesture_dir.exists():
            print(f"Skip missing gesture dir: {gesture_dir}")
            continue

        code = gesture_code(gesture_dir.name)
        clip_dirs = sorted(
            [p.resolve() for p in gesture_dir.iterdir() if p.is_dir() and p.name.startswith("clip")],
            key=lambda p: p.name,
        )

        for clip_dir in clip_dirs:
            ann_map = image_stem_map(clip_dir / "annotation")
            depth_map = image_stem_map(clip_dir / "depth")
            rgb_map = image_stem_map(clip_dir / "rgb")

            common_stems = sorted(set(ann_map) & set(depth_map) & set(rgb_map))
            if not common_stems:
                print(f"Skip {gesture_dir.name}/{clip_dir.name}: no matched frames across annotation/depth/rgb")
                continue

            copied_for_clip = 0
            for stem in common_stems:
                filename = f"{counters[code]:04d}.png"
                shutil.copy2(ann_map[stem], output_root / "annotation" / code / filename)
                shutil.copy2(depth_map[stem], output_root / "depth" / code / filename)
                shutil.copy2(rgb_map[stem], output_root / "rgb" / code / filename)
                counters[code] += 1
                copied_for_clip += 1
                total_copied += 1

            print(f"{gesture_dir.name}/{clip_dir.name}: copied {copied_for_clip} matched frames")

    print("-" * 80)
    print("Original nested input folders were kept.")
    print(f"Output root     : {output_root}")
    print(f"Total triplets  : {total_copied}")
    for code in codes:
        print(f"{code}: {counters[code] - 1}")


def main() -> None:
    input_root = INPUT_ROOT.resolve()
    output_root = OUTPUT_ROOT.resolve()

    if not input_root.exists():
        raise FileNotFoundError(
            f"Input root not found: {input_root}\n"
            "Update INPUT_ROOT in src/temp.py to the dataset folder you want to convert."
        )

    source_root = find_payload_root(input_root)
    print(f"Input root   : {input_root}")
    print(f"Source root  : {source_root}")
    print(f"Output root  : {output_root}")

    if is_flat_training_style(source_root):
        print("Detected flat training dataset format already.")
        print("No conversion needed.")
        return

    if is_nested_clip_style(source_root):
        print("Detected nested clip dataset format.")
        copy_nested_to_flat(source_root, output_root)
        return

    raise RuntimeError(
        "Unrecognized dataset structure.\n"
        "Expected either:\n"
        "1. annotation/depth/rgb/G0X style, or\n"
        "2. G0X_xxx/clipxx/annotation|depth|rgb style."
    )


if __name__ == "__main__":
    main()

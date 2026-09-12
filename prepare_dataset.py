from pathlib import Path
import json

# KaggleHub dataset location
DATASET_ROOT = Path(
    r"C:\Users\omaka\.cache\kagglehub\datasets\gti-upm\leapgestrecog\versions\1\leapGestRecog"
)

# Subject-wise split
TRAIN_SUBJECTS = [f"{i:02d}" for i in range(8)]  # 00-07
VAL_SUBJECTS = ["08"]
TEST_SUBJECTS = ["09"]

# Output metadata
OUTPUT_DIR = Path("dataset")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# Expected gesture classes
CLASS_NAMES = [
    "01_palm",
    "02_l",
    "03_fist",
    "04_fist_moved",
    "05_thumb",
    "06_index",
    "07_ok",
    "08_palm_moved",
    "09_c",
    "10_down",
]

def collect_files(subjects):
    records = []

    for subject in subjects:
        subject_dir = DATASET_ROOT / subject

        if not subject_dir.exists():
            print(f"WARNING: Subject folder not found: {subject_dir}")
            continue

        for class_name in CLASS_NAMES:
            class_dir = subject_dir / class_name

            if not class_dir.exists():
                print(f"WARNING: Class folder not found: {class_dir}")
                continue

            for image_path in sorted(class_dir.glob("*.png")):
                records.append({
                    "path": str(image_path),
                    "subject": subject,
                    "class": class_name,
                })

    return records


train_records = collect_files(TRAIN_SUBJECTS)
val_records = collect_files(VAL_SUBJECTS)
test_records = collect_files(TEST_SUBJECTS)

# Save metadata
metadata = {
    "dataset": "LeapGestRecog",
    "dataset_root": str(DATASET_ROOT),
    "class_names": CLASS_NAMES,
    "split_strategy": "Subject-wise split",
    "train_subjects": TRAIN_SUBJECTS,
    "validation_subjects": VAL_SUBJECTS,
    "test_subjects": TEST_SUBJECTS,
    "train_images": len(train_records),
    "validation_images": len(val_records),
    "test_images": len(test_records),
}

with open(OUTPUT_DIR / "metadata.json", "w", encoding="utf-8") as f:
    json.dump(metadata, f, indent=4)

with open(OUTPUT_DIR / "train.json", "w", encoding="utf-8") as f:
    json.dump(train_records, f, indent=2)

with open(OUTPUT_DIR / "validation.json", "w", encoding="utf-8") as f:
    json.dump(val_records, f, indent=2)

with open(OUTPUT_DIR / "test.json", "w", encoding="utf-8") as f:
    json.dump(test_records, f, indent=2)

print("\nDataset preparation completed!")
print(f"Training images   : {len(train_records)}")
print(f"Validation images : {len(val_records)}")
print(f"Test images       : {len(test_records)}")
print(f"Total images      : {len(train_records) + len(val_records) + len(test_records)}")
print("\nSubjects:")
print(f"Train: {TRAIN_SUBJECTS}")
print(f"Val  : {VAL_SUBJECTS}")
print(f"Test : {TEST_SUBJECTS}")
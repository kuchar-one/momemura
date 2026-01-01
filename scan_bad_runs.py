import pickle
from pathlib import Path
import shutil

deprecated_dir = Path("/cluster/home/kuchar/code/momemura/output/_deprecated")
bad_dir = Path("/cluster/home/kuchar/code/momemura/output/_evenmoredeprecated")

bad_dir.mkdir(exist_ok=True)

print(f"Scanning {deprecated_dir}...")

for run_dir in deprecated_dir.iterdir():
    if not run_dir.is_dir():
        continue

    chk_path = run_dir / "checkpoint_latest.pkl"
    if not chk_path.exists():
        chk_path = run_dir / "results.pkl"
        if not chk_path.exists():
            continue

    try:
        with open(chk_path, "rb") as f:
            data = pickle.load(f)

        # Check for 'repertoire'
        repertoire = data.get("repertoire")
        if repertoire:
            # Repertoire.genotypes could be a list or array
            genotypes = repertoire.genotypes
            if hasattr(genotypes, "shape"):
                shape = genotypes.shape
                # We want explicitly 161
                if len(shape) >= 2 and shape[-1] == 161:
                    pass  # Good
                else:
                    print(f"FOUND NON-161 RUN: {run_dir.name} | Shape: {shape}")
                    shutil.move(str(run_dir), str(bad_dir / run_dir.name))
                    print(f"Moved to {bad_dir}")
            else:
                print(f"Genotypes has no shape in {run_dir.name}")
        else:
            print(f"No repertoire in {run_dir.name} - moving just in case?")
            # shutil.move(str(run_dir), str(bad_dir / run_dir.name))

    except Exception as e:
        print(f"Error loading {run_dir.name}: {e} - moving safely")
        try:
            shutil.move(str(run_dir), str(bad_dir / run_dir.name))
        except Exception:
            pass

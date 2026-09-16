"""Read TB scalars from v3.1 training run."""
import sys, glob, time
from pathlib import Path

LOG = "/home/jovyan/dsuhoi/weather_time_interpolation/logs/exp_hier_v31_split4lvl_6yr"
event_files = sorted(glob.glob(f"{LOG}/version_*/events.out.tfevents.*"),
                     key=lambda p: Path(p).stat().st_mtime, reverse=True)
if not event_files:
    print("no event files")
    sys.exit(0)
print(f"Found {len(event_files)} event files, using latest: {event_files[0]}")
print(f"  mtime: {time.ctime(Path(event_files[0]).stat().st_mtime)}")
print(f"  size:  {Path(event_files[0]).stat().st_size} bytes")

from tensorboard.backend.event_processing.event_accumulator import EventAccumulator
ea = EventAccumulator(event_files[0], size_guidance={'scalars': 0})
ea.Reload()
tags = ea.Tags()['scalars']
print(f"\nScalar tags ({len(tags)}):")
for t in tags:
    pts = ea.Scalars(t)
    if pts:
        first = pts[0]; last = pts[-1]
        print(f"  {t:<30s}  n={len(pts):<5d}  first(step={first.step})={first.value:.4f}  "
              f"last(step={last.step})={last.value:.4f}")
print(f"\nWall-time elapsed: {(time.time() - Path(event_files[0]).stat().st_ctime)/60:.1f} min")

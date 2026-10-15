# Maintenance notes

## Folder migration

The previous flat Python modules now live in `traffic_optimizer/`, grouped by responsibility. Root launchers execute dashboard modules on each Streamlit rerun. Imports must use the full package path.

Model files moved from the root into `models/`. Local clips moved from the enclosing Downloads folder into `videos/` with direction names. The optional accident clip still takes precedence for West. The original Jetson folder is preserved under the enclosing workspace's `archive/`; it is not part of this Git repository. No original backup files were deleted.

New logs and snapshots go into `runtime/`. Existing backup logs were left in the archive. Runtime data, videos, secrets, caches, and TensorRT engines are ignored by Git.

## Verified fixes

- Execute Streamlit dashboard code again on reruns instead of relying on a cached import.
- Determine emergency phase geometry from both road endpoints, including the destination.
- Bound the logger's in-memory event list and atomically replace JSON output.
- Match the optional QAOA implementation to a V1 sampler and constrain its dependency family.
- Separate simulator dependencies from optional detection, SMS, and QAOA installations.

## Known limits

The live UI uses rule-based timing, although optimization helpers are available separately. Turning emergency routes can express competing phase preferences, and emergency preferences are soft costs. QAOA availability is checked at runtime. Video detection accuracy, accident recall, and calibrated speed accuracy require a labeled evaluation dataset. The inherited model licensing/provenance should be confirmed before redistribution outside the existing project.

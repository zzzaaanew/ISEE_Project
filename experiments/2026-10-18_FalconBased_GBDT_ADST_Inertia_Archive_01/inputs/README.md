# External inputs and replay limits

This is a compact source/evidence archive, not a redistribution of AcmeTrace.

- Raw files: telemetry_5m_{util,temp,power,fb}.parquet and xid_onsets_metadata.parquet.
- Original raw identities and upstream repository revision: data_inventory.json.
- Falcon45 engine cache: grid_meta.npz plus telemetry_{metric}_5m.npy and telemetry_{metric}_node_5m.npy (see cache_inventory.json).
- Independently verified Parquet cache: {metric}.mean.npy with {metric}.identity.json; the Falcon143 engine also uses observed-count caches.
- The 64 E19 reference score parts supply timestamp scheduling only to the training runners; baseline learned scores are used only for comparison.
- Omitted full score tapes, Top-100 files and learned checkpoints have SHA256 and size entries in omitted_artifacts.json. Their absence means the full artifact auditors cannot run on this compact archive alone.

The copied config.json and execution contracts retain the exact historical machine paths. Edit a new config file for another machine. Install the pinned dependencies, supply the inputs/caches/reference schedule, and run the original ML entry point with that new config. Fresh runs generate new execution contracts; do not reuse historical resume state or supervisor PIDs/policies for a new training run.

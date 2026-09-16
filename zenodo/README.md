# Archiving a release

The GitHub repository is
[dsuhoi/weatherbridge](https://github.com/dsuhoi/weatherbridge).
No Zenodo DOI has been assigned to this release.

Build the source archive from a clean commit:

```bash
SOURCE_ONLY=1 bash zenodo/build_release.sh
python zenodo/validate_release.py --source-only
```

Build the standalone model archive from `weatherbridge-release/` after
downloading the weights. It includes normalization, static fields and
a single ERA5 test window.

For a Zenodo deposit, create software records with the metadata in
`code_metadata.json` and `weatherbridge-release/zenodo_metadata.json`.
Keep the Apache-2.0 source and MIT model-package licenses separate, including
their third-party exceptions. Add assigned DOIs to the citation metadata
and manuscript only after the records exist.

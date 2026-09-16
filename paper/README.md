# Manuscript

From the repository root:

```bash
bash paper/build_npj_submission.sh
```

The build requires TeX Live with latexmk and Poppler. The Springer Nature
class is supplied here. The script checks references, text overflow and
embedded fonts, and writes `main.pdf`, `manuscript_npj.pdf` and
`supplementary.pdf`.

Figure sources are in `figs/`; generated illustrations are in `images/`.
Plotting commands are indexed in `repro/paper_experiments.json`.

`FINAL_PUBLICATION=1` adds figure-size and provenance checks. These can reject
an incomplete export even when the ordinary manuscript build succeeds.

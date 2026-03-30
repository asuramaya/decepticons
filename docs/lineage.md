# Lineage and Attribution

This repository is meant to stand on its own.

Its code, examples, and tests live here and should run from this repo without depending on sibling workspace paths or
temporary directories.

## What Is In This Repo

- the reusable kernel in `src/open_predictive_coder/`
- project-layer descendant reconstructions under `examples/projects/`
- docs that explain the extraction boundary, research anchors, and downstream pattern language

The example descendants in this repo are original reconstructions built from the kernel primitives. They are not
vendored copies of sibling repositories.

## Upstream Lineage

The design language comes from the broader `carving_machine` workspace line.

In this repo, references such as `carving_machine/models.py`, `carving_machine/reservoir.py`, `ablations.py`, or
`v6.py` are attribution coordinates for the upstream workspace lineage. Those files are not vendored here, so repo
docs should cite them as plain paths rather than dead filesystem links.

## Related Descendants

These descendant families informed the boundary and naming, but they are not part of this repository:

- `conker`
- `blinx`
- `giddy-up`

When this repo mentions those names, it is using them as attribution for the broader problem family, not as imported
code dependencies.

## Public External Reference

The public downstream reference for the byte-latent branch is:

- [`guilhhotina/brelt`](https://github.com/guilhhotina/brelt)

The `brelt_like` example in this repo is a project-layer reconstruction shaped by that public repository. It is not a
checkout in `/tmp`, and it should not rely on any external local clone to make sense.

## Citation Rule In This Repo

- repo-local files should use relative links
- non-vendored workspace lineage should be cited as plain code paths
- public external descendants can use normal HTTPS links

That rule keeps the repository readable on GitHub and avoids dead local links.

# Yggdrasil

Generating megastructure 3D models to be build in minecraft.

## Layout

```
utils/          shared utilities (atlas, structure format, visualization)
islands/        floating island generation
islands/out/    generated models + screenshots (gitignored)
```

## Model format

A model is:

- a **3D numpy array** of integer block indices (`int16`), axes `(X, Y, Z)`
  with Y up; index `0` is air
- an **atlas** mapping each integer index to a block name
  (e.g. `1 -> stone`)

On disk it's a single `.npz` file containing `data` (the voxel array) and
`atlas` (a JSON legend).

Current island palette: `stone` (lower part), `dirt`, and `grass`
(grass is a dirt block with no other block on top).

## Setup

```sh
uv venv .venv
uv pip install --python .venv/bin/python -r requirements.txt
```

## Run

```sh
.venv/bin/python islands/generate.py
```

Writes `islands/out/island.npz` (the model) and `islands/out/island.png`
(a screenshot for inspection).

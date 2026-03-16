# STELLA Docker Image Package

The Docker image package can be downloaded from [Google Drive](https://drive.google.com/file/d/1iN9AOJpi0FBDz7i_gjmW8EbcjTY0z8bF/view). This folder documents how to use the exported package after download.

## Package

- File name: `stella_4images_20260313.tar.zst`
- Compressed size: ~8.5 GiB
- Estimated local Docker storage after import: typically 20+ GiB (depends on existing layer reuse on your machine).
- Contains 4 image tags:
  - `stella:baseline_20260313`
  - `stella:ablate_toolon_templateon_20260313`
  - `stella:ablate_tooloff_templateon_20260313`
  - `stella:ablate_tooloff_templateoff_20260313`

Meaning of these tags:

- `stella:baseline_20260313`: default STELLA baseline image used in the main run setting.
- `stella:ablate_toolon_templateon_20260313`: ablation image with `tool_creation=on`, `template_reuse=on`.
- `stella:ablate_tooloff_templateon_20260313`: ablation image with `tool_creation=off`, `template_reuse=on`.
- `stella:ablate_tooloff_templateoff_20260313`: ablation image with `tool_creation=off`, `template_reuse=off`.

Note: this package does not include a separate `tool_creation=on`, `template_reuse=off` image tag because that condition was executed via runtime/config control in our runs rather than a separately distributed image tag.

## Prerequisites

- Linux/macOS with Docker installed and running (`docker --version` should work).
- Recommended free disk space: at least 40 GiB before loading images.
- `zstd` installed (required for `.tar.zst` package import).
- `biomlbench` available in your environment if you want to run benchmark tasks.

## Platform Notes

- Tested target platform: `linux/amd64`.
- On Apple Silicon (`arm64`), Docker may use emulation for `linux/amd64` images.
- If needed, use explicit platform in runtime commands, e.g. `--platform linux/amd64`.

## Download from Drive

The Docker image package can be downloaded from:

- **Google Drive:** https://drive.google.com/file/d/1iN9AOJpi0FBDz7i_gjmW8EbcjTY0z8bF/view

Open the link above and download `stella_4images_20260313.tar.zst`.
Recommended: keep the filename unchanged.

## Load Images into Docker

### Option A (recommended)

```bash
zstd -dc stella_4images_20260313.tar.zst | docker load
```

### Option B (decompress first, then load)

```bash
zstd -d stella_4images_20260313.tar.zst -o stella_4images_20260313.tar
docker load -i stella_4images_20260313.tar
```

Expected result: Docker prints `Loaded image: ...` for each tag.

## Verify Imported Tags

```bash
docker images --format "{{.Repository}}:{{.Tag}}" | grep "^stella:"
```

Expected tags include:

- `stella:baseline_20260313`
- `stella:ablate_toolon_templateon_20260313`
- `stella:ablate_tooloff_templateon_20260313`
- `stella:ablate_tooloff_templateoff_20260313`

## Minimal Reproducibility Mapping

Use the tags/flags below to reproduce the core conditions:

- Baseline: `stella:baseline_20260313`
- Ablation (`tool_creation=on`, `template_reuse=on`): `stella:ablate_toolon_templateon_20260313`
- Ablation (`tool_creation=off`, `template_reuse=on`): `stella:ablate_tooloff_templateon_20260313`
- Ablation (`tool_creation=off`, `template_reuse=off`): `stella:ablate_tooloff_templateoff_20260313`
- Ablation (`tool_creation=on`, `template_reuse=off`): run using baseline image plus runtime/config switch (not a separate image in this package).

Practical mapping (image + agent ID):

| Condition | Docker tag | Agent ID |
|---|---|---|
| Baseline (default) | `stella:baseline_20260313` | `stella` |
| `tool_creation=on`, `template_reuse=on` | `stella:ablate_toolon_templateon_20260313` | `stella_toolon_templateon` |
| `tool_creation=off`, `template_reuse=on` | `stella:ablate_tooloff_templateon_20260313` | `stella_tooloff_templateon` |
| `tool_creation=off`, `template_reuse=off` | `stella:ablate_tooloff_templateoff_20260313` | `stella_tooloff_templateoff` |
| `tool_creation=on`, `template_reuse=off` | baseline image + runtime switch | `stella_toolon_templateoff` |

## Running the benchmark

This Docker package provides the images only. To run tasks, please use the benchmark instructions in [`Tool_Creation_Benchmark/README.md`](../Tool_Creation_Benchmark/README.md).

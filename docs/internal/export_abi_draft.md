# OPC Export ABI

This document is the first draft of the stable export contract between `decepticons` and descendant systems such as `Chronohorn`.

The goal is simple:

- `opc` defines the reusable Python kernel
- descendant systems train or search in Python
- Chronohorn Rust consumes a versioned export without needing Python-specific checkpoint assumptions

The ABI is manifest-first. Every export must include a small deterministic manifest, plus one or more payload blobs.

## ABI Name

The ABI name is `opc-export`.

The first stable line is `opc-export/1.0.0`.

## Export Shape

An export is a directory, tarball, or object bundle with at least one manifest and one learned payload.

Required top-level entries:

- `manifest`
- `learned_state`
- `checksums`

Optional top-level entries:

- `packed_memory`
- `notes`
- `debug`

Chronohorn Rust should treat the manifest as authoritative and the payloads as verified attachments.

## Required Manifest Fields

The manifest must contain these fields.

| Field | Type | Required | Meaning |
|---|---:|---:|---|
| `abi_name` | string | yes | Must be `opc-export` |
| `abi_version` | string | yes | Semantic version of the ABI, starting at `1.0.0` |
| `exporter_version` | string | yes | Version of the exporter that wrote the artifact |
| `exported_utc` | string | yes | UTC timestamp in RFC 3339 format |
| `model_family_id` | string | yes | Stable family label, for example `causal-bank` |
| `model_variant_id` | string | yes | Concrete variant label, for example `window4_scale18_routed_e8` |
| `kernel_version` | string | yes | Version of the OPC kernel contract used to build deterministic pieces |
| `tokenizer_id` | string | yes | Tokenizer family and variant identifier |
| `data_root_id` | string | yes | Dataset/root identifier used for training or export |
| `deterministic_substrate` | object | yes | Code-derived structure that can be regenerated exactly |
| `learned_state` | object | yes | Learned tensor payload description and names |
| `checksums` | object | yes | Hashes for all payload blobs and the canonical manifest |
| `artifact_role` | string | yes | Intended use such as `replay`, `eval`, or `packed_residual` |

Recommended additional manifest fields:

- `source_commit`
- `train_step`
- `train_wallclock_s`
- `sequence_length`
- `vocab_size`
- `dtype_policy`
- `quantization_policy`
- `export_notes`

## Deterministic vs Learned Separation

The ABI must separate state into two categories.

### Deterministic State

Deterministic state is everything that can be rebuilt exactly from code plus configuration.

Examples:

- substrate topology
- number of layers
- readout wiring shape
- routing/expert counts
- tokenizer identity
- fixed hash tables or code-derived basis choices
- any preset or factory configuration that does not depend on training noise

Deterministic state must be stored as structured configuration, not as opaque learned tensors.

### Learned State

Learned state is anything produced by optimization.

Examples:

- trainable weight matrices
- trainable biases
- trainable scalars
- learned readout matrices
- learned routing weights
- learned packed-memory coefficients

Learned state must be listed by stable tensor name, shape, dtype, and payload checksum.

### Hard Rule

If a field can be regenerated exactly from source code plus the manifest, it is deterministic.

If a field changes with seed, optimizer step, or training schedule, it is learned.

Chronohorn Rust should never need to infer that distinction from tensor statistics.

## Manifest Subsections

### `deterministic_substrate`

This object describes the exact architecture shape that Rust must reconstruct.

Required fields inside `deterministic_substrate`:

- `substrate_family`
- `layer_count`
- `hidden_size`
- `readout_kind`
- `readout_shape`
- `routing_kind`
- `routing_shape`
- `activation_kind`
- `memory_kind`
- `feature_view_kind`

Optional fields inside `deterministic_substrate`:

- `local_window`
- `oscillatory_schedule`
- `half_life_policy`
- `expert_count`
- `top_k`
- `normalization_policy`
- `gate_policy`

### `learned_state`

This object describes the tensors Chronohorn must load.

Required fields inside `learned_state`:

- `tensor_format`
- `tensor_count`
- `tensor_index`

Each tensor entry in `tensor_index` must include:

- `name`
- `shape`
- `dtype`
- `storage`
- `checksum`

Allowed `storage` values for v1:

- `inline`
- `blob_ref`

The manifest may not rename tensors implicitly. Tensor names are part of the ABI.

## Optional Packed-Memory Attachment

Packed memory is optional and is treated as an attachment, not as part of the base learned state.

Use it only when the export includes a residual memory or table-backed augmentation that Chronohorn Rust can apply at runtime.

Required `packed_memory` fields when present:

- `packed_memory.kind`
- `packed_memory.version`
- `packed_memory.byte_budget`
- `packed_memory.layout`
- `packed_memory.blob_ref`
- `packed_memory.checksum`

Recommended `packed_memory` fields:

- `packed_memory.order_budget`
- `packed_memory.row_count`
- `packed_memory.support_policy`
- `packed_memory.rank_policy`

Rules:

- If `packed_memory` is absent, the export is still valid.
- If `packed_memory` is present, Chronohorn Rust must verify its checksum before attaching it.
- Packed memory must not silently change the meaning of the base deterministic substrate.
- Packed memory is an additive residual, not a replacement for the learned state.

## Checksums

The manifest must include canonical checksums for:

- the manifest itself
- every learned-state blob
- every packed-memory blob, if present

Recommended hash algorithm:

- `blake3`

Chronohorn Rust should reject any export whose checksum does not match the manifest.

## Versioning Rules

The ABI uses semantic versioning.

### Major Version

Increment the major version when a change is not backward compatible.

Examples:

- required field removal
- required field rename
- tensor name contract change
- payload layout change that old Chronohorn loaders cannot parse
- meaning change for a field that existing exports rely on

Chronohorn Rust must reject incompatible major versions by default.

### Minor Version

Increment the minor version when adding backward-compatible structure.

Examples:

- new optional manifest field
- new optional packed-memory attachment field
- new optional payload kind

Chronohorn Rust should ignore unknown optional fields and continue when the major version matches.

### Patch Version

Increment the patch version for documentation fixes, validation improvements, or non-contract clarifications that do not change parsing behavior.

### Compatibility Rule

Chronohorn Rust should use the following rule:

- accept same-major, any-minor-or-patch exports if all required fields parse
- reject different-major exports unless an explicit compatibility layer exists

## Chronohorn Rust Consumption Contract

Chronohorn Rust should consume this ABI in a fixed order.

1. Read the manifest first.
2. Validate `abi_name` and `abi_version`.
3. Validate required fields and reject missing entries.
4. Rebuild the deterministic substrate from `deterministic_substrate`.
5. Load learned tensors by exact tensor name.
6. Verify all checksums.
7. If `packed_memory` is present, verify and attach it as an optional residual.
8. Run a short parity probe before full replay.

Chronohorn Rust must not:

- infer architecture from payload shapes alone
- accept unnamed tensors
- guess at a missing deterministic setting
- silently coerce unsupported dtypes
- silently ignore checksum failures

Chronohorn Rust may:

- map exported tensor names to internal Rust parameter names
- provide explicit compatibility shims for same-major versions
- refuse optional attachments that it does not know how to execute

## Canonical Failure Modes

Chronohorn Rust should fail closed on:

- unknown `abi_name`
- unsupported major version
- missing required field
- checksum mismatch
- tensor shape mismatch
- unsupported dtype
- packed-memory attachment mismatch
- deterministic substrate rebuild mismatch

## Minimal Example

```json
{
  "manifest": {
    "abi_name": "opc-export",
    "abi_version": "1.0.0",
    "exporter_version": "chronohorn-export-0.1",
    "exported_utc": "2026-03-30T18:00:00Z",
    "model_family_id": "causal-bank",
    "model_variant_id": "window4_scale18_routed_e8",
    "kernel_version": "opc-kernel-0.8",
    "tokenizer_id": "fineweb-bpe-8k",
    "data_root_id": "fineweb-2026-03",
    "artifact_role": "replay",
    "deterministic_substrate": {
      "substrate_family": "causal-bank",
      "layer_count": 18,
      "hidden_size": 512,
      "readout_kind": "routed_sqrelu",
      "readout_shape": {
        "experts": 8,
        "expert_width": 287,
        "top_k": 2
      },
      "routing_kind": "softmax_router",
      "routing_shape": {
        "input_dim": 512,
        "expert_count": 8
      },
      "activation_kind": "sqrelu"
    },
    "learned_state": {
      "tensor_format": "dense",
      "tensor_count": 42,
      "tensor_index": [
        {
          "name": "readout.router.weight",
          "shape": [8, 512],
          "dtype": "f32",
          "storage": "blob_ref",
          "checksum": "blake3:..."
        }
      ]
    },
    "checksums": {
      "manifest": "blake3:...",
      "tensors": "blake3:...",
      "packed_memory": null
    }
  }
}
```

## Open Questions For v1.1

- whether the manifest should be JSON, CBOR, or a canonical binary encoding
- whether tensor blobs should be raw, `.npz`, or a custom packed layout
- whether Chronohorn Rust should allow multiple learned-state shards
- whether packed memory should support more than one attachment kind

For now, the first draft rule is:

- one ABI name
- one semver line
- one manifest-first loader
- deterministic state separated from learned state
- packed memory optional and additive
- Chronohorn Rust fails closed on ambiguity

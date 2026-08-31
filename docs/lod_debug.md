# LoD debugging

The viewer exposes a compact `Gaussian LoD` panel per named cloud. It contains
the detail slider, camera-freeze toggle, named debug preset, lifecycle, selected
record count, quality outcome, residency, and typed failure state. The same
presets are available from the CLI. `LodDebugSettings` contains only that
preset; palettes and boundary styling are fixed so scenes and screenshots do
not depend on hidden normalization or tuning controls.

## Presets

| Preset | Meaning |
|---|---|
| `off` | Authored Gaussian color |
| `level` | Stable color for the selected hierarchy level |
| `page` | Stable color for the physical page containing each record |
| `residency` | Direct resident data versus ancestor fallback |
| `boundaries` | Support-aware logical chunk boundaries |
| `selection-pressure` | Effective selector pressure; `<=1` meets the target |

Example:

```bash
cargo run --release --bin bevy_gaussian_splatting -- \
  --input-scene=https://mitchell.mosure.me/trellis.glb \
  --lod-quality=0.75 --lod-debug=selection-pressure
```

Quality `1` renders the exact flat source for ordinary cloud inputs, which
intentionally owns no hierarchy metadata. A prebuilt LoD package instead keeps
its exact leaf frontier and can still provide hierarchy presets at quality one.
The panel reports `metadata ready` without claiming that an adapter-specific
pipeline has already drawn the annotation.

## Runtime cost and adapter limits

Prebuilt packages build debug records lazily from already validated decoded
pages. CPU preparation is capped at 32,768 records per package update, and the
render path writes only changed physical slots, capped at 64 MiB and 256 slots
per cloud per frame. Switching among enabled presets updates only the 32-byte
configuration uniform; it does not rebuild the package, replace the active cut,
or re-upload the full metadata address space. Pending cut annotations are
prepared outside the visible epoch and become current with the logical cut.
Level and Page metadata is pre-uploaded before a replacement may activate, so
camera motion does not briefly fall back to authored color. Residency provenance
travels with each compacted candidate in two packed entry bits; it therefore
switches atomically with the exact per-view cut without another record upload.

The GPU binding still reserves one fixed-address record buffer sized for the
configured physical atlas. A low-limit adapter that cannot bind that buffer
falls back to authored color and logs a warning. `metadata ready` describes the
main-world annotation state, not adapter capability; a render-to-main capability
status is not yet exposed in the panel. The native Garden qualification and its
bounded sparse-upload evidence are recorded in
[the Garden report](lod_garden_report.md).

## Reading selection pressure

Selection pressure is the most useful view for quality diagnosis. It mirrors
the CPU selector and includes structural demand, projected pixel error, and the
builder-authored high-fidelity certificate. Certificate pressure is inactive
through quality `.90` and ramps to full authority at `.95`; projected-error
authority uses the same cubic curve as CPU selection. Cool colors are
comfortably below the target, green is near the boundary, and warm colors
exceed it. A pressure above one may be visible transiently inside hysteresis or
when the status reports a residency/budget degradation.

Raw geometric, appearance, opacity, and combined-error ramps are deliberately
not presets: those values use different units and are not the user-facing
quality contract.

## Freeze workflow

To inspect what a distant camera actually selected:

1. Move to the distance of interest.
2. Press `F` or enable `Freeze camera selection`.
3. Move closer without changing the captured selection camera.
4. Compare `level`, `page`, and `selection-pressure`.
5. Unfreeze to resume camera-driven selection.

Freezing captures only the selection view. Streaming and residency continue to
converge, so a frozen cut can still replace an ancestor fallback with its
requested page.

## Interpreting other modes

- `level` shows logical hierarchy depth and is the quickest way to see a mixed
  cut across the object.
- `page` shows physical packing. Multiple logical node slices may share one
  page, so page color is not a quality level. Its hashed hues are normalized
  to a fixed `0.30` Rec.709 linear luminance; light/dark bands therefore do not
  masquerade as page-local opacity or density changes.
- `residency` is most informative on prebuilt streamed packages. A fully
  resident steady cut is expected to look uniform. Its provenance is owned by
  the per-view candidate, so different cameras can classify the same physical
  page independently without racing a cloud-wide annotation update.
- The panel's `Cached pages` value is decoded bounded-cache occupancy,
  including the permanent coarse guard and warm completed pages. It is not a
  visible-page, selected-frontier, or GPU-current count, and can rise after
  camera motion stops while work already in flight finishes before settling.
- `boundaries` marks logical chunk support. It is not a source-geometry leakage
  detector; use selection pressure and the morphology/PSNR qualification for
  elongated-representative regressions.

The canonical Trellis report and its regression thresholds are documented in
[lod_quality_report.md](lod_quality_report.md).

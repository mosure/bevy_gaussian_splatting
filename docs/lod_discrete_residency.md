# Discrete package residency planning

CPU-selected bounded-refinement packages in explicit `Discrete` mode compute a complete
camera-dependent destination and publish complete categorical transactions.
Every ordinary step replaces a parent with all immediate children, or complete
siblings with their parent. Current pages stay retained through upload and GPU
publication. Independently ready branches can advance without waiting for a
whole deep destination.

Physical admission deduplicates pages across views, permanent roots, retained
output and admitted replacement cohorts. A cohort reserves pages only after its
logical active/change budget passes. Wave priority rotates between views.
Refinement prioritizes conservative visibility, selection pressure and projected
error; equal bounds use camera-to-center distance before stable node identity.
Distance only schedules work and does not change quality/error acceptance.

If all pages on the ideal cut's navigation paths fit the page/byte/record limits,
the ideal needs no additional replacement reserve. This includes a hierarchy
whose nodes all share one already-resident page. Otherwise the planner contracts
complete low-priority sibling groups until the destination plus a measured
immediate-cohort reserve fits. This reserve is conservative: it uses the largest
cohort on the original ideal paths, rather than recomputing exact marginal
cohort cost after every contraction. Shared-page contraction is not a globally
optimal page/quality allocator.

A packed complete destination can be cheaper than its intermediate path. At an
adjacent-capacity stall, the planner admits that complete destination directly
when the exact retained-output + destination + root union fits. Existing decode,
materialization and upload limits still govern its preparation before atomic
publication. If even that union does not fit, already-resident complete roots
provide an escape from an inherited fully pinned cut. A lowered active cap also
uses a resident root recovery so the replacement obeys renderer admission.

The constrained destination is cached by exact camera, policy, view membership
and physical limits. Reaching it can settle orchestration while quality remains
`Residency` or `ActiveBudget`; it never implies original quality. Camera/policy
changes invalidate the plan. The reserve, greedy contraction and distance tie
break require measured rate/quality and multi-view qualification beyond the
CPU coverage/progress regressions.

## GPU selection during camera movement

With `--lod-gpu-traversal` and either global quad ordering or Gaussian point
splatting, the displayed cut is selected by compute from the current extracted
camera. CPU page demand does not publish a camera-dependent displayed cut.
Each frame traverses the resident hierarchy, selects complete child cohorts
within the active bounds, and expands physical indices for that frame's shared
renderer. A missing child retains its complete resident parent.

Residency publication and camera selection have separate responsibilities:

- A replacement residency snapshot becomes usable only after its atlas slots
  are uploaded. Until then, the previous authenticated snapshot is traversed
  using the current camera; the previous camera image is not the fallback.
- Compatible residency updates reuse traversal allocations. In-flight readbacks
  retain their own snapshot identities and page lifetimes.
- Stable queue order and deterministic prefix admission prevent budget races
  from changing a held camera's cut or equal-depth ordering between frames.
- Async page requests may come from an older residency generation of the same
  immutable package. Requests must advance monotonically and contain valid page
  IDs. Display acknowledgements still require the exact rendered generation.
- Full telemetry buffers skip telemetry capture, rather than camera rendering.

This removes feedback-paced display updates for resident data. Disk/network
loads remain asynchronous, and newly available representatives can still change
the image. Discrete cuts do not provide continuous geometric interpolation or
repair low-quality proxies. Deterministic prefix admission is not a global
screen-error optimizer; cold descendant prefetch and representative transition
quality still require separate work and visual qualification.

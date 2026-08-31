use super::*;
use crate::{
    gaussian::formats::{
        lodge::LodgeClusterId,
        planar_3d::PlanarGaussian3d,
        planar_3d_chunked::{LodNodeId, LodPageId},
    },
    render::{lod_debug_sparse_identity_matches, skip_unready_candidate_required_draw},
    stream::{cache::AtlasSlot, lodge::LodgePairIdentity, runtime::LodRuntimeViewId},
};
use bevy::render::sync_world::MainEntity;

fn test_extracted_view() -> ExtractedView {
    ExtractedView {
        retained_view_entity: RetainedViewEntity::new(
            MainEntity::from(Entity::from_bits(1)),
            None,
            0,
        ),
        clip_from_view: Mat4::perspective_infinite_reverse_rh(1.0, 16.0 / 9.0, 0.1),
        world_from_view: GlobalTransform::IDENTITY,
        clip_from_world: None,
        target_format: bevy::render::render_resource::TextureFormat::Rgba8Unorm,
        viewport: UVec4::new(0, 0, 1920, 1080),
        color_grading: bevy::render::view::ColorGrading::default(),
        invert_culling: false,
    }
}

fn test_view_blend_edge_key(id: u64) -> LodViewBlendEdgeKey {
    let metric = LodViewBlendMetricKey {
        center_bits: [id as u32, 0, 0],
        radius_bits: 1.0_f32.to_bits(),
        geometric_error_bits: (id as f32 + 1.0).to_bits(),
        quality_min_bits: 0.0_f32.to_bits(),
        quality_max_bits: 1.0_f32.to_bits(),
        certificate_bits: 0.5_f32.to_bits(),
        original_representation: false,
    };
    LodViewBlendEdgeKey {
        parent: LodNodeId(id),
        children: vec![LodNodeId(id + 1_000)],
        parent_metric: metric,
        child_metrics: vec![metric],
    }
}

#[test]
fn indirect_layout_matches_wgpu_offsets() {
    assert_eq!(
        std::mem::size_of::<LodIndirectArgs>(),
        LOD_INDIRECT_ARGS_SIZE as usize
    );
    assert_eq!(std::mem::offset_of!(LodIndirectArgs, vertex_count), 0);
    assert_eq!(std::mem::offset_of!(LodIndirectArgs, instance_count), 4);
    assert_eq!(
        std::mem::offset_of!(LodIndirectArgs, dispatch_x),
        DISPATCH_A_INDIRECT_OFFSET as usize
    );
    assert_eq!(
        std::mem::offset_of!(LodIndirectArgs, dispatch_c_x),
        DISPATCH_C_INDIRECT_OFFSET as usize
    );
    assert_eq!(std::mem::offset_of!(LodIndirectArgs, candidate_hits), 40);
    assert_eq!(std::mem::size_of::<LodScanRecord>(), 8);
    assert_eq!(std::mem::offset_of!(LodScanRecord, count), 0);
    assert_eq!(std::mem::offset_of!(LodScanRecord, offset), 4);

    let expected = finalized_indirect_args(257, 512, 256, 128);
    let decoded = bytemuck::pod_read_unaligned::<LodIndirectArgs>(bytemuck::bytes_of(&expected));
    assert_eq!(decoded, expected);
    let host = include_str!("../lod.rs");
    assert!(host.contains("size: LOD_INDIRECT_ARGS_SIZE"));
    assert!(host.contains("slice(..LOD_INDIRECT_ARGS_SIZE)"));
}

#[test]
fn candidate_upload_cache_reuses_identical_frontiers_and_tracks_content_changes() {
    let base_range = LodPhysicalRange {
        node: LodNodeId(1),
        page: LodPageId(2),
        slot: AtlasSlot {
            index: 3,
            generation: 4,
        },
        physical_start: 12,
        count: 3,
    };
    let base = lod_candidate_parts_fingerprint(7, &[base_range], 3);
    let first_version = Arc::new(AtomicU8::new(LOD_RENDER_WAITING));
    let mut tracker = LodCandidateUploadTracker::default();
    assert_eq!(
        tracker.plan_fingerprint(&first_version, base),
        LodCandidateUploadPlan::Upload(base)
    );
    tracker.mark_synchronized(&first_version, base);
    assert_eq!(
        tracker.plan_fingerprint(&first_version, base),
        LodCandidateUploadPlan::ReuseVersion
    );

    let identical_version = Arc::new(AtomicU8::new(LOD_RENDER_WAITING));
    assert_eq!(
        tracker.plan_fingerprint(&identical_version, base),
        LodCandidateUploadPlan::ReuseFingerprint(base)
    );
    tracker.mark_synchronized(&identical_version, base);

    let generation_change = lod_candidate_parts_fingerprint(
        7,
        &[LodPhysicalRange {
            slot: AtlasSlot {
                generation: 5,
                ..base_range.slot
            },
            ..base_range
        }],
        3,
    );
    let changed_generation_version = Arc::new(AtomicU8::new(LOD_RENDER_WAITING));
    assert_eq!(
        tracker.plan_fingerprint(&changed_generation_version, generation_change),
        LodCandidateUploadPlan::Upload(generation_change)
    );

    let index_change = lod_candidate_parts_fingerprint(
        7,
        &[LodPhysicalRange {
            physical_start: 13,
            ..base_range
        }],
        3,
    );
    let changed_index_version = Arc::new(AtomicU8::new(LOD_RENDER_WAITING));
    assert_eq!(
        tracker.plan_fingerprint(&changed_index_version, index_change),
        LodCandidateUploadPlan::Upload(index_change)
    );

    let resident = lod_candidate_parts_fingerprint_with_residency(
        7,
        &[base_range],
        3,
        |_| LodDebugResidency::Resident as u32,
        None,
        None,
    );
    let fallback = lod_candidate_parts_fingerprint_with_residency(
        7,
        &[base_range],
        3,
        |_| LodDebugResidency::AncestorFallback as u32,
        None,
        None,
    );
    assert_ne!(
        resident, fallback,
        "same physical payload with new per-view Residency must re-upload and recompute"
    );
}

fn external_candidate_for_fingerprint(
    pair: LodgePairIdentity,
    second_center: [f32; 3],
    classes: Vec<LodgeMembershipClass>,
) -> LodRenderCandidate {
    let ranges = vec![
        LodPhysicalRange {
            node: LodNodeId(1),
            page: LodPageId(1),
            slot: AtlasSlot {
                index: 0,
                generation: 1,
            },
            physical_start: 0,
            count: 2,
        },
        LodPhysicalRange {
            node: LodNodeId(2),
            page: LodPageId(1),
            slot: AtlasSlot {
                index: 1,
                generation: 1,
            },
            physical_start: 2,
            count: 2,
        },
    ];
    let frontier =
        LodCandidateFrontier::complete_external_active_set(LodRuntimeViewId(7), ranges, false)
            .unwrap();
    let presentation =
        LodExternalActiveSetPresentation::new(pair, [0.0, 0.0, 0.0], second_center, classes)
            .unwrap();
    LodRenderCandidate::new_external_active_set(frontier, presentation).unwrap()
}

#[test]
fn external_candidate_fingerprint_covers_pair_centers_and_membership_classes() {
    let pair = LodgePairIdentity {
        first: LodgeClusterId(1),
        second: LodgeClusterId(2),
    };
    let base = external_candidate_for_fingerprint(
        pair,
        [2.0, 0.0, 0.0],
        vec![
            LodgeMembershipClass::Shared,
            LodgeMembershipClass::SecondOnly,
        ],
    );
    let identical = external_candidate_for_fingerprint(
        pair,
        [2.0, 0.0, 0.0],
        vec![
            LodgeMembershipClass::Shared,
            LodgeMembershipClass::SecondOnly,
        ],
    );
    let changed_pair = external_candidate_for_fingerprint(
        LodgePairIdentity {
            first: LodgeClusterId(1),
            second: LodgeClusterId(3),
        },
        [2.0, 0.0, 0.0],
        vec![
            LodgeMembershipClass::Shared,
            LodgeMembershipClass::SecondOnly,
        ],
    );
    let changed_center = external_candidate_for_fingerprint(
        pair,
        [3.0, 0.0, 0.0],
        vec![
            LodgeMembershipClass::Shared,
            LodgeMembershipClass::SecondOnly,
        ],
    );
    let changed_classes = external_candidate_for_fingerprint(
        pair,
        [2.0, 0.0, 0.0],
        vec![
            LodgeMembershipClass::FirstOnly,
            LodgeMembershipClass::SecondOnly,
        ],
    );

    let fingerprint = lod_bridge_candidate_fingerprint(&base);
    assert_eq!(fingerprint, lod_bridge_candidate_fingerprint(&identical));
    assert_ne!(fingerprint, lod_bridge_candidate_fingerprint(&changed_pair));
    assert_ne!(
        fingerprint,
        lod_bridge_candidate_fingerprint(&changed_center)
    );
    assert_ne!(
        fingerprint,
        lod_bridge_candidate_fingerprint(&changed_classes)
    );
}

#[test]
fn radix_drawable_metadata_promotes_only_with_matching_sort_generation() {
    let range = LodPhysicalRange {
        node: LodNodeId(1),
        page: LodPageId(2),
        slot: AtlasSlot {
            index: 3,
            generation: 4,
        },
        physical_start: 12,
        count: 3,
    };
    let first_fingerprint = lod_candidate_parts_fingerprint(7, &[range], 3);
    let second_fingerprint = lod_candidate_parts_fingerprint(
        7,
        &[LodPhysicalRange {
            node: LodNodeId(5),
            physical_start: 15,
            count: 5,
            ..range
        }],
        5,
    );
    let first_phase = Arc::new(AtomicU8::new(LOD_RENDER_ACTIVE));
    let second_phase = Arc::new(AtomicU8::new(LOD_RENDER_PREPARED));
    let morph_identity = LodCandidateFrontier::complete_empty_for_test(
        crate::stream::runtime::LodRuntimeViewId(7),
        &GaussianLodSettings::default(),
    )
    .with_temporal_transition_for_test(LodTemporalTransitionMode::Morphing)
    .temporal_transition()
    .and_then(|transition| transition.morph())
    .map(LodViewBlendBatch::identity);
    let view_blend = morph_identity.map(|identity| LodLastRadixViewBlendForTesting {
        identity,
        edges: Vec::new(),
        weights: Vec::new(),
        endpoints: Vec::new(),
        recovery_lag: Vec::new(),
        invalid_pressure: Vec::new(),
        evaluation_view: None,
        evaluation_target: None,
        desired_evaluation_complete: false,
        upload: LodViewBlendUploadStats {
            immutable_table_upload_count: 1,
            weight_write_count: 0,
            buffer_allocation_count: 1,
            weight_bytes_written: 0,
            edge_count: 0,
            word_capacity: LOD_MORPH_HEADER_WORDS,
            lagging_edge_count: 0,
            last_max_delta: 0.0,
            last_weighted_record_energy: 0.0,
            max_weight_delta_per_frame: LOD_VIEW_BLEND_MAX_WEIGHT_DELTA_PER_FRAME,
        },
    });
    let first = LodRadixCandidateSnapshot {
        version: Some(Arc::clone(&first_phase)),
        phase_at_compaction: Some(LOD_RENDER_ACTIVE),
        fingerprint: Some(first_fingerprint),
        candidate_content_signature: Some(101),
        candidate_atlas_allocation_epoch: Some(11),
        rendered_candidate_count: 3,
        morph_identity,
        compute_input_generation: 4,
        compaction_signature: 41,
        view_blend: view_blend.clone(),
    };
    let second = LodRadixCandidateSnapshot {
        version: Some(Arc::clone(&second_phase)),
        phase_at_compaction: Some(LOD_RENDER_PREPARED),
        fingerprint: Some(second_fingerprint),
        candidate_content_signature: Some(202),
        candidate_atlas_allocation_epoch: Some(12),
        rendered_candidate_count: 5,
        morph_identity,
        compute_input_generation: 5,
        compaction_signature: 52,
        view_blend,
    };

    let mut tracker = LodRadixDrawableTracker::default();
    tracker.latch_compacted(first);
    assert!(tracker.drawable.is_none());
    assert!(!tracker.promote(40), "a mismatched radix cannot publish");
    assert!(tracker.drawable.is_none());
    assert_eq!(tracker.drawable_publication_generation, 0);

    // Relatch the compacted generation after the deliberate mismatch and
    // publish it through the one matching radix completion.
    tracker.latch_compacted(LodRadixCandidateSnapshot {
        version: Some(Arc::clone(&first_phase)),
        phase_at_compaction: Some(LOD_RENDER_ACTIVE),
        fingerprint: Some(first_fingerprint),
        candidate_content_signature: Some(101),
        candidate_atlas_allocation_epoch: Some(11),
        rendered_candidate_count: 3,
        morph_identity,
        compute_input_generation: 4,
        compaction_signature: 41,
        view_blend: morph_identity.map(|identity| LodLastRadixViewBlendForTesting {
            identity,
            edges: Vec::new(),
            weights: Vec::new(),
            endpoints: Vec::new(),
            recovery_lag: Vec::new(),
            invalid_pressure: Vec::new(),
            evaluation_view: None,
            evaluation_target: None,
            desired_evaluation_complete: false,
            upload: LodViewBlendUploadStats {
                immutable_table_upload_count: 1,
                weight_write_count: 0,
                buffer_allocation_count: 1,
                weight_bytes_written: 0,
                edge_count: 0,
                word_capacity: LOD_MORPH_HEADER_WORDS,
                lagging_edge_count: 0,
                last_max_delta: 0.0,
                last_weighted_record_energy: 0.0,
                max_weight_delta_per_frame: LOD_VIEW_BLEND_MAX_WEIGHT_DELTA_PER_FRAME,
            },
        }),
    });
    assert!(tracker.promote(41));
    let drawable = tracker.drawable.as_ref().unwrap();
    assert!(
        drawable
            .version
            .as_ref()
            .is_some_and(|version| Arc::ptr_eq(version, &first_phase))
    );
    assert_eq!(drawable.fingerprint, Some(first_fingerprint));
    assert_eq!(drawable.rendered_candidate_count, 3);
    assert_eq!(drawable.compute_input_generation, 4);
    assert_eq!(tracker.drawable_publication_generation, 1);

    // A no-write selector evaluation may complete after this exact physical
    // output has already promoted. Only its evaluation metadata changes: the
    // candidate/radix generations, immutable identity, weights, and upload
    // counters remain paired with the same drawable.
    let evaluation_view = LodView::perspective(Vec3::new(0.0, 0.0, 2.0), 1080.0, 1.0, 0.1);
    let evaluation_target = LodQualityTarget::Balanced {
        detail_fraction: 0.65,
        max_error_px: 1.25,
    };
    let before = tracker.drawable.as_ref().unwrap();
    let before_fingerprint = before.fingerprint;
    let before_content_signature = before.candidate_content_signature;
    let before_allocation_epoch = before.candidate_atlas_allocation_epoch;
    let before_compute_generation = before.compute_input_generation;
    let before_weights = before.view_blend.as_ref().unwrap().weights.clone();
    let before_upload = before.view_blend.as_ref().unwrap().upload;
    assert!(tracker.refresh_complete_view_blend_evaluation(
        morph_identity.unwrap(),
        &[],
        &[],
        Some(evaluation_view),
        Some(evaluation_target),
    ));
    let refreshed = tracker.drawable.as_ref().unwrap();
    let refreshed_blend = refreshed.view_blend.as_ref().unwrap();
    assert_eq!(refreshed_blend.evaluation_view, Some(evaluation_view));
    assert_eq!(refreshed_blend.evaluation_target, Some(evaluation_target));
    assert!(refreshed_blend.desired_evaluation_complete);
    assert_eq!(refreshed.fingerprint, before_fingerprint);
    assert_eq!(
        refreshed.candidate_content_signature,
        before_content_signature
    );
    assert_eq!(
        refreshed.candidate_atlas_allocation_epoch,
        before_allocation_epoch
    );
    assert_eq!(
        refreshed.compute_input_generation,
        before_compute_generation
    );
    assert_eq!(refreshed_blend.weights, before_weights);
    assert_eq!(refreshed_blend.upload, before_upload);
    assert_eq!(tracker.drawable_publication_generation, 1);

    let mismatched_state = LodViewBlendEdgeState {
        key: test_view_blend_edge_key(99),
        weight: LodViewBlendWeight {
            displayed: 0.5,
            desired: 0.5,
        },
        record_count: 1,
        recovery_lag: false,
        desired_initialized: true,
        initial_drawable_pending: false,
    };
    let rejected_view = LodView::perspective(Vec3::new(0.0, 0.0, 4.0), 1080.0, 1.0, 0.1);
    assert!(!tracker.refresh_complete_view_blend_evaluation(
        morph_identity.unwrap(),
        std::slice::from_ref(&mismatched_state),
        &[0.5],
        Some(rejected_view),
        Some(evaluation_target),
    ));
    assert_eq!(
        tracker
            .drawable
            .as_ref()
            .unwrap()
            .view_blend
            .as_ref()
            .unwrap()
            .evaluation_view,
        Some(evaluation_view)
    );
    assert_eq!(tracker.drawable_publication_generation, 1);

    tracker.latch_compacted(second);
    let still_drawable = tracker.drawable.as_ref().unwrap();
    assert!(
        still_drawable
            .version
            .as_ref()
            .is_some_and(|version| Arc::ptr_eq(version, &first_phase))
    );
    assert_eq!(still_drawable.rendered_candidate_count, 3);
    assert_eq!(tracker.drawable_publication_generation, 1);

    assert!(tracker.promote(52));
    let drawable = tracker.drawable.as_ref().unwrap();
    assert!(
        drawable
            .version
            .as_ref()
            .is_some_and(|version| Arc::ptr_eq(version, &second_phase))
    );
    assert_eq!(drawable.fingerprint, Some(second_fingerprint));
    assert_eq!(drawable.rendered_candidate_count, 5);
    assert_eq!(drawable.compute_input_generation, 5);
    assert_eq!(drawable.candidate_content_signature, Some(202));
    assert_eq!(drawable.candidate_atlas_allocation_epoch, Some(12));
    assert_eq!(drawable.morph_identity, morph_identity);
    assert_eq!(tracker.drawable_publication_generation, 2);

    let host = include_str!("../lod.rs");
    let compacted = host
        .split("fn mark_compacted")
        .nth(1)
        .expect("compaction metadata latch")
        .split("pub(crate) fn radix_sort_is_current")
        .next()
        .unwrap();
    assert!(compacted.contains("self.radix_drawable.latch_compacted(snapshot)"));
    let promoted = host
        .split("pub(crate) fn mark_radix_sorted")
        .nth(1)
        .expect("radix metadata promotion")
        .split("fn publish_bridge_activation_after_radix")
        .next()
        .unwrap();
    assert!(promoted.contains("self.radix_drawable.promote(signature)"));
}

#[test]
fn radix_promoted_morph_state_seeds_replacement_without_cpu_rollback() {
    let identity = LodCandidateFrontier::complete_empty_for_test(
        crate::stream::runtime::LodRuntimeViewId(7),
        &GaussianLodSettings::default(),
    )
    .with_temporal_transition_for_test(LodTemporalTransitionMode::Morphing)
    .temporal_transition()
    .and_then(|transition| transition.morph())
    .map(LodViewBlendBatch::identity)
    .expect("test morph identity exists");
    let key = test_view_blend_edge_key(17);
    let stale_weight = f32::from_bits(1_061_316_637);
    let promoted_weight = f32::from_bits(1_061_812_911);
    let staged_weight = 0.9_f32;
    let state = |weight| LodViewBlendEdgeState {
        key: key.clone(),
        weight: LodViewBlendWeight {
            displayed: weight,
            desired: weight,
        },
        record_count: 13,
        recovery_lag: false,
        desired_initialized: true,
        initial_drawable_pending: false,
    };

    let stale_cpu_state = state(stale_weight);
    let stale_cpu_snapshot = LodDrawableViewBlendSnapshot::from_edge_states(
        std::slice::from_ref(&stale_cpu_state),
        0.0,
        0.0,
    )
    .unwrap();
    assert_eq!(stale_cpu_snapshot.displayed, vec![stale_weight]);

    let mut tracker = LodRadixMorphStateTracker::default();
    assert!(tracker.latch_compacted(
        identity,
        41,
        std::slice::from_ref(&stale_cpu_state),
        &[false],
        true,
        0.0,
        0.0,
    ));
    assert!(tracker.promote(41));

    let actual_radix_state = state(promoted_weight);
    assert!(tracker.latch_compacted(
        identity,
        52,
        std::slice::from_ref(&actual_radix_state),
        &[false],
        true,
        0.0,
        0.0,
    ));
    let staged_live = [state(staged_weight)];
    let before_promotion = tracker
        .reconciliation_seed(Some(identity), &staged_live)
        .unwrap()
        .unwrap();
    assert_eq!(
        before_promotion[0].weight.displayed.to_bits(),
        stale_weight.to_bits()
    );

    assert!(tracker.promote(52));
    assert_eq!(tracker.drawable_signature, Some(52));
    assert_eq!(tracker.drawable_invalid_pressure, vec![false]);
    assert!(tracker.drawable_evaluation_complete(identity));
    let promoted_drawable = tracker.drawable_snapshot(identity).unwrap().unwrap();
    assert_eq!(promoted_drawable.displayed, vec![promoted_weight]);
    assert_eq!(promoted_drawable.desired, vec![promoted_weight]);
    assert_eq!(promoted_drawable.max_delta.to_bits(), 0.0_f32.to_bits());
    let promoted_seed = tracker
        .reconciliation_seed(Some(identity), &staged_live)
        .unwrap()
        .unwrap();
    assert_eq!(
        promoted_seed[0].weight.displayed.to_bits(),
        promoted_weight.to_bits(),
        "replacement inheritance must use the newest radix-proven weight, not the stale Prepare snapshot",
    );
    assert_eq!(
        promoted_seed[0].weight.desired.to_bits(),
        promoted_weight.to_bits()
    );

    let replacement = reconcile_lod_view_blend_edge_admissions(
        &promoted_seed,
        &[
            LodViewBlendEdgeAdmission {
                key,
                initial_weight: 0.0,
                record_count: 17,
                activation_requires_slew: false,
            },
            LodViewBlendEdgeAdmission {
                key: test_view_blend_edge_key(23),
                initial_weight: 1.0,
                record_count: 19,
                activation_requires_slew: false,
            },
        ],
    )
    .unwrap();
    assert_eq!(
        replacement[0].weight.displayed.to_bits(),
        promoted_weight.to_bits()
    );
    assert_eq!(
        replacement[0].weight.desired.to_bits(),
        promoted_weight.to_bits()
    );
    assert_eq!(
        replacement[1].weight,
        LodViewBlendWeight::initial(1.0).unwrap()
    );
    let replacement_drawable =
        LodDrawableViewBlendSnapshot::from_edge_states(&replacement, 0.0, 0.0).unwrap();
    assert_eq!(replacement_drawable.max_delta.to_bits(), 0.0_f32.to_bits());
    assert_eq!(
        replacement_drawable.weighted_record_energy.to_bits(),
        0.0_f64.to_bits()
    );

    let mut retargeted = actual_radix_state.clone();
    retargeted.weight.desired = staged_weight;
    assert!(tracker.refresh_drawable_evaluation(
        identity,
        std::slice::from_ref(&retargeted),
        &[false],
        true,
    ));
    let refreshed = tracker.drawable_snapshot(identity).unwrap().unwrap();
    assert_eq!(refreshed.displayed, vec![promoted_weight]);
    assert_eq!(refreshed.desired, vec![staged_weight]);
    assert_eq!(refreshed.lagging_count(), 1);
    assert!(tracker.drawable_evaluation_complete(identity));

    let host = include_str!("../lod.rs");
    let synchronize = host
        .split("fn synchronize_candidate_morph")
        .nth(1)
        .expect("candidate morph synchronization exists")
        .split("fn clear_drawable_view_blend_snapshot")
        .next()
        .expect("candidate morph synchronization has a bounded body");
    assert!(synchronize.contains("morph_radix_state"));
    assert!(synchronize.contains("reconciliation_seed"));
    assert!(!synchronize.contains("lod_view_blend_drawable_reconciliation_seed"));
    let compacted = host
        .split("fn mark_compacted")
        .nth(1)
        .expect("compaction state latch exists")
        .split("pub(crate) fn radix_sort_is_current")
        .next()
        .expect("compaction state latch has a bounded body");
    assert!(compacted.contains("self.morph_radix_state.latch_compacted"));
    let promoted = host
        .split("pub(crate) fn mark_radix_sorted")
        .nth(1)
        .expect("radix state promotion exists")
        .split("fn publish_bridge_activation_after_radix")
        .next()
        .expect("radix state promotion has a bounded body");
    assert!(promoted.contains("self.morph_radix_state.promote(signature)"));
}

#[test]
fn radix_promotion_consumes_view_blend_frame_energy_once() {
    let identity = LodCandidateFrontier::complete_empty_for_test(
        crate::stream::runtime::LodRuntimeViewId(71),
        &GaussianLodSettings::default(),
    )
    .with_temporal_transition_for_test(LodTemporalTransitionMode::Morphing)
    .temporal_transition()
    .and_then(|transition| transition.morph())
    .map(LodViewBlendBatch::identity)
    .expect("test morph identity exists");
    let edge_state = LodViewBlendEdgeState {
        key: test_view_blend_edge_key(17),
        weight: LodViewBlendWeight {
            displayed: 0.75,
            desired: 0.75,
        },
        record_count: 13,
        recovery_lag: false,
        desired_initialized: true,
        initial_drawable_pending: false,
    };
    let mut tracker = LodRadixMorphStateTracker::default();
    let mut live_max_delta = 0.125_f32;
    let mut live_energy = 19.5_f64;

    assert!(tracker.latch_compacted(
        identity,
        41,
        std::slice::from_ref(&edge_state),
        &[false],
        true,
        live_max_delta,
        live_energy,
    ));
    let promoted = tracker.promote(41);
    clear_lod_view_blend_frame_energy_after_promotion(
        promoted,
        &mut live_max_delta,
        &mut live_energy,
    );
    let moved = tracker.drawable_snapshot(identity).unwrap().unwrap();
    assert_eq!(moved.max_delta.to_bits(), 0.125_f32.to_bits());
    assert_eq!(moved.weighted_record_energy.to_bits(), 19.5_f64.to_bits());
    assert_eq!(live_max_delta.to_bits(), 0.0_f32.to_bits());
    assert_eq!(live_energy.to_bits(), 0.0_f64.to_bits());

    // A later camera-only compaction consumes the same weight suffix. Its
    // newly published drawable frame must not replay the movement event.
    assert!(tracker.latch_compacted(
        identity,
        52,
        std::slice::from_ref(&edge_state),
        &[false],
        true,
        live_max_delta,
        live_energy,
    ));
    let promoted = tracker.promote(52);
    clear_lod_view_blend_frame_energy_after_promotion(
        promoted,
        &mut live_max_delta,
        &mut live_energy,
    );
    let sort_only = tracker.drawable_snapshot(identity).unwrap().unwrap();
    assert_eq!(sort_only.displayed, moved.displayed);
    assert_eq!(sort_only.max_delta.to_bits(), 0.0_f32.to_bits());
    assert_eq!(
        sort_only.weighted_record_energy.to_bits(),
        0.0_f64.to_bits()
    );

    // A pending generation which did not promote still owns its event data.
    live_max_delta = 0.25;
    live_energy = 7.0;
    clear_lod_view_blend_frame_energy_after_promotion(false, &mut live_max_delta, &mut live_energy);
    assert_eq!(live_max_delta.to_bits(), 0.25_f32.to_bits());
    assert_eq!(live_energy.to_bits(), 7.0_f64.to_bits());

    let host = include_str!("../lod.rs");
    let promoted = host
        .split("pub(crate) fn mark_radix_sorted")
        .nth(1)
        .expect("radix promotion exists")
        .split("pub(crate) fn publish_bridge_activation_after_radix")
        .next()
        .expect("radix promotion has a bounded body");
    let production_promotion = promoted
        .find("self.morph_radix_state.promote(signature)")
        .expect("production morph state promotes");
    let testing_promotion = promoted
        .find("self.radix_drawable.promote(signature)")
        .expect("testing drawable metadata promotes");
    let clear = promoted
        .find("clear_lod_view_blend_frame_energy_after_promotion")
        .expect("live frame energy clears after promotion");
    assert!(production_promotion < testing_promotion && testing_promotion < clear);
}

#[test]
fn retirement_uses_the_radix_promoted_invalid_mask() {
    let identity = LodCandidateFrontier::complete_empty_for_test(
        crate::stream::runtime::LodRuntimeViewId(79),
        &GaussianLodSettings::default(),
    )
    .with_temporal_transition_for_test(LodTemporalTransitionMode::Morphing)
    .temporal_transition()
    .and_then(|transition| transition.morph())
    .map(LodViewBlendBatch::identity)
    .expect("test morph identity exists");
    let edge_state = LodViewBlendEdgeState {
        key: test_view_blend_edge_key(31),
        weight: LodViewBlendWeight {
            displayed: 0.0,
            desired: 0.0,
        },
        record_count: 5,
        recovery_lag: false,
        desired_initialized: true,
        initial_drawable_pending: false,
    };
    let stale_prepare_invalid_pressure = [false];
    let mut tracker = LodRadixMorphStateTracker::default();
    assert!(tracker.latch_compacted(
        identity,
        73,
        std::slice::from_ref(&edge_state),
        &[true],
        true,
        0.0,
        0.0,
    ));
    assert!(tracker.promote(73));
    let promoted = tracker.drawable_snapshot(identity).unwrap().unwrap();
    assert_eq!(stale_prepare_invalid_pressure, [false]);
    assert_eq!(promoted.invalid_pressure_edges, [true]);
    assert!(
        !lod_view_blend_retirement_endpoint_is_current(
            promoted.displayed[0],
            Some(0.0),
            promoted.invalid_pressure_edges[0],
            LodViewBlendEndpoint::ParentExact,
        ),
        "a stale-clear Prepare mask must not authorize retirement of the invalid radix-promoted edge",
    );

    let host = include_str!("../lod.rs");
    let attestation = host
        .split("fn view_blend_predecessor_attestation_is_current")
        .nth(1)
        .and_then(|source| {
            source
                .split("fn stage_view_blend_pressure_evaluation")
                .next()
        })
        .expect("bounded render retirement attestation body");
    assert!(attestation.contains("morph_radix_state\n            .drawable_snapshot"));
    assert!(attestation.contains("drawable_snapshot.invalid_pressure_edges[index]"));
    assert!(!attestation.contains("self.morph_drawable_invalid_pressure_edges[index]"));
}

#[test]
fn staged_first_morph_has_no_predecessor_until_radix_promotion() {
    let identity = LodCandidateFrontier::complete_empty_for_test(
        crate::stream::runtime::LodRuntimeViewId(80),
        &GaussianLodSettings::default(),
    )
    .with_temporal_transition_for_test(LodTemporalTransitionMode::Morphing)
    .temporal_transition()
    .and_then(|transition| transition.morph())
    .map(LodViewBlendBatch::identity)
    .expect("test morph identity exists");
    let edge_state = LodViewBlendEdgeState {
        key: test_view_blend_edge_key(32),
        weight: LodViewBlendWeight {
            displayed: 0.0,
            desired: 0.0,
        },
        record_count: 5,
        recovery_lag: false,
        desired_initialized: false,
        initial_drawable_pending: true,
    };
    let mut tracker = LodRadixMorphStateTracker::default();

    assert!(missing_promoted_morph_predecessor_is_safe(false, None));
    assert!(
        missing_promoted_morph_predecessor_is_safe(true, None),
        "a categorical drawable has no morph predecessor to attest"
    );
    assert!(
        !missing_promoted_morph_predecessor_is_safe(true, Some(identity)),
        "a live captured morph cannot lose its promoted tracker identity"
    );
    let fingerprint = lod_bridge_candidate_fingerprint(&LodRenderCandidate::new(
        LodCandidateFrontier::complete_empty_for_test(
            crate::stream::runtime::LodRuntimeViewId(81),
            &GaussianLodSettings::default(),
        ),
    ));
    assert!(!view_blend_predecessor_attestation_required(
        false,
        LodCandidateUploadPlan::Upload(fingerprint),
    ));
    assert!(view_blend_predecessor_attestation_required(
        true,
        LodCandidateUploadPlan::Upload(fingerprint),
    ));
    assert!(!view_blend_predecessor_attestation_required(
        true,
        LodCandidateUploadPlan::ReuseVersion,
    ));
    assert_eq!(
        tracker.drawable_identity(),
        None,
        "installing a live table is not drawable predecessor evidence"
    );
    assert!(tracker.latch_compacted(
        identity,
        74,
        std::slice::from_ref(&edge_state),
        &[false],
        false,
        0.0,
        0.0,
    ));
    assert_eq!(
        tracker.drawable_identity(),
        None,
        "compaction alone must not create a predecessor attestation"
    );
    assert!(tracker.promote(74));
    assert_eq!(tracker.drawable_identity(), Some(identity));

    let host = include_str!("../lod.rs");
    let attestation = host
        .split("fn view_blend_predecessor_attestation_is_current")
        .nth(1)
        .and_then(|source| {
            source
                .split("fn stage_view_blend_pressure_evaluation")
                .next()
        })
        .expect("bounded render retirement attestation body");
    assert!(attestation.contains("self.morph_radix_state.drawable_identity()"));
    assert!(
        !attestation.contains("let Some(drawable_identity) = self.morph_identity"),
        "a merely staged table must not masquerade as the drawable predecessor"
    );

    let commit = host
        .split("fn commit_lod_bridge_candidates")
        .nth(1)
        .expect("render candidate commit exists");
    assert!(commit.contains("view_blend_predecessor_attestation_required("));
    assert!(commit.contains("retained_package_replacement"));
    assert!(commit.contains("state.candidate_upload.plan(candidate)"));
}

#[test]
fn effective_morph_mode_is_part_of_candidate_upload_identity() {
    let settings = GaussianLodSettings::default();
    let frontier = LodCandidateFrontier::complete_empty_for_test(
        crate::stream::runtime::LodRuntimeViewId(7),
        &settings,
    )
    .with_temporal_transition_for_test(LodTemporalTransitionMode::Morphing);
    let candidate = LodRenderCandidate::new(frontier);
    let morphing = lod_bridge_candidate_fingerprint(&candidate);
    candidate.publish_temporal_transition_mode(LodTemporalTransitionMode::BoundedHardCohort);
    let bounded_hard = lod_bridge_candidate_fingerprint(&candidate);
    assert_ne!(
        morphing, bounded_hard,
        "adapter downgrade must invalidate descriptors even when refinement presentation ranges equal the target"
    );

    let package_gaussian_4d = LodRenderCandidate::new(
        LodCandidateFrontier::complete_empty_for_test(
            crate::stream::runtime::LodRuntimeViewId(8),
            &settings,
        )
        .with_temporal_transition_for_test(LodTemporalTransitionMode::Morphing),
    );
    package_gaussian_4d
        .phase
        .store(LOD_RENDER_WAITING, Ordering::Release);
    assert!(enforce_lod_candidate_gaussian_morph_capability(
        &package_gaussian_4d,
        GaussianMode::Gaussian4d,
        LodCandidateHardFallbackPolicy::RequestPackageReplan,
    ));
    assert_eq!(
        package_gaussian_4d.temporal_transition_mode(),
        Some(LodTemporalTransitionMode::Morphing),
        "a package veto must preserve the authored mode until main-world hard replan"
    );
    assert!(package_gaussian_4d.render_hard_fallback_requested());
    assert_eq!(
        package_gaussian_4d.phase.load(Ordering::Acquire),
        LOD_RENDER_WAITING
    );

    let source_gaussian_4d = LodRenderCandidate::new(
        LodCandidateFrontier::complete_empty_for_test(
            crate::stream::runtime::LodRuntimeViewId(9),
            &settings,
        )
        .with_temporal_transition_for_test(LodTemporalTransitionMode::Morphing),
    );
    source_gaussian_4d
        .phase
        .store(LOD_RENDER_WAITING, Ordering::Release);
    assert!(enforce_lod_candidate_gaussian_morph_capability(
        &source_gaussian_4d,
        GaussianMode::Gaussian4d,
        LodCandidateHardFallbackPolicy::RenderHardTarget,
    ));
    assert_eq!(
        source_gaussian_4d.temporal_transition_mode(),
        Some(LodTemporalTransitionMode::BoundedHardCohort),
        "a source-backed bridge can safely author the complete hard target"
    );
    assert!(!source_gaussian_4d.render_hard_fallback_requested());

    let gaussian_3d = LodRenderCandidate::new(
        LodCandidateFrontier::complete_empty_for_test(
            crate::stream::runtime::LodRuntimeViewId(10),
            &settings,
        )
        .with_temporal_transition_for_test(LodTemporalTransitionMode::Morphing),
    );
    gaussian_3d
        .phase
        .store(LOD_RENDER_WAITING, Ordering::Release);
    assert!(!enforce_lod_candidate_gaussian_morph_capability(
        &gaussian_3d,
        GaussianMode::Gaussian3d,
        LodCandidateHardFallbackPolicy::RequestPackageReplan,
    ));
    assert_eq!(
        gaussian_3d.temporal_transition_mode(),
        Some(LodTemporalTransitionMode::Morphing)
    );
}

#[test]
fn hard_fallback_request_blocks_an_already_armed_radix_activation() {
    let settings = GaussianLodSettings::default();
    let candidate = LodRenderCandidate::new(
        LodCandidateFrontier::complete_empty_for_test(
            crate::stream::runtime::LodRuntimeViewId(11),
            &settings,
        )
        .with_temporal_transition_for_test(LodTemporalTransitionMode::Morphing),
    );
    candidate
        .phase
        .store(LOD_RENDER_PREPARED, Ordering::Release);
    let before = lod_bridge_candidate_fingerprint(&candidate);
    let mut tracker = LodCandidateUploadTracker::default();
    tracker.mark_synchronized(&candidate.phase, before);

    candidate.request_hard_fallback();

    assert_eq!(lod_bridge_candidate_fingerprint(&candidate), before);
    assert_eq!(
        tracker.plan(&candidate),
        LodCandidateUploadPlan::ReuseVersion
    );
    assert_eq!(
        candidate.temporal_transition_mode(),
        Some(LodTemporalTransitionMode::Morphing)
    );
    assert!(candidate.render_hard_fallback_requested());
    assert_eq!(candidate.phase.load(Ordering::Acquire), LOD_RENDER_WAITING);
    assert!(!publish_bridge_activation_after_radix(&candidate.phase));
    assert!(candidate.active_presentation().is_none());
}

#[test]
fn package_hard_fallback_hold_preserves_the_retained_gpu_payload_contract() {
    let host = include_str!("../lod.rs");
    let synchronization = host
        .split("fn synchronize_candidate_morph")
        .nth(1)
        .expect("morph synchronization")
        .split("fn clear_drawable_view_blend_snapshot")
        .next()
        .unwrap();
    let unsupported = synchronization
        .split("LodCandidateMorphPlan::Unsupported =>")
        .nth(1)
        .expect("unsupported morph branch")
        .split("LodCandidateMorphPlan::Enabled")
        .next()
        .unwrap();
    assert!(unsupported.contains("publish_lod_candidate_hard_fallback"));
    assert!(unsupported.contains("if fallback == LodCandidateMorphSynchronization::Disabled"));
    assert!(
        !unsupported
            .split("return Ok(fallback)")
            .next()
            .unwrap()
            .contains("mark_compute_input_dirty")
    );

    let prepare = host
        .split("fn prepare_lod_compaction_buffers")
        .nth(1)
        .expect("compaction buffer preparation")
        .split("fn commit_lod_bridge_candidates")
        .next()
        .unwrap();
    assert!(prepare.contains("current_morph_word_capacity"));
    assert!(prepare.contains("if request.8"));
    assert!(prepare.contains("if hard_fallback_requested"));
    assert!(prepare.contains("pinned_existing: true"));
    assert!(prepare.contains("continue;"));
    let prepare_hold = prepare
        .find("if candidate.render_hard_fallback_requested()")
        .expect("prepare-buffer fallback hold");
    assert!(
        prepare_hold
            < prepare
                .find("if !storage_buffer_count_supported")
                .expect("storage-buffer capability rejection")
    );
    assert!(
        prepare_hold
            < prepare
                .find("plan_lod_compaction_allocation")
                .expect("device allocation planning")
    );

    let commit = host
        .split("fn commit_lod_bridge_candidates")
        .nth(1)
        .expect("bridge candidate commit");
    let hold = commit
        .split("if candidate.render_hard_fallback_requested()")
        .nth(1)
        .expect("render fallback hold")
        .split("let cold_staging")
        .next()
        .unwrap();
    assert!(hold.contains("defer_bridge_activation_for"));
    assert!(hold.contains("continue;"));
    assert!(!hold.contains("deactivate_morph"));
    assert!(!hold.contains("synchronize_bridge_candidate_frontier"));
    assert!(
        commit
            .find("if candidate.render_hard_fallback_requested()")
            .unwrap()
            < commit
                .find("if raster_gate.readiness == LodCandidateRasterPipelineReadiness::Failed")
                .expect("raster rejection")
    );
}

#[test]
fn frozen_cut_control_separates_candidate_identity_from_live_camera_sort_inputs() {
    let range = LodPhysicalRange {
        node: LodNodeId(1),
        page: LodPageId(2),
        slot: AtlasSlot {
            index: 3,
            generation: 4,
        },
        physical_start: 12,
        count: 3,
    };
    let frozen_candidate = lod_candidate_parts_fingerprint(7, &[range], 3);
    let initial = test_extracted_view();
    let mut moved = test_extracted_view();
    moved.world_from_view = GlobalTransform::from_translation(Vec3::new(3.0, -1.0, 8.0));

    assert_eq!(
        frozen_candidate,
        lod_candidate_parts_fingerprint(7, &[range], 3),
        "Frozen selection keeps the exact candidate payload"
    );
    assert_ne!(
        lod_live_camera_sort_signature(&initial),
        lod_live_camera_sort_signature(&moved),
        "the live render camera still changes compaction/frustum/sort inputs"
    );
}

#[test]
fn cold_allocation_is_not_a_drawable_bridge_output_until_radix_publication() {
    let host = include_str!("../lod.rs");
    let allocation = host
        .split("Self {")
        .find(|body| body.contains("candidate_and_scan_buffer: Some(candidate_and_scan_buffer)"))
        .expect("GpuLodCompaction allocation initializer");
    assert!(allocation.contains("has_drawable_bridge_output: false"));
    assert!(
        host.contains(".filter(|state| state.is_ready() && state.has_drawable_bridge_output())")
    );
    let radix_publication = host
        .split("pub(crate) fn mark_radix_sorted")
        .nth(1)
        .expect("radix publication method")
        .split("fn publish_bridge_activation_after_radix")
        .next()
        .unwrap();
    assert!(radix_publication.contains("self.candidate_descriptor_committed"));
    assert!(radix_publication.contains("self.has_drawable_bridge_output = true"));
}

#[test]
fn delayed_debug_gate_can_publish_an_already_sorted_candidate() {
    assert!(bridge_activation_can_publish_immediately(true, true, true));
    assert!(!bridge_activation_can_publish_immediately(
        false, true, true
    ));
    assert!(!bridge_activation_can_publish_immediately(
        true, false, true
    ));
    assert!(!bridge_activation_can_publish_immediately(
        true, true, false
    ));
}

#[test]
fn debug_activation_holds_off_to_on_until_the_matching_sidecar_is_ready() {
    use LodDebugRenderCapability::{SupportedPending, Unknown, Unsupported};

    assert!(lod_debug_candidate_activation_ready(
        false, Unknown, false, false
    ));
    assert!(!lod_debug_candidate_activation_ready(
        true, Unknown, false, false
    ));
    assert!(!lod_debug_candidate_activation_ready(
        true,
        SupportedPending,
        false,
        true
    ));
    assert!(!lod_debug_candidate_activation_ready(
        true,
        SupportedPending,
        true,
        false
    ));
    assert!(lod_debug_candidate_activation_ready(
        true,
        SupportedPending,
        true,
        true
    ));
    assert!(lod_debug_candidate_activation_ready(
        true,
        Unsupported,
        false,
        false
    ));

    assert!(lod_debug_sparse_identity_matches(Some(7), 7));
    assert!(!lod_debug_sparse_identity_matches(Some(7), 8));
    assert!(!lod_debug_sparse_identity_matches(None, 7));
}

#[test]
fn cold_handoff_prepares_the_bounded_atlas_while_the_source_remains_drawable() {
    fn atlas_id(value: u128) -> AssetId<PlanarGaussian3d> {
        AssetId::Uuid {
            uuid: bevy::asset::uuid::Uuid::from_u128(value),
        }
    }

    let source = atlas_id(1);
    let bounded_atlas = atlas_id(2);
    let candidates = LodRenderCandidates {
        staging_atlas: Some(bounded_atlas),
        candidate_draw_required: false,
        ..default()
    };

    assert_eq!(
        lod_compaction_asset_id(source, Some(&candidates)),
        Some(bounded_atlas),
        "compaction capacity and state must be keyed to the bounded atlas, not the exact source"
    );
    assert!(
        !skip_unready_candidate_required_draw(false, false),
        "a missing staging output must not suppress the still-bound exact source"
    );

    assert_eq!(
        cold_staging_candidate_phase(false, true, true, true, true),
        LOD_RENDER_WAITING,
        "target-atlas generation mismatch cannot publish PREPARED"
    );
    assert_eq!(
        cold_staging_candidate_phase(true, false, true, true, true),
        LOD_RENDER_WAITING,
        "uncompiled compaction/radix pipelines cannot publish PREPARED"
    );
    assert_eq!(
        cold_staging_candidate_phase(true, true, false, true, true),
        LOD_RENDER_WAITING,
        "a queued LoD raster pipeline must retain the exact source"
    );
    assert_eq!(
        cold_staging_candidate_phase(true, true, true, false, true),
        LOD_RENDER_WAITING,
        "a delayed LoD debug binding must retain the exact source"
    );
    assert_eq!(
        cold_staging_candidate_phase(true, true, true, true, false),
        LOD_RENDER_FAILED,
        "an invalid target-atlas frontier fails closed"
    );
    assert_eq!(
        cold_staging_candidate_phase(true, true, true, true, true),
        LOD_RENDER_PREPARED,
        "only a current, valid target atlas with ready compute and raster pipelines may publish PREPARED"
    );

    let host = include_str!("../lod.rs");
    let staging_branch = host
        .split("let phase = if atlas_current")
        .nth(1)
        .expect("cold staging branch exists")
        .split("record_cold_staging_phase(&mut cold_staging_updates, candidate, phase);")
        .next()
        .expect("cold staging branch terminates before active synchronization");
    assert!(staging_branch.contains("raster_pipeline_ready"));
    assert!(staging_branch.contains("cold_staging_candidate_phase"));
}

#[test]
fn retained_pending_debug_staging_reaches_prepared_before_activation() {
    assert_eq!(
        retained_candidate_preparation_phase(false, true, false, true, true),
        LOD_RENDER_WAITING,
        "uncompiled compaction/radix pipelines cannot acknowledge a replacement"
    );
    assert_eq!(
        retained_candidate_preparation_phase(true, false, false, true, true),
        LOD_RENDER_WAITING,
        "a queued raster permutation cannot acknowledge a replacement"
    );
    assert_eq!(
        retained_candidate_preparation_phase(true, true, false, true, false),
        LOD_RENDER_FAILED,
        "an invalid replacement frontier fails closed"
    );

    assert_eq!(
        retained_candidate_preparation_phase(true, true, false, false, true),
        LOD_RENDER_WAITING,
        "a cold or bridge candidate cannot bypass an incomplete debug binding"
    );
    assert_eq!(
        debug_incomplete_candidate_phase(LOD_RENDER_ACTIVE, true, true, true, false, true,),
        LOD_RENDER_ACTIVE,
        "a presentation-only debug toggle must preserve the current drawable output"
    );
    assert_eq!(
        debug_incomplete_candidate_phase(LOD_RENDER_WAITING, false, true, true, false, true,),
        LOD_RENDER_WAITING,
        "a cold or bridge candidate remains WAITING while debug is incomplete"
    );
    let phase = AtomicU8::new(retained_candidate_preparation_phase(
        true, true, false, true, true,
    ));
    assert_eq!(
        phase.load(Ordering::Acquire),
        LOD_RENDER_PREPARED,
        "pending debug metadata must not deadlock the package staging handshake"
    );
    let mut debug_activation_ready = false;
    if debug_activation_ready {
        publish_bridge_activation_after_radix(&phase);
    }
    assert_eq!(
        phase.load(Ordering::Acquire),
        LOD_RENDER_PREPARED,
        "debug-incomplete preparation must not activate the replacement"
    );
    debug_activation_ready = true;
    assert!(debug_activation_ready && publish_bridge_activation_after_radix(&phase));
    assert_eq!(phase.load(Ordering::Acquire), LOD_RENDER_ACTIVE);

    let host = include_str!("../lod.rs");
    let retained = host
        .split("LodBridgeAtlasDecision::RetainCurrent =>")
        .nth(1)
        .expect("retained replacement branch exists")
        .split("LodBridgeAtlasDecision::SynchronizePending =>")
        .next()
        .expect("retained replacement branch terminates");
    assert!(retained.contains("retained_candidate_preparation_phase"));
    assert!(
        !retained.contains("&& debug_activation_ready"),
        "debug-sidecar staging readiness must not block PREPARED"
    );
    let synchronized_hold = host
        .split("LodBridgeAtlasDecision::SynchronizePending => {}")
        .nth(1)
        .expect("synchronized replacement branch exists")
        .split("let atlas_content_revision")
        .next()
        .expect("debug hold precedes descriptor synchronization");
    assert!(synchronized_hold.contains("if !debug_activation_ready"));
    assert!(synchronized_hold.contains("debug_incomplete_candidate_phase"));
    assert!(synchronized_hold.contains("continue;"));
    assert!(
        !synchronized_hold.contains("synchronize_bridge_candidate_frontier"),
        "debug-incomplete PREPARED must retain the old descriptor/output"
    );
}

#[test]
fn retained_replacement_is_validate_only_until_every_draw_prerequisite_is_ready() {
    for missing in 0..4 {
        let mut ready = [true; 4];
        ready[missing] = false;
        assert!(
            !retained_replacement_synchronization_ready(ready[0], ready[1], ready[2], ready[3]),
            "missing prerequisite {missing} must preserve the retained descriptor/output"
        );
    }
    assert!(retained_replacement_synchronization_ready(
        true, true, true, true
    ));

    let host = include_str!("../lod.rs");
    let synchronized = host
        .split("LodBridgeAtlasDecision::SynchronizePending => {}")
        .nth(1)
        .expect("synchronized replacement branch exists")
        .split("let atlas_content_revision")
        .next()
        .expect("retained readiness hold precedes descriptor synchronization");
    let hold = synchronized
        .split("if retained_package_replacement")
        .nth(1)
        .expect("retained replacement readiness hold exists");
    assert!(hold.contains("retained_replacement_synchronization_ready"));
    assert!(hold.contains(".validate_bridge_candidate_presentation(candidate)"));
    assert!(hold.contains("state.defer_bridge_activation_for(candidate)"));
    assert!(hold.contains("continue;"));
    assert!(
        !hold.contains("synchronize_bridge_candidate_frontier"),
        "a retained replacement may not overwrite live buffers before every draw prerequisite is ready"
    );
}

#[test]
fn candidate_raster_gate_aggregates_all_subviews_and_fails_on_any_error() {
    use bevy::shader::ShaderCacheError;

    assert_eq!(
        LodCandidateRasterPipelineReadiness::Ready
            .merge(LodCandidateRasterPipelineReadiness::Pending),
        LodCandidateRasterPipelineReadiness::Pending,
        "one compiled and one queued HDR/MSAA subview cannot publish PREPARED"
    );
    assert_eq!(
        LodCandidateRasterPipelineReadiness::Ready
            .merge(LodCandidateRasterPipelineReadiness::Failed),
        LodCandidateRasterPipelineReadiness::Failed,
        "one failed subview permutation rejects the shared candidate"
    );
    let failed = CachedPipelineState::Err(ShaderCacheError::CreateShaderModule(
        "synthetic candidate raster failure".to_owned(),
    ));
    assert_eq!(
        lod_candidate_raster_pipeline_readiness(&failed),
        LodCandidateRasterPipelineReadiness::Failed
    );

    let ready = LodCandidateRasterGate {
        readiness: LodCandidateRasterPipelineReadiness::Ready,
        debug_activation_ready: true,
        consumer_count: 1,
    };
    let delayed_debug = LodCandidateRasterGate {
        readiness: LodCandidateRasterPipelineReadiness::Ready,
        debug_activation_ready: false,
        consumer_count: 1,
    };
    assert_eq!(
        ready.merge(delayed_debug),
        LodCandidateRasterGate {
            readiness: LodCandidateRasterPipelineReadiness::Ready,
            debug_activation_ready: false,
            consumer_count: 2,
        }
    );
    assert!(!multi_subview_activation_ready(2, 0));
    assert!(!multi_subview_activation_ready(2, 1));
    assert!(multi_subview_activation_ready(2, 2));
    assert!(!multi_subview_activation_ready(2, 3));
    assert!(!multi_subview_activation_ready(1, 1));

    let host = include_str!("../lod.rs");
    let gate = host
        .split("// Prewarm both stable debug permutations")
        .nth(1)
        .and_then(|source| source.split("let gate = LodCandidateRasterGate").next())
        .expect("candidate raster prewarm gate");
    assert!(gate.contains("[false, true].into_iter().take(debug_variant_count)"));
    assert!(gate.contains("readiness = readiness.merge(variant_readiness)"));
    assert!(host.contains("state.defer_bridge_activation_for(candidate);"));
    assert!(host.contains("state.has_current_drawable_bridge_candidate(candidate)"));
    assert!(host.contains("record_multi_subview_drawable_output("));
    assert!(host.contains("publish_bridge_activation_after_radix(&candidate_phase);"));
    assert!(host.contains("record_drawable_view_blend_publication("));
    assert!(host.contains("publish_complete_view_blend_publication(&publication)"));
    assert!(host.contains("publish_lod_view_blend_after_radix::<R>"));
}

#[test]
fn morph_activation_waits_for_a_complete_published_consumer_aggregate() {
    let frontier = LodCandidateFrontier::complete_empty_for_test(
        crate::stream::runtime::LodRuntimeViewId(77),
        &GaussianLodSettings::default(),
    )
    .with_temporal_transition_for_test(LodTemporalTransitionMode::Morphing);
    let phase = Arc::new(AtomicU8::new(LOD_RENDER_PREPARED));
    let candidate = LodRenderCandidate::with_phase(frontier, Arc::clone(&phase));
    // Complete-empty candidates intentionally activate at construction because
    // no render pass will visit them. Reset the synthetic capability so this
    // pure test can exercise the Morphing aggregate-before-ACTIVE barrier.
    phase.store(LOD_RENDER_PREPARED, Ordering::Release);
    let empty = LodDrawableViewBlendSnapshot::from_edge_states(&[], 0.0, 0.0).unwrap();
    let complete = LodViewBlendPublication {
        candidate: &candidate,
        expected_consumers: 1,
        drawable_consumers: 1,
        activation_allowed_consumers: 1,
        snapshot: Some(empty.clone()),
    };

    assert!(view_blend_publication_is_complete(&complete));
    assert!(view_blend_publication_can_activate(&complete));
    assert_eq!(phase.load(Ordering::Acquire), LOD_RENDER_PREPARED);
    assert!(publish_complete_view_blend_publication(&complete));
    assert_eq!(
        phase.load(Ordering::Acquire),
        LOD_RENDER_PREPARED,
        "coherent aggregate publication must precede the ACTIVE capability"
    );
    assert!(publish_bridge_activation_after_radix(&phase));
    assert_eq!(phase.load(Ordering::Acquire), LOD_RENDER_ACTIVE);

    let missing = LodViewBlendPublication {
        candidate: &candidate,
        expected_consumers: 2,
        drawable_consumers: 1,
        activation_allowed_consumers: 1,
        snapshot: Some(empty.clone()),
    };
    assert!(!view_blend_publication_is_complete(&missing));
    assert!(!view_blend_publication_can_activate(&missing));

    let mut invalid_snapshot = empty;
    invalid_snapshot.invalid_pressure_edges.push(true);
    let invalid = LodViewBlendPublication {
        candidate: &candidate,
        expected_consumers: 1,
        drawable_consumers: 1,
        activation_allowed_consumers: 1,
        snapshot: Some(invalid_snapshot),
    };
    assert!(!view_blend_publication_can_activate(&invalid));

    let host = include_str!("../lod.rs");
    let mark_radix = host
        .split("pub(crate) fn mark_radix_sorted")
        .nth(1)
        .and_then(|source| {
            source
                .split("pub(crate) fn publish_bridge_activation_after_radix")
                .next()
        })
        .expect("radix promotion body");
    assert!(mark_radix.contains("if self.morph_identity.is_none()"));
    let cleanup = host
        .split("fn publish_lod_view_blend_after_radix")
        .nth(1)
        .expect("Cleanup aggregate barrier");
    let aggregate_publish = cleanup
        .find("publish_complete_view_blend_publication")
        .expect("Cleanup publishes the coherent aggregate");
    let activation = cleanup
        .find("publish_bridge_activation_after_radix")
        .expect("Cleanup publishes ACTIVE");
    assert!(aggregate_publish < activation);
}

#[test]
fn removed_edge_retirement_requires_private_and_current_view_endpoint_agreement() {
    assert!(lod_view_blend_retirement_endpoint_is_current(
        0.0,
        Some(0.0),
        false,
        LodViewBlendEndpoint::ParentExact,
    ));
    assert!(lod_view_blend_retirement_endpoint_is_current(
        1.0,
        Some(1.0),
        false,
        LodViewBlendEndpoint::ChildrenExact,
    ));
    assert!(!lod_view_blend_retirement_endpoint_is_current(
        0.0,
        Some(1.0),
        false,
        LodViewBlendEndpoint::ParentExact,
    ));
    assert!(!lod_view_blend_retirement_endpoint_is_current(
        1.0,
        Some(1.0),
        true,
        LodViewBlendEndpoint::ChildrenExact,
    ));
    assert!(!lod_view_blend_retirement_endpoint_is_current(
        0.0,
        None,
        false,
        LodViewBlendEndpoint::ParentExact,
    ));
    assert!(!lod_view_blend_retirement_endpoint_is_current(
        0.5,
        Some(0.5),
        false,
        LodViewBlendEndpoint::Fractional,
    ));

    let frontier = LodCandidateFrontier::complete_empty_for_test(
        crate::stream::runtime::LodRuntimeViewId(78),
        &GaussianLodSettings::default(),
    )
    .with_temporal_transition_for_test(LodTemporalTransitionMode::Morphing);
    let candidate = LodRenderCandidate::new(frontier);
    let published_before = candidate
        .view_blend_snapshot_for_testing()
        .expect("Morphing candidates publish an initial view-blend snapshot");
    assert_eq!(candidate.phase.load(Ordering::Acquire), LOD_RENDER_ACTIVE);
    let first_sync = LodCandidateUploadPlan::Upload(lod_bridge_candidate_fingerprint(&candidate));
    assert!(view_blend_predecessor_attestation_required(
        true, first_sync,
    ));
    if view_blend_predecessor_attestation_required(true, first_sync)
        && !lod_view_blend_retirement_endpoint_is_current(
            0.0,
            Some(1.0),
            false,
            LodViewBlendEndpoint::ParentExact,
        )
    {
        candidate.request_view_blend_replan();
    }
    assert!(candidate.view_blend_replan_requested());
    assert_eq!(candidate.phase.load(Ordering::Acquire), LOD_RENDER_WAITING);
    assert_eq!(
        candidate.temporal_transition_mode(),
        Some(LodTemporalTransitionMode::Morphing),
        "the pipelined endpoint mismatch requests a non-hard replan without rewriting authored mode"
    );
    assert!(!candidate.render_hard_fallback_requested());
    assert_eq!(
        candidate.view_blend_snapshot_for_testing(),
        Some(published_before),
        "the replan request must preserve the predecessor's published presentation evidence"
    );

    let host = include_str!("../lod.rs");
    let commit = host
        .split("fn commit_lod_bridge_candidates")
        .nth(1)
        .expect("render candidate commit exists");
    let attestation = commit
        .find("view_blend_predecessor_attestation_is_current")
        .expect("pipelined predecessor endpoint attestation");
    let synchronize = commit
        .find("synchronize_bridge_candidate_frontier")
        .expect("candidate descriptor synchronization");
    assert!(attestation < synchronize);
    let hold = &commit[attestation..synchronize];
    assert!(hold.contains("candidate.request_view_blend_replan()"));
    assert!(hold.contains("state.defer_bridge_activation_for(candidate)"));
    assert!(hold.contains("continue;"));
}

#[test]
fn cold_handoff_requires_compaction_and_radix_after_the_atomic_handle_swap() {
    fn atlas_id(value: u128) -> AssetId<PlanarGaussian3d> {
        AssetId::Uuid {
            uuid: bevy::asset::uuid::Uuid::from_u128(value),
        }
    }

    let bounded_atlas = atlas_id(3);
    let candidates = LodRenderCandidates {
        staging_atlas: Some(bounded_atlas),
        candidate_draw_required: true,
        ..default()
    };
    assert_eq!(
        lod_compaction_asset_id(bounded_atlas, Some(&candidates)),
        Some(bounded_atlas),
        "the preallocated staging state is reused after the entity handle swaps"
    );

    let phase = AtomicU8::new(LOD_RENDER_PREPARED);
    assert_eq!(
        lod_bridge_atlas_decision(phase.load(Ordering::Acquire), true),
        LodBridgeAtlasDecision::SynchronizePending
    );
    assert_eq!(
        LodCompactionReadiness::AwaitingCandidates.after_candidate_commit(true),
        LodCompactionReadiness::Ready,
        "the prepared descriptor may execute in the first frame after the swap"
    );
    assert!(
        skip_unready_candidate_required_draw(true, false),
        "the bounded atlas must never be drawn raw before candidate radix output exists"
    );
    assert_eq!(phase.load(Ordering::Acquire), LOD_RENDER_PREPARED);

    assert!(publish_bridge_activation_after_radix(&phase));
    assert_eq!(phase.load(Ordering::Acquire), LOD_RENDER_ACTIVE);
    assert!(
        !skip_unready_candidate_required_draw(true, true),
        "the candidate draw opens only after compaction and radix publish output"
    );
}

#[test]
fn cold_candidate_runs_in_the_observed_ready_frame_but_waits_for_radix_to_draw() {
    let readiness = LodCompactionReadiness::AwaitingCandidates.after_candidate_commit(true);
    assert_eq!(readiness, LodCompactionReadiness::Ready);

    // Readiness permits same-frame compaction/radix dispatch, but the draw
    // remains fail-closed until that radix pass publishes a complete output.
    assert!(skip_unready_candidate_required_draw(true, false));
    assert!(!skip_unready_candidate_required_draw(true, true));

    let phase = AtomicU8::new(LOD_RENDER_PREPARED);
    assert!(publish_bridge_activation_after_radix(&phase));
    assert_eq!(phase.load(Ordering::Acquire), LOD_RENDER_ACTIVE);

    assert_eq!(
        LodCompactionReadiness::AwaitingCandidates.after_candidate_commit(false),
        LodCompactionReadiness::PendingCandidates
    );
}

#[test]
fn identical_fingerprint_new_phase_forces_recompute_and_reaches_active() {
    let range = LodPhysicalRange {
        node: LodNodeId(1),
        page: LodPageId(2),
        slot: AtlasSlot {
            index: 3,
            generation: 4,
        },
        physical_start: 12,
        count: 3,
    };
    let fingerprint = lod_candidate_parts_fingerprint(7, &[range], 3);
    let old_phase = Arc::new(AtomicU8::new(LOD_RENDER_ACTIVE));
    let new_phase = Arc::new(AtomicU8::new(LOD_RENDER_PREPARED));
    let mut tracker = LodCandidateUploadTracker::default();
    tracker.mark_synchronized(&old_phase, fingerprint);

    let plan = tracker.plan_fingerprint(&new_phase, fingerprint);
    assert_eq!(plan, LodCandidateUploadPlan::ReuseFingerprint(fingerprint));
    assert!(plan.requires_recompute());
    let host = include_str!("../lod.rs");
    let reuse_branch = host
        .split("LodCandidateUploadPlan::ReuseFingerprint(fingerprint) =>")
        .nth(1)
        .expect("reuse-fingerprint synchronization branch")
        .split("LodCandidateUploadPlan::Upload(fingerprint) =>")
        .next()
        .unwrap();
    assert!(reuse_branch.contains("self.mark_compute_input_dirty()"));

    // The forced compaction/radix pass owns the one publication point for the
    // new Arc even though its descriptor bytes match the cached fingerprint.
    assert!(publish_bridge_activation_after_radix(&new_phase));
    assert_eq!(new_phase.load(Ordering::Acquire), LOD_RENDER_ACTIVE);
}

#[test]
fn deferred_package_generations_retain_old_draw_then_publish_once_after_radix() {
    const SLOT_COUNT: usize = 300;
    const PER_FRAME_LIMIT: usize = 256;

    let phase = AtomicU8::new(LOD_RENDER_PREPARED);
    let mut generations_current = vec![false; SLOT_COUNT];
    let mut visible_output = "old";

    generations_current[..PER_FRAME_LIMIT].fill(true);
    let all_current = generations_current.iter().all(|current| *current);
    assert_eq!(
        lod_bridge_atlas_decision(phase.load(Ordering::Acquire), all_current),
        LodBridgeAtlasDecision::RetainCurrent
    );
    assert_eq!(visible_output, "old");
    assert_eq!(phase.load(Ordering::Acquire), LOD_RENDER_PREPARED);
    assert_eq!(
        LodCompactionReadiness::Ready.after_commit(),
        LodCompactionReadiness::Ready,
        "an existing ready output remains drawable while 44 slots are deferred"
    );

    generations_current[PER_FRAME_LIMIT..].fill(true);
    let all_current = generations_current.iter().all(|current| *current);
    assert_eq!(
        lod_bridge_atlas_decision(phase.load(Ordering::Acquire), all_current),
        LodBridgeAtlasDecision::SynchronizePending
    );
    assert_eq!(visible_output, "old", "descriptor sync is not publication");
    assert!(publish_bridge_activation_after_radix(&phase));
    visible_output = "new";
    assert_eq!(visible_output, "new");
    assert_eq!(phase.load(Ordering::Acquire), LOD_RENDER_ACTIVE);
    assert!(
        !publish_bridge_activation_after_radix(&phase),
        "the transaction has exactly one publication point"
    );
}

#[test]
fn multiview_and_failed_pending_transactions_preserve_each_current_output() {
    let host = include_str!("../lod.rs");
    assert!(host.contains("set.by_camera.values().all(|candidate|"));

    let left = AtomicU8::new(LOD_RENDER_PREPARED);
    let right = AtomicU8::new(LOD_RENDER_PREPARED);

    assert_eq!(
        lod_bridge_atlas_decision(left.load(Ordering::Acquire), false),
        LodBridgeAtlasDecision::RetainCurrent
    );
    assert_eq!(
        lod_bridge_atlas_decision(right.load(Ordering::Acquire), false),
        LodBridgeAtlasDecision::RetainCurrent
    );
    assert_eq!(left.load(Ordering::Acquire), LOD_RENDER_PREPARED);
    assert_eq!(right.load(Ordering::Acquire), LOD_RENDER_PREPARED);

    // Package code supplies one union-generation result to every view. Once
    // it becomes true, both descriptors may switch in the same render frame.
    assert_eq!(
        lod_bridge_atlas_decision(left.load(Ordering::Acquire), true),
        LodBridgeAtlasDecision::SynchronizePending
    );
    assert_eq!(
        lod_bridge_atlas_decision(right.load(Ordering::Acquire), true),
        LodBridgeAtlasDecision::SynchronizePending
    );
    assert!(publish_bridge_activation_after_radix(&left));
    assert!(publish_bridge_activation_after_radix(&right));
    assert_eq!(left.load(Ordering::Acquire), LOD_RENDER_ACTIVE);
    assert_eq!(right.load(Ordering::Acquire), LOD_RENDER_ACTIVE);

    let failed = AtomicU8::new(LOD_RENDER_FAILED);
    let retained_output = LodCompactionReadiness::Ready;
    assert_eq!(
        retained_output,
        LodCompactionReadiness::Ready,
        "the other camera retains its independent old draw"
    );
    assert!(
        !publish_bridge_activation_after_radix(&failed),
        "a failed replacement cannot become active from a late radix callback"
    );
    assert_eq!(failed.load(Ordering::Acquire), LOD_RENDER_FAILED);
}

#[test]
fn transitioning_cut_revokes_visibility_when_any_required_atlas_generation_is_stale() {
    assert_eq!(
        lod_bridge_atlas_decision(LOD_RENDER_TRANSITIONING, false),
        LodBridgeAtlasDecision::RejectActive,
        "a morph reads both presentation records and parent lookup sources"
    );
    assert_eq!(
        lod_bridge_atlas_decision(LOD_RENDER_TRANSITIONING, true),
        LodBridgeAtlasDecision::SynchronizePending
    );
}

#[test]
fn waiting_candidate_cannot_retain_drawable_output_from_a_recreated_atlas_allocation() {
    assert_eq!(
        lod_bridge_atlas_decision(LOD_RENDER_WAITING, false),
        LodBridgeAtlasDecision::RetainCurrent,
        "ordinary slot streaming may retain output from the same allocation",
    );
    assert!(lod_drawable_atlas_allocation_is_current(
        true,
        Some(7),
        Some(7),
    ));
    assert!(!lod_drawable_atlas_allocation_is_current(
        true,
        Some(7),
        Some(8),
    ));
    assert!(!lod_drawable_atlas_allocation_is_current(
        true,
        Some(7),
        None,
    ));
    assert!(lod_drawable_atlas_allocation_is_current(true, None, None,));
    assert!(lod_drawable_atlas_allocation_is_current(
        false,
        Some(7),
        Some(8),
    ));
}

#[test]
fn latched_multi_view_transition_keeps_companion_prepared_payload_synchronizable() {
    assert!(lod_pending_candidate_policy_allows_synchronization(
        false, true, false,
    ));
    assert!(lod_pending_candidate_policy_allows_synchronization(
        false, false, true,
    ));
    assert!(lod_pending_candidate_policy_allows_synchronization(
        true, false, false,
    ));
    assert!(
        !lod_pending_candidate_policy_allows_synchronization(false, false, false),
        "an ordinary stale PREPARED candidate remains cancellable before any transition is drawable"
    );
}

#[test]
fn view_blend_edges_are_phase_independent_and_recovering_reversal_changes_sign_immediately() {
    let mut recovering = LodViewBlendEdgeState {
        key: test_view_blend_edge_key(1),
        weight: LodViewBlendWeight::initial(0.0).unwrap(),
        record_count: 9,
        recovery_lag: true,
        desired_initialized: false,
        initial_drawable_pending: false,
    };
    assert!(update_lod_view_blend_edge_weight(&mut recovering, 1.0, false).unwrap());
    assert_eq!(
        recovering.weight.displayed.to_bits(),
        LOD_VIEW_BLEND_MAX_WEIGHT_DELTA_PER_FRAME.to_bits()
    );
    assert!(recovering.recovery_lag);
    let before_reversal = recovering.weight.displayed;
    assert!(update_lod_view_blend_edge_weight(&mut recovering, 0.0, false).unwrap());
    assert!(recovering.weight.displayed < before_reversal);
    assert_eq!(recovering.weight.displayed.to_bits(), 0.0_f32.to_bits());
    assert!(!recovering.recovery_lag);

    let mut independently_lagging = LodViewBlendEdgeState {
        key: test_view_blend_edge_key(3),
        weight: LodViewBlendWeight {
            displayed: 0.4,
            desired: 1.0,
        },
        record_count: 2,
        recovery_lag: true,
        desired_initialized: true,
        initial_drawable_pending: false,
    };
    assert!(update_lod_view_blend_edge_weight(&mut independently_lagging, 1.0, false).unwrap());
    assert_eq!(
        independently_lagging.weight.displayed.to_bits(),
        (0.4 + LOD_VIEW_BLEND_MAX_WEIGHT_DELTA_PER_FRAME).to_bits(),
        "each late edge advances from its own retained weight, not a cohort phase",
    );

    let mut fully_resident = LodViewBlendEdgeState {
        key: test_view_blend_edge_key(2),
        weight: LodViewBlendWeight::initial(0.0).unwrap(),
        record_count: 4,
        recovery_lag: false,
        desired_initialized: false,
        initial_drawable_pending: false,
    };
    assert!(update_lod_view_blend_edge_weight(&mut fully_resident, 0.91, false).unwrap());
    assert_eq!(
        fully_resident.weight.displayed.to_bits(),
        0.91_f32.to_bits()
    );
    assert_eq!(fully_resident.weight.desired.to_bits(), 0.91_f32.to_bits());
    assert!(!fully_resident.recovery_lag);
    assert!(
        !update_lod_view_blend_edge_weight(&mut fully_resident, 0.91, false).unwrap(),
        "a stationary fully resident mid-band edge is a fixed point",
    );
    assert!(update_lod_view_blend_edge_weight(&mut fully_resident, 0.1, false).unwrap());
    assert_eq!(fully_resident.weight.displayed.to_bits(), 0.1_f32.to_bits());
    assert!(update_lod_view_blend_edge_weight(&mut fully_resident, 0.91, false).unwrap());
    assert_eq!(
        fully_resident.weight.displayed.to_bits(),
        0.91_f32.to_bits()
    );

    assert!(update_lod_view_blend_edge_weight(&mut fully_resident, 0.0, true).unwrap());
    assert_eq!(
        fully_resident.weight.displayed.to_bits(),
        (0.91 - LOD_VIEW_BLEND_MAX_WEIGHT_DELTA_PER_FRAME).to_bits()
    );
    assert!(fully_resident.recovery_lag);

    let host = include_str!("../lod.rs");
    assert!(!host.contains("LOD_MORPH_PROGRESS_FRAMES"));
    assert!(!host.contains("advance_morph_progress"));
    assert!(!host.contains("active_morph_phase"));
    assert!(!host.contains("morph_finalizing"));
    assert!(host.contains("publish_bridge_activation_after_radix(phase)"));
    assert!(host.contains("LOD_RENDER_ACTIVE | LOD_RENDER_TRANSITIONING"));
    assert!(host.contains("capture_drawable_view_blend_snapshot"));
    assert!(host.contains("publish_view_blend_aggregate_snapshot"));
}

#[test]
fn active_fractional_invalid_pressure_holds_then_recovers_with_bounded_slew() {
    let mut active = LodViewBlendEdgeState {
        key: test_view_blend_edge_key(4),
        weight: LodViewBlendWeight {
            displayed: 0.4,
            desired: 0.4,
        },
        record_count: 12,
        recovery_lag: false,
        desired_initialized: true,
        initial_drawable_pending: false,
    };
    let held_bits = (
        active.weight.displayed.to_bits(),
        active.weight.desired.to_bits(),
    );

    hold_lod_view_blend_weights_for_invalid_pressure(std::slice::from_mut(&mut active));
    assert_eq!(
        (
            active.weight.displayed.to_bits(),
            active.weight.desired.to_bits(),
        ),
        held_bits,
        "an invalid ACTIVE pressure evaluation must not mutate either published weight bit",
    );
    assert!(active.recovery_lag);

    let mut drawable =
        LodDrawableViewBlendSnapshot::from_edge_states(std::slice::from_ref(&active), 0.0, 0.0)
            .unwrap();
    drawable.invalid_pressure_edges[0] = true;

    assert!(update_lod_view_blend_edge_weight(&mut active, 0.9, false).unwrap());
    assert_eq!(
        active.weight.displayed.to_bits(),
        (0.4 + LOD_VIEW_BLEND_MAX_WEIGHT_DELTA_PER_FRAME).to_bits(),
        "the first later-valid frame must recover from the held fractional value, not jump",
    );
    assert_eq!(active.weight.desired.to_bits(), 0.9_f32.to_bits());
    assert!(active.recovery_lag);
    drawable
        .recover_pressure_targets(std::slice::from_ref(&active))
        .unwrap();
    assert_eq!(drawable.displayed[0].to_bits(), 0.4_f32.to_bits());
    assert_eq!(drawable.desired[0].to_bits(), 0.9_f32.to_bits());
    assert_eq!(drawable.invalid_pressure_count(), 0);
    assert_eq!(drawable.lagging_count(), 1);
    assert!((drawable.max_lag - 0.5).abs() <= f32::EPSILON);

    let before_reversal = active.weight.displayed;
    assert!(update_lod_view_blend_edge_weight(&mut active, 0.0, false).unwrap());
    assert!(active.weight.displayed < before_reversal);
    assert!(
        before_reversal - active.weight.displayed
            <= LOD_VIEW_BLEND_MAX_WEIGHT_DELTA_PER_FRAME + f32::EPSILON,
        "recovery retargeting must reverse on the next frame without exceeding the safety bound",
    );

    let host = include_str!("../lod.rs");
    let update = host
        .split("fn update_view_blend_weights")
        .nth(1)
        .expect("live weight update")
        .split("pub fn readiness")
        .next()
        .expect("bounded live update body");
    assert!(update.contains("stage_view_blend_pressure_evaluation"));
    assert!(update.contains("LodViewBlendPressureEvaluation::Invalid"));
    assert!(!update.contains("request_hard_fallback"));
    assert!(!update.contains("LOD_RENDER_WAITING"));

    let prepare = host
        .split("fn commit_lod_bridge_candidates")
        .nth(1)
        .expect("render preparation system");
    assert!(prepare.contains("preflight_view_blend_activation"));
    assert!(prepare.contains("state.defer_bridge_activation_for(candidate)"));
}

#[test]
fn checked_exact_retarget_retires_recovery_before_the_next_camera_view() {
    let mut recovering = LodViewBlendEdgeState {
        key: test_view_blend_edge_key(6),
        weight: LodViewBlendWeight {
            displayed: 0.4,
            desired: 0.9,
        },
        record_count: 12,
        recovery_lag: true,
        desired_initialized: true,
        initial_drawable_pending: false,
    };

    retarget_checked_lod_view_blend_edge_desired(&mut recovering, 0.4).unwrap();
    assert_eq!(
        recovering.weight,
        LodViewBlendWeight {
            displayed: 0.4,
            desired: 0.4,
        },
        "a checked view which meets the retained drawable must complete recovery",
    );
    assert!(
        !recovering.recovery_lag,
        "an exact checked retarget must not leave stale recovery provenance",
    );

    assert!(update_lod_view_blend_edge_weight(&mut recovering, 0.9, false).unwrap());
    assert_eq!(
        recovering.weight.displayed.to_bits(),
        0.9_f32.to_bits(),
        "the next ordinary camera view must track exactly after recovery completed",
    );
    assert!(!recovering.recovery_lag);
}

#[test]
fn newly_admitted_edges_publish_the_retained_endpoint_before_tracking_or_recovery() {
    let admissions = [
        LodViewBlendEdgeAdmission {
            key: test_view_blend_edge_key(10),
            initial_weight: 0.0,
            record_count: 5,
            activation_requires_slew: false,
        },
        LodViewBlendEdgeAdmission {
            key: test_view_blend_edge_key(20),
            initial_weight: 0.0,
            record_count: 7,
            activation_requires_slew: true,
        },
    ];
    let mut states = reconcile_lod_view_blend_edge_admissions(&[], &admissions).unwrap();
    let initial = LodDrawableViewBlendSnapshot::from_edge_states(&states, 0.0, 0.0).unwrap();
    assert_eq!(
        initial.endpoints,
        vec![
            LodViewBlendEndpoint::ParentExact,
            LodViewBlendEndpoint::ParentExact,
        ],
    );

    // The initial radix result may already have been cached before pressure
    // preflight completes. Priming must refresh only desired/lag telemetry on
    // that cached publication; displayed endpoint bits remain authored.
    let mut cached_first_publication = initial.clone();
    cached_first_publication.invalid_pressure_edges[0] = true;
    retarget_checked_lod_view_blend_edge_desired(&mut states[1], 1.0).unwrap();
    cached_first_publication
        .retarget_pressure_targets(&states)
        .unwrap();
    assert_eq!(cached_first_publication.displayed, vec![0.0, 0.0]);
    assert_eq!(cached_first_publication.desired, vec![0.0, 1.0]);
    assert_eq!(
        cached_first_publication.endpoints,
        vec![
            LodViewBlendEndpoint::ParentExact,
            LodViewBlendEndpoint::ParentExact,
        ],
    );
    assert_eq!(cached_first_publication.lagging_edges, vec![false, true]);
    assert_eq!(
        cached_first_publication.max_lag.to_bits(),
        1.0_f32.to_bits()
    );
    assert_eq!(
        cached_first_publication.invalid_pressure_edges,
        vec![true, false],
        "retargeting a cached authored snapshot must not rewrite its drawable invalid mask",
    );

    let first_publication =
        LodDrawableViewBlendSnapshot::from_edge_states(&states, 0.0, 0.0).unwrap();
    assert_eq!(first_publication.displayed, vec![0.0, 0.0]);
    assert_eq!(first_publication.desired, vec![0.0, 1.0]);
    assert_eq!(first_publication.lagging_edges, vec![false, true]);
    assert_eq!(first_publication.max_lag.to_bits(), 1.0_f32.to_bits());

    assert!(
        !update_lod_view_blend_edge_after_initial_draw(&mut states[0], Some(0.75), false,).unwrap()
    );
    assert!(
        !update_lod_view_blend_edge_after_initial_draw(&mut states[1], Some(1.0), false,).unwrap()
    );
    assert_eq!(states[0].weight.desired.to_bits(), 0.0_f32.to_bits());
    assert_eq!(states[1].weight.desired.to_bits(), 1.0_f32.to_bits());
    assert!(states.iter().all(|state| !state.initial_drawable_pending));

    assert!(
        update_lod_view_blend_edge_after_initial_draw(&mut states[0], Some(0.75), false,).unwrap()
    );
    assert!(
        update_lod_view_blend_edge_after_initial_draw(&mut states[1], Some(1.0), false,).unwrap()
    );
    assert_eq!(states[0].weight.displayed.to_bits(), 0.75_f32.to_bits());
    assert_eq!(
        states[1].weight.displayed.to_bits(),
        LOD_VIEW_BLEND_MAX_WEIGHT_DELTA_PER_FRAME.to_bits(),
    );
    assert!(states[1].recovery_lag);

    let frozen_admission = LodViewBlendEdgeAdmission {
        key: test_view_blend_edge_key(30),
        initial_weight: 0.0,
        record_count: 3,
        activation_requires_slew: false,
    };
    let mut frozen_state = reconcile_lod_view_blend_edge_admissions(&[], &[frozen_admission])
        .unwrap()
        .remove(0);
    let frozen_first_publication = LodDrawableViewBlendSnapshot::from_edge_states(
        std::slice::from_ref(&frozen_state),
        0.0,
        0.0,
    )
    .unwrap();
    assert_eq!(frozen_first_publication.displayed, vec![0.0]);
    assert_eq!(frozen_first_publication.desired, vec![0.0]);
    assert_eq!(frozen_first_publication.lagging_count(), 0);
    assert!(
        !update_lod_view_blend_edge_after_initial_draw(&mut frozen_state, None, false).unwrap()
    );
    assert!(!frozen_state.desired_initialized);
    assert!(
        update_lod_view_blend_edge_after_initial_draw(&mut frozen_state, Some(0.8), true).unwrap()
    );
    assert_eq!(
        frozen_state.weight.displayed.to_bits(),
        LOD_VIEW_BLEND_MAX_WEIGHT_DELTA_PER_FRAME.to_bits(),
        "Frozen-to-Dynamic first initialization must recover from the held endpoint",
    );
    assert_eq!(frozen_state.weight.desired.to_bits(), 0.8_f32.to_bits());
    assert!(frozen_state.recovery_lag);

    let invalid_admission = LodViewBlendEdgeAdmission {
        key: test_view_blend_edge_key(40),
        initial_weight: 0.0,
        record_count: 11,
        activation_requires_slew: true,
    };
    let mut invalid_state =
        reconcile_lod_view_blend_edge_admissions(&[], &[invalid_admission]).unwrap();
    hold_lod_view_blend_weights_for_invalid_pressure(&mut invalid_state);
    let mut invalid_first_publication =
        LodDrawableViewBlendSnapshot::from_edge_states(&invalid_state, 0.0, 0.0).unwrap();
    invalid_first_publication.invalid_pressure_edges[0] = true;
    assert_eq!(invalid_first_publication.displayed, vec![0.0]);
    assert_eq!(invalid_first_publication.desired, vec![0.0]);
    assert_eq!(invalid_first_publication.lagging_count(), 0);
    assert_eq!(invalid_first_publication.invalid_pressure_count(), 1);

    // A common edge can survive a table replacement with retained displayed
    // bits which differ from both the replacement's authored initial endpoint
    // and the current selector oracle. Its displayed value stays exact, while
    // desired must become the checked oracle before the replacement snapshot
    // is called complete.
    let mut inherited = LodViewBlendEdgeState {
        key: test_view_blend_edge_key(50),
        weight: LodViewBlendWeight {
            displayed: 0.0,
            desired: 0.0,
        },
        record_count: 13,
        recovery_lag: false,
        desired_initialized: true,
        initial_drawable_pending: false,
    };
    let mut inherited_drawable =
        LodDrawableViewBlendSnapshot::from_edge_states(std::slice::from_ref(&inherited), 0.0, 0.0)
            .unwrap();
    retarget_checked_lod_view_blend_edge_desired(&mut inherited, 0.73).unwrap();
    inherited_drawable
        .retarget_pressure_targets(std::slice::from_ref(&inherited))
        .unwrap();
    assert_eq!(inherited_drawable.displayed, vec![0.0]);
    assert_eq!(inherited_drawable.desired, vec![0.73]);
    assert_eq!(
        inherited_drawable.endpoints,
        vec![LodViewBlendEndpoint::ParentExact]
    );
    assert_eq!(inherited_drawable.lagging_count(), 1);
    assert_eq!(inherited_drawable.max_lag.to_bits(), 0.73_f32.to_bits());

    let mut second_inherited = LodViewBlendEdgeState {
        key: test_view_blend_edge_key(55),
        weight: LodViewBlendWeight {
            displayed: 1.0,
            desired: 1.0,
        },
        record_count: 19,
        recovery_lag: false,
        desired_initialized: true,
        initial_drawable_pending: false,
    };
    retarget_checked_lod_view_blend_edge_desired(&mut second_inherited, 0.25).unwrap();

    // Exercise the identity-replacement reconciliation used by production:
    // common keys retain their exact lagged drawable state, while the disjoint
    // ordinary edge starts at its authored endpoint with no artificial lag.
    let replacement_admissions = [
        LodViewBlendEdgeAdmission {
            key: inherited.key.clone(),
            initial_weight: 1.0,
            record_count: 23,
            activation_requires_slew: false,
        },
        LodViewBlendEdgeAdmission {
            key: second_inherited.key.clone(),
            initial_weight: 0.0,
            record_count: 29,
            activation_requires_slew: false,
        },
        LodViewBlendEdgeAdmission {
            key: test_view_blend_edge_key(60),
            initial_weight: 1.0,
            record_count: 17,
            activation_requires_slew: false,
        },
    ];
    let replacement_states = reconcile_lod_view_blend_edge_admissions(
        &[inherited, second_inherited],
        &replacement_admissions,
    )
    .unwrap();
    assert_eq!(
        replacement_states[0].weight,
        LodViewBlendWeight {
            displayed: 0.0,
            desired: 0.73,
        }
    );
    assert_eq!(
        replacement_states[1].weight,
        LodViewBlendWeight {
            displayed: 1.0,
            desired: 0.25,
        }
    );
    assert_eq!(
        replacement_states[2].weight,
        LodViewBlendWeight::initial(1.0).unwrap()
    );
    assert!(replacement_states[2].initial_drawable_pending);
    assert_eq!(
        lod_view_blend_lagging_edge_count(&replacement_states),
        2,
        "common retargets contribute lag while the ordinary authored guard does not",
    );
    let primed_snapshot =
        LodDrawableViewBlendSnapshot::from_edge_states(&replacement_states, 0.0, 0.0).unwrap();
    assert_eq!(primed_snapshot.lagging_count(), 2);
    assert_eq!(primed_snapshot.max_delta.to_bits(), 0.0_f32.to_bits());
    assert_eq!(
        primed_snapshot.weighted_record_energy.to_bits(),
        0.0_f64.to_bits()
    );

    let host = include_str!("../lod.rs");
    let pending = host
        .split("// Validate every pressure pair before priming any late edge.")
        .nth(1)
        .expect("pending candidate checked-pressure preparation exists")
        .split("// Consume the exact authored first draw only after it has")
        .next()
        .expect("pending preparation has a bounded body");
    let preflight = pending
        .find("preflight_view_blend_activation")
        .expect("pending path performs checked all-edge preflight");
    let prime = pending
        .find("prime_initial_recovery_view_blend_desired")
        .expect("valid pending path primes truthful late-edge desired telemetry");
    let capture = pending
        .find("capture_drawable_view_blend_snapshot")
        .expect("pending path captures the authored drawable publication");
    assert!(preflight < prime && prime < capture);
    assert!(pending.contains("if pressure_valid"));

    let prime_body = host
        .split("fn prime_initial_recovery_view_blend_desired")
        .nth(1)
        .expect("late-recovery priming helper exists")
        .split("fn drawable_view_blend_snapshot")
        .next()
        .expect("late-recovery priming helper has a bounded body");
    assert!(prime_body.contains("let mut desired_weights"));
    assert!(prime_body.contains("retarget_checked_lod_view_blend_edge_desired"));
    assert!(prime_body.contains("lod_view_blend_lagging_edge_count"));
    assert!(prime_body.contains("desired_evaluation_complete"));
    assert!(prime_body.contains("snapshot.retarget_pressure_targets"));

    let synchronize = host
        .split("fn synchronize_candidate_morph")
        .nth(1)
        .expect("candidate morph synchronization exists")
        .split("fn clear_drawable_view_blend_snapshot")
        .next()
        .expect("candidate morph synchronization has a bounded body");
    assert!(synchronize.contains("lod_view_blend_lagging_edge_count(&self.morph_edge_states)"));
    assert!(!synchronize.contains("self.morph_lagging_edge_count = 0;"));

    let radix_latch = host
        .split("fn view_blend_for_pending_radix_for_testing")
        .nth(1)
        .expect("testing radix latch exists")
        .split("fn mark_compute_input_dirty")
        .next()
        .expect("testing radix latch has a bounded body");
    assert!(radix_latch.contains("exact_lagging_edge_count"));
    assert!(radix_latch.contains("upload.lagging_edge_count = exact_lagging_edge_count"));
}

#[test]
fn frozen_resume_prime_publishes_recovery_before_moving_the_radix_suffix() {
    let identity = LodCandidateFrontier::complete_empty_for_test(
        crate::stream::runtime::LodRuntimeViewId(83),
        &GaussianLodSettings::default(),
    )
    .with_temporal_transition_for_test(LodTemporalTransitionMode::Morphing)
    .temporal_transition()
    .and_then(|transition| transition.morph())
    .map(LodViewBlendBatch::identity)
    .expect("test morph identity exists");
    let state = |id, displayed, desired_initialized| LodViewBlendEdgeState {
        key: test_view_blend_edge_key(id),
        weight: LodViewBlendWeight {
            displayed,
            desired: displayed,
        },
        record_count: 5,
        recovery_lag: false,
        desired_initialized,
        initial_drawable_pending: false,
    };
    let mut states = vec![
        state(83, 0.25, true),
        state(84, 0.25, true),
        // An edge first drawn while Frozen has consumed its authored guard but
        // has not yet initialized a selector desired value.
        state(85, 0.0, false),
    ];
    let mut tracker = LodRadixMorphStateTracker::default();
    assert!(tracker.latch_compacted(identity, 91, &states, &[false; 3], false, 0.0, 0.0));
    assert!(tracker.promote(91));
    let frozen = tracker.drawable_snapshot(identity).unwrap().unwrap();

    for (state, desired) in states.iter_mut().zip([0.9, 0.3, 0.8]) {
        retarget_checked_lod_view_blend_edge_desired(state, desired).unwrap();
    }
    mark_lod_view_blend_frozen_resume_recovery(&mut states);
    assert!(states.iter().all(|state| state.recovery_lag));
    assert!(states.iter().all(|state| state.desired_initialized));
    assert!(tracker.refresh_drawable_evaluation(identity, &states, &[false; 3], true));

    let overlay = tracker.drawable_snapshot(identity).unwrap().unwrap();
    assert_eq!(tracker.drawable_signature, Some(91));
    assert_eq!(overlay.displayed, frozen.displayed);
    assert_eq!(overlay.displayed, vec![0.25, 0.25, 0.0]);
    assert_eq!(overlay.desired, vec![0.9, 0.3, 0.8]);
    assert_eq!(overlay.recovery_edges, vec![true, true, true]);
    assert_eq!(overlay.lagging_count(), 3);
    assert_eq!(overlay.max_delta.to_bits(), 0.0_f32.to_bits());
    assert_eq!(overlay.weighted_record_energy.to_bits(), 0.0_f64.to_bits());
    assert!(tracker.drawable_evaluation_complete(identity));

    let mut ordinary_authored = LodViewBlendEdgeState {
        key: test_view_blend_edge_key(86),
        weight: LodViewBlendWeight::initial(0.0).unwrap(),
        record_count: 7,
        recovery_lag: false,
        desired_initialized: false,
        initial_drawable_pending: true,
    };
    retarget_checked_lod_view_blend_edge_desired(&mut ordinary_authored, 0.75).unwrap();
    mark_lod_view_blend_frozen_resume_recovery(std::slice::from_mut(&mut ordinary_authored));
    assert_eq!(
        ordinary_authored.weight,
        LodViewBlendWeight::initial(0.0).unwrap()
    );
    assert!(!ordinary_authored.desired_initialized);
    assert!(!ordinary_authored.recovery_lag);
    let mixed_overlay = LodDrawableViewBlendSnapshot::from_edge_states(
        &[states[0].clone(), ordinary_authored],
        0.0,
        0.0,
    )
    .unwrap();
    assert_eq!(mixed_overlay.displayed, vec![0.25, 0.0]);
    assert_eq!(mixed_overlay.desired, vec![0.9, 0.0]);
    assert_eq!(mixed_overlay.recovery_edges, vec![true, false]);
    assert_eq!(mixed_overlay.lagging_count(), 1);

    let slow_previous = states[0].weight.displayed;
    assert!(update_lod_view_blend_edge_weight(&mut states[0], 0.9, true).unwrap());
    assert_eq!(
        states[0].weight.displayed.to_bits(),
        (slow_previous + LOD_VIEW_BLEND_MAX_WEIGHT_DELTA_PER_FRAME).to_bits(),
        "the first physical Dynamic suffix must consume one bounded recovery step",
    );
    assert!(states[0].recovery_lag);
    assert!(update_lod_view_blend_edge_weight(&mut states[1], 0.3, true).unwrap());
    assert_eq!(states[1].weight.displayed.to_bits(), 0.3_f32.to_bits());
    assert!(
        !states[1].recovery_lag,
        "a sub-limit resume delta must catch up exactly and clear recovery"
    );

    let host = include_str!("../lod.rs");
    let prime = host
        .split("fn prime_initial_recovery_view_blend_desired")
        .nth(1)
        .expect("Frozen-resume priming helper exists")
        .split("fn drawable_view_blend_snapshot")
        .next()
        .expect("Frozen-resume priming helper has a bounded body");
    let retarget = prime
        .find("retarget_checked_lod_view_blend_edge_desired")
        .expect("prime retargets the complete checked table");
    let recovery = prime
        .find("mark_lod_view_blend_frozen_resume_recovery")
        .expect("prime marks Frozen-resume recovery");
    let lag = prime
        .find("lod_view_blend_lagging_edge_count")
        .expect("prime re-derives coherent lag");
    let production_refresh = prime
        .find("morph_radix_state.refresh_drawable_evaluation")
        .expect("prime refreshes production radix metadata");
    let testing_refresh = prime
        .find("radix_drawable.refresh_checked_view_blend_evaluation")
        .expect("prime refreshes testing radix metadata");
    assert!(
        retarget < recovery
            && recovery < lag
            && lag < production_refresh
            && production_refresh < testing_refresh,
        "recovery provenance must be coherent before either radix publication is refreshed",
    );
}

#[test]
fn drawable_publication_keeps_previous_weights_view_and_counters_while_next_is_staged() {
    let previous_view = LodView::perspective(Vec3::ZERO, 1080.0, 1.0, 0.1);
    let next_view = LodView::perspective(Vec3::new(0.0, 0.0, 2.0), 1080.0, 1.0, 0.1);
    let target = LodQualityTarget::Balanced {
        detail_fraction: 0.65,
        max_error_px: 1.25,
    };
    let mut staged_state = LodViewBlendEdgeState {
        key: test_view_blend_edge_key(40),
        weight: LodViewBlendWeight {
            displayed: 0.25,
            desired: 0.25,
        },
        record_count: 8,
        recovery_lag: false,
        desired_initialized: true,
        initial_drawable_pending: false,
    };
    let drawable = LodDrawableViewBlendSnapshot::from_edge_states(
        std::slice::from_ref(&staged_state),
        0.04,
        0.32,
    )
    .unwrap();
    let live_at_capture = LodViewBlendUploadStats {
        immutable_table_upload_count: 2,
        weight_write_count: 7,
        buffer_allocation_count: 1,
        weight_bytes_written: 28,
        edge_count: 1,
        word_capacity: 64,
        lagging_edge_count: 99,
        last_max_delta: 0.9,
        last_weighted_record_energy: 9.0,
        max_weight_delta_per_frame: LOD_VIEW_BLEND_MAX_WEIGHT_DELTA_PER_FRAME,
    };
    let publication = LodViewBlendDrawablePublicationForTesting {
        compaction_generation: 3,
        publication_generation: 11,
        evaluation_view: Some(previous_view),
        evaluation_target: Some(target),
        desired_evaluation_complete: true,
        upload: lod_view_blend_upload_stats_for_drawable_snapshot(live_at_capture, &drawable),
    };

    assert!(update_lod_view_blend_edge_weight(&mut staged_state, 0.75, false).unwrap());
    let staged_upload = LodViewBlendUploadStats {
        weight_write_count: 8,
        weight_bytes_written: 32,
        last_max_delta: 0.5,
        last_weighted_record_energy: 4.0,
        ..live_at_capture
    };

    assert_eq!(drawable.displayed, vec![0.25]);
    assert_eq!(drawable.desired, vec![0.25]);
    assert_eq!(publication.evaluation_view, Some(previous_view));
    assert_eq!(publication.evaluation_target, Some(target));
    assert!(publication.desired_evaluation_complete);
    assert_eq!(publication.upload.weight_write_count, 7);
    assert_eq!(publication.upload.weight_bytes_written, 28);
    assert_eq!(publication.upload.lagging_edge_count, 0);
    assert_eq!(
        publication.upload.last_max_delta.to_bits(),
        0.04_f32.to_bits()
    );
    assert_eq!(
        publication.upload.last_weighted_record_energy.to_bits(),
        0.32_f64.to_bits()
    );
    assert_eq!(staged_state.weight.displayed.to_bits(), 0.75_f32.to_bits());
    assert_eq!(staged_upload.weight_write_count, 8);
    assert_eq!(next_view.camera_position, Vec3::new(0.0, 0.0, 2.0));

    let host = include_str!("../lod.rs");
    let capture = host
        .split("fn capture_drawable_view_blend_snapshot")
        .nth(1)
        .expect("drawable capture exists")
        .split("fn prime_initial_recovery_view_blend_desired")
        .next()
        .expect("drawable capture has a bounded body");
    assert!(capture.contains("view_blend_upload_stats_for_drawable(Some(&snapshot))"));
    assert!(capture.contains("self.morph_drawable_snapshot = Some(snapshot)"));
    assert!(!capture.contains("last_sorted_signature:"));
}

#[test]
fn authored_drawable_preflight_attaches_only_coherent_evaluation_metadata() {
    let prior_view = LodView::perspective(Vec3::ZERO, 1080.0, 1.0, 0.1);
    let preflight_view = LodView::perspective(Vec3::new(0.0, 0.0, 3.0), 1440.0, 1.2, 0.1);
    let prior_target = LodQualityTarget::Balanced {
        detail_fraction: 0.35,
        max_error_px: 2.0,
    };
    let preflight_target = LodQualityTarget::Balanced {
        detail_fraction: 0.65,
        max_error_px: 1.25,
    };
    let upload = LodViewBlendUploadStats {
        immutable_table_upload_count: 4,
        weight_write_count: 9,
        buffer_allocation_count: 2,
        weight_bytes_written: 36,
        edge_count: 3,
        word_capacity: 128,
        lagging_edge_count: 0,
        last_max_delta: 0.0,
        last_weighted_record_energy: 0.0,
        max_weight_delta_per_frame: LOD_VIEW_BLEND_MAX_WEIGHT_DELTA_PER_FRAME,
    };
    let mut publication = LodViewBlendDrawablePublicationForTesting {
        compaction_generation: 7,
        publication_generation: 13,
        evaluation_view: Some(prior_view),
        evaluation_target: Some(prior_target),
        desired_evaluation_complete: true,
        upload,
    };

    attach_view_blend_preflight_evaluation_for_testing(
        &mut publication,
        Some(preflight_view),
        Some(preflight_target),
    );
    assert_eq!(publication.compaction_generation, 7);
    assert_eq!(publication.publication_generation, 13);
    assert_eq!(publication.evaluation_view, Some(preflight_view));
    assert_eq!(publication.evaluation_target, Some(preflight_target));
    assert!(!publication.desired_evaluation_complete);
    assert_eq!(publication.upload, upload);

    // A preflight whose selector view cannot be constructed must not retain a
    // stale half-pair from an earlier pose. Constructible invalid edge metrics
    // still take the paired path above, allowing the raw pressure oracle to
    // report `None` against the exact attempted view.
    attach_view_blend_preflight_evaluation_for_testing(
        &mut publication,
        None,
        Some(preflight_target),
    );
    assert_eq!(publication.compaction_generation, 7);
    assert_eq!(publication.publication_generation, 13);
    assert_eq!(publication.evaluation_view, None);
    assert_eq!(publication.evaluation_target, None);
    assert!(!publication.desired_evaluation_complete);
    assert_eq!(publication.upload, upload);

    refresh_complete_view_blend_evaluation_for_testing(
        &mut publication,
        Some(preflight_view),
        Some(preflight_target),
    );
    assert_eq!(publication.compaction_generation, 7);
    assert_eq!(publication.publication_generation, 13);
    assert_eq!(publication.evaluation_view, Some(preflight_view));
    assert_eq!(publication.evaluation_target, Some(preflight_target));
    assert!(publication.desired_evaluation_complete);
    assert_eq!(publication.upload, upload);

    let host = include_str!("../lod.rs");
    let preflight = host
        .split("fn preflight_view_blend_activation")
        .nth(1)
        .expect("pending view-blend preflight exists")
        .split("fn recover_drawable_view_blend_pressure_targets")
        .next()
        .expect("preflight has a bounded body");
    assert!(preflight.contains("stage_view_blend_pressure_evaluation"));
    assert!(preflight.contains("attach_view_blend_preflight_evaluation_for_testing"));

    let active_update = host
        .split("fn update_view_blend_weights")
        .nth(1)
        .expect("ACTIVE view-blend update exists")
        .split("pub fn readiness")
        .next()
        .expect("ACTIVE update has a bounded body");
    assert!(active_update.contains("authored_publication_pending"));
    assert!(active_update.contains("initial_drawable_pending"));
    assert!(active_update.contains("stage_view_blend_pressure_evaluation"));
    assert!(active_update.contains("attach_view_blend_preflight_evaluation_for_testing"));
    assert!(active_update.contains("refresh_complete_view_blend_evaluation_for_testing"));
    assert!(active_update.contains("state.weight.displayed.to_bits() == desired.to_bits()"));
    assert!(active_update.contains("state.weight.desired.to_bits() == desired.to_bits()"));
}

#[test]
fn overlapping_table_replacement_preserves_common_lag_and_initializes_only_new_edges() {
    fn key(id: u64) -> LodViewBlendEdgeKey {
        let metric = LodViewBlendMetricKey {
            center_bits: [id as u32, 0, 0],
            radius_bits: 1.0_f32.to_bits(),
            geometric_error_bits: (id as f32 + 1.0).to_bits(),
            quality_min_bits: 0.0_f32.to_bits(),
            quality_max_bits: 1.0_f32.to_bits(),
            certificate_bits: 0.5_f32.to_bits(),
            original_representation: false,
        };
        LodViewBlendEdgeKey {
            parent: LodNodeId(id),
            children: vec![LodNodeId(id + 1_000)],
            parent_metric: metric,
            child_metrics: vec![metric],
        }
    }

    let old_a = LodViewBlendEdgeState {
        key: key(1),
        weight: LodViewBlendWeight {
            displayed: 1.0,
            desired: 0.4,
        },
        record_count: 10,
        recovery_lag: true,
        desired_initialized: true,
        initial_drawable_pending: false,
    };
    let old_b = LodViewBlendEdgeState {
        key: key(2),
        weight: LodViewBlendWeight {
            displayed: 0.35,
            desired: 0.8,
        },
        record_count: 20,
        recovery_lag: true,
        desired_initialized: true,
        initial_drawable_pending: false,
    };
    let admissions = [
        LodViewBlendEdgeAdmission {
            key: key(2),
            initial_weight: 1.0,
            record_count: 22,
            activation_requires_slew: false,
        },
        LodViewBlendEdgeAdmission {
            key: key(3),
            initial_weight: 0.0,
            record_count: 30,
            activation_requires_slew: true,
        },
    ];
    let next = reconcile_lod_view_blend_edge_admissions(&[old_a, old_b.clone()], &admissions)
        .expect("exact edge A may retire while fractional edge B overlaps new edge C");
    assert_eq!(next.len(), 2);
    assert_eq!(next[0].key, old_b.key);
    assert_eq!(next[0].weight, old_b.weight);
    assert!(next[0].recovery_lag);
    assert!(!next[0].initial_drawable_pending);
    assert_eq!(next[0].record_count, 22);
    assert_eq!(next[1].weight, LodViewBlendWeight::initial(0.0).unwrap());
    assert!(next[1].recovery_lag);
    assert!(!next[1].desired_initialized);
    assert!(next[1].initial_drawable_pending);

    assert!(
        reconcile_lod_view_blend_edge_admissions(
            &[old_b],
            &[LodViewBlendEdgeAdmission {
                key: key(3),
                initial_weight: 0.0,
                record_count: 30,
                activation_requires_slew: false,
            }],
        )
        .is_err(),
        "a fractional removed edge must remain in the overlapping table"
    );
}

#[test]
fn table_replacement_seeds_common_edges_from_the_radix_proven_drawable() {
    let common_key = test_view_blend_edge_key(70);
    let drawable_state = LodViewBlendEdgeState {
        key: common_key.clone(),
        weight: LodViewBlendWeight {
            displayed: 0.4,
            desired: 0.6,
        },
        record_count: 12,
        recovery_lag: true,
        desired_initialized: true,
        initial_drawable_pending: false,
    };
    let drawable = LodDrawableViewBlendSnapshot::from_edge_states(
        std::slice::from_ref(&drawable_state),
        0.0,
        0.0,
    )
    .unwrap();

    // This newer CPU suffix has not completed compaction/radix. Neither its
    // displayed bit nor its prematurely cleared recovery state may seed the
    // replacement table.
    let mut staged = drawable_state;
    staged.weight = LodViewBlendWeight {
        displayed: 0.6,
        desired: 0.9,
    };
    staged.recovery_lag = false;
    let seed = lod_view_blend_drawable_reconciliation_seed(&[staged], &drawable).unwrap();
    assert_eq!(seed[0].weight.displayed.to_bits(), 0.4_f32.to_bits());
    assert_eq!(seed[0].weight.desired.to_bits(), 0.6_f32.to_bits());
    assert!(seed[0].recovery_lag);

    let admissions = [
        LodViewBlendEdgeAdmission {
            key: common_key,
            initial_weight: 1.0,
            record_count: 14,
            activation_requires_slew: false,
        },
        LodViewBlendEdgeAdmission {
            key: test_view_blend_edge_key(80),
            initial_weight: 0.0,
            record_count: 5,
            activation_requires_slew: false,
        },
    ];
    let mut replacement = reconcile_lod_view_blend_edge_admissions(&seed, &admissions).unwrap();
    assert_eq!(
        replacement[0].weight.displayed.to_bits(),
        0.4_f32.to_bits(),
        "a common edge must inherit the image which was actually drawable",
    );
    assert_eq!(replacement[0].weight.desired.to_bits(), 0.6_f32.to_bits());
    assert!(replacement[0].recovery_lag);
    assert_eq!(
        replacement[1].weight,
        LodViewBlendWeight::initial(0.0).unwrap(),
        "a genuinely new edge still starts at its authored endpoint",
    );

    // Checked preflight retargets desired without moving the inherited suffix.
    retarget_checked_lod_view_blend_edge_desired(&mut replacement[0], 0.9).unwrap();
    let replacement_drawable =
        LodDrawableViewBlendSnapshot::from_edge_states(&replacement, 0.0, 0.0).unwrap();
    assert_eq!(
        replacement_drawable.displayed[0].to_bits(),
        0.4_f32.to_bits()
    );
    assert_eq!(replacement_drawable.desired[0].to_bits(), 0.9_f32.to_bits());
    assert_eq!(replacement_drawable.lagging_count(), 1);
    assert!(replacement_drawable.recovery_edges[0]);
    assert_eq!(replacement_drawable.max_delta.to_bits(), 0.0_f32.to_bits());
    assert_eq!(
        replacement_drawable.weighted_record_energy.to_bits(),
        0.0_f64.to_bits(),
        "installing a new immutable table must not report the staged CPU jump as drawable energy",
    );
}

#[test]
fn multi_subview_view_blend_publication_requires_unanimous_endpoints() {
    fn edge_state(id: u64, displayed: f32, desired: f32) -> LodViewBlendEdgeState {
        LodViewBlendEdgeState {
            key: test_view_blend_edge_key(id),
            weight: LodViewBlendWeight { displayed, desired },
            record_count: 10,
            recovery_lag: displayed.to_bits() != desired.to_bits(),
            desired_initialized: true,
            initial_drawable_pending: false,
        }
    }

    let mut left = LodDrawableViewBlendSnapshot::from_edge_states(
        &[edge_state(1, 0.0, 0.0), edge_state(2, 1.0, 0.6)],
        0.08,
        0.8,
    )
    .unwrap();
    let right = LodDrawableViewBlendSnapshot::from_edge_states(
        &[edge_state(1, 1.0, 1.0), edge_state(2, 1.0, 1.0)],
        0.03,
        0.3,
    )
    .unwrap();
    left.invalid_pressure_edges = vec![true, false];
    let mut right = right;
    right.invalid_pressure_edges = vec![true, true];
    left.merge_consumer(&right).unwrap();

    assert_eq!(
        left.endpoints,
        vec![
            LodViewBlendEndpoint::Fractional,
            LodViewBlendEndpoint::ChildrenExact,
        ],
        "one parent/children disagreement must fail closed to fractional while a unanimous child endpoint may retire",
    );
    assert_eq!(left.lagging_count(), 1);
    assert_eq!(left.invalid_pressure_edges, vec![true, true]);
    assert_eq!(
        left.invalid_pressure_count(),
        2,
        "invalid pressure is reduced by per-edge OR, not summed across consumers",
    );
    assert!((left.max_lag - 0.4).abs() <= f32::EPSILON);
    assert_eq!(left.max_delta.to_bits(), 0.08_f32.to_bits());
    assert!((left.weighted_record_energy - 1.1).abs() <= f64::EPSILON);
}

#[test]
fn non_radix_bridge_candidates_fail_fast() {
    assert_eq!(
        validate_bridge_candidate_sort_mode(&SortMode::Radix),
        Ok(())
    );
    assert_eq!(
        validate_bridge_candidate_sort_mode(&SortMode::None),
        Err(LodCandidateConfigError::UnsupportedSortMode)
    );
    let host = include_str!("../lod.rs");
    assert!(host.contains("validate_bridge_candidate_sort_mode(&cloud_settings.sort_mode)"));
    assert!(host.contains(".store(LOD_RENDER_FAILED, Ordering::Release)"));
}

#[test]
fn finalize_produces_exact_bounded_draw_and_dispatch_counts() {
    let exact = finalized_indirect_args(1025, 2000, 256, 1024);
    assert_eq!(exact.instance_count, 1025);
    assert_eq!(exact.dispatch_x, 5);
    assert_eq!((exact.dispatch_c_x, exact.dispatch_c_y), (1, 2));
    assert_eq!(exact.overflow_count, 0);

    let limited = finalized_indirect_args(3000, 2000, 256, 1024);
    assert_eq!(limited.instance_count, 2000);
    assert_eq!(limited.dispatch_x, 8);
    assert_eq!(limited.overflow_count, 1000);

    let empty = finalized_indirect_args(0, 2000, 256, 1024);
    assert_eq!(empty.instance_count, 0);
    assert_eq!(empty.dispatch_x, 0);
    assert_eq!((empty.dispatch_y, empty.dispatch_z), (1, 1));
}

#[test]
fn two_level_scan_scatter_preserves_candidate_order_across_blocks() {
    fn stable_two_level_scatter(accepted: &[bool]) -> Vec<u32> {
        let group_size = LOD_COMPACTION_WORKGROUP_SIZE as usize;
        let block_size = LOD_COMPACTION_SCAN_BLOCK_SIZE as usize;
        let group_counts = accepted
            .chunks(group_size)
            .map(|group| group.iter().filter(|&&keep| keep).count())
            .collect::<Vec<_>>();
        let block_counts = group_counts
            .chunks(block_size)
            .map(|block| block.iter().sum::<usize>())
            .collect::<Vec<_>>();
        let mut block_offsets = Vec::with_capacity(block_counts.len());
        let mut total = 0usize;
        for count in block_counts {
            block_offsets.push(total);
            total += count;
        }

        let mut group_offsets = vec![0usize; group_counts.len()];
        for (block_index, block) in group_counts.chunks(block_size).enumerate() {
            let mut offset = block_offsets[block_index];
            for (local_group, &count) in block.iter().enumerate() {
                group_offsets[block_index * block_size + local_group] = offset;
                offset += count;
            }
        }

        let mut output = vec![u32::MAX; total];
        for (group_index, group) in accepted.chunks(group_size).enumerate() {
            let mut local_prefix = 0usize;
            for (local_index, &keep) in group.iter().enumerate() {
                if keep {
                    output[group_offsets[group_index] + local_prefix] =
                        (group_index * group_size + local_index) as u32;
                    local_prefix += 1;
                }
            }
        }
        output
    }

    // Cross both candidate-workgroup and scan-block boundaries, including
    // sparse, all-rejected, and adjacent accepted regions. Treating every
    // output as an equal-depth radix entry makes source order the tie-break.
    let len = (LOD_COMPACTION_WORKGROUP_SIZE * LOD_COMPACTION_SCAN_BLOCK_SIZE + 37) as usize;
    let accepted = (0..len)
        .map(|index| index % 17 == 0 || index % 251 == 250 || (65_530..65_545).contains(&index))
        .collect::<Vec<_>>();
    let expected = accepted
        .iter()
        .enumerate()
        .filter_map(|(index, &keep)| keep.then_some(index as u32))
        .collect::<Vec<_>>();
    assert_eq!(stable_two_level_scatter(&accepted), expected);

    let shader = include_str!("../lod_compaction.wgsl");
    for entry_point in [
        "fn lod_count(",
        "fn lod_scan_groups(",
        "fn lod_scan_blocks(",
        "fn lod_add_block_offsets(",
        "fn lod_scatter(",
    ] {
        assert!(shader.contains(entry_point));
    }
    assert!(shader.contains("scan_record_offset(workgroup_id.x)"));
    assert!(!shader.contains("atomicAdd(&lod_indirect.candidate_hits"));
    assert_eq!(shader.matches("@group(3) @binding(").count(), 4);
    assert_eq!(shader.matches("var<storage").count(), 3);
    assert!(!shader.contains("@group(3) @binding(4)"));
}

#[test]
fn compaction_storage_buffer_capability_is_counted_before_admission() {
    let gaussian_layout = crate::render::gaussian_storage_layout_descriptor::<
        crate::gaussian::formats::planar_3d::Gaussian3d,
    >("storage_count_contract", true);
    let compaction_entries = lod_compaction_layout_entries();
    let compaction_layout = BindGroupLayoutDescriptor::new(
        "lod_compaction_storage_count_contract",
        &compaction_entries,
    );
    let required = lod_compute_storage_buffer_count(&[gaussian_layout, compaction_layout]);
    #[cfg(feature = "precompute_covariance_3d")]
    assert_eq!(required, 9, "five planar planes plus four LoD buffers");
    #[cfg(not(feature = "precompute_covariance_3d"))]
    assert_eq!(required, 8, "four planar planes plus four LoD buffers");

    assert!(lod_storage_buffer_count_is_supported(required, required));
    assert!(!lod_storage_buffer_count_is_supported(
        required,
        required - 1
    ));

    let host = include_str!("../lod.rs");
    let prepare = host
        .split("fn prepare_lod_compaction_buffers")
        .nth(1)
        .expect("compaction admission system");
    let capability = prepare
        .find("if !storage_buffer_count_supported")
        .expect("ordinary-allocation storage-buffer capability gate");
    let allocation = prepare
        .find("plan_lod_compaction_allocation(")
        .expect("first allocation planning site");
    let request = allocation
        + prepare[allocation..]
            .find("requests.push(")
            .expect("ordinary aggregate allocation request after planning");
    assert!(capability < allocation && allocation < request);
    assert!(prepare[..request].contains("LOD_RENDER_FAILED"));
    assert!(
        prepare[..capability].contains("pinned_existing: true"),
        "an already-drawable hard-fallback hold intentionally bypasses new adapter allocation capability"
    );
}

#[test]
fn zero_candidate_path_resets_and_finalizes_without_scanning() {
    let shader = include_str!("../lod_compaction.wgsl");
    assert!(shader.contains("atomicStore(&lod_indirect.candidate_hits, 0u)"));
    assert!(shader.contains("let hits = atomicLoad(&lod_indirect.candidate_hits)"));

    let host = include_str!("../lod.rs");
    let guarded_passes = host
        .split("if state.candidate_count() > 0")
        .nth(1)
        .expect("scan passes are candidate-count guarded");
    assert!(guarded_passes.contains("pipelines.count"));
    assert!(guarded_passes.contains("pipelines.scatter"));
    assert!(guarded_passes.contains("pipelines.finalize"));
    assert!(guarded_passes.find("pipelines.finalize") > guarded_passes.find("pipelines.scatter"));

    let empty = finalized_indirect_args(0, 1, 256, 1024);
    assert_eq!(empty.instance_count, 0);
    assert_eq!(empty.dispatch_x, 0);
    assert_eq!(empty.dispatch_c_y, 0);
}

#[test]
fn candidate_list_mode_cannot_exceed_allocated_capacity() {
    assert_eq!(
        LodCompactionUniform::identity(100_000_000, 2_000_000, LodQualityEndpoint::Original, true,),
        Err(LodCandidateConfigError::IdentitySourceExceedsCapacity {
            source_count: 100_000_000,
            output_capacity: 2_000_000,
        })
    );

    let identity =
        LodCompactionUniform::identity(2_000_000, 2_000_000, LodQualityEndpoint::Original, true)
            .expect("complete identity allocation");
    assert_eq!(
        identity.with_physical_ranges(2_000_001, 1),
        Err(LodCandidateConfigError::CandidateCountExceedsCapacity {
            candidate_count: 2_000_001,
            output_capacity: 2_000_000,
        })
    );
    assert_eq!(identity.candidate_count, 2_000_000);
    assert_eq!(std::mem::size_of::<LodCompactionUniform>(), 64);
}

#[test]
fn physical_range_descriptors_replace_steady_explicit_indices() {
    let ranges = [
        LodPhysicalRange {
            node: LodNodeId(1),
            page: LodPageId(2),
            slot: AtlasSlot {
                index: 3,
                generation: 4,
            },
            physical_start: 12,
            count: 3,
        },
        LodPhysicalRange {
            node: LodNodeId(5),
            page: LodPageId(6),
            slot: AtlasSlot {
                index: 7,
                generation: 8,
            },
            physical_start: 40,
            count: 2,
        },
    ];
    let (descriptors, candidate_count) = build_gpu_physical_range_descriptors(&ranges, 64).unwrap();
    assert_eq!(candidate_count, 5);
    assert_eq!(
        descriptors,
        vec![
            LodGpuPhysicalRangeDescriptor {
                candidate_start: 0,
                physical_start: 12,
                count: 3,
                metadata: LodDebugResidency::Unknown as u32,
            },
            LodGpuPhysicalRangeDescriptor {
                candidate_start: 3,
                physical_start: 40,
                count: 2,
                metadata: LodDebugResidency::Unknown as u32,
            },
        ]
    );
    let (annotated, annotated_count) =
        build_gpu_physical_range_descriptors_with_residency(&ranges, 64, |node| {
            if node == LodNodeId(5) {
                LodDebugResidency::AncestorFallback as u32
            } else {
                LodDebugResidency::Resident as u32
            }
        })
        .unwrap();
    assert_eq!(annotated_count, candidate_count);
    assert_eq!(annotated[0].metadata, LodDebugResidency::Resident as u32);
    assert_eq!(
        annotated[1].metadata,
        LodDebugResidency::AncestorFallback as u32
    );
    let config = LodCompactionUniform::identity(64, 64, LodQualityEndpoint::Continuous, true)
        .unwrap()
        .with_physical_ranges(5, descriptors.len() as u32)
        .unwrap();
    assert_eq!(config.candidate_source_mode, LOD_CANDIDATE_SOURCE_RANGES);
    assert_eq!(config.candidate_range_count, 2);

    let expanded = descriptors
        .iter()
        .flat_map(|descriptor| {
            descriptor.physical_start..descriptor.physical_start + descriptor.count
        })
        .collect::<Vec<_>>();
    assert_eq!(expanded, [12, 13, 14, 40, 41]);

    let two_million_range = [LodPhysicalRange {
        physical_start: 0,
        count: 2_000_000,
        ..ranges[0]
    }];
    let (large_descriptors, large_count) =
        build_gpu_physical_range_descriptors(&two_million_range, 2_000_000).unwrap();
    assert_eq!(large_count, 2_000_000);
    assert_eq!(large_descriptors.len(), 1);
    assert_eq!(
        large_descriptors.len() * std::mem::size_of::<LodGpuPhysicalRangeDescriptor>(),
        16,
        "a 2M steady frontier remains O(range_count) on CPU and upload"
    );
    assert_eq!(
        candidate_binding_bytes(2_000_000, 4).unwrap(),
        candidate_evaluations_and_scan_record_bytes(2_000_000).unwrap() + 16
    );
}

#[test]
fn candidate_entries_carry_residency_without_growing_gpu_storage() {
    assert_eq!(std::mem::size_of::<LodGpuPhysicalRangeDescriptor>(), 16);
    assert_eq!(std::mem::size_of::<SortEntry>(), 8);

    let source_index = LOD_ENTRY_SOURCE_INDEX_MASK;
    let packed =
        source_index | LOD_ENTRY_MORPH_FLAG | ((LodDebugResidency::AncestorFallback as u32) << 30);
    assert_eq!(packed & LOD_ENTRY_SOURCE_INDEX_MASK, source_index);
    assert_ne!(packed & LOD_ENTRY_MORPH_FLAG, 0);
    assert_eq!(
        (packed & LOD_ENTRY_PRESENTATION_CLASS_MASK) >> LOD_ENTRY_PRESENTATION_CLASS_SHIFT,
        LodExternalActiveSetClass::FirstOnly as u32
    );
    assert_eq!(packed >> 30, LodDebugResidency::AncestorFallback as u32);

    let shader = include_str!("../lod_compaction.wgsl");
    assert!(shader.contains("let range_metadata = candidate_and_scan_words[word + 3u];"));
    assert!(shader.contains("let residency = range_metadata & 3u;"));
    assert!(shader.contains("let presentation_class = (range_metadata >> 2u) & 3u;"));
    assert!(shader.contains("Entry(key, pack_lod_entry_value(source))"));
    assert!(shader.contains("const LOD_ENTRY_SOURCE_INDEX_MASK: u32 = 0x0fffffffu;"));
    assert!(shader.contains("const LOD_ENTRY_PRESENTATION_CLASS_SHIFT: u32 = 28u;"));
    assert!(shader.contains(
        "const LOD_ENTRY_PRESENTATION_CLASS_MASK: u32 = 3u << LOD_ENTRY_PRESENTATION_CLASS_SHIFT;"
    ));
    assert!(shader.contains("const LOD_ENTRY_RESIDENCY_SHIFT: u32 = 30u;"));
}

#[test]
fn external_active_set_classes_share_the_existing_range_and_entry_words() {
    let ranges = [
        LodPhysicalRange {
            node: LodNodeId(1),
            page: LodPageId(1),
            slot: AtlasSlot {
                index: 1,
                generation: 1,
            },
            physical_start: 8,
            count: 2,
        },
        LodPhysicalRange {
            node: LodNodeId(2),
            page: LodPageId(2),
            slot: AtlasSlot {
                index: 2,
                generation: 1,
            },
            physical_start: 24,
            count: 3,
        },
        LodPhysicalRange {
            node: LodNodeId(3),
            page: LodPageId(3),
            slot: AtlasSlot {
                index: 3,
                generation: 1,
            },
            physical_start: 40,
            count: 1,
        },
    ];
    let classes = [
        LodgeMembershipClass::Shared,
        LodgeMembershipClass::FirstOnly,
        LodgeMembershipClass::SecondOnly,
    ];
    let (descriptors, candidate_count) =
        build_gpu_external_active_set_range_descriptors(&ranges, &classes, 64).unwrap();
    assert_eq!(candidate_count, 6);
    assert_eq!(descriptors.len(), ranges.len());
    for (descriptor, class) in descriptors.iter().zip(classes) {
        assert_eq!(descriptor.metadata & 3, LodDebugResidency::Resident as u32);
        assert_eq!(
            (descriptor.metadata & LOD_RANGE_PRESENTATION_CLASS_MASK)
                >> LOD_RANGE_PRESENTATION_CLASS_SHIFT,
            LodExternalActiveSetClass::from(class) as u32
        );
    }
    assert_eq!(
        build_gpu_external_active_set_range_descriptors(&ranges, &classes[..2], 64),
        Err(
            LodCandidateConfigError::ExternalActiveSetClassCountMismatch {
                range_count: 3,
                class_count: 2,
            }
        )
    );
    assert_eq!(
        LOD_RANGE_MORPH_FLAG,
        1u32 << LOD_RANGE_PRESENTATION_CLASS_SHIFT
    );
}

#[test]
fn presentation_header_preserves_exact_external_weights_and_mode_qualifies_class_one() {
    assert_eq!(std::mem::size_of::<LodPresentationHeader>(), 32);
    assert_eq!(LOD_PRESENTATION_HEADER_WORDS, 8);

    let second = f32::from_bits(0x3f40_0001);
    let first = 1.0_f32 - second;
    let external = LodPresentationHeader::external_active_set(first, second).unwrap();
    let words = external.words();
    assert_eq!(words[..5], [0; 5]);
    assert_eq!(words[5], LodPresentationMode::ExternalActiveSet as u32);
    assert_eq!(words[6], first.to_bits());
    assert_eq!(words[7], second.to_bits());
    assert_eq!(
        external.external_active_set_coefficient(LodExternalActiveSetClass::Shared),
        1.0
    );
    assert_eq!(
        external
            .external_active_set_coefficient(LodExternalActiveSetClass::FirstOnly)
            .to_bits(),
        first.to_bits()
    );
    assert_eq!(
        external
            .external_active_set_coefficient(LodExternalActiveSetClass::SecondOnly)
            .to_bits(),
        second.to_bits()
    );

    let morph = LodPresentationHeader::morph(2, 24, 3, 30, 1).words();
    assert_eq!(morph[5], LodPresentationMode::Morph as u32);
    assert_eq!(morph[6], 1.0_f32.to_bits());
    assert_eq!(morph[7], 1.0_f32.to_bits());
    assert_eq!(
        LodPresentationHeader::external_active_set(f32::NAN, 0.0),
        Err(LodCandidateConfigError::InvalidExternalActiveSetWeight)
    );
    assert_eq!(
        LodPresentationHeader::external_active_set(0.0, 1.01),
        Err(LodCandidateConfigError::InvalidExternalActiveSetWeight)
    );
    assert_eq!(
        LodPresentationHeader::external_active_set(0.2, 0.2),
        Err(LodCandidateConfigError::InvalidExternalActiveSetWeight)
    );
}

#[test]
fn external_active_set_installation_is_fail_closed_and_weight_updates_are_header_only() {
    let host = include_str!("../lod.rs");
    let update = host
        .split("pub fn update_external_active_set_weights")
        .nth(1)
        .and_then(|body| {
            body.split("pub fn install_external_active_set_candidate")
                .next()
        })
        .expect("external weight-only update primitive");
    assert!(update.contains("LodPresentationHeader::external_active_set("));
    assert!(update.contains("bytemuck::bytes_of(&header)"));
    assert!(update.contains("self.presentation_header = header"));
    assert!(!update.contains("self.mark_compute_input_dirty()"));
    assert!(!update.contains("compute_input_generation"));
    assert!(!update.contains("upload_candidate_descriptors"));

    let install = host
        .split("pub fn install_external_active_set_candidate")
        .nth(1)
        .and_then(|body| {
            body.split("pub fn upload_physical_ranges_for_testing")
                .next()
        })
        .expect("external union installation primitive");
    let validate_header = install
        .find("LodPresentationHeader::external_active_set(")
        .expect("weight validation");
    let validate_ranges = install
        .find("build_gpu_external_active_set_range_descriptors(")
        .expect("class-aware range validation");
    let mutate_header = install
        .find("render_queue.write_buffer(")
        .expect("header mutation");
    let upload_ranges = install
        .find("self.upload_candidate_descriptors(")
        .expect("shared compaction descriptor upload");
    assert!(validate_header < mutate_header);
    assert!(validate_ranges < mutate_header);
    assert!(mutate_header < upload_ranges);
    assert!(!install.contains("active_entries_buffer"));
    assert!(!install.contains("radix_scratch_buffer"));
}

#[test]
fn external_candidate_hook_bypasses_hierarchy_planning_and_preserves_weight_only_union() {
    let host = include_str!("../lod.rs");
    let prepare = host
        .split("fn prepare_lod_compaction_buffers")
        .nth(1)
        .and_then(|body| body.split("fn commit_lod_bridge_candidates").next())
        .expect("external candidate preparation branch");
    assert!(prepare.contains("candidate.is_external_active_set()"));
    assert!(prepare.contains("Some(LOD_PRESENTATION_HEADER_WORDS)"));
    assert!(prepare.contains("plan_lod_candidate_morph("));
    assert!(
        prepare
            .find("candidate.is_external_active_set()")
            .expect("external mode branch")
            < prepare
                .find("plan_lod_candidate_morph(")
                .expect("hierarchy morph planner")
    );

    let commit = host
        .split("fn commit_lod_bridge_candidates")
        .nth(1)
        .and_then(|body| body.split("fn publish_lod_view_blend_after_radix").next())
        .expect("external candidate render hook");
    for required in [
        "lod_external_active_set_weights(view, transform, presentation)",
        "synchronize_bridge_external_active_set(",
        "validate_bridge_external_active_set(candidate, presentation)",
        "lod_resident_catalog_epoch(resident_catalog_tick)",
        "candidate.is_external_active_set() && debug_required",
        "let debug_variant_count = if !candidate.is_external_active_set()",
    ] {
        assert!(
            commit.contains(required),
            "missing external hook: {required}"
        );
    }

    let synchronize = host
        .split("fn synchronize_bridge_external_active_set")
        .nth(1)
        .and_then(|body| body.split("fn validate_bridge_external_active_set").next())
        .expect("external synchronization method");
    let reuse_version = synchronize
        .split("LodCandidateUploadPlan::ReuseVersion =>")
        .nth(1)
        .and_then(|body| {
            body.split("LodCandidateUploadPlan::ReuseFingerprint")
                .next()
        })
        .expect("stable external candidate branch");
    assert!(reuse_version.contains("update_external_active_set_weights("));
    assert!(!reuse_version.contains("upload_candidate_descriptors"));
    let dirty = reuse_version
        .find("self.mark_compute_input_dirty()")
        .expect("catalog-content changes still invalidate the union");
    let content_guard = reuse_version
        .find("candidate_content_signature_changed(")
        .expect("catalog-content guard");
    assert!(content_guard < dirty);
}

#[test]
fn identical_physical_range_can_have_exact_per_view_residency() {
    let range = [LodPhysicalRange {
        node: LodNodeId(9),
        page: LodPageId(4),
        slot: AtlasSlot {
            index: 2,
            generation: 7,
        },
        physical_start: 32,
        count: 8,
    }];
    let (resident_view, _) =
        build_gpu_physical_range_descriptors_with_residency(&range, 64, |_| {
            LodDebugResidency::Resident as u32
        })
        .unwrap();
    let (fallback_view, _) =
        build_gpu_physical_range_descriptors_with_residency(&range, 64, |_| {
            LodDebugResidency::AncestorFallback as u32
        })
        .unwrap();
    assert_eq!(
        resident_view[0].physical_start,
        fallback_view[0].physical_start
    );
    assert_eq!(resident_view[0].count, fallback_view[0].count);
    assert_eq!(
        resident_view[0].metadata,
        LodDebugResidency::Resident as u32
    );
    assert_eq!(
        fallback_view[0].metadata,
        LodDebugResidency::AncestorFallback as u32
    );
}

#[test]
fn one_physical_range_per_candidate_fits_the_planned_prefix() {
    const CAPACITY: u32 = 2_000_000;
    let config =
        LodCompactionUniform::identity(CAPACITY, CAPACITY, LodQualityEndpoint::Continuous, true)
            .unwrap()
            .with_physical_ranges(CAPACITY, CAPACITY)
            .expect("one positive-count range per candidate is representable");
    assert_eq!(config.candidate_range_count, CAPACITY);

    let maximum_words = maximum_candidate_source_words(u64::from(CAPACITY)).unwrap();
    assert_eq!(maximum_words, u64::from(CAPACITY) * 4);
    assert_eq!(
        candidate_and_scan_record_bytes(u64::from(CAPACITY)).unwrap(),
        candidate_binding_bytes(u64::from(CAPACITY), maximum_words).unwrap()
    );
    assert_eq!(
        config.with_physical_ranges(CAPACITY, CAPACITY + 1),
        Err(
            LodCandidateConfigError::PhysicalRangeDescriptorCapacityExceeded {
                range_count: CAPACITY + 1,
                descriptor_capacity: CAPACITY,
            }
        )
    );
}

#[test]
fn missing_bridge_candidates_remain_fail_closed_for_production_ownership() {
    assert_eq!(
        readiness_without_bridge_candidate(
            LodCompactionReadiness::Ready,
            LodCandidateOwnership::Bridge,
        ),
        LodCompactionReadiness::AwaitingCandidates,
    );
    assert_eq!(
        readiness_without_bridge_candidate(
            LodCompactionReadiness::PendingCandidates,
            LodCandidateOwnership::Bridge,
        ),
        LodCompactionReadiness::AwaitingCandidates,
    );
    assert_eq!(
        LodCandidateOwnership::default(),
        LodCandidateOwnership::Bridge
    );
}

#[cfg(feature = "testing")]
#[test]
fn testing_range_ownership_survives_an_absent_bridge_until_explicitly_revoked() {
    let readiness = readiness_without_bridge_candidate(
        LodCompactionReadiness::Ready,
        LodCandidateOwnership::TestingPhysicalRanges,
    );
    assert_eq!(
        readiness,
        LodCompactionReadiness::Ready,
        "an unchanged testing-managed physical-range frontier remains executable"
    );
    assert_eq!(
        readiness_without_bridge_candidate(readiness, LodCandidateOwnership::Bridge),
        LodCompactionReadiness::AwaitingCandidates,
        "a real bridge candidate or explicit invalidation reclaims fail-closed ownership"
    );

    let host = include_str!("../lod.rs");
    let testing_upload = host
        .find("pub fn upload_physical_ranges_for_testing")
        .expect("testing range upload API");
    let testing_upload = &host[testing_upload..];
    let next_method = testing_upload
        .find("pub fn configure_sort_dispatch")
        .expect("end of testing range upload API");
    assert!(
        testing_upload[..next_method]
            .contains("self.candidate_ownership = LodCandidateOwnership::TestingPhysicalRanges")
    );
    assert!(
        testing_upload[..next_method]
            .contains("self.candidate_upload.revoke_for_testing_override()")
    );

    let range = LodPhysicalRange {
        node: LodNodeId(1),
        page: LodPageId(2),
        slot: AtlasSlot {
            index: 3,
            generation: 4,
        },
        physical_start: 12,
        count: 3,
    };
    let fingerprint = lod_candidate_parts_fingerprint(7, &[range], 3);
    let production_version = Arc::new(AtomicU8::new(LOD_RENDER_ACTIVE));
    let mut tracker = LodCandidateUploadTracker::default();
    tracker.mark_synchronized(&production_version, fingerprint);
    assert_eq!(
        tracker.plan_fingerprint(&production_version, fingerprint),
        LodCandidateUploadPlan::ReuseVersion
    );

    tracker.revoke_for_testing_override();
    assert_eq!(
        tracker.plan_fingerprint(&production_version, fingerprint),
        LodCandidateUploadPlan::Upload(fingerprint),
        "the same real bridge version must upload after a testing override"
    );
}

#[test]
fn candidate_prefix_grows_once_and_replacement_is_peak_bounded() {
    let tail = candidate_evaluations_and_scan_record_bytes(2_000_000).unwrap();
    let stable_bytes = candidate_binding_bytes(2_000_000, 4).unwrap();
    let maximum_source_words = maximum_candidate_source_words(2_000_000).unwrap();
    let maximum_prefix_bytes = candidate_binding_bytes(2_000_000, maximum_source_words).unwrap();
    assert_eq!(stable_bytes, tail + 16);
    assert_eq!(maximum_prefix_bytes, tail + 32_000_000);
    assert!(stable_bytes < maximum_prefix_bytes);
    assert_eq!(candidate_source_capacity_after_upload(4, 4, 8_000_000), 4);
    assert_eq!(
        candidate_source_capacity_after_upload(4, 5, 8_000_000),
        8_000_000,
        "the first non-trivial prefix grows directly to the admitted maximum"
    );
    assert_eq!(
        candidate_source_capacity_after_upload(8_000_000, 4, 8_000_000),
        8_000_000,
        "range frontiers retain peak prefix capacity until state destruction"
    );
    assert_eq!(
        candidate_source_capacity_after_upload(8_000_000, 1_000_000, 8_000_000),
        8_000_000,
        "descriptor churn cannot allocate another generation"
    );

    let host = include_str!("../lod.rs");
    let resize = host
        .find("fn resize_candidate_source_prefix")
        .expect("candidate prefix resize implementation");
    let resize = &host[resize..];
    let drop_bind_group = resize
        .find("let old_bind_group = self.bind_group.take()")
        .expect("dependent bind group drop");
    let drop_old_buffer = resize
        .find("let old_candidate_and_scan_buffer = self.candidate_and_scan_buffer.take()")
        .expect("old candidate/evaluation buffer drop");
    let replacement = resize
        .find("let candidate_and_scan_buffer = render_device.create_buffer")
        .expect("replacement allocation");
    assert!(drop_bind_group < drop_old_buffer && drop_old_buffer < replacement);
    assert!(!host.contains("LodGpuCandidateStorageStats"));
    assert!(!host.contains("candidate_storage_stats"));

    let plan = plan_lod_compaction_allocation(
        2_000_000,
        256 * 1024 * 1024,
        128 * 1024 * 1024,
        64 * 1024,
        u32::MAX,
    )
    .unwrap();
    assert_eq!(
        plan.candidate_replacement_reserve_bytes, stable_bytes,
        "aggregate admission must reserve the sole initial predecessor"
    );
    assert!(
        plan.total_bytes >= maximum_prefix_bytes + stable_bytes,
        "the admitted peak includes old and replacement candidate bindings"
    );
    let invalidate = host
        .find("pub fn invalidate_candidates")
        .expect("candidate invalidation implementation");
    let invalidate = &host[invalidate..];
    let next_method = invalidate
        .find("fn synchronize_pipeline_readiness")
        .expect("end of invalidation method");
    assert!(
        !invalidate[..next_method].contains("resize_candidate_source_prefix"),
        "normal invalidation must not shrink and recreate the full-tail binding"
    );
    assert!(invalidate[..next_method].contains("self.morph_buffer"));
    assert!(invalidate[..next_method].contains("LodCandidateUploadTracker::default()"));
}

#[test]
fn morph_buffer_admission_charges_base_residency_and_growth_overlap_exactly() {
    let base_total = 1_000_u64;
    assert_eq!(LOD_MORPH_MIN_BUFFER_BYTES, 32);
    assert_eq!(lod_morph_word_capacity(8).unwrap(), 8);
    assert_eq!(lod_morph_word_capacity(9).unwrap(), 16);
    assert_eq!(lod_morph_word_capacity(17).unwrap(), 32);

    assert_eq!(
        lod_compaction_admission_bytes_with_morph(base_total, 8, 8),
        Some(base_total),
        "the allocation plan already owns the minimum header"
    );
    assert_eq!(
        lod_compaction_admission_bytes_with_morph(base_total, 8, 9),
        Some(base_total + 64),
        "first growth charges the retained 32-byte base plus the new 64-byte buffer"
    );
    assert_eq!(
        lod_compaction_admission_bytes_with_morph(base_total, 16, 17),
        Some(base_total - 32 + 64 + 128),
        "grow-only replacement charges both current and next power-of-two buffers"
    );
    assert_eq!(
        lod_compaction_admission_bytes_with_morph(base_total, 64, 8),
        Some(base_total - 32 + 256),
        "a settled hard cut still charges its resident grow-only morph allocation"
    );

    let plan =
        plan_lod_compaction_allocation(2_000, 1024 * 1024, 8_192, 64 * 1024, u32::MAX).unwrap();
    assert_eq!(plan.morph_base_bytes, LOD_MORPH_MIN_BUFFER_BYTES);
    let first = lod_compaction_admission_bytes_with_morph(plan.total_bytes, 8, 9).unwrap();
    let second = lod_compaction_admission_bytes_with_morph(plan.total_bytes, 8, 9).unwrap();
    let phase_a = AtomicU8::new(LOD_RENDER_WAITING);
    let phase_b = AtomicU8::new(LOD_RENDER_WAITING);
    let admitted = admit_lod_compaction_requests(
        vec![
            LodCompactionAdmissionRequest {
                payload: 1_u8,
                total_bytes: first,
                class: LodCompactionAdmissionClass::FallbackCapable,
                required_phase: Some(&phase_a),
                pinned_existing: false,
            },
            LodCompactionAdmissionRequest {
                payload: 2_u8,
                total_bytes: second,
                class: LodCompactionAdmissionClass::FallbackCapable,
                required_phase: Some(&phase_b),
                pinned_existing: false,
            },
        ],
        first + second - 1,
    );
    assert_eq!(admitted, vec![1]);
    assert_eq!(phase_a.load(Ordering::Acquire), LOD_RENDER_WAITING);
    assert_eq!(phase_b.load(Ordering::Acquire), LOD_RENDER_FAILED);

    let host = include_str!("../lod.rs");
    let prepare = host
        .split("fn prepare_lod_compaction_buffers")
        .nth(1)
        .expect("compaction admission system");
    assert!(prepare.contains("lod_compaction_admission_bytes_with_morph"));
    assert!(prepare.contains("total_bytes: admission_total_bytes"));
}

#[test]
fn static_signature_skips_and_every_compute_generation_input_invalidates() {
    let view = test_extracted_view();
    let transform = GlobalTransform::IDENTITY;
    let settings = CloudSettings::default();
    let static_signature = compaction_signature(7, &view, &transform, &settings, 11);
    assert_eq!(
        static_signature,
        compaction_signature(7, &view, &transform, &settings, 11),
        "an unchanged view/frontier/transform/storage tuple must hit the complete skip"
    );

    let mut moving_view = test_extracted_view();
    moving_view.world_from_view = GlobalTransform::from(Transform::from_translation(Vec3::X));
    assert_ne!(
        static_signature,
        compaction_signature(7, &moving_view, &transform, &settings, 11),
        "camera motion must invalidate compaction and sorting"
    );
    assert_ne!(
        static_signature,
        compaction_signature(
            7,
            &view,
            &GlobalTransform::from(Transform::from_scale(Vec3::splat(2.0))),
            &settings,
            11,
        ),
        "cloud transform changes must invalidate compaction and sorting"
    );
    assert_ne!(
        static_signature,
        compaction_signature(8, &view, &transform, &settings, 11),
        "frontier/config generation changes must invalidate compaction and sorting"
    );
    assert_ne!(
        static_signature,
        compaction_signature(7, &view, &transform, &settings, 12),
        "same-sized GPU storage replacement must invalidate cached results"
    );
}

#[test]
fn live_gpu_writers_force_consecutive_compaction_generations() {
    assert!(lod_compaction_cache_allowed(false, false));
    assert!(!lod_compaction_cache_allowed(true, false));
    assert!(!lod_compaction_cache_allowed(false, true));
    assert!(!lod_compaction_cache_allowed(true, true));

    let view = test_extracted_view();
    let transform = GlobalTransform::IDENTITY;
    let settings = CloudSettings::default();
    let first = compaction_signature(7, &view, &transform, &settings, 11);
    let second = compaction_signature(8, &view, &transform, &settings, 11);
    let third = compaction_signature(9, &view, &transform, &settings, 11);
    assert_ne!(first, second);
    assert_ne!(second, third);

    let host = include_str!("../lod.rs");
    let run = host
        .split("fn run_lod_compaction")
        .nth(1)
        .expect("LoD compaction runner");
    assert!(run.contains("With<GaussianInterpolate<R>>"));
    assert!(run.contains("With<ParticleBehaviorsHandle>"));
    assert!(run.contains("if !lod_compaction_cache_allowed(has_interpolate, has_particles)"));
    assert!(run.contains("state.mark_compute_input_dirty();"));
}

#[test]
fn committed_frontier_content_epoch_invalidates_only_the_synchronized_descriptor() {
    assert!(candidate_content_signature_changed(None, 7));
    assert!(!candidate_content_signature_changed(Some(7), 7));
    assert!(candidate_content_signature_changed(Some(7), 8));
    let fingerprint = LodCandidateFrontierFingerprint {
        primary: 1,
        secondary: 2,
        range_count: 1,
        candidate_count: 4,
    };
    assert!(!candidate_content_signature_must_refresh(
        LodCandidateUploadPlan::ReuseVersion,
        Some(11),
        11,
        Some(7),
    ));
    assert!(candidate_content_signature_must_refresh(
        LodCandidateUploadPlan::ReuseVersion,
        Some(11),
        12,
        Some(7),
    ));
    assert!(candidate_content_signature_must_refresh(
        LodCandidateUploadPlan::Upload(fingerprint),
        Some(11),
        11,
        Some(7),
    ));

    let host = include_str!("../lod.rs");
    let retain_current = host
        .split("LodBridgeAtlasDecision::RetainCurrent =>")
        .nth(1)
        .expect("pending-candidate retention branch")
        .split("LodBridgeAtlasDecision::SynchronizePending =>")
        .next()
        .unwrap();
    assert!(
        !retain_current.contains("frontier_content_signature"),
        "pending ranges must not dirty the retained descriptor"
    );

    let synchronized = host
        .split("LodBridgeAtlasDecision::SynchronizePending =>")
        .nth(1)
        .expect("candidate synchronization branch")
        .split("#[derive(Clone, Copy, Debug, Eq, PartialEq)]")
        .next()
        .unwrap();
    assert!(synchronized.contains("atlas_generations.content_revision(atlas)"));
    assert!(
        synchronized
            .contains(".frontier_content_signature(atlas, candidate.required_atlas_ranges())")
    );

    let reuse_version = host
        .split("LodCandidateUploadPlan::ReuseVersion =>")
        .nth(1)
        .expect("same-candidate synchronization branch")
        .split("LodCandidateUploadPlan::ReuseFingerprint")
        .next()
        .unwrap();
    assert!(reuse_version.contains("candidate_content_signature_changed"));
    assert!(reuse_version.contains("self.mark_compute_input_dirty()"));

    let run = host
        .split("fn run_lod_compaction")
        .nth(1)
        .expect("LoD compaction runner");
    assert!(
        !run.contains("content_revision("),
        "LoD must not invalidate every camera for unrelated atlas writes"
    );
}

#[test]
fn first_pass_evaluation_is_cached_for_scatter() {
    let shader = include_str!("../lod_compaction.wgsl");
    assert_eq!(shader.matches("evaluate_candidate(global_id.x)").count(), 1);
    assert!(shader.contains("store_candidate_evaluation(global_id.x, evaluation)"));
    assert!(shader.contains("load_candidate_evaluation(global_id.x)"));
    assert!(shader.contains("candidate_from_physical_ranges(candidate_offset)"));
    assert!(shader.contains("lod_config.transform_scale_bound"));
    assert_eq!(
        shader
            .matches("if global_id.x < lod_config.candidate_count")
            .count(),
        2,
        "padded count/scatter lanes must not address cached evaluations out of bounds"
    );
}
#[test]
fn allocation_plan_preserves_requested_capacity_when_device_limits_fit() {
    let plan = plan_lod_compaction_allocation(
        2_000_000,
        256 * 1024 * 1024,
        128 * 1024 * 1024,
        64 * 1024,
        u32::MAX,
    )
    .unwrap();
    assert_eq!(plan.effective_capacity, 2_000_000);
    assert_eq!(plan.config_bytes, 64);
    assert_eq!(plan.candidate_indices_bytes, 32_000_000);
    assert_eq!(plan.candidate_evaluations_bytes, 16_000_000);
    assert_eq!(plan.scan_group_count, 7_813);
    assert_eq!(plan.scan_block_count, 31);
    assert_eq!(plan.scan_records_bytes, 62_752);
    assert_eq!(plan.candidate_and_scan_records_bytes, 48_062_752);
    assert_eq!(plan.active_entries_bytes, 16_000_000);
    assert_eq!(plan.radix_scratch_bytes, 16_000_000);
    assert_eq!(plan.indirect_args_bytes, LOD_INDIRECT_ARGS_SIZE);
    assert_eq!(plan.sorting_pass_bytes, 4);
    assert_eq!(
        plan.total_bytes,
        checked_lod_compaction_total_bytes([
            plan.config_bytes,
            plan.candidate_and_scan_records_bytes,
            plan.candidate_replacement_reserve_bytes,
            plan.active_entries_bytes,
            plan.radix_scratch_bytes,
            plan.sorting_global_bytes,
            plan.sorting_status_counter_bytes,
            plan.sorting_pass_bytes * 4,
            plan.indirect_args_bytes,
            plan.morph_base_bytes,
        ])
        .unwrap()
    );
    assert_eq!(
        plan.sorting_status_counter_bytes,
        ShaderDefines::default().sorting_status_counters_buffer_size(2_000_000) as u64
    );
    for bytes in [
        plan.candidate_and_scan_records_bytes,
        plan.active_entries_bytes,
        plan.radix_scratch_bytes,
        plan.sorting_global_bytes,
        plan.sorting_status_counter_bytes,
        plan.indirect_args_bytes,
    ] {
        assert!(bytes <= 128 * 1024 * 1024);
        assert!(bytes <= 256 * 1024 * 1024);
    }
}

#[test]
fn allocation_plan_clamps_each_capacity_dependent_storage_buffer() {
    let storage_limited =
        plan_lod_compaction_allocation(2_000, 1024 * 1024, 8_192, 64 * 1024, u32::MAX).unwrap();
    assert_eq!(storage_limited.effective_capacity, 340);
    assert_eq!(storage_limited.candidate_indices_bytes, 5_440);
    assert_eq!(storage_limited.candidate_evaluations_bytes, 2_720);
    assert_eq!(storage_limited.scan_records_bytes, 24);
    assert_eq!(storage_limited.candidate_and_scan_records_bytes, 8_184);
    assert_eq!(storage_limited.active_entries_bytes, 2_720);
    assert_eq!(storage_limited.radix_scratch_bytes, 2_720);
    assert_eq!(storage_limited.sorting_status_counter_bytes, 1_024);

    let buffer_limited =
        plan_lod_compaction_allocation(2_000, 6_000, 1024 * 1024, 64 * 1024, u32::MAX).unwrap();
    assert_eq!(buffer_limited.effective_capacity, 249);
    assert_eq!(buffer_limited.scan_records_bytes, 16);
    assert_eq!(buffer_limited.candidate_and_scan_records_bytes, 5_992);
    assert_eq!(buffer_limited.active_entries_bytes, 1_992);
    assert_eq!(buffer_limited.radix_scratch_bytes, 1_992);
    assert!(buffer_limited.sorting_global_bytes <= 6_000);
    assert!(buffer_limited.sorting_status_counter_bytes <= 6_000);
}

#[test]
fn allocation_plan_caps_direct_filter_dispatch_to_device_limit() {
    let max_workgroups = 65_535;
    let plan =
        plan_lod_compaction_allocation(20_000_000, u64::MAX, u64::MAX, u64::MAX, max_workgroups)
            .unwrap();
    assert_eq!(
        plan.effective_capacity,
        max_workgroups * LOD_COMPACTION_WORKGROUP_SIZE
    );
    assert_eq!(
        plan.effective_capacity
            .div_ceil(LOD_COMPACTION_WORKGROUP_SIZE),
        max_workgroups
    );
    assert_eq!(
        plan_lod_compaction_allocation(1, u64::MAX, u64::MAX, u64::MAX, 0),
        Err(LodCompactionAllocationError::ZeroComputeDispatchCapacity)
    );
}

#[test]
fn allocation_plan_caps_the_bounded_two_level_scan_topology() {
    assert_eq!(max_candidate_capacity_for_combined_storage(39), 0);
    assert_eq!(max_candidate_capacity_for_combined_storage(40), 1);
    assert_eq!(max_candidate_capacity_for_combined_storage(6_159), 255);
    assert_eq!(max_candidate_capacity_for_combined_storage(6_160), 256);
    assert_eq!(max_candidate_capacity_for_combined_storage(6_191), 256);
    assert_eq!(max_candidate_capacity_for_combined_storage(6_192), 257);

    let storage_limit = 64 * 1024 * 1024;
    let boundary = max_candidate_capacity_for_combined_storage(storage_limit);
    assert!(candidate_and_scan_record_bytes(boundary).unwrap() <= storage_limit);
    assert!(candidate_and_scan_record_bytes(boundary + 1).unwrap() > storage_limit);

    let plan =
        plan_lod_compaction_allocation(u32::MAX, u64::MAX, u64::MAX, u64::MAX, u32::MAX).unwrap();
    assert_eq!(
        plan.effective_capacity,
        LOD_COMPACTION_MAX_CANDIDATE_WORKGROUPS * LOD_COMPACTION_WORKGROUP_SIZE
    );
    assert_eq!(
        plan.scan_group_count,
        LOD_COMPACTION_MAX_CANDIDATE_WORKGROUPS
    );
    assert_eq!(plan.scan_block_count, LOD_COMPACTION_MAX_SCAN_BLOCKS);
    assert_eq!(
        plan.scan_records_bytes,
        u64::from(LOD_COMPACTION_MAX_CANDIDATE_WORKGROUPS + LOD_COMPACTION_MAX_SCAN_BLOCKS) * 8
    );
    assert_eq!(
        plan.candidate_and_scan_records_bytes,
        u64::from(plan.effective_capacity) * 24 + plan.scan_records_bytes
    );
}

#[test]
fn allocation_plan_rejects_fixed_buffers_before_gpu_allocation() {
    let config_bytes = std::mem::size_of::<LodCompactionUniform>() as u64;
    assert_eq!(
        plan_lod_compaction_allocation(1, u64::MAX, u64::MAX, config_bytes - 1, u32::MAX,),
        Err(LodCompactionAllocationError::UniformBindingSizeLimit {
            buffer: LodCompactionBufferRole::Config,
            required: config_bytes,
            limit: config_bytes - 1,
        })
    );
    assert_eq!(
        plan_lod_compaction_allocation(1, u64::MAX, 47, u64::MAX, u32::MAX),
        Err(LodCompactionAllocationError::StorageBindingSizeLimit {
            buffer: LodCompactionBufferRole::IndirectArgs,
            required: LOD_INDIRECT_ARGS_SIZE,
            limit: 47,
        })
    );
    assert_eq!(
        plan_lod_compaction_allocation(0, u64::MAX, u64::MAX, u64::MAX, u32::MAX),
        Err(LodCompactionAllocationError::ZeroRequestedCapacity)
    );
    assert_eq!(
        checked_lod_compaction_total_bytes([u64::MAX, 1]),
        Err(LodCompactionAllocationError::SizeOverflow(
            LodCompactionBufferRole::Aggregate,
        ))
    );
}

#[test]
fn aggregate_budget_is_device_bounded_and_reservations_are_checked() {
    assert_eq!(
        effective_lod_compaction_aggregate_budget(
            DEFAULT_LOD_COMPACTION_AGGREGATE_BYTES,
            128 * 1024 * 1024,
        ),
        256 * 1024 * 1024
    );
    assert_eq!(
        effective_lod_compaction_aggregate_budget(64 * 1024 * 1024, u64::MAX),
        64 * 1024 * 1024
    );

    let mut used = 0;
    assert!(reserve_lod_compaction_bytes(&mut used, 40, 64));
    assert_eq!(used, 40);
    assert!(!reserve_lod_compaction_bytes(&mut used, 25, 64));
    assert_eq!(used, 40);
    assert!(reserve_lod_compaction_bytes(&mut used, 24, 64));
    assert_eq!(used, 64);

    let mut nearly_overflowed = u64::MAX;
    assert!(!reserve_lod_compaction_bytes(
        &mut nearly_overflowed,
        1,
        u64::MAX
    ));
    assert_eq!(nearly_overflowed, u64::MAX);
}

#[test]
fn undersized_budget_retains_active_package_and_fails_cold_package() {
    let cold_phase = Arc::new(AtomicU8::new(LOD_RENDER_WAITING));
    let active_phase = Arc::new(AtomicU8::new(LOD_RENDER_ACTIVE));
    let requests = vec![
        // Identity order deliberately places the cold package first. Admission
        // must not let that fixed prefix evict the already-drawable package.
        LodCompactionAdmissionRequest {
            payload: "cold",
            total_bytes: 64,
            class: LodCompactionAdmissionClass::RequiredOutput,
            required_phase: Some(cold_phase.as_ref()),
            pinned_existing: false,
        },
        LodCompactionAdmissionRequest {
            payload: "active",
            total_bytes: 64,
            class: LodCompactionAdmissionClass::RetainedRequiredOutput,
            required_phase: Some(active_phase.as_ref()),
            pinned_existing: false,
        },
    ];

    let admitted = admit_lod_compaction_requests(requests, 64);

    assert_eq!(admitted, ["active"]);
    assert_eq!(active_phase.load(Ordering::Acquire), LOD_RENDER_ACTIVE);
    assert_eq!(cold_phase.load(Ordering::Acquire), LOD_RENDER_FAILED);
}

#[test]
fn hard_fallback_pins_existing_state_above_a_lowered_budget_and_rejects_new_work() {
    let held_phase = AtomicU8::new(LOD_RENDER_WAITING);
    let competing_phase = AtomicU8::new(LOD_RENDER_WAITING);
    let admitted = admit_lod_compaction_requests(
        vec![
            LodCompactionAdmissionRequest {
                payload: "competing",
                total_bytes: 1,
                class: LodCompactionAdmissionClass::RetainedRequiredOutput,
                required_phase: Some(&competing_phase),
                pinned_existing: false,
            },
            LodCompactionAdmissionRequest {
                payload: "held",
                total_bytes: 64,
                class: LodCompactionAdmissionClass::RetainedRequiredOutput,
                required_phase: Some(&held_phase),
                pinned_existing: true,
            },
        ],
        0,
    );

    assert_eq!(admitted, ["held"]);
    assert_eq!(held_phase.load(Ordering::Acquire), LOD_RENDER_WAITING);
    assert_eq!(competing_phase.load(Ordering::Acquire), LOD_RENDER_FAILED);
}

#[test]
fn crossed_replacements_drop_all_old_states_before_allocating() {
    #[derive(Clone, Copy)]
    enum Event {
        ReleaseDependent(usize),
        DropOld(usize),
        AllocateNew(usize),
    }

    fn peak_live_bytes(old: &[u64], new: &[u64], events: &[Event]) -> Option<u64> {
        let mut dependent_live = vec![true; old.len()];
        let mut old_live = vec![true; old.len()];
        let mut new_live = vec![false; new.len()];
        let mut live = old
            .iter()
            .try_fold(0u64, |sum, bytes| sum.checked_add(*bytes))?;
        let mut peak = live;

        for event in events {
            match *event {
                Event::ReleaseDependent(index) => dependent_live[index] = false,
                Event::DropOld(index) => {
                    if dependent_live[index] || !old_live[index] {
                        return None;
                    }
                    old_live[index] = false;
                    live = live.checked_sub(old[index])?;
                }
                Event::AllocateNew(index) => {
                    if old_live[index] || new_live[index] {
                        return None;
                    }
                    new_live[index] = true;
                    live = live.checked_add(new[index])?;
                    peak = peak.max(live);
                }
            }
        }
        Some(peak)
    }

    // Both steady-state totals fit 100 bytes. Growing key 0 before key 1's
    // old allocation is dropped creates a 160-byte transient peak.
    let old = [20, 80];
    let new = [80, 20];
    let interleaved = [
        Event::ReleaseDependent(0),
        Event::DropOld(0),
        Event::AllocateNew(0),
        Event::ReleaseDependent(1),
        Event::DropOld(1),
        Event::AllocateNew(1),
    ];
    assert_eq!(peak_live_bytes(&old, &new, &interleaved), Some(160));

    let two_phase = [
        Event::ReleaseDependent(0),
        Event::ReleaseDependent(1),
        Event::DropOld(0),
        Event::DropOld(1),
        Event::AllocateNew(0),
        Event::AllocateNew(1),
    ];
    assert_eq!(peak_live_bytes(&old, &new, &two_phase), Some(100));
    assert_eq!(
        peak_live_bytes(&old, &new, &[Event::DropOld(0)]),
        None,
        "a state cannot release memory while its radix group retains it"
    );

    let host = include_str!("../lod.rs");
    let dependent_drop = host
        .find("for key in &recreate_keys {\n        radix_groups.remove(key);")
        .expect("dependent pre-drop phase");
    let state_drop = host
        .find("for key in &recreate_keys {\n        buffers.entries.remove(key);")
        .expect("state pre-drop phase");
    let allocation = host
        .find("if recreate_keys.contains(&key) {")
        .expect("replacement allocation phase");
    assert!(dependent_drop < state_drop && state_drop < allocation);
}

#[test]
fn device_clamping_never_turns_a_flat_identity_into_a_partial_draw() {
    let plan =
        plan_lod_compaction_allocation(2_000, 1024 * 1024, 8_192, 64 * 1024, u32::MAX).unwrap();
    let (flat, readiness) = LodCompactionUniform::initial(
        2_000,
        plan.effective_capacity,
        LodQualityEndpoint::Original,
        true,
    );
    assert_eq!(plan.effective_capacity, 340);
    assert_eq!(readiness, LodCompactionReadiness::AwaitingCandidates);
    assert_eq!(flat.candidate_count, 0);
    assert_eq!(flat.candidate_source_mode, LOD_CANDIDATE_SOURCE_RANGES);
    assert_eq!(
        LodCompactionUniform::identity(
            2_000,
            plan.effective_capacity,
            LodQualityEndpoint::Original,
            true,
        ),
        Err(LodCandidateConfigError::IdentitySourceExceedsCapacity {
            source_count: 2_000,
            output_capacity: 340,
        })
    );
    assert_eq!(
        build_gpu_physical_range_descriptors(&[], LOD_ENTRY_MAX_SOURCE_COUNT + 1),
        Err(LodCandidateConfigError::SourceIndexExceedsEntryEncoding {
            source_count: LOD_ENTRY_MAX_SOURCE_COUNT + 1,
            max_source_count: LOD_ENTRY_MAX_SOURCE_COUNT,
        })
    );
    let bounded_frontier = flat.with_physical_ranges(340, 1).unwrap();
    assert_eq!(bounded_frontier.candidate_count, 340);
    assert_eq!(
        flat.with_physical_ranges(341, 1),
        Err(LodCandidateConfigError::CandidateCountExceedsCapacity {
            candidate_count: 341,
            output_capacity: 340,
        })
    );
}

#[test]
fn frustum_policy_is_explicit_and_compaction_preserves_draw_mode_metadata() {
    let mut settings = GaussianLodSettings {
        quality: 0.5,
        frustum_culling: true,
        frustum_margin: 2.5,
        ..default()
    };
    let enabled = LodCompactionUniform::identity(1, 1, LodQualityEndpoint::Continuous, true)
        .expect("one-entry identity")
        .with_policy(LodCompactionPolicy::hierarchy(&settings));
    let disabled = LodCompactionUniform::identity(1, 1, LodQualityEndpoint::Continuous, false)
        .expect("one-entry identity");
    assert_eq!(enabled.frustum_culling, 1);
    assert_eq!(enabled.frustum_margin, 2.5);
    assert_eq!(disabled.frustum_culling, 0);
    assert_eq!(
        std::mem::offset_of!(LodCompactionUniform, frustum_margin),
        32
    );
    assert_eq!(std::mem::size_of::<LodCompactionUniform>(), 64);
    assert_eq!(
        std::mem::offset_of!(LodCompactionUniform, candidate_range_count),
        36
    );

    settings.frustum_margin = f32::NAN;
    let sanitized = enabled.with_policy(LodCompactionPolicy::hierarchy(&settings));
    assert_eq!(sanitized.frustum_margin, 0.0);

    let shader = include_str!("../lod_compaction.wgsl");
    assert!(shader.contains("lod_config.frustum_culling != 0u"));
    assert!(shader.contains("support_sphere_in_frustum(transformed_position, support_radius)"));
    assert!(shader.contains("view.frustum[plane_index]"));
    assert!(shader.contains("lod_config.frustum_margin"));
    assert!(shader.contains("gaussian_mip_support_radius_world(position_world, 3.0)"));
    assert!(shader.contains("authored_radius_world + mip_radius_world"));
    assert!(!shader.contains("get_visibility"));
    assert!(!shader.contains("visibility <= 0.0"));
    assert!(!shader.contains("lod_morph_visibility"));
    assert!(!shader.contains("min_projected_radius_px"));
    assert!(!shader.contains("min_projected_opacity"));
    assert!(!shader.contains("!in_frustum(clip_space_pos.xyz)"));
}

#[test]
fn support_sphere_visibility_keeps_large_edge_splats() {
    fn sphere_visible(center: Vec3, radius: f32, margin: f32, planes: &[Vec4; 6]) -> bool {
        planes.iter().all(|plane| {
            let normal = plane.truncate();
            normal.dot(center) + plane.w >= -(radius + margin.max(0.0)) * normal.length()
        })
    }

    // The first plane represents x >= -1. The center lies outside, but a
    // sufficiently large support sphere still intersects the frustum.
    let planes = [
        Vec4::new(1.0, 0.0, 0.0, 1.0),
        Vec4::ZERO,
        Vec4::ZERO,
        Vec4::ZERO,
        Vec4::ZERO,
        Vec4::ZERO,
    ];
    let center = Vec3::new(-1.5, 0.0, 0.0);
    assert!(sphere_visible(center, 0.6, 0.0, &planes));
    assert!(!sphere_visible(center, 0.4, 0.0, &planes));
    assert!(sphere_visible(center, 0.4, 0.1, &planes));

    // The fixed projected filter has a 3*sqrt(1.2) shader-unit footprint.
    // Convert it at perspective depth exactly as the shared WGSL helper does:
    // a tiny authored sphere alone misses the edge, while filtered support
    // still overlaps and must remain a compaction candidate.
    let mip_radius = 3.0_f32 * 1.2_f32.sqrt() * 5.0 / 300.0;
    let tiny_center = Vec3::new(-1.015, 0.0, 0.0);
    let authored_radius = 3.0 * 0.001;
    assert!(!sphere_visible(tiny_center, authored_radius, 0.0, &planes));
    assert!(sphere_visible(
        tiny_center,
        authored_radius + mip_radius,
        0.0,
        &planes
    ));
}

#[test]
fn quality_one_policy_preserves_the_complete_identity_frontier() {
    let settings = GaussianLodSettings {
        quality: 1.0,
        frustum_margin: 1.0,
        ..default()
    };
    let identity = LodCompactionUniform::identity(
        4_096,
        4_096,
        LodQualityEndpoint::Original,
        settings.frustum_culling,
    )
    .unwrap()
    .with_policy(LodCompactionPolicy::hierarchy(&settings));
    assert_eq!(identity.quality_endpoint, 2);
    assert_eq!(identity.source_count, 4_096);
    assert_eq!(identity.candidate_count, 4_096);
    assert_eq!(
        identity.candidate_source_mode,
        LOD_CANDIDATE_SOURCE_IDENTITY
    );
}

#[test]
fn compaction_allocation_requires_a_published_candidate_at_every_endpoint() {
    assert!(!lod_compaction_request_is_eligible(false, false, false));
    assert!(
        lod_compaction_request_is_eligible(true, true, false),
        "a prebuilt package must submit its exact leaf frontier at quality one"
    );
    assert!(
        !lod_compaction_request_is_eligible(false, false, false),
        "a requested approximate quality must not allocate before orchestration publishes a complete candidate"
    );
    assert!(lod_compaction_request_is_eligible(true, true, false));
    assert!(
        !lod_compaction_request_is_eligible(true, false, false),
        "a candidate selected before a quality change must not allocate under the new target"
    );
    assert!(
        lod_compaction_request_is_eligible(true, false, true),
        "the retained complete cut must remain drawable while a matching replacement streams"
    );
    assert!(
        !lod_compaction_request_is_eligible(false, true, true),
        "an incomplete cold-start observation must not allocate compaction without a draw-required candidate"
    );
    assert!(lod_compaction_request_is_eligible(true, true, false));
}

#[test]
fn retained_package_capacity_does_not_shrink_before_replacement_activation() {
    assert_eq!(
        lod_compaction_requested_capacity(8_000_000, 2_000_000, Some(4_000_000), None),
        4_000_000,
        "a pending smaller replacement must preserve the current drawable allocation"
    );
    assert_eq!(
        lod_compaction_requested_capacity(8_000_000, 2_000_000, None, Some(3_994_792)),
        3_994_792,
        "device recovery must size a retained current candidate independently of the new cap"
    );
    assert_eq!(
        lod_compaction_requested_capacity(8_000_000, 2_000_000, None, None),
        2_000_000,
        "a cold package still honors the requested active budget"
    );

    let mut settings = GaussianLodSettings {
        quality: 0.65,
        ..default()
    };
    settings.budgets.max_active_gaussians = 2_000_000;
    let target = settings.quality_target();
    assert!(lod_frontier_matches_extracted_policy(
        target, 4_000_000, false, target, 3_994_792, false,
    ));
    assert!(
        !lod_frontier_matches_extracted_policy(target, 2_000_000, false, target, 3_994_792, false,),
        "a pending candidate selected under the old cap must not activate after a budget decrease"
    );
    assert!(
        !lod_frontier_matches_extracted_policy(
            LodQualityTarget::Original,
            4_000_000,
            false,
            target,
            3_994_792,
            false,
        ),
        "a pending candidate selected under the old quality must not activate"
    );
}

#[test]
fn oversized_flat_sources_wait_for_candidates_without_truncating_identity() {
    let (identity, identity_readiness) =
        LodCompactionUniform::initial(2_000_000, 2_000_000, LodQualityEndpoint::Original, true);
    assert_eq!(
        identity_readiness,
        LodCompactionReadiness::PendingCandidates
    );
    assert_eq!(
        identity.candidate_source_mode,
        LOD_CANDIDATE_SOURCE_IDENTITY
    );
    assert_eq!(identity.candidate_count, 2_000_000);

    let (bounded, bounded_readiness) =
        LodCompactionUniform::initial(8_000_000, 2_000_000, LodQualityEndpoint::Continuous, true);
    assert_eq!(
        bounded_readiness,
        LodCompactionReadiness::AwaitingCandidates
    );
    assert_eq!(bounded.source_count, 8_000_000);
    assert_eq!(bounded.output_capacity, 2_000_000);
    assert_eq!(bounded.candidate_source_mode, LOD_CANDIDATE_SOURCE_RANGES);
    assert_eq!(bounded.candidate_count, 0);
    assert_eq!(
        bounded_readiness.after_commit(),
        LodCompactionReadiness::PendingCandidates
    );
    assert_eq!(
        bounded_readiness.after_commit().after_prepare(),
        LodCompactionReadiness::Ready
    );
    assert_eq!(
        LodCompactionReadiness::PendingCandidates.synchronize_pipeline_readiness(false),
        LodCompactionReadiness::PendingCandidates
    );
    assert_eq!(
        LodCompactionReadiness::PendingCandidates.synchronize_pipeline_readiness(true),
        LodCompactionReadiness::Ready
    );
    assert_eq!(
        LodCompactionReadiness::Ready.synchronize_pipeline_readiness(false),
        LodCompactionReadiness::PendingCandidates
    );
    assert_eq!(
        LodCompactionReadiness::AwaitingCandidates.synchronize_pipeline_readiness(true),
        LodCompactionReadiness::AwaitingCandidates
    );
    assert_eq!(
        LodCompactionReadiness::Ready.after_commit(),
        LodCompactionReadiness::Ready
    );
    assert_eq!(
        bounded.with_physical_ranges(2_000_001, 1),
        Err(LodCandidateConfigError::CandidateCountExceedsCapacity {
            candidate_count: 2_000_001,
            output_capacity: 2_000_000,
        })
    );

    assert_eq!(representable_source_count(2_000_000), Some(2_000_000));
    assert_eq!(representable_source_count(8_000_000), Some(8_000_000));
    assert_eq!(
        representable_source_count(LOD_ENTRY_MAX_SOURCE_COUNT as usize),
        Some(LOD_ENTRY_MAX_SOURCE_COUNT)
    );
    assert_eq!(
        representable_source_count(LOD_ENTRY_MAX_SOURCE_COUNT as usize + 1),
        None
    );
    assert_eq!(
        LodCompactionUniform::identity(
            LOD_ENTRY_MAX_SOURCE_COUNT + 1,
            LOD_ENTRY_MAX_SOURCE_COUNT + 1,
            LodQualityEndpoint::Continuous,
            true,
        ),
        Err(LodCandidateConfigError::SourceIndexExceedsEntryEncoding {
            source_count: LOD_ENTRY_MAX_SOURCE_COUNT + 1,
            max_source_count: LOD_ENTRY_MAX_SOURCE_COUNT,
        })
    );

    #[cfg(target_pointer_width = "64")]
    assert_eq!(representable_source_count(u32::MAX as usize + 1), None);
}

#[test]
fn radix_output_parity_matches_all_supported_depths() {
    assert_eq!(
        ShaderDefines::for_radix_depth_bits(RadixSortDepthBits::Bits16).radix_key_shift,
        16
    );
    assert_eq!(
        ShaderDefines::for_radix_depth_bits(RadixSortDepthBits::Bits24).radix_key_shift,
        8
    );
    assert_eq!(
        ShaderDefines::for_radix_depth_bits(RadixSortDepthBits::Bits32).radix_key_shift,
        0
    );
    let variant_keys = HashSet::from([
        (GaussianMode::Gaussian3d, RadixSortDepthBits::Bits16),
        (GaussianMode::Gaussian3d, RadixSortDepthBits::Bits24),
        (GaussianMode::Gaussian3d, RadixSortDepthBits::Bits32),
    ]);
    assert_eq!(variant_keys.len(), 3);

    assert_eq!(
        radix_sorted_output_buffer_index(RadixSortDepthBits::Bits16),
        0
    );
    assert_eq!(
        radix_sorted_output_buffer_index(RadixSortDepthBits::Bits24),
        1
    );
    assert_eq!(
        radix_sorted_output_buffer_index(RadixSortDepthBits::Bits32),
        0
    );
}

#[test]
fn radix_treats_exact_draw_count_as_read_only() {
    let shader = include_str!("../../sort/radix.wgsl");
    assert!(shader.contains("return draw_indirect.instance_count;"));
    assert!(!shader.contains("atomicLoad(&draw_indirect.instance_count)"));
    assert_eq!(shader.matches("draw_indirect.instance_count").count(), 1);

    let host = include_str!("../../sort/radix.rs");
    assert!(!host.contains("clear_buffer(cloud.draw_indirect_buffer()"));
    assert_eq!(host.matches("dispatch_workgroups_indirect").count(), 1);
    assert_eq!(host.matches("radix_indirect_stage!(").count(), 3);
}

#[test]
fn compaction_is_ordered_after_live_gpu_writers() {
    let host = include_str!("../lod.rs");
    assert!(host.contains("TypeId::of::<R::PlanarType>() == TypeId::of::<PlanarGaussian3d>()"));
    assert!(host.contains(".after(InterpolateLabel)"));
    assert!(host.contains("LodCompactionLabel.after(MorphLabel)"));
}

#[test]
fn compaction_is_modular_and_active_bridge_cuts_require_uploaded_generations() {
    let host = include_str!("../lod.rs");
    assert!(host.contains(".init_gpu_resource::<LodAtlasGpuGenerations>()"));
    assert_eq!(
        lod_bridge_atlas_decision(LOD_RENDER_ACTIVE, false),
        LodBridgeAtlasDecision::RejectActive,
        "an ACTIVE adjacency cannot remain drawable from stale atlas generations"
    );
    assert_eq!(
        lod_bridge_atlas_decision(LOD_RENDER_ACTIVE, true),
        LodBridgeAtlasDecision::SynchronizePending,
        "an ACTIVE adjacency may synchronize only after every required generation uploads"
    );
    assert!(host.contains("frontier_is_current(atlas, candidate.required_atlas_ranges())"));
    assert!(host.contains(".store(LOD_RENDER_WAITING, Ordering::Release)"));
}

#[test]
fn view_entity_cloud_keys_isolate_render_instances_sharing_an_asset() {
    fn retained_view(entity: u32, subview: u32) -> RetainedViewEntity {
        RetainedViewEntity::new(
            MainEntity::from(Entity::from_raw_u32(entity).expect("valid test entity")),
            None,
            subview,
        )
    }

    let view_a = retained_view(1, 0);
    let entity_a = Entity::from_raw_u32(10).expect("valid test entity");
    let entity_b = Entity::from_raw_u32(11).expect("valid test entity");
    let cloud_a = AssetId::<PlanarGaussian3d>::default();
    let keys = HashSet::from([
        lod_view_cloud_key(view_a, entity_a, cloud_a),
        lod_view_cloud_key(view_a, entity_b, cloud_a),
    ]);

    assert_eq!(keys.len(), 2);
}

#[test]
fn view_entity_cloud_keys_isolate_cameras_subviews_and_assets() {
    fn retained_view(entity: u32, subview: u32) -> RetainedViewEntity {
        RetainedViewEntity::new(
            MainEntity::from(Entity::from_raw_u32(entity).expect("valid test entity")),
            None,
            subview,
        )
    }

    let view_a = retained_view(1, 0);
    let view_b = retained_view(1, 1);
    let entity = Entity::from_raw_u32(10).expect("valid test entity");
    let cloud_a = AssetId::<PlanarGaussian3d>::default();
    let cloud_b = AssetId::<PlanarGaussian3d>::invalid();
    let keys = HashSet::from([
        lod_view_cloud_key(view_a, entity, cloud_a),
        lod_view_cloud_key(view_b, entity, cloud_a),
        lod_view_cloud_key(view_a, entity, cloud_b),
    ]);

    assert_eq!(keys.len(), 3);
}

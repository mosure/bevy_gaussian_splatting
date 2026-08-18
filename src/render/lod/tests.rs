use super::*;
use crate::{
    gaussian::formats::{
        planar_3d::PlanarGaussian3d,
        planar_3d_chunked::{LodNodeId, LodPageId},
    },
    stream::cache::AtlasSlot,
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
                _padding: 0,
            },
            LodGpuPhysicalRangeDescriptor {
                candidate_start: 3,
                physical_start: 40,
                count: 2,
                _padding: 0,
            },
        ]
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
    let maximum_prefix_bytes = candidate_binding_bytes(2_000_000, 2_000_000).unwrap();
    assert_eq!(stable_bytes, tail + 16);
    assert_eq!(maximum_prefix_bytes, tail + 8_000_000);
    assert!(stable_bytes < maximum_prefix_bytes);
    assert_eq!(candidate_source_capacity_after_upload(4, 4, 2_000_000), 4);
    assert_eq!(
        candidate_source_capacity_after_upload(4, 5, 2_000_000),
        2_000_000,
        "the first non-trivial prefix grows directly to the admitted maximum"
    );
    assert_eq!(
        candidate_source_capacity_after_upload(2_000_000, 4, 2_000_000),
        2_000_000,
        "range frontiers retain peak prefix capacity until state destruction"
    );
    assert_eq!(
        candidate_source_capacity_after_upload(2_000_000, 1_000_000, 2_000_000),
        2_000_000,
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
    assert_eq!(plan.candidate_indices_bytes, 8_000_000);
    assert_eq!(plan.candidate_evaluations_bytes, 16_000_000);
    assert_eq!(plan.scan_group_count, 7_813);
    assert_eq!(plan.scan_block_count, 31);
    assert_eq!(plan.scan_records_bytes, 62_752);
    assert_eq!(plan.candidate_and_scan_records_bytes, 24_062_752);
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
    assert_eq!(storage_limited.effective_capacity, 680);
    assert_eq!(storage_limited.candidate_indices_bytes, 2_720);
    assert_eq!(storage_limited.candidate_evaluations_bytes, 5_440);
    assert_eq!(storage_limited.scan_records_bytes, 32);
    assert_eq!(storage_limited.candidate_and_scan_records_bytes, 8_192);
    assert_eq!(storage_limited.active_entries_bytes, 5_440);
    assert_eq!(storage_limited.radix_scratch_bytes, 5_440);
    assert_eq!(storage_limited.sorting_status_counter_bytes, 1_024);

    let buffer_limited =
        plan_lod_compaction_allocation(2_000, 6_000, 1024 * 1024, 64 * 1024, u32::MAX).unwrap();
    assert_eq!(buffer_limited.effective_capacity, 498);
    assert_eq!(buffer_limited.scan_records_bytes, 24);
    assert_eq!(buffer_limited.candidate_and_scan_records_bytes, 6_000);
    assert_eq!(buffer_limited.active_entries_bytes, 3_984);
    assert_eq!(buffer_limited.radix_scratch_bytes, 3_984);
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
    assert_eq!(max_candidate_capacity_for_combined_storage(27), 0);
    assert_eq!(max_candidate_capacity_for_combined_storage(28), 1);
    assert_eq!(max_candidate_capacity_for_combined_storage(3_088), 256);
    assert_eq!(max_candidate_capacity_for_combined_storage(3_107), 256);
    assert_eq!(max_candidate_capacity_for_combined_storage(3_108), 257);

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
        u64::from(plan.effective_capacity) * 12 + plan.scan_records_bytes
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
    assert_eq!(plan.effective_capacity, 680);
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
            output_capacity: 680,
        })
    );
    let bounded_frontier = flat.with_physical_ranges(680, 1).unwrap();
    assert_eq!(bounded_frontier.candidate_count, 680);
    assert_eq!(
        flat.with_physical_ranges(681, 1),
        Err(LodCandidateConfigError::CandidateCountExceedsCapacity {
            candidate_count: 681,
            output_capacity: 680,
        })
    );
}

#[test]
fn visibility_policy_is_explicit_in_the_gpu_uniform_and_shader() {
    let mut settings = GaussianLodSettings {
        quality: 0.5,
        frustum_culling: true,
        frustum_margin: 2.5,
        ..default()
    };
    let enabled = LodCompactionUniform::identity(1, 1, LodQualityEndpoint::Continuous, true)
        .expect("one-entry identity")
        .with_policy(&settings);
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
    let sanitized = enabled.with_policy(&settings);
    assert_eq!(sanitized.frustum_margin, 0.0);

    let shader = include_str!("../lod_compaction.wgsl");
    assert!(shader.contains("lod_config.frustum_culling != 0u"));
    assert!(shader.contains("support_sphere_in_frustum(transformed_position, support_radius)"));
    assert!(shader.contains("view.frustum[plane_index]"));
    assert!(shader.contains("lod_config.frustum_margin"));
    assert!(shader.contains("let visibility = get_visibility(source_index)"));
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
    .with_policy(&settings);
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
    let mut settings = GaussianLodSettings::default();
    assert_eq!(settings.quality_endpoint(), LodQualityEndpoint::Original);
    let original = settings.quality_target();
    assert!(!lod_compaction_request_is_eligible(original, None, false));
    assert!(
        lod_compaction_request_is_eligible(original, Some(original), true),
        "a prebuilt package must submit its exact leaf frontier at quality one"
    );

    settings.quality = 0.75;
    assert_eq!(settings.quality_endpoint(), LodQualityEndpoint::Continuous);
    let balanced = settings.quality_target();
    assert!(
        !lod_compaction_request_is_eligible(balanced, None, false),
        "a requested approximate quality must not allocate before orchestration publishes a complete candidate"
    );
    assert!(lod_compaction_request_is_eligible(
        balanced,
        Some(balanced),
        true
    ));
    assert!(
        !lod_compaction_request_is_eligible(original, Some(balanced), true),
        "a candidate selected before a quality change must not allocate under the new target"
    );
    assert!(
        !lod_compaction_request_is_eligible(balanced, Some(balanced), false),
        "a retained ephemeral flat-source candidate keeps status without allocating compaction"
    );

    settings.quality = 0.0;
    assert_eq!(settings.quality_endpoint(), LodQualityEndpoint::Coarsest);
    let coarsest = settings.quality_target();
    assert!(lod_compaction_request_is_eligible(
        coarsest,
        Some(coarsest),
        true
    ));
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
fn planar_3d_compaction_is_ordered_after_morph_interpolation() {
    let host = include_str!("../lod.rs");
    assert!(host.contains("TypeId::of::<R::PlanarType>() == TypeId::of::<PlanarGaussian3d>()"));
    assert!(host.contains(".after(InterpolateLabel)"));
}

#[test]
fn compaction_is_modular_and_active_bridge_cuts_require_uploaded_generations() {
    let host = include_str!("../lod.rs");
    assert!(host.contains(".init_gpu_resource::<LodAtlasGpuGenerations>()"));
    assert!(host.contains("requested_phase == LOD_RENDER_ACTIVE"));
    assert!(host.contains(".frontier_is_current(handle.handle().id().untyped()"));
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

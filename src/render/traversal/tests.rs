use super::*;
use crate::{
    Gaussian3d, GaussianLodBuildSettings,
    gaussian::{
        f32::Rotation,
        formats::{
            planar_3d_chunked::LodNodeId,
            planar_3d_lod::{GaussianLodManifest, build_planar_3d_lod},
        },
    },
    stream::hierarchy::{
        LodHierarchy, LodView, ManifestLodHierarchy, select_frontier_with_visibility,
    },
};

fn test_frustum(clip: Mat4) -> [[f32; 4]; 6] {
    let rows = clip.transpose();
    [
        rows.w_axis + rows.x_axis,
        rows.w_axis - rows.x_axis,
        rows.w_axis + rows.y_axis,
        rows.w_axis - rows.y_axis,
        rows.w_axis - rows.z_axis,
        rows.z_axis,
    ]
    .map(|plane| {
        let length = plane.truncate().length();
        if length > 0.0 {
            (plane / length).to_array()
        } else {
            [0.0; 4]
        }
    })
}

fn fixture() -> (
    GaussianLodManifest,
    Arc<GpuLodHierarchyTree>,
    Vec<Option<GpuLodPagePlacement>>,
) {
    let source = PlanarGaussian3d::from(
        (0..32)
            .map(|i| Gaussian3d {
                position_visibility: [
                    ((i % 4) as f32 - 1.5) * 0.15,
                    (((i / 4) % 4) as f32 - 1.5) * 0.15,
                    ((i / 16) as f32 - 0.5) * 0.2,
                    1.0,
                ]
                .into(),
                rotation: Rotation {
                    rotation: [1.0, 0.0, 0.0, 0.0],
                },
                scale_opacity: [0.025, 0.02, 0.015, 0.5].into(),
                spherical_harmonic: default(),
            })
            .collect::<Vec<_>>(),
    );
    let manifest = build_planar_3d_lod(
        &source,
        GaussianLodBuildSettings {
            // One record per physical page makes every binary sibling pair
            // span distinct pages, so removing one genuinely leaves a loaded
            // sibling that must remain demanded during the replacement.
            leaf_capacity: 1,
            ..default()
        },
    )
    .unwrap()
    .manifest;
    let tree = Arc::new(GpuLodHierarchyTree::from_manifest(&manifest).unwrap());
    assert!(tree.max_depth >= 2);
    let mut start = 0;
    let placements = manifest
        .pages
        .iter()
        .map(|page| {
            let placement = GpuLodPagePlacement {
                start,
                count: page.gaussian_count,
            };
            start += page.gaussian_count;
            Some(placement)
        })
        .collect();
    (manifest, tree, placements)
}

// One small forest exposes more concurrent reservations than the retired
// 64-retry cutoff. Every split fits; the exact result is all 1024 child records.
fn contention_fixture() -> (
    Arc<GpuLodHierarchyTree>,
    Vec<Option<GpuLodPagePlacement>>,
    Vec<u32>,
) {
    let roots = 512u32;
    let parent = |index| Node {
        center_radius: [0.0, 0.0, 0.0, 0.1],
        half_extents: [0.05, 0.05, 0.05, 0.0],
        error_quality: [1.0, 0.5, 0.0, 0.0],
        topology: [roots * 2 + index * 2, 2, index, 0],
        counts: [1, 2, 0, 0],
    };
    let nodes = (0..roots)
        .map(parent)
        .chain((0..roots).map(parent))
        .chain((roots..roots * 3).map(|page| Node {
            center_radius: [0.0, 0.0, 0.0, 0.1],
            half_extents: [0.05, 0.05, 0.05, 0.0],
            error_quality: [0.0, 1.0, 1.0, 0.0],
            topology: [0, 0, page, 0],
            counts: [1, 0, 1, 0],
        }))
        .collect();
    let tree = Arc::new(GpuLodHierarchyTree {
        nodes,
        source_order: Vec::new(),
        candidates: Vec::new(),
        root_count: roots,
        support_sigma: 3.0,
        root_records: roots,
        max_depth: 1,
        page_lengths: vec![1; (roots * 3) as usize],
        page_ids: (0..roots * 3)
            .map(|page| LodPageId(u64::from(page)))
            .collect(),
    });
    let placements = (0..roots * 3)
        .map(|start| Some(GpuLodPagePlacement { start, count: 1 }))
        .collect();
    (tree, placements, (roots..roots * 3).collect())
}

// The synthetic forest has contiguous, disjoint source intervals under each
// root. Build its source-domain order without changing its BFS child indices.
fn order_contention_fixture(tree: &mut GpuLodHierarchyTree) {
    let mut order = Vec::with_capacity(tree.nodes.len());
    for root in 0..tree.root_count {
        order.push(root);
        order.push(tree.root_count + root);
        let node = tree.nodes[root as usize];
        let mut stack = (node.topology[0]..node.topology[0] + node.topology[1])
            .rev()
            .collect::<Vec<_>>();
        while let Some(index) = stack.pop() {
            order.push(index);
            let child = tree.nodes[index as usize];
            stack.extend((child.topology[0]..child.topology[0] + child.topology[1]).rev());
        }
    }
    assert_eq!(order.len(), tree.nodes.len());
    for (rank, &index) in order.iter().enumerate() {
        tree.nodes[index as usize].counts[3] = rank as u32;
    }
    tree.source_order = order;
    let mut candidates = Vec::new();
    for root in 0..tree.root_count {
        let mut stack = vec![(root, u32::MAX, 0)];
        while let Some((index, parent, depth)) = stack.pop() {
            tree.nodes[index as usize].error_quality[3] = f32::from_bits(parent);
            let node = tree.nodes[index as usize];
            if node.topology[1] == 0 {
                continue;
            }
            candidates.push([
                index,
                parent,
                depth,
                if index < tree.root_count {
                    tree.root_count + index
                } else {
                    u32::MAX
                },
            ]);
            stack.extend(
                (node.topology[0]..node.topology[0] + node.topology[1])
                    .rev()
                    .map(|child| (child, index, depth + 1)),
            );
        }
    }
    tree.candidates = candidates;
}

#[test]
fn compiled_topology_preserves_cpu_oracle_metrics_and_child_cohorts() {
    let (manifest, tree, _) = fixture();
    let oracle = ManifestLodHierarchy::new(&manifest).unwrap();
    assert_eq!(size_of::<Node>(), 80);
    assert_eq!(size_of::<Config>(), 352);
    let projection = Mat4::perspective_infinite_reverse_rh(0.9, 1.4, 0.1);
    let supported = omission_parameters(
        3.0,
        1.0,
        Mat4::IDENTITY,
        1.0,
        projection,
        Vec2::new(128.0, 96.0),
        true,
        true,
    );
    assert!(supported[0] >= 1.0 && supported[1] > 0.0);
    assert_eq!(&supported[2..], &[1.0, 1.0]);
    for (support, scale, transform, frustum_enabled) in [
        (2.0, 1.0, Mat4::IDENTITY, true),
        (3.0, 2.0, Mat4::IDENTITY, true),
        (3.0, 1.0, Mat4::from_scale(Vec3::new(0.0, 1.0, 1.0)), true),
        (3.0, 1.0, Mat4::IDENTITY, false),
    ] {
        assert_eq!(
            omission_parameters(
                support,
                scale,
                transform,
                1.0,
                projection,
                Vec2::new(128.0, 96.0),
                frustum_enabled,
                true
            ),
            [0.0, 0.0, 0.0, 1.0],
            "declining physical omission must retain active annotation ownership"
        );
    }
    assert_eq!(tree.node_count(), manifest.nodes.len());
    assert_eq!(tree.source_order.len(), tree.nodes.len());
    for (rank, &compiled) in tree.source_order.iter().enumerate() {
        assert_eq!(tree.nodes[compiled as usize].counts[3], rank as u32);
    }
    for (index, authored) in manifest.nodes.iter().enumerate() {
        let node = tree.nodes[index + tree.root_count as usize];
        let metrics = oracle.metrics(authored.id).unwrap();
        assert_eq!(
            node.center_radius,
            [
                metrics.center.x,
                metrics.center.y,
                metrics.center.z,
                metrics.radius
            ]
        );
        for axis in 0..3 {
            assert_eq!(
                node.half_extents[axis],
                (authored.bounds.max[axis] - metrics.center[axis])
                    .max(metrics.center[axis] - authored.bounds.min[axis])
            );
        }
        assert_eq!(
            node.error_quality[..3],
            [
                metrics.geometric_error,
                metrics.quality_threshold(),
                metrics.high_fidelity_certificate
            ]
        );
        assert_eq!(node.counts[0], metrics.representative_count);
        assert_eq!(
            node.counts[1],
            oracle
                .children(authored.id)
                .iter()
                .map(|child| oracle.metrics(*child).unwrap().representative_count)
                .sum::<u32>()
        );
        assert_eq!(
            tree.page_ids[node.topology[2] as usize],
            authored.representation.page
        );
        assert_eq!(node.topology[3], authored.representation.offset);
        assert_eq!(node.topology[0], tree.root_count + authored.children.start);
    }
}

#[test]
fn snapshot_rejects_partial_overlapping_and_out_of_range_page_placements() {
    let (_, tree, placements) = fixture();
    let create = |placements| {
        GpuLodHierarchy::new(
            tree.clone(),
            1,
            AssetId::default(),
            placements,
            Arc::new(()),
        )
    };
    assert!(create(placements.clone()).is_ok());
    let mut changed = placements.clone();
    changed[0].as_mut().unwrap().count += 1;
    assert!(create(changed).is_err());
    let mut changed = placements.clone();
    changed[1].as_mut().unwrap().start = changed[0].unwrap().start;
    assert!(create(changed).is_err());
    let mut changed = placements;
    changed[0].as_mut().unwrap().start = 0x1000_0000;
    assert!(create(changed).is_err());

    let slot_count = tree.page_ids.len() as u32 + 1;
    let slot_size = tree.page_lengths.iter().copied().max().unwrap() + 1;
    let residents = (0..tree.page_ids.len())
        .step_by(2)
        .map(|page| {
            (
                page,
                crate::stream::cache::AtlasSlot {
                    index: slot_count - 1 - page as u32,
                    generation: 7,
                },
            )
        })
        .collect::<Vec<_>>();
    let fixed = |residents: &[(usize, crate::stream::cache::AtlasSlot)]| {
        GpuLodHierarchy::from_fixed_slots(
            tree.clone(),
            1,
            AssetId::default(),
            slot_count,
            slot_size,
            residents.iter().copied(),
            Arc::new(()),
        )
    };
    let mut expected = vec![None; tree.page_ids.len()];
    for &(page, slot) in &residents {
        expected[page] = Some(GpuLodPagePlacement {
            start: slot.index * slot_size,
            count: tree.page_lengths[page],
        });
    }
    let fixed_snapshot = fixed(&residents).unwrap();
    assert!(fixed_snapshot.page_is_resident(0));
    assert!(!fixed_snapshot.page_is_resident(1));
    assert!(!fixed_snapshot.page_is_resident(tree.page_ids.len()));
    assert_eq!(
        fixed_snapshot.0.placements,
        create(expected).unwrap().0.placements
    );
    assert_eq!(
        fixed_snapshot.0.required_atlas_slots,
        residents.iter().map(|&(_, slot)| slot).collect::<Vec<_>>()
    );
    let mut changed = residents.clone();
    changed[1].1.index = changed[0].1.index;
    changed[1].1.generation += 1;
    assert!(
        fixed(&changed).is_err(),
        "generation differences cannot hide slot overlap"
    );
    changed = residents.clone();
    changed[1].0 = changed[0].0;
    assert!(fixed(&changed).is_err(), "one page cannot occupy two slots");
    changed = residents.clone();
    changed[0].1.index = slot_count;
    assert!(fixed(&changed).is_err());
    changed = residents;
    changed[0].0 = tree.page_ids.len();
    assert!(fixed(&changed).is_err());
    assert!(
        GpuLodTraversalSettings {
            max_selected_gaussians: 1,
            ..default()
        }
        .validate()
        .is_ok()
    );
}

#[test]
fn allocation_preflight_checks_roots_device_limits_and_shared_view_footprint() {
    // A forest may consist entirely of roots packed into one page. The old
    // 160*N estimate left no scratch allowance after 80-byte root duplication.
    let forest_nodes = 4;
    let topology = GpuLodHierarchyTree::compilation_bytes(forest_nodes, forest_nodes, 1).unwrap();
    let compiled_nodes =
        2 * forest_nodes * (size_of::<Node>() + size_of::<u32>() + size_of::<[u32; 4]>());
    let root_scratch = forest_nodes * (size_of::<(LodNodeId, u32)>() + size_of::<u32>());
    let page_vectors = size_of::<(LodPageId, u32)>() + size_of::<LodPageId>() + size_of::<u32>();
    assert_eq!(
        topology,
        (compiled_nodes + root_scratch + page_vectors) as u64
    );
    assert!(topology > (160 * forest_nodes) as u64);
    assert!(GpuLodHierarchyTree::compilation_bytes(usize::MAX, usize::MAX, 1).is_err());

    let (_, tree, placements) = fixture();
    let snapshot = GpuLodHierarchy::new(tree, 1, AssetId::default(), placements, Arc::new(()))
        .unwrap()
        .0;
    let limits = wgpu::Limits::default();
    let mut settings = GpuLodTraversalSettings::default();
    let plan = plan_allocation(&limits, &snapshot, &settings, None).unwrap();
    let shared = tree_allocation_bytes(&snapshot) + page_allocation_bytes(&snapshot);
    settings.max_gpu_bytes = plan.bytes + shared;
    assert!(plan_allocation(&limits, &snapshot, &settings, None).is_ok());
    settings.max_gpu_bytes -= 1;
    assert!(plan_allocation(&limits, &snapshot, &settings, None).is_err());
    settings.max_gpu_bytes = u64::MAX;
    settings.max_selected_gaussians = snapshot.tree.root_gaussian_count() - 1;
    assert!(plan_allocation(&limits, &snapshot, &settings, None).is_err());
    settings = GpuLodTraversalSettings::default();
    let small_device = wgpu::Limits {
        max_storage_buffer_binding_size: plan.entry_bytes - 1,
        ..limits
    };
    assert!(plan_allocation(&small_device, &snapshot, &settings, None).is_err());
    let small_workgroup = wgpu::Limits {
        max_compute_workgroup_storage_size: SCAN_WORKGROUP_BYTES - 1,
        ..wgpu::Limits::default()
    };
    assert!(plan_allocation(&small_workgroup, &snapshot, &settings, None).is_err());

    let two_cloud_bytes = 2 * plan.bytes + shared;
    let mut footprint = ViewFootprint::default();
    footprint.admit(&snapshot, &plan, two_cloud_bytes).unwrap();
    footprint.admit(&snapshot, &plan, two_cloud_bytes).unwrap();
    assert_eq!(footprint.bytes, two_cloud_bytes);
    let mut insufficient = ViewFootprint::default();
    insufficient
        .admit(&snapshot, &plan, two_cloud_bytes - 1)
        .unwrap();
    assert!(
        insufficient
            .admit(&snapshot, &plan, two_cloud_bytes - 1)
            .is_err()
    );
}

#[test]
fn feedback_dedup_preserves_request_priority_and_rejects_invalid_references() {
    let pages = [10, 20, 30, 40].map(LodPageId);
    assert_eq!(
        deduplicate_page_requests(&[2, 2, 0, 2, 3, 0, 1], &pages),
        Some([30, 10, 40, 20].map(LodPageId).to_vec())
    );
    assert!(deduplicate_page_requests(&[2, 9], &pages).is_none());
}

/// The actual production WGSL, with a 32-source hierarchy and bounded
/// scenarios. Explicitly opt in; this checks numerical/cut correctness, not speed.
#[test]
#[ignore = "requires an explicitly requested wgpu adapter"]
fn gpu_traversal_matches_cpu_cuts_and_preserves_bounded_parent_fallback() {
    if std::env::var("RUN_GPU_RENDER_TESTS").as_deref() != Ok("1") {
        return;
    }
    use wgpu::util::DeviceExt;
    let instance = wgpu::Instance::default();
    let adapter = pollster::block_on(wgpu::util::initialize_adapter_from_env_or_default(
        &instance, None,
    ))
    .unwrap();
    let (device, queue) = pollster::block_on(adapter.request_device(&wgpu::DeviceDescriptor {
        label: Some("lod_traversal_oracle"),
        ..default()
    }))
    .unwrap();
    let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
        label: Some("production_lod_traversal"),
        source: wgpu::ShaderSource::Wgsl(
            format!(
                "{}\n{}",
                include_str!("../spatial_morph/metrics.wgsl")
                    .lines()
                    .filter(|line| !line.starts_with("#define_import_path"))
                    .collect::<Vec<_>>()
                    .join("\n"),
                include_str!("traversal.wgsl")
                    .lines()
                    .filter(|line| !line.starts_with("#import"))
                    .collect::<Vec<_>>()
                    .join("\n"),
            )
            .into(),
        ),
    });
    let bindings = (0..7)
        .map(|binding| wgpu::BindGroupLayoutEntry {
            binding,
            visibility: wgpu::ShaderStages::COMPUTE,
            ty: wgpu::BindingType::Buffer {
                ty: if binding == 0 {
                    wgpu::BufferBindingType::Uniform
                } else {
                    wgpu::BufferBindingType::Storage {
                        read_only: binding < 3 || binding == 6,
                    }
                },
                has_dynamic_offset: false,
                min_binding_size: None,
            },
            count: None,
        })
        .collect::<Vec<_>>();
    let layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
        label: None,
        entries: &bindings,
    });
    let pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
        label: None,
        bind_group_layouts: &[Some(&layout)],
        immediate_size: 0,
    });
    let stages = STAGES.map(|entry| {
        device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some(entry),
            layout: Some(&pipeline_layout),
            module: &shader,
            entry_point: Some(entry),
            compilation_options: default(),
            cache: None,
        })
    });
    let (manifest, tree, placements) = fixture();
    let oracle = ManifestLodHierarchy::new(&manifest).unwrap();
    let camera = Vec3::new(0.6, 0.3, 3.0);
    let clip = Mat4::perspective_infinite_reverse_rh(0.9, 1.4, 0.1)
        * Mat4::look_at_rh(camera, Vec3::ZERO, Vec3::Y);
    let world = Mat4::from_scale_rotation_translation(
        Vec3::new(1.3, 0.7, 0.9),
        Quat::from_rotation_y(0.4),
        Vec3::ZERO,
    );
    let root_pages = tree.nodes[..tree.root_count as usize]
        .iter()
        .map(|node| node.topology[2] as usize)
        .collect::<HashSet<_>>();
    let missing = (0..placements.len())
        .find(|page| !root_pages.contains(page))
        .unwrap();
    for (case, quality, offscreen, absent, bound) in [
        ("coarse", 0.0, false, false, 0),
        ("balanced", 0.65, false, false, 0),
        ("exact", 1.0, false, false, 0),
        ("offscreen", 1.0, true, false, 0),
        ("thin_offscreen", 1.0, false, false, 0),
        ("thin_transformed", 1.0, false, false, 0),
        ("thin_boundary", 1.0, false, false, 0),
        ("thin_margin", 1.0, false, false, 0),
        ("missing", 1.0, false, true, 0),
        ("missing_request_limit", 1.0, false, true, 4),
        ("missing_records", 1.0, false, true, 1),
        ("missing_frontier", 1.0, false, true, 3),
        ("records", 1.0, false, false, 1),
        ("visits", 1.0, false, false, 2),
        ("frontier", 1.0, false, false, 3),
        ("contention", 1.0, false, false, 0),
        ("canonical_motion", 1.0, false, false, 0),
        ("camera_motion", 1.0, false, false, 0),
        ("priority_near_resident", 1.0, false, false, 0),
        ("priority_near_spine", 1.0, false, false, 0),
        ("priority_error_resident", 1.0, false, false, 0),
        ("priority_footprint_resident", 1.0, false, false, 0),
        ("priority_near_demand", 1.0, false, false, 0),
        ("pending_demand", 1.0, false, false, 0),
        ("desired_residency_early", 1.0, false, false, 0),
        ("desired_residency_late", 1.0, false, false, 0),
        ("cutoff_budget", 0.95, false, false, 0),
        ("cutoff_missing", 0.95, false, false, 0),
        ("cutoff_tie", 0.95, false, false, 0),
        ("cutoff_exact", 0.95, false, false, 0),
        ("omission_motion", 1.0, false, false, 0),
        ("omission_active_band", 1.0, false, false, 0),
    ] {
        let omission_case = case.starts_with("omission_");
        let active_omission = case == "omission_active_band";
        let cutoff_case = case.starts_with("cutoff_");
        let priority_case = case.starts_with("priority_");
        let priority_demand = case == "priority_near_demand";
        let near_spine = case == "priority_near_spine";
        let canonical_motion = case == "canonical_motion";
        let clip = if cutoff_case || omission_case {
            Mat4::IDENTITY
        } else if priority_case || canonical_motion {
            Mat4::perspective_infinite_reverse_rh(0.9, 1.4, 0.1)
        } else if case.starts_with("thin_") {
            Mat4::IDENTITY
        } else {
            clip
        };
        let boundary_x = 1.0 - manifest.nodes[0].bounds.min[0] * 0.01;
        let camera_motion = case == "camera_motion";
        let world =
            if camera_motion || priority_case || canonical_motion || cutoff_case || omission_case {
                Mat4::IDENTITY
            } else if case == "thin_transformed" {
                // Reflection and shear must transform the plane normal, without
                // an inverse or a uniform-scale assumption.
                Mat4::from_cols(
                    Vec4::new(0.01, 0.03, 0.0, 0.0),
                    Vec4::new(0.002, 30.0, 0.004, 0.0),
                    Vec4::new(0.01, 0.0, -0.02, 0.0),
                    Vec4::new(1.2, 0.0, 0.5, 1.0),
                )
            } else if case.starts_with("thin_") {
                let x = match case {
                    "thin_boundary" => boundary_x,
                    "thin_margin" => boundary_x + 0.02,
                    _ => 1.2,
                };
                Mat4::from_scale_rotation_translation(
                    Vec3::new(0.01, 30.0, 0.01),
                    Quat::IDENTITY,
                    Vec3::new(x, 0.0, 0.5),
                )
            } else if offscreen {
                Mat4::from_translation(Vec3::X * 100.0) * world
            } else {
                world
            };
        let settings = GaussianLodSettings {
            quality,
            hysteresis: 0.0,
            frustum_margin: if case == "thin_margin" { 0.02 } else { 0.0 },
            ..default()
        };
        let view = LodView::perspective(camera, 96.0, 0.9, 0.1)
            .with_view_projection(clip, Vec2::new(128.0, 96.0))
            .with_world_from_local(world);
        let mut pages = placements.clone();
        if absent {
            pages[missing] = None;
        }
        let resident = |id: LodNodeId| {
            let node = manifest.nodes.iter().find(|node| node.id == id).unwrap();
            let page = tree
                .page_ids
                .iter()
                .position(|id| *id == node.representation.page)
                .unwrap();
            pages[page].is_some()
        };
        let frontier = if bound == 0 || bound == 4 {
            select_frontier_with_visibility(&oracle, &resident, view, &settings, |node, _| {
                view.bounds_are_visible(oracle.node(node).unwrap().bounds, settings.frustum_margin)
            })
            .unwrap()
            .nodes
        } else {
            manifest.roots.clone()
        };
        if matches!(case, "thin_offscreen" | "thin_transformed") {
            assert!(view.node_is_visible(oracle.metrics(manifest.roots[0]).unwrap(), 0.0));
            assert_eq!(
                frontier, manifest.roots,
                "AABB must avoid the sphere's false refinement"
            );
        } else if matches!(case, "thin_boundary" | "thin_margin") {
            assert_ne!(
                frontier, manifest.roots,
                "touching support must retain refinement"
            );
        }
        let mut cohort_pages = HashSet::new();
        if absent {
            for &parent in &frontier {
                let children = oracle.children(parent);
                if !children.iter().any(|child| !resident(*child)) {
                    continue;
                }
                assert!(children.len() > 1, "fixture requires a sibling cohort");
                for &child in children {
                    let page = oracle.page(child).unwrap();
                    cohort_pages
                        .insert(tree.page_ids.iter().position(|id| *id == page).unwrap() as u32);
                }
            }
        }
        let mut expected = frontier
            .iter()
            .flat_map(|id| {
                let node = manifest.nodes.iter().find(|node| node.id == *id).unwrap();
                let page = tree
                    .page_ids
                    .iter()
                    .position(|id| *id == node.representation.page)
                    .unwrap();
                let start = pages[page].unwrap().start + node.representation.offset;
                start..start + node.representation.count
            })
            .collect::<Vec<_>>();
        expected.sort_unstable();
        let pending_demand = case == "pending_demand";
        let desired_residency = case.starts_with("desired_residency_");
        let contention = case == "contention"
            || pending_demand
            || desired_residency
            || camera_motion
            || canonical_motion
            || cutoff_case
            || priority_case
            || omission_case;
        let (tree, pages, expected) = if contention {
            let (mut tree, mut pages, mut expected) = contention_fixture();
            if omission_case {
                let tree = Arc::get_mut(&mut tree).unwrap();
                for (index, node) in tree.nodes.iter_mut().enumerate() {
                    let parent = if index < 1024 {
                        index % 512
                    } else {
                        (index - 1024) / 2
                    };
                    let x = match parent {
                        0 => -1.0,
                        1 => 3.0,
                        _ => 100.0,
                    };
                    node.center_radius = [x, 0.0, 0.5, 0.36];
                    node.half_extents = [0.3, 0.05, 0.05, 0.0];
                    if index >= 1024 {
                        node.center_radius[0] += if index % 2 == 0 { -0.2 } else { 0.2 };
                        node.center_radius[3] = 0.05;
                        node.half_extents = [0.025, 0.025, 0.025, 0.0];
                    }
                }
                // Root pages remain resident even outside view. Logical
                // resolution requires complete child cohorts before omission.
                for (page, placement) in pages.iter_mut().enumerate() {
                    if page >= 512 && ![512, 513, 514, 515].contains(&page) {
                        *placement = None;
                    }
                }
                if !active_omission {
                    pages[512] = None;
                }
                expected = if active_omission {
                    vec![512, 513]
                } else {
                    vec![0]
                };
            }
            if camera_motion {
                let tree = Arc::get_mut(&mut tree).unwrap();
                for (index, node) in tree.nodes.iter_mut().enumerate() {
                    let parent = if index < 1024 {
                        index % 512
                    } else {
                        (index - 1024) / 2
                    };
                    node.center_radius[0] = if parent < 256 { -3.0 } else { 3.0 };
                    node.center_radius[2] = 0.5;
                }
            }
            if canonical_motion {
                let tree = Arc::get_mut(&mut tree).unwrap();
                for (index, node) in tree.nodes.iter_mut().enumerate() {
                    let parent = if index < 1024 {
                        index % 512
                    } else {
                        (index - 1024) / 2
                    };
                    node.center_radius[2] = if parent < 256 { -0.1 } else { -10.0 };
                }
            }
            if priority_case {
                let tree = Arc::get_mut(&mut tree).unwrap();
                for (index, node) in tree.nodes.iter_mut().enumerate() {
                    let parent = if index < 1024 {
                        index % 512
                    } else {
                        (index - 1024) / 2
                    };
                    node.center_radius = [0.0, 0.0, -10.0, 0.1];
                    node.half_extents = [0.05, 0.05, 0.05, 0.0];
                    node.error_quality[0] = 0.0;
                    if case.starts_with("priority_near_") {
                        if parent == 511 {
                            node.center_radius[2] = -0.1;
                        } else {
                            // A wide, thin far box has a sphere crossing near,
                            // but its actual transformed AABB does not. It must
                            // not tie the true near-plane proxy's priority.
                            node.center_radius[3] = 20.001;
                            node.half_extents[0] = 20.0;
                        }
                    } else if parent == 511 {
                        if case == "priority_error_resident" {
                            node.error_quality[0] = 5.0;
                        } else {
                            node.center_radius[3] = 3.5;
                            node.half_extents = [2.0, 2.0, 2.0, 0.0];
                        }
                    }
                }
                expected = (0..511).chain([1534, 1535]).collect();
                if near_spine {
                    // A near descendant competes with 511 lower-priority root
                    // splits. Spending by level would consume its last record.
                    let mut leaf = tree.nodes[2046];
                    tree.nodes[2046].topology[0] = 2048;
                    tree.nodes[2046].topology[1] = 2;
                    tree.nodes[2046].counts[1] = 2;
                    tree.nodes[2046].counts[2] = 0;
                    for page in 1536..1538 {
                        leaf.topology[2] = page;
                        tree.nodes.push(leaf);
                        tree.page_lengths.push(1);
                        tree.page_ids.push(LodPageId(u64::from(page)));
                        pages.push(Some(GpuLodPagePlacement {
                            start: page,
                            count: 1,
                        }));
                    }
                    tree.max_depth = 2;
                    expected = (0..511).chain([1535, 1536, 1537]).collect();
                }
            }
            if pending_demand || desired_residency || priority_demand {
                // Each root has one loaded child and one missing child. Only
                // half of these future splits fit the shared record capacity.
                for page in (tree.root_count as usize + 1..pages.len()).step_by(2) {
                    pages[page] = None;
                }
                expected = (0..tree.root_count).collect();
            }
            if desired_residency {
                // Exactly one ready split competes with 511 missing cohorts
                // for the sole spare record. Exercise both ends of the queue,
                // across workgroups, so dispatch order cannot decide admission.
                let ready = if case == "desired_residency_early" {
                    0
                } else {
                    tree.root_count - 1
                };
                let first_child = tree.root_count + ready * 2;
                let loaded_child = first_child + 1;
                pages[loaded_child as usize] = Some(GpuLodPagePlacement {
                    start: loaded_child,
                    count: 1,
                });
                // Residency cannot redirect the logical target: root zero owns
                // the sole spare record, even when a later cohort arrives first.
                if ready == 0 {
                    expected.retain(|index| *index != ready);
                    expected.extend([first_child, loaded_child]);
                }
            }
            if cutoff_case {
                let tree = Arc::get_mut(&mut tree).unwrap();
                // Three versus two child records gives unequal fixed edge costs.
                // The excluded equal-score tail defines tau, not the last
                // included 1.5x-severity edge. All candidates are uncertified.
                tree.nodes[0].counts[1] = 3;
                tree.nodes[512].counts[1] = 3;
                tree.nodes[1024].counts = [2, 2, 0, 0];
                tree.nodes[1024].topology[0] = 2048;
                tree.nodes[1024].topology[1] = 2;
                tree.nodes[1024].error_quality[0] = 1000.0;
                let mut leaf = tree.nodes[1025];
                for (page, start) in [(1536, 5000), (1537, 5001)] {
                    leaf.topology[2] = page;
                    tree.nodes.push(leaf);
                    tree.page_lengths.push(1);
                    tree.page_ids.push(LodPageId(u64::from(page)));
                    pages.push(Some(GpuLodPagePlacement { start, count: 1 }));
                }
                tree.max_depth = 2;
                tree.page_lengths[512] = 2;
                pages[512] = Some(GpuLodPagePlacement {
                    start: 4096,
                    count: 2,
                });
                if case != "cutoff_tie" {
                    tree.nodes[0].error_quality[0] = 1.5;
                    tree.nodes[512].error_quality[0] = 1.5;
                }
                if case == "cutoff_exact" {
                    expected = (513..1536).chain([5000, 5001]).collect();
                } else if case == "cutoff_missing" {
                    pages[513] = None;
                    expected = (0..512).collect();
                } else if case == "cutoff_tie" {
                    expected = (0..512).collect();
                } else {
                    expected = (1..512).chain([513, 4096, 4097]).collect();
                }
            }
            order_contention_fixture(Arc::get_mut(&mut tree).unwrap());
            (tree, pages, expected)
        } else {
            (tree.clone(), pages, expected)
        };
        let full_capacity = if cutoff_case {
            1025
        } else if contention {
            1024
        } else {
            64
        };
        let capacity = if omission_case {
            tree.root_records + 1
        } else if cutoff_case && case != "cutoff_exact" {
            tree.root_records + 2
        } else if camera_motion {
            tree.root_records + 128
        } else if desired_residency || priority_case {
            tree.root_records + if near_spine { 2 } else { 1 }
        } else if pending_demand {
            768
        } else if bound == 1 {
            tree.root_records
        } else {
            full_capacity
        };
        let frontier_cap = if bound == 3 {
            tree.root_count
        } else {
            full_capacity
        };
        let visit_cap = if cutoff_case {
            tree.node_count() as u32
        } else if bound == 2 {
            tree.root_count
        } else if contention {
            1536
        } else {
            256
        };
        let bitmap = (pages.len() as u32).div_ceil(32);
        let offsets = [0, frontier_cap, frontier_cap * 2, frontier_cap * 6];
        let request_cap = if bound == 4 {
            1
        } else if pending_demand {
            1024
        } else {
            64
        };
        let feedback_offsets = [
            offsets[3] + bitmap,
            offsets[3] + bitmap * 2,
            offsets[3] + bitmap * 2 + request_cap,
            bitmap,
        ];
        let linear = Mat3::from_mat4(world);
        let gram = linear.transpose() * linear;
        let scale = gram
            .x_axis
            .abs()
            .element_sum()
            .max(gram.y_axis.abs().element_sum())
            .max(gram.z_axis.abs().element_sum())
            .sqrt();
        let target = match settings.quality_target() {
            LodQualityTarget::Coarsest => [0.0, 0.0, 0.0, 1.0],
            LodQualityTarget::Original => [1.0, 0.0, 2.0, 1.0],
            LodQualityTarget::Balanced {
                detail_fraction,
                max_error_px,
            } => [detail_fraction, max_error_px, 1.0, 1.0],
        };
        let mut config = Config {
            world_from_local: world.to_cols_array(),
            clip_from_world: clip.to_cols_array(),
            view: [128.0, 96.0, scale, settings.frustum_margin],
            quality: target,
            counts: [
                tree.nodes.len() as u32,
                pages.len() as u32,
                tree.root_count,
                tree.root_records,
            ],
            limits: [capacity, frontier_cap, visit_cap, request_cap],
            offsets,
            feedback_offsets,
            spatial: [frontier_cap, 0, 0, 0],
            omission: [0.0; 4],
            frustum: [[0.0; 4]; 6],
        };
        if canonical_motion || cutoff_case {
            config.quality[3] = 0.0;
        }
        if omission_case {
            config.frustum = test_frustum(clip);
            config.omission = [1.0, 0.001, 0.0, u32::from(active_omission) as f32];
            if active_omission {
                config.spatial[1] = 1;
            }
        }
        if cutoff_case {
            config.spatial[2] = feedback_offsets[2]
                + frontier_cap
                + scan_work_words(frontier_cap, tree.nodes.len() as u32) as u32;
            config.spatial[3] = tree.candidates.len() as u32;
        }
        let config_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: None,
            contents: bytemuck::bytes_of(&config),
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
        });
        let node_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: None,
            contents: bytemuck::cast_slice(&tree.nodes),
            usage: wgpu::BufferUsages::STORAGE,
        });
        let source_order = tree
            .source_order
            .iter()
            .copied()
            .chain(tree.candidates.iter().flatten().copied())
            .collect::<Vec<_>>();
        let source_order_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("canonical source-domain ranks"),
            contents: bytemuck::cast_slice(&source_order),
            usage: wgpu::BufferUsages::STORAGE,
        });
        let pages = pages
            .into_iter()
            .map(Option::unwrap_or_default)
            .collect::<Vec<_>>();
        let page_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: None,
            contents: bytemuck::cast_slice(&pages),
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
        });
        let make = |size, usage| {
            device.create_buffer(&wgpu::BufferDescriptor {
                label: None,
                size,
                usage,
                mapped_at_creation: false,
            })
        };
        let work_size = HEADER_BYTES
            + (u64::from(feedback_offsets[2] + frontier_cap)
                + scan_work_words(frontier_cap, tree.nodes.len() as u32)
                + if cutoff_case {
                    cutoff_work_words(tree.nodes.len() as u32, tree.candidates.len() as u32)
                } else {
                    0
                })
                * 4;
        let work = make(
            work_size,
            wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
        );
        let entry_tail_bytes = if active_omission {
            32 + 32 * u64::from(frontier_cap)
        } else {
            0
        };
        let entries = make(
            u64::from(capacity) * 8 + entry_tail_bytes,
            wgpu::BufferUsages::STORAGE
                | wgpu::BufferUsages::COPY_SRC
                | wgpu::BufferUsages::COPY_DST,
        );
        // Feed the final physical compactor the same descriptor contract that
        // actual annotation supplies. The full ordered integration fixture
        // exercises annotation itself; this case isolates a crossing edge whose
        // first child has no visible endpoint support but must remain emitted.
        let annotation = active_omission.then(|| {
            let mut words = vec![0u32; entry_tail_bytes as usize / 4];
            words[0] = 1;
            words[2] = tree.root_count + 1;
            words[3] = 1;
            words[6] = 2;
            words[7] = 2;
            for range in 0..2 {
                words[8 + range * 8 + 1] = 1;
                words[8 + range * 8 + 7] = 1;
            }
            device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: Some("complete active edge at the view boundary"),
                contents: bytemuck::cast_slice(&words),
                usage: wgpu::BufferUsages::COPY_SRC,
            })
        });
        let draw = make(
            32,
            wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
        );
        let dispatch = make(
            16,
            wgpu::BufferUsages::INDIRECT | wgpu::BufferUsages::COPY_DST,
        );
        let readback = make(
            32 + work_size + u64::from(capacity) * 8,
            wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
        );
        let binding_entries = [
            &config_buffer,
            &node_buffer,
            &page_buffer,
            &work,
            &entries,
            &draw,
            &source_order_buffer,
        ]
        .into_iter()
        .enumerate()
        .map(|(binding, buffer)| wgpu::BindGroupEntry {
            binding: binding as u32,
            resource: buffer.as_entire_binding(),
        })
        .collect::<Vec<_>>();
        let bind = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: None,
            layout: &layout,
            entries: &binding_entries,
        });
        let mut held_order = None;
        let mut held_entries = None;
        let mut held_omission_ranges = None;
        let mut held_resolver_pages = HashSet::new();
        // The same buffers and immutable resident snapshot are reused. Only
        // the current camera uniform changes; no CPU frontier is uploaded.
        for frame in 0..if camera_motion || omission_case {
            4
        } else if canonical_motion {
            3
        } else {
            1
        } {
            let mut expected = expected.clone();
            let annotation_active = active_omission && (frame == 0 || frame == 3);
            if active_omission && frame != 0 {
                expected = vec![513];
                if frame == 2 {
                    // Model pressure removing every unnecessary non-root page.
                    // A bypassed sibling still needed by resolution must survive.
                    let retained = pages
                        .iter()
                        .enumerate()
                        .map(|(page, placement)| {
                            if page < tree.root_count as usize
                                || held_resolver_pages.contains(&(page as u32))
                            {
                                *placement
                            } else {
                                GpuLodPagePlacement::default()
                            }
                        })
                        .collect::<Vec<_>>();
                    queue.write_buffer(&page_buffer, 0, bytemuck::cast_slice(&retained));
                }
            }
            if active_omission && frame == 3 {
                config.omission = omission_parameters(
                    tree.support_sigma,
                    1.0,
                    world,
                    scale,
                    clip,
                    Vec2::new(128.0, 96.0),
                    false,
                    true,
                );
                queue.write_buffer(&config_buffer, 0, bytemuck::bytes_of(&config));
                expected = (1..tree.root_count + 2).collect();
            }
            if omission_case && !active_omission {
                config.clip_from_world = Mat4::from_translation(if frame == 2 {
                    Vec3::X * -3.0
                } else {
                    Vec3::ZERO
                })
                .to_cols_array();
                config.frustum = test_frustum(Mat4::from_cols_array(&config.clip_from_world));
                queue.write_buffer(&config_buffer, 0, bytemuck::bytes_of(&config));
                let mut resident = pages.clone();
                if frame != 0 {
                    resident[512] = GpuLodPagePlacement {
                        start: 512,
                        count: 1,
                    };
                }
                queue.write_buffer(&page_buffer, 0, bytemuck::cast_slice(&resident));
                expected = match frame {
                    0 => vec![0],
                    2 => vec![514, 515],
                    _ => vec![513],
                };
            }
            if camera_motion {
                let first = if frame == 2 { 256 } else { 0 };
                config.clip_from_world =
                    Mat4::from_translation(Vec3::X * if first == 0 { 3.0 } else { -3.0 })
                        .to_cols_array();
                queue.write_buffer(&config_buffer, 0, bytemuck::bytes_of(&config));
                expected = (0..512)
                    .filter(|root| !(first..first + 128).contains(root))
                    .collect();
                expected.extend(512 + first * 2..512 + (first + 128) * 2);
                expected.sort_unstable();
            }
            if canonical_motion {
                // Change only priority, then relocate the same immutable nodes.
                config.clip_from_world = (clip
                    * Mat4::from_translation(Vec3::Z * if frame == 1 { 9.9 } else { 0.0 }))
                .to_cols_array();
                queue.write_buffer(&config_buffer, 0, bytemuck::bytes_of(&config));
                if frame == 2 {
                    let relocated = pages
                        .iter()
                        .map(|page| GpuLodPagePlacement {
                            start: page.start + 4096,
                            count: page.count,
                        })
                        .collect::<Vec<_>>();
                    queue.write_buffer(&page_buffer, 0, bytemuck::cast_slice(&relocated));
                    for value in &mut expected {
                        *value += 4096;
                    }
                }
            }
            let mut encoder = device.create_command_encoder(&default());
            let encode_block =
                |encoder: &mut wgpu::CommandEncoder, dispatches: &[(usize, Option<u32>)]| {
                    let encode_dispatch =
                        |pass: &mut wgpu::ComputePass<'_>,
                         &(stage, direct): &(usize, Option<u32>)| {
                            pass.set_pipeline(&stages[stage]);
                            if let Some(groups) = direct {
                                pass.dispatch_workgroups(groups, 1, 1);
                            } else {
                                pass.dispatch_workgroups_indirect(&dispatch, 0);
                            }
                        };
                    if camera_motion && frame == 0 {
                        // Reference command scheduling from before pass batching.
                        for instruction in dispatches {
                            let mut pass = encoder.begin_compute_pass(&default());
                            pass.set_bind_group(0, &bind, &[]);
                            encode_dispatch(&mut pass, instruction);
                        }
                    } else {
                        let mut pass = encoder.begin_compute_pass(&default());
                        pass.set_bind_group(0, &bind, &[]);
                        for instruction in dispatches {
                            encode_dispatch(&mut pass, instruction);
                        }
                    }
                };
            traversal_schedule(&config, tree.max_depth, |copy_args, _label, steps| {
                if steps.is_empty() {
                    if let Some(annotation) = &annotation {
                        if annotation_active {
                            encoder.copy_buffer_to_buffer(
                                annotation,
                                0,
                                &entries,
                                u64::from(capacity) * 8,
                                entry_tail_bytes,
                            );
                        } else {
                            // Near-safety or resource bypass leaves no active
                            // descriptors, without changing the finite target.
                            encoder.clear_buffer(&entries, u64::from(capacity) * 8, None);
                        }
                    }
                    return;
                }
                if copy_args {
                    encoder.copy_buffer_to_buffer(&work, 0, &dispatch, 0, 12);
                }
                encode_block(&mut encoder, steps);
            });
            encoder.copy_buffer_to_buffer(&draw, 0, &readback, 0, 32);
            encoder.copy_buffer_to_buffer(&work, 0, &readback, 32, work_size);
            encoder.copy_buffer_to_buffer(
                &entries,
                0,
                &readback,
                32 + work_size,
                u64::from(capacity) * 8,
            );
            let submission = queue.submit([encoder.finish()]);
            let (sender, receiver) = std::sync::mpsc::channel();
            readback
                .slice(..)
                .map_async(wgpu::MapMode::Read, move |result| {
                    sender.send(result).unwrap()
                });
            device
                .poll(wgpu::PollType::Wait {
                    submission_index: Some(submission),
                    timeout: Some(std::time::Duration::from_secs(10)),
                })
                .unwrap();
            receiver
                .recv_timeout(std::time::Duration::from_secs(1))
                .unwrap()
                .unwrap();
            let bytes = readback.slice(..).get_mapped_range();
            let words: &[u32] = bytemuck::cast_slice(&bytes);
            assert_eq!(words[0], 4, "{case}");
            assert_eq!(words[4] & 1, 0, "{case}: incomplete cut");
            if contention
                && !pending_demand
                && !desired_residency
                && !camera_motion
                && !cutoff_case
                && !omission_case
                && !priority_case
            {
                assert_eq!(words[4], 0, "contention must not report exhausted budgets");
                assert_eq!(
                    words[8 + 9],
                    visit_cap,
                    "every child cohort must be examined"
                );
            }
            assert!(words[8 + 9] <= visit_cap, "{case}: exceeded visit cap");
            assert!(
                words[8 + 5] <= frontier_cap,
                "{case}: exceeded frontier cap"
            );
            assert_eq!(words[1], expected.len() as u32, "{case}");
            let start = (32 + work_size) as usize / 4;
            let mut actual = (0..words[1] as usize)
                .map(|index| words[start + index * 2 + 1])
                .collect::<Vec<_>>();
            if canonical_motion {
                let relocation = if frame == 2 { 4096 } else { 0 };
                let stable = (0..words[1] as usize)
                    .map(|index| {
                        (
                            words[start + index * 2],
                            words[start + index * 2 + 1] - relocation,
                        )
                    })
                    .collect::<Vec<_>>();
                if frame == 0 {
                    held_entries = Some(stable);
                } else {
                    assert_eq!(
                        held_entries.as_ref().unwrap(),
                        &stable,
                        "same cut must preserve exact-depth tie order and sampling identities across priority changes and page relocation"
                    );
                }
            }
            // The visible range order is canonical even when admission priority
            // differs. Every key is a function of immutable node and local record.
            let range_start = 8 + 16 + offsets[2] as usize;
            let mut previous_rank = None;
            let mut physical_end = 0;
            let mut omitted = 0;
            for range in words[range_start..range_start + words[8 + 5] as usize * 4].chunks_exact(4)
            {
                let node = range[0];
                assert!(
                    previous_rank
                        .is_none_or(|previous| previous < tree.nodes[node as usize].counts[3]),
                    "{case}: noncanonical source-domain range order"
                );
                previous_rank = Some(tree.nodes[node as usize].counts[3]);
                assert_eq!(
                    range[1], physical_end,
                    "{case}: noncontiguous physical ranges"
                );
                physical_end += range[2];
                omitted += u32::from(range[3] != 0);
                assert!(range[3] == 0 || range[3] == super::super::OMITTED_OUTSIDE_VIEW);
                assert_eq!(
                    range[2],
                    if range[3] == 0 {
                        tree.nodes[node as usize].counts[0]
                    } else {
                        0
                    }
                );
                for local in 0..range[2] {
                    let mut key = node.wrapping_mul(0x9e37_79b9) ^ local;
                    key = (key ^ (key >> 16)).wrapping_mul(0x7feb_352d);
                    key = (key ^ (key >> 15)).wrapping_mul(0x846c_a68b);
                    key ^= key >> 16;
                    assert_eq!(
                        words[start + (range[1] + local) as usize * 2],
                        key,
                        "{case}: unstable sampling identity"
                    );
                }
            }
            assert_eq!(physical_end, words[1]);
            if omission_case {
                let fallback = !active_omission && frame == 0;
                let logical_count = tree.root_count + u32::from(!fallback);
                assert_eq!(
                    words[8 + 5],
                    logical_count,
                    "physical omission cannot alter logical coverage"
                );
                assert_eq!(omitted, logical_count - expected.len() as u32);
                assert_eq!(
                    words[8 + 10],
                    u32::from(fallback) * 2,
                    "an absent offscreen sibling still requires the complete cohort"
                );
                assert_eq!(
                    words[4], 0,
                    "resident fallback must preserve complete coverage"
                );
                assert_eq!(
                    words[8 + 11],
                    logical_count + u32::from(annotation_active),
                    "every logical page remains pinned independently of physical omission"
                );
                let selected_start = 8 + 16 + feedback_offsets[2] as usize;
                let retained = words[selected_start..selected_start + words[8 + 11] as usize]
                    .iter()
                    .copied()
                    .collect::<HashSet<_>>();
                let replaced_root = u32::from(!active_omission && frame == 2);
                let mut expected_retained = (0..tree.root_count).collect::<HashSet<_>>();
                if !fallback {
                    if !annotation_active {
                        expected_retained.remove(&replaced_root);
                    }
                    expected_retained.extend(512 + replaced_root * 2..514 + replaced_root * 2);
                }
                assert_eq!(retained, expected_retained);
                let ranges = &words[range_start..range_start + words[8 + 5] as usize * 4];
                if frame == 1 {
                    held_resolver_pages = retained;
                    held_omission_ranges = Some(ranges.to_vec());
                } else if frame == if active_omission { 2 } else { 3 } {
                    assert_eq!(
                        held_omission_ranges.as_deref().unwrap(),
                        ranges,
                        "the same fully resident logical cut and physical intervals must return after camera motion or removal of unneeded pages"
                    );
                }
            }
            if camera_motion {
                assert_eq!(
                    words[4], 2,
                    "only record admission limits this resident cut"
                );
                assert_eq!(
                    words[8 + 10],
                    0,
                    "resident camera motion needs no page demand"
                );
                if frame == 0 {
                    held_order = Some(actual.clone());
                } else if frame == 2 {
                    assert_ne!(
                        held_order.as_ref().unwrap(),
                        &actual,
                        "current camera must replace the visible refinement in this dispatch"
                    );
                } else {
                    assert_eq!(
                        held_order.as_ref().unwrap(),
                        &actual,
                        "batched passes must preserve the separate-pass cut and record order for held and returned cameras"
                    );
                }
            }
            if cutoff_case {
                let base = 8 + 16 + config.spatial[2] as usize;
                let tau = f32::from_bits(words[base]);
                assert_eq!(words[base + 1], 1, "{case}: exhaustive threshold required");
                let parent_score = f32::from_bits(words[base + 8]);
                let other_score = f32::from_bits(words[base + 9]);
                assert!(parent_score.is_finite() && other_score > 0.0);
                let descendant_score = f32::from_bits(words[base + 8 + 1024]);
                assert_eq!(
                    descendant_score,
                    0.5 * parent_score,
                    "large child severity must obey strict ancestry score decay"
                );
                assert_eq!(
                    f32::from_bits(words[base + 8 + 512]),
                    parent_score,
                    "canonical root alias must carry the same annotation score"
                );
                if case == "cutoff_exact" {
                    assert_eq!(tau, 0.0, "no excluded candidate means actual children");
                    assert_eq!(words[4], 0);
                } else {
                    assert_eq!(tau, other_score, "first excluded score defines cutoff");
                    assert_eq!(
                        words[4], 10,
                        "record cap and bounded pre-band demand limit this cut"
                    );
                    if case == "cutoff_tie" {
                        assert_eq!(parent_score, tau);
                        assert_eq!(words[base + 3], 0, "threshold ties retain actual parents");
                    } else {
                        let weight = (parent_score - tau) / tau;
                        assert!(
                            (weight - 0.5).abs() < 1e-5,
                            "{case}: uncertified budget-driven cohort needs a finite interior weight"
                        );
                        assert_eq!(words[base + 3], 1);
                    }
                }
                assert_eq!(
                    words[8 + 10],
                    if case == "cutoff_exact" {
                        0
                    } else {
                        request_cap
                    },
                    "bounded pre-band demand must not change the logical cutoff"
                );
            }
            if case == "cutoff_budget" {
                assert_eq!(
                    &actual[..4],
                    &[4096, 4097, 513, 1],
                    "refined source domain must precede unrelated coarse roots at exact depth ties"
                );
            }
            if case == "cutoff_missing" {
                let request_start = 8 + 16 + feedback_offsets[1] as usize;
                assert_eq!(
                    &words[request_start..request_start + 2],
                    &[512, 513],
                    "necessary missing cohort must precede speculative pre-band demand"
                );
            }
            if desired_residency {
                // Every synthetic root/child center is identical: this is an
                // exact depth tie. Refining root zero must keep its optical
                // group before root one, even though the child node indices
                // occur after all coarse parents in the compiled topology.
                if case == "desired_residency_early" {
                    assert_eq!(&actual[..3], &[512, 513, 1]);
                } else {
                    assert_eq!(&actual[..2], &[0, 1]);
                }
            }
            actual.sort_unstable();
            assert_eq!(actual, expected, "{case}: GPU cut differs from CPU oracle");
            let requests = words[8 + 10];
            assert!(requests <= request_cap, "{case}: exceeded request cap");
            if desired_residency {
                assert_eq!(words[4], 2, "{case}: only the record budget is exhausted");
                assert_eq!(
                    requests,
                    if case == "desired_residency_early" {
                        0
                    } else {
                        2
                    },
                    "{case}: unrelated resident coverage must not redirect desired demand"
                );
                assert_eq!(
                    words[8 + 9],
                    tree.root_count + 2,
                    "{case}: logical visits must be independent of child residency"
                );
            }
            let request_start = 8 + 16 + feedback_offsets[1] as usize;
            let unique = words[request_start..request_start + requests as usize]
                .iter()
                .copied()
                .collect::<HashSet<_>>();
            if omission_case {
                assert_eq!(
                    unique,
                    if !active_omission && frame == 0 {
                        HashSet::from([512, 513])
                    } else {
                        HashSet::new()
                    },
                    "missing offscreen children use the existing complete-cohort demand"
                );
            }
            if desired_residency {
                let node_bitmap = config.counts[0].div_ceil(32);
                let desired_start = 8
                    + 16
                    + feedback_offsets[2] as usize
                    + frontier_cap as usize
                    + 6 * frontier_cap as usize
                    + 8 * frontier_cap.div_ceil(256) as usize
                    + 13
                    + 8 * node_bitmap as usize;
                assert_eq!(
                    words[desired_start], 1,
                    "root zero must own the logical split independent of availability"
                );
                assert!(
                    words[desired_start + 1..desired_start + node_bitmap as usize]
                        .iter()
                        .all(|word| *word == 0)
                );
                if case == "desired_residency_late" {
                    assert_eq!(
                        unique,
                        HashSet::from([512, 513]),
                        "a later ready cohort cannot cancel the earlier desired cohort"
                    );
                }
            }
            if absent {
                if bound != 0 {
                    assert_eq!(requests, 0, "a rejected cohort must publish no demand");
                    if bound == 4 {
                        assert_eq!(
                            words[8 + 9] as usize,
                            manifest.nodes.len(),
                            "request capacity must not alter the original logical target"
                        );
                    }
                } else {
                    assert!(
                        cohort_pages.len() > 1,
                        "fixture must include resident siblings"
                    );
                    assert_eq!(
                        unique, cohort_pages,
                        "loaded siblings must remain demanded until their cohort is complete"
                    );
                }
            }
            if priority_case {
                assert_eq!(
                    words[4], 2,
                    "{case}: priority may not break the record budget or covering cut"
                );
                if priority_demand {
                    assert_eq!(
                        requests, 2,
                        "only the highest-priority whole sibling cohort fits"
                    );
                    assert_eq!(
                        unique,
                        HashSet::from([1534, 1535]),
                        "the near proxy must receive demand before earlier distant branches"
                    );
                } else {
                    assert_eq!(
                        requests, 0,
                        "resident priority needs no missing-page demand"
                    );
                    if near_spine {
                        assert!(
                            actual.contains(&1535)
                                && actual.contains(&1536)
                                && actual.contains(&1537)
                        );
                        assert!(
                            !actual.contains(&1534),
                            "near descendants must precede lower-priority root splits"
                        );
                    } else {
                        assert!(actual.contains(&1534) && actual.contains(&1535));
                    }
                    assert!(
                        !actual.contains(&511),
                        "the highest-priority proxy must be replaced by its complete children"
                    );
                }
            }
            if pending_demand {
                assert_eq!(
                    words[4], 2,
                    "only the actual record limit should stop demand"
                );
                assert_eq!(requests, 512, "only 256 complete future splits fit");
                assert_eq!(
                    words[1] + requests / 2,
                    capacity,
                    "parent cover and future split deltas must share one budget"
                );
                for pair in words[request_start..request_start + requests as usize].chunks_exact(2)
                {
                    assert_eq!(pair[0] % 2, 0);
                    assert_eq!(
                        pair[1],
                        pair[0] + 1,
                        "demand must preserve complete sibling pairs"
                    );
                }
            }
            if bound != 0 {
                assert_ne!(
                    words[4] & [0, 2, 16, 4, 8][bound],
                    0,
                    "{case}: no pressure flag"
                );
            }
            drop(bytes);
            readback.unmap();
        }
    }
}

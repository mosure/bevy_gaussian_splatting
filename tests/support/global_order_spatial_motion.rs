//! Test-only camera plan and one outstanding readback of existing GPU output.
use bevy::{
    prelude::*,
    render::{
        render_resource::{
            BufferDescriptor, BufferUsages, CommandEncoderDescriptor, MapMode, PollType,
        },
        renderer::{RenderDevice, RenderQueue},
        view::ExtractedView,
    },
};
use bevy_gaussian_splatting::{
    gaussian::formats::planar_3d_lod::GaussianLodManifest,
    render::traversal::{GpuLodHierarchy, GpuLodTraversalOutputs},
};
use std::sync::{Arc, Mutex};

pub const FOV: f32 = 0.35;

pub struct Motion {
    pub depths: [f32; 12],
    pub step: usize,
    pub baseline: Vec<u8>,
    pub previous: Vec<u8>,
    pub previous_weight: f32,
    pub near_cut: Option<(u64, Vec<[u32; 2]>)>,
}

impl Motion {
    pub fn new(manifest: &GaussianLodManifest) -> Self {
        let mut nodes: Vec<_> = manifest.nodes.iter().filter(|n| !n.is_leaf()).collect();
        let root = nodes.remove(
            nodes
                .iter()
                .position(|n| n.id == manifest.roots[0])
                .unwrap(),
        );
        assert_eq!(nodes.len(), 2);
        nodes.sort_by(|a, b| a.bounds.center()[2].total_cmp(&b.bounds.center()[2]));
        assert!(nodes[0].bounds.center()[2] < -0.9 && nodes[1].bounds.center()[2] > 0.9);
        // Exact axial-camera specialization of the production finite scheduling
        // estimate. It only plans poses; the test asserts GPU descriptor weights.
        let ratio = |camera_z: f32| {
            let score = |node: &bevy_gaussian_splatting::gaussian::formats::planar_3d_lod::GaussianLodNode| {
                let center = node.bounds.center();
                let depth = camera_z - center[2];
                let support_z = (node.bounds.max[2] - center[2]).max(center[2] - node.bounds.min[2]);
                let focal = 32.0 / (FOV * 0.5).tan();
                let norm = focal * (1.0 + (center[0] * center[0] + center[1] * center[1]) / (depth * depth)).sqrt();
                let minimum = (depth - support_z).max(0.1);
                let error_minimum = (minimum - node.error.geometric).max(0.1);
                (node.error.geometric * (norm / error_minimum) * (depth / error_minimum))
                    .max(node.bounds.radius() * (norm / minimum) * 0.125)
            };
            let inherited = score(root) * 0.5;
            score(nodes[1]).min(inherited) / score(nodes[0]).min(inherited)
        };
        let solve = |target: f32| {
            let (mut near, mut far) = (3.0, 30.0);
            assert!(ratio(near) > target && ratio(far) < target);
            // Fixed scalar bisection, not a rendered pose sweep.
            for _ in 0..32 {
                let midpoint = (near + far) * 0.5;
                if ratio(midpoint) > target {
                    near = midpoint;
                } else {
                    far = midpoint;
                }
            }
            if (ratio(near) - target).abs() < (ratio(far) - target).abs() {
                near
            } else {
                far
            }
        };
        let middle = solve(1.5);
        let parent = solve(1.0);
        let child = solve(2.0);
        // The orthographic follow-up keeps score ratio 1.5 and XY projection
        // fixed while only camera Z crosses this parent's expanded near box.
        let near_parent = nodes[0];
        let center_z = near_parent.bounds.center()[2];
        let half_z =
            (near_parent.bounds.max[2] - center_z).max(center_z - near_parent.bounds.min[2]);
        let tolerance = 0.00002 * near_parent.bounds.radius().max(1.0);
        let envelope = (half_z + tolerance) * 2.0;
        let boundary = center_z + 0.1 + envelope;
        Self {
            depths: [
                middle,
                middle + 0.0001,
                solve(1.25),
                parent,
                parent,
                child - 0.0001,
                child + 0.0001,
                middle,
                boundary + 1.1 * envelope,
                boundary + 0.01 * envelope,
                boundary - 0.01 * envelope,
                boundary + 1.1 * envelope,
            ],
            step: 0,
            baseline: Vec::new(),
            previous: Vec::new(),
            previous_weight: 0.0,
            near_cut: None,
        }
    }

    pub fn capacity(&self) -> u32 {
        if self.step == 4 { 2 } else { 3 }
    }
}

#[derive(Clone, Debug)]
pub struct Sample {
    pub step: usize,
    pub submission: u64,
    pub residency_generation: u64,
    pub selected: u32,
    pub traversal_flags: u32,
    pub tail: [u32; 8],
    pub weights: Vec<f32>,
    pub entries: Vec<[u32; 2]>,
}

#[derive(Debug, Default)]
struct Mailbox {
    request: Option<(usize, Entity, Entity)>,
    pending: bool,
    result: Option<Sample>,
    lookup_misses: u32,
}

#[derive(Resource, Clone, Debug, Default)]
pub struct Probe(Arc<Mutex<Mailbox>>);

impl Probe {
    pub fn request(&self, step: usize, camera: Entity, cloud: Entity) {
        let mut mailbox = self.0.lock().unwrap();
        if !mailbox.pending && mailbox.result.is_none() && mailbox.request.is_none() {
            mailbox.request = Some((step, camera, cloud));
            mailbox.lookup_misses = 0;
        }
    }

    pub fn sample(&self) -> Option<Sample> {
        self.0.lock().unwrap().result.clone()
    }
    pub fn clear(&self) {
        self.0.lock().unwrap().result = None;
    }
}

pub fn readback(
    probe: Res<Probe>,
    device: Res<RenderDevice>,
    queue: Res<RenderQueue>,
    views: Query<&ExtractedView>,
    clouds: Query<Entity, With<GpuLodHierarchy>>,
    outputs: Res<GpuLodTraversalOutputs>,
) {
    let _ = device.poll(PollType::Poll);
    let Some((step, camera, cloud)) = probe.0.lock().unwrap().request else {
        return;
    };
    // A main camera can own several retained views (including UI subviews).
    // Locate the actual hierarchy output instead of taking the first view.
    let output = views
        .iter()
        .filter(|view| view.retained_view_entity.main_entity.id() == camera)
        .find_map(|view| {
            clouds.iter().find_map(|entity| {
                outputs
                    .get(view.retained_view_entity, entity)
                    .filter(|output| {
                        output.main_cloud == cloud && output.is_ready() && output.submission > 0
                    })
            })
        });
    let Some(output) = output else {
        // A projection/capacity change can prepare a replacement pipeline or
        // workspace after the main-world request. Keep this one request pending;
        // the app's existing timeout bounds the retry without copying stale data.
        let mut mailbox = probe.0.lock().unwrap();
        mailbox.lookup_misses += 1;
        if mailbox.lookup_misses == 8 {
            let view_keys: Vec<_> = views.iter().map(|view| view.retained_view_entity).collect();
            let entities: Vec<_> = clouds.iter().collect();
            let mut candidates = Vec::new();
            for view in &view_keys {
                for entity in &entities {
                    if let Some(output) = outputs.get(*view, *entity) {
                        candidates.push((
                            *view,
                            *entity,
                            output.main_cloud,
                            output.is_ready(),
                            output.submission,
                        ));
                    }
                }
            }
            eprintln!(
                "spatial probe lookup pending after 8 renders: request={:?}; views={view_keys:?}; render_clouds={entities:?}; outputs={candidates:?}",
                mailbox.request
            );
        }
        return;
    };
    // Existing public output buffers suffice. The fixture admits 16 node ranges:
    // 64B feedback, 32B tail header, 16 descriptors and up to 16 entries:
    // one 736B slot, using only existing producer buffers.
    const TAIL_BYTES: u64 = 32 + 16 * 32;
    const BYTES: u64 = 64 + TAIL_BYTES + 16 * 8;
    let tail_offset = u64::from(output.capacity) * 8;
    assert!(output.entries.size() >= tail_offset + TAIL_BYTES);
    let buffer = device.create_buffer(&BufferDescriptor {
        label: Some("spatial_motion_test_receipt"),
        size: BYTES,
        usage: BufferUsages::MAP_READ | BufferUsages::COPY_DST,
        mapped_at_creation: false,
    });
    let mut encoder = device.create_command_encoder(&CommandEncoderDescriptor {
        label: Some("spatial_motion_test_receipt"),
    });
    encoder.copy_buffer_to_buffer(&output.feedback, 0, &buffer, 0, 64);
    encoder.copy_buffer_to_buffer(&output.entries, tail_offset, &buffer, 64, TAIL_BYTES);
    assert!(output.capacity <= 16);
    encoder.copy_buffer_to_buffer(
        &output.entries,
        0,
        &buffer,
        64 + TAIL_BYTES,
        u64::from(output.capacity) * 8,
    );
    queue.submit([encoder.finish()]);
    let mut mailbox = probe.0.lock().unwrap();
    mailbox.request = None;
    mailbox.pending = true;
    drop(mailbox);
    let mailbox = probe.0.clone();
    let read = buffer.clone();
    let submission = output.submission;
    let residency_generation = output.residency_generation;
    buffer.slice(..).map_async(MapMode::Read, move |result| {
        result.expect("spatial motion descriptor readback");
        let data = read.slice(..).get_mapped_range();
        let words: Vec<u32> = data
            .chunks_exact(4)
            .map(bytemuck::pod_read_unaligned)
            .collect();
        let tail: [u32; 8] = words[16..24].try_into().unwrap();
        assert!(tail[2] <= 16);
        assert!(words[8] <= 16);
        let weights = words[24..24 + tail[2] as usize * 8]
            .chunks_exact(8)
            .filter(|descriptor| descriptor[7] != 0)
            .map(|descriptor| f32::from_bits(descriptor[6]))
            .collect();
        let entry_start = (64 + TAIL_BYTES) as usize / 4;
        let entries = words[entry_start..entry_start + words[8] as usize * 2]
            .chunks_exact(2)
            .map(|entry| [entry[0], entry[1]])
            .collect();
        let sample = Sample {
            step,
            submission,
            residency_generation,
            selected: words[8],
            traversal_flags: words[12],
            tail,
            weights,
            entries,
        };
        drop(data);
        read.unmap();
        let mut mailbox = mailbox.lock().unwrap();
        mailbox.pending = false;
        mailbox.result = Some(sample);
    });
}

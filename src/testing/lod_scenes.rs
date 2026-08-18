use bevy::prelude::{Quat, Vec3};
use bevy_interleave::prelude::Planar;

use crate::{
    gaussian::formats::planar_3d::{Gaussian3d, PlanarGaussian3d},
    material::spherical_harmonics::{SH_COEFF_COUNT, SphericalHarmonicCoefficients},
};

#[derive(Clone, Copy, Debug, PartialEq)]
pub enum LodProjection {
    Perspective { vertical_fov_radians: f32 },
    Orthographic { vertical_world_size: f32 },
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct LodTestCamera {
    pub position: Vec3,
    pub target: Vec3,
    pub up: Vec3,
    pub projection: LodProjection,
    pub near: f32,
    pub far: f32,
    pub viewport: [u32; 2],
}

impl Default for LodTestCamera {
    fn default() -> Self {
        Self {
            position: Vec3::new(0.0, 0.0, 8.0),
            target: Vec3::ZERO,
            up: Vec3::Y,
            projection: LodProjection::Perspective {
                vertical_fov_radians: 60.0_f32.to_radians(),
            },
            near: 0.01,
            far: 1_000.0,
            viewport: [640, 480],
        }
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum LodScenePattern {
    NestedOctants,
    ScreenSpaceLadder,
    CheckerboardFacade,
    SharpEdge,
    TransparentDepthStack,
    NeedlesWiresFoliage,
    BoundaryStraddlers,
    ProjectionAdversaries,
    WorkgroupBoundary,
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct LodTestGaussian {
    pub stable_id: u64,
    pub gaussian: Gaussian3d,
}

#[derive(Clone, Debug, PartialEq)]
pub struct LodTestScene {
    pub name: &'static str,
    pub pattern: LodScenePattern,
    pub gaussians: Vec<LodTestGaussian>,
    pub camera: LodTestCamera,
}

impl LodTestScene {
    pub fn cloud(&self) -> PlanarGaussian3d {
        PlanarGaussian3d::from_interleaved(
            self.gaussians.iter().map(|entry| entry.gaussian).collect(),
        )
    }

    pub fn stable_selection_hash<'a>(ids: impl IntoIterator<Item = &'a u64>) -> u64 {
        ids.into_iter().fold(FNV_OFFSET, |hash, id| {
            id.to_le_bytes().into_iter().fold(hash, |hash, byte| {
                (hash ^ u64::from(byte)).wrapping_mul(FNV_PRIME)
            })
        })
    }

    pub fn nested_octants(levels: u32) -> Self {
        assert!((1..=5).contains(&levels), "levels must be in 1..=5");
        let mut points = vec![(Vec3::ZERO, 4.0_f32, 1_u64)];
        for level in 0..levels {
            let mut children = Vec::with_capacity(points.len() * 8);
            for (center, extent, path) in points {
                let child_extent = extent * 0.5;
                let offset = child_extent * 0.5;
                for child in 0..8_u64 {
                    let sign = Vec3::new(
                        if child & 1 == 0 { -1.0 } else { 1.0 },
                        if child & 2 == 0 { -1.0 } else { 1.0 },
                        if child & 4 == 0 { -1.0 } else { 1.0 },
                    );
                    children.push((center + sign * offset, child_extent, (path << 3) | child));
                }
            }
            points = children;
            debug_assert_eq!(points.len(), 8_usize.pow(level + 1));
        }

        let gaussians = points
            .into_iter()
            .enumerate()
            .map(|(index, (position, extent, path))| LodTestGaussian {
                stable_id: (u64::from(levels) << 56) | path,
                gaussian: gaussian(
                    position,
                    Vec3::splat((extent * 0.08).max(0.005)),
                    octant_color(index as u64),
                    0.85,
                    Quat::IDENTITY,
                    index as u64,
                ),
            })
            .collect();

        Self {
            name: "nested_octants",
            pattern: LodScenePattern::NestedOctants,
            gaussians,
            camera: LodTestCamera::default(),
        }
    }

    pub fn screen_space_ladder() -> Self {
        let mut gaussians = Vec::new();
        let depths = [2.0_f32, 4.0, 8.0, 16.0, 32.0];
        for (rung, depth) in depths.into_iter().enumerate() {
            let center = Vec3::new((rung as f32 - 2.0) * 1.4, 0.0, -depth);
            let spacing = depth * 0.015;
            for y in -2..=2 {
                for x in -2..=2 {
                    let local = Vec3::new(x as f32 * spacing, y as f32 * spacing, 0.0);
                    let id = (rung * 25 + ((y + 2) * 5 + (x + 2)) as usize) as u64;
                    gaussians.push(LodTestGaussian {
                        stable_id: id,
                        gaussian: gaussian(
                            center + local,
                            Vec3::splat(spacing * 0.45),
                            [0.15 + rung as f32 * 0.15, 0.7, 0.3],
                            0.9,
                            Quat::IDENTITY,
                            id,
                        ),
                    });
                }
            }
        }
        let camera = LodTestCamera {
            position: Vec3::ZERO,
            target: -Vec3::Z,
            far: 64.0,
            ..Default::default()
        };
        Self {
            name: "screen_space_ladder",
            pattern: LodScenePattern::ScreenSpaceLadder,
            gaussians,
            camera,
        }
    }

    pub fn checkerboard_facade(width: u32, height: u32) -> Self {
        assert!(width > 1 && height > 1);
        let mut gaussians = Vec::with_capacity((width * height) as usize);
        for y in 0..height {
            for x in 0..width {
                let id = u64::from(y) * u64::from(width) + u64::from(x);
                let color = if (x + y) % 2 == 0 {
                    [0.95, 0.05, 0.05]
                } else {
                    [0.03, 0.05, 0.95]
                };
                gaussians.push(LodTestGaussian {
                    stable_id: id,
                    gaussian: gaussian(
                        Vec3::new(
                            (x as f32 - (width - 1) as f32 * 0.5) * 0.08,
                            (y as f32 - (height - 1) as f32 * 0.5) * 0.08,
                            0.0,
                        ),
                        Vec3::new(0.042, 0.042, 0.008),
                        color,
                        0.96,
                        Quat::IDENTITY,
                        id,
                    ),
                });
            }
        }
        Self {
            name: "checkerboard_facade",
            pattern: LodScenePattern::CheckerboardFacade,
            gaussians,
            camera: LodTestCamera::default(),
        }
    }

    pub fn sharp_edge(width: u32, height: u32) -> Self {
        assert!(width > 1 && height > 1);
        let mut gaussians = Vec::with_capacity((width * height * 2) as usize);
        for layer in 0..2_u32 {
            for y in 0..height {
                for x in 0..width {
                    let ordinal = layer * width * height + y * width + x;
                    let side = x < width / 2;
                    let z = if side { -0.025 } else { 0.025 } + layer as f32 * 0.012;
                    let color = if side {
                        [1.0, 0.08, 0.02]
                    } else {
                        [0.02, 0.8, 1.0]
                    };
                    gaussians.push(LodTestGaussian {
                        stable_id: u64::from(ordinal),
                        gaussian: gaussian(
                            Vec3::new(
                                (x as f32 - (width - 1) as f32 * 0.5) * 0.055,
                                (y as f32 - (height - 1) as f32 * 0.5) * 0.055,
                                z,
                            ),
                            Vec3::new(0.032, 0.032, 0.004),
                            color,
                            0.72,
                            Quat::IDENTITY,
                            u64::from(ordinal),
                        ),
                    });
                }
            }
        }
        Self {
            name: "sharp_edge",
            pattern: LodScenePattern::SharpEdge,
            gaussians,
            camera: LodTestCamera::default(),
        }
    }

    pub fn transparent_depth_stack(layers: u32) -> Self {
        assert!(layers > 1);
        let mut gaussians = Vec::with_capacity((layers * 25) as usize);
        for layer in 0..layers {
            for y in -2..=2 {
                for x in -2..=2 {
                    let ordinal = layer * 25 + ((y + 2) * 5 + x + 2) as u32;
                    let mut color = [0.15, 0.15, 0.15];
                    color[layer as usize % 3] = 0.9;
                    gaussians.push(LodTestGaussian {
                        stable_id: u64::from(ordinal),
                        gaussian: gaussian(
                            Vec3::new(x as f32 * 0.11, y as f32 * 0.11, layer as f32 * 0.015),
                            Vec3::new(0.075, 0.075, 0.01),
                            color,
                            0.18,
                            Quat::from_rotation_y(layer as f32 * 0.07),
                            u64::from(ordinal),
                        ),
                    });
                }
            }
        }
        Self {
            name: "transparent_depth_stack",
            pattern: LodScenePattern::TransparentDepthStack,
            gaussians,
            camera: LodTestCamera::default(),
        }
    }

    pub fn needles_wires_foliage() -> Self {
        let mut gaussians = Vec::with_capacity(384);
        for i in 0..128_u64 {
            let t = i as f32 / 127.0;
            gaussians.push(LodTestGaussian {
                stable_id: i,
                gaussian: gaussian(
                    Vec3::new(-1.1 + t * 2.2, (t * 12.0).sin() * 0.12, -0.2),
                    Vec3::new(0.035, 0.004, 0.004),
                    [0.85, 0.65, 0.1],
                    0.82,
                    Quat::from_rotation_z((t * 12.0).cos() * 0.35),
                    i,
                ),
            });
            gaussians.push(LodTestGaussian {
                stable_id: 128 + i,
                gaussian: gaussian(
                    Vec3::new((t * 31.0).sin(), -0.8 + t * 1.6, (t * 19.0).cos() * 0.2),
                    Vec3::new(0.006, 0.055, 0.009),
                    [0.08, 0.7, 0.12],
                    0.7,
                    Quat::from_rotation_x(t * 1.3),
                    128 + i,
                ),
            });
            gaussians.push(LodTestGaussian {
                stable_id: 256 + i,
                gaussian: gaussian(
                    Vec3::new((t * 97.0).sin(), (t * 53.0).cos(), 0.25 + t * 0.3),
                    Vec3::new(0.055, 0.015, 0.0025),
                    [0.15, 0.85, 0.3],
                    0.62,
                    Quat::from_rotation_y(t * 5.0),
                    256 + i,
                ),
            });
        }
        Self {
            name: "needles_wires_foliage",
            pattern: LodScenePattern::NeedlesWiresFoliage,
            gaussians,
            camera: LodTestCamera::default(),
        }
    }

    pub fn boundary_straddlers() -> Self {
        let coordinates = [-1.0_f32, -f32::EPSILON, 0.0, f32::EPSILON, 1.0];
        let mut gaussians = Vec::with_capacity(coordinates.len().pow(3) + 3);
        for (ordinal, (&x, &y, &z)) in coordinates
            .iter()
            .flat_map(|x| coordinates.iter().map(move |y| (x, y)))
            .flat_map(|(x, y)| coordinates.iter().map(move |z| (x, y, z)))
            .enumerate()
        {
            gaussians.push(LodTestGaussian {
                stable_id: ordinal as u64,
                gaussian: gaussian(
                    Vec3::new(x, y, z),
                    Vec3::new(0.14, 0.025, 0.008),
                    octant_color(ordinal as u64),
                    0.8,
                    Quat::from_rotation_z(ordinal as f32 * 0.17),
                    ordinal as u64,
                ),
            });
        }
        for (offset, position) in [
            Vec3::new(100.0, 0.0, 0.0),
            Vec3::new(-100.0, 0.0, 0.0),
            Vec3::new(0.0, 0.0, 100.0),
        ]
        .into_iter()
        .enumerate()
        {
            let id = gaussians.len() as u64;
            gaussians.push(LodTestGaussian {
                stable_id: id,
                gaussian: gaussian(
                    position,
                    Vec3::splat(2.0 + offset as f32),
                    [1.0, 1.0, 0.0],
                    0.4,
                    Quat::IDENTITY,
                    id,
                ),
            });
        }
        Self {
            name: "boundary_straddlers",
            pattern: LodScenePattern::BoundaryStraddlers,
            gaussians,
            camera: LodTestCamera::default(),
        }
    }

    pub fn projection_adversaries() -> Self {
        let positions = [
            Vec3::ZERO,
            Vec3::new(0.0, 0.0, 0.009),
            Vec3::new(0.0, 0.0, -0.009),
            Vec3::new(4.0, 0.0, -2.0),
            Vec3::new(-4.0, 0.0, -2.0),
            Vec3::new(0.0, 4.0, -2.0),
            Vec3::new(0.0, -4.0, -2.0),
            Vec3::new(0.0, 0.0, -999.0),
        ];
        let gaussians = positions
            .into_iter()
            .enumerate()
            .map(|(index, position)| LodTestGaussian {
                stable_id: index as u64,
                gaussian: gaussian(
                    position,
                    if index < 3 {
                        Vec3::splat(0.03)
                    } else {
                        Vec3::new(2.0, 0.15, 0.15)
                    },
                    octant_color(index as u64),
                    0.75,
                    Quat::from_rotation_z(index as f32 * 0.31),
                    index as u64,
                ),
            })
            .collect();
        let camera = LodTestCamera {
            position: Vec3::ZERO,
            target: -Vec3::Z,
            near: 0.01,
            far: 1_000.0,
            ..Default::default()
        };
        Self {
            name: "projection_adversaries",
            pattern: LodScenePattern::ProjectionAdversaries,
            gaussians,
            camera,
        }
    }

    pub fn workgroup_boundary(count: usize) -> Self {
        let mut gaussians = Vec::with_capacity(count);
        for index in 0..count {
            let x = (index % 257) as f32 * 0.003;
            let y = (index / 257) as f32 * 0.003;
            gaussians.push(LodTestGaussian {
                stable_id: index as u64,
                gaussian: gaussian(
                    Vec3::new(x, y, 0.0),
                    Vec3::splat(0.002),
                    octant_color(index as u64),
                    0.8,
                    Quat::IDENTITY,
                    index as u64,
                ),
            });
        }
        Self {
            name: "workgroup_boundary",
            pattern: LodScenePattern::WorkgroupBoundary,
            gaussians,
            camera: LodTestCamera::default(),
        }
    }
}

pub fn all_small_lod_scenes() -> Vec<LodTestScene> {
    vec![
        LodTestScene::nested_octants(2),
        LodTestScene::screen_space_ladder(),
        LodTestScene::checkerboard_facade(24, 16),
        LodTestScene::sharp_edge(24, 16),
        LodTestScene::transparent_depth_stack(12),
        LodTestScene::needles_wires_foliage(),
        LodTestScene::boundary_straddlers(),
        LodTestScene::projection_adversaries(),
    ]
}

/// Lazy deterministic city-scale source. It represents well over 100M source Gaussians without
/// retaining any per-Gaussian allocation; tests and builders request individual pages on demand.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct VirtualCityScene {
    pub seed: u64,
    pub page_count: u32,
    pub gaussians_per_page: u32,
    pub grid_width: u32,
}

impl Default for VirtualCityScene {
    fn default() -> Self {
        Self {
            seed: 0x47a5_51a7_d15c_1a5e,
            page_count: 32_768,
            gaussians_per_page: 4_096,
            grid_width: 256,
        }
    }
}

impl VirtualCityScene {
    pub fn source_gaussian_count(self) -> u64 {
        u64::from(self.page_count) * u64::from(self.gaussians_per_page)
    }

    pub fn generate_page(self, page_index: u32) -> Option<Vec<LodTestGaussian>> {
        if page_index >= self.page_count || self.grid_width == 0 {
            return None;
        }
        let block_x = page_index % self.grid_width;
        let block_z = page_index / self.grid_width;
        let origin = Vec3::new(block_x as f32 * 24.0, 0.0, block_z as f32 * 24.0);
        let mut rng = StableRng::new(self.seed ^ u64::from(page_index));
        let mut page = Vec::with_capacity(self.gaussians_per_page as usize);
        for local_index in 0..self.gaussians_per_page {
            let facade = local_index % 5 != 0;
            let floor = (local_index / 128) % 32;
            let lane = local_index % 128;
            let (position, scale, color) = if facade {
                let side = lane % 4;
                let along = (lane / 4) as f32 * 0.28;
                let height = floor as f32 * 0.22;
                let building = (local_index / (128 * 32)) as f32;
                let base = origin + Vec3::new(3.0 + building * 5.0, height, 3.0 + building * 4.0);
                let offset = match side {
                    0 => Vec3::new(along, 0.0, 0.0),
                    1 => Vec3::new(0.0, 0.0, along),
                    2 => Vec3::new(-along, 0.0, 4.0),
                    _ => Vec3::new(4.0, 0.0, -along),
                };
                (
                    base + offset + rng.vec3_signed() * 0.015,
                    Vec3::new(0.18, 0.12, 0.025),
                    if (lane + floor) % 7 == 0 {
                        [0.9, 0.65, 0.2]
                    } else {
                        [0.22, 0.27, 0.32]
                    },
                )
            } else {
                (
                    origin
                        + Vec3::new(
                            rng.next_f32() * 24.0,
                            rng.next_f32() * 3.0,
                            rng.next_f32() * 24.0,
                        ),
                    Vec3::new(0.08, 0.18, 0.08),
                    [0.08, 0.42 + rng.next_f32() * 0.25, 0.1],
                )
            };
            let stable_id =
                u64::from(page_index) * u64::from(self.gaussians_per_page) + u64::from(local_index);
            page.push(LodTestGaussian {
                stable_id,
                gaussian: gaussian(
                    position,
                    scale,
                    color,
                    0.55 + rng.next_f32() * 0.4,
                    Quat::from_rotation_y(rng.next_signed_f32() * 0.2),
                    stable_id,
                ),
            });
        }
        Some(page)
    }
}

const FNV_OFFSET: u64 = 0xcbf2_9ce4_8422_2325;
const FNV_PRIME: u64 = 0x0000_0100_0000_01b3;

fn gaussian(
    position: Vec3,
    scale: Vec3,
    color: [f32; 3],
    opacity: f32,
    rotation: Quat,
    seed: u64,
) -> Gaussian3d {
    let mut coefficients = [0.0; SH_COEFF_COUNT];
    let dc_coefficient_count = 3.min(SH_COEFF_COUNT);
    coefficients[..dc_coefficient_count].copy_from_slice(&color[..dc_coefficient_count]);
    let mut rng = StableRng::new(seed ^ 0x6a09_e667_f3bc_c909);
    for coefficient in coefficients.iter_mut().skip(3) {
        *coefficient = rng.next_signed_f32() * 0.025;
    }
    Gaussian3d {
        position_visibility: [position.x, position.y, position.z, 1.0].into(),
        spherical_harmonic: SphericalHarmonicCoefficients { coefficients },
        rotation: rotation.to_array().into(),
        scale_opacity: [
            scale.x.max(1e-6),
            scale.y.max(1e-6),
            scale.z.max(1e-6),
            opacity.clamp(0.0, 1.0),
        ]
        .into(),
    }
}

fn octant_color(index: u64) -> [f32; 3] {
    [
        0.15 + 0.8 * ((index & 1) as f32),
        0.15 + 0.8 * (((index >> 1) & 1) as f32),
        0.15 + 0.8 * (((index >> 2) & 1) as f32),
    ]
}

#[derive(Clone, Copy, Debug)]
struct StableRng {
    state: u64,
}

impl StableRng {
    fn new(seed: u64) -> Self {
        Self { state: seed }
    }

    fn next_u64(&mut self) -> u64 {
        self.state = self.state.wrapping_add(0x9e37_79b9_7f4a_7c15);
        let mut value = self.state;
        value = (value ^ (value >> 30)).wrapping_mul(0xbf58_476d_1ce4_e5b9);
        value = (value ^ (value >> 27)).wrapping_mul(0x94d0_49bb_1331_11eb);
        value ^ (value >> 31)
    }

    fn next_f32(&mut self) -> f32 {
        let mantissa = (self.next_u64() >> 40) as u32;
        mantissa as f32 / (1_u32 << 24) as f32
    }

    fn next_signed_f32(&mut self) -> f32 {
        self.next_f32() * 2.0 - 1.0
    }

    fn vec3_signed(&mut self) -> Vec3 {
        Vec3::new(
            self.next_signed_f32(),
            self.next_signed_f32(),
            self.next_signed_f32(),
        )
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn fixtures_are_deterministic_and_finite() {
        let first = all_small_lod_scenes();
        let second = all_small_lod_scenes();
        assert_eq!(first, second);
        for scene in first {
            assert!(!scene.gaussians.is_empty(), "{} is empty", scene.name);
            for entry in scene.gaussians {
                assert!(
                    entry
                        .gaussian
                        .position_visibility
                        .position
                        .iter()
                        .all(|v| v.is_finite())
                );
                assert!(
                    entry
                        .gaussian
                        .scale_opacity
                        .scale
                        .iter()
                        .all(|v| v.is_finite() && *v > 0.0)
                );
                assert!(entry.gaussian.scale_opacity.opacity.is_finite());
            }
        }
    }

    #[test]
    fn nested_octants_have_exact_leaf_counts_and_unique_ids() {
        for levels in 1..=4 {
            let scene = LodTestScene::nested_octants(levels);
            assert_eq!(scene.gaussians.len(), 8_usize.pow(levels));
            let mut ids = scene
                .gaussians
                .iter()
                .map(|entry| entry.stable_id)
                .collect::<Vec<_>>();
            ids.sort_unstable();
            ids.dedup();
            assert_eq!(ids.len(), scene.gaussians.len());
        }
    }

    #[test]
    fn workgroup_fixture_preserves_requested_tail_counts() {
        for count in [0, 1, 63, 64, 65, 255, 256, 257, 511, 512, 513] {
            assert_eq!(
                LodTestScene::workgroup_boundary(count).gaussians.len(),
                count
            );
        }
    }

    #[test]
    fn virtual_city_exceeds_one_hundred_million_without_eager_storage() {
        let city = VirtualCityScene::default();
        assert!(city.source_gaussian_count() > 100_000_000);
        assert!(std::mem::size_of::<VirtualCityScene>() <= 32);

        let first = city.generate_page(17).unwrap();
        let repeated = city.generate_page(17).unwrap();
        assert_eq!(first, repeated);
        assert_eq!(first.len(), city.gaussians_per_page as usize);
        assert_eq!(first[0].stable_id, 17 * u64::from(city.gaussians_per_page));
        assert!(city.generate_page(city.page_count).is_none());
    }

    #[test]
    fn selection_hash_is_order_sensitive_and_repeatable() {
        let a = [1_u64, 2, 3];
        let b = [3_u64, 2, 1];
        assert_eq!(
            LodTestScene::stable_selection_hash(a.iter()),
            LodTestScene::stable_selection_hash(a.iter())
        );
        assert_ne!(
            LodTestScene::stable_selection_hash(a.iter()),
            LodTestScene::stable_selection_hash(b.iter())
        );
    }
}

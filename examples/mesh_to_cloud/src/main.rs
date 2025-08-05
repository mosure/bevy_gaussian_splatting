// Converts a mesh (monkey.glb) into a Gaussian cloud on CPU: one splat per vertex, edge, and face,
// with color derived from the primitive normal.
//
// Run: cargo run
// Ensure assets/scenes/monkey.glb exists under bevy_gaussian_splatting/assets.

use std::collections::HashSet;
use bevy::prelude::*;
use bevy::render::mesh::{
    Indices,
    PrimitiveTopology,
    VertexAttributeValues,
};

use bevy_gaussian_splatting::{
    CloudSettings,
    Gaussian3d,
    GaussianSplattingPlugin,
    PlanarGaussian3d,
    PlanarGaussian3dHandle,
    RasterizeMode,
};

const GLB_PATH: &str = "scenes/monkey.glb";

// Tunables for splat appearance
const DEFAULT_OPACITY: f32 = 0.8;
const DEFAULT_SCALE: f32 = 0.02; // isotropic; adjust to your mesh units
const EDGE_SCALE: f32 = 0.015;
const FACE_SCALE: f32 = 0.03;

// Entry
fn main() {
    App::new()
        .add_plugins(DefaultPlugins)
        .add_plugins(GaussianSplattingPlugin)
        .add_systems(Startup, (spawn_camera_and_light, load_monkey))
        .add_systems(Update, try_convert_loaded_mesh)
        .run();
}

fn spawn_camera_and_light(mut commands: Commands) {
    commands.spawn((
        Camera3d::default(),
        Transform::from_translation(Vec3::new(0.0, 0.6, 2.2)).looking_at(Vec3::ZERO, Vec3::Y),
    ));
    commands.spawn((
        DirectionalLight::default(),
        Transform::from_xyz(2.0, 4.0, 2.0).looking_at(Vec3::ZERO, Vec3::Y),
    ));
}

#[derive(Resource, Default)]
struct PendingScene(Handle<Scene>);

fn load_monkey(mut commands: Commands, assets: Res<AssetServer>) {
    let scene: Handle<Scene> = assets.load(GLB_PATH.to_string() + "#Scene0");
    commands.insert_resource(PendingScene(scene.clone()));
    commands.spawn((
        SceneRoot(scene),
        Transform::default(),
        Visibility::Visible,
    ));
}

fn try_convert_loaded_mesh(
    mut commands: Commands,
    pending: Option<Res<PendingScene>>,
    pnp: Query<(&Mesh3d, &GlobalTransform)>,
    meshes: Res<Assets<Mesh>>,
    mut planar_gaussians: ResMut<Assets<PlanarGaussian3d>>,
) {
    if pending.is_none() {
        return;
    }

    let mut collected: Vec<(Handle<Mesh>, Transform)> = Vec::new();
    for (mesh3d, transform) in pnp.iter() {
        collected.push((mesh3d.0.clone(), transform.compute_transform()));
    }

    if collected.is_empty() {
        return;
    }

    commands.remove_resource::<PendingScene>();

    let mut gaussians: Vec<Gaussian3d> = Vec::new();
    for (mh, transform) in collected {
        if let Some(mesh) = meshes.get(&mh) {
            gaussians.extend(convert_mesh_to_gaussians(mesh, transform));
        }
    }

    if gaussians.is_empty() {
        warn!("mesh_to_cloud: No gaussians produced");
        return;
    }

    let cloud: PlanarGaussian3d = PlanarGaussian3d::from(gaussians);

    let handle = planar_gaussians.add(cloud);
    commands.spawn((
        PlanarGaussian3dHandle(handle),
        CloudSettings {
            rasterize_mode: RasterizeMode::Color,
            global_scale: 1.0,
            ..default()
        },
        Transform::default(),
        Visibility::Visible,
    ));

    info!("mesh_to_cloud: spawned Gaussian cloud from {}", GLB_PATH);
}

// Convert a Mesh into Gaussian3d instances for vertices, edges, and faces
fn convert_mesh_to_gaussians(mesh: &Mesh, transform: Transform) -> Vec<Gaussian3d> {
    let topology = mesh.primitive_topology();
    let positions = match read_positions(mesh) {
        Some(v) => v,
        None => {
            warn!("mesh_to_cloud: mesh missing positions");
            return Vec::new();
        }
    };
    let normals_opt = read_normals(mesh);

    // Build index buffer as u32
    let indices_u32: Option<Vec<u32>> = match mesh.indices() {
        Some(Indices::U32(ix)) => Some(ix.clone()),
        Some(Indices::U16(ix)) => Some(ix.iter().map(|&x| x as u32).collect()),
        None => None,
    };

    // Vertex normals: either from attribute or computed from faces
    let vertex_normals = normals_opt.unwrap_or_else(|| compute_vertex_normals(topology, &positions, indices_u32.as_ref()));

    let mut out: Vec<Gaussian3d> = Vec::new();

    // 1) Vertices
    for (vpos, vnorm) in positions.iter().zip(vertex_normals.iter()) {
        let pos = transform.transform_point(*vpos);
        let rot = Quat::IDENTITY;
        let scale = Vec3::splat(DEFAULT_SCALE);
        out.push(gaussian_from_transform(pos, rot, scale, *vnorm, DEFAULT_OPACITY));
    }

    // For edges and faces we need indices and triangles
    if let Some(indices) = indices_u32 {
        // 2) Faces: assumes triangle topology
        let tri_iter = triangles_from(topology, &indices);
        let tris: Vec<[u32; 3]> = tri_iter.collect();
        for tri in &tris {
            let p0 = positions[tri[0] as usize];
            let p1 = positions[tri[1] as usize];
            let p2 = positions[tri[2] as usize];

            let centroid = (p0 + p1 + p2) / 3.0;

            let u = p1 - p0;
            let v = p2 - p0;

            let x_axis = u.normalize_or_zero();
            let z_axis = u.cross(v).normalize_or_zero();
            let y_axis = z_axis.cross(x_axis);

            let rot = Quat::from_mat3(&Mat3::from_cols(x_axis, y_axis, z_axis));

            let u_len = u.length();
            let v_on_y = v.dot(y_axis).abs();

            let scale = Vec3::new(u_len, v_on_y, FACE_SCALE);
            let face_n = z_axis;

            out.push(gaussian_from_transform(
                transform.transform_point(centroid),
                rot,
                scale,
                face_n,
                DEFAULT_OPACITY,
            ));
        }

        // 3) Edges: dedupe undirected
        let mut set: HashSet<(u32, u32)> = HashSet::new();
        for tri in &tris {
            let e = [
                (tri[0], tri[1]),
                (tri[1], tri[2]),
                (tri[2], tri[0]),
            ];
            for (a, b) in e {
                let (lo, hi) = if a < b { (a, b) } else { (b, a) };
                if set.insert((lo, hi)) {
                    let pa = positions[lo as usize];
                    let pb = positions[hi as usize];
                    let mid = (pa + pb) * 0.5;
                    let na = vertex_normals[lo as usize];
                    let nb = vertex_normals[hi as usize];
                    let n = (na + nb).normalize_or_zero();

                    let edge_vec = pb - pa;
                    let rot = Quat::from_rotation_arc(Vec3::X, edge_vec.normalize_or_zero());
                    let scale = Vec3::new(edge_vec.length(), EDGE_SCALE, EDGE_SCALE);

                    out.push(gaussian_from_transform(
                        transform.transform_point(mid),
                        rot,
                        scale,
                        n,
                        DEFAULT_OPACITY,
                    ));
                }
            }
        }
    } else {
        // No indices; treat as point cloud of vertices only
        debug!("mesh_to_cloud: mesh had no indices; produced only vertex splats");
    }

    out
}

fn triangles_from(topology: PrimitiveTopology, indices: &[u32]) -> impl Iterator<Item = [u32; 3]> + '_ {
    match topology {
        PrimitiveTopology::TriangleList => Box::new(indices.chunks_exact(3).map(|c| [c[0], c[1], c[2]])) as Box<dyn Iterator<Item = [u32; 3]> + '_>,
        _ => {
            warn!("mesh_to_cloud: non-triangle topology {:?} not fully supported; attempting naive 3-chunking", topology);
            Box::new(indices.chunks(3).filter(|c| c.len() == 3).map(|c| [c[0], c[1], c[2]]))
        }
    }
}

// --- Mesh attribute readers ---

fn read_positions(mesh: &Mesh) -> Option<Vec<Vec3>> {
    // Bevy standard attribute
    let attr = Mesh::ATTRIBUTE_POSITION;
    mesh.attribute(attr).and_then(|a| {
        // Convert any supported format to f32 Vec3
        match a {
            VertexAttributeValues::Float32x3(v) => {
                Some(v.iter().map(|p| Vec3::from_slice(p)).collect())
            }
            VertexAttributeValues::Float32x2(v) => {
                Some(v.iter().map(|p| Vec3::new(p[0], p[1], 0.0)).collect())
            }
            VertexAttributeValues::Float32x4(v) => {
                Some(v.iter().map(|p| Vec3::new(p[0], p[1], p[2])).collect())
            }
            VertexAttributeValues::Uint32x3(v) => {
                Some(v.iter().map(|p| Vec3::new(p[0] as f32, p[1] as f32, p[2] as f32)).collect())
            }
            _ => None,
        }
    })
}

fn read_normals(mesh: &Mesh) -> Option<Vec<Vec3>> {
    let attr = Mesh::ATTRIBUTE_NORMAL;
    mesh.attribute(attr).and_then(|a| {
        match a {
            VertexAttributeValues::Float32x3(v) => {
                Some(v.iter().map(|p| Vec3::from_slice(p)).collect())
            }
            VertexAttributeValues::Float32x4(v) => {
                Some(v.iter().map(|p| Vec3::new(p[0], p[1], p[2])).collect())
            }
            VertexAttributeValues::Uint32x3(v) => {
                Some(v.iter().map(|p| Vec3::new(p[0] as f32, p[1] as f32, p[2] as f32)).collect())
            }
            _ => None,
        }
    })
}

// Compute per-vertex normals if missing
fn compute_vertex_normals(topology: PrimitiveTopology, positions: &[Vec3], indices: Option<&Vec<u32>>) -> Vec<Vec3> {
    let mut normals = vec![Vec3::ZERO; positions.len()];

    if let Some(ix) = indices {
        for tri in triangles_from(topology, ix) {
            let p0 = positions[tri[0] as usize];
            let p1 = positions[tri[1] as usize];
            let p2 = positions[tri[2] as usize];
            let n = face_normal(p0, p1, p2);
            normals[tri[0] as usize] += n;
            normals[tri[1] as usize] += n;
            normals[tri[2] as usize] += n;
        }
    }

    for n in &mut normals {
        *n = n.normalize_or_zero();
    }
    normals
}

fn face_normal(p0: Vec3, p1: Vec3, p2: Vec3) -> Vec3 {
    (p1 - p0).cross(p2 - p0).normalize_or_zero()
}

fn normal_to_rgb(n: Vec3) -> [f32; 3] {
    let c = (n * 0.5) + Vec3::splat(0.5);
    [c.x, c.y, c.z]
}

// Construct a Gaussian3d from a transform, a normal for color, and an opacity.
fn gaussian_from_transform(
    pos: Vec3,
    rot: Quat,
    scale: Vec3,
    norm: Vec3,
    opacity: f32,
) -> Gaussian3d {
    let mut g = Gaussian3d::default();
    // position + visibility
    g.position_visibility.position = pos.to_array();
    g.position_visibility.visibility = 1.0;

    // rotation
    g.rotation.rotation = rot.to_array();

    // scale and opacity
    g.scale_opacity.scale = scale.to_array();
    g.scale_opacity.opacity = opacity;

    // Color via SH DC coefficients (first 3 channels as RGB)
    let rgb = normal_to_rgb(norm);
    g.spherical_harmonic.set(0, rgb[0]);
    g.spherical_harmonic.set(1, rgb[1]);
    g.spherical_harmonic.set(2, rgb[2]);
    // zero the rest for determinism
    for i in 3..bevy_gaussian_splatting::material::spherical_harmonics::SH_COEFF_COUNT {
        g.spherical_harmonic.set(i, 0.0);
    }

    g
}

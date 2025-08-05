// Converts a mesh (monkey.glb) into a Gaussian cloud on CPU: one splat per vertex, edge, and face,
// with color derived from the primitive normal.
//
// Run: cargo run --example mesh_to_cloud --features="viewer io_ply planar buffer_storage bevy/bevy_ui bevy/bevy_scene"
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
    GaussianCamera,
    GaussianSplattingPlugin,
    PlanarGaussian3d,
    PlanarGaussian3dHandle,
};

const GLB_PATH: &str = "scenes/monkey.glb";

// Tunables for splat appearance
const DEFAULT_OPACITY: f32 = 0.8;
const DEFAULT_SCALE: f32 = 0.01; // Small vertices
const EDGE_SCALE: f32 = 0.005;   // Thin edges
const FACE_SCALE: f32 = 0.002;   // Very flat faces

// Entry
fn main() {
    App::new()
        .add_plugins(DefaultPlugins)
        .add_plugins(GaussianSplattingPlugin)
        .add_systems(Startup, (spawn_camera_and_light, load_monkey))
        .add_systems(Update, (try_convert_loaded_mesh, camera_controls))
        .run();
}

fn spawn_camera_and_light(mut commands: Commands) {
    commands.spawn((
        GaussianCamera {
            warmup: true,
        },
        Camera3d::default(),
        Transform::from_translation(Vec3::new(0.0, 1.0, 8.0)).looking_at(Vec3::ZERO, Vec3::Y),
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
    pnp: Query<(Entity, &Mesh3d, &GlobalTransform)>,
    meshes: Res<Assets<Mesh>>,
    mut planar_gaussians: ResMut<Assets<PlanarGaussian3d>>,
) {
    if pending.is_none() {
        return;
    }

    let mut collected: Vec<(Handle<Mesh>, Transform)> = Vec::new();
    let mut mesh_entities: Vec<Entity> = Vec::new();
    for (entity, mesh3d, transform) in pnp.iter() {
        info!("Found mesh entity with Mesh3d component");
        collected.push((mesh3d.0.clone(), transform.compute_transform()));
        mesh_entities.push(entity);
    }

    if collected.is_empty() {
        info!("No meshes found with Mesh3d component, waiting...");
        return;
    }

    info!("Converting {} mesh(es) to Gaussian cloud", collected.len());

    commands.remove_resource::<PendingScene>();

    // Hide original mesh entities now that we know the Gaussians work
    for entity in mesh_entities {
        commands.entity(entity).insert(Visibility::Hidden);
    }

    let mut all_vertices: Vec<Gaussian3d> = Vec::new();
    let mut all_edges: Vec<Gaussian3d> = Vec::new();
    let mut all_faces: Vec<Gaussian3d> = Vec::new();
    let mesh_count = collected.len();
    
    for (mh, transform) in collected {
        if let Some(mesh) = meshes.get(&mh) {
            info!("Converting mesh with {} vertices", mesh.attribute(Mesh::ATTRIBUTE_POSITION).map(|attr| attr.len()).unwrap_or(0));
            let (vertices, edges, faces) = convert_mesh_to_gaussians_separated(mesh, transform);
            info!("Generated {} vertices, {} edges, {} faces", vertices.len(), edges.len(), faces.len());
            all_vertices.extend(vertices);
            all_edges.extend(edges);
            all_faces.extend(faces);
        } else {
            warn!("Mesh handle not found in assets!");
        }
    }

    if all_vertices.is_empty() && all_edges.is_empty() && all_faces.is_empty() {
        warn!("mesh_to_cloud: No gaussians produced from {} meshes", mesh_count);
        return;
    }

    info!("Total gaussians: {} vertices, {} edges, {} faces", all_vertices.len(), all_edges.len(), all_faces.len());

    // Spawn three separate clouds positioned side by side
    let spacing = 3.0;
    
    // Vertices cloud (left)
    if !all_vertices.is_empty() {
        let vertices_cloud = PlanarGaussian3d::from(all_vertices);
        let vertices_handle = planar_gaussians.add(vertices_cloud);
        let vertices_entity = commands.spawn((
            PlanarGaussian3dHandle(vertices_handle),
            CloudSettings {
                aabb: true,
                ..default()
            },
            Transform::from_xyz(-spacing, 0.0, 0.0),
        )).id();
        info!("Spawned vertices cloud entity {:?}", vertices_entity);
    }

    // Edges cloud (center)
    if !all_edges.is_empty() {
        let edges_cloud = PlanarGaussian3d::from(all_edges);
        let edges_handle = planar_gaussians.add(edges_cloud);
        let edges_entity = commands.spawn((
            PlanarGaussian3dHandle(edges_handle),
            CloudSettings {
                aabb: true,
                ..default()
            },
            Transform::from_xyz(0.0, 0.0, 0.0),
        )).id();
        info!("Spawned edges cloud entity {:?}", edges_entity);
    }

    // Faces cloud (right)
    if !all_faces.is_empty() {
        let faces_cloud = PlanarGaussian3d::from(all_faces);
        let faces_handle = planar_gaussians.add(faces_cloud);
        let faces_entity = commands.spawn((
            PlanarGaussian3dHandle(faces_handle),
            CloudSettings {
                aabb: true,
                ..default()
            },
            Transform::from_xyz(spacing, 0.0, 0.0),
        )).id();
        info!("Spawned faces cloud entity {:?}", faces_entity);
    }
}

// Convert a Mesh into separate Gaussian3d collections for vertices, edges, and faces
fn convert_mesh_to_gaussians_separated(mesh: &Mesh, transform: Transform) -> (Vec<Gaussian3d>, Vec<Gaussian3d>, Vec<Gaussian3d>) {
    info!("Starting mesh conversion...");
    let topology = mesh.primitive_topology();
    info!("Mesh topology: {:?}", topology);
    
    let positions = match read_positions(mesh) {
        Some(v) => {
            info!("Found {} vertex positions", v.len());
            v
        },
        None => {
            warn!("mesh_to_cloud: mesh missing positions");
            return (Vec::new(), Vec::new(), Vec::new());
        }
    };
    let normals_opt = read_normals(mesh);
    info!("Normals available: {}", normals_opt.is_some());

    // Build index buffer as u32
    let indices_u32: Option<Vec<u32>> = match mesh.indices() {
        Some(Indices::U32(ix)) => Some(ix.clone()),
        Some(Indices::U16(ix)) => Some(ix.iter().map(|&x| x as u32).collect()),
        None => None,
    };

    // Vertex normals: either from attribute or computed from faces
    let vertex_normals = normals_opt.unwrap_or_else(|| compute_vertex_normals(topology, &positions, indices_u32.as_ref()));

    let mut vertices: Vec<Gaussian3d> = Vec::new();
    let mut edges: Vec<Gaussian3d> = Vec::new();
    let mut faces: Vec<Gaussian3d> = Vec::new();

    // 1) Vertices - small isotropic splats
    for (vpos, vnorm) in positions.iter().zip(vertex_normals.iter()) {
        let pos = transform.transform_point(*vpos);
        let rot = Quat::IDENTITY; // No special rotation needed for isotropic vertex splats
        let scale = Vec3::splat(DEFAULT_SCALE);
        vertices.push(gaussian_from_transform(pos, rot, scale, *vnorm, DEFAULT_OPACITY));
    }

    // For edges and faces we need indices and triangles
    if let Some(indices) = indices_u32 {
        // 2) Faces - flat splats oriented in the triangle plane
        let tri_iter = triangles_from(topology, &indices);
        let tris: Vec<[u32; 3]> = tri_iter.collect();
        for tri in &tris {
            let p0 = positions[tri[0] as usize];
            let p1 = positions[tri[1] as usize];
            let p2 = positions[tri[2] as usize];

            let centroid = (p0 + p1 + p2) / 3.0;

            // Calculate face normal and create local coordinate system
            let edge1 = p1 - p0;
            let edge2 = p2 - p0;
            let face_normal = edge1.cross(edge2).normalize_or_zero();

            // Create a rotation that aligns the Z-axis with the face normal
            // This makes the XY plane of the splat lie in the triangle plane
            let rot = Quat::from_rotation_arc(Vec3::Z, face_normal);

            // Scale: make it flat in Z direction, and sized to cover the triangle area
            // Shrink by 25% total (15% + 10% additional) for better visual separation
            let edge1_len = edge1.length();
            let edge2_len = edge2.length();
            let scale_factor = 0.55; // 25% smaller total
            let scale = Vec3::new(edge1_len * 0.5 * scale_factor, edge2_len * 0.5 * scale_factor, FACE_SCALE);

            faces.push(gaussian_from_transform(
                transform.transform_point(centroid),
                rot,
                scale,
                face_normal,
                DEFAULT_OPACITY,
            ));
        }

        // 3) Edges - elongated splats along edge direction
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
                    let avg_normal = (na + nb).normalize_or_zero();

                    let edge_vec = pb - pa;
                    let edge_length = edge_vec.length();
                    let edge_dir = edge_vec.normalize_or_zero();

                    // Create rotation that aligns the splat's long axis with edge direction
                    // Since Gaussian splats are longest in their X direction by default,
                    // we want to align X with the edge direction
                    let rot = if edge_dir.length() > 0.001 {
                        Quat::from_rotation_arc(Vec3::X, edge_dir)
                    } else {
                        Quat::IDENTITY
                    };
                    
                    // Scale: long along edge (X), thin in other directions
                    // Reduce edge length by 5x for better proportions
                    let scale = Vec3::new(edge_length * 0.14, EDGE_SCALE, EDGE_SCALE);

                    edges.push(gaussian_from_transform(
                        transform.transform_point(mid),
                        rot,
                        scale,
                        avg_normal,
                        DEFAULT_OPACITY,
                    ));
                }
            }
        }
    } else {
        // No indices; treat as point cloud of vertices only
        debug!("mesh_to_cloud: mesh had no indices; produced only vertex splats");
    }

    info!("Conversion complete: {} vertices, {} edges, {} faces", vertices.len(), edges.len(), faces.len());
    (vertices, edges, faces)
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
    // Enhanced contrast: map normals to more vibrant colors
    let normalized = n.normalize_or_zero();
    
    // Map from [-1, 1] to [0, 1] with enhanced contrast
    let base = (normalized * 0.5) + Vec3::splat(0.5);
    
    // Apply stronger contrast enhancement: make colors much more saturated
    let contrast_factor = 100.0; // Increased from 2.2 to 3.0 for maximum contrast
    let enhanced = ((base - Vec3::splat(0.5)) * contrast_factor) + Vec3::splat(0.5);
    
    // Clamp to valid range
    let clamped = enhanced.clamp(Vec3::ZERO, Vec3::ONE);
    
    [clamped.x, clamped.y, clamped.z]
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

    // rotation - use the rotation as provided (each caller handles orientation appropriately)
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

// Camera controls: orbit around origin with arrow keys
fn camera_controls(
    mut camera_query: Query<&mut Transform, With<GaussianCamera>>,
    input: Res<ButtonInput<KeyCode>>,
    time: Res<Time>,
) {
    if let Ok(mut camera_transform) = camera_query.get_single_mut() {
        let rotation_speed = 1.5; // radians per second
        let distance = camera_transform.translation.length();
        
        // Current spherical coordinates (relative to origin)
        let current_pos = camera_transform.translation;
        let mut azimuth = current_pos.z.atan2(current_pos.x); // angle around Y axis
        let mut elevation = (current_pos.y / distance).asin(); // angle up from XZ plane
        
        // Adjust angles based on input
        if input.pressed(KeyCode::ArrowLeft) {
            azimuth += rotation_speed * time.delta_secs();
        }
        if input.pressed(KeyCode::ArrowRight) {
            azimuth -= rotation_speed * time.delta_secs();
        }
        if input.pressed(KeyCode::ArrowUp) {
            elevation += rotation_speed * time.delta_secs();
        }
        if input.pressed(KeyCode::ArrowDown) {
            elevation -= rotation_speed * time.delta_secs();
        }
        
        // Clamp elevation to avoid flipping
        elevation = elevation.clamp(-std::f32::consts::FRAC_PI_2 + 0.1, std::f32::consts::FRAC_PI_2 - 0.1);
        
        // Convert back to cartesian coordinates
        let new_pos = Vec3::new(
            distance * elevation.cos() * azimuth.cos(),
            distance * elevation.sin(),
            distance * elevation.cos() * azimuth.sin(),
        );
        
        // Update camera position and make it look at origin
        camera_transform.translation = new_pos;
        camera_transform.look_at(Vec3::ZERO, Vec3::Y);
    }
}
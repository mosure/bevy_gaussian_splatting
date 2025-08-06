// Converts a mesh (monkey.glb) into a Gaussian cloud on CPU: one splat per vertex, edge, and face,
// with color derived from the primitive normal.
//
// Run: cargo run --example mesh_to_cloud --features="viewer io_ply planar buffer_storage bevy/bevy_ui bevy/bevy_scene"
// Ensure assets/scenes/monkey.glb exists under bevy_gaussian_splatting/assets.

use std::collections::HashSet;
use bevy::prelude::*;
use bevy::math::Mat3;
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
use bevy::ui::Val::*;

const GLB_PATH: &str = "scenes/monkey.glb";

// Tunables for splat appearance
const DEFAULT_OPACITY: f32 = 0.8;
const DEFAULT_SCALE: f32 = 0.01; // Small vertices
const EDGE_SCALE: f32 = 0.025;   // Thin edges (X/Y for edges)
const FACE_SCALE: f32 = 0.02;    // Very flat faces (Z for faces)

// Entry
fn main() {
    App::new()
        .add_plugins(DefaultPlugins)
        .add_plugins(GaussianSplattingPlugin)
        .init_resource::<CloudVisibility>()
        .add_systems(Startup, (spawn_camera_and_light, setup_ui, load_monkey))
        .add_systems(Update, (try_convert_loaded_mesh, camera_controls, visibility_controls, update_info_text))
        .run();
}

fn spawn_camera_and_light(mut commands: Commands) {
    // UI camera for the overlay text
    commands.spawn(Camera2d);
    
    // 3D camera for Gaussian rendering
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

fn setup_ui(mut commands: Commands) {
    // Simple text without background
    commands.spawn((
        Text::new("Loading..."),
        TextFont {
            font_size: 24.0,
            ..default()
        },
        TextColor(Color::srgb(1.0, 1.0, 1.0)),
        Node {
            position_type: PositionType::Absolute,
            top: Px(10.0),
            left: Px(10.0),
            ..default()
        },
        InfoText,
    ));
}

#[derive(Resource, Default)]
struct PendingScene(Handle<Scene>);

#[derive(Resource)]
struct CloudVisibility {
    vertices: bool,
    edges: bool, 
    faces: bool,
}

impl Default for CloudVisibility {
    fn default() -> Self {
        Self {
            vertices: true,
            edges: false,
            faces: false,
        }
    }
}

#[derive(Component)]
struct VerticesCloud;

#[derive(Component)]
struct EdgesCloud;

#[derive(Component)]
struct FacesCloud;

#[derive(Component)]
struct OriginalMesh;

#[derive(Component)]
struct InfoText;

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

    // Hide original mesh entities initially
    for entity in mesh_entities {
        commands.entity(entity).insert((Visibility::Hidden, OriginalMesh));
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

    // Spawn three clouds stacked at the same position
    
    // Vertices cloud
    if !all_vertices.is_empty() {
        let vertices_cloud = PlanarGaussian3d::from(all_vertices);
        let vertices_handle = planar_gaussians.add(vertices_cloud);
        let vertices_entity = commands.spawn((
            PlanarGaussian3dHandle(vertices_handle),
            CloudSettings {
                aabb: true,
                ..default()
            },
            Transform::from_xyz(0.0, 0.0, 0.0),
            VerticesCloud,
            Visibility::Visible,
        )).id();
        info!("Spawned vertices cloud entity {:?}", vertices_entity);
    }

    // Edges cloud
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
            EdgesCloud,
            Visibility::Hidden,
        )).id();
        info!("Spawned edges cloud entity {:?}", edges_entity);
    }

    // Faces cloud
    if !all_faces.is_empty() {
        let faces_cloud = PlanarGaussian3d::from(all_faces);
        let faces_handle = planar_gaussians.add(faces_cloud);
        let faces_entity = commands.spawn((
            PlanarGaussian3dHandle(faces_handle),
            CloudSettings {
                aabb: true,
                ..default()
            },
            Transform::from_xyz(0.0, 0.0, 0.0),
            FacesCloud,
            Visibility::Hidden,
        )).id();
        info!("Spawned faces cloud entity {:?}", faces_entity);
    }
}

fn convert_mesh_to_gaussians_separated(
    mesh: &Mesh,
    transform: Transform,
) -> (Vec<Gaussian3d>, Vec<Gaussian3d>, Vec<Gaussian3d>) {
    let topology = mesh.primitive_topology();

    // ─────────── vertex positions & normals ───────────
    let positions = match read_positions(mesh) {
        Some(p) => p,
        None    => return (Vec::new(), Vec::new(), Vec::new()),
    };
    let indices_u32: Option<Vec<u32>> = match mesh.indices() {
        Some(Indices::U32(ix)) => Some(ix.clone()),
        Some(Indices::U16(ix)) => Some(ix.iter().map(|&x| x as u32).collect()),
        None                   => None,
    };
    let vertex_normals = read_normals(mesh)
        .unwrap_or_else(|| compute_vertex_normals(topology, &positions, indices_u32.as_ref()));

    // Precompute world positions for de-duplication and placement
    let world_positions: Vec<Vec3> = positions.iter().map(|p| transform.transform_point(*p)).collect();

    // ─────────── output storages ───────────
    let mut verts  = Vec::new();
    let mut edges  = Vec::new();
    let mut faces  = Vec::new();

    // ─────────── 1) vertices (isotropic) ───────────
    for (p_world, n_local) in world_positions.iter().zip(vertex_normals.iter()) {
        verts.push(gaussian_from_transform(
            *p_world,                                  // world centre
            Quat::IDENTITY,                            // isotropic
            Vec3::splat(DEFAULT_SCALE),
            (transform.rotation * *n_local).normalize(), // world normal
            DEFAULT_OPACITY,
        ));
    }

    // Stop here if the mesh has no indices
    let indices = if let Some(ix) = indices_u32 { ix } else { return (verts, edges, faces) };
    let tris: Vec<[u32; 3]> = triangles_from(topology, &indices).collect();

    // ─────────── 2) faces (Z = normal = thin axis) ───────────
    for tri in &tris {
        // local-space corners
        let (p0, p1, p2) = (positions[tri[0] as usize], positions[tri[1] as usize], positions[tri[2] as usize]);
        let edge1 = p1 - p0;
        let edge2 = p2 - p0;

        let normal_l_raw = edge1.cross(edge2);
        if normal_l_raw.length_squared() < 1e-8 { continue; }

        // Build an ONB with Z along the face normal (so scale.z controls thinness)
        let z_axis_l = normal_l_raw.normalize(); // thin axis = Z
        // Choose X within the plane; start from edge1 projected to plane
        let mut x_axis_l = edge1 - edge1.dot(z_axis_l) * z_axis_l;
        if x_axis_l.length_squared() < 1e-12 {
            // edge1 is nearly parallel to normal? use edge2
            x_axis_l = edge2 - edge2.dot(z_axis_l) * z_axis_l;
        }
        let x_axis_l = x_axis_l.normalize();
        let y_axis_l = z_axis_l.cross(x_axis_l).normalize(); // X×Y=Z ⇒ Y = Z×X

        let rot_local = Quat::from_mat3(&Mat3::from_cols(x_axis_l, y_axis_l, z_axis_l));
        let rot_world   = transform.rotation * rot_local;
        let normal_world = (transform.rotation * z_axis_l).normalize();

        // Scale: coverage in-plane on X/Y, very thin along Z
        let edge1_len = edge1.length();
        let edge2_len = edge2.length();
        let avg_len   = 0.5 * (edge1_len + edge2_len);
        let scale     = Vec3::new(avg_len * 0.3, avg_len * 0.3, FACE_SCALE);

        let centroid_world = (world_positions[tri[0] as usize]
                            + world_positions[tri[1] as usize]
                            + world_positions[tri[2] as usize]) / 3.0;

        faces.push(gaussian_from_transform(
            centroid_world,
            rot_world,
            scale,
            normal_world,
            1.0,
        ));
    }

    // ─────────── 3) edges (Z = edge direction = long axis), geometric de-dup ───────────

    // Quantize world positions so indices split at seams still map to the same geometric vertex
    fn qkey(p: Vec3, eps: f32) -> (i32, i32, i32) {
        let inv = 1.0 / eps;
        (
            (p.x * inv).round() as i32,
            (p.y * inv).round() as i32,
            (p.z * inv).round() as i32,
        )
    }
    let eps = 1e-5;
    let canon: Vec<(i32,i32,i32)> = world_positions.iter().map(|&p| qkey(p, eps)).collect();
    let mut seen_geo = HashSet::<((i32,i32,i32),(i32,i32,i32))>::new();

    for tri in &tris {
        for &(a, b) in &[(tri[0],tri[1]), (tri[1],tri[2]), (tri[2],tri[0])] {
            // geometric edge key (not raw index), insensitive to seam splits
            let key_a = canon[a as usize];
            let key_b = canon[b as usize];
            let edge_key = if key_a <= key_b { (key_a, key_b) } else { (key_b, key_a) };
            if !seen_geo.insert(edge_key) { continue; }

            // Build edge frame from local geometry, then promote to world
            let pa_l = positions[a as usize];
            let pb_l = positions[b as usize];
            let edge_vec_l = pb_l - pa_l;
            let len_l = edge_vec_l.length();
            if len_l < 1e-4 { continue; }

            let z_axis_l = edge_vec_l / len_l; // long axis = Z

            // Use average vertex normal to stabilize twist; project to plane ⟂ Z
            let n_avg_l  = (vertex_normals[a as usize] + vertex_normals[b as usize]).normalize_or_zero();
            let mut x_axis_l = n_avg_l - n_avg_l.dot(z_axis_l) * z_axis_l;
            if x_axis_l.length_squared() < 1e-8 {
                // fallback to a world axis most orthogonal to Z
                let candidate = if z_axis_l.dot(Vec3::X).abs() < 0.9 { Vec3::X }
                                else if z_axis_l.dot(Vec3::Y).abs() < 0.9 { Vec3::Y }
                                else { Vec3::Z };
                x_axis_l = candidate - candidate.dot(z_axis_l) * z_axis_l;
            }
            let x_axis_l = x_axis_l.normalize();
            let y_axis_l = z_axis_l.cross(x_axis_l).normalize(); // X×Y=Z ⇒ Y = Z×X

            let rot_local = Quat::from_mat3(&Mat3::from_cols(x_axis_l, y_axis_l, z_axis_l));
            let rot_world = transform.rotation * rot_local;
            let normal_w  = (transform.rotation * n_avg_l).normalize();

            // Long along Z, thin on X/Y
            let scale = Vec3::new(EDGE_SCALE, EDGE_SCALE, len_l * 0.08);

            let mid_world = (world_positions[a as usize] + world_positions[b as usize]) * 0.5;

            edges.push(gaussian_from_transform(
                mid_world,
                rot_world,
                scale,
                normal_w,
                DEFAULT_OPACITY,
            ));
        }
    }

    (verts, edges, faces)
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
    let contrast_factor = 100.0;
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
    let a = rot.to_array();                 // [x, y, z, w]
    g.rotation.rotation = [a[3], a[0], a[1], a[2]]; // -> [w, x, y, z]


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
    if let Ok(mut camera_transform) = camera_query.single_mut() {
        let rotation_speed = 1.5; // radians per second
        let zoom_speed = 5.0; // units per second
        let mut distance = camera_transform.translation.length();
        
        // Current spherical coordinates (relative to origin)
        let current_pos = camera_transform.translation;
        let mut azimuth = current_pos.z.atan2(current_pos.x); // angle around Y axis
        let mut elevation = (current_pos.y / distance).asin(); // angle up from XZ plane
        
        // Adjust angles based on input
        if input.pressed(KeyCode::KeyD) {
            azimuth += rotation_speed * time.delta_secs();
        }
        if input.pressed(KeyCode::KeyA) {
            azimuth -= rotation_speed * time.delta_secs();
        }
        if input.pressed(KeyCode::KeyW) {
            elevation += rotation_speed * time.delta_secs();
        }
        if input.pressed(KeyCode::KeyS) {
            elevation -= rotation_speed * time.delta_secs();
        }
        
        // Zoom controls
        if input.pressed(KeyCode::KeyE) || input.pressed(KeyCode::NumpadAdd) {
            distance -= zoom_speed * time.delta_secs();
        }
        if input.pressed(KeyCode::KeyQ) || input.pressed(KeyCode::NumpadSubtract) {
            distance += zoom_speed * time.delta_secs();
        }
        
        // Clamp elevation to avoid flipping and distance to reasonable bounds
        elevation = elevation.clamp(-std::f32::consts::FRAC_PI_2 + 0.1, std::f32::consts::FRAC_PI_2 - 0.1);
        distance = distance.clamp(1.0, 50.0);
        
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

// Visibility controls: toggle cloud types with keys 1, 2, 3
fn visibility_controls(
    mut cloud_visibility: ResMut<CloudVisibility>,
    mut vertices_query: Query<&mut Visibility, (With<VerticesCloud>, Without<EdgesCloud>, Without<FacesCloud>, Without<OriginalMesh>)>,
    mut edges_query: Query<&mut Visibility, (With<EdgesCloud>, Without<VerticesCloud>, Without<FacesCloud>, Without<OriginalMesh>)>,
    mut faces_query: Query<&mut Visibility, (With<FacesCloud>, Without<VerticesCloud>, Without<EdgesCloud>, Without<OriginalMesh>)>,
    mut original_query: Query<&mut Visibility, (With<OriginalMesh>, Without<VerticesCloud>, Without<EdgesCloud>, Without<FacesCloud>)>,
    input: Res<ButtonInput<KeyCode>>,
) {
    let mut changed = false;
    
    if input.just_pressed(KeyCode::Digit1) {
        cloud_visibility.vertices = !cloud_visibility.vertices;
        changed = true;
    }
    if input.just_pressed(KeyCode::Digit2) {
        cloud_visibility.edges = !cloud_visibility.edges;
        changed = true;
    }
    if input.just_pressed(KeyCode::Digit3) {
        cloud_visibility.faces = !cloud_visibility.faces;
        changed = true;
    }
    
    if changed {
        // Update cloud visibilities
        if let Ok(mut vis) = vertices_query.single_mut() {
            *vis = if cloud_visibility.vertices { Visibility::Visible } else { Visibility::Hidden };
        }
        if let Ok(mut vis) = edges_query.single_mut() {
            *vis = if cloud_visibility.edges { Visibility::Visible } else { Visibility::Hidden };
        }
        if let Ok(mut vis) = faces_query.single_mut() {
            *vis = if cloud_visibility.faces { Visibility::Visible } else { Visibility::Hidden };
        }
        
        // Show original mesh if no clouds are visible
        let show_original = !cloud_visibility.vertices && !cloud_visibility.edges && !cloud_visibility.faces;
        for mut vis in original_query.iter_mut() {
            *vis = if show_original { Visibility::Visible } else { Visibility::Hidden };
        }
        
        info!("Visibility - Vertices: {}, Edges: {}, Faces: {}, Original: {}", 
              cloud_visibility.vertices, cloud_visibility.edges, cloud_visibility.faces, show_original);
    }
}

// Update the info text with current controls and visibility state
fn update_info_text(
    mut text_query: Query<&mut Text, With<InfoText>>,
    cloud_visibility: Res<CloudVisibility>,
) {
    if let Ok(mut text) = text_query.single_mut() {
        // Simple approach: use basic text with visual indicators
        let v_indicator = if cloud_visibility.vertices { "[ON]" } else { "[OFF]" };
        let e_indicator = if cloud_visibility.edges { "[ON]" } else { "[OFF]" };
        let f_indicator = if cloud_visibility.faces { "[ON]" } else { "[OFF]" };
        
        **text = format!(
            "Controls:\n\
            WASD: Rotate camera\n\
            QE: Zoom in/out\n\
            1: Toggle vertices {}\n\
            2: Toggle edges {}\n\
            3: Toggle faces {}"
            , v_indicator, e_indicator, f_indicator
        );
    }
}

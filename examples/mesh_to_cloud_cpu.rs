//! # Mesh to Gaussian Cloud Converter
//!
//! This example demonstrates converting a 3D mesh into Gaussian splats on the CPU,
//! creating separate visualizations for vertices, edges, and faces.
//!
//! ## Features
//!
//! - **Vertex Splats**: Small isotropic splats at each mesh vertex
//! - **Edge Splats**: Elongated splats along each unique geometric edge
//! - **Face Splats**: Flat splats covering each triangle face
//! - **Interactive Controls**: Toggle visibility of each splat type
//! - **Geometric Deduplication**: Handles mesh seams and UV splits correctly
//!
//! ## Controls
//!
//! - `WASD`: Orbit camera around the model
//! - `Q/E`: Zoom in/out
//! - `1`: Toggle vertex splats
//! - `2`: Toggle edge splats  
//! - `3`: Toggle face splats
//!
//! ## Run Command
//!
//! ```bash
//! cargo run --example mesh_to_cloud --features="viewer io_ply planar buffer_storage bevy/bevy_ui bevy/bevy_scene"
//! ```
//!
//! Ensure `assets/scenes/monkey.glb` exists in the bevy_gaussian_splatting assets directory.

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
use bevy::ui::Val::Px;

/// Path to the mesh asset to convert
const MESH_PATH: &str = "scenes/monkey.glb";

/// Configuration for splat appearance and behavior
mod config {
    /// Base opacity for all splats
    pub const DEFAULT_OPACITY: f32 = 0.8;
    /// Scale for isotropic vertex splats
    pub const VERTEX_SCALE: f32 = 0.01;
    /// Thickness of edge splats (in non-elongated directions)
    pub const EDGE_THICKNESS: f32 = 0.025;
    /// Thickness of face splats (in face-normal direction)
    pub const FACE_THICKNESS: f32 = 0.02;
    /// Length scaling factor for edges relative to geometric length
    pub const EDGE_LENGTH_SCALE: f32 = 0.08;
    /// Size scaling factor for faces relative to triangle size
    pub const FACE_SIZE_SCALE: f32 = 0.3;
    /// Geometric deduplication epsilon for floating-point comparison
    pub const GEOMETRIC_EPSILON: f32 = 1e-5;
}

fn main() {
    App::new()
        .add_plugins(DefaultPlugins)
        .add_plugins(GaussianSplattingPlugin)
        .init_resource::<CloudVisibility>()
        .init_resource::<ConversionMetrics>()
        .add_systems(Startup, (setup_scene, setup_ui, load_mesh))
        .add_systems(Update, (
            convert_loaded_mesh,
            camera_controls,
            visibility_controls,
            update_info_text,
        ))
        .run();
}

/// Resource tracking which types of splats are currently visible
#[derive(Resource)]
struct CloudVisibility {
    vertices: bool,
    edges: bool,
    faces: bool,
}

/// Resource tracking conversion performance metrics
#[derive(Resource, Default)]
struct ConversionMetrics {
    conversion_time_ms: f32,
    total_gaussians: usize,
    vertex_count: usize,
    edge_count: usize,
    face_count: usize,
}

impl Default for CloudVisibility {
    fn default() -> Self {
        Self {
            vertices: true,  // Start with vertices visible
            edges: false,
            faces: false,
        }
    }
}

/// Resource holding the scene handle while waiting for it to load
#[derive(Resource, Default)]
struct PendingMeshScene(Handle<Scene>);

/// Marker components for different splat cloud types
#[derive(Component)]
struct VerticesCloud;

#[derive(Component)]
struct EdgesCloud;

#[derive(Component)]
struct FacesCloud;

/// Marker for the original mesh entities (hidden during splat visualization)
#[derive(Component)]
struct OriginalMesh;

/// Marker for the UI info text
#[derive(Component)]
struct InfoText;

/// Set up the 3D scene with camera and lighting
fn setup_scene(mut commands: Commands) {
    // UI camera for the overlay text
    commands.spawn(Camera2d);
    
    // 3D camera for Gaussian rendering - positioned to view the model
    commands.spawn((
        GaussianCamera {
            warmup: true,
        },
        Camera3d::default(),
        Transform::from_translation(Vec3::new(0.0, 1.0, 8.0))
            .looking_at(Vec3::ZERO, Vec3::Y),
    ));

    // Directional light to illuminate the scene
    commands.spawn((
        DirectionalLight::default(),
        Transform::from_xyz(2.0, 4.0, 2.0)
            .looking_at(Vec3::ZERO, Vec3::Y),
    ));
}

/// Set up the UI overlay showing controls and status
fn setup_ui(mut commands: Commands) {
    commands.spawn((
        Text::new("Loading mesh..."),
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

/// Load the mesh asset and prepare for conversion
fn load_mesh(mut commands: Commands, assets: Res<AssetServer>) {
    let scene: Handle<Scene> = assets.load(MESH_PATH.to_string() + "#Scene0");
    commands.insert_resource(PendingMeshScene(scene.clone()));
    commands.spawn((
        SceneRoot(scene),
        Transform::default(),
        Visibility::Visible,
    ));
}

/// System that converts loaded meshes to Gaussian splat clouds
/// 
/// This system waits for the mesh scene to load, then processes each mesh
/// to generate separate splat clouds for vertices, edges, and faces.
fn convert_loaded_mesh(
    mut commands: Commands,
    pending: Option<Res<PendingMeshScene>>,
    mesh_query: Query<(Entity, &Mesh3d, &GlobalTransform)>,
    meshes: Res<Assets<Mesh>>,
    mut planar_gaussians: ResMut<Assets<PlanarGaussian3d>>,
    mut metrics: ResMut<ConversionMetrics>,
) {
    let Some(_pending) = pending else { return };

    // Collect all mesh entities and their data
    let mut mesh_data: Vec<(Handle<Mesh>, Transform)> = Vec::new();
    let mut mesh_entities: Vec<Entity> = Vec::new();
    
    for (entity, mesh3d, global_transform) in mesh_query.iter() {
        info!("Found mesh entity to convert");
        mesh_data.push((mesh3d.0.clone(), global_transform.compute_transform()));
        mesh_entities.push(entity);
    }

    if mesh_data.is_empty() {
        info!("No meshes found yet, waiting for scene to load...");
        return;
    }

    info!("Converting {} mesh(es) to Gaussian splat clouds", mesh_data.len());
    commands.remove_resource::<PendingMeshScene>();

    // Start timing the conversion process
    let start_time = std::time::Instant::now();

    // Hide original mesh entities during splat visualization
    for entity in mesh_entities {
        commands.entity(entity).insert((Visibility::Hidden, OriginalMesh));
    }

    // Process all meshes and collect splats by type
    let mut all_vertices: Vec<Gaussian3d> = Vec::new();
    let mut all_edges: Vec<Gaussian3d> = Vec::new();
    let mut all_faces: Vec<Gaussian3d> = Vec::new();
    
    for (mesh_handle, transform) in mesh_data {
        if let Some(mesh) = meshes.get(&mesh_handle) {
            let vertex_count = mesh
                .attribute(Mesh::ATTRIBUTE_POSITION)
                .map(|attr| attr.len())
                .unwrap_or(0);
            info!("Converting mesh with {} vertices", vertex_count);
            
            let (vertices, edges, faces) = convert_mesh_to_splats(mesh, transform);
            info!("Generated {} vertex, {} edge, {} face splats", 
                  vertices.len(), edges.len(), faces.len());
            
            all_vertices.extend(vertices);
            all_edges.extend(edges);
            all_faces.extend(faces);
        } else {
            warn!("Mesh handle not found in assets - may still be loading");
        }
    }

    if all_vertices.is_empty() && all_edges.is_empty() && all_faces.is_empty() {
        warn!("No Gaussian splats generated from meshes");
        return;
    }

    // Calculate conversion time
    let conversion_time = start_time.elapsed();
    let conversion_time_ms = conversion_time.as_secs_f32() * 1000.0;
    let total_gaussians = all_vertices.len() + all_edges.len() + all_faces.len();

    // Update metrics for UI display
    metrics.conversion_time_ms = conversion_time_ms;
    metrics.total_gaussians = total_gaussians;
    metrics.vertex_count = all_vertices.len();
    metrics.edge_count = all_edges.len();
    metrics.face_count = all_faces.len();

    info!("Converted {} vertices, {} edges, {} faces → {} gaussians in {:.2} ms", 
          all_vertices.len(), all_edges.len(), all_faces.len(), total_gaussians, conversion_time_ms);

    // Create separate splat clouds for each primitive type
    spawn_splat_clouds(&mut commands, &mut planar_gaussians, 
                       all_vertices, all_edges, all_faces);
}

/// Helper function to spawn the three different splat cloud entities
fn spawn_splat_clouds(
    commands: &mut Commands,
    planar_gaussians: &mut ResMut<Assets<PlanarGaussian3d>>,
    vertices: Vec<Gaussian3d>,
    edges: Vec<Gaussian3d>,
    faces: Vec<Gaussian3d>,
) {
    // Vertices cloud - visible by default
    if !vertices.is_empty() {
        let cloud = PlanarGaussian3d::from(vertices);
        let handle = planar_gaussians.add(cloud);
        commands.spawn((
            PlanarGaussian3dHandle(handle),
            CloudSettings { aabb: true, ..default() },
            Transform::IDENTITY,
            VerticesCloud,
            Visibility::Visible,
        ));
        info!("Spawned vertices cloud");
    }

    // Edges cloud - hidden by default
    if !edges.is_empty() {
        let cloud = PlanarGaussian3d::from(edges);
        let handle = planar_gaussians.add(cloud);
        commands.spawn((
            PlanarGaussian3dHandle(handle),
            CloudSettings { aabb: true, ..default() },
            Transform::IDENTITY,
            EdgesCloud,
            Visibility::Hidden,
        ));
        info!("Spawned edges cloud");
    }

    // Faces cloud - hidden by default
    if !faces.is_empty() {
        let cloud = PlanarGaussian3d::from(faces);
        let handle = planar_gaussians.add(cloud);
        commands.spawn((
            PlanarGaussian3dHandle(handle),
            CloudSettings { aabb: true, ..default() },
            Transform::IDENTITY,
            FacesCloud,
            Visibility::Hidden,
        ));
        info!("Spawned faces cloud");
    }
}

/// Convert a single mesh into separate collections of Gaussian splats
/// 
/// Returns (vertices, edges, faces) where each vector contains the splats
/// for that primitive type. Uses geometric deduplication to handle mesh
/// seams and UV splits correctly.
fn convert_mesh_to_splats(
    mesh: &Mesh,
    transform: Transform,
) -> (Vec<Gaussian3d>, Vec<Gaussian3d>, Vec<Gaussian3d>) {
    let topology = mesh.primitive_topology();

    // Extract vertex positions and normals
    let positions = match read_vertex_positions(mesh) {
        Some(positions) => positions,
        None => {
            warn!("Mesh missing vertex positions");
            return (Vec::new(), Vec::new(), Vec::new());
        }
    };
    
    let indices_u32: Option<Vec<u32>> = match mesh.indices() {
        Some(Indices::U32(indices)) => Some(indices.clone()),
        Some(Indices::U16(indices)) => Some(indices.iter().map(|&i| i as u32).collect()),
        None => None,
    };
    
    let vertex_normals = read_vertex_normals(mesh)
        .unwrap_or_else(|| compute_vertex_normals(topology, &positions, indices_u32.as_ref()));

    // Pre-compute world positions for efficient processing
    let world_positions: Vec<Vec3> = positions
        .iter()
        .map(|&pos| transform.transform_point(pos))
        .collect();

    let mut vertex_splats = Vec::new();
    let mut edge_splats = Vec::new(); 
    let mut face_splats = Vec::new();

    // Generate vertex splats - isotropic spheres at each vertex
    for (&world_pos, &local_normal) in world_positions.iter().zip(vertex_normals.iter()) {
        let world_normal = (transform.rotation * local_normal).normalize();
        vertex_splats.push(create_gaussian_splat(
            world_pos,
            Quat::IDENTITY, // Isotropic - no special orientation
            Vec3::splat(config::VERTEX_SCALE),
            world_normal,
            config::DEFAULT_OPACITY,
        ));
    }

    // Process faces and edges if mesh has indices
    let Some(indices) = indices_u32 else { 
        return (vertex_splats, edge_splats, face_splats);
    };
    
    let triangles: Vec<[u32; 3]> = extract_triangles(topology, &indices).collect();

    // Generate face splats - flat ellipsoids covering each triangle
    for &triangle in &triangles {
        let [i0, i1, i2] = triangle;
        let p0 = positions[i0 as usize];
        let p1 = positions[i1 as usize]; 
        let p2 = positions[i2 as usize];

        let edge1 = p1 - p0;
        let edge2 = p2 - p0;
        let face_normal_raw = edge1.cross(edge2);
        
        // Skip degenerate triangles
        if face_normal_raw.length_squared() < 1e-8 {
            continue;
        }

        let face_normal = face_normal_raw.normalize();
        
        // Build coordinate system: Z-axis = face normal (thin direction)
        let z_axis = face_normal;
        let mut x_axis = edge1 - edge1.dot(z_axis) * z_axis; // Project edge1 to face plane
        if x_axis.length_squared() < 1e-12 {
            x_axis = edge2 - edge2.dot(z_axis) * z_axis; // Fallback to edge2
        }
        let x_axis = x_axis.normalize();
        let y_axis = z_axis.cross(x_axis).normalize();

        let rotation = Quat::from_mat3(&Mat3::from_cols(x_axis, y_axis, z_axis));
        let world_rotation = transform.rotation * rotation;
        let world_normal = (transform.rotation * face_normal).normalize();

        // Scale to cover triangle area, thin in face normal direction
        let edge1_length = edge1.length();
        let edge2_length = edge2.length(); 
        let average_size = (edge1_length + edge2_length) * 0.5;
        let scale = Vec3::new(
            average_size * config::FACE_SIZE_SCALE,
            average_size * config::FACE_SIZE_SCALE,
            config::FACE_THICKNESS,
        );

        let centroid = (p0 + p1 + p2) / 3.0;
        let world_centroid = transform.transform_point(centroid);

        face_splats.push(create_gaussian_splat(
            world_centroid,
            world_rotation,
            scale,
            world_normal,
            1.0, // High opacity for face visualization
        ));
    }

    // Generate edge splats with geometric deduplication
    let mut unique_edges = HashSet::new();
    
    for &triangle in &triangles {
        let edges = [
            (triangle[0], triangle[1]),
            (triangle[1], triangle[2]), 
            (triangle[2], triangle[0]),
        ];
        
        for (vertex_a, vertex_b) in edges {
            // Use geometric positions for deduplication (handles UV seams)
            let pos_a = world_positions[vertex_a as usize];
            let pos_b = world_positions[vertex_b as usize];
            let edge_key = create_geometric_edge_key(pos_a, pos_b, config::GEOMETRIC_EPSILON);
            
            if unique_edges.insert(edge_key) {
                let local_pos_a = positions[vertex_a as usize];
                let local_pos_b = positions[vertex_b as usize];
                let edge_vector = local_pos_b - local_pos_a;
                let edge_length = edge_vector.length();
                
                if edge_length < 1e-4 {
                    continue; // Skip degenerate edges
                }

                let edge_direction = edge_vector / edge_length;
                
                // Build coordinate system: Z-axis = edge direction (long axis)
                let z_axis = edge_direction;
                
                // Use average vertex normal for secondary axis, projected perpendicular to edge
                let normal_a = vertex_normals[vertex_a as usize];
                let normal_b = vertex_normals[vertex_b as usize];
                let average_normal = (normal_a + normal_b).normalize_or_zero();
                
                let mut x_axis = average_normal - average_normal.dot(z_axis) * z_axis;
                if x_axis.length_squared() < 1e-8 {
                    // Fallback: use most perpendicular world axis
                    x_axis = find_most_perpendicular_axis(z_axis);
                    x_axis = x_axis - x_axis.dot(z_axis) * z_axis;
                }
                let x_axis = x_axis.normalize();
                let y_axis = z_axis.cross(x_axis).normalize();

                let rotation = Quat::from_mat3(&Mat3::from_cols(x_axis, y_axis, z_axis));
                let world_rotation = transform.rotation * rotation;
                let world_normal = (transform.rotation * average_normal).normalize();

                // Scale: thin in X/Y, long along Z (edge direction)
                let scale = Vec3::new(
                    config::EDGE_THICKNESS,
                    config::EDGE_THICKNESS,
                    edge_length * config::EDGE_LENGTH_SCALE,
                );

                let midpoint = (local_pos_a + local_pos_b) * 0.5;
                let world_midpoint = transform.transform_point(midpoint);

                edge_splats.push(create_gaussian_splat(
                    world_midpoint,
                    world_rotation,
                    scale,
                    world_normal,
                    config::DEFAULT_OPACITY,
                ));
            }
        }
    }

    (vertex_splats, edge_splats, face_splats)
}

/// Extract triangles from mesh indices based on topology
fn extract_triangles(topology: PrimitiveTopology, indices: &[u32]) -> impl Iterator<Item = [u32; 3]> + '_ {
    match topology {
        PrimitiveTopology::TriangleList => {
            Box::new(indices.chunks_exact(3).map(|chunk| [chunk[0], chunk[1], chunk[2]]))
                as Box<dyn Iterator<Item = [u32; 3]> + '_>
        }
        _ => {
            warn!("Non-triangle topology {:?} - attempting triangle extraction", topology);
            Box::new(
                indices
                    .chunks(3)
                    .filter(|chunk| chunk.len() == 3)
                    .map(|chunk| [chunk[0], chunk[1], chunk[2]])
            )
        }
    }
}

/// Create a geometric key for edge deduplication
/// 
/// This quantizes positions to handle floating-point precision issues
/// and orders the vertices consistently regardless of edge direction.
fn create_geometric_edge_key(pos_a: Vec3, pos_b: Vec3, epsilon: f32) -> ((i32, i32, i32), (i32, i32, i32)) {
    fn quantize_position(pos: Vec3, epsilon: f32) -> (i32, i32, i32) {
        let inv_epsilon = 1.0 / epsilon;
        (
            (pos.x * inv_epsilon).round() as i32,
            (pos.y * inv_epsilon).round() as i32,
            (pos.z * inv_epsilon).round() as i32,
        )
    }
    
    let key_a = quantize_position(pos_a, epsilon);
    let key_b = quantize_position(pos_b, epsilon);
    
    // Order consistently for deduplication
    if key_a <= key_b { (key_a, key_b) } else { (key_b, key_a) }
}

/// Find the world axis most perpendicular to the given direction
fn find_most_perpendicular_axis(direction: Vec3) -> Vec3 {
    let dot_x = direction.dot(Vec3::X).abs();
    let dot_y = direction.dot(Vec3::Y).abs();
    let dot_z = direction.dot(Vec3::Z).abs();
    
    if dot_x <= dot_y && dot_x <= dot_z {
        Vec3::X
    } else if dot_y <= dot_z {
        Vec3::Y
    } else {
        Vec3::Z
    }
}

// === Mesh Attribute Reading ===

/// Read vertex positions from mesh attributes
fn read_vertex_positions(mesh: &Mesh) -> Option<Vec<Vec3>> {
    mesh.attribute(Mesh::ATTRIBUTE_POSITION).and_then(|attr| {
        match attr {
            VertexAttributeValues::Float32x3(positions) => {
                Some(positions.iter().map(|&pos| Vec3::from(pos)).collect())
            }
            VertexAttributeValues::Float32x2(positions) => {
                Some(positions.iter().map(|&pos| Vec3::new(pos[0], pos[1], 0.0)).collect())
            }
            VertexAttributeValues::Float32x4(positions) => {
                Some(positions.iter().map(|&pos| Vec3::new(pos[0], pos[1], pos[2])).collect())
            }
            _ => {
                warn!("Unsupported position attribute format");
                None
            }
        }
    })
}

/// Read vertex normals from mesh attributes
fn read_vertex_normals(mesh: &Mesh) -> Option<Vec<Vec3>> {
    mesh.attribute(Mesh::ATTRIBUTE_NORMAL).and_then(|attr| {
        match attr {
            VertexAttributeValues::Float32x3(normals) => {
                Some(normals.iter().map(|&normal| Vec3::from(normal)).collect())
            }
            VertexAttributeValues::Float32x4(normals) => {
                Some(normals.iter().map(|&normal| Vec3::new(normal[0], normal[1], normal[2])).collect())
            }
            _ => {
                warn!("Unsupported normal attribute format");
                None
            }
        }
    })
}

/// Compute vertex normals from face normals when mesh normals are missing
fn compute_vertex_normals(
    topology: PrimitiveTopology,
    positions: &[Vec3],
    indices: Option<&Vec<u32>>,
) -> Vec<Vec3> {
    let mut normals = vec![Vec3::ZERO; positions.len()];

    if let Some(indices) = indices {
        for triangle in extract_triangles(topology, indices) {
            let [i0, i1, i2] = triangle;
            let p0 = positions[i0 as usize];
            let p1 = positions[i1 as usize];
            let p2 = positions[i2 as usize];
            
            let face_normal = compute_face_normal(p0, p1, p2);
            
            // Accumulate face normal at each vertex
            normals[i0 as usize] += face_normal;
            normals[i1 as usize] += face_normal;
            normals[i2 as usize] += face_normal;
        }
    }

    // Normalize accumulated normals
    for normal in &mut normals {
        *normal = normal.normalize_or_zero();
    }

    normals
}

/// Compute the normal vector for a triangle face
fn compute_face_normal(p0: Vec3, p1: Vec3, p2: Vec3) -> Vec3 {
    (p1 - p0).cross(p2 - p0).normalize_or_zero()
}

/// Convert a normal vector to RGB color for visualization
fn normal_to_color(normal: Vec3) -> [f32; 3] {
    let normalized = normal.normalize_or_zero();
    
    // Map from [-1, 1] to [0, 1] range
    let base = (normalized * 0.5) + Vec3::splat(0.5);
    
    // Apply contrast enhancement for better visibility
    let contrast_factor = 2.0;
    let enhanced = ((base - Vec3::splat(0.5)) * contrast_factor) + Vec3::splat(0.5);
    
    // Clamp to valid color range
    let clamped = enhanced.clamp(Vec3::ZERO, Vec3::ONE);
    
    [clamped.x, clamped.y, clamped.z]
}

/// Create a Gaussian splat with the specified properties
/// 
/// Note: Handles the critical quaternion component order conversion
/// from Bevy's [x,y,z,w] format to GPU's expected [w,x,y,z] format.
fn create_gaussian_splat(
    position: Vec3,
    rotation: Quat,
    scale: Vec3,
    normal: Vec3,
    opacity: f32,
) -> Gaussian3d {
    let mut splat = Gaussian3d::default();
    
    // Set position and visibility
    splat.position_visibility.position = position.to_array();
    splat.position_visibility.visibility = 1.0;

    // CRITICAL: Convert quaternion component order from Bevy [x,y,z,w] to GPU [w,x,y,z]
    let components = rotation.to_array(); // [x, y, z, w]
    splat.rotation.rotation = [components[3], components[0], components[1], components[2]]; // [w, x, y, z]

    // Set scale and opacity
    splat.scale_opacity.scale = scale.to_array();
    splat.scale_opacity.opacity = opacity;

    // Set color based on normal vector using spherical harmonics
    let color = normal_to_color(normal);
    splat.spherical_harmonic.set(0, color[0]);
    splat.spherical_harmonic.set(1, color[1]);
    splat.spherical_harmonic.set(2, color[2]);
    
    // Zero out remaining spherical harmonic coefficients
    for i in 3..bevy_gaussian_splatting::material::spherical_harmonics::SH_COEFF_COUNT {
        splat.spherical_harmonic.set(i, 0.0);
    }

    splat
}

// === Interactive Controls ===

/// Camera orbit controls using WASD keys and QE for zoom
fn camera_controls(
    mut camera_query: Query<&mut Transform, With<GaussianCamera>>,
    input: Res<ButtonInput<KeyCode>>,
    time: Res<Time>,
) {
    let Ok(mut camera_transform) = camera_query.single_mut() else { return };
    
    const ROTATION_SPEED: f32 = 1.5; // radians per second
    const ZOOM_SPEED: f32 = 5.0;     // units per second
    
    let mut distance = camera_transform.translation.length();
    
    // Convert current position to spherical coordinates
    let current_pos = camera_transform.translation;
    let mut azimuth = current_pos.z.atan2(current_pos.x);   // rotation around Y-axis
    let mut elevation = (current_pos.y / distance).asin();  // angle from XZ-plane
    
    // Handle rotation input
    if input.pressed(KeyCode::KeyD) {
        azimuth += ROTATION_SPEED * time.delta_secs();
    }
    if input.pressed(KeyCode::KeyA) {
        azimuth -= ROTATION_SPEED * time.delta_secs();
    }
    if input.pressed(KeyCode::KeyW) {
        elevation += ROTATION_SPEED * time.delta_secs();
    }
    if input.pressed(KeyCode::KeyS) {
        elevation -= ROTATION_SPEED * time.delta_secs();
    }
    
    // Handle zoom input
    if input.pressed(KeyCode::KeyE) || input.pressed(KeyCode::NumpadAdd) {
        distance -= ZOOM_SPEED * time.delta_secs();
    }
    if input.pressed(KeyCode::KeyQ) || input.pressed(KeyCode::NumpadSubtract) {
        distance += ZOOM_SPEED * time.delta_secs();
    }
    
    // Clamp values to reasonable bounds
    elevation = elevation.clamp(-std::f32::consts::FRAC_PI_2 + 0.1, std::f32::consts::FRAC_PI_2 - 0.1);
    distance = distance.clamp(1.0, 50.0);
    
    // Convert back to Cartesian coordinates
    let new_position = Vec3::new(
        distance * elevation.cos() * azimuth.cos(),
        distance * elevation.sin(),
        distance * elevation.cos() * azimuth.sin(),
    );
    
    // Update camera transform
    camera_transform.translation = new_position;
    camera_transform.look_at(Vec3::ZERO, Vec3::Y);
}

/// Handle keyboard input for toggling splat cloud visibility
fn visibility_controls(
    mut cloud_visibility: ResMut<CloudVisibility>,
    mut vertices_query: Query<&mut Visibility, (With<VerticesCloud>, Without<EdgesCloud>, Without<FacesCloud>, Without<OriginalMesh>)>,
    mut edges_query: Query<&mut Visibility, (With<EdgesCloud>, Without<VerticesCloud>, Without<FacesCloud>, Without<OriginalMesh>)>,
    mut faces_query: Query<&mut Visibility, (With<FacesCloud>, Without<VerticesCloud>, Without<EdgesCloud>, Without<OriginalMesh>)>,
    mut original_query: Query<&mut Visibility, (With<OriginalMesh>, Without<VerticesCloud>, Without<EdgesCloud>, Without<FacesCloud>)>,
    input: Res<ButtonInput<KeyCode>>,
) {
    let mut visibility_changed = false;
    
    // Toggle visibility based on number key input
    if input.just_pressed(KeyCode::Digit1) {
        cloud_visibility.vertices = !cloud_visibility.vertices;
        visibility_changed = true;
    }
    if input.just_pressed(KeyCode::Digit2) {
        cloud_visibility.edges = !cloud_visibility.edges;
        visibility_changed = true;
    }
    if input.just_pressed(KeyCode::Digit3) {
        cloud_visibility.faces = !cloud_visibility.faces;
        visibility_changed = true;
    }
    
    if visibility_changed {
        // Update splat cloud visibility
        if let Ok(mut visibility) = vertices_query.single_mut() {
            *visibility = if cloud_visibility.vertices { 
                Visibility::Visible 
            } else { 
                Visibility::Hidden 
            };
        }
        if let Ok(mut visibility) = edges_query.single_mut() {
            *visibility = if cloud_visibility.edges { 
                Visibility::Visible 
            } else { 
                Visibility::Hidden 
            };
        }
        if let Ok(mut visibility) = faces_query.single_mut() {
            *visibility = if cloud_visibility.faces { 
                Visibility::Visible 
            } else { 
                Visibility::Hidden 
            };
        }
        
        // Show original mesh only when no splat clouds are visible
        let show_original = !cloud_visibility.vertices 
            && !cloud_visibility.edges 
            && !cloud_visibility.faces;
            
        for mut visibility in original_query.iter_mut() {
            *visibility = if show_original { 
                Visibility::Visible 
            } else { 
                Visibility::Hidden 
            };
        }
        
        info!(
            "Visibility updated - Vertices: {}, Edges: {}, Faces: {}, Original: {}", 
            cloud_visibility.vertices, 
            cloud_visibility.edges, 
            cloud_visibility.faces, 
            show_original
        );
    }
}

/// Update the UI text showing controls and current state
fn update_info_text(
    mut text_query: Query<&mut Text, With<InfoText>>,
    cloud_visibility: Res<CloudVisibility>,
    metrics: Res<ConversionMetrics>,
) {
    let Ok(mut text) = text_query.single_mut() else { return };
    
    let vertex_indicator = if cloud_visibility.vertices { "[ON]" } else { "[OFF]" };
    let edge_indicator = if cloud_visibility.edges { "[ON]" } else { "[OFF]" };
    let face_indicator = if cloud_visibility.faces { "[ON]" } else { "[OFF]" };
    
    // Display conversion metrics if available
    let metrics_text = if metrics.total_gaussians > 0 {
        format!(
            "\n\
            Conversion Metrics:\n\
            • Total Gaussians: {}\n\
            • Conversion time: {:.1} ms",
            metrics.total_gaussians,
            metrics.conversion_time_ms
        )
    } else {
        String::new()
    };
    
    **text = format!(
        "Mesh to Gaussian Splats Demo\n\
        \n\
        Camera Controls:\n\
        • WASD: Orbit camera\n\
        • Q/E: Zoom in/out\n\
        \n\
        Splat Visibility:\n\
        • 1: Vertices {}\n\
        • 2: Edges {}\n\
        • 3: Faces {}{}",
        vertex_indicator, edge_indicator, face_indicator, metrics_text
    );
}

use bevy::{
    ecs::query::{Or, Without},
    prelude::*,
};
use mcubes::{MarchingCubes, MeshSide, Vec3 as McVec3};

use crate::{
    Gaussian3d as StoredGaussian3d,
    PlanarGaussian3d,
    PlanarGaussian3dHandle,
};

#[derive(Debug, Clone, Copy, Reflect)]
#[reflect(Default)]
pub struct ProxyParams {
    pub voxel_size: f32,
    pub iso_threshold: f32,
    pub nsig: f32,
}

impl Default for ProxyParams {
    fn default() -> Self {
        Self {
            voxel_size: 0.05,
            iso_threshold: 0.25,
            nsig: 3.0,
        }
    }
}

#[derive(Component, Debug, Clone, Copy, Reflect)]
#[reflect(Component)]
pub struct ProxyMeshSettings {
    pub params: ProxyParams,
}

impl Default for ProxyMeshSettings {
    fn default() -> Self {
        Self {
            params: ProxyParams::default(),
        }
    }
}

#[derive(Component, Debug, Clone, Reflect)]
#[reflect(Component)]
pub struct ProxyMesh(pub Handle<Mesh>);

#[derive(Debug, Clone, Copy)]
struct Gaussian3D {
    center: Vec3,
    rot: Quat,
    sigma: Vec3,
    weight: f32,
}

pub struct ProxyMeshPlugin;

impl Plugin for ProxyMeshPlugin {
    fn build(&self, app: &mut App) {
        app.register_type::<ProxyParams>();
        app.register_type::<ProxyMeshSettings>();
        app.register_type::<ProxyMesh>();
        app.add_systems(Update, generate_mesh_proxyes);
    }
}

fn generate_mesh_proxyes(
    mut commands: Commands,
    mut meshes: ResMut<Assets<Mesh>>,
    planar_clouds: Res<Assets<PlanarGaussian3d>>,
    query: Query<
        (
            Entity,
            &ProxyMeshSettings,
            &PlanarGaussian3dHandle,
            Option<&ProxyMesh>,
        ),
        Or<(
            Changed<ProxyMeshSettings>,
            Added<ProxyMeshSettings>,
            Without<ProxyMesh>,
        )>,
    >,
) {
    for (entity, settings, handle, existing_proxy) in &query {
        let Some(cloud) = planar_clouds.get(handle.handle()) else {
            continue;
        };

        let gaussians: Vec<Gaussian3D> = cloud
            .iter()
            .map(convert_gaussian)
            .collect();

        if gaussians.is_empty() {
            continue;
        }

        let mesh_proxy = mesh_from_gaussians(&gaussians, settings.params, &mut meshes);

        match existing_proxy {
            Some(current) if current.0 == mesh_proxy => {}
            _ => {
                commands.entity(entity).insert(ProxyMesh(mesh_proxy));
            }
        }
    }
}

fn convert_gaussian(gaussian: StoredGaussian3d) -> Gaussian3D {
    let mut rotation = Quat::from_array(gaussian.rotation.rotation);
    if !rotation.is_normalized() {
        rotation = rotation.normalize();
    }

    let sigma = Vec3::from_array(gaussian.scale_opacity.scale).max(Vec3::splat(1e-4));

    Gaussian3D {
        center: Vec3::from_array(gaussian.position_visibility.position),
        rot: rotation,
        sigma,
        weight: gaussian.position_visibility.visibility * gaussian.scale_opacity.opacity,
    }
}

fn mesh_from_gaussians(
    gaussians: &[Gaussian3D],
    params: ProxyParams,
    meshes: &mut Assets<Mesh>,
) -> Handle<Mesh> {
    let (minv, maxv) = union_aabb(gaussians, params.nsig);
    let size = maxv - minv;

    let steps = Vec3::splat(params.voxel_size);
    let nx = (size.x / steps.x).ceil().max(1.0) as usize + 1;
    let ny = (size.y / steps.y).ceil().max(1.0) as usize + 1;
    let nz = (size.z / steps.z).ceil().max(1.0) as usize + 1;

    let mut values = Vec::with_capacity(nx * ny * nz);

    for kz in 0..nz {
        let z = minv.z + kz as f32 * steps.z;
        for jy in 0..ny {
            let y = minv.y + jy as f32 * steps.y;
            for ix in 0..nx {
                let x = minv.x + ix as f32 * steps.x;
                values.push(density_at(Vec3::new(x, y, z), gaussians));
            }
        }
    }

    let marching_cubes = MarchingCubes::new(
        (nx, ny, nz),
        McVec3::new(size.x, size.y, size.z),
        McVec3::new(steps.x, steps.y, steps.z),
        McVec3::new(minv.x, minv.y, minv.z),
        values,
        params.iso_threshold,
    )
    .expect("failed to initialise marching cubes");

    let mesh_data = marching_cubes.generate(MeshSide::Both);

    let positions: Vec<[f32; 3]> = mesh_data
        .vertices
        .iter()
        .map(|vertex| {
            let p = vertex.posit;
            [p.x, p.y, p.z]
        })
        .collect();

    let normals: Vec<[f32; 3]> = mesh_data
        .vertices
        .iter()
        .map(|vertex| {
            let n = vertex.normal;
            [n.x, n.y, n.z]
        })
        .collect();

    let indices: Vec<u32> = mesh_data.indices.iter().copied().collect();

    let mut mesh = Mesh::new(PrimitiveTopology::TriangleList);
    mesh.insert_attribute(Mesh::ATTRIBUTE_POSITION, positions);
    mesh.insert_attribute(Mesh::ATTRIBUTE_NORMAL, normals);
    mesh.insert_attribute(Mesh::ATTRIBUTE_UV_0, vec![[0.0f32, 0.0f32]; mesh.count_vertices()]);

    mesh.set_indices(Some(Indices::U32(indices)));

    let normals_attr = mesh
        .attribute(Mesh::ATTRIBUTE_NORMAL)
        .unwrap()
        .as_float3()
        .unwrap()
        .to_vec();

    let tangents: Vec<[f32; 4]> = normals_attr
        .iter()
        .map(|normal| {
            let n = Vec3::from_array(*normal);
            let up = if n.z.abs() < 0.999 { Vec3::Z } else { Vec3::X };
            let tangent = n.cross(up).normalize_or_zero();
            [tangent.x, tangent.y, tangent.z, 1.0]
        })
        .collect();

    mesh.insert_attribute(Mesh::ATTRIBUTE_TANGENT, tangents);

    mesh.compute_normals();

    meshes.add(mesh)
}

fn union_aabb(gaussians: &[Gaussian3D], nsig: f32) -> (Vec3, Vec3) {
    let mut minv = Vec3::splat(f32::INFINITY);
    let mut maxv = Vec3::splat(f32::NEG_INFINITY);

    for gaussian in gaussians {
        let ax = gaussian.rot * (Vec3::X * gaussian.sigma.x * nsig);
        let ay = gaussian.rot * (Vec3::Y * gaussian.sigma.y * nsig);
        let az = gaussian.rot * (Vec3::Z * gaussian.sigma.z * nsig);
        let extent = Vec3::new(
            ax.x.abs() + ay.x.abs() + az.x.abs(),
            ax.y.abs() + ay.y.abs() + az.y.abs(),
            ax.z.abs() + ay.z.abs() + az.z.abs(),
        );

        minv = minv.min(gaussian.center - extent);
        maxv = maxv.max(gaussian.center + extent);
    }

    (minv, maxv)
}

fn density_at(p: Vec3, gaussians: &[Gaussian3D]) -> f32 {
    let mut s = 0.0;
    for gaussian in gaussians {
        let delta = p - gaussian.center;
        let local = gaussian.rot.conjugate() * delta;
        let q = (local.x * local.x) / (gaussian.sigma.x * gaussian.sigma.x)
            + (local.y * local.y) / (gaussian.sigma.y * gaussian.sigma.y)
            + (local.z * local.z) / (gaussian.sigma.z * gaussian.sigma.z);

        if q < 36.0 {
            s += gaussian.weight * (-0.5 * q).exp();
        }
    }
    s
}

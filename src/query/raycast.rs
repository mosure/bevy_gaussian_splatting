use bevy::{prelude::*, render::mesh::PrimitiveTopology};
use std::collections::BTreeMap;

struct Triangle {
    vertices: [Vec3; 3],
}

// TODO: update Handle<Mesh>
fn point_in_mesh_system(
    mesh_query: Query<(&Handle<Mesh>, &Transform)>,
    point_query: Query<(&Point, Entity)>,
    meshes: Res<Assets<Mesh>>,
    mut commands: Commands,
) {
    for (mesh_handle, transform) in mesh_query.iter() {
        if let Some(mesh) = meshes.get(mesh_handle) {
            for (point, entity) in point_query.iter() {
                let local_point = transform
                    .compute_matrix()
                    .inverse()
                    .transform_point3(point.position);

                if is_point_in_mesh(&local_point, mesh) {
                    commands.entity(entity).insert(InsideMesh);
                }
            }
        }
    }
}

fn is_point_in_mesh(point: &Vec3, mesh: &Mesh) -> bool {
    if mesh.primitive_topology() != PrimitiveTopology::TriangleList {
        panic!("Mesh must be a triangle list");
    }

    let vertices = if let Some(vertex_attribute) = mesh.attribute(Mesh::ATTRIBUTE_POSITION) {
        vertex_attribute
            .as_float3()
            .expect("Expected vertex positions as Vec3")
    } else {
        panic!("Mesh does not contain vertex positions");
    };

    let indices = if let Some(Indices::U32(indices)) = &mesh.indices() {
        indices
    } else {
        panic!("Mesh indices must be of type U32");
    };

    let mut intersections = 0;
    for chunk in indices.chunks_exact(3) {
        let triangle = Triangle {
            vertices: [
                vertices[chunk[0] as usize],
                vertices[chunk[1] as usize],
                vertices[chunk[2] as usize],
            ],
        };

        let ray_direction = Vec3::new(1.0, 0.0, 0.0);
        if ray_intersects_triangle(point, &ray_direction, &triangle) {
            intersections += 1;
        }
    }

    intersections % 2 != 0
}

fn ray_intersects_triangle(ray_origin: &Vec3, ray_direction: &Vec3, triangle: &Triangle) -> bool {
    let epsilon = 0.000_001;
    let vertex0 = triangle.vertices[0];
    let vertex1 = triangle.vertices[1];
    let vertex2 = triangle.vertices[2];

    let edge1 = vertex1 - vertex0;
    let edge2 = vertex2 - vertex0;
    let h = ray_direction.cross(edge2);
    let a = edge1.dot(h);

    if a > -epsilon && a < epsilon {
        return false;
    }

    let f = 1.0 / a;
    let s = *ray_origin - vertex0;
    let u = f * s.dot(h);

    if u < 0.0 || u > 1.0 {
        return false;
    }

    let q = s.cross(edge1);
    let v = f * ray_direction.dot(q);

    if v < 0.0 || u + v > 1.0 {
        return false;
    }

    let t = f * edge2.dot(q);

    t > epsilon
}

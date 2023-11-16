#import bevy_gaussian_splatting::bindings::{
    globals,
    points,
    particle_behaviors,
    ParticleBehavior,
}
#import bevy_gaussian_splatting::spherical_harmonics::spherical_harmonics_lookup
#import bevy_gaussian_splatting::transform::{
    world_to_clip,
    in_frustum,
}


@compute @workgroup_size(16)
fn apply_particle_behaviors(
    @builtin(global_invocation_id) gl_GlobalInvocationID: vec3<u32>,
) {
    let behavior_index = gl_GlobalInvocationID.x;
    let behavior = particle_behaviors[behavior_index];

    if (behavior.indicies.x < 0) {
        return;
    }

    let point_index = behavior.indicies.x;
    let point = points[point_index];

    let delta_position = behavior.velocity * globals.delta_time + 0.5 * behavior.acceleration * globals.delta_time * globals.delta_time;
    let delta_velocity = behavior.acceleration * globals.delta_time;

    let new_position = point.position + delta_position;
    let new_velocity = behavior.velocity + delta_velocity;

    workgroupBarrier();

    points[point_index].position = new_position;
    particle_behaviors[behavior_index].velocity = new_velocity;
}

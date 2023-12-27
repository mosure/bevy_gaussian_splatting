pub mod cloud;
pub mod f32;
pub mod packed;
pub mod rand;
pub mod settings;

#[cfg(feature = "f16")]
pub mod f16;

// TODO: add plugin with type registration
// TODO: add buffer/texture creation helpers (e.g. fn create_buffers() -> [Buffer; 1])

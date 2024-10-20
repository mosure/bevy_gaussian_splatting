use bevy::{
    prelude::*,
    render::extract_component::ExtractComponent
};


#[derive(
    Clone,
    Component,
    Debug,
    Default,
    ExtractComponent,
    Reflect,
)]
pub struct GaussianCamera;

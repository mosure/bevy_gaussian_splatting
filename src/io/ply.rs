use core::panic;
use std::io::BufRead;

use bevy_interleave::prelude::Planar;
use ply_rs::{
    parser::Parser,
    ply::{Property, PropertyAccess},
};

use crate::{
    gaussian::formats::{
        planar_3d::{Gaussian3d, PlanarGaussian3d},
        planar_4d::{Gaussian4d, PlanarGaussian4d},
    },
    material::{
        spherical_harmonics::{SH_CHANNELS, SH_COEFF_COUNT, SH_COEFF_COUNT_PER_CHANNEL},
        spherindrical_harmonics::SH_4D_COEFF_COUNT,
    },
};

pub const MAX_SIZE_VARIANCE: f32 = 4.0;

impl PropertyAccess for Gaussian3d {
    fn new() -> Self {
        Gaussian3d::default()
    }

    fn set_property(&mut self, key: String, property: Property) {
        match (key.as_ref(), property) {
            ("x", Property::Float(v)) => self.position_visibility.position[0] = v,
            ("y", Property::Float(v)) => self.position_visibility.position[1] = v,
            ("z", Property::Float(v)) => self.position_visibility.position[2] = v,
            ("visibility", Property::Float(v)) => self.position_visibility.visibility = v,
            ("f_dc_0", Property::Float(v)) => self.spherical_harmonic.set(0, v),
            ("f_dc_1", Property::Float(v)) => self.spherical_harmonic.set(1, v),
            ("f_dc_2", Property::Float(v)) => self.spherical_harmonic.set(2, v),
            ("scale_0", Property::Float(v)) => self.scale_opacity.scale[0] = v,
            ("scale_1", Property::Float(v)) => self.scale_opacity.scale[1] = v,
            ("scale_2", Property::Float(v)) => self.scale_opacity.scale[2] = v,
            ("opacity", Property::Float(v)) => {
                self.scale_opacity.opacity = 1.0 / (1.0 + (-v).exp())
            }
            ("rot_0", Property::Float(v)) => self.rotation.rotation[0] = v,
            ("rot_1", Property::Float(v)) => self.rotation.rotation[1] = v,
            ("rot_2", Property::Float(v)) => self.rotation.rotation[2] = v,
            ("rot_3", Property::Float(v)) => self.rotation.rotation[3] = v,
            (_, Property::Float(v)) if key.starts_with("f_rest_") => {
                let i = key[7..].parse::<usize>().unwrap();

                // interleaved
                // if (i + 3) < SH_COEFF_COUNT {
                //     self.spherical_harmonic.coefficients[i + 3] = v;
                // }

                // planar
                let channel = i / SH_COEFF_COUNT_PER_CHANNEL;
                let coefficient = if SH_COEFF_COUNT_PER_CHANNEL == 1 {
                    1
                } else {
                    (i % (SH_COEFF_COUNT_PER_CHANNEL - 1)) + 1
                };

                let interleaved_idx = coefficient * SH_CHANNELS + channel;

                if interleaved_idx < SH_COEFF_COUNT {
                    self.spherical_harmonic.set(interleaved_idx, v);
                } else {
                    // TODO: convert higher degree SH to lower degree SH
                }
            }
            (_, _) => {}
        }
    }
}

pub fn parse_ply_3d(mut reader: &mut dyn BufRead) -> Result<PlanarGaussian3d, std::io::Error> {
    let gaussian_parser = Parser::<Gaussian3d>::new();
    let header = gaussian_parser.read_header(&mut reader)?;

    let mut cloud = Vec::new();

    let required_properties = vec![
        "x", "y", "z", "f_dc_0", "f_dc_1", "f_dc_2", "scale_0", "scale_1", "opacity", "rot_0",
        "rot_1", "rot_2", "rot_3",
    ];
    let mut required_property_count = required_properties.len();

    for (_key, element) in &header.elements {
        if element.name == "vertex" {
            for (key, _prop) in &element.properties {
                required_property_count -= required_properties.contains(&key.as_str()) as usize;
            }

            if required_property_count > 0 {
                return Err(std::io::Error::new(
                    std::io::ErrorKind::InvalidData,
                    "missing required properties",
                ));
            }

            cloud = gaussian_parser.read_payload_for_element(&mut reader, element, &header)?;
        }
    }

    for gaussian in &mut cloud {
        // TODO: add automatic scaling normalization detection (e.g. don't normalize twice)
        let mean_scale = (gaussian.scale_opacity.scale[0]
            + gaussian.scale_opacity.scale[1]
            + gaussian.scale_opacity.scale[2])
            / 3.0;
        for i in 0..3 {
            gaussian.scale_opacity.scale[i] = gaussian.scale_opacity.scale[i]
                .max(mean_scale - MAX_SIZE_VARIANCE)
                .min(mean_scale + MAX_SIZE_VARIANCE)
                .exp();
        }

        let norm = (0..4)
            .map(|i| gaussian.rotation.rotation[i].powf(2.0))
            .sum::<f32>()
            .sqrt();
        for i in 0..4 {
            gaussian.rotation.rotation[i] /= norm;
        }
    }

    // pad with empty gaussians to multiple of 32
    let pad = 32 - (cloud.len() % 32);
    cloud.extend(std::iter::repeat_n(Gaussian3d::default(), pad));

    Ok(PlanarGaussian3d::from_interleaved(cloud))
}

impl PropertyAccess for Gaussian4d {
    fn new() -> Self {
        Gaussian4d::default()
    }

    fn set_property(&mut self, key: String, property: Property) {
        match (key.as_ref(), property) {
            ("x", Property::Float(v)) => self.position_visibility.position[0] = v,
            ("y", Property::Float(v)) => self.position_visibility.position[1] = v,
            ("z", Property::Float(v)) => self.position_visibility.position[2] = v,
            ("visibility", Property::Float(v)) => self.position_visibility.visibility = v,

            ("t", Property::Float(v)) => self.timestamp_timescale.timestamp = v,
            ("st", Property::Float(v)) => self.timestamp_timescale.timescale = v,

            (_, Property::Float(v)) if key.starts_with("feat_") => {
                let channel = match key.chars().nth(5).unwrap() {
                    'r' => 0,
                    'g' => 1,
                    'b' => 2,
                    _ => panic!("invalid feature channel, expected r, g, or b"),
                };
                let i = key[7..].parse::<usize>().unwrap();
                let interleaved_idx = i * SH_CHANNELS + channel;

                if interleaved_idx < SH_4D_COEFF_COUNT {
                    self.spherindrical_harmonic.set(interleaved_idx, v);
                } else {
                    // TODO: handle higher-degree if needed
                }
            }

            ("sx", Property::Float(v)) => self.scale_opacity.scale[0] = v,
            ("sy", Property::Float(v)) => self.scale_opacity.scale[1] = v,
            ("sz", Property::Float(v)) => self.scale_opacity.scale[2] = v,
            ("opacity", Property::Float(v)) => self.scale_opacity.opacity = v,

            ("rot_x", Property::Float(v)) => self.isotropic_rotations.rotation[0] = v,
            ("rot_y", Property::Float(v)) => self.isotropic_rotations.rotation[1] = v,
            ("rot_z", Property::Float(v)) => self.isotropic_rotations.rotation[2] = v,
            ("rot_w", Property::Float(v)) => self.isotropic_rotations.rotation[3] = v,

            ("rot_r_x", Property::Float(v)) => self.isotropic_rotations.rotation_r[0] = v,
            ("rot_r_y", Property::Float(v)) => self.isotropic_rotations.rotation_r[1] = v,
            ("rot_r_z", Property::Float(v)) => self.isotropic_rotations.rotation_r[2] = v,
            ("rot_r_w", Property::Float(v)) => self.isotropic_rotations.rotation_r[3] = v,
            _ => {}
        }
    }
}

pub fn parse_ply_4d(mut reader: &mut dyn BufRead) -> Result<PlanarGaussian4d, std::io::Error> {
    let parser = Parser::<Gaussian4d>::new();
    let header = parser.read_header(&mut reader)?;

    let mut cloud = Vec::new();

    let required_properties = vec![
        "x", "y", "z", "t", "st", "sx", "sy", "sz", "opacity", "rot_x", "rot_y", "rot_z", "rot_w",
        "rot_r_x", "rot_r_y", "rot_r_z", "rot_r_w",
    ];
    let mut required_property_count = required_properties.len();

    for (_key, element) in &header.elements {
        if element.name == "vertex" {
            for (key, _prop) in &element.properties {
                required_property_count -= required_properties.contains(&key.as_str()) as usize;
            }

            if required_property_count > 0 {
                return Err(std::io::Error::new(
                    std::io::ErrorKind::InvalidData,
                    "missing required properties",
                ));
            }

            cloud = parser.read_payload_for_element(&mut reader, element, &header)?;
        }
    }

    for g in &mut cloud {
        let norm = g
            .isotropic_rotations
            .rotation
            .iter()
            .map(|v| v.powi(2))
            .sum::<f32>()
            .sqrt();

        for v in &mut g.isotropic_rotations.rotation {
            *v /= norm;
        }

        let norm = g
            .isotropic_rotations
            .rotation_r
            .iter()
            .map(|v| v.powi(2))
            .sum::<f32>()
            .sqrt();

        for v in &mut g.isotropic_rotations.rotation_r {
            *v /= norm;
        }

        // TODO: normalize timescale between 0 and 1
    }

    // pad to multiple of 32
    let pad = 32 - (cloud.len() % 32);
    cloud.extend(std::iter::repeat_n(Gaussian4d::default(), pad));

    Ok(PlanarGaussian4d::from_interleaved(cloud))
}

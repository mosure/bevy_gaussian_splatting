use std::io::BufRead;

use bevy_interleave::prelude::Planar;
use ply_rs::{
    ply::{
        Property,
        PropertyAccess,
    },
    parser::Parser,
};

use crate::{
    gaussian::packed::{Gaussian3d, PlanarGaussian3d},
    material::spherical_harmonics::{
        SH_CHANNELS,
        SH_COEFF_COUNT,
        SH_COEFF_COUNT_PER_CHANNEL,
    },
};


pub const MAX_SIZE_VARIANCE: f32 = 5.0;

impl PropertyAccess for Gaussian3d {
    fn new() -> Self {
        Gaussian3d::default()
    }

    fn set_property(&mut self, key: String, property: Property) {
        match (key.as_ref(), property) {
            ("x", Property::Float(v))           => self.position_visibility.position[0] = v,
            ("y", Property::Float(v))           => self.position_visibility.position[1] = v,
            ("z", Property::Float(v))           => self.position_visibility.position[2] = v,
            ("f_dc_0", Property::Float(v))      => self.spherical_harmonic.set(0, v),
            ("f_dc_1", Property::Float(v))      => self.spherical_harmonic.set(1, v),
            ("f_dc_2", Property::Float(v))      => self.spherical_harmonic.set(2, v),
            ("scale_0", Property::Float(v))     => self.scale_opacity.scale[0] = v,
            ("scale_1", Property::Float(v))     => self.scale_opacity.scale[1] = v,
            ("scale_2", Property::Float(v))     => self.scale_opacity.scale[2] = v,
            ("opacity", Property::Float(v))     => self.scale_opacity.opacity = 1.0 / (1.0 + (-v).exp()),
            ("rot_0", Property::Float(v))       => self.rotation.rotation[0] = v,
            ("rot_1", Property::Float(v))       => self.rotation.rotation[1] = v,
            ("rot_2", Property::Float(v))       => self.rotation.rotation[2] = v,
            ("rot_3", Property::Float(v))       => self.rotation.rotation[3] = v,
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
            (_, _) => {},
        }
    }
}

pub fn parse_ply(mut reader: &mut dyn BufRead) -> Result<PlanarGaussian3d, std::io::Error> {
    // TODO: detect and parse Gaussian vs Gaussian4d
    let gaussian_parser = Parser::<Gaussian3d>::new();
    let header = gaussian_parser.read_header(&mut reader)?;

    let mut cloud = Vec::new();

    for (_ignore_key, element) in &header.elements {
        if element.name == "vertex" {
            cloud = gaussian_parser.read_payload_for_element(&mut reader, element, &header)?;
        }
    }

    for gaussian in &mut cloud {
        gaussian.position_visibility.visibility = 1.0;

        let mean_scale = (gaussian.scale_opacity.scale[0] + gaussian.scale_opacity.scale[1] + gaussian.scale_opacity.scale[2]) / 3.0;
        for i in 0..3 {
            gaussian.scale_opacity.scale[i] = gaussian.scale_opacity.scale[i]
                .max(mean_scale - MAX_SIZE_VARIANCE)
                .min(mean_scale + MAX_SIZE_VARIANCE)
                .exp();
        }

        let norm = (0..4).map(|i| gaussian.rotation.rotation[i].powf(2.0)).sum::<f32>().sqrt();
        for i in 0..4 {
            gaussian.rotation.rotation[i] /= norm;
        }
    }

    // pad with empty gaussians to multiple of 32
    let pad = 32 - (cloud.len() % 32);
    cloud.extend(std::iter::repeat(Gaussian3d::default()).take(pad));

    Ok(PlanarGaussian3d::from_interleaved(cloud))
}

use std::io::BufRead;

use bevy::asset::Error;
use ply_rs::{
    ply::{
        Property,
        PropertyAccess,
    },
    parser::Parser,
};

use crate::gaussian::{
    Gaussian,
    MAX_SH_COEFF_COUNT_PER_CHANNEL,
    MAX_SIZE_VARIANCE,
    SH_CHANNELS,
};


impl PropertyAccess for Gaussian {
    fn new() -> Self {
        Gaussian::default()
    }

    fn set_property(&mut self, key: String, property: Property) {
        match (key.as_ref(), property) {
            ("x", Property::Float(v))           => self.position[0] = v,
            ("y", Property::Float(v))           => self.position[1] = v,
            ("z", Property::Float(v))           => self.position[2] = v,
            ("f_dc_0", Property::Float(v))      => self.spherical_harmonic.coefficients[0] = v,
            ("f_dc_1", Property::Float(v))      => self.spherical_harmonic.coefficients[1] = v,
            ("f_dc_2", Property::Float(v))      => self.spherical_harmonic.coefficients[2] = v,
            ("scale_0", Property::Float(v))     => self.scale_opacity[0] = v,
            ("scale_1", Property::Float(v))     => self.scale_opacity[1] = v,
            ("scale_2", Property::Float(v))     => self.scale_opacity[2] = v,
            ("opacity", Property::Float(v))     => self.scale_opacity[3] = 1.0 / (1.0 + (-v).exp()),
            ("rot_0", Property::Float(v))       => self.rotation[0] = v,
            ("rot_1", Property::Float(v))       => self.rotation[1] = v,
            ("rot_2", Property::Float(v))       => self.rotation[2] = v,
            ("rot_3", Property::Float(v))       => self.rotation[3] = v,
            (_, Property::Float(v)) if key.starts_with("f_rest_") => {
                let i = key[7..].parse::<usize>().unwrap();

                match i {
                    _ if i + 3 < self.spherical_harmonic.coefficients.len() => {
                        self.spherical_harmonic.coefficients[i + 3] = v;
                    },
                    _ => { },
                }
            }
            (_, _) => {},
        }
    }
}

pub fn parse_ply(mut reader: &mut dyn BufRead) -> Result<Vec<Gaussian>, Error> {
    let gaussian_parser = Parser::<Gaussian>::new();
    let header = gaussian_parser.read_header(&mut reader)?;

    let mut cloud = Vec::new();

    for (_ignore_key, element) in &header.elements {
        match element.name.as_ref() {
            "vertex" => { cloud = gaussian_parser.read_payload_for_element(&mut reader, &element, &header)?; },
            _ => {},
        }
    }

    for gaussian in &mut cloud {
        gaussian.position[3] = 1.0;

        let mean_scale = (gaussian.scale_opacity[0] + gaussian.scale_opacity[1] + gaussian.scale_opacity[2]) / 3.0;
        for i in 0..3 {
            gaussian.scale_opacity[i] = gaussian.scale_opacity[i]
                .max(mean_scale - MAX_SIZE_VARIANCE)
                .min(mean_scale + MAX_SIZE_VARIANCE)
                .exp();
        }

        let sh_src = gaussian.spherical_harmonic.coefficients.clone();
        let sh = &mut gaussian.spherical_harmonic.coefficients;
        for i in SH_CHANNELS..sh_src.len() {
            let j = i - SH_CHANNELS;

            let channel = j / (MAX_SH_COEFF_COUNT_PER_CHANNEL - 1);
            let coefficient = (j % (MAX_SH_COEFF_COUNT_PER_CHANNEL - 1)) + 1;

            let interleaved_idx = coefficient * SH_CHANNELS + channel;
            assert!(interleaved_idx >= SH_CHANNELS);

            sh[interleaved_idx] = sh_src[i];
        }
    }

    Ok(cloud)
}

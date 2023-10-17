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
            // ("nx", Property::Float(v))          => self.normal.x = v,
            // ("ny", Property::Float(v))          => self.normal.y = v,
            // ("nz", Property::Float(v))          => self.normal.z = v,
            ("f_dc_0", Property::Float(v))      => self.spherical_harmonic.coefficients[0] = v,
            ("f_dc_1", Property::Float(v))      => self.spherical_harmonic.coefficients[1] = v,
            ("f_dc_2", Property::Float(v))      => self.spherical_harmonic.coefficients[2] = v,
            ("opacity", Property::Float(v))     => self.opacity = 1.0 / (1.0 + (-v).exp()),
            ("scale_0", Property::Float(v))     => self.scale.x = v,
            ("scale_1", Property::Float(v))     => self.scale.y = v,
            ("scale_2", Property::Float(v))     => self.scale.z = v,
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
        // let mean_scale = (gaussian.scale.x + gaussian.scale.y + gaussian.scale.z) / 3.0;
        gaussian.scale = gaussian.scale
            // .max(Vec3::splat(mean_scale - MAX_SIZE_VARIANCE))
            // .min(Vec3::splat(mean_scale + MAX_SIZE_VARIANCE))
            .exp();

        // let rot = &gaussian.rotation;
        // let qlen = (rot[0] * rot[0] + rot[1] * rot[1] + rot[2] * rot[2] + rot[3] * rot[3]).sqrt();
        // gaussian.rotation = [
        //     rot[0] / qlen,
        //     rot[1] / qlen,
        //     rot[2] / qlen,
        //     rot[3] / qlen,
        // ];

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

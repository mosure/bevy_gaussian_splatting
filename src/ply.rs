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
    MAX_SH_COEFF_COUNT,
};


impl PropertyAccess for Gaussian {
    fn new() -> Self {
        Gaussian::default()
    }

    fn set_property(&mut self, key: String, property: Property) {
        match (key.as_ref(), property) {
            ("x", Property::Float(v))           => self.position.x = v,
            ("y", Property::Float(v))           => self.position.y = v,
            ("z", Property::Float(v))           => self.position.z = v,
            ("nx", Property::Float(v))          => self.normal.x = v,
            ("ny", Property::Float(v))          => self.normal.y = v,
            ("nz", Property::Float(v))          => self.normal.z = v,
            ("f_dc_0", Property::Float(v))      => self.spherical_harmonic.coefficients[0].x = v,
            ("f_dc_1", Property::Float(v))      => self.spherical_harmonic.coefficients[0].y = v,
            ("f_dc_2", Property::Float(v))      => self.spherical_harmonic.coefficients[0].z = v,
            ("opacity", Property::Float(v))     => self.opacity = v,
            ("scale_0", Property::Float(v))     => self.scale.x = v,
            ("scale_1", Property::Float(v))     => self.scale.y = v,
            ("scale_2", Property::Float(v))     => self.scale.z = v,
            ("rot_0", Property::Float(v))       => self.rotation[0] = v,
            ("rot_1", Property::Float(v))       => self.rotation[1] = v,
            ("rot_2", Property::Float(v))       => self.rotation[2] = v,
            ("rot_3", Property::Float(v))       => self.rotation[3] = v,
            (_, Property::Float(v)) if key.starts_with("f_rest_") => {
                let i = key[7..].parse::<usize>().unwrap();
                let sh_upper_bound = MAX_SH_COEFF_COUNT - 3;

                match i {
                    _ if i < sh_upper_bound => {
                        let i = i + 3;
                        let j = i / 3;
                        let k = i % 3;

                        // TODO: verify this is the correct sh order
                        self.spherical_harmonic.coefficients[j][k] = v;
                    },
                    _ => {
                        // println!("unmapped property: {}", key);
                        // println!("value: {}", v);
                    },
                }
            }
            (_, _) => {},
        }
    }
}

pub fn parse_ply(mut reader: &mut dyn BufRead) -> Result<Vec<Gaussian>, Error> {
    let gaussian_parser = Parser::<Gaussian>::new();
    let header = gaussian_parser.read_header(&mut reader)?;

    // TODO: determine spherical harmonic order from header (for speedup on lower orders)

    let mut cloud = Vec::new();

    for (_ignore_key, element) in &header.elements {
        match element.name.as_ref() {
            "vertex" => { cloud = gaussian_parser.read_payload_for_element(&mut reader, &element, &header)?; },
            _ => {},
        }
    }

    Ok(cloud)
}

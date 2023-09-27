use bevy::{
    prelude::*,
    asset::{
        AssetLoader,
        LoadContext,
        LoadedAsset,
    },
    reflect::TypeUuid,
    utils::BoxedFuture,
};
use ply_rs::{
    ply::{
        Property,
        PropertyAccess,
    },
    parser::Parser,
};

use std::io::{
    BufReader,
    Cursor,
};


#[derive(Clone, Debug, Reflect)]
pub struct Gaussian {
    // TODO: store position, scale, and rotation as a Transform
    pub x: f32,
    pub y: f32,
    pub z: f32,
    pub nx: f32,
    pub ny: f32,
    pub nz: f32,
    pub f_dc_0: f32,
    pub f_dc_1: f32,
    pub f_dc_2: f32,
    pub f_rest_0: f32,
    // TODO: store f_rest_0 through f_rest_44 as an array
    pub opacity: f32,
    pub scale_0: f32,
    pub scale_1: f32,
    pub scale_2: f32,
    pub rot_0: f32,
    pub rot_1: f32,
    pub rot_2: f32,
    pub rot_3: f32,
}

impl PropertyAccess for Gaussian {
    fn new() -> Self {
        Gaussian {
            x: 0.0,
            y: 0.0,
            z: 0.0,
            nx: 0.0,
            ny: 0.0,
            nz: 0.0,
            f_dc_0: 0.0,
            f_dc_1: 0.0,
            f_dc_2: 0.0,
            f_rest_0: 0.0,
            opacity: 0.0,
            scale_0: 0.0,
            scale_1: 0.0,
            scale_2: 0.0,
            rot_0: 0.0,
            rot_1: 0.0,
            rot_2: 0.0,
            rot_3: 0.0,
        }
    }

    fn set_property(&mut self, key: String, property: Property) {
        match (key.as_ref(), property) {
            ("x", Property::Float(v))           => self.x = v,
            ("y", Property::Float(v))           => self.y = v,
            ("z", Property::Float(v))           => self.z = v,
            ("nx", Property::Float(v))          => self.nx = v,
            ("ny", Property::Float(v))          => self.ny = v,
            ("nz", Property::Float(v))          => self.nz = v,
            ("f_dc_0", Property::Float(v))      => self.f_dc_0 = v,
            ("f_dc_1", Property::Float(v))      => self.f_dc_1 = v,
            ("f_dc_2", Property::Float(v))      => self.f_dc_2 = v,
            ("f_rest_0", Property::Float(v))    => self.f_rest_0 = v,
            ("opacity", Property::Float(v))     => self.opacity = v,
            ("scale_0", Property::Float(v))     => self.scale_0 = v,
            ("scale_1", Property::Float(v))     => self.scale_1 = v,
            ("scale_2", Property::Float(v))     => self.scale_2 = v,
            ("rot_0", Property::Float(v))       => self.rot_0 = v,
            ("rot_1", Property::Float(v))       => self.rot_1 = v,
            ("rot_2", Property::Float(v))       => self.rot_2 = v,
            ("rot_3", Property::Float(v))       => self.rot_3 = v,
            (_, _) => {},
            //(k, _) => panic!("Gaussian: unexpected key/value combination: key: {}", k),
        }
    }
}

#[derive(Clone, Debug, Reflect, TypeUuid)]
#[uuid = "ac2f08eb-bc32-aabb-ff21-51571ea332d5"]
pub struct GaussianCloud(Vec<Gaussian>);


#[derive(Default)]
pub struct GaussianCloudLoader;

impl AssetLoader for GaussianCloudLoader {
    fn load<'a>(
        &'a self,
        bytes: &'a [u8],
        load_context: &'a mut LoadContext,
    ) -> BoxedFuture<'a, Result<(), bevy::asset::Error>> {
        Box::pin(async move {
            let cursor = Cursor::new(bytes);
            let mut f = BufReader::new(cursor);

            let gaussian_parser = Parser::<Gaussian>::new();
            let header = gaussian_parser.read_header(&mut f)?;

            let mut cloud = GaussianCloud(Vec::new());

            for (_ignore_key, element) in &header.elements {
                match element.name.as_ref() {
                    "vertex" => { cloud = GaussianCloud(gaussian_parser.read_payload_for_element(&mut f, &element, &header)?); },
                    _ => {},
                    //_ => panic!("GaussianCloudLoader: unexpected element: {}", element.name)
                }
            }

            println!("GaussianCloudLoader: loaded {} verticies", cloud.0.len());

            load_context.set_default_asset(LoadedAsset::new(cloud));
            Ok(())
        })
    }

    fn extensions(&self) -> &[&str] {
        &["ply"]
    }
}

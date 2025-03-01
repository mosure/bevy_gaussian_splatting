use rkyv::{
    archived_root,
};
use crate::{
    io::codec::CloudCodec,
    gaussian::formats::{
        planar_3d::PlanarGaussian3d,
        planar_4d::PlanarGaussian4d,
    },
};


// TODO: add rkyv support to bevy_interleave
impl CloudCodec for PlanarGaussian3d {
    fn encode(&self) -> Vec<u8> {
        let _bytes = rkyv::to_bytes::<rkyv::rancor::Error>(self).unwrap();

        let mut serializer = AlignedVecSerializer::new();
        serializer
            .serialize_value(self)
            .expect("failed to serialize cloud");
        serializer.into_inner()
    }

    fn decode(data: &[u8]) -> Self {
        let archived = unsafe { rkyv::archived_root::<Self>(data) };
        archived
            .deserialize(&mut rkyv::Infallible)
            .expect("deserialization failed")
    }
}

impl CloudCodec for PlanarGaussian4d {
    fn encode(&self) -> Vec<u8> {
        let mut serializer = AlignedVecSerializer::new();
        serializer
            .serialize_value(self)
            .expect("failed to serialize cloud");
        serializer.into_inner()
    }

    fn decode(data: &[u8]) -> Self {
        let archived = unsafe { rkyv::archived_root::<Self>(data) };
        archived
            .deserialize(&mut rkyv::Infallible)
            .expect("deserialization failed")
    }
}

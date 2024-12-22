use flexbuffers::{
    FlexbufferSerializer,
    Reader,
};
use serde::{
    Deserialize,
    Serialize,
};

use crate::{
    Cloud,
    io::codec::CloudCodec,
};


impl CloudCodec for Cloud {
    fn encode(&self) -> Vec<u8> {
        let mut serializer = FlexbufferSerializer::new();
        self.serialize(&mut serializer).expect("failed to serialize cloud");

        serializer.view().to_vec()
    }

    fn decode(data: &[u8]) -> Self {
        let reader = Reader::get_root(data).expect("failed to read flexbuffer");
        let cloud = Cloud::deserialize(reader).expect("deserialization failed");

        cloud
    }
}

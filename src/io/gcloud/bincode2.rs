use bincode2::{
    deserialize_from,
    serialize_into,
};
use flate2::{
    Compression,
    read::GzDecoder,
    write::GzEncoder,
};

use crate::{
    gaussian::GaussianCloud,
    io::codec::GaussianCloudCodec,
};


impl GaussianCloudCodec for GaussianCloud {
    fn encode(&self) -> Vec<u8> {
        let mut output = Vec::new();

        {
            let mut gz_encoder = GzEncoder::new(&mut output, Compression::default());
            serialize_into(&mut gz_encoder, &self).expect("failed to encode cloud");
        }

        output
    }

    fn decode(data: &[u8]) -> Self {
        let decompressed = GzDecoder::new(data);
        let cloud: GaussianCloud = deserialize_from(decompressed).expect("failed to decode cloud");

        cloud
    }
}

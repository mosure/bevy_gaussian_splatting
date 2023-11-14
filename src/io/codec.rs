
// TODO: support streamed codecs
pub trait GaussianCloudCodec {
    fn encode(&self) -> Vec<u8>;
    fn decode(data: &[u8]) -> Self;
}

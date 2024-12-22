
// TODO: support streamed codecs
pub trait CloudCodec {
    fn encode(&self) -> Vec<u8>;
    fn decode(data: &[u8]) -> Self;
}

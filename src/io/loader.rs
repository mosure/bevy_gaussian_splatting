use std::io::{
    BufReader,
    Cursor,
    ErrorKind,
};

use bevy::{
    asset::{
        AssetLoader,
        AsyncReadExt,
        LoadContext,
        io::Reader,
    },
    utils::BoxedFuture,
};

use crate::{
    GaussianCloud,
    io::codec::GaussianCloudCodec,
};


#[derive(Default)]
pub struct GaussianCloudLoader;

impl AssetLoader for GaussianCloudLoader {
    type Asset = GaussianCloud;
    type Settings = ();
    type Error = std::io::Error;

    fn load<'a>(
        &'a self,
        reader: &'a mut Reader,
        _settings: &'a Self::Settings,
        load_context: &'a mut LoadContext,
    ) -> BoxedFuture<'a, Result<Self::Asset, Self::Error>> {

        Box::pin(async move {
            let mut bytes = Vec::new();
            reader.read_to_end(&mut bytes).await?;

            match load_context.path().extension() {
                Some(ext) if ext == "ply" => {
                    #[cfg(feature = "io_ply")]
                    {
                        let cursor = Cursor::new(bytes);
                        let mut f = BufReader::new(cursor);

                        let gaussians = crate::io::ply::parse_ply(&mut f)?;

                        Ok(GaussianCloud::from_gaussians(gaussians))
                    }

                    #[cfg(not(feature = "io_ply"))]
                    {
                        Err(std::io::Error::new(ErrorKind::Other, "ply support not enabled, enable with io_ply feature"))
                    }
                },
                Some(ext) if ext == "gcloud" => {
                    let cloud = GaussianCloud::decode(bytes.as_slice());

                    Ok(cloud)
                },
                _ => Err(std::io::Error::new(ErrorKind::Other, "only .ply and .gcloud supported")),
            }
        })
    }

    fn extensions(&self) -> &[&str] {
        &["ply", "gcloud"]
    }
}

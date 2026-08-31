#[cfg(feature = "lod")]
pub mod atlas_upload;
#[cfg(feature = "lod")]
pub mod bridge;
pub mod cache;
pub mod hierarchy;
#[cfg(feature = "lod")]
pub mod http;
#[cfg(feature = "lod")]
pub mod lodge;
// The portable LODGE format and CPU pair planner remain available in every
// `lod` build.  Resident ECS presentation additionally needs the storage-
// buffer radix consumer; WebGL2/buffer-texture builds must not compile a
// component which they cannot make drawable.
#[cfg(lod_render_path)]
pub mod lodge_resident;
#[cfg(feature = "lod")]
pub mod lodge_status;
#[cfg(feature = "lod")]
pub mod package;
#[cfg(feature = "lod")]
pub mod package_source;
#[cfg(feature = "lod")]
pub mod persistent_cache;
#[cfg(feature = "lod")]
pub mod preprocess;
#[cfg(feature = "lod")]
pub mod render_commit;
#[cfg(feature = "lod")]
pub mod runtime;
#[cfg(feature = "lod")]
pub mod status;
pub mod transport;

/// Whether this build contains the complete render-world consumer required by
/// the LoD bridge and package two-phase commit protocol.
///
/// The portable hierarchy, codec, and streaming APIs remain available in
/// other feature combinations. Runtime LoD rendering, however, requires the
/// storage-buffer radix path and is intentionally unavailable on WebGL2.
#[cfg(feature = "lod")]
pub const fn lod_render_path_is_supported() -> bool {
    cfg!(lod_render_path)
}

/// Typed rejection returned before a bridge or package allocates streaming
/// state that no render-world system could ever commit.
#[cfg(feature = "lod")]
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
#[non_exhaustive]
pub enum LodRenderPathSupportError {
    UnsupportedBuildConfiguration,
}

#[cfg(feature = "lod")]
impl std::fmt::Display for LodRenderPathSupportError {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::UnsupportedBuildConfiguration => write!(
                formatter,
                "LoD rendering requires sort_radix + buffer_storage without buffer_texture or webgl2"
            ),
        }
    }
}

#[cfg(feature = "lod")]
impl std::error::Error for LodRenderPathSupportError {}

#[cfg(feature = "lod")]
pub(crate) fn require_lod_render_path() -> Result<(), LodRenderPathSupportError> {
    if lod_render_path_is_supported() {
        Ok(())
    } else {
        Err(LodRenderPathSupportError::UnsupportedBuildConfiguration)
    }
}

#[cfg(all(test, feature = "lod"))]
mod tests {
    use super::*;

    #[test]
    fn lod_render_path_support_matches_the_functional_handshake_cfg() {
        let expected = cfg!(lod_render_path);
        assert_eq!(lod_render_path_is_supported(), expected);
        assert_eq!(require_lod_render_path().is_ok(), expected);
        if !expected {
            assert_eq!(
                require_lod_render_path(),
                Err(LodRenderPathSupportError::UnsupportedBuildConfiguration)
            );
        }
    }
}

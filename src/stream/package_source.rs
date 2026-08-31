//! Canonical package-source derivation from a manifest URI.

use std::fmt;
#[cfg(not(target_arch = "wasm32"))]
use std::path::Path;

use super::{
    http::{HttpRangeTransportError, validate_base_url},
    package::GaussianLodPackageSource,
};

/// Typed failure returned before a manifest URI is converted into a package
/// page-source root.
#[derive(Clone, Debug, Eq, PartialEq)]
#[non_exhaustive]
pub enum GaussianLodPackageSourceError {
    EmptyManifestUri,
    UnsupportedScheme(String),
    MissingManifestPath(String),
    InvalidHttpUri(HttpRangeTransportError),
    BrowserLocationUnavailable,
    BrowserUrlResolutionFailed,
}

impl fmt::Display for GaussianLodPackageSourceError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::EmptyManifestUri => write!(formatter, "LoD manifest URI is empty"),
            Self::UnsupportedScheme(scheme) => {
                write!(formatter, "unsupported LoD manifest URI scheme '{scheme}'")
            }
            Self::MissingManifestPath(uri) => {
                write!(formatter, "LoD manifest URI does not name a file: '{uri}'")
            }
            Self::InvalidHttpUri(error) => write!(formatter, "invalid LoD manifest URL: {error}"),
            Self::BrowserLocationUnavailable => {
                write!(formatter, "browser location is unavailable")
            }
            Self::BrowserUrlResolutionFailed => {
                write!(formatter, "could not resolve the LoD manifest URL")
            }
        }
    }
}

impl std::error::Error for GaussianLodPackageSourceError {}

impl GaussianLodPackageSource {
    /// Derive the page-source root adjacent to a `.gsplatlod` manifest.
    ///
    /// Native relative paths and `file://` URIs produce a confined native
    /// directory source. Browser-relative paths are resolved against the
    /// document URL and produce an HTTP range source. HTTP URLs use the
    /// transport's deliberately hardened URL subset: query strings,
    /// fragments, credentials, percent escapes, backslashes, controls and
    /// spaces are rejected rather than normalized silently.
    pub fn try_from_manifest_uri(
        manifest_uri: &str,
    ) -> Result<Self, GaussianLodPackageSourceError> {
        if manifest_uri.is_empty() {
            return Err(GaussianLodPackageSourceError::EmptyManifestUri);
        }

        #[cfg(target_arch = "wasm32")]
        {
            let window = web_sys::window()
                .ok_or(GaussianLodPackageSourceError::BrowserLocationUnavailable)?;
            let base = window
                .location()
                .href()
                .map_err(|_| GaussianLodPackageSourceError::BrowserLocationUnavailable)?;
            let resolved = web_sys::Url::new_with_base(manifest_uri, &base)
                .map_err(|_| GaussianLodPackageSourceError::BrowserUrlResolutionFailed)?;
            http_source_from_manifest_url(&resolved.href())
        }

        #[cfg(not(target_arch = "wasm32"))]
        {
            if manifest_uri.starts_with("https://") || manifest_uri.starts_with("http://") {
                return http_source_from_manifest_url(manifest_uri);
            }
            if let Some((scheme, _)) = manifest_uri.split_once("://")
                && scheme != "file"
            {
                return Err(GaussianLodPackageSourceError::UnsupportedScheme(
                    scheme.to_owned(),
                ));
            }
            let native_path = manifest_uri.strip_prefix("file://").unwrap_or(manifest_uri);
            if native_path.is_empty() {
                return Err(GaussianLodPackageSourceError::MissingManifestPath(
                    manifest_uri.to_owned(),
                ));
            }
            let root = Path::new(native_path)
                .parent()
                .filter(|parent| !parent.as_os_str().is_empty())
                .unwrap_or_else(|| Path::new("."))
                .to_string_lossy()
                .into_owned();
            Ok(Self::native_directory(root))
        }
    }
}

fn http_source_from_manifest_url(
    manifest_uri: &str,
) -> Result<GaussianLodPackageSource, GaussianLodPackageSourceError> {
    validate_base_url(manifest_uri).map_err(GaussianLodPackageSourceError::InvalidHttpUri)?;
    let authority_start = manifest_uri.find("://").expect("HTTP scheme was validated") + 3;
    let separator = manifest_uri[authority_start..]
        .rfind('/')
        .map(|offset| authority_start + offset)
        .ok_or_else(|| {
            GaussianLodPackageSourceError::MissingManifestPath(manifest_uri.to_owned())
        })?;
    if separator + 1 == manifest_uri.len() {
        return Err(GaussianLodPackageSourceError::MissingManifestPath(
            manifest_uri.to_owned(),
        ));
    }
    let base_url = &manifest_uri[..=separator];
    validate_base_url(base_url).map_err(GaussianLodPackageSourceError::InvalidHttpUri)?;
    Ok(GaussianLodPackageSource::url(base_url))
}

#[cfg(all(test, not(target_arch = "wasm32")))]
mod tests {
    use super::*;

    #[test]
    fn derives_native_and_http_roots_without_silent_url_normalization() {
        assert_eq!(
            GaussianLodPackageSource::try_from_manifest_uri(
                "https://cdn.example/scene/model.gsplatlod"
            )
            .unwrap(),
            GaussianLodPackageSource::url("https://cdn.example/scene/")
        );
        assert_eq!(
            GaussianLodPackageSource::try_from_manifest_uri("assets/scene/model.gsplatlod")
                .unwrap(),
            GaussianLodPackageSource::native_directory("assets/scene")
        );
        assert_eq!(
            GaussianLodPackageSource::try_from_manifest_uri("model.gsplatlod").unwrap(),
            GaussianLodPackageSource::native_directory(".")
        );
        assert!(
            GaussianLodPackageSource::try_from_manifest_uri(
                "https://cdn.example/x%20y/model.gsplatlod"
            )
            .is_err()
        );
        assert!(
            GaussianLodPackageSource::try_from_manifest_uri(
                "https://cdn.example/model.gsplatlod?signature=secret"
            )
            .is_err()
        );
        assert!(
            GaussianLodPackageSource::try_from_manifest_uri("ftp://example/model.gsplatlod")
                .is_err()
        );
    }
}

#[cfg(all(test, target_arch = "wasm32"))]
mod browser_tests {
    use super::*;
    use wasm_bindgen_test::wasm_bindgen_test;

    wasm_bindgen_test::wasm_bindgen_test_configure!(run_in_browser);

    #[wasm_bindgen_test]
    fn relative_manifest_uses_same_origin_http_source() {
        let source =
            GaussianLodPackageSource::try_from_manifest_uri("assets/scene/model.gsplatlod")
                .unwrap();
        let GaussianLodPackageSource::Url { base_url } = source else {
            panic!("browser-relative packages must use HTTP range transport");
        };
        assert!(base_url.starts_with("http://") || base_url.starts_with("https://"));
        assert!(base_url.ends_with("/assets/scene/"));
    }
}

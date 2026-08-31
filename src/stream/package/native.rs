use std::{
    collections::HashMap,
    path::{Path, PathBuf},
    sync::{Arc, Mutex},
};

use bevy::prelude::{App, ResMut};

use crate::stream::{
    http::{HttpRangePageTransport, HttpRangeTransportError, NativeUreqHttpClient},
    persistent_cache::{
        NativePersistentCacheConfig, NativePersistentCacheService, PersistentCachePageIdentities,
        PersistentCacheTransportError, SharedPersistentCachePageTransport,
    },
    transport::{
        LodPageId, LodPageTransport, LodPageTransportFailure, ManifestPageLocations,
        NativeFilePageTransport, NativeFileTransportError, PagePoll, PageRequest,
    },
};

use super::{
    GaussianLodPackageConfig, GaussianLodPackageError, GaussianLodPackageManager,
    GaussianLodPackageSource, GaussianLodPackageTransportError, GaussianStreamingSettings,
    map_package_poll, package_cache_name, package_http_config, validated_cache_namespace,
};

pub(super) type PackageManagerParam<'w> = ResMut<'w, GaussianLodPackageManager>;

pub(super) fn init_package_manager(app: &mut App) {
    app.init_resource::<GaussianLodPackageManager>();
}

#[derive(Default)]
pub(super) struct PackageCacheRegistry {
    caches: HashMap<PathBuf, RegisteredNativeCache>,
}

struct RegisteredNativeCache {
    config: NativePersistentCacheConfig,
    max_pending_operations: u32,
    service: Arc<Mutex<NativePersistentCacheService>>,
}

impl PackageCacheRegistry {
    pub(super) fn prune_unused(&mut self) {
        self.caches
            .retain(|_, cache| Arc::strong_count(&cache.service) > 1);
    }

    fn shared_cache(
        &mut self,
        config: NativePersistentCacheConfig,
        max_pending_operations: u32,
    ) -> Result<Arc<Mutex<NativePersistentCacheService>>, GaussianLodPackageError> {
        if let Some(registered) = self.caches.get(&config.root) {
            if registered.config != config
                || registered.max_pending_operations != max_pending_operations
            {
                return Err(GaussianLodPackageError::PersistentCacheConfigConflict {
                    key: config.root.display().to_string(),
                });
            }
            return Ok(registered.service.clone());
        }

        let service = Arc::new(Mutex::new(
            NativePersistentCacheService::spawn_from_config(config.clone(), max_pending_operations)
                .map_err(|error| GaussianLodPackageError::PersistentCache(error.to_string()))?,
        ));
        self.caches.insert(
            config.root.clone(),
            RegisteredNativeCache {
                config,
                max_pending_operations,
                service: service.clone(),
            },
        );
        Ok(service)
    }

    #[cfg(all(test, feature = "sort_radix", not(feature = "buffer_texture")))]
    pub(super) fn len(&self) -> usize {
        self.caches.len()
    }

    #[cfg(all(test, feature = "sort_radix", not(feature = "buffer_texture")))]
    pub(super) fn is_empty(&self) -> bool {
        self.caches.is_empty()
    }
}

type NativeHttpPageTransport = HttpRangePageTransport<NativeUreqHttpClient>;

/// Native package transport set with one stable ticket/error ABI.
#[allow(clippy::enum_variant_names)]
pub(super) enum PackagePageTransport {
    NativeFile(NativeFilePageTransport),
    NativeFileCached(SharedPersistentCachePageTransport<NativeFilePageTransport>),
    NativeHttp(NativeHttpPageTransport),
    NativeHttpCached(SharedPersistentCachePageTransport<NativeHttpPageTransport>),
}

impl LodPageTransport for PackagePageTransport {
    type Ticket = u64;
    type Error = GaussianLodPackageTransportError;

    fn begin(&mut self, request: PageRequest) -> Result<Self::Ticket, Self::Error> {
        match self {
            Self::NativeFile(transport) => transport.begin(request).map_err(native_file_error),
            Self::NativeFileCached(transport) => {
                transport.begin(request).map_err(native_file_cache_error)
            }
            Self::NativeHttp(transport) => transport.begin(request).map_err(http_error),
            Self::NativeHttpCached(transport) => transport.begin(request).map_err(http_cache_error),
        }
    }

    fn poll(&mut self, ticket: &Self::Ticket) -> PagePoll<Self::Error> {
        match self {
            Self::NativeFile(transport) => {
                map_package_poll(transport.poll(ticket), native_file_error)
            }
            Self::NativeFileCached(transport) => {
                map_package_poll(transport.poll(ticket), native_file_cache_error)
            }
            Self::NativeHttp(transport) => map_package_poll(transport.poll(ticket), http_error),
            Self::NativeHttpCached(transport) => {
                map_package_poll(transport.poll(ticket), http_cache_error)
            }
        }
    }

    fn cancel(&mut self, ticket: &Self::Ticket) {
        match self {
            Self::NativeFile(transport) => transport.cancel(ticket),
            Self::NativeFileCached(transport) => transport.cancel(ticket),
            Self::NativeHttp(transport) => transport.cancel(ticket),
            Self::NativeHttpCached(transport) => transport.cancel(ticket),
        }
    }

    fn classify_error(error: &Self::Error) -> LodPageTransportFailure {
        error.runtime_failure()
    }
}

fn native_file_error(error: NativeFileTransportError) -> GaussianLodPackageTransportError {
    GaussianLodPackageTransportError::NativeFile(error.to_string())
}

fn http_error(error: HttpRangeTransportError) -> GaussianLodPackageTransportError {
    GaussianLodPackageTransportError::Http(error.to_string())
}

fn native_file_cache_error(
    error: PersistentCacheTransportError<NativeFileTransportError>,
) -> GaussianLodPackageTransportError {
    match error {
        PersistentCacheTransportError::Upstream(error) => native_file_error(error),
        error => GaussianLodPackageTransportError::PersistentCache(error.to_string()),
    }
}

fn http_cache_error(
    error: PersistentCacheTransportError<HttpRangeTransportError>,
) -> GaussianLodPackageTransportError {
    match error {
        PersistentCacheTransportError::Upstream(error) => http_error(error),
        error => GaussianLodPackageTransportError::PersistentCache(error.to_string()),
    }
}

impl PackagePageTransport {
    pub(super) fn invalidate_cached_page(
        &mut self,
        page: LodPageId,
    ) -> Result<(), GaussianLodPackageError> {
        match self {
            Self::NativeFileCached(transport) => transport
                .invalidate_page(page)
                .map_err(|error| GaussianLodPackageError::PersistentCache(error.to_string())),
            Self::NativeHttpCached(transport) => transport
                .invalidate_page(page)
                .map_err(|error| GaussianLodPackageError::PersistentCache(error.to_string())),
            Self::NativeFile(_) | Self::NativeHttp(_) => Ok(()),
        }
    }

    pub(super) fn maintain_cache(&mut self) -> Result<bool, GaussianLodPackageError> {
        match self {
            Self::NativeFileCached(transport) => transport
                .maintain_cache()
                .map_err(|error| GaussianLodPackageError::PersistentCache(error.to_string())),
            Self::NativeHttpCached(transport) => transport
                .maintain_cache()
                .map_err(|error| GaussianLodPackageError::PersistentCache(error.to_string())),
            Self::NativeFile(_) | Self::NativeHttp(_) => Ok(true),
        }
    }
}

#[cfg(all(test, feature = "sort_radix", not(feature = "buffer_texture")))]
impl PackagePageTransport {
    pub(super) fn shared_native_cache_service(
        &self,
    ) -> Option<&Arc<Mutex<NativePersistentCacheService>>> {
        match self {
            Self::NativeFileCached(transport) => Some(transport.shared_cache()),
            Self::NativeHttpCached(transport) => Some(transport.shared_cache()),
            Self::NativeFile(_) | Self::NativeHttp(_) => None,
        }
    }
}

pub(super) fn package_page_transport(
    manifest: &crate::GaussianLodManifest,
    source: &GaussianLodPackageSource,
    config: &GaussianLodPackageConfig,
    streaming: &GaussianStreamingSettings,
    caches: &mut PackageCacheRegistry,
) -> Result<PackagePageTransport, GaussianLodPackageError> {
    let identities = if streaming.persistent_cache {
        Some(
            PersistentCachePageIdentities::from_validated_manifest(manifest)
                .map_err(|error| GaussianLodPackageError::PersistentCache(error.to_string()))?,
        )
    } else {
        None
    };

    match source {
        GaussianLodPackageSource::NativeDirectory { root } => {
            let root = validate_native_root(root)?;
            let locations = ManifestPageLocations::from_validated_manifest(manifest)
                .map_err(|error| GaussianLodPackageError::NativeTransport(error.to_string()))?;
            let upstream = NativeFilePageTransport::with_max_encoded_page_bytes(
                root,
                locations,
                streaming.effective_max_encoded_page_bytes(),
            )
            .map_err(|error| GaussianLodPackageError::NativeTransport(error.to_string()))?;
            if let Some(identities) = identities {
                let cache = caches.shared_cache(
                    native_cache_config(manifest, config, streaming)?,
                    streaming.max_concurrent_requests.saturating_mul(2).max(1),
                )?;
                Ok(PackagePageTransport::NativeFileCached(
                    SharedPersistentCachePageTransport::new(upstream, cache, identities),
                ))
            } else {
                Ok(PackagePageTransport::NativeFile(upstream))
            }
        }
        GaussianLodPackageSource::Url { base_url } => {
            let locations = ManifestPageLocations::from_validated_manifest(manifest)
                .map_err(|error| GaussianLodPackageError::HttpTransport(error.to_string()))?;
            let http_config = package_http_config(base_url, streaming)?;
            let client = NativeUreqHttpClient::with_max_workers(
                http_config.request_timeout,
                streaming.max_concurrent_requests,
            )
            .map_err(|error| GaussianLodPackageError::HttpTransport(error.to_string()))?;
            // Package HTTP transports own only byte-range, response-shape, and
            // immutable-object validation. The runtime's bounded page
            // preprocessor is the single owner of checksum, codec, manifest,
            // and support-bound validation.
            let upstream = HttpRangePageTransport::new(http_config, locations, client)
                .map_err(|error| GaussianLodPackageError::HttpTransport(error.to_string()))?;
            if let Some(identities) = identities {
                let cache = caches.shared_cache(
                    native_cache_config(manifest, config, streaming)?,
                    streaming.max_concurrent_requests.saturating_mul(2).max(1),
                )?;
                Ok(PackagePageTransport::NativeHttpCached(
                    SharedPersistentCachePageTransport::new(upstream, cache, identities),
                ))
            } else {
                Ok(PackagePageTransport::NativeHttp(upstream))
            }
        }
    }
}

pub(super) fn validate_cache_config(
    config: &GaussianLodPackageConfig,
) -> Result<(), GaussianLodPackageError> {
    validated_cache_namespace(config)?;
    validate_cache_root_config(config)?;
    Ok(())
}

fn validate_cache_root_config(
    config: &GaussianLodPackageConfig,
) -> Result<&str, GaussianLodPackageError> {
    let root = config
        .persistent_cache_root
        .as_deref()
        .ok_or(GaussianLodPackageError::MissingPersistentCacheRoot)?;
    if root.is_empty() || root.split_once("://").is_some() || !Path::new(root).is_absolute() {
        return Err(GaussianLodPackageError::InvalidPersistentCacheRoot(
            root.to_owned(),
        ));
    }
    Ok(root)
}

fn native_cache_config(
    manifest: &crate::GaussianLodManifest,
    config: &GaussianLodPackageConfig,
    streaming: &GaussianStreamingSettings,
) -> Result<NativePersistentCacheConfig, GaussianLodPackageError> {
    let root = validate_cache_root_config(config)?;
    Ok(NativePersistentCacheConfig {
        root: PathBuf::from(root).join(package_cache_name(manifest, config)?),
        max_bytes: streaming.max_compressed_cache_bytes,
        max_entries: config.persistent_cache_max_entries,
    })
}

pub(super) fn validate_native_root(root: &str) -> Result<&Path, GaussianLodPackageError> {
    if root.is_empty() {
        return Err(GaussianLodPackageError::EmptyNativeRoot);
    }
    if let Some(scheme) = root.split_once("://").map(|(scheme, _)| scheme) {
        return Err(GaussianLodPackageError::UnsupportedUrlScheme(
            scheme.to_owned(),
        ));
    }
    Ok(Path::new(root))
}

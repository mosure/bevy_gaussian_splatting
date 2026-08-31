use std::{cell::RefCell, collections::HashMap, rc::Rc};

use bevy::prelude::{App, NonSendMut};

use crate::stream::{
    http::{BrowserFetchHttpClient, HttpRangePageTransport, HttpRangeTransportError},
    persistent_cache::{
        BrowserPersistentCacheConfig, BrowserPersistentPageCache, PersistentCachePageIdentities,
        PersistentCacheTransportError, SharedBrowserPersistentCachePageTransport,
    },
    transport::{
        LodPageId, LodPageTransport, LodPageTransportFailure, ManifestPageLocations, PagePoll,
        PageRequest,
    },
};

use super::{
    GaussianLodPackageConfig, GaussianLodPackageError, GaussianLodPackageManager,
    GaussianLodPackageSource, GaussianLodPackageTransportError, GaussianStreamingSettings,
    map_package_poll, package_cache_name, package_http_config, validated_cache_namespace,
};

pub(super) type PackageManagerParam<'w> = NonSendMut<'w, GaussianLodPackageManager>;

pub(super) fn init_package_manager(app: &mut App) {
    app.insert_non_send(GaussianLodPackageManager::default());
}

#[derive(Default)]
pub(super) struct PackageCacheRegistry {
    caches: HashMap<String, RegisteredBrowserCache>,
}

struct RegisteredBrowserCache {
    config: BrowserPersistentCacheConfig,
    cache: Rc<RefCell<BrowserPersistentPageCache>>,
}

impl PackageCacheRegistry {
    pub(super) fn prune_unused(&mut self) {
        self.caches
            .retain(|_, cache| Rc::strong_count(&cache.cache) > 1);
    }

    fn shared_cache(
        &mut self,
        config: BrowserPersistentCacheConfig,
    ) -> Result<Rc<RefCell<BrowserPersistentPageCache>>, GaussianLodPackageError> {
        if let Some(registered) = self.caches.get(&config.cache_name) {
            if registered.config != config {
                return Err(GaussianLodPackageError::PersistentCacheConfigConflict {
                    key: config.cache_name.clone(),
                });
            }
            return Ok(registered.cache.clone());
        }

        let cache = BrowserPersistentPageCache::shared(config.clone())
            .map_err(|error| GaussianLodPackageError::PersistentCache(error.to_string()))?;
        self.caches.insert(
            config.cache_name.clone(),
            RegisteredBrowserCache {
                config,
                cache: cache.clone(),
            },
        );
        Ok(cache)
    }
}

type BrowserHttpPageTransport = HttpRangePageTransport<BrowserFetchHttpClient>;

/// Browser package transport set with one stable ticket/error ABI.
#[allow(clippy::enum_variant_names)]
pub(super) enum PackagePageTransport {
    BrowserHttp(BrowserHttpPageTransport),
    BrowserHttpCached(SharedBrowserPersistentCachePageTransport<BrowserHttpPageTransport>),
}

impl LodPageTransport for PackagePageTransport {
    type Ticket = u64;
    type Error = GaussianLodPackageTransportError;

    fn begin(&mut self, request: PageRequest) -> Result<Self::Ticket, Self::Error> {
        match self {
            Self::BrowserHttp(transport) => transport.begin(request).map_err(http_error),
            Self::BrowserHttpCached(transport) => {
                transport.begin(request).map_err(http_cache_error)
            }
        }
    }

    fn poll(&mut self, ticket: &Self::Ticket) -> PagePoll<Self::Error> {
        match self {
            Self::BrowserHttp(transport) => map_package_poll(transport.poll(ticket), http_error),
            Self::BrowserHttpCached(transport) => {
                map_package_poll(transport.poll(ticket), http_cache_error)
            }
        }
    }

    fn cancel(&mut self, ticket: &Self::Ticket) {
        match self {
            Self::BrowserHttp(transport) => transport.cancel(ticket),
            Self::BrowserHttpCached(transport) => transport.cancel(ticket),
        }
    }

    fn classify_error(error: &Self::Error) -> LodPageTransportFailure {
        error.runtime_failure()
    }
}

fn http_error(error: HttpRangeTransportError) -> GaussianLodPackageTransportError {
    GaussianLodPackageTransportError::Http(error.to_string())
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
            Self::BrowserHttpCached(transport) => transport
                .invalidate_page(page)
                .map_err(|error| GaussianLodPackageError::PersistentCache(error.to_string())),
            Self::BrowserHttp(_) => Ok(()),
        }
    }

    pub(super) fn maintain_cache(&mut self) -> Result<bool, GaussianLodPackageError> {
        match self {
            Self::BrowserHttpCached(transport) => transport
                .maintain_cache()
                .map_err(|error| GaussianLodPackageError::PersistentCache(error.to_string())),
            Self::BrowserHttp(_) => Ok(true),
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
        GaussianLodPackageSource::NativeDirectory { .. } => {
            Err(GaussianLodPackageError::NativeSourceUnsupportedInBrowser)
        }
        GaussianLodPackageSource::Url { base_url } => {
            let locations = ManifestPageLocations::from_validated_manifest(manifest)
                .map_err(|error| GaussianLodPackageError::HttpTransport(error.to_string()))?;
            let http_config = package_http_config(base_url, streaming)?;
            let client =
                BrowserFetchHttpClient::with_max_requests(streaming.max_concurrent_requests)
                    .map_err(|error| GaussianLodPackageError::HttpTransport(error.to_string()))?;
            // Package HTTP transports own only byte-range, response-shape, and
            // immutable-object validation. The runtime's bounded page
            // preprocessor is the single owner of checksum, codec, manifest,
            // and support-bound validation.
            let upstream = HttpRangePageTransport::new(http_config, locations, client)
                .map_err(|error| GaussianLodPackageError::HttpTransport(error.to_string()))?;
            if let Some(identities) = identities {
                let cache =
                    caches.shared_cache(browser_cache_config(manifest, config, streaming)?)?;
                Ok(PackagePageTransport::BrowserHttpCached(
                    SharedBrowserPersistentCachePageTransport::new(upstream, cache, identities),
                ))
            } else {
                Ok(PackagePageTransport::BrowserHttp(upstream))
            }
        }
    }
}

pub(super) fn validate_cache_config(
    config: &GaussianLodPackageConfig,
) -> Result<(), GaussianLodPackageError> {
    validated_cache_namespace(config).map(|_| ())
}

fn browser_cache_config(
    manifest: &crate::GaussianLodManifest,
    config: &GaussianLodPackageConfig,
    streaming: &GaussianStreamingSettings,
) -> Result<BrowserPersistentCacheConfig, GaussianLodPackageError> {
    Ok(BrowserPersistentCacheConfig {
        cache_name: package_cache_name(manifest, config)?,
        max_bytes: streaming.max_compressed_cache_bytes,
        max_entries: config.persistent_cache_max_entries,
        max_pending_operations: streaming.max_concurrent_requests.saturating_mul(2).max(1),
    })
}

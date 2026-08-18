use std::sync::atomic::{AtomicU64, Ordering};

use super::super::*;
use super::*;
use crate::{
    gaussian::formats::{
        planar_3d::{Gaussian3d, PlanarGaussian3d},
        planar_3d_chunked::LodPageStorage,
        planar_3d_lod::CpuGaussianLodBuilder,
    },
    io::lod::encode_page,
    stream::transport::{MemoryPageTransport, PageRequestPriority},
};

static NEXT_ROOT: AtomicU64 = AtomicU64::new(1);
const LOCK_HOLDER_ROOT_ENV: &str = "BGS_PERSISTENT_CACHE_LOCK_HOLDER_ROOT";
const LOCK_HOLDER_READY: &str = ".lock-holder-ready";
const LOCK_HOLDER_RELEASE: &str = ".lock-holder-release";

struct PendingCountingTransport {
    next_ticket: u64,
    canceled: Arc<AtomicU64>,
}

impl LodPageTransport for PendingCountingTransport {
    type Ticket = u64;
    type Error = std::convert::Infallible;

    fn begin(&mut self, _request: PageRequest) -> Result<Self::Ticket, Self::Error> {
        let ticket = self.next_ticket;
        self.next_ticket += 1;
        Ok(ticket)
    }

    fn poll(&mut self, _ticket: &Self::Ticket) -> PagePoll<Self::Error> {
        PagePoll::Pending
    }

    fn cancel(&mut self, _ticket: &Self::Ticket) {
        self.canceled.fetch_add(1, Ordering::Relaxed);
    }
}

struct TestRoot(PathBuf);

impl TestRoot {
    fn new() -> Self {
        let unique = NEXT_ROOT.fetch_add(1, Ordering::Relaxed);
        let path = std::env::temp_dir().join(format!(
            "bevy-gaussian-lod-persistent-cache-{}-{unique}",
            std::process::id()
        ));
        fs::create_dir(&path).unwrap();
        Self(path)
    }
}

impl Drop for TestRoot {
    fn drop(&mut self) {
        let _ = fs::remove_dir_all(&self.0);
    }
}

fn identities(count: usize) -> PersistentCachePageIdentities {
    let mut gaussian = Gaussian3d::default();
    gaussian.rotation.rotation = [1.0, 0.0, 0.0, 0.0];
    let source: PlanarGaussian3d = vec![gaussian; count.max(1)].into();
    let mut built = CpuGaussianLodBuilder::default().build(&source).unwrap();
    for descriptor in &mut built.manifest.pages {
        descriptor.storage = Some(LodPageStorage {
            uri: format!("{}.gspage", descriptor.id.0),
            byte_range: None,
            encoded_len: 4,
        });
    }
    PersistentCachePageIdentities::from_manifest(&built.manifest).unwrap()
}

fn encoded_transport_fixture() -> (
    PersistentCachePageIdentities,
    PersistentCachePageIdentity,
    PagePayload,
) {
    let mut gaussian = Gaussian3d::default();
    gaussian.rotation.rotation = [1.0, 0.0, 0.0, 0.0];
    let source: PlanarGaussian3d = vec![gaussian].into();
    let mut built = CpuGaussianLodBuilder::default().build(&source).unwrap();
    let page = built.pages[0].clone();
    let encoded = encode_page(&page).unwrap();
    let descriptor = built
        .manifest
        .pages
        .iter_mut()
        .find(|descriptor| descriptor.id == page.id)
        .unwrap();
    descriptor.storage = Some(LodPageStorage {
        uri: format!("{}.gspage", page.id.0),
        byte_range: None,
        encoded_len: encoded.len() as u64,
    });
    let identities = PersistentCachePageIdentities::from_manifest(&built.manifest).unwrap();
    let identity = identities.get(page.id).unwrap().clone();
    let payload = PagePayload::new(page.id, encoded);
    (identities, identity, payload)
}

fn cache(root: &TestRoot, max_entries: u32) -> NativePersistentPageCache {
    NativePersistentPageCache::open(NativePersistentCacheConfig {
        root: root.0.clone(),
        max_bytes: u64::from(max_entries) * (CACHE_HEADER_LEN as u64 + 4),
        max_entries,
    })
    .unwrap()
}

fn transport_cache(root: &TestRoot) -> NativePersistentPageCache {
    NativePersistentPageCache::open(NativePersistentCacheConfig {
        root: root.0.clone(),
        max_bytes: 1024 * 1024,
        max_entries: 4,
    })
    .unwrap()
}

#[test]
fn key_is_content_identity_not_url_identity() {
    let mut gaussian = Gaussian3d::default();
    gaussian.rotation.rotation = [1.0, 0.0, 0.0, 0.0];
    let source: PlanarGaussian3d = vec![gaussian].into();
    let mut first = CpuGaussianLodBuilder::default()
        .build(&source)
        .unwrap()
        .manifest;
    let mut second = first.clone();
    first.pages[0].storage = Some(LodPageStorage {
        uri: "mirror-a/page".to_owned(),
        byte_range: None,
        encoded_len: 4,
    });
    second.pages[0].storage = Some(LodPageStorage {
        uri: "mirror-b/page".to_owned(),
        byte_range: None,
        encoded_len: 4,
    });
    let first = PersistentCachePageIdentities::from_manifest(&first).unwrap();
    let second = PersistentCachePageIdentities::from_manifest(&second).unwrap();
    let page = first.entries.keys().next().copied().unwrap();
    assert_eq!(
        first.get(page).unwrap().key(),
        second.get(page).unwrap().key()
    );

    let changed = first
        .get(page)
        .unwrap()
        .package
        .clone()
        .with_package_version("deployment-2")
        .unwrap();
    let changed = PersistentCachePageIdentity {
        package: changed,
        ..first.get(page).unwrap().clone()
    };
    assert_ne!(first.get(page).unwrap().key(), changed.key());
}

#[test]
fn cache_survives_reopen_for_offline_reuse() {
    let root = TestRoot::new();
    let identities = identities(1);
    let identity = identities.entries.values().next().unwrap();
    let payload = PagePayload::new(identity.page_id, vec![1, 2, 3, 4]);
    {
        let mut cache = cache(&root, 2);
        assert!(matches!(
            cache.insert(identity, &payload).unwrap(),
            PersistentCacheInsert::Written { .. }
        ));
    }
    let mut reopened = cache(&root, 2);
    assert_eq!(
        reopened.lookup(identity).unwrap(),
        PersistentCacheLookup::Hit(payload)
    );
    assert_eq!(reopened.stats().hits, 1);
}

#[test]
fn startup_scan_discards_excess_records_before_index_allocation() {
    let root = TestRoot::new();
    let base = identities(1).entries.values().next().unwrap().clone();
    let records = (1..=5)
        .map(|id| PersistentCachePageIdentity {
            page_id: LodPageId(id),
            content_hash: id * 31,
            ..base.clone()
        })
        .collect::<Vec<_>>();
    {
        let mut initial = cache(&root, 5);
        for identity in &records {
            initial
                .insert(
                    identity,
                    &PagePayload::new(identity.page_id, vec![identity.page_id.0 as u8; 4]),
                )
                .unwrap();
        }
    }

    let reopened = cache(&root, 2);
    assert_eq!(reopened.stats().entries, 2);
    assert_eq!(reopened.stats().evictions, 3);
    assert_eq!(
        fs::read_dir(&root.0)
            .unwrap()
            .filter_map(Result::ok)
            .filter(|entry| {
                entry.path().extension().and_then(|value| value.to_str()) == Some(CACHE_EXTENSION)
            })
            .count(),
        2
    );
}

#[test]
fn checksum_corruption_is_removed_and_refetched() {
    let root = TestRoot::new();
    let identities = identities(1);
    let identity = identities.entries.values().next().unwrap();
    let payload = PagePayload::new(identity.page_id, vec![1, 2, 3, 4]);
    let mut cache = cache(&root, 2);
    cache.insert(identity, &payload).unwrap();
    let key = identity.key().unwrap();
    let path = root.0.join(key.file_name());
    let mut bytes = fs::read(&path).unwrap();
    *bytes.last_mut().unwrap() ^= 0xff;
    fs::write(&path, bytes).unwrap();
    assert!(matches!(
        cache.lookup(identity).unwrap(),
        PersistentCacheLookup::CorruptionRecovered(PersistentCacheCorruption {
            reason: PersistentCacheCorruptionReason::PayloadChecksumMismatch { .. },
            ..
        })
    ));
    assert!(!path.exists());
    assert_eq!(cache.stats().corruptions_recovered, 1);
}

#[test]
fn byte_and_entry_budget_evict_deterministic_lru() {
    let root = TestRoot::new();
    let base = identities(1).entries.values().next().unwrap().clone();
    let identities = (1..=3)
        .map(|id| PersistentCachePageIdentity {
            page_id: LodPageId(id),
            content_hash: id * 17,
            ..base.clone()
        })
        .collect::<Vec<_>>();
    let mut cache = cache(&root, 2);
    for identity in &identities[..2] {
        cache
            .insert(
                identity,
                &PagePayload::new(identity.page_id, vec![identity.page_id.0 as u8; 4]),
            )
            .unwrap();
    }
    assert!(matches!(
        cache.lookup(&identities[0]).unwrap(),
        PersistentCacheLookup::Hit(_)
    ));
    let inserted = cache
        .insert(
            &identities[2],
            &PagePayload::new(identities[2].page_id, vec![3; 4]),
        )
        .unwrap();
    let PersistentCacheInsert::Written { evicted } = inserted else {
        panic!("expected write");
    };
    assert_eq!(evicted, vec![identities[1].key().unwrap()]);
    assert!(cache.contains(&identities[0]));
    assert!(!cache.contains(&identities[1]));
    assert!(cache.contains(&identities[2]));
    assert_eq!(cache.stats().entries, 2);
    assert!(cache.stats().bytes <= cache.config().max_bytes);
}

#[test]
fn cache_first_transport_reuses_bytes_when_upstream_is_offline() {
    let root = TestRoot::new();
    let (identities, identity, payload) = encoded_transport_fixture();
    let mut primed = transport_cache(&root);
    primed.insert(&identity, &payload).unwrap();
    drop(primed);

    let upstream = MemoryPageTransport::default();
    let cache = transport_cache(&root);
    let mut transport = PersistentCachePageTransport::new(upstream, cache, identities);
    let mut request = PageRequest::new(identity.page_id, PageRequestPriority::visible(1));
    request.expected_bytes = Some(identity.encoded_len);
    let ticket = transport.begin(request).unwrap();
    assert_eq!(transport.poll(&ticket), PagePoll::Ready(payload));
}

#[test]
fn cache_first_transport_populates_after_miss() {
    let root = TestRoot::new();
    let (identities, identity, payload) = encoded_transport_fixture();
    let mut upstream = MemoryPageTransport::default();
    upstream.insert(identity.page_id, payload.bytes.clone());
    let persistent_cache = transport_cache(&root);
    let mut transport =
        PersistentCachePageTransport::new(upstream, persistent_cache, identities.clone());
    let mut request = PageRequest::new(identity.page_id, PageRequestPriority::visible(1));
    request.expected_bytes = Some(identity.encoded_len);
    let ticket = transport.begin(request).unwrap();
    assert_eq!(transport.poll(&ticket), PagePoll::Ready(payload.clone()));
    drop(transport);

    let mut reopened = transport_cache(&root);
    assert_eq!(
        reopened.lookup(&identity).unwrap(),
        PersistentCacheLookup::Hit(payload)
    );
}

#[test]
fn downstream_rejection_invalidates_cache_hit_before_upstream_repair() {
    let root = TestRoot::new();
    let (identities, identity, valid_payload) = encoded_transport_fixture();
    let mut corrupt_bytes = valid_payload.bytes.clone();
    *corrupt_bytes.last_mut().unwrap() ^= 0x01;
    let corrupt_payload = PagePayload::new(identity.page_id, corrupt_bytes);
    let mut primed = transport_cache(&root);
    primed.insert(&identity, &corrupt_payload).unwrap();
    drop(primed);

    let mut upstream = MemoryPageTransport::default();
    upstream.insert(identity.page_id, valid_payload.bytes.clone());
    let mut transport =
        PersistentCachePageTransport::new(upstream, transport_cache(&root), identities.clone());
    let mut request = PageRequest::new(identity.page_id, PageRequestPriority::visible(1));
    request.expected_bytes = Some(identity.encoded_len);
    let ticket = transport.begin(request).unwrap();
    assert_eq!(transport.poll(&ticket), PagePoll::Ready(corrupt_payload));
    assert!(transport.invalidate_page(identity.page_id).unwrap());

    let ticket = transport.begin(request).unwrap();
    assert_eq!(
        transport.poll(&ticket),
        PagePoll::Ready(valid_payload.clone())
    );
    drop(transport);

    let mut reopened = transport_cache(&root);
    assert_eq!(
        reopened.lookup(&identity),
        Ok(PersistentCacheLookup::Hit(valid_payload))
    );
}

#[test]
fn downstream_rejection_invalidation_removes_corrupt_upstream_payload() {
    let root = TestRoot::new();
    let (identities, identity, valid_payload) = encoded_transport_fixture();
    let mut corrupt_bytes = valid_payload.bytes.clone();
    *corrupt_bytes.last_mut().unwrap() ^= 0x01;
    let mut upstream = MemoryPageTransport::default();
    upstream.insert(identity.page_id, corrupt_bytes);
    let mut transport =
        PersistentCachePageTransport::new(upstream, transport_cache(&root), identities.clone());
    let mut request = PageRequest::new(identity.page_id, PageRequestPriority::visible(1));
    request.expected_bytes = Some(identity.encoded_len);
    let ticket = transport.begin(request).unwrap();
    let PagePoll::Ready(corrupt_payload) = transport.poll(&ticket) else {
        panic!("encoded payload must reach the downstream preprocessor")
    };
    assert_ne!(corrupt_payload, valid_payload);
    assert!(transport.cache().contains(&identity));
    assert!(transport.invalidate_page(identity.page_id).unwrap());
    drop(transport);
    assert!(!transport_cache(&root).contains(&identity));

    let mut corrected = MemoryPageTransport::default();
    corrected.insert(identity.page_id, valid_payload.bytes.clone());
    let mut transport =
        PersistentCachePageTransport::new(corrected, transport_cache(&root), identities);
    let mut request = PageRequest::new(identity.page_id, PageRequestPriority::visible(1));
    request.expected_bytes = Some(identity.encoded_len);
    let ticket = transport.begin(request).unwrap();
    assert_eq!(transport.poll(&ticket), PagePoll::Ready(valid_payload));
}

#[test]
fn shared_transport_invalidation_is_bounded_and_removes_rejected_page() {
    let root = TestRoot::new();
    let (identities, identity, valid_payload) = encoded_transport_fixture();
    let mut corrupt_bytes = valid_payload.bytes.clone();
    *corrupt_bytes.last_mut().unwrap() ^= 0x01;
    let mut upstream = MemoryPageTransport::default();
    upstream.insert(identity.page_id, corrupt_bytes);
    let service = NativePersistentCacheService::spawn(transport_cache(&root), 8).unwrap();
    let shared = Arc::new(Mutex::new(service));
    let mut transport =
        SharedPersistentCachePageTransport::new(upstream, shared.clone(), identities.clone());
    let mut request = PageRequest::new(identity.page_id, PageRequestPriority::visible(1));
    request.expected_bytes = Some(identity.encoded_len);
    let ticket = transport.begin(request).unwrap();
    let payload = loop {
        match transport.poll(&ticket) {
            PagePoll::Pending => std::thread::yield_now(),
            PagePoll::Ready(payload) => break payload,
            PagePoll::Failed(error) => panic!("shared cache transport failed: {error}"),
        }
    };
    assert_ne!(payload, valid_payload);

    transport.invalidate_page(identity.page_id).unwrap();
    for _ in 0..10_000 {
        if transport.maintain_cache().unwrap() {
            break;
        }
        std::thread::yield_now();
    }
    assert!(transport.maintain_cache().unwrap());
    let lookup = shared
        .lock()
        .unwrap()
        .begin_lookup(identities.validation(identity.page_id).unwrap())
        .unwrap()
        .recv_timeout(std::time::Duration::from_secs(10))
        .unwrap()
        .unwrap();
    assert_eq!(lookup, PersistentCacheLookup::Miss);
}

#[test]
fn owned_cache_transport_drop_and_into_parts_cancel_upstream_tickets() {
    for extract_parts in [false, true] {
        let root = TestRoot::new();
        let identities = identities(1);
        let identity = identities.entries.values().next().unwrap().clone();
        let canceled = Arc::new(AtomicU64::new(0));
        let upstream = PendingCountingTransport {
            next_ticket: 1,
            canceled: canceled.clone(),
        };
        let mut transport =
            PersistentCachePageTransport::new(upstream, cache(&root, 2), identities);
        for _ in 0..2 {
            let mut request = PageRequest::new(identity.page_id, PageRequestPriority::visible(1));
            request.expected_bytes = Some(identity.encoded_len);
            transport.begin(request).unwrap();
        }
        if extract_parts {
            let (_upstream, _cache) = transport.into_parts();
        } else {
            drop(transport);
        }
        assert_eq!(canceled.load(Ordering::Relaxed), 2);
    }
}

#[test]
fn shared_cache_transport_never_waits_for_blocked_filesystem_worker() {
    let root = TestRoot::new();
    let (identities, identity, payload) = encoded_transport_fixture();
    let mut upstream = MemoryPageTransport::default();
    upstream.insert(identity.page_id, payload.bytes.clone());
    let service = NativePersistentCacheService::spawn(transport_cache(&root), 4).unwrap();
    let (release, blocked) = std::sync::mpsc::sync_channel(1);
    service.block_until(blocked).unwrap();
    let shared = Arc::new(Mutex::new(service));
    let mut transport = SharedPersistentCachePageTransport::new(upstream, shared, identities);
    let mut request = PageRequest::new(identity.page_id, PageRequestPriority::visible(1));
    request.expected_bytes = Some(identity.encoded_len);

    let started = std::time::Instant::now();
    let ticket = transport.begin(request).unwrap();
    assert!(started.elapsed() < std::time::Duration::from_millis(100));
    assert!(matches!(transport.poll(&ticket), PagePoll::Pending));
    release.send(()).unwrap();
    for _ in 0..10_000 {
        match transport.poll(&ticket) {
            PagePoll::Pending => std::thread::yield_now(),
            PagePoll::Ready(actual) => {
                assert_eq!(actual, payload);
                return;
            }
            PagePoll::Failed(error) => panic!("shared cache transport failed: {error}"),
        }
    }
    panic!("shared cache service did not complete")
}

#[test]
fn native_cache_service_is_process_lifetime_single_writer_per_root() {
    let root = TestRoot::new();
    let canonical = root.0.join("canonical");
    let alias_parent = root.0.join("alias-parent");
    fs::create_dir_all(&alias_parent).unwrap();
    let config = NativePersistentCacheConfig {
        root: canonical.clone(),
        max_bytes: 4096,
        max_entries: 4,
    };
    let first = NativePersistentCacheService::spawn_from_config(config.clone(), 4).unwrap();
    let second = NativePersistentCacheService::spawn_from_config(
        NativePersistentCacheConfig {
            root: alias_parent.join("..").join("canonical"),
            ..config.clone()
        },
        4,
    )
    .unwrap();
    assert!(Arc::ptr_eq(&first.inner, &second.inner));
    assert!(matches!(
        NativePersistentCacheService::spawn_from_config(config, 5),
        Err(PersistentCacheError::CacheServiceConfigConflict(_))
    ));
    drop((first, second));
    assert!(
        native_persistent_cache_services()
            .lock()
            .unwrap()
            .values()
            .any(|registered| Arc::strong_count(&registered.service.inner) >= 1)
    );
}

#[test]
fn native_cache_service_recovers_after_external_lock_release() {
    if let Some(root) = std::env::var_os(LOCK_HOLDER_ROOT_ENV) {
        let root = PathBuf::from(root);
        let cache = NativePersistentPageCache::open(NativePersistentCacheConfig {
            root: root.clone(),
            max_bytes: 4096,
            max_entries: 4,
        })
        .unwrap();
        fs::write(root.join(LOCK_HOLDER_READY), b"ready").unwrap();
        let deadline = std::time::Instant::now() + std::time::Duration::from_secs(30);
        while !root.join(LOCK_HOLDER_RELEASE).exists() {
            assert!(
                std::time::Instant::now() < deadline,
                "parent did not release child cache lock"
            );
            std::thread::sleep(std::time::Duration::from_millis(10));
        }
        drop(cache);
        return;
    }

    let root = TestRoot::new();
    let config = NativePersistentCacheConfig {
        root: root.0.clone(),
        max_bytes: 4096,
        max_entries: 4,
    };
    let mut child = std::process::Command::new(std::env::current_exe().unwrap())
            .arg("--exact")
            .arg(
                "stream::persistent_cache::native::tests::native_cache_service_recovers_after_external_lock_release",
            )
            .arg("--nocapture")
            .env(LOCK_HOLDER_ROOT_ENV, &root.0)
            .stdin(std::process::Stdio::null())
            .stdout(std::process::Stdio::null())
            .spawn()
            .unwrap();
    let ready = root.0.join(LOCK_HOLDER_READY);
    let deadline = std::time::Instant::now() + std::time::Duration::from_secs(10);
    while !ready.exists() {
        assert!(
            std::time::Instant::now() < deadline,
            "child did not acquire cache lock"
        );
        assert!(
            child.try_wait().unwrap().is_none(),
            "lock holder exited early"
        );
        std::thread::sleep(std::time::Duration::from_millis(10));
    }

    let identities = identities(1);
    let page_id = *identities.entries.keys().next().unwrap();
    let validation = identities.validation(page_id).unwrap();
    let first = NativePersistentCacheService::spawn_from_config(config.clone(), 4).unwrap();
    let initial_failure = first
        .begin_lookup(validation.clone())
        .unwrap()
        .recv_timeout(std::time::Duration::from_secs(10))
        .unwrap()
        .unwrap_err();
    assert!(matches!(
        initial_failure,
        PersistentCacheError::CacheRootAlreadyOwned(_)
    ));

    fs::write(root.0.join(LOCK_HOLDER_RELEASE), b"release").unwrap();
    assert!(child.wait().unwrap().success());

    let reopened = NativePersistentCacheService::spawn_from_config(config, 4).unwrap();
    assert!(Arc::ptr_eq(&first.inner, &reopened.inner));
    assert_eq!(
        reopened
            .begin_lookup(validation)
            .unwrap()
            .recv_timeout(std::time::Duration::from_secs(10))
            .unwrap()
            .unwrap(),
        PersistentCacheLookup::Miss
    );
}

use wasm_bindgen::JsCast as _;
use wasm_bindgen_test::wasm_bindgen_test;

use super::super::browser_contract::*;
use super::super::*;
use super::*;
use crate::gaussian::formats::{
    planar_3d::{Gaussian3d, PlanarGaussian3d},
    planar_3d_chunked::LodPageStorage,
    planar_3d_lod::CpuGaussianLodBuilder,
};
use crate::gaussian::lod_settings::GaussianStreamingSettings;
use crate::io::lod::encode_page;
use crate::stream::http::{
    BrowserFetchHttpClient, HttpClientFailureKind, HttpClientPoll, HttpFetchRequest,
    HttpObjectVersion, HttpRangeClient, HttpRangePageTransport, HttpRangeTransportConfig,
    browser_http_unsettled_tasks_for_testing,
};
use crate::stream::transport::{
    LodPageTransport, ManifestPageLocations, MemoryPageTransport, PagePoll, PageRequest,
    PageRequestPriority,
};

wasm_bindgen_test::wasm_bindgen_test_configure!(run_in_browser);

#[wasm_bindgen::prelude::wasm_bindgen(inline_js = r#"
        export function install_bgs_range_fetch_fixture() {
            const original = window.fetch;
            window.fetch = function(request) {
                const range = request.headers.get("Range");
                const ifMatch = request.headers.get("If-Match");
                if (range !== "bytes=2-5") {
                    return Promise.reject(new Error("unexpected Range header: " + range));
                }
                if (ifMatch !== "\"fixture-v1\"") {
                    return Promise.reject(new Error("unexpected If-Match header: " + ifMatch));
                }
                return Promise.resolve(new Response(
                    new Uint8Array([2, 3, 4, 5]),
                    {
                        status: 206,
                        headers: new Headers({
                            "Content-Length": "4",
                            "Content-Range": "bytes 2-5/8",
                            "ETag": "\"fixture-v1\"",
                            "X-Lod-Version": "fixture-version-1"
                        })
                    }
                ));
            };
            return function() { window.fetch = original; };
        }

        export async function hold_bgs_cache_lock(lockName) {
            let release;
            let acquired;
            const acquiredPromise = new Promise(resolve => { acquired = resolve; });
            navigator.locks.request(lockName, async () => {
                acquired();
                await new Promise(resolve => { release = resolve; });
            });
            await acquiredPromise;
            return function() { release(); };
        }

        export function bgs_cache_lock_is_available(lockName) {
            return navigator.locks.request(
                lockName,
                { ifAvailable: true },
                lock => lock !== null
            );
        }
    "#)]
extern "C" {
    fn install_bgs_range_fetch_fixture() -> js_sys::Function;
    fn hold_bgs_cache_lock(lock_name: &str) -> js_sys::Promise;
    fn bgs_cache_lock_is_available(lock_name: &str) -> js_sys::Promise;
}

struct FetchFixtureRestore(js_sys::Function);

impl Drop for FetchFixtureRestore {
    fn drop(&mut self) {
        let _ = self.0.call0(&wasm_bindgen::JsValue::NULL);
    }
}

async fn browser_turn() {
    let promise = js_sys::Promise::new(&mut |resolve, _reject| {
        web_sys::window()
            .unwrap()
            .set_timeout_with_callback_and_timeout_and_arguments_0(&resolve, 0)
            .unwrap();
    });
    wasm_bindgen_futures::JsFuture::from(promise).await.unwrap();
}

async fn browser_delay(milliseconds: i32) {
    let promise = js_sys::Promise::new(&mut |resolve, _reject| {
        web_sys::window()
            .unwrap()
            .set_timeout_with_callback_and_timeout_and_arguments_0(&resolve, milliseconds)
            .unwrap();
    });
    wasm_bindgen_futures::JsFuture::from(promise).await.unwrap();
}

#[wasm_bindgen_test(async)]
async fn browser_timeout_guard_clears_callback_and_coordination_errors_stay_typed() {
    let fired = std::rc::Rc::new(std::cell::Cell::new(false));
    {
        let fired = fired.clone();
        let _timeout =
            BrowserTimeoutGuard::schedule(&web_sys::window().unwrap(), 10, move || fired.set(true))
                .unwrap();
    }
    browser_delay(30).await;
    assert!(
        !fired.get(),
        "dropped timeout guard left its JS timer armed"
    );

    let mapped = map_browser_coordination_error(
        "fixture Web Locks setup failed",
        js_sys::Error::new("fixture failure").into(),
    );
    assert!(matches!(
        mapped,
        PersistentCacheError::BrowserCoordinationUnavailable(message)
            if message.contains("fixture Web Locks setup failed")
                && message.contains("fixture failure")
    ));
}

fn identity() -> PersistentCachePageIdentity {
    PersistentCachePageIdentity {
        package: PersistentCachePackageIdentity {
            manifest_version: 2,
            page_schema_version: 2,
            required_features: 0,
            source_gaussian_count: 1,
            stored_gaussian_count: 1,
            source_fingerprint: 11,
            config_fingerprint: 22,
            builder_abi_version: 5,
            reducer_version: 1,
            package_version: Some("browser-test".to_owned()),
        },
        page_id: LodPageId(1),
        content_hash: 33,
        encoded_len: 4,
    }
}

fn encoded_browser_transport_fixture() -> (
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

async fn await_insert(
    cache: &std::rc::Rc<std::cell::RefCell<BrowserPersistentPageCache>>,
    ticket: u64,
) -> PersistentCacheInsert {
    for _ in 0..2_000 {
        let polled = cache.borrow_mut().poll_insert(&ticket);
        match polled {
            BrowserPersistentCachePoll::Pending => browser_turn().await,
            BrowserPersistentCachePoll::Ready(value) => return value,
            BrowserPersistentCachePoll::Failed(error) => {
                panic!("browser cache insert failed: {error:?}")
            }
        }
    }
    panic!("browser cache insert did not complete")
}

async fn await_lookup(
    cache: &std::rc::Rc<std::cell::RefCell<BrowserPersistentPageCache>>,
    ticket: u64,
) -> PersistentCacheLookup {
    for _ in 0..2_000 {
        let polled = cache.borrow_mut().poll_lookup(&ticket);
        match polled {
            BrowserPersistentCachePoll::Pending => browser_turn().await,
            BrowserPersistentCachePoll::Ready(value) => return value,
            BrowserPersistentCachePoll::Failed(error) => {
                panic!("browser cache lookup failed: {error:?}")
            }
        }
    }
    panic!("browser cache lookup did not complete")
}

async fn await_invalidate(
    cache: &std::rc::Rc<std::cell::RefCell<BrowserPersistentPageCache>>,
    ticket: u64,
) -> bool {
    for _ in 0..2_000 {
        let polled = cache.borrow_mut().poll_invalidate(&ticket);
        match polled {
            BrowserPersistentCachePoll::Pending => browser_turn().await,
            BrowserPersistentCachePoll::Ready(value) => return value,
            BrowserPersistentCachePoll::Failed(error) => {
                panic!("browser cache invalidation failed: {error:?}")
            }
        }
    }
    panic!("browser cache invalidation did not complete")
}

#[wasm_bindgen_test(async)]
async fn browser_cache_invalidation_removes_downstream_rejection() {
    let cache_name = format!("bgs-lod-invalidate-{}", js_sys::Date::now() as u64);
    let cache = BrowserPersistentPageCache::shared(BrowserPersistentCacheConfig {
        cache_name: cache_name.clone(),
        max_bytes: 4096,
        max_entries: 4,
        max_pending_operations: 4,
    })
    .unwrap();
    let identity = identity();
    let payload = PagePayload::new(identity.page_id, vec![1, 2, 3, 4]);
    let insert = cache
        .borrow_mut()
        .begin_insert(identity.clone(), payload)
        .unwrap();
    let _ = await_insert(&cache, insert).await;
    let invalidate = cache
        .borrow_mut()
        .begin_invalidate(identity.clone())
        .unwrap();
    assert!(await_invalidate(&cache, invalidate).await);
    let lookup = cache.borrow_mut().begin_lookup(identity).unwrap();
    assert_eq!(
        await_lookup(&cache, lookup).await,
        PersistentCacheLookup::Miss
    );

    let storage = web_sys::window().unwrap().caches().unwrap();
    wasm_bindgen_futures::JsFuture::from(storage.delete(&cache_name))
        .await
        .unwrap();
}

#[wasm_bindgen_test(async)]
async fn cache_storage_round_trip_and_dirty_index_recovery() {
    let cache_name = format!("bgs-lod-test-{}", js_sys::Date::now() as u64);
    let config = BrowserPersistentCacheConfig {
        cache_name: cache_name.clone(),
        max_bytes: 4096,
        max_entries: 4,
        max_pending_operations: 4,
    };
    let cache = BrowserPersistentPageCache::shared(config.clone()).unwrap();
    let identity = identity();
    let payload = PagePayload::new(identity.page_id, vec![1, 2, 3, 4]);
    let insert = cache
        .borrow_mut()
        .begin_insert(identity.clone(), payload.clone())
        .unwrap();
    assert!(matches!(
        await_insert(&cache, insert).await,
        PersistentCacheInsert::Written { .. }
    ));
    let lookup = cache.borrow_mut().begin_lookup(identity.clone()).unwrap();
    assert_eq!(
        await_lookup(&cache, lookup).await,
        PersistentCacheLookup::Hit(payload)
    );

    let window = web_sys::window().unwrap();
    let origin = window.location().origin().unwrap();
    let storage = window.caches().unwrap();
    let opened = wasm_bindgen_futures::JsFuture::from(storage.open(&cache_name))
        .await
        .unwrap();
    let raw_cache = opened.dyn_into::<web_sys::Cache>().unwrap();
    let dirty = BrowserCacheIndex {
        next_epoch: 1,
        entries: BTreeMap::new(),
    };
    assert_eq!(
        decode_browser_index(&encode_browser_index(&dirty).unwrap(), 4).unwrap(),
        dirty
    );
    write_browser_index_with_flags(&raw_cache, &origin, &dirty, BROWSER_INDEX_DIRTY_FLAG)
        .await
        .unwrap();
    let lookup = cache.borrow_mut().begin_lookup(identity).unwrap();
    assert_eq!(
        await_lookup(&cache, lookup).await,
        PersistentCacheLookup::Miss
    );
    wasm_bindgen_futures::JsFuture::from(storage.delete(&cache_name))
        .await
        .unwrap();
}

#[wasm_bindgen_test(async)]
async fn dropped_owned_browser_caches_remain_globally_charged_until_settlement() {
    let oversized = BrowserPersistentCacheConfig {
        cache_name: "bgs-lod-oversized-queue".to_owned(),
        max_bytes: 4096,
        max_entries: 4,
        max_pending_operations: MAX_PERSISTENT_CACHE_PENDING_OPERATIONS + 1,
    };
    assert!(matches!(
        oversized.validate(),
        Err(PersistentCacheError::ServiceQueueCapacityTooLarge { .. })
    ));
    assert_eq!(
        browser_persistent_cache_unsettled_operations_for_testing(),
        0
    );
    let prefix = format!("bgs-lod-owned-churn-{}", js_sys::Date::now() as u64);
    let mut caches = Vec::new();
    for index in 0..BROWSER_PERSISTENT_CACHE_EFFECTIVE_OPERATION_CAPACITY {
        let mut cache = BrowserPersistentPageCache::new(BrowserPersistentCacheConfig {
            cache_name: format!("{prefix}-{index}"),
            max_bytes: 4096,
            max_entries: 4,
            max_pending_operations: 1,
        })
        .unwrap();
        cache.begin_lookup(identity()).unwrap();
        caches.push(cache);
    }
    assert_eq!(
        browser_persistent_cache_unsettled_operations_for_testing(),
        BROWSER_PERSISTENT_CACHE_EFFECTIVE_OPERATION_CAPACITY
    );
    drop(caches);

    let mut rejected = BrowserPersistentPageCache::new(BrowserPersistentCacheConfig {
        cache_name: format!("{prefix}-rejected"),
        max_bytes: 4096,
        max_entries: 4,
        max_pending_operations: 1,
    })
    .unwrap();
    let ticket = rejected.begin_lookup(identity()).unwrap();
    assert!(matches!(
        rejected.poll_lookup(&ticket),
        BrowserPersistentCachePoll::Failed(
            PersistentCacheError::BrowserOperationCapacityExceeded {
                maximum: BROWSER_PERSISTENT_CACHE_EFFECTIVE_OPERATION_CAPACITY
            }
        )
    ));

    for _ in 0..2_000 {
        if browser_persistent_cache_unsettled_operations_for_testing() == 0 {
            return;
        }
        browser_turn().await;
    }
    panic!("dropped browser cache operations did not settle and release admission");
}

#[wasm_bindgen_test(async)]
async fn independent_browser_coordinators_serialize_with_web_locks() {
    let cache_name = format!("bgs-lod-lock-test-{}", js_sys::Date::now() as u64);
    let config = BrowserPersistentCacheConfig {
        cache_name: cache_name.clone(),
        max_bytes: 4096,
        max_entries: 4,
        max_pending_operations: 4,
    };
    let first = std::rc::Rc::new(std::cell::RefCell::new(
        BrowserPersistentPageCache::new(config.clone()).unwrap(),
    ));
    let second = std::rc::Rc::new(std::cell::RefCell::new(
        BrowserPersistentPageCache::new(config).unwrap(),
    ));
    let first_identity = identity();
    let mut second_identity = first_identity.clone();
    second_identity.page_id = LodPageId(2);
    second_identity.content_hash = 44;
    let first_ticket = first
        .borrow_mut()
        .begin_insert(
            first_identity.clone(),
            PagePayload::new(first_identity.page_id, vec![1, 2, 3, 4]),
        )
        .unwrap();
    let second_ticket = second
        .borrow_mut()
        .begin_insert(
            second_identity.clone(),
            PagePayload::new(second_identity.page_id, vec![5, 6, 7, 8]),
        )
        .unwrap();
    assert!(matches!(
        first.borrow_mut().poll_insert(&first_ticket),
        BrowserPersistentCachePoll::Pending
    ));
    assert!(matches!(
        second.borrow_mut().poll_insert(&second_ticket),
        BrowserPersistentCachePoll::Pending
    ));
    assert!(matches!(
        await_insert(&first, first_ticket).await,
        PersistentCacheInsert::Written { .. }
    ));
    assert!(matches!(
        await_insert(&second, second_ticket).await,
        PersistentCacheInsert::Written { .. }
    ));

    let first_lookup = first
        .borrow_mut()
        .begin_lookup(first_identity.clone())
        .unwrap();
    assert_eq!(
        await_lookup(&first, first_lookup).await,
        PersistentCacheLookup::Hit(PagePayload::new(first_identity.page_id, vec![1, 2, 3, 4]))
    );
    let second_lookup = first
        .borrow_mut()
        .begin_lookup(second_identity.clone())
        .unwrap();
    assert_eq!(
        await_lookup(&first, second_lookup).await,
        PersistentCacheLookup::Hit(PagePayload::new(second_identity.page_id, vec![5, 6, 7, 8]))
    );

    let storage = web_sys::window().unwrap().caches().unwrap();
    wasm_bindgen_futures::JsFuture::from(storage.delete(&cache_name))
        .await
        .unwrap();
}

#[wasm_bindgen_test(async)]
async fn browser_cache_lock_acquisition_times_out_and_releases_capacity() {
    let cache_name = format!("bgs-lod-lock-timeout-{}", js_sys::Date::now() as u64);
    let lock_name = format!("bevy-gaussian-lod-cache::{cache_name}");
    let release = wasm_bindgen_futures::JsFuture::from(hold_bgs_cache_lock(&lock_name))
        .await
        .unwrap()
        .dyn_into::<js_sys::Function>()
        .unwrap();
    let mut cache = BrowserPersistentPageCache::new(BrowserPersistentCacheConfig {
        cache_name,
        max_bytes: 4096,
        max_entries: 4,
        max_pending_operations: 1,
    })
    .unwrap();
    let ticket = cache.begin_lookup(identity()).unwrap();
    for _ in 0..2_000 {
        match cache.poll_lookup(&ticket) {
            BrowserPersistentCachePoll::Pending => browser_turn().await,
            BrowserPersistentCachePoll::Failed(
                PersistentCacheError::BrowserCoordinationUnavailable(_),
            ) => {
                release.call0(&wasm_bindgen::JsValue::NULL).unwrap();
                for _ in 0..2_000 {
                    if browser_persistent_cache_unsettled_operations_for_testing() == 0 {
                        return;
                    }
                    browser_turn().await;
                }
                panic!("timed-out browser cache operation did not release admission");
            }
            BrowserPersistentCachePoll::Failed(error) => {
                panic!("unexpected browser cache lock failure: {error:?}")
            }
            BrowserPersistentCachePoll::Ready(_) => {
                panic!("browser cache acquired an externally held Web Lock")
            }
        }
    }
    release.call0(&wasm_bindgen::JsValue::NULL).unwrap();
    panic!("browser cache Web Lock acquisition did not time out")
}

#[wasm_bindgen_test(async)]
async fn post_grant_cache_timeout_bypasses_without_releasing_lock_or_permit() {
    assert_eq!(
        browser_persistent_cache_unsettled_operations_for_testing(),
        0
    );
    let cache_name = format!("bgs-lod-operation-timeout-{}", js_sys::Date::now() as u64);
    let lock_name = format!("bevy-gaussian-lod-cache::{cache_name}");
    let config = BrowserPersistentCacheConfig {
        cache_name: cache_name.clone(),
        max_bytes: 1024 * 1024,
        max_entries: 4,
        max_pending_operations: 4,
    };
    let cache = std::rc::Rc::new(std::cell::RefCell::new(
        BrowserPersistentPageCache::new(config.clone()).unwrap(),
    ));
    let mut release = None;
    let gate = js_sys::Promise::new(&mut |resolve, _reject| release = Some(resolve));
    let gate_ticket = cache
        .borrow_mut()
        .enqueue(BrowserCacheOperation::TestGate(gate))
        .unwrap();
    // This request is queued before the gate times out. `pump` must bypass
    // it without spending a second realm permit once the timeout is polled.
    let queued_ticket = cache.borrow_mut().begin_lookup(identity()).unwrap();
    assert_eq!(
        browser_persistent_cache_unsettled_operations_for_testing(),
        1
    );

    for _ in 0..4_000 {
        match cache.borrow_mut().poll_lookup(&gate_ticket) {
            BrowserPersistentCachePoll::Pending => browser_turn().await,
            BrowserPersistentCachePoll::Failed(
                PersistentCacheError::BrowserCacheOperationTimedOut { timeout_millis },
            ) => {
                assert_eq!(timeout_millis, BROWSER_CACHE_OPERATION_TIMEOUT_MS as u32);
                break;
            }
            BrowserPersistentCachePoll::Failed(error) => {
                panic!("unexpected post-grant cache failure: {error:?}")
            }
            BrowserPersistentCachePoll::Ready(_) => {
                panic!("never-settling cache operation completed")
            }
        }
    }
    assert!(cache.borrow().namespace_state.is_temporarily_bypassed());
    assert_eq!(
        cache.borrow_mut().poll_lookup(&queued_ticket),
        BrowserPersistentCachePoll::Failed(PersistentCacheError::BrowserCacheTemporarilyBypassed)
    );
    assert_eq!(
        browser_persistent_cache_unsettled_operations_for_testing(),
        1,
        "queued work consumed a second permit behind the draining lock"
    );

    let independent = std::rc::Rc::new(std::cell::RefCell::new(
        BrowserPersistentPageCache::new(config.clone()).unwrap(),
    ));
    assert!(std::rc::Rc::ptr_eq(
        &cache.borrow().namespace_state,
        &independent.borrow().namespace_state
    ));
    assert_eq!(
        independent.borrow_mut().begin_lookup(identity()),
        Err(PersistentCacheError::BrowserCacheTemporarilyBypassed)
    );
    let lock_available =
        wasm_bindgen_futures::JsFuture::from(bgs_cache_lock_is_available(&lock_name))
            .await
            .unwrap()
            .as_bool()
            .unwrap();
    assert!(
        !lock_available,
        "caller timeout released the Web Lock early"
    );

    let (identities, transport_identity, expected_payload) = encoded_browser_transport_fixture();
    let mut upstream = MemoryPageTransport::default();
    upstream.insert(transport_identity.page_id, expected_payload.bytes.clone());
    let mut transport =
        SharedBrowserPersistentCachePageTransport::new(upstream, cache.clone(), identities);
    let mut request = PageRequest::new(
        transport_identity.page_id,
        PageRequestPriority::fallback_critical(u32::MAX),
    );
    request.expected_bytes = Some(transport_identity.encoded_len);
    let transport_ticket = transport.begin(request).unwrap();
    assert_eq!(
        transport.poll(&transport_ticket),
        PagePoll::Ready(expected_payload),
        "cache-first wrapper did not bypass safely to its validated upstream"
    );

    release
        .take()
        .unwrap()
        .call0(&wasm_bindgen::JsValue::NULL)
        .unwrap();
    for _ in 0..2_000 {
        if browser_persistent_cache_unsettled_operations_for_testing() == 0
            && !cache.borrow().namespace_state.is_temporarily_bypassed()
        {
            break;
        }
        browser_turn().await;
    }
    assert_eq!(
        browser_persistent_cache_unsettled_operations_for_testing(),
        0,
        "settled cache operation retained its realm permit"
    );
    assert!(!cache.borrow().namespace_state.is_temporarily_bypassed());
    assert!(
        cache.borrow().results.borrow().is_empty(),
        "late cache settlement published an orphan result"
    );
    let lock_available =
        wasm_bindgen_futures::JsFuture::from(bgs_cache_lock_is_available(&lock_name))
            .await
            .unwrap()
            .as_bool()
            .unwrap();
    assert!(
        lock_available,
        "settled cache operation retained its Web Lock"
    );

    let recovery_ticket = independent.borrow_mut().begin_lookup(identity()).unwrap();
    assert_eq!(
        await_lookup(&independent, recovery_ticket).await,
        PersistentCacheLookup::Miss
    );
    let storage = web_sys::window().unwrap().caches().unwrap();
    wasm_bindgen_futures::JsFuture::from(storage.delete(&cache_name))
        .await
        .unwrap();
}

#[wasm_bindgen_test(async)]
async fn browser_fetch_stream_is_bounded_and_cancelable() {
    let request = HttpFetchRequest {
        url: "data:application/octet-stream,0123456789".to_owned(),
        byte_range: None,
        expected_bytes: 10,
        max_response_bytes: 4,
        timeout: std::time::Duration::from_secs(2),
        if_match: None,
        expected_version: None,
        object_version_header: None,
    };
    let mut client = BrowserFetchHttpClient::with_max_requests(1).unwrap();
    let ticket = client.begin(request.clone()).unwrap();
    assert_eq!(
        client.begin(request).unwrap_err().kind,
        HttpClientFailureKind::ConcurrencyLimit
    );
    for _ in 0..2_000 {
        match client.poll(&ticket) {
            HttpClientPoll::Pending => browser_turn().await,
            HttpClientPoll::Ready(_) => panic!("oversized browser body was accepted"),
            HttpClientPoll::Failed(error) => {
                assert_eq!(error.kind, HttpClientFailureKind::ResponseTooLarge);
                return;
            }
        }
    }
    client.cancel(&ticket);
    panic!("bounded browser fetch did not complete")
}

#[wasm_bindgen_test(async)]
async fn browser_http_range_transport_validates_request_and_response() {
    GaussianStreamingSettings::default().validate().unwrap();
    let _restore = FetchFixtureRestore(install_bgs_range_fetch_fixture());
    let mut gaussian = Gaussian3d::default();
    gaussian.rotation.rotation = [1.0, 0.0, 0.0, 0.0];
    let source: PlanarGaussian3d = vec![gaussian].into();
    let mut built = CpuGaussianLodBuilder::default().build(&source).unwrap();
    let page_id = built.manifest.pages[0].id;
    for page in &mut built.manifest.pages {
        page.storage = Some(LodPageStorage {
            uri: "range-fixture.bin".to_owned(),
            byte_range: Some((2, 4)),
            encoded_len: 4,
        });
    }
    let locations = ManifestPageLocations::from_manifest(&built.manifest).unwrap();
    let origin = web_sys::window().unwrap().location().origin().unwrap();
    let mut transport = HttpRangePageTransport::new(
        HttpRangeTransportConfig {
            base_url: origin,
            request_timeout: std::time::Duration::from_secs(2),
            retry_limit: 0,
            retry_base_delay: std::time::Duration::ZERO,
            retry_max_delay: std::time::Duration::ZERO,
            max_encoded_page_bytes: 4,
            max_concurrent_requests: 1,
            require_content_length: true,
            require_object_validator: true,
            object_version_header: Some("x-lod-version".to_owned()),
        },
        locations,
        BrowserFetchHttpClient::with_max_requests(1).unwrap(),
    )
    .unwrap();
    transport
        .expect_object_version(
            "range-fixture.bin",
            HttpObjectVersion {
                etag: Some("\"fixture-v1\"".to_owned()),
                version: Some("fixture-version-1".to_owned()),
            },
        )
        .unwrap();
    let ticket = transport
        .begin(PageRequest::new(
            page_id,
            PageRequestPriority::fallback_critical(u32::MAX),
        ))
        .unwrap();
    for _ in 0..2_000 {
        match transport.poll(&ticket) {
            PagePoll::Pending => browser_turn().await,
            PagePoll::Ready(payload) => {
                assert_eq!(payload.page_id, page_id);
                assert_eq!(payload.bytes, vec![2, 3, 4, 5]);
                return;
            }
            PagePoll::Failed(error) => panic!("browser range transport failed: {error:?}"),
        }
    }
    transport.cancel(&ticket);
    panic!("browser range transport did not complete")
}

#[wasm_bindgen_test(async)]
async fn cancelled_fetches_remain_globally_charged_until_settlement() {
    let request = HttpFetchRequest {
        url: "data:application/octet-stream,pending".to_owned(),
        byte_range: None,
        expected_bytes: 7,
        max_response_bytes: 7,
        timeout: std::time::Duration::from_secs(2),
        if_match: None,
        expected_version: None,
        object_version_header: None,
    };
    let mut clients = Vec::new();
    for _ in 0..4 {
        let mut client = BrowserFetchHttpClient::with_max_requests(1).unwrap();
        let ticket = client.begin(request.clone()).unwrap();
        client.cancel(&ticket);
        clients.push(client);
    }
    assert_eq!(browser_http_unsettled_tasks_for_testing(), 4);
    let mut blocked = BrowserFetchHttpClient::with_max_requests(1).unwrap();
    assert_eq!(
        blocked.begin(request.clone()).unwrap_err().kind,
        HttpClientFailureKind::ConcurrencyLimit
    );

    for _ in 0..2_000 {
        if browser_http_unsettled_tasks_for_testing() == 0 {
            let ticket = blocked.begin(request).unwrap();
            blocked.cancel(&ticket);
            break;
        }
        browser_turn().await;
    }
    for _ in 0..2_000 {
        if browser_http_unsettled_tasks_for_testing() == 0 {
            return;
        }
        browser_turn().await;
    }
    panic!("cancelled browser Fetch tasks did not release the global budget")
}

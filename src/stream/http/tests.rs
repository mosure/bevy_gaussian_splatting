use std::{
    collections::VecDeque,
    sync::{
        Arc,
        atomic::{AtomicU32, Ordering},
    },
};

use super::*;
use crate::gaussian::formats::{
    planar_3d::{Gaussian3d, PlanarGaussian3d},
    planar_3d_chunked::LodPageStorage,
    planar_3d_lod::CpuGaussianLodBuilder,
};
use crate::io::lod::encode_page;

#[derive(Default)]
struct MockClient {
    responses: VecDeque<Result<HttpFetchResponse, HttpClientFailure>>,
    requests: Vec<HttpFetchRequest>,
    tickets: BTreeMap<u64, Result<HttpFetchResponse, HttpClientFailure>>,
    next_ticket: u64,
    cancellations: Arc<AtomicU32>,
}

impl HttpRangeClient for MockClient {
    type Ticket = u64;

    fn begin(&mut self, request: HttpFetchRequest) -> Result<Self::Ticket, HttpClientFailure> {
        self.requests.push(request);
        let result = self.responses.pop_front().expect("mock response");
        let ticket = self.next_ticket;
        self.next_ticket += 1;
        self.tickets.insert(ticket, result);
        Ok(ticket)
    }

    fn poll(&mut self, ticket: &Self::Ticket) -> HttpClientPoll {
        match self.tickets.remove(ticket) {
            Some(Ok(response)) => HttpClientPoll::Ready(response),
            Some(Err(error)) => HttpClientPoll::Failed(error),
            None => HttpClientPoll::Pending,
        }
    }

    fn cancel(&mut self, ticket: &Self::Ticket) {
        self.tickets.remove(ticket);
        self.cancellations.fetch_add(1, Ordering::Relaxed);
    }
}

fn fixture() -> (ManifestPageLocations, LodPageId) {
    let mut gaussian = Gaussian3d::default();
    gaussian.rotation.rotation = [1.0, 0.0, 0.0, 0.0];
    let source: PlanarGaussian3d = vec![gaussian].into();
    let mut built = CpuGaussianLodBuilder::default().build(&source).unwrap();
    let page = built.manifest.pages[0].id;
    for descriptor in &mut built.manifest.pages {
        descriptor.storage = Some(LodPageStorage {
            uri: format!("{}.gspage", descriptor.id.0),
            byte_range: None,
            encoded_len: 4,
        });
    }
    (
        ManifestPageLocations::from_manifest(&built.manifest).unwrap(),
        page,
    )
}

fn validation_fixture() -> (
    GaussianLodManifest,
    ManifestPageLocations,
    LodPageId,
    Vec<u8>,
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
    let locations = ManifestPageLocations::from_manifest(&built.manifest).unwrap();
    (built.manifest, locations, page.id, encoded)
}

fn config() -> HttpRangeTransportConfig {
    HttpRangeTransportConfig {
        base_url: "https://cdn.example/scene".to_owned(),
        request_timeout: Duration::from_secs(5),
        retry_limit: 2,
        retry_base_delay: Duration::ZERO,
        retry_max_delay: Duration::from_secs(1),
        max_encoded_page_bytes: 64 * 1024 * 1024,
        max_concurrent_requests: 4,
        require_content_length: true,
        require_object_validator: true,
        object_version_header: None,
    }
}

fn response(bytes: Vec<u8>, etag: &str) -> HttpFetchResponse {
    HttpFetchResponse {
        status: 200,
        redirected: false,
        content_length: Some(bytes.len() as u64),
        bytes,
        content_range: None,
        content_encoding: None,
        etag: Some(etag.to_owned()),
        object_version: None,
        retry_after: None,
    }
}

#[test]
fn default_transport_returns_encoded_payload_without_codec_retry() {
    let (manifest, locations, page, valid) = validation_fixture();
    let mut corrupt = valid.clone();
    *corrupt.last_mut().unwrap() ^= 0x01;
    let mut client = MockClient::default();
    client
        .responses
        .push_back(Ok(response(corrupt.clone(), "\"v1\"")));
    let mut transport = HttpRangePageTransport::new(config(), locations, client).unwrap();
    let mut request = PageRequest::new(
        page,
        super::super::transport::PageRequestPriority::visible(1),
    );
    request.expected_bytes = Some(valid.len() as u64);

    let ticket = transport.begin(request).unwrap();
    let PagePoll::Ready(payload) = transport.poll(&ticket) else {
        panic!("structurally valid encoded bytes must cross the HTTP boundary")
    };

    assert_eq!(payload.bytes, corrupt);
    assert_eq!(transport.client().requests.len(), 1);
    let descriptor = manifest
        .pages
        .iter()
        .find(|descriptor| descriptor.id == page)
        .unwrap();
    assert!(
        decode_page_with_descriptor(&payload.bytes, descriptor, LodCodecLimits::default()).is_err(),
        "the downstream codec boundary must still reject the corrupt page"
    );
}

#[test]
fn decoded_manifest_mismatch_retries_inside_authoritative_http_budget() {
    let (manifest, locations, page, valid) = validation_fixture();
    let mut corrupt = valid.clone();
    *corrupt.last_mut().unwrap() ^= 0x01;
    let mut client = MockClient::default();
    client.responses.push_back(Ok(response(corrupt, "\"v1\"")));
    client
        .responses
        .push_back(Ok(response(valid.clone(), "\"v1\"")));
    let mut policy = config();
    policy.retry_limit = 1;
    policy.retry_base_delay = Duration::ZERO;
    policy.retry_max_delay = Duration::ZERO;
    let mut transport = HttpRangePageTransport::new(policy, locations, client)
        .unwrap()
        .with_manifest_validation(&manifest)
        .unwrap();
    let mut request = PageRequest::new(
        page,
        super::super::transport::PageRequestPriority::visible(1),
    );
    request.expected_bytes = Some(valid.len() as u64);
    let ticket = transport.begin(request).unwrap();
    let mut ready = None;
    for _ in 0..8 {
        match transport.poll(&ticket) {
            PagePoll::Pending => {}
            PagePoll::Ready(payload) => {
                ready = Some(payload);
                break;
            }
            PagePoll::Failed(error) => panic!("manifest retry failed: {error:?}"),
        }
    }
    assert_eq!(ready.unwrap().bytes, valid);
    assert_eq!(transport.client().requests.len(), 2);
}

#[test]
fn response_validation_rejects_full_object_for_pack_range() {
    let page = LodPageId(9);
    let location = ManifestPageLocation {
        uri: "scene.pack".into(),
        byte_range: Some((40, 4)),
        encoded_len: 4,
    };
    let error = validate_http_response(
        page,
        &location,
        &response(vec![1, 2, 3, 4], "\"v1\""),
        &config(),
        None,
    )
    .unwrap_err();
    assert_eq!(
        error,
        HttpRangeTransportError::UnexpectedStatus {
            page,
            expected: 206,
            actual: 200,
        }
    );

    let mut ranged = response(vec![1, 2, 3, 4], "\"v1\"");
    ranged.status = 206;
    for malformed in [
        "bytes 40-43",
        "bytes 40-43/garbage",
        "bytes 40-43/43",
        "bytes 40-43/44/extra",
    ] {
        ranged.content_range = Some(malformed.to_owned());
        assert!(matches!(
            validate_http_response(page, &location, &ranged, &config(), None),
            Err(HttpRangeTransportError::InvalidContentRange(value)) if value == malformed
        ));
    }
    ranged.content_range = Some("bytes 40-43/44".to_owned());
    assert!(validate_http_response(page, &location, &ranged, &config(), None).is_ok());
    ranged.content_range = Some("bytes 40-43/*".to_owned());
    assert!(validate_http_response(page, &location, &ranged, &config(), None).is_ok());
}

#[test]
fn byte_range_end_rejects_zero_length_and_overflow() {
    assert_eq!(byte_range_end(2, 4), Some(5));
    assert_eq!(byte_range_end(2, 0), None);
    assert_eq!(byte_range_end(u64::MAX, 2), None);
}

#[test]
fn content_length_policy_is_enforced_above_the_bounded_client() {
    let page = LodPageId(5);
    let location = ManifestPageLocation {
        uri: "page.gspage".into(),
        byte_range: None,
        encoded_len: 4,
    };
    let mut without_header = response(vec![1, 2, 3, 4], "\"v1\"");
    without_header.content_length = None;
    let mut optional = config();
    optional.require_content_length = false;
    assert!(validate_http_response(page, &location, &without_header, &optional, None).is_ok());
    optional.require_content_length = true;
    assert_eq!(
        validate_http_response(page, &location, &without_header, &optional, None),
        Err(HttpRangeTransportError::MissingContentLength(page))
    );
}

#[test]
fn retries_transient_status_and_preserves_exact_request_bound() {
    let (locations, page) = fixture();
    let expected = locations.get(page).unwrap().encoded_len;
    let mut client = MockClient::default();
    let mut unavailable = response(Vec::new(), "\"v1\"");
    unavailable.status = 503;
    unavailable.content_length = Some(0);
    client.responses.push_back(Ok(unavailable));
    client
        .responses
        .push_back(Ok(response(vec![7; expected as usize], "\"v1\"")));
    let mut transport = HttpRangePageTransport::new(config(), locations, client).unwrap();
    let mut request = PageRequest::new(
        page,
        super::super::transport::PageRequestPriority::visible(1),
    );
    request.expected_bytes = Some(expected);
    let ticket = transport.begin(request).unwrap();
    assert!(matches!(transport.poll(&ticket), PagePoll::Pending));
    let payload = loop {
        match transport.poll(&ticket) {
            PagePoll::Pending => std::thread::yield_now(),
            PagePoll::Ready(payload) => break payload,
            PagePoll::Failed(error) => panic!("unexpected failure: {error:?}"),
        }
    };
    assert_eq!(payload.bytes.len() as u64, expected);
    assert_eq!(transport.client().requests.len(), 2);
    assert!(
        transport
            .client()
            .requests
            .iter()
            .all(|request| request.max_response_bytes == expected)
    );
}

#[test]
fn shared_pack_etag_change_is_typed() {
    let page = LodPageId(1);
    let location = ManifestPageLocation {
        uri: "scene.pack".into(),
        byte_range: None,
        encoded_len: 4,
    };
    let version = HttpObjectVersion {
        etag: Some("\"first\"".to_owned()),
        version: None,
    };
    assert_eq!(
        validate_http_response(
            page,
            &location,
            &response(vec![0; 4], "\"second\""),
            &config(),
            Some(&version),
        ),
        Err(HttpRangeTransportError::ResponseValidatorMismatch {
            page,
            expected: version,
            actual: HttpObjectVersion {
                etag: Some("\"second\"".to_owned()),
                version: None,
            },
        })
    );
}

#[test]
fn url_resolution_rejects_escape_and_credentials() {
    assert!(matches!(
        resolve_page_url("https://cdn.example/scene", "../secret"),
        Err(HttpRangeTransportError::UnsafeUri(_))
    ));
    assert!(matches!(
        validate_base_url("https://token@cdn.example/scene"),
        Err(HttpRangeTransportError::InvalidBaseUrl(_))
    ));
    for unsafe_uri in ["%2e%2e/secret", "pages%2fsecret", "page?redirect=secret"] {
        assert!(matches!(
            resolve_page_url("https://cdn.example/scene", unsafe_uri),
            Err(HttpRangeTransportError::UnsafeUri(_))
        ));
    }
    assert!(matches!(
        validate_base_url("https://cdn.example/scene?token=secret"),
        Err(HttpRangeTransportError::InvalidBaseUrl(_))
    ));
}

#[test]
fn transport_ticket_capacity_bounds_fetch_and_retry_state() {
    let (locations, page) = fixture();
    let mut client = MockClient::default();
    client
        .responses
        .push_back(Ok(response(vec![0; 4], "\"v1\"")));
    let mut policy = config();
    policy.max_concurrent_requests = 1;
    let mut transport = HttpRangePageTransport::new(policy, locations, client).unwrap();
    let request = PageRequest::new(
        page,
        super::super::transport::PageRequestPriority::visible(1),
    );
    let _first = transport.begin(request).unwrap();
    assert_eq!(
        transport.begin(request),
        Err(HttpRangeTransportError::RequestCapacityExceeded { maximum: 1 })
    );
}

#[test]
fn dropping_transport_cancels_every_fetching_ticket() {
    let (locations, page) = fixture();
    let mut client = MockClient::default();
    let cancellations = client.cancellations.clone();
    client
        .responses
        .push_back(Ok(response(vec![0; 4], "\"v1\"")));
    let mut transport = HttpRangePageTransport::new(config(), locations, client).unwrap();
    transport
        .begin(PageRequest::new(
            page,
            super::super::transport::PageRequestPriority::visible(1),
        ))
        .unwrap();
    drop(transport);
    assert_eq!(cancellations.load(Ordering::Relaxed), 1);
}

#[test]
fn retry_delay_is_exponential_and_capped() {
    let mut policy = config();
    policy.retry_base_delay = Duration::from_millis(25);
    policy.retry_max_delay = Duration::from_millis(70);
    assert_eq!(policy.retry_delay(1, None), Duration::from_millis(25));
    assert_eq!(policy.retry_delay(2, None), Duration::from_millis(50));
    assert_eq!(policy.retry_delay(3, None), Duration::from_millis(70));
    assert_eq!(
        policy.retry_delay(1, Some(Duration::from_secs(10))),
        Duration::from_millis(70)
    );
    policy.retry_base_delay = Duration::MAX;
    policy.retry_max_delay = Duration::MAX;
    assert!(matches!(
        policy.validate(),
        Err(HttpRangeTransportError::RetryDeadlineOutOfRange(
            Duration::MAX
        ))
    ));
}

#[test]
fn request_timeout_validation_rejects_zero_and_unrepresentable_deadlines() {
    for timeout in [Duration::ZERO, Duration::MAX] {
        let mut policy = config();
        policy.request_timeout = timeout;
        assert!(matches!(
            policy.validate(),
            Err(HttpRangeTransportError::ZeroTimeout)
                | Err(HttpRangeTransportError::RequestDeadlineOutOfRange(_))
        ));
        assert_eq!(
            validate_fetch_request_timeout(timeout).unwrap_err().kind,
            HttpClientFailureKind::InvalidRequest
        );
    }
}

#[cfg(not(target_arch = "wasm32"))]
#[test]
fn completed_client_result_wins_when_transport_poll_is_late() {
    let (locations, page) = fixture();
    let mut client = MockClient::default();
    client
        .responses
        .push_back(Ok(response(vec![0; 4], "\"on-time\"")));
    let mut policy = config();
    policy.request_timeout = Duration::from_millis(1);
    policy.retry_limit = 0;
    let mut transport = HttpRangePageTransport::new(policy, locations, client).unwrap();
    let ticket = transport
        .begin(PageRequest::new(
            page,
            super::super::transport::PageRequestPriority::visible(1),
        ))
        .unwrap();
    std::thread::sleep(Duration::from_millis(10));
    assert!(matches!(transport.poll(&ticket), PagePoll::Ready(_)));
}

#[test]
fn bounded_chunk_accounting_rejects_understated_stream_before_growth() {
    assert_eq!(bounded_chunk_end(0, 4, 8).unwrap(), 4);
    assert_eq!(bounded_chunk_end(4, 4, 8).unwrap(), 8);
    let error = bounded_chunk_end(8, 1, 8).unwrap_err();
    assert_eq!(error.kind, HttpClientFailureKind::ResponseTooLarge);
    let overflow = bounded_chunk_end(usize::MAX, 1, u64::MAX).unwrap_err();
    assert_eq!(overflow.kind, HttpClientFailureKind::ResponseTooLarge);
}

#[test]
fn browser_timer_delay_and_body_growth_are_capped_geometrically() {
    assert_eq!(browser_timer_delay_ms(Duration::from_nanos(1)), 1);
    assert_eq!(browser_timer_delay_ms(Duration::from_millis(25)), 25);
    assert_eq!(
        browser_timer_delay_ms(Duration::from_secs(i32::MAX as u64 + 1)),
        i32::MAX
    );

    assert_eq!(bounded_growth_capacity(0, 1, 64).unwrap(), 2);
    assert_eq!(bounded_growth_capacity(2, 3, 64).unwrap(), 4);
    assert_eq!(bounded_growth_capacity(4, 5, 6).unwrap(), 6);
    assert_eq!(
        bounded_growth_capacity(6, 7, 6).unwrap_err().kind,
        HttpClientFailureKind::ResponseTooLarge
    );
}

#[cfg(not(target_arch = "wasm32"))]
#[test]
fn native_client_and_direct_begin_reject_unrepresentable_timeouts() {
    assert!(matches!(
        NativeUreqHttpClient::new(Duration::MAX),
        Err(HttpRangeTransportError::RequestDeadlineOutOfRange(
            Duration::MAX
        ))
    ));
    let pool = create_native_http_worker_pool_with_limits(1, 1).unwrap();
    let mut client = NativeUreqHttpClient::new(Duration::from_secs(1)).unwrap();
    let error = client
        .begin_with_pool(
            HttpFetchRequest {
                url: "http://127.0.0.1/never-started".to_owned(),
                byte_range: None,
                expected_bytes: 1,
                max_response_bytes: 1,
                timeout: Duration::MAX,
                if_match: None,
                expected_version: None,
                object_version_header: None,
            },
            &pool,
        )
        .unwrap_err();
    assert_eq!(error.kind, HttpClientFailureKind::InvalidRequest);
}

#[cfg(not(target_arch = "wasm32"))]
#[test]
fn process_dns_pool_bounds_stalled_and_cancelled_resolution_work() {
    use std::sync::atomic::AtomicBool;
    use ureq::unversioned::resolver::Resolver as _;

    assert!(
        create_native_dns_resolver_pool_with_limits(0, 1, Arc::new(|_| Ok(Vec::new()))).is_err()
    );

    let release = Arc::new(AtomicBool::new(false));
    let calls = Arc::new(AtomicU32::new(0));
    let (started_sender, started_receiver) = std::sync::mpsc::sync_channel(1);
    let lookup_release = release.clone();
    let lookup_calls = calls.clone();
    let pool = create_native_dns_resolver_pool_with_limits(
        1,
        1,
        Arc::new(move |_| {
            let call = lookup_calls.fetch_add(1, Ordering::AcqRel);
            if call == 0 {
                started_sender.send(()).unwrap();
                while !lookup_release.load(Ordering::Acquire) {
                    std::thread::yield_now();
                }
            }
            Ok(vec!["127.0.0.1:80".parse().unwrap()])
        }),
    )
    .unwrap();
    let resolver = BoundedNativeResolver { pool };
    let uri: ureq::http::Uri = "http://resolver.test/page".parse().unwrap();

    let first_resolver = resolver.clone();
    let first_uri = uri.clone();
    let first = std::thread::spawn(move || {
        first_resolver.resolve(
            &first_uri,
            &ureq::config::Config::default(),
            ureq::unversioned::transport::NextTimeout {
                after: ureq::unversioned::transport::time::Duration::from_millis(20),
                reason: ureq::Timeout::Resolve,
            },
        )
    });
    started_receiver.recv().unwrap();
    assert!(matches!(
        first.join().unwrap(),
        Err(ureq::Error::Timeout(_))
    ));

    let immediate_timeout_started = Instant::now();
    let second = resolver.resolve(
        &uri,
        &ureq::config::Config::default(),
        ureq::unversioned::transport::NextTimeout {
            after: ureq::unversioned::transport::time::Duration::from_millis(0),
            reason: ureq::Timeout::Resolve,
        },
    );
    assert!(matches!(second, Err(ureq::Error::Timeout(_))));
    assert!(immediate_timeout_started.elapsed() < Duration::from_millis(100));

    let saturated = resolver.resolve(
        &uri,
        &ureq::config::Config::default(),
        ureq::unversioned::transport::NextTimeout {
            after: ureq::unversioned::transport::time::Duration::from_millis(10),
            reason: ureq::Timeout::Resolve,
        },
    );
    assert!(matches!(
        saturated,
        Err(ureq::Error::Io(ref error))
            if error.kind() == std::io::ErrorKind::WouldBlock
    ));

    release.store(true, Ordering::Release);
    let resolved = loop {
        match resolver.resolve(
            &uri,
            &ureq::config::Config::default(),
            ureq::unversioned::transport::NextTimeout {
                after: ureq::unversioned::transport::time::Duration::from_millis(200),
                reason: ureq::Timeout::Resolve,
            },
        ) {
            Err(ureq::Error::Io(error)) if error.kind() == std::io::ErrorKind::WouldBlock => {
                std::thread::yield_now();
            }
            result => break result,
        }
    }
    .unwrap();
    assert_eq!(resolved.len(), 1);
    assert_eq!(calls.load(Ordering::Acquire), 2);
}

#[cfg(not(target_arch = "wasm32"))]
#[test]
fn native_http_pool_initialization_retries_after_transient_failure() {
    let pool = RetryableNativeHttpWorkerPool::new();
    let first = match native_http_worker_pool_with(&pool, || Err("injected failure".to_owned())) {
        Ok(_) => panic!("the injected native HTTP pool failure was unexpectedly accepted"),
        Err(error) => error,
    };
    assert_eq!(first.kind, HttpClientFailureKind::Network);
    assert!(first.retryable);
    assert!(first.message.contains("injected failure"));
    assert!(pool.initialized.get().is_none());

    let initialized =
        native_http_worker_pool_with(&pool, || create_native_http_worker_pool_with_limits(1, 1))
            .unwrap();
    let (completed_sender, completed_receiver) = std::sync::mpsc::sync_channel(1);
    initialized
        .sender
        .try_send(Box::new(move || completed_sender.send(()).unwrap()))
        .unwrap();
    completed_receiver.recv().unwrap();

    let reused = native_http_worker_pool_with(&pool, || {
        panic!("an initialized native HTTP pool must be reused")
    })
    .unwrap();
    assert!(std::ptr::eq(initialized, reused));
}

#[cfg(not(target_arch = "wasm32"))]
#[test]
fn process_http_pool_bounds_running_and_queued_jobs() {
    let pool = create_native_http_worker_pool_with_limits(1, 1).unwrap();
    let (started_sender, started_receiver) = std::sync::mpsc::sync_channel(1);
    let (release_sender, release_receiver) = std::sync::mpsc::sync_channel(1);
    pool.sender
        .try_send(Box::new(move || {
            started_sender.send(()).unwrap();
            release_receiver.recv().unwrap();
        }))
        .unwrap();
    started_receiver.recv().unwrap();
    pool.sender.try_send(Box::new(|| {})).unwrap();
    assert!(matches!(
        pool.sender.try_send(Box::new(|| {})),
        Err(std::sync::mpsc::TrySendError::Full(_))
    ));
    release_sender.send(()).unwrap();
}

#[cfg(not(target_arch = "wasm32"))]
#[test]
fn dropped_clients_remain_charged_to_the_process_http_pool() {
    use std::io::{Read as _, Write as _};

    let pool = create_native_http_worker_pool_with_limits(1, 1).unwrap();
    let listener = std::net::TcpListener::bind("127.0.0.1:0").unwrap();
    let address = listener.local_addr().unwrap();
    let (started_sender, started_receiver) = std::sync::mpsc::sync_channel(1);
    let (release_sender, release_receiver) = std::sync::mpsc::sync_channel(1);
    let server = std::thread::spawn(move || {
        let (mut socket, _) = listener.accept().unwrap();
        let mut request = [0_u8; 2048];
        let _ = socket.read(&mut request);
        started_sender.send(()).unwrap();
        release_receiver.recv().unwrap();
        let _ = socket.write_all(
            b"HTTP/1.1 200 OK\r\nContent-Length: 1\r\nETag: \"released\"\r\nConnection: close\r\n\r\nx",
        );
    });
    let request = HttpFetchRequest {
        url: format!("http://{address}/blocked"),
        byte_range: None,
        expected_bytes: 1,
        max_response_bytes: 1,
        timeout: Duration::from_secs(2),
        if_match: None,
        expected_version: None,
        object_version_header: None,
    };

    let mut running = NativeUreqHttpClient::new(Duration::from_secs(2)).unwrap();
    running.begin_with_pool(request.clone(), &pool).unwrap();
    started_receiver.recv().unwrap();
    drop(running);

    let mut queued = NativeUreqHttpClient::new(Duration::from_secs(2)).unwrap();
    queued.begin_with_pool(request.clone(), &pool).unwrap();
    drop(queued);

    let mut rejected = NativeUreqHttpClient::new(Duration::from_secs(2)).unwrap();
    assert_eq!(
        rejected.begin_with_pool(request, &pool).unwrap_err().kind,
        HttpClientFailureKind::ConcurrencyLimit
    );

    release_sender.send(()).unwrap();
    server.join().unwrap();
    let (drained_sender, drained_receiver) = std::sync::mpsc::sync_channel(1);
    loop {
        let sender = drained_sender.clone();
        match pool.sender.try_send(Box::new(move || {
            let _ = sender.send(());
        })) {
            Ok(()) => break,
            Err(std::sync::mpsc::TrySendError::Full(_)) => std::thread::yield_now(),
            Err(std::sync::mpsc::TrySendError::Disconnected(_)) => {
                panic!("native HTTP test pool disconnected")
            }
        }
    }
    drained_receiver.recv().unwrap();
}

#[cfg(not(target_arch = "wasm32"))]
#[test]
fn native_client_reads_local_range_without_public_network() {
    use std::io::{Read as _, Write as _};

    let listener = std::net::TcpListener::bind("127.0.0.1:0").unwrap();
    let address = listener.local_addr().unwrap();
    let server = std::thread::spawn(move || {
        let (mut socket, _) = listener.accept().unwrap();
        let mut request = [0_u8; 2048];
        let count = socket.read(&mut request).unwrap();
        let request = String::from_utf8_lossy(&request[..count]);
        assert!(request.contains("Range: bytes=2-5") || request.contains("range: bytes=2-5"));
        socket
            .write_all(
                b"HTTP/1.1 206 Partial Content\r\nContent-Length: 4\r\nContent-Range: bytes 2-5/8\r\nETag: \"local-v1\"\r\nConnection: close\r\n\r\n2345",
            )
            .unwrap();
    });
    let request = HttpFetchRequest {
        url: format!("http://{address}/pack"),
        byte_range: Some((2, 4)),
        expected_bytes: 4,
        max_response_bytes: 4,
        timeout: Duration::from_secs(2),
        if_match: None,
        expected_version: None,
        object_version_header: None,
    };
    let mut client = NativeUreqHttpClient::new(Duration::from_secs(2)).unwrap();
    let ticket = client.begin(request).unwrap();
    let response = loop {
        match client.poll(&ticket) {
            HttpClientPoll::Pending => std::thread::yield_now(),
            HttpClientPoll::Ready(response) => break response,
            HttpClientPoll::Failed(error) => panic!("local HTTP failed: {error:?}"),
        }
    };
    assert_eq!(response.bytes, b"2345");
    assert_eq!(response.etag.as_deref(), Some("\"local-v1\""));
    server.join().unwrap();
}

#[cfg(not(target_arch = "wasm32"))]
#[test]
fn native_request_timeout_overrides_the_agent_default() {
    use std::io::{Read as _, Write as _};

    let listener = std::net::TcpListener::bind("127.0.0.1:0").unwrap();
    let address = listener.local_addr().unwrap();
    let server = std::thread::spawn(move || {
        let (mut socket, _) = listener.accept().unwrap();
        let mut request = [0_u8; 2048];
        let _ = socket.read(&mut request);
        std::thread::sleep(Duration::from_millis(250));
        let _ = socket.write_all(
            b"HTTP/1.1 200 OK\r\nContent-Length: 1\r\nETag: \"late\"\r\nConnection: close\r\n\r\nx",
        );
    });
    let agent = native_ureq_agent(Duration::from_secs(2)).unwrap();
    let request = HttpFetchRequest {
        url: format!("http://{address}/slow"),
        byte_range: None,
        expected_bytes: 1,
        max_response_bytes: 1,
        timeout: Duration::from_millis(25),
        if_match: None,
        expected_version: None,
        object_version_header: None,
    };
    let started = Instant::now();
    assert!(fetch_with_ureq(&agent, request).is_err());
    assert!(started.elapsed() < Duration::from_secs(1));
    server.join().unwrap();
}

#[cfg(not(target_arch = "wasm32"))]
#[test]
fn cancelled_native_workers_remain_charged_until_reaped() {
    use std::{
        io::{Read as _, Write as _},
        sync::{
            Arc,
            atomic::{AtomicBool, AtomicU32, Ordering},
        },
    };

    let listener = std::net::TcpListener::bind("127.0.0.1:0").unwrap();
    let address = listener.local_addr().unwrap();
    let release = Arc::new(AtomicBool::new(false));
    let accepted = Arc::new(AtomicU32::new(0));
    let server_release = release.clone();
    let server_accepted = accepted.clone();
    let server = std::thread::spawn(move || {
        let mut handlers = Vec::new();
        for index in 0..3 {
            let (mut socket, _) = listener.accept().unwrap();
            server_accepted.fetch_add(1, Ordering::Release);
            let handler_release = server_release.clone();
            handlers.push(std::thread::spawn(move || {
                let mut request = [0_u8; 2048];
                let _ = socket.read(&mut request);
                if index < 2 {
                    while !handler_release.load(Ordering::Acquire) {
                        std::thread::yield_now();
                    }
                }
                socket
                    .write_all(
                        b"HTTP/1.1 200 OK\r\nContent-Length: 1\r\nETag: \"bounded-v1\"\r\nConnection: close\r\n\r\nx",
                    )
                    .unwrap();
            }));
        }
        for handler in handlers {
            handler.join().unwrap();
        }
    });
    let request = HttpFetchRequest {
        url: format!("http://{address}/page"),
        byte_range: None,
        expected_bytes: 1,
        max_response_bytes: 1,
        timeout: Duration::from_secs(2),
        if_match: None,
        expected_version: None,
        object_version_header: None,
    };
    let mut client = NativeUreqHttpClient::with_max_workers(Duration::from_secs(2), 2).unwrap();
    let first = client.begin(request.clone()).unwrap();
    let second = client.begin(request.clone()).unwrap();
    while accepted.load(Ordering::Acquire) < 2 {
        std::thread::yield_now();
    }
    client.cancel(&first);
    client.cancel(&second);
    let saturated = client.begin(request.clone()).unwrap_err();
    assert_eq!(saturated.kind, HttpClientFailureKind::ConcurrencyLimit);

    release.store(true, Ordering::Release);
    let third = loop {
        match client.begin(request.clone()) {
            Ok(ticket) => break ticket,
            Err(error) if error.kind == HttpClientFailureKind::ConcurrencyLimit => {
                std::thread::yield_now()
            }
            Err(error) => panic!("unexpected worker admission error: {error:?}"),
        }
    };
    loop {
        match client.poll(&third) {
            HttpClientPoll::Pending => std::thread::yield_now(),
            HttpClientPoll::Ready(response) => {
                assert_eq!(response.bytes, b"x");
                break;
            }
            HttpClientPoll::Failed(error) => panic!("third worker failed: {error:?}"),
        }
    }
    server.join().unwrap();
}

use super::*;

/// Native HTTP client. Each request uses a bounded worker while the clonable
/// `ureq` agent retains its connection pool across requests.
pub struct NativeUreqHttpClient {
    agent: ureq::Agent,
    workers: BTreeMap<u64, NativeUreqWorker>,
    max_workers: u32,
    next_ticket: u64,
}

impl Drop for NativeUreqHttpClient {
    fn drop(&mut self) {
        for worker in self.workers.values() {
            worker
                .cancelled
                .store(true, std::sync::atomic::Ordering::Relaxed);
        }
    }
}

struct NativeUreqWorker {
    receiver: std::sync::mpsc::Receiver<Result<HttpFetchResponse, HttpClientFailure>>,
    cancelled: std::sync::Arc<std::sync::atomic::AtomicBool>,
}

pub(super) type NativeHttpJob = Box<dyn FnOnce() + Send + 'static>;

/// Process-wide bounds retained across package/client teardown. Requests use a
/// shared pool so dropping a client cannot detach an unaccounted OS thread.
pub const NATIVE_HTTP_GLOBAL_WORKERS: usize = 32;
pub const NATIVE_HTTP_GLOBAL_QUEUE_CAPACITY: usize = 256;

/// Process-wide DNS bounds. System name resolution is blocking on the
/// platforms supported by `std`, so DNS runs on its own fixed pool instead of
/// ureq's default per-resolution timeout thread.
pub const NATIVE_DNS_GLOBAL_WORKERS: usize = 8;
pub const NATIVE_DNS_GLOBAL_QUEUE_CAPACITY: usize = 64;

pub(super) struct NativeHttpWorkerPool {
    pub(super) sender: std::sync::mpsc::SyncSender<NativeHttpJob>,
}

pub(super) struct RetryableNativeHttpWorkerPool {
    pub(super) initialized: std::sync::OnceLock<NativeHttpWorkerPool>,
    initialization: std::sync::Mutex<()>,
}

impl RetryableNativeHttpWorkerPool {
    pub(super) const fn new() -> Self {
        Self {
            initialized: std::sync::OnceLock::new(),
            initialization: std::sync::Mutex::new(()),
        }
    }

    fn get_or_try_init(
        &self,
        create: impl FnOnce() -> Result<NativeHttpWorkerPool, String>,
    ) -> Result<&NativeHttpWorkerPool, HttpClientFailure> {
        if let Some(pool) = self.initialized.get() {
            return Ok(pool);
        }

        let _initialization_guard = self
            .initialization
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner);
        if let Some(pool) = self.initialized.get() {
            return Ok(pool);
        }

        let pool = create().map_err(native_http_worker_pool_failure)?;
        if self.initialized.set(pool).is_err() {
            return self.initialized.get().ok_or_else(|| {
                native_http_worker_pool_failure(
                    "native HTTP worker pool publication raced without a winner".to_owned(),
                )
            });
        }
        self.initialized.get().ok_or_else(|| {
            native_http_worker_pool_failure(
                "native HTTP worker pool was not visible after publication".to_owned(),
            )
        })
    }
}

fn native_http_worker_pool() -> Result<&'static NativeHttpWorkerPool, HttpClientFailure> {
    static POOL: RetryableNativeHttpWorkerPool = RetryableNativeHttpWorkerPool::new();
    native_http_worker_pool_with(&POOL, create_native_http_worker_pool)
}

pub(super) fn native_http_worker_pool_with(
    pool: &RetryableNativeHttpWorkerPool,
    create: impl FnOnce() -> Result<NativeHttpWorkerPool, String>,
) -> Result<&NativeHttpWorkerPool, HttpClientFailure> {
    pool.get_or_try_init(create)
}

fn native_http_worker_pool_failure(error: String) -> HttpClientFailure {
    HttpClientFailure::new(
        HttpClientFailureKind::Network,
        format!("native HTTP worker pool is unavailable: {error}"),
        true,
    )
}

fn create_native_http_worker_pool() -> Result<NativeHttpWorkerPool, String> {
    create_native_http_worker_pool_with_limits(
        NATIVE_HTTP_GLOBAL_WORKERS,
        NATIVE_HTTP_GLOBAL_QUEUE_CAPACITY,
    )
}

pub(super) fn create_native_http_worker_pool_with_limits(
    worker_count: usize,
    queue_capacity: usize,
) -> Result<NativeHttpWorkerPool, String> {
    if worker_count == 0 || queue_capacity == 0 {
        return Err("native HTTP worker and queue limits must be nonzero".to_owned());
    }
    let (sender, receiver) = std::sync::mpsc::sync_channel::<NativeHttpJob>(queue_capacity);
    let receiver = std::sync::Arc::new(std::sync::Mutex::new(receiver));
    for worker in 0..worker_count {
        let receiver = receiver.clone();
        std::thread::Builder::new()
            .name(format!("gaussian-lod-http-{worker}"))
            .spawn(move || {
                loop {
                    let job = {
                        let Ok(receiver) = receiver.lock() else {
                            return;
                        };
                        receiver.recv()
                    };
                    let Ok(job) = job else {
                        return;
                    };
                    // A malformed direct-client request or a future backend
                    // regression must not permanently shrink this process-wide
                    // fixed worker pool.
                    let _ = std::panic::catch_unwind(std::panic::AssertUnwindSafe(job));
                }
            })
            .map_err(|error| format!("failed to spawn worker {worker}: {error}"))?;
    }
    Ok(NativeHttpWorkerPool { sender })
}

pub(super) type NativeDnsLookup = std::sync::Arc<
    dyn Fn(&str) -> std::io::Result<Vec<std::net::SocketAddr>> + Send + Sync + 'static,
>;

struct NativeDnsJob {
    address: String,
    cancelled: std::sync::Arc<std::sync::atomic::AtomicBool>,
    response: std::sync::mpsc::SyncSender<std::io::Result<Vec<std::net::SocketAddr>>>,
}

#[derive(Clone)]
pub(super) struct NativeDnsResolverPool {
    sender: std::sync::mpsc::SyncSender<NativeDnsJob>,
    queue_capacity: usize,
}

struct RetryableNativeDnsResolverPool {
    initialized: std::sync::OnceLock<NativeDnsResolverPool>,
    initialization: std::sync::Mutex<()>,
}

impl RetryableNativeDnsResolverPool {
    const fn new() -> Self {
        Self {
            initialized: std::sync::OnceLock::new(),
            initialization: std::sync::Mutex::new(()),
        }
    }

    fn get_or_try_init(
        &self,
        create: impl FnOnce() -> Result<NativeDnsResolverPool, String>,
    ) -> Result<&NativeDnsResolverPool, String> {
        if let Some(pool) = self.initialized.get() {
            return Ok(pool);
        }

        let _initialization_guard = self
            .initialization
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner);
        if let Some(pool) = self.initialized.get() {
            return Ok(pool);
        }

        let pool = create()?;
        if self.initialized.set(pool).is_err() {
            return self.initialized.get().ok_or_else(|| {
                "native DNS resolver pool publication raced without a winner".to_owned()
            });
        }
        self.initialized
            .get()
            .ok_or_else(|| "native DNS resolver pool was not visible after publication".to_owned())
    }
}

fn native_dns_resolver_pool() -> Result<&'static NativeDnsResolverPool, HttpRangeTransportError> {
    static POOL: RetryableNativeDnsResolverPool = RetryableNativeDnsResolverPool::new();
    native_dns_resolver_pool_with(&POOL, create_native_dns_resolver_pool)
        .map_err(HttpRangeTransportError::NativeResolverUnavailable)
}

fn native_dns_resolver_pool_with(
    pool: &RetryableNativeDnsResolverPool,
    create: impl FnOnce() -> Result<NativeDnsResolverPool, String>,
) -> Result<&NativeDnsResolverPool, String> {
    pool.get_or_try_init(create)
}

fn create_native_dns_resolver_pool() -> Result<NativeDnsResolverPool, String> {
    use std::net::ToSocketAddrs as _;

    create_native_dns_resolver_pool_with_limits(
        NATIVE_DNS_GLOBAL_WORKERS,
        NATIVE_DNS_GLOBAL_QUEUE_CAPACITY,
        std::sync::Arc::new(|address| address.to_socket_addrs().map(Iterator::collect)),
    )
}

pub(super) fn create_native_dns_resolver_pool_with_limits(
    worker_count: usize,
    queue_capacity: usize,
    lookup: NativeDnsLookup,
) -> Result<NativeDnsResolverPool, String> {
    if worker_count == 0 || queue_capacity == 0 {
        return Err("native DNS worker and queue limits must be nonzero".to_owned());
    }
    let (sender, receiver) = std::sync::mpsc::sync_channel::<NativeDnsJob>(queue_capacity);
    let receiver = std::sync::Arc::new(std::sync::Mutex::new(receiver));
    for worker in 0..worker_count {
        let receiver = receiver.clone();
        let lookup = lookup.clone();
        std::thread::Builder::new()
            .name(format!("gaussian-lod-dns-{worker}"))
            .spawn(move || {
                loop {
                    let job = {
                        let Ok(receiver) = receiver.lock() else {
                            return;
                        };
                        receiver.recv()
                    };
                    let Ok(job) = job else {
                        return;
                    };
                    if job.cancelled.load(std::sync::atomic::Ordering::Acquire) {
                        continue;
                    }
                    let resolved = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
                        lookup(&job.address)
                    }))
                    .unwrap_or_else(|_| {
                        Err(std::io::Error::other("native DNS resolver worker panicked"))
                    });
                    if !job.cancelled.load(std::sync::atomic::Ordering::Acquire) {
                        let _ = job.response.try_send(resolved);
                    }
                }
            })
            .map_err(|error| format!("failed to spawn DNS worker {worker}: {error}"))?;
    }
    Ok(NativeDnsResolverPool {
        sender,
        queue_capacity,
    })
}

#[derive(Clone)]
pub(super) struct BoundedNativeResolver {
    pub(super) pool: NativeDnsResolverPool,
}

impl fmt::Debug for BoundedNativeResolver {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter
            .debug_struct("BoundedNativeResolver")
            .finish_non_exhaustive()
    }
}

impl ureq::unversioned::resolver::Resolver for BoundedNativeResolver {
    fn resolve(
        &self,
        uri: &ureq::http::Uri,
        config: &ureq::config::Config,
        timeout: ureq::unversioned::transport::NextTimeout,
    ) -> Result<ureq::unversioned::resolver::ResolvedSocketAddrs, ureq::Error> {
        use ureq::unversioned::resolver::DefaultResolver;

        let scheme = uri
            .scheme()
            .ok_or_else(|| ureq::Error::BadUri(format!("{uri} is missing scheme")))?;
        let authority = uri
            .authority()
            .ok_or_else(|| ureq::Error::BadUri(format!("{uri} is missing host")))?;
        let address = DefaultResolver::host_and_port(scheme, authority)
            .ok_or_else(|| ureq::Error::BadUri(format!("{uri} has no usable port")))?;
        let cancelled = std::sync::Arc::new(std::sync::atomic::AtomicBool::new(false));
        let (response, receiver) = std::sync::mpsc::sync_channel(1);
        let job = NativeDnsJob {
            address,
            cancelled: cancelled.clone(),
            response,
        };
        self.pool.sender.try_send(job).map_err(|error| {
            let (kind, message) = match error {
                std::sync::mpsc::TrySendError::Full(_) => (
                    std::io::ErrorKind::WouldBlock,
                    format!(
                        "native DNS queue capacity {} is fully charged",
                        self.pool.queue_capacity
                    ),
                ),
                std::sync::mpsc::TrySendError::Disconnected(_) => (
                    std::io::ErrorKind::BrokenPipe,
                    "native DNS resolver pool disconnected".to_owned(),
                ),
            };
            ureq::Error::Io(std::io::Error::new(kind, message))
        })?;

        let received = if timeout.after.is_not_happening() {
            receiver.recv().map_err(|_| {
                ureq::Error::Io(std::io::Error::new(
                    std::io::ErrorKind::BrokenPipe,
                    "native DNS resolver worker disconnected",
                ))
            })?
        } else {
            receiver
                .recv_timeout(*timeout.after)
                .map_err(|error| match error {
                    std::sync::mpsc::RecvTimeoutError::Timeout => {
                        cancelled.store(true, std::sync::atomic::Ordering::Release);
                        ureq::Error::Timeout(timeout.reason)
                    }
                    std::sync::mpsc::RecvTimeoutError::Disconnected => {
                        ureq::Error::Io(std::io::Error::new(
                            std::io::ErrorKind::BrokenPipe,
                            "native DNS resolver worker disconnected",
                        ))
                    }
                })?
        };
        cancelled.store(true, std::sync::atomic::Ordering::Release);
        let addresses = received.map_err(ureq::Error::Io)?;
        let mut resolved = self.empty();
        for address in config
            .ip_family()
            .keep_wanted(addresses.into_iter())
            .take(16)
        {
            resolved.push(address);
        }
        if resolved.is_empty() {
            Err(ureq::Error::HostNotFound)
        } else {
            Ok(resolved)
        }
    }
}

pub(super) fn native_ureq_agent(
    request_timeout: Duration,
) -> Result<ureq::Agent, HttpRangeTransportError> {
    let config = ureq::Agent::config_builder()
        .timeout_global(Some(request_timeout))
        .http_status_as_error(false)
        .max_redirects(0)
        .build();
    let resolver = BoundedNativeResolver {
        pool: native_dns_resolver_pool()?.clone(),
    };
    Ok(ureq::Agent::with_parts(
        config,
        ureq::unversioned::transport::DefaultConnector::default(),
        resolver,
    ))
}

impl NativeUreqHttpClient {
    const DEFAULT_MAX_WORKERS: u32 = 32;

    pub fn new(request_timeout: Duration) -> Result<Self, HttpRangeTransportError> {
        Self::with_max_workers(request_timeout, Self::DEFAULT_MAX_WORKERS)
    }

    pub fn with_max_workers(
        request_timeout: Duration,
        max_workers: u32,
    ) -> Result<Self, HttpRangeTransportError> {
        validate_request_timeout(request_timeout)?;
        if max_workers == 0 {
            return Err(HttpRangeTransportError::ZeroMaxWorkers);
        }
        let agent = native_ureq_agent(request_timeout)?;
        Ok(Self {
            agent,
            workers: BTreeMap::new(),
            max_workers,
            next_ticket: 1,
        })
    }

    fn reap_cancelled_workers(&mut self) {
        self.workers.retain(|_, worker| {
            !worker.cancelled.load(std::sync::atomic::Ordering::Relaxed)
                || matches!(
                    worker.receiver.try_recv(),
                    Err(std::sync::mpsc::TryRecvError::Empty)
                )
        });
    }

    pub(super) fn begin_with_pool(
        &mut self,
        request: HttpFetchRequest,
        pool: &NativeHttpWorkerPool,
    ) -> Result<u64, HttpClientFailure> {
        validate_fetch_request_timeout(request.timeout)?;
        self.reap_cancelled_workers();
        if self.workers.len() >= self.max_workers as usize {
            return Err(HttpClientFailure::new(
                HttpClientFailureKind::ConcurrencyLimit,
                format!(
                    "native HTTP worker limit {} is fully charged (including cancelled in-flight workers)",
                    self.max_workers
                ),
                true,
            ));
        }
        let agent = self.agent.clone();
        let cancelled = std::sync::Arc::new(std::sync::atomic::AtomicBool::new(false));
        let cancelled_for_job = cancelled.clone();
        let (sender, receiver) = std::sync::mpsc::sync_channel(1);
        let job: NativeHttpJob = Box::new(move || {
            let result = if cancelled_for_job.load(std::sync::atomic::Ordering::Relaxed) {
                Err(HttpClientFailure::new(
                    HttpClientFailureKind::Cancelled,
                    "native HTTP request was cancelled before execution",
                    false,
                ))
            } else {
                fetch_with_ureq(&agent, request)
            };
            let _ = sender.send(result);
        });
        pool.sender.try_send(job).map_err(|error| match error {
            std::sync::mpsc::TrySendError::Full(_) => HttpClientFailure::new(
                HttpClientFailureKind::ConcurrencyLimit,
                format!(
                    "native HTTP global queue capacity {} is fully charged",
                    NATIVE_HTTP_GLOBAL_QUEUE_CAPACITY
                ),
                true,
            ),
            std::sync::mpsc::TrySendError::Disconnected(_) => HttpClientFailure::new(
                HttpClientFailureKind::Network,
                "native HTTP worker pool disconnected",
                true,
            ),
        })?;
        let ticket = self.next_ticket;
        self.next_ticket = self.next_ticket.wrapping_add(1).max(1);
        self.workers.insert(
            ticket,
            NativeUreqWorker {
                receiver,
                cancelled,
            },
        );
        Ok(ticket)
    }
}

impl HttpRangeClient for NativeUreqHttpClient {
    type Ticket = u64;

    fn begin(&mut self, request: HttpFetchRequest) -> Result<Self::Ticket, HttpClientFailure> {
        let pool = native_http_worker_pool()?;
        self.begin_with_pool(request, pool)
    }

    fn poll(&mut self, ticket: &Self::Ticket) -> HttpClientPoll {
        let Some(worker) = self.workers.get(ticket) else {
            return HttpClientPoll::Failed(HttpClientFailure::new(
                HttpClientFailureKind::InvalidRequest,
                format!("invalid native HTTP ticket {ticket}"),
                false,
            ));
        };
        if worker.cancelled.load(std::sync::atomic::Ordering::Relaxed) {
            return HttpClientPoll::Failed(HttpClientFailure::new(
                HttpClientFailureKind::Cancelled,
                format!("native HTTP ticket {ticket} was cancelled"),
                false,
            ));
        }
        match worker.receiver.try_recv() {
            Ok(Ok(response)) => {
                self.workers.remove(ticket);
                HttpClientPoll::Ready(response)
            }
            Ok(Err(error)) => {
                self.workers.remove(ticket);
                HttpClientPoll::Failed(error)
            }
            Err(std::sync::mpsc::TryRecvError::Empty) => HttpClientPoll::Pending,
            Err(std::sync::mpsc::TryRecvError::Disconnected) => {
                self.workers.remove(ticket);
                HttpClientPoll::Failed(HttpClientFailure::new(
                    HttpClientFailureKind::Network,
                    "native HTTP worker disconnected",
                    true,
                ))
            }
        }
    }

    fn cancel(&mut self, ticket: &Self::Ticket) {
        // ureq cannot abort an in-flight blocking call. Keep the worker charged
        // until it actually exits so cancellation churn cannot create an
        // unbounded population of detached threads within the timeout window.
        if let Some(worker) = self.workers.get_mut(ticket) {
            worker
                .cancelled
                .store(true, std::sync::atomic::Ordering::Relaxed);
        }
        self.reap_cancelled_workers();
    }
}

pub(super) fn fetch_with_ureq(
    agent: &ureq::Agent,
    request: HttpFetchRequest,
) -> Result<HttpFetchResponse, HttpClientFailure> {
    let mut builder = agent
        .get(&request.url)
        .config()
        .timeout_global(Some(request.timeout))
        .build()
        .header("Accept-Encoding", "identity");
    if let Some((start, len)) = request.byte_range {
        let end = byte_range_end(start, len).ok_or_else(|| {
            HttpClientFailure::new(
                HttpClientFailureKind::InvalidRequest,
                "HTTP byte range overflow",
                false,
            )
        })?;
        builder = builder.header("Range", format!("bytes={start}-{end}"));
    }
    if let Some(etag) = request.if_match.as_deref() {
        builder = builder.header("If-Match", etag);
    }
    let mut response = builder.call().map_err(map_ureq_error)?;
    let status = response.status().as_u16();
    let content_length = response
        .headers()
        .get("content-length")
        .and_then(|value| value.to_str().ok())
        .and_then(|value| value.parse().ok());
    let content_range = header_string(response.headers(), "content-range");
    let content_encoding = header_string(response.headers(), "content-encoding");
    let etag = header_string(response.headers(), "etag");
    let object_version = request
        .object_version_header
        .as_deref()
        .and_then(|header| header_string(response.headers(), header));
    let retry_after = header_string(response.headers(), "retry-after")
        .and_then(|value| value.parse::<u64>().ok())
        .map(Duration::from_secs);
    if is_retryable_status(status) {
        return Ok(HttpFetchResponse {
            status,
            redirected: false,
            bytes: Vec::new(),
            content_length,
            content_range,
            content_encoding,
            etag,
            object_version,
            retry_after,
        });
    }
    if content_length.is_some_and(|length| length > request.max_response_bytes) {
        return Err(HttpClientFailure::new(
            HttpClientFailureKind::ResponseTooLarge,
            format!(
                "HTTP Content-Length exceeds {} byte bound",
                request.max_response_bytes
            ),
            false,
        ));
    }
    let body_limit = request.max_response_bytes.checked_add(1).ok_or_else(|| {
        HttpClientFailure::new(
            HttpClientFailureKind::InvalidRequest,
            "HTTP body limit does not fit address space",
            false,
        )
    })?;
    let bytes = response
        .body_mut()
        .with_config()
        .limit(body_limit)
        .read_to_vec()
        .map_err(map_ureq_error)?;
    if bytes.len() as u64 > request.max_response_bytes {
        return Err(HttpClientFailure::new(
            HttpClientFailureKind::ResponseTooLarge,
            format!(
                "HTTP body exceeds {} byte bound",
                request.max_response_bytes
            ),
            false,
        ));
    }
    Ok(HttpFetchResponse {
        status,
        redirected: false,
        bytes,
        content_length,
        content_range,
        content_encoding,
        etag,
        object_version,
        retry_after,
    })
}

fn header_string(headers: &ureq::http::HeaderMap, name: &str) -> Option<String> {
    headers
        .get(name)
        .and_then(|value| value.to_str().ok())
        .map(str::to_owned)
}

fn map_ureq_error(error: ureq::Error) -> HttpClientFailure {
    use ureq::Error;
    let kind = match &error {
        Error::Timeout(_) => HttpClientFailureKind::Timeout,
        Error::Io(_)
        | Error::HostNotFound
        | Error::ConnectionFailed
        | Error::Protocol(_)
        | Error::BodyStalled => HttpClientFailureKind::Network,
        Error::BodyExceedsLimit(_) => HttpClientFailureKind::ResponseTooLarge,
        Error::Http(_) | Error::BadUri(_) | Error::RequireHttpsOnly(_) => {
            HttpClientFailureKind::InvalidRequest
        }
        _ => HttpClientFailureKind::Protocol,
    };
    let retryable = matches!(
        kind,
        HttpClientFailureKind::Network | HttpClientFailureKind::Timeout
    );
    HttpClientFailure::new(kind, error.to_string(), retryable)
}

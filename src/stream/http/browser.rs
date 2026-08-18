use super::*;

/// Browser fetch client. Its public contract is identical to the native client,
/// so package/runtime code does not depend on JS promises or a futures runtime.
///
/// A realm-wide task budget remains charged until each Fetch promise settles,
/// including after the public ticket is cancelled or its client is dropped.
/// This prevents rapid client churn from accumulating an unbounded number of
/// aborting promises behind the browser event loop.
pub struct BrowserFetchHttpClient {
    tickets: BTreeMap<u64, BrowserFetchTicket>,
    max_requests: u32,
    next_ticket: u64,
}

struct BrowserFetchTicket {
    abort: web_sys::AbortController,
    shared: std::rc::Rc<BrowserFetchShared>,
    deadline_timer: BrowserFetchTimerSlot,
}

struct BrowserFetchShared {
    published: std::cell::Cell<bool>,
    result: std::cell::RefCell<Option<Result<HttpFetchResponse, HttpClientFailure>>>,
}

impl BrowserFetchShared {
    fn new() -> Self {
        Self {
            published: std::cell::Cell::new(false),
            result: std::cell::RefCell::new(None),
        }
    }

    fn publish_once(&self, result: Result<HttpFetchResponse, HttpClientFailure>) {
        if !self.published.replace(true) {
            *self.result.borrow_mut() = Some(result);
        }
    }

    fn take_result(&self) -> Option<Result<HttpFetchResponse, HttpClientFailure>> {
        self.result.borrow_mut().take()
    }

    fn abandon(&self) {
        self.published.set(true);
        self.result.borrow_mut().take();
    }
}

type BrowserFetchTimerSlot = std::rc::Rc<std::cell::RefCell<Option<BrowserFetchDeadlineTimer>>>;

struct BrowserFetchDeadlineTimer {
    window: web_sys::Window,
    handle: Option<i32>,
    callback: wasm_bindgen::closure::Closure<dyn FnMut()>,
}

impl Drop for BrowserFetchDeadlineTimer {
    fn drop(&mut self) {
        if let Some(handle) = self.handle.take() {
            self.window.clear_timeout_with_handle(handle);
        }
    }
}

/// Hard upper bound for unsettled browser Fetch tasks in one JavaScript realm.
pub const BROWSER_HTTP_GLOBAL_TASK_CAPACITY: u32 = 256;

#[cfg(test)]
const BROWSER_HTTP_EFFECTIVE_TASK_CAPACITY: u32 = 4;
#[cfg(not(test))]
const BROWSER_HTTP_EFFECTIVE_TASK_CAPACITY: u32 = BROWSER_HTTP_GLOBAL_TASK_CAPACITY;

std::thread_local! {
    static BROWSER_HTTP_UNSETTLED_TASKS: std::cell::Cell<u32> = const {
        std::cell::Cell::new(0)
    };
}

struct BrowserHttpTaskPermit;

impl BrowserHttpTaskPermit {
    fn acquire() -> Result<Self, HttpClientFailure> {
        BROWSER_HTTP_UNSETTLED_TASKS.with(|active| {
            let current = active.get();
            if current >= BROWSER_HTTP_EFFECTIVE_TASK_CAPACITY {
                return Err(HttpClientFailure::new(
                    HttpClientFailureKind::ConcurrencyLimit,
                    format!(
                        "browser HTTP realm task limit {} is fully charged",
                        BROWSER_HTTP_EFFECTIVE_TASK_CAPACITY
                    ),
                    true,
                ));
            }
            active.set(current + 1);
            Ok(Self)
        })
    }
}

impl Drop for BrowserHttpTaskPermit {
    fn drop(&mut self) {
        BROWSER_HTTP_UNSETTLED_TASKS.with(|active| {
            let current = active.get();
            debug_assert!(current > 0, "browser HTTP task budget underflow");
            active.set(current.saturating_sub(1));
        });
    }
}

#[cfg(test)]
pub(crate) fn browser_http_unsettled_tasks_for_testing() -> u32 {
    BROWSER_HTTP_UNSETTLED_TASKS.with(std::cell::Cell::get)
}

impl Drop for BrowserFetchHttpClient {
    fn drop(&mut self) {
        for state in std::mem::take(&mut self.tickets).into_values() {
            state.shared.abandon();
            clear_browser_fetch_timer(&state.deadline_timer);
            state.abort.abort();
        }
    }
}

impl Default for BrowserFetchHttpClient {
    fn default() -> Self {
        Self {
            tickets: BTreeMap::new(),
            max_requests: 32,
            next_ticket: 1,
        }
    }
}

impl BrowserFetchHttpClient {
    pub fn with_max_requests(max_requests: u32) -> Result<Self, HttpRangeTransportError> {
        if max_requests == 0 {
            return Err(HttpRangeTransportError::ZeroMaxConcurrentRequests);
        }
        Ok(Self {
            tickets: BTreeMap::new(),
            max_requests,
            next_ticket: 1,
        })
    }
}

impl HttpRangeClient for BrowserFetchHttpClient {
    type Ticket = u64;

    fn begin(&mut self, request: HttpFetchRequest) -> Result<Self::Ticket, HttpClientFailure> {
        use wasm_bindgen::JsCast as _;

        validate_fetch_request_timeout(request.timeout)?;
        if self.tickets.len() >= self.max_requests as usize {
            return Err(HttpClientFailure::new(
                HttpClientFailureKind::ConcurrencyLimit,
                format!(
                    "browser HTTP request limit {} is fully charged",
                    self.max_requests
                ),
                true,
            ));
        }

        let abort = web_sys::AbortController::new().map_err(js_client_failure)?;
        let init = web_sys::RequestInit::new();
        init.set_method("GET");
        init.set_signal(Some(&abort.signal()));
        let browser_request = web_sys::Request::new_with_str_and_init(&request.url, &init)
            .map_err(js_client_failure)?;
        // `Accept-Encoding` is a forbidden browser request header. The server
        // must serve package objects with identity encoding; response
        // validation below rejects any content encoding that would make byte
        // ranges refer to a different representation.
        if let Some((start, len)) = request.byte_range {
            let end = byte_range_end(start, len).ok_or_else(|| {
                HttpClientFailure::new(
                    HttpClientFailureKind::InvalidRequest,
                    "HTTP byte range overflow",
                    false,
                )
            })?;
            browser_request
                .headers()
                .set("Range", &format!("bytes={start}-{end}"))
                .map_err(js_client_failure)?;
        }
        if let Some(etag) = request.if_match.as_deref() {
            browser_request
                .headers()
                .set("If-Match", etag)
                .map_err(js_client_failure)?;
        }
        let window = web_sys::window().ok_or_else(|| {
            HttpClientFailure::new(
                HttpClientFailureKind::InvalidRequest,
                "browser window is unavailable",
                false,
            )
        })?;
        let task_permit = BrowserHttpTaskPermit::acquire()?;
        let shared = std::rc::Rc::new(BrowserFetchShared::new());
        let deadline_timer =
            start_browser_fetch_timer(&window, request.timeout, abort.clone(), shared.clone())?;
        let shared_for_task = shared.clone();
        let timer_for_task = deadline_timer.clone();
        wasm_bindgen_futures::spawn_local(async move {
            let _task_permit = task_permit;
            let fetched = async {
                let value = wasm_bindgen_futures::JsFuture::from(
                    window.fetch_with_request(&browser_request),
                )
                .await
                .map_err(js_client_failure)?;
                let response = value
                    .dyn_into::<web_sys::Response>()
                    .map_err(js_client_failure)?;
                browser_fetch_response(response, &request).await
            }
            .await;
            // Clearing the deadline and publishing happen before the task
            // permit is released. If the timer already published Timeout,
            // publish_once deliberately leaves that caller-visible result in
            // place while this task still accounts for the unsettled Fetch or
            // cancellation promise.
            clear_browser_fetch_timer(&timer_for_task);
            shared_for_task.publish_once(fetched);
        });
        let ticket = self.next_ticket;
        self.next_ticket = self.next_ticket.wrapping_add(1).max(1);
        self.tickets.insert(
            ticket,
            BrowserFetchTicket {
                abort,
                shared,
                deadline_timer,
            },
        );
        Ok(ticket)
    }

    fn poll(&mut self, ticket: &Self::Ticket) -> HttpClientPoll {
        let Some(state) = self.tickets.get(ticket) else {
            return HttpClientPoll::Failed(HttpClientFailure::new(
                HttpClientFailureKind::InvalidRequest,
                format!("invalid browser fetch ticket {ticket}"),
                false,
            ));
        };
        let result = state.shared.take_result();
        match result {
            None => HttpClientPoll::Pending,
            Some(Ok(response)) => {
                if let Some(state) = self.tickets.remove(ticket) {
                    clear_browser_fetch_timer(&state.deadline_timer);
                }
                HttpClientPoll::Ready(response)
            }
            Some(Err(error)) => {
                if let Some(state) = self.tickets.remove(ticket) {
                    clear_browser_fetch_timer(&state.deadline_timer);
                }
                HttpClientPoll::Failed(error)
            }
        }
    }

    fn cancel(&mut self, ticket: &Self::Ticket) {
        if let Some(state) = self.tickets.remove(ticket) {
            state.shared.abandon();
            clear_browser_fetch_timer(&state.deadline_timer);
            state.abort.abort();
        }
    }
}

fn start_browser_fetch_timer(
    window: &web_sys::Window,
    timeout: Duration,
    abort: web_sys::AbortController,
    shared: std::rc::Rc<BrowserFetchShared>,
) -> Result<BrowserFetchTimerSlot, HttpClientFailure> {
    use wasm_bindgen::JsCast as _;

    let deadline = Instant::now().checked_add(timeout).ok_or_else(|| {
        HttpClientFailure::new(
            HttpClientFailureKind::InvalidRequest,
            format!("browser HTTP timeout {timeout:?} has no representable deadline"),
            false,
        )
    })?;
    let slot: BrowserFetchTimerSlot = std::rc::Rc::new(std::cell::RefCell::new(None));
    let weak_slot = std::rc::Rc::downgrade(&slot);
    let callback = wasm_bindgen::closure::Closure::new(move || {
        let Some(slot) = weak_slot.upgrade() else {
            return;
        };
        let now = Instant::now();
        let mut timer_slot = slot.borrow_mut();
        let Some(timer) = timer_slot.as_mut() else {
            return;
        };
        // The callback corresponding to this handle is executing now.
        timer.handle = None;
        if now >= deadline {
            drop(timer_slot);
            abort.abort();
            shared.publish_once(Err(HttpClientFailure::new(
                HttpClientFailureKind::Timeout,
                format!("browser HTTP request exceeded {timeout:?} timeout"),
                true,
            )));
            return;
        }

        let delay = browser_timer_delay_ms(deadline.saturating_duration_since(now));
        let rearmed = timer
            .window
            .set_timeout_with_callback_and_timeout_and_arguments_0(
                timer.callback.as_ref().unchecked_ref(),
                delay,
            );
        match rearmed {
            Ok(handle) => timer.handle = Some(handle),
            Err(value) => {
                drop(timer_slot);
                abort.abort();
                shared.publish_once(Err(js_client_failure(value)));
            }
        }
    });
    let delay = browser_timer_delay_ms(timeout);
    let handle = window
        .set_timeout_with_callback_and_timeout_and_arguments_0(
            callback.as_ref().unchecked_ref(),
            delay,
        )
        .map_err(js_client_failure)?;
    *slot.borrow_mut() = Some(BrowserFetchDeadlineTimer {
        window: window.clone(),
        handle: Some(handle),
        callback,
    });
    Ok(slot)
}

fn clear_browser_fetch_timer(slot: &BrowserFetchTimerSlot) {
    slot.borrow_mut().take();
}

async fn browser_fetch_response(
    response: web_sys::Response,
    request: &HttpFetchRequest,
) -> Result<HttpFetchResponse, HttpClientFailure> {
    let body_for_cancel = response.body();
    let result = browser_fetch_response_inner(response, request).await;
    // This is a no-op after a fully consumed stream, and actively stops bodies
    // skipped because of retry status, invalid headers, redirects, or limits.
    if let Some(body) = body_for_cancel {
        let _ = wasm_bindgen_futures::JsFuture::from(body.cancel()).await;
    }
    result
}

async fn browser_fetch_response_inner(
    response: web_sys::Response,
    request: &HttpFetchRequest,
) -> Result<HttpFetchResponse, HttpClientFailure> {
    let headers = response.headers();
    let content_length =
        browser_header(&headers, "content-length")?.and_then(|value| value.parse().ok());
    let status = response.status();
    let redirected = response.redirected();
    let content_range = browser_header(&headers, "content-range")?;
    let content_encoding = browser_header(&headers, "content-encoding")?;
    let etag = browser_header(&headers, "etag")?;
    let object_version = match request.object_version_header.as_deref() {
        Some(header) => browser_header(&headers, header)?,
        None => None,
    };
    let retry_after = browser_header(&headers, "retry-after")?
        .and_then(|value| value.parse::<u64>().ok())
        .map(Duration::from_secs);
    if is_retryable_status(status) {
        return Ok(HttpFetchResponse {
            status,
            redirected,
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
    // Do not trust Content-Length as an allocation bound: an origin can
    // understate it. The high-level transport applies its optional header
    // policy; the client always consumes a bounded ReadableStream.
    let bytes =
        browser_bounded_response_body(response, request.max_response_bytes, content_length).await?;
    Ok(HttpFetchResponse {
        status,
        redirected,
        bytes,
        content_length,
        content_range,
        content_encoding,
        etag,
        object_version,
        retry_after,
    })
}

async fn browser_bounded_response_body(
    response: web_sys::Response,
    maximum: u64,
    content_length_hint: Option<u64>,
) -> Result<Vec<u8>, HttpClientFailure> {
    use wasm_bindgen::JsCast as _;

    let stream = response.body().ok_or_else(|| {
        HttpClientFailure::new(
            HttpClientFailureKind::Protocol,
            "browser response body stream is unavailable",
            false,
        )
    })?;
    let reader = stream
        .get_reader()
        .dyn_into::<web_sys::ReadableStreamDefaultReader>()
        .map_err(|value| js_client_failure(value.into()))?;
    let result = browser_read_bounded_body(&reader, maximum, content_length_hint).await;
    if result.is_err() {
        // The reader owns the stream lock, so settle its cancellation before
        // releasing that lock and before the caller releases the realm task
        // permit. The response-level cancel is awaited after this function.
        let _ = wasm_bindgen_futures::JsFuture::from(reader.cancel()).await;
    }
    reader.release_lock();
    result
}

async fn browser_read_bounded_body(
    reader: &web_sys::ReadableStreamDefaultReader,
    maximum: u64,
    content_length_hint: Option<u64>,
) -> Result<Vec<u8>, HttpClientFailure> {
    let initial_capacity = content_length_hint
        .unwrap_or(0)
        .min(maximum)
        .min(usize::MAX as u64) as usize;
    let mut bytes = Vec::new();
    bytes.try_reserve_exact(initial_capacity).map_err(|_| {
        HttpClientFailure::new(
            HttpClientFailureKind::ResponseTooLarge,
            format!("failed to allocate bounded {initial_capacity} byte HTTP body"),
            false,
        )
    })?;
    loop {
        let result = wasm_bindgen_futures::JsFuture::from(reader.read())
            .await
            .map_err(js_client_failure)?;
        let done = js_sys::Reflect::get(&result, &wasm_bindgen::JsValue::from_str("done"))
            .map_err(js_client_failure)?
            .as_bool()
            .unwrap_or(false);
        if done {
            return Ok(bytes);
        }
        let value = js_sys::Reflect::get(&result, &wasm_bindgen::JsValue::from_str("value"))
            .map_err(js_client_failure)?;
        let chunk = js_sys::Uint8Array::new(&value);
        let start = bytes.len();
        let end = bounded_chunk_end(start, chunk.length() as usize, maximum)?;
        if end > bytes.capacity() {
            let capacity = bounded_growth_capacity(bytes.capacity(), end, maximum)?;
            bytes
                .try_reserve_exact(capacity.saturating_sub(bytes.len()))
                .map_err(|_| {
                    HttpClientFailure::new(
                        HttpClientFailureKind::ResponseTooLarge,
                        format!("failed to allocate bounded {capacity} byte HTTP body"),
                        false,
                    )
                })?;
        }
        bytes.resize(end, 0);
        chunk.copy_to(&mut bytes[start..end]);
    }
}

fn browser_header(
    headers: &web_sys::Headers,
    name: &str,
) -> Result<Option<String>, HttpClientFailure> {
    headers.get(name).map_err(js_client_failure)
}

fn js_client_failure(value: wasm_bindgen::JsValue) -> HttpClientFailure {
    HttpClientFailure::new(
        HttpClientFailureKind::Network,
        value.as_string().unwrap_or_else(|| format!("{value:?}")),
        true,
    )
}

#[cfg(test)]
mod browser_lifecycle_tests {
    use wasm_bindgen::JsCast as _;
    use wasm_bindgen_test::wasm_bindgen_test;

    use super::*;

    #[wasm_bindgen::prelude::wasm_bindgen(inline_js = r#"
        export function install_bgs_http_lifecycle_fixture(mode) {
            const original = window.fetch;
            let resolveFetch = null;
            let cancelResolvers = [];
            let cancelCount = 0;
            let abortSeen = false;

            function streamResponse(status, oversized) {
                const stream = new ReadableStream({
                    start(controller) {
                        if (oversized) {
                            controller.enqueue(new Uint8Array([1, 2, 3, 4, 5, 6, 7, 8]));
                        }
                    },
                    cancel() {
                        cancelCount += 1;
                        return new Promise(resolve => { cancelResolvers.push(resolve); });
                    }
                });
                return new Response(stream, { status });
            }

            window.fetch = function(request) {
                if (request.signal.aborted) {
                    abortSeen = true;
                }
                request.signal.addEventListener("abort", () => { abortSeen = true; });
                if (mode === "pending-fetch") {
                    return new Promise(resolve => { resolveFetch = resolve; });
                }
                if (mode === "retry-cancel") {
                    return Promise.resolve(streamResponse(503, false));
                }
                if (mode === "reader-cancel") {
                    return Promise.resolve(streamResponse(200, true));
                }
                if (mode === "multi-chunk") {
                    const stream = new ReadableStream({
                        start(controller) {
                            controller.enqueue(new Uint8Array([1]));
                            controller.enqueue(new Uint8Array([2, 3, 4]));
                            controller.enqueue(new Uint8Array([5, 6, 7]));
                            controller.close();
                        }
                    });
                    return Promise.resolve(new Response(stream, { status: 200 }));
                }
                return Promise.reject(new Error("unknown HTTP lifecycle fixture mode: " + mode));
            };

            return {
                restore: () => { window.fetch = original; },
                releaseFetch: () => {
                    if (resolveFetch !== null) {
                        const resolve = resolveFetch;
                        resolveFetch = null;
                        resolve(new Response(new Uint8Array([7]), {
                            status: 200,
                            headers: new Headers({ "Content-Length": "1" })
                        }));
                    }
                },
                releaseCancel: () => {
                    const resolvers = cancelResolvers;
                    cancelResolvers = [];
                    for (const resolve of resolvers) {
                        resolve();
                    }
                },
                cancelCount: () => cancelCount,
                abortSeen: () => abortSeen
            };
        }
    "#)]
    extern "C" {
        fn install_bgs_http_lifecycle_fixture(mode: &str) -> wasm_bindgen::JsValue;
    }

    struct BrowserLifecycleFixture(wasm_bindgen::JsValue);

    impl BrowserLifecycleFixture {
        fn install(mode: &str) -> Self {
            Self(install_bgs_http_lifecycle_fixture(mode))
        }

        fn call(&self, name: &str) {
            let function = js_sys::Reflect::get(&self.0, &wasm_bindgen::JsValue::from_str(name))
                .unwrap()
                .dyn_into::<js_sys::Function>()
                .unwrap();
            function.call0(&self.0).unwrap();
        }

        fn bool(&self, name: &str) -> bool {
            let function = js_sys::Reflect::get(&self.0, &wasm_bindgen::JsValue::from_str(name))
                .unwrap()
                .dyn_into::<js_sys::Function>()
                .unwrap();
            function.call0(&self.0).unwrap().as_bool().unwrap()
        }

        fn number(&self, name: &str) -> u32 {
            let function = js_sys::Reflect::get(&self.0, &wasm_bindgen::JsValue::from_str(name))
                .unwrap()
                .dyn_into::<js_sys::Function>()
                .unwrap();
            function.call0(&self.0).unwrap().as_f64().unwrap() as u32
        }
    }

    impl Drop for BrowserLifecycleFixture {
        fn drop(&mut self) {
            let function =
                js_sys::Reflect::get(&self.0, &wasm_bindgen::JsValue::from_str("restore"))
                    .ok()
                    .and_then(|value| value.dyn_into::<js_sys::Function>().ok());
            if let Some(function) = function {
                let _ = function.call0(&self.0);
            }
        }
    }

    fn browser_request(timeout: Duration, maximum: u64) -> HttpFetchRequest {
        HttpFetchRequest {
            url: "https://fixture.invalid/lod-page".to_owned(),
            byte_range: None,
            expected_bytes: maximum,
            max_response_bytes: maximum,
            timeout,
            if_match: None,
            expected_version: None,
            object_version_header: None,
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

    async fn wait_for_cancels(fixture: &BrowserLifecycleFixture, expected: u32) {
        for _ in 0..2_000 {
            if fixture.number("cancelCount") >= expected {
                return;
            }
            browser_turn().await;
        }
        panic!("browser fixture cancellation did not start");
    }

    async fn wait_for_task_release() {
        for _ in 0..2_000 {
            if browser_http_unsettled_tasks_for_testing() == 0 {
                return;
            }
            browser_turn().await;
        }
        panic!("browser HTTP task permit did not release after settlement");
    }

    #[wasm_bindgen_test(async)]
    async fn browser_timeout_publishes_before_unsettled_fetch_releases_its_permit() {
        assert_eq!(browser_http_unsettled_tasks_for_testing(), 0);
        let fixture = BrowserLifecycleFixture::install("pending-fetch");
        let mut client = BrowserFetchHttpClient::with_max_requests(1).unwrap();
        let ticket = client
            .begin(browser_request(Duration::from_millis(30), 1))
            .unwrap();

        browser_delay(60).await;
        let error = match client.poll(&ticket) {
            HttpClientPoll::Failed(error) => error,
            result => panic!("browser timeout was not published: {result:?}"),
        };
        assert_eq!(error.kind, HttpClientFailureKind::Timeout);
        assert!(fixture.bool("abortSeen"));
        assert_eq!(browser_http_unsettled_tasks_for_testing(), 1);

        fixture.call("releaseFetch");
        wait_for_task_release().await;
    }

    #[wasm_bindgen_test(async)]
    async fn browser_timed_out_body_cancels_hold_capacity_until_settlement_then_recover() {
        assert_eq!(browser_http_unsettled_tasks_for_testing(), 0);
        let fixture = BrowserLifecycleFixture::install("retry-cancel");
        let mut clients = Vec::new();
        let mut tickets = Vec::new();
        for _ in 0..BROWSER_HTTP_EFFECTIVE_TASK_CAPACITY {
            let mut client = BrowserFetchHttpClient::with_max_requests(1).unwrap();
            tickets.push(
                client
                    .begin(browser_request(Duration::from_millis(50), 8))
                    .unwrap(),
            );
            clients.push(client);
        }

        wait_for_cancels(&fixture, BROWSER_HTTP_EFFECTIVE_TASK_CAPACITY).await;
        browser_delay(80).await;
        for (client, ticket) in clients.iter_mut().zip(&tickets) {
            let error = match client.poll(ticket) {
                HttpClientPoll::Failed(error) => error,
                result => panic!("body-cancel timeout was not published: {result:?}"),
            };
            assert_eq!(error.kind, HttpClientFailureKind::Timeout);
        }
        assert_eq!(
            browser_http_unsettled_tasks_for_testing(),
            BROWSER_HTTP_EFFECTIVE_TASK_CAPACITY
        );
        let mut blocked = BrowserFetchHttpClient::with_max_requests(1).unwrap();
        assert_eq!(
            blocked
                .begin(browser_request(Duration::from_secs(2), 8))
                .unwrap_err()
                .kind,
            HttpClientFailureKind::ConcurrencyLimit
        );

        fixture.call("releaseCancel");
        wait_for_task_release().await;

        let recovered = blocked
            .begin(browser_request(Duration::from_secs(2), 8))
            .unwrap();
        wait_for_cancels(&fixture, BROWSER_HTTP_EFFECTIVE_TASK_CAPACITY + 1).await;
        fixture.call("releaseCancel");

        for _ in 0..2_000 {
            match blocked.poll(&recovered) {
                HttpClientPoll::Pending => browser_turn().await,
                HttpClientPoll::Ready(response) => {
                    assert_eq!(response.status, 503);
                    assert_eq!(browser_http_unsettled_tasks_for_testing(), 0);
                    return;
                }
                HttpClientPoll::Failed(error) => {
                    panic!("retry response cancellation failed: {error:?}")
                }
            }
        }
        panic!("retry response cancellation promise did not settle");
    }

    #[wasm_bindgen_test(async)]
    async fn browser_reader_cancel_releases_lock_and_permit_only_after_settlement() {
        assert_eq!(browser_http_unsettled_tasks_for_testing(), 0);
        let fixture = BrowserLifecycleFixture::install("reader-cancel");
        let mut client = BrowserFetchHttpClient::with_max_requests(1).unwrap();
        let ticket = client
            .begin(browser_request(Duration::from_secs(2), 4))
            .unwrap();

        wait_for_cancels(&fixture, 1).await;
        assert!(matches!(client.poll(&ticket), HttpClientPoll::Pending));
        assert_eq!(browser_http_unsettled_tasks_for_testing(), 1);
        fixture.call("releaseCancel");

        for _ in 0..2_000 {
            match client.poll(&ticket) {
                HttpClientPoll::Pending => browser_turn().await,
                HttpClientPoll::Ready(_) => panic!("oversized browser stream was accepted"),
                HttpClientPoll::Failed(error) => {
                    assert_eq!(error.kind, HttpClientFailureKind::ResponseTooLarge);
                    assert_eq!(browser_http_unsettled_tasks_for_testing(), 0);
                    return;
                }
            }
        }
        panic!("reader cancellation promise did not settle");
    }

    #[wasm_bindgen_test(async)]
    async fn browser_multichunk_body_preserves_exact_bytes_through_geometric_growth() {
        assert_eq!(browser_http_unsettled_tasks_for_testing(), 0);
        let _fixture = BrowserLifecycleFixture::install("multi-chunk");
        let mut client = BrowserFetchHttpClient::with_max_requests(1).unwrap();
        let ticket = client
            .begin(browser_request(Duration::from_secs(2), 7))
            .unwrap();

        for _ in 0..2_000 {
            match client.poll(&ticket) {
                HttpClientPoll::Pending => browser_turn().await,
                HttpClientPoll::Ready(response) => {
                    assert_eq!(response.bytes, vec![1, 2, 3, 4, 5, 6, 7]);
                    assert_eq!(browser_http_unsettled_tasks_for_testing(), 0);
                    return;
                }
                HttpClientPoll::Failed(error) => {
                    panic!("multi-chunk browser body failed: {error:?}")
                }
            }
        }
        panic!("multi-chunk browser body did not complete");
    }
}

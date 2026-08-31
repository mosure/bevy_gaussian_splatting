//! Bounded HTTP range transport for immutable LoD page objects and pack files.
//!
//! The high-level transport is independent of an async runtime. Native builds use
//! a small `ureq` worker client, browser builds use `fetch`, and deterministic
//! tests can inject a client without opening a socket. Range responses, object
//! versions, body lengths, timeouts, and retry scheduling are validated before
//! bytes reach the page codec.

use std::{collections::BTreeMap, fmt, time::Duration};

use bevy::platform::time::Instant;

use crate::{
    gaussian::formats::{planar_3d_chunked::LodPageDescriptor, planar_3d_lod::GaussianLodManifest},
    io::lod::{LodCodecLimits, decode_page_with_descriptor},
};

use super::transport::{
    LodPageId, LodPageTransport, ManifestPageLocation, ManifestPageLocations, PagePayload,
    PagePoll, PageRequest,
};

/// HTTP policy applied to every page request.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct HttpRangeTransportConfig {
    /// Absolute `http` or `https` URL below which relative manifest URIs live.
    pub base_url: String,
    /// End-to-end timeout for one attempt, including its response body.
    pub request_timeout: Duration,
    /// Number of attempts after the initial attempt.
    pub retry_limit: u32,
    /// Initial exponential-retry delay. Zero retries immediately.
    pub retry_base_delay: Duration,
    /// Retry-After and exponential delays are clamped to this duration.
    pub retry_max_delay: Duration,
    /// Hard encoded-byte limit independent of response headers.
    pub max_encoded_page_bytes: u64,
    /// Aggregate fetching and retry-wait ticket bound.
    pub max_concurrent_requests: u32,
    /// Reject responses without an exact Content-Length.
    pub require_content_length: bool,
    /// Require a strong ETag or configured version header on every object.
    pub require_object_validator: bool,
    /// Optional response header carrying an immutable object version, for
    /// example `x-amz-version-id`.
    pub object_version_header: Option<String>,
}

impl HttpRangeTransportConfig {
    pub fn validate(&self) -> Result<(), HttpRangeTransportError> {
        validate_base_url(&self.base_url)?;
        validate_request_timeout(self.request_timeout)?;
        if self.retry_max_delay < self.retry_base_delay {
            return Err(HttpRangeTransportError::RetryDelayOrder {
                base: self.retry_base_delay,
                maximum: self.retry_max_delay,
            });
        }
        if Instant::now().checked_add(self.retry_max_delay).is_none() {
            return Err(HttpRangeTransportError::RetryDeadlineOutOfRange(
                self.retry_max_delay,
            ));
        }
        if self.max_encoded_page_bytes == 0 {
            return Err(HttpRangeTransportError::ZeroMaxEncodedPageBytes);
        }
        if self.max_concurrent_requests == 0 {
            return Err(HttpRangeTransportError::ZeroMaxConcurrentRequests);
        }
        if let Some(header) = self.object_version_header.as_deref() {
            validate_header_name(header)?;
        }
        Ok(())
    }

    fn retry_delay(&self, retry_index: u32, retry_after: Option<Duration>) -> Duration {
        debug_assert!(retry_index > 0);
        let shift = retry_index.saturating_sub(1).min(31);
        let exponential = self.retry_base_delay.saturating_mul(1_u32 << shift);
        exponential
            .max(retry_after.unwrap_or(Duration::ZERO))
            .min(self.retry_max_delay)
    }
}

/// Validator learned from, or explicitly required for, one immutable object.
#[derive(Clone, Debug, Default, Eq, PartialEq)]
pub struct HttpObjectVersion {
    pub etag: Option<String>,
    pub version: Option<String>,
}

impl HttpObjectVersion {
    fn is_empty(&self) -> bool {
        self.etag.is_none() && self.version.is_none()
    }
}

/// Low-level fetch request. Clients must never return more than
/// `max_response_bytes`; the high-level transport independently checks again.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct HttpFetchRequest {
    pub url: String,
    pub byte_range: Option<(u64, u64)>,
    pub expected_bytes: u64,
    pub max_response_bytes: u64,
    pub timeout: Duration,
    pub if_match: Option<String>,
    pub expected_version: Option<String>,
    pub object_version_header: Option<String>,
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct HttpFetchResponse {
    pub status: u16,
    /// Redirects are rejected after fetch so immutable package validators cannot
    /// silently change origin.
    pub redirected: bool,
    pub bytes: Vec<u8>,
    pub content_length: Option<u64>,
    pub content_range: Option<String>,
    pub content_encoding: Option<String>,
    pub etag: Option<String>,
    pub object_version: Option<String>,
    pub retry_after: Option<Duration>,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum HttpClientFailureKind {
    Network,
    Timeout,
    ConcurrencyLimit,
    ResponseTooLarge,
    Protocol,
    InvalidRequest,
    Cancelled,
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct HttpClientFailure {
    pub kind: HttpClientFailureKind,
    pub message: String,
    pub retryable: bool,
}

impl HttpClientFailure {
    pub fn new(kind: HttpClientFailureKind, message: impl Into<String>, retryable: bool) -> Self {
        Self {
            kind,
            message: message.into(),
            retryable,
        }
    }
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub enum HttpClientPoll {
    Pending,
    Ready(HttpFetchResponse),
    Failed(HttpClientFailure),
}

/// Injectable client contract used by native workers, browser fetch, and tests.
pub trait HttpRangeClient {
    type Ticket: Clone + Eq;

    fn begin(&mut self, request: HttpFetchRequest) -> Result<Self::Ticket, HttpClientFailure>;
    fn poll(&mut self, ticket: &Self::Ticket) -> HttpClientPoll;
    fn cancel(&mut self, ticket: &Self::Ticket);
}

#[derive(Clone, Debug)]
struct FetchAttempt<Ticket> {
    client_ticket: Ticket,
    request: PageRequest,
    location: ManifestPageLocation,
    url: String,
    attempt: u32,
    started: Instant,
}

#[derive(Clone, Debug)]
struct RetryWait {
    request: PageRequest,
    location: ManifestPageLocation,
    url: String,
    next_attempt: u32,
    ready_at: Instant,
}

#[derive(Clone, Debug)]
enum HttpTicketState<Ticket> {
    Fetching(FetchAttempt<Ticket>),
    Retry(RetryWait),
}

/// Range-aware page transport with bounded bodies, timeout cancellation,
/// exponential retries, strong-validator tracking, and exact response checks.
pub struct HttpRangePageTransport<Client: HttpRangeClient> {
    config: HttpRangeTransportConfig,
    locations: ManifestPageLocations,
    client: Client,
    tickets: BTreeMap<u64, HttpTicketState<Client::Ticket>>,
    next_ticket: u64,
    /// Validators are keyed by absolute object URL. Multiple ranges in one pack
    /// must therefore observe the same immutable object version.
    observed_versions: BTreeMap<String, HttpObjectVersion>,
    expected_versions: BTreeMap<String, HttpObjectVersion>,
    validation_descriptors: BTreeMap<LodPageId, LodPageDescriptor>,
}

impl<Client: HttpRangeClient> Drop for HttpRangePageTransport<Client> {
    fn drop(&mut self) {
        let tickets = std::mem::take(&mut self.tickets);
        for state in tickets.into_values() {
            if let HttpTicketState::Fetching(attempt) = state {
                self.client.cancel(&attempt.client_ticket);
            }
        }
    }
}

impl<Client: HttpRangeClient> HttpRangePageTransport<Client> {
    pub fn new(
        config: HttpRangeTransportConfig,
        locations: ManifestPageLocations,
        client: Client,
    ) -> Result<Self, HttpRangeTransportError> {
        config.validate()?;
        for page_id in locations.page_ids() {
            let location = locations
                .get(page_id)
                .expect("page ID originated from locations");
            validate_http_location(page_id, location, config.max_encoded_page_bytes)?;
            resolve_page_url(&config.base_url, &location.uri)?;
        }
        Ok(Self {
            config,
            locations,
            client,
            tickets: BTreeMap::new(),
            next_ticket: 1,
            observed_versions: BTreeMap::new(),
            expected_versions: BTreeMap::new(),
            validation_descriptors: BTreeMap::new(),
        })
    }

    /// Enables optional manifest-level decoded page validation inside the HTTP
    /// retry loop.
    ///
    /// This mode is intended for standalone transport users that do not have a
    /// downstream validation stage. Package runtimes deliberately leave it
    /// disabled so their bounded page preprocessor is the single owner of
    /// checksum, codec, manifest, and support-bound validation.
    pub fn with_manifest_validation(
        mut self,
        manifest: &GaussianLodManifest,
    ) -> Result<Self, HttpRangeTransportError> {
        manifest
            .validate()
            .map_err(|error| HttpRangeTransportError::InvalidManifest(error.to_string()))?;
        let mut descriptors = BTreeMap::new();
        for descriptor in &manifest.pages {
            let location = self
                .locations
                .get(descriptor.id)
                .ok_or(HttpRangeTransportError::MissingPage(descriptor.id))?;
            let storage = descriptor
                .storage
                .as_ref()
                .ok_or(HttpRangeTransportError::MissingPage(descriptor.id))?;
            if location.uri != storage.uri
                || location.byte_range != storage.byte_range
                || location.encoded_len != storage.encoded_len
            {
                return Err(HttpRangeTransportError::ManifestLocationMismatch(
                    descriptor.id,
                ));
            }
            let mut descriptor = descriptor.clone();
            descriptor.storage = None;
            descriptors.insert(descriptor.id, descriptor);
        }
        self.validation_descriptors = descriptors;
        Ok(self)
    }

    pub fn config(&self) -> &HttpRangeTransportConfig {
        &self.config
    }

    pub fn client(&self) -> &Client {
        &self.client
    }

    pub fn client_mut(&mut self) -> &mut Client {
        &mut self.client
    }

    pub fn object_version(&self, absolute_url: &str) -> Option<&HttpObjectVersion> {
        self.observed_versions.get(absolute_url)
    }

    /// Pins the expected validator for a relative manifest URI. This is useful
    /// when a signed manifest or package index supplies an ETag/version before
    /// any page is fetched.
    pub fn expect_object_version(
        &mut self,
        relative_uri: &str,
        version: HttpObjectVersion,
    ) -> Result<(), HttpRangeTransportError> {
        validate_object_version(&version)?;
        let url = resolve_page_url(&self.config.base_url, relative_uri)?;
        self.expected_versions.insert(url, version);
        Ok(())
    }

    fn version_for_request(&self, url: &str) -> Option<&HttpObjectVersion> {
        self.expected_versions
            .get(url)
            .or_else(|| self.observed_versions.get(url))
    }

    fn make_fetch_request(&self, url: &str, location: &ManifestPageLocation) -> HttpFetchRequest {
        let version = self.version_for_request(url);
        HttpFetchRequest {
            url: url.to_owned(),
            byte_range: location.byte_range,
            expected_bytes: location.encoded_len,
            max_response_bytes: self.config.max_encoded_page_bytes.min(location.encoded_len),
            timeout: self.config.request_timeout,
            if_match: version.and_then(|value| value.etag.clone()),
            expected_version: version.and_then(|value| value.version.clone()),
            object_version_header: self.config.object_version_header.clone(),
        }
    }

    fn start_attempt(
        &mut self,
        request: PageRequest,
        location: ManifestPageLocation,
        url: String,
        attempt: u32,
    ) -> Result<HttpTicketState<Client::Ticket>, HttpRangeTransportError> {
        let fetch = self.make_fetch_request(&url, &location);
        match self.client.begin(fetch) {
            Ok(client_ticket) => Ok(HttpTicketState::Fetching(FetchAttempt {
                client_ticket,
                request,
                location,
                url,
                attempt,
                started: Instant::now(),
            })),
            Err(failure) => self.retry_or_fail(request, location, url, attempt, failure, None),
        }
    }

    fn retry_or_fail(
        &self,
        request: PageRequest,
        location: ManifestPageLocation,
        url: String,
        attempt: u32,
        failure: HttpClientFailure,
        retry_after: Option<Duration>,
    ) -> Result<HttpTicketState<Client::Ticket>, HttpRangeTransportError> {
        if !failure.retryable || attempt >= self.config.retry_limit {
            return Err(HttpRangeTransportError::Client {
                page: request.page_id,
                attempts: attempt.saturating_add(1),
                failure,
            });
        }
        let next_attempt = attempt + 1;
        let delay = self.config.retry_delay(next_attempt, retry_after);
        let ready_at = Instant::now()
            .checked_add(delay)
            .ok_or(HttpRangeTransportError::RetryDeadlineOutOfRange(delay))?;
        Ok(HttpTicketState::Retry(RetryWait {
            request,
            location,
            url,
            next_attempt,
            ready_at,
        }))
    }

    fn handle_response(
        &mut self,
        attempt: FetchAttempt<Client::Ticket>,
        response: HttpFetchResponse,
    ) -> Result<PagePayload, ResponseDisposition<Client::Ticket>> {
        if is_retryable_status(response.status) {
            let failure = HttpClientFailure::new(
                HttpClientFailureKind::Network,
                format!("HTTP status {}", response.status),
                true,
            );
            return Err(ResponseDisposition::Retry {
                state: self
                    .retry_or_fail(
                        attempt.request,
                        attempt.location,
                        attempt.url,
                        attempt.attempt,
                        failure,
                        response.retry_after,
                    )
                    .map(Box::new),
            });
        }
        let version = match validate_http_response(
            attempt.request.page_id,
            &attempt.location,
            &response,
            &self.config,
            self.version_for_request(&attempt.url),
        ) {
            Ok(version) => version,
            Err(error) => return Err(ResponseDisposition::Failed(error)),
        };
        if let Some(expected) = self.expected_versions.get(&attempt.url)
            && !object_version_matches(expected, &version)
        {
            return Err(ResponseDisposition::Failed(
                HttpRangeTransportError::ObjectVersionChanged {
                    url: attempt.url,
                    expected: expected.clone(),
                    actual: version,
                },
            ));
        }
        if let Some(observed) = self.observed_versions.get(&attempt.url)
            && observed != &version
        {
            return Err(ResponseDisposition::Failed(
                HttpRangeTransportError::ObjectVersionChanged {
                    url: attempt.url,
                    expected: observed.clone(),
                    actual: version,
                },
            ));
        }
        let payload = PagePayload::new(attempt.request.page_id, response.bytes);
        if let Some(descriptor) = self.validation_descriptors.get(&attempt.request.page_id) {
            let mut limits = LodCodecLimits::default();
            limits.max_page_bytes = limits.max_page_bytes.max(attempt.location.encoded_len);
            limits.max_page_gaussians = descriptor.gaussian_count;
            if let Err(error) = decode_page_with_descriptor(&payload.bytes, descriptor, limits) {
                let failure = HttpClientFailure::new(
                    HttpClientFailureKind::Protocol,
                    format!(
                        "page {} failed manifest validation: {error}",
                        attempt.request.page_id.0
                    ),
                    true,
                );
                return Err(ResponseDisposition::Retry {
                    state: self
                        .retry_or_fail(
                            attempt.request,
                            attempt.location,
                            attempt.url,
                            attempt.attempt,
                            failure,
                            None,
                        )
                        .map(Box::new),
                });
            }
        }
        self.observed_versions.insert(attempt.url, version);
        Ok(payload)
    }
}

enum ResponseDisposition<Ticket> {
    Retry {
        state: Result<Box<HttpTicketState<Ticket>>, HttpRangeTransportError>,
    },
    Failed(HttpRangeTransportError),
}

impl<Client: HttpRangeClient> LodPageTransport for HttpRangePageTransport<Client> {
    type Ticket = u64;
    type Error = HttpRangeTransportError;

    fn begin(&mut self, request: PageRequest) -> Result<Self::Ticket, Self::Error> {
        if self.tickets.len() >= self.config.max_concurrent_requests as usize {
            return Err(HttpRangeTransportError::RequestCapacityExceeded {
                maximum: self.config.max_concurrent_requests,
            });
        }
        if !request.page_id.is_valid() {
            return Err(HttpRangeTransportError::InvalidPageId);
        }
        let location = self
            .locations
            .get(request.page_id)
            .cloned()
            .ok_or(HttpRangeTransportError::MissingPage(request.page_id))?;
        if let Some(expected) = request.expected_bytes
            && expected != location.encoded_len
        {
            return Err(HttpRangeTransportError::SizeMismatch {
                expected,
                actual: location.encoded_len,
            });
        }
        validate_http_location(
            request.page_id,
            &location,
            self.config.max_encoded_page_bytes,
        )?;
        let url = resolve_page_url(&self.config.base_url, &location.uri)?;
        let state = self.start_attempt(request, location, url, 0)?;
        let ticket = self.next_ticket;
        self.next_ticket = self.next_ticket.wrapping_add(1).max(1);
        self.tickets.insert(ticket, state);
        Ok(ticket)
    }

    fn poll(&mut self, ticket: &Self::Ticket) -> PagePoll<Self::Error> {
        let Some(state) = self.tickets.remove(ticket) else {
            return PagePoll::Failed(HttpRangeTransportError::InvalidTicket(*ticket));
        };
        match state {
            HttpTicketState::Retry(wait) => {
                if Instant::now() < wait.ready_at {
                    self.tickets.insert(*ticket, HttpTicketState::Retry(wait));
                    return PagePoll::Pending;
                }
                match self.start_attempt(wait.request, wait.location, wait.url, wait.next_attempt) {
                    Ok(state) => {
                        self.tickets.insert(*ticket, state);
                        PagePoll::Pending
                    }
                    Err(error) => PagePoll::Failed(error),
                }
            }
            HttpTicketState::Fetching(attempt) => {
                match self.client.poll(&attempt.client_ticket) {
                    HttpClientPoll::Pending => {
                        // Poll the client before applying the transport's
                        // fallback wall-clock timeout. Native and browser
                        // clients enforce the same deadline internally, so a
                        // response already completed on time must not be
                        // discarded merely because the render loop polls it
                        // after the deadline.
                        if attempt.started.elapsed() >= self.config.request_timeout {
                            self.client.cancel(&attempt.client_ticket);
                            let failure = HttpClientFailure::new(
                                HttpClientFailureKind::Timeout,
                                format!(
                                    "request exceeded {:?} transport timeout",
                                    self.config.request_timeout
                                ),
                                true,
                            );
                            return match self.retry_or_fail(
                                attempt.request,
                                attempt.location,
                                attempt.url,
                                attempt.attempt,
                                failure,
                                None,
                            ) {
                                Ok(state) => {
                                    self.tickets.insert(*ticket, state);
                                    PagePoll::Pending
                                }
                                Err(error) => PagePoll::Failed(error),
                            };
                        }
                        self.tickets
                            .insert(*ticket, HttpTicketState::Fetching(attempt));
                        PagePoll::Pending
                    }
                    HttpClientPoll::Ready(response) => {
                        match self.handle_response(attempt, response) {
                            Ok(payload) => PagePoll::Ready(payload),
                            Err(ResponseDisposition::Failed(error)) => PagePoll::Failed(error),
                            Err(ResponseDisposition::Retry { state }) => match state {
                                Ok(state) => {
                                    self.tickets.insert(*ticket, *state);
                                    PagePoll::Pending
                                }
                                Err(error) => PagePoll::Failed(error),
                            },
                        }
                    }
                    HttpClientPoll::Failed(failure) => match self.retry_or_fail(
                        attempt.request,
                        attempt.location,
                        attempt.url,
                        attempt.attempt,
                        failure,
                        None,
                    ) {
                        Ok(state) => {
                            self.tickets.insert(*ticket, state);
                            PagePoll::Pending
                        }
                        Err(error) => PagePoll::Failed(error),
                    },
                }
            }
        }
    }

    fn cancel(&mut self, ticket: &Self::Ticket) {
        if let Some(HttpTicketState::Fetching(attempt)) = self.tickets.remove(ticket) {
            self.client.cancel(&attempt.client_ticket);
        }
    }
}

pub(crate) fn validate_base_url(base_url: &str) -> Result<(), HttpRangeTransportError> {
    if !(base_url.starts_with("https://") || base_url.starts_with("http://")) {
        return Err(HttpRangeTransportError::UnsupportedScheme(
            base_url.split(':').next().unwrap_or_default().to_owned(),
        ));
    }
    let authority_start = base_url.find("://").expect("validated scheme") + 3;
    let authority_end = base_url[authority_start..]
        .find('/')
        .map_or(base_url.len(), |offset| authority_start + offset);
    if authority_end == authority_start
        || base_url.contains('#')
        || base_url.contains('?')
        || base_url.contains('%')
        || base_url.contains('\\')
        || base_url
            .bytes()
            .any(|byte| byte.is_ascii_control() || byte == b' ')
        || base_url[authority_start..authority_end].contains('@')
    {
        return Err(HttpRangeTransportError::InvalidBaseUrl(base_url.to_owned()));
    }
    Ok(())
}

fn validate_header_name(header: &str) -> Result<(), HttpRangeTransportError> {
    if header.is_empty()
        || !header
            .bytes()
            .all(|byte| byte.is_ascii_alphanumeric() || byte == b'-')
    {
        return Err(HttpRangeTransportError::InvalidVersionHeader(
            header.to_owned(),
        ));
    }
    Ok(())
}

fn resolve_page_url(base_url: &str, uri: &str) -> Result<String, HttpRangeTransportError> {
    if uri.is_empty()
        || uri.starts_with('/')
        || uri.contains("://")
        || uri.contains('#')
        || uri.contains('?')
        || uri.contains('%')
        || uri.contains('\\')
        || uri
            .bytes()
            .any(|byte| byte.is_ascii_control() || byte == b' ')
        || uri
            .split('/')
            .any(|component| component.is_empty() || matches!(component, "." | ".."))
    {
        return Err(HttpRangeTransportError::UnsafeUri(uri.to_owned()));
    }
    let separator = if base_url.ends_with('/') { "" } else { "/" };
    Ok(format!("{base_url}{separator}{uri}"))
}

fn validate_http_location(
    page: LodPageId,
    location: &ManifestPageLocation,
    max_encoded_page_bytes: u64,
) -> Result<(), HttpRangeTransportError> {
    if location.encoded_len == 0 {
        return Err(HttpRangeTransportError::ZeroEncodedPageBytes(page));
    }
    if location.encoded_len > max_encoded_page_bytes {
        return Err(HttpRangeTransportError::EncodedPageTooLarge {
            page,
            encoded_len: location.encoded_len,
            maximum: max_encoded_page_bytes,
        });
    }
    if let Some((start, len)) = location.byte_range {
        if len == 0 || start.checked_add(len).is_none() {
            return Err(HttpRangeTransportError::InvalidByteRange { page, start, len });
        }
        if len != location.encoded_len {
            return Err(HttpRangeTransportError::ByteRangeLengthMismatch {
                page,
                range_len: len,
                encoded_len: location.encoded_len,
            });
        }
    }
    usize::try_from(location.encoded_len)
        .map_err(|_| HttpRangeTransportError::EncodedLengthOverflow)?;
    Ok(())
}

fn validate_object_version(version: &HttpObjectVersion) -> Result<(), HttpRangeTransportError> {
    if let Some(etag) = version.etag.as_deref()
        && !is_strong_etag(etag)
    {
        return Err(HttpRangeTransportError::WeakOrInvalidEtag(etag.to_owned()));
    }
    if version
        .version
        .as_deref()
        .is_some_and(|value| value.is_empty() || value.bytes().any(|byte| byte.is_ascii_control()))
    {
        return Err(HttpRangeTransportError::InvalidObjectVersion);
    }
    Ok(())
}

fn validate_http_response(
    page: LodPageId,
    location: &ManifestPageLocation,
    response: &HttpFetchResponse,
    config: &HttpRangeTransportConfig,
    expected_version: Option<&HttpObjectVersion>,
) -> Result<HttpObjectVersion, HttpRangeTransportError> {
    let expected_status = if location.byte_range.is_some() {
        206
    } else {
        200
    };
    if response.status != expected_status {
        return Err(HttpRangeTransportError::UnexpectedStatus {
            page,
            expected: expected_status,
            actual: response.status,
        });
    }
    if response.redirected {
        return Err(HttpRangeTransportError::UnexpectedRedirect(page));
    }
    if response
        .content_encoding
        .as_deref()
        .is_some_and(|value| !value.eq_ignore_ascii_case("identity"))
    {
        return Err(HttpRangeTransportError::UnsupportedContentEncoding(
            response.content_encoding.clone().unwrap_or_default(),
        ));
    }
    if config.require_content_length && response.content_length.is_none() {
        return Err(HttpRangeTransportError::MissingContentLength(page));
    }
    if let Some(actual) = response.content_length
        && actual != location.encoded_len
    {
        return Err(HttpRangeTransportError::ContentLengthMismatch {
            page,
            expected: location.encoded_len,
            actual,
        });
    }
    let actual = response.bytes.len() as u64;
    if actual != location.encoded_len {
        return Err(HttpRangeTransportError::BodyLengthMismatch {
            page,
            expected: location.encoded_len,
            actual,
        });
    }
    if let Some((start, len)) = location.byte_range {
        let expected_end =
            byte_range_end(start, len).ok_or(HttpRangeTransportError::EncodedLengthOverflow)?;
        let (actual_start, actual_end) = response
            .content_range
            .as_deref()
            .and_then(parse_content_range)
            .ok_or_else(|| {
                HttpRangeTransportError::InvalidContentRange(
                    response.content_range.clone().unwrap_or_default(),
                )
            })?;
        if (actual_start, actual_end) != (start, expected_end) {
            return Err(HttpRangeTransportError::ContentRangeMismatch {
                page,
                expected_start: start,
                expected_end,
                actual_start,
                actual_end,
            });
        }
    }
    let version = HttpObjectVersion {
        etag: response.etag.clone(),
        version: response.object_version.clone(),
    };
    validate_object_version(&version)?;
    if config.require_object_validator && version.is_empty() {
        return Err(HttpRangeTransportError::MissingObjectValidator(page));
    }
    if let Some(expected) = expected_version
        && !object_version_matches(expected, &version)
    {
        return Err(HttpRangeTransportError::ResponseValidatorMismatch {
            page,
            expected: expected.clone(),
            actual: version,
        });
    }
    Ok(version)
}

fn parse_content_range(value: &str) -> Option<(u64, u64)> {
    let (unit, value) = value.split_once(' ')?;
    if !unit.eq_ignore_ascii_case("bytes") {
        return None;
    }
    let (range, total) = value.split_once('/')?;
    if total.contains('/') {
        return None;
    }
    let (start, end) = range.split_once('-')?;
    let start = start.parse().ok()?;
    let end = end.parse().ok()?;
    if end < start {
        return None;
    }
    if total != "*" {
        let total = total.parse::<u64>().ok()?;
        if total <= end {
            return None;
        }
    }
    Some((start, end))
}

fn byte_range_end(start: u64, len: u64) -> Option<u64> {
    len.checked_sub(1).and_then(|last| start.checked_add(last))
}

fn validate_request_timeout(timeout: Duration) -> Result<(), HttpRangeTransportError> {
    if timeout.is_zero() {
        return Err(HttpRangeTransportError::ZeroTimeout);
    }
    if Instant::now().checked_add(timeout).is_none() {
        return Err(HttpRangeTransportError::RequestDeadlineOutOfRange(timeout));
    }
    Ok(())
}

fn validate_fetch_request_timeout(timeout: Duration) -> Result<(), HttpClientFailure> {
    validate_request_timeout(timeout).map_err(|error| {
        HttpClientFailure::new(
            HttpClientFailureKind::InvalidRequest,
            format!("invalid HTTP request timeout: {error}"),
            false,
        )
    })
}

fn object_version_matches(expected: &HttpObjectVersion, actual: &HttpObjectVersion) -> bool {
    expected
        .etag
        .as_ref()
        .is_none_or(|value| actual.etag.as_ref() == Some(value))
        && expected
            .version
            .as_ref()
            .is_none_or(|value| actual.version.as_ref() == Some(value))
}

fn is_strong_etag(value: &str) -> bool {
    !value.starts_with("W/")
        && value.len() >= 2
        && value.starts_with('"')
        && value.ends_with('"')
        && !value[1..value.len() - 1]
            .bytes()
            .any(|byte| byte == b'"' || byte.is_ascii_control())
}

fn is_retryable_status(status: u16) -> bool {
    matches!(status, 408 | 425 | 429 | 500 | 502 | 503 | 504)
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub enum HttpRangeTransportError {
    InvalidManifest(String),
    ManifestLocationMismatch(LodPageId),
    InvalidBaseUrl(String),
    UnsupportedScheme(String),
    UnsafeUri(String),
    InvalidVersionHeader(String),
    ZeroTimeout,
    ZeroMaxWorkers,
    RequestDeadlineOutOfRange(Duration),
    NativeResolverUnavailable(String),
    RetryDelayOrder {
        base: Duration,
        maximum: Duration,
    },
    RetryDeadlineOutOfRange(Duration),
    ZeroMaxEncodedPageBytes,
    ZeroMaxConcurrentRequests,
    RequestCapacityExceeded {
        maximum: u32,
    },
    InvalidPageId,
    MissingPage(LodPageId),
    InvalidTicket(u64),
    SizeMismatch {
        expected: u64,
        actual: u64,
    },
    ZeroEncodedPageBytes(LodPageId),
    EncodedPageTooLarge {
        page: LodPageId,
        encoded_len: u64,
        maximum: u64,
    },
    InvalidByteRange {
        page: LodPageId,
        start: u64,
        len: u64,
    },
    ByteRangeLengthMismatch {
        page: LodPageId,
        range_len: u64,
        encoded_len: u64,
    },
    EncodedLengthOverflow,
    Client {
        page: LodPageId,
        attempts: u32,
        failure: HttpClientFailure,
    },
    UnexpectedStatus {
        page: LodPageId,
        expected: u16,
        actual: u16,
    },
    UnexpectedRedirect(LodPageId),
    MissingContentLength(LodPageId),
    ContentLengthMismatch {
        page: LodPageId,
        expected: u64,
        actual: u64,
    },
    BodyLengthMismatch {
        page: LodPageId,
        expected: u64,
        actual: u64,
    },
    InvalidContentRange(String),
    ContentRangeMismatch {
        page: LodPageId,
        expected_start: u64,
        expected_end: u64,
        actual_start: u64,
        actual_end: u64,
    },
    UnsupportedContentEncoding(String),
    MissingObjectValidator(LodPageId),
    WeakOrInvalidEtag(String),
    InvalidObjectVersion,
    ResponseValidatorMismatch {
        page: LodPageId,
        expected: HttpObjectVersion,
        actual: HttpObjectVersion,
    },
    ObjectVersionChanged {
        url: String,
        expected: HttpObjectVersion,
        actual: HttpObjectVersion,
    },
}

impl fmt::Display for HttpRangeTransportError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(formatter, "{self:?}")
    }
}

impl std::error::Error for HttpRangeTransportError {}

#[cfg(not(target_arch = "wasm32"))]
mod native;

#[cfg(not(target_arch = "wasm32"))]
pub use native::{
    NATIVE_DNS_GLOBAL_QUEUE_CAPACITY, NATIVE_DNS_GLOBAL_WORKERS, NATIVE_HTTP_GLOBAL_QUEUE_CAPACITY,
    NATIVE_HTTP_GLOBAL_WORKERS, NativeUreqHttpClient,
};

#[cfg(all(test, not(target_arch = "wasm32")))]
use native::{
    BoundedNativeResolver, RetryableNativeHttpWorkerPool,
    create_native_dns_resolver_pool_with_limits, create_native_http_worker_pool_with_limits,
    fetch_with_ureq, native_http_worker_pool_with, native_ureq_agent,
};

#[cfg(any(target_arch = "wasm32", test))]
fn browser_timer_delay_ms(remaining: Duration) -> i32 {
    let whole_millis = remaining.as_millis();
    let rounded_up = whole_millis.saturating_add(u128::from(
        !remaining.subsec_nanos().is_multiple_of(1_000_000),
    ));
    rounded_up.clamp(1, i32::MAX as u128) as i32
}

#[cfg(any(target_arch = "wasm32", test))]
fn bounded_chunk_end(
    current: usize,
    chunk: usize,
    maximum: u64,
) -> Result<usize, HttpClientFailure> {
    let end = current.checked_add(chunk).ok_or_else(|| {
        HttpClientFailure::new(
            HttpClientFailureKind::ResponseTooLarge,
            "HTTP body length overflow",
            false,
        )
    })?;
    if end as u64 > maximum {
        return Err(HttpClientFailure::new(
            HttpClientFailureKind::ResponseTooLarge,
            format!("HTTP body exceeds {maximum} byte bound"),
            false,
        ));
    }
    Ok(end)
}

#[cfg(any(target_arch = "wasm32", test))]
fn bounded_growth_capacity(
    current_capacity: usize,
    required: usize,
    maximum: u64,
) -> Result<usize, HttpClientFailure> {
    let addressable_maximum = maximum.min(usize::MAX as u64) as usize;
    if required > addressable_maximum {
        return Err(HttpClientFailure::new(
            HttpClientFailureKind::ResponseTooLarge,
            format!("HTTP body exceeds {maximum} byte bound"),
            false,
        ));
    }
    let geometric = current_capacity
        .max(1)
        .saturating_mul(2)
        .min(addressable_maximum);
    Ok(required.max(geometric))
}

#[cfg(target_arch = "wasm32")]
mod browser;

#[cfg(target_arch = "wasm32")]
pub use browser::{BROWSER_HTTP_GLOBAL_TASK_CAPACITY, BrowserFetchHttpClient};

#[cfg(all(test, target_arch = "wasm32"))]
pub(crate) use browser::browser_http_unsettled_tasks_for_testing;

#[cfg(test)]
mod tests;

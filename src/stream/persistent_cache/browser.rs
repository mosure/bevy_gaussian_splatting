use super::browser_contract::*;
use super::*;

/// Persistent Cache Storage policy for browsers. The cache name is explicit so
/// opting into persistence never creates origin storage under an implicit name.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct BrowserPersistentCacheConfig {
    pub cache_name: String,
    pub max_bytes: u64,
    pub max_entries: u32,
    /// Aggregate active-plus-queued Cache Storage operations shared by every
    /// package using this cache name.
    pub max_pending_operations: u32,
}

impl BrowserPersistentCacheConfig {
    pub fn validate(&self) -> Result<(), PersistentCacheError> {
        if self.cache_name.is_empty()
            || self.cache_name.len() > 128
            || !self
                .cache_name
                .bytes()
                .all(|byte| byte.is_ascii_alphanumeric() || matches!(byte, b'-' | b'_' | b'.'))
        {
            return Err(PersistentCacheError::InvalidBrowserCacheName(
                self.cache_name.clone(),
            ));
        }
        if self.max_bytes == 0 {
            return Err(PersistentCacheError::ZeroByteBudget);
        }
        if self.max_entries == 0 {
            return Err(PersistentCacheError::ZeroEntryBudget);
        }
        if self.max_entries > MAX_PERSISTENT_CACHE_ENTRIES {
            return Err(PersistentCacheError::EntryBudgetTooLarge {
                configured: self.max_entries,
                maximum: MAX_PERSISTENT_CACHE_ENTRIES,
            });
        }
        validate_service_queue_capacity(self.max_pending_operations)?;
        Ok(())
    }
}

#[derive(Clone, Debug)]
enum BrowserCacheOperation {
    Lookup(PersistentCachePageIdentity),
    Insert(PersistentCachePageIdentity, PagePayload),
    Invalidate(PersistentCachePageIdentity),
    #[cfg(test)]
    TestGate(js_sys::Promise),
}

#[derive(Clone, Debug)]
enum BrowserCacheOperationValue {
    Lookup(PersistentCacheLookup),
    Insert(PersistentCacheInsert),
    Invalidate(bool),
}

#[derive(Clone, Debug)]
struct BrowserCacheOperationResult {
    value: BrowserCacheOperationValue,
    entries: u32,
    bytes: u64,
}

/// Separates caller-visible completion from the lifetime of an unabortable
/// Cache Storage operation. A timeout may publish once, but the operation's
/// Web Lock and realm admission permit remain owned by its spawned future until
/// the browser promise actually settles.
#[derive(Clone)]
struct BrowserCacheOperationPublication {
    ticket: u64,
    results: std::rc::Rc<
        std::cell::RefCell<
            BTreeMap<u64, Result<BrowserCacheOperationResult, PersistentCacheError>>,
        >,
    >,
    claimed: std::rc::Rc<std::cell::Cell<bool>>,
    timed_out: std::rc::Rc<std::cell::Cell<bool>>,
    namespace_state: std::rc::Rc<BrowserPersistentCacheNamespaceState>,
}

impl BrowserCacheOperationPublication {
    fn publish_timeout(&self) -> bool {
        if self.claimed.replace(true) {
            return false;
        }
        // Publish bypass state before the result. A caller polling the result
        // can therefore never enqueue a Store behind the still-running task.
        self.timed_out.set(true);
        self.namespace_state.begin_timed_out_operation();
        self.results.borrow_mut().insert(
            self.ticket,
            Err(PersistentCacheError::BrowserCacheOperationTimedOut {
                timeout_millis: BROWSER_CACHE_OPERATION_TIMEOUT_MS as u32,
            }),
        );
        true
    }

    fn settle(&self, result: Result<BrowserCacheOperationResult, PersistentCacheError>) {
        if self.timed_out.replace(false) {
            // `settle` is called only after the Web Locks request promise has
            // settled, which means the callback released the exclusive lock.
            self.namespace_state.finish_timed_out_operation();
        }
        if !self.claimed.replace(true) {
            self.results.borrow_mut().insert(self.ticket, result);
        }
    }
}

/// Owns both sides of a browser timer registration. Clearing the browser timer
/// in Drop guarantees that a synchronous setup error cannot leave JavaScript
/// holding a callback whose Rust closure has already been destroyed.
struct BrowserTimeoutGuard {
    window: web_sys::Window,
    handle: i32,
    _callback: wasm_bindgen::closure::Closure<dyn FnMut()>,
}

impl BrowserTimeoutGuard {
    fn schedule(
        window: &web_sys::Window,
        timeout_millis: i32,
        callback: impl FnMut() + 'static,
    ) -> Result<Self, wasm_bindgen::JsValue> {
        use wasm_bindgen::JsCast as _;

        let callback = wasm_bindgen::closure::Closure::<dyn FnMut()>::new(callback);
        let handle = window.set_timeout_with_callback_and_timeout_and_arguments_0(
            callback.as_ref().unchecked_ref(),
            timeout_millis,
        )?;
        Ok(Self {
            window: window.clone(),
            handle,
            _callback: callback,
        })
    }
}

impl Drop for BrowserTimeoutGuard {
    fn drop(&mut self) {
        self.window.clear_timeout_with_handle(self.handle);
    }
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub enum BrowserPersistentCachePoll<Value> {
    Pending,
    Ready(Value),
    Failed(PersistentCacheError),
}

#[derive(Default)]
struct BrowserPersistentCacheNamespaceState {
    timed_out_operations: std::cell::Cell<u32>,
}

impl BrowserPersistentCacheNamespaceState {
    fn is_temporarily_bypassed(&self) -> bool {
        self.timed_out_operations.get() != 0
    }

    fn begin_timed_out_operation(&self) {
        self.timed_out_operations
            .set(self.timed_out_operations.get().saturating_add(1));
    }

    fn finish_timed_out_operation(&self) {
        self.timed_out_operations
            .set(self.timed_out_operations.get().saturating_sub(1));
    }
}

/// Serialized Cache Storage backend. Operations are deliberately sequenced so
/// concurrent page requests cannot race the persistent LRU index.
pub struct BrowserPersistentPageCache {
    config: BrowserPersistentCacheConfig,
    namespace_state: std::rc::Rc<BrowserPersistentCacheNamespaceState>,
    pending: std::collections::VecDeque<(u64, BrowserCacheOperation)>,
    active: Option<u64>,
    results: std::rc::Rc<
        std::cell::RefCell<
            BTreeMap<u64, Result<BrowserCacheOperationResult, PersistentCacheError>>,
        >,
    >,
    cancelled: std::collections::BTreeSet<u64>,
    next_ticket: u64,
    stats: PersistentCacheStats,
}

struct RegisteredBrowserPersistentCache {
    config: BrowserPersistentCacheConfig,
    cache: std::rc::Rc<std::cell::RefCell<BrowserPersistentPageCache>>,
}

/// Realm-wide bound on Cache Storage/Web Lock promises that may outlive their
/// owning public cache object. The permit is owned by the spawned future, so
/// dropping an owned cache cannot reset admission accounting.
pub const BROWSER_PERSISTENT_CACHE_GLOBAL_OPERATION_CAPACITY: u32 = 256;
const BROWSER_PERSISTENT_CACHE_EFFECTIVE_OPERATION_CAPACITY: u32 = if cfg!(test) {
    4
} else {
    BROWSER_PERSISTENT_CACHE_GLOBAL_OPERATION_CAPACITY
};
const BROWSER_CACHE_LOCK_TIMEOUT_MS: i32 = if cfg!(test) { 100 } else { 5_000 };
const BROWSER_CACHE_OPERATION_TIMEOUT_MS: i32 = if cfg!(test) { 1_000 } else { 30_000 };

std::thread_local! {
    static BROWSER_PERSISTENT_CACHE_UNSETTLED_OPERATIONS: std::cell::Cell<u32> =
        const { std::cell::Cell::new(0) };
}

struct BrowserPersistentCacheOperationPermit;

impl BrowserPersistentCacheOperationPermit {
    fn acquire() -> Result<Self, PersistentCacheError> {
        BROWSER_PERSISTENT_CACHE_UNSETTLED_OPERATIONS.with(|active| {
            let current = active.get();
            if current >= BROWSER_PERSISTENT_CACHE_EFFECTIVE_OPERATION_CAPACITY {
                return Err(PersistentCacheError::BrowserOperationCapacityExceeded {
                    maximum: BROWSER_PERSISTENT_CACHE_EFFECTIVE_OPERATION_CAPACITY,
                });
            }
            active.set(current + 1);
            Ok(Self)
        })
    }
}

impl Drop for BrowserPersistentCacheOperationPermit {
    fn drop(&mut self) {
        BROWSER_PERSISTENT_CACHE_UNSETTLED_OPERATIONS.with(|active| {
            active.set(active.get().saturating_sub(1));
        });
    }
}

#[cfg(test)]
fn browser_persistent_cache_unsettled_operations_for_testing() -> u32 {
    BROWSER_PERSISTENT_CACHE_UNSETTLED_OPERATIONS.with(std::cell::Cell::get)
}

std::thread_local! {
    static BROWSER_PERSISTENT_CACHE_SERVICES: std::cell::RefCell<
        BTreeMap<String, RegisteredBrowserPersistentCache>
    > = const { std::cell::RefCell::new(BTreeMap::new()) };
    static BROWSER_PERSISTENT_CACHE_NAMESPACE_STATES: std::cell::RefCell<
        BTreeMap<String, std::rc::Weak<BrowserPersistentCacheNamespaceState>>
    > = const { std::cell::RefCell::new(BTreeMap::new()) };
}

fn browser_persistent_cache_namespace_state(
    cache_name: &str,
) -> Result<std::rc::Rc<BrowserPersistentCacheNamespaceState>, PersistentCacheError> {
    BROWSER_PERSISTENT_CACHE_NAMESPACE_STATES.with(|states| {
        let mut states = states.borrow_mut();
        states.retain(|_, state| state.strong_count() != 0);
        if let Some(state) = states.get(cache_name).and_then(std::rc::Weak::upgrade) {
            return Ok(state);
        }
        if states.len() >= MAX_PERSISTENT_CACHE_SERVICES {
            return Err(PersistentCacheError::CacheServiceRegistryFull {
                maximum: MAX_PERSISTENT_CACHE_SERVICES,
            });
        }
        let state = std::rc::Rc::new(BrowserPersistentCacheNamespaceState::default());
        states.insert(cache_name.to_owned(), std::rc::Rc::downgrade(&state));
        Ok(state)
    })
}

impl BrowserPersistentPageCache {
    pub fn new(config: BrowserPersistentCacheConfig) -> Result<Self, PersistentCacheError> {
        config.validate()?;
        let window = web_sys::window().ok_or(PersistentCacheError::BrowserStorageUnavailable)?;
        window.caches().map_err(map_browser_storage_error)?;
        let namespace_state = browser_persistent_cache_namespace_state(&config.cache_name)?;
        Ok(Self {
            config,
            namespace_state,
            pending: std::collections::VecDeque::new(),
            active: None,
            results: std::rc::Rc::new(std::cell::RefCell::new(BTreeMap::new())),
            cancelled: std::collections::BTreeSet::new(),
            next_ticket: 1,
            stats: PersistentCacheStats::default(),
        })
    }

    /// Returns the tab-lifetime coordinator for `config.cache_name`. Keeping
    /// the coordinator alive across package teardown prevents an old
    /// unabortable Cache Storage promise from racing a newly created index
    /// writer for the same namespace.
    pub fn shared(
        config: BrowserPersistentCacheConfig,
    ) -> Result<std::rc::Rc<std::cell::RefCell<Self>>, PersistentCacheError> {
        config.validate()?;
        BROWSER_PERSISTENT_CACHE_SERVICES.with(|services| {
            let mut services = services.borrow_mut();
            if let Some(registered) = services.get(&config.cache_name) {
                if registered.config != config {
                    return Err(PersistentCacheError::CacheServiceConfigConflict(
                        config.cache_name.clone(),
                    ));
                }
                return Ok(registered.cache.clone());
            }
            if services.len() >= MAX_PERSISTENT_CACHE_SERVICES {
                return Err(PersistentCacheError::CacheServiceRegistryFull {
                    maximum: MAX_PERSISTENT_CACHE_SERVICES,
                });
            }
            let cache = std::rc::Rc::new(std::cell::RefCell::new(Self::new(config.clone())?));
            services.insert(
                config.cache_name.clone(),
                RegisteredBrowserPersistentCache {
                    config,
                    cache: cache.clone(),
                },
            );
            Ok(cache)
        })
    }

    pub fn config(&self) -> &BrowserPersistentCacheConfig {
        &self.config
    }

    pub fn stats(&self) -> PersistentCacheStats {
        self.stats
    }

    pub fn begin_lookup(
        &mut self,
        identity: PersistentCachePageIdentity,
    ) -> Result<u64, PersistentCacheError> {
        identity.key()?;
        self.enqueue(BrowserCacheOperation::Lookup(identity))
    }

    fn begin_validated_lookup(
        &mut self,
        validation: PersistentCachePageValidation,
    ) -> Result<u64, PersistentCacheError> {
        self.begin_lookup(validation.identity)
    }

    pub fn poll_lookup(
        &mut self,
        ticket: &u64,
    ) -> BrowserPersistentCachePoll<PersistentCacheLookup> {
        match self.poll_operation(ticket) {
            BrowserPersistentCachePoll::Pending => BrowserPersistentCachePoll::Pending,
            BrowserPersistentCachePoll::Failed(error) => BrowserPersistentCachePoll::Failed(error),
            BrowserPersistentCachePoll::Ready(BrowserCacheOperationValue::Lookup(value)) => {
                match &value {
                    PersistentCacheLookup::Hit(_) => {
                        self.stats.hits = self.stats.hits.saturating_add(1)
                    }
                    PersistentCacheLookup::Miss => {
                        self.stats.misses = self.stats.misses.saturating_add(1)
                    }
                    PersistentCacheLookup::CorruptionRecovered(_) => {
                        self.stats.misses = self.stats.misses.saturating_add(1);
                        self.stats.corruptions_recovered =
                            self.stats.corruptions_recovered.saturating_add(1);
                    }
                }
                BrowserPersistentCachePoll::Ready(value)
            }
            BrowserPersistentCachePoll::Ready(_) => BrowserPersistentCachePoll::Failed(
                PersistentCacheError::BrowserOperationKindMismatch,
            ),
        }
    }

    pub fn begin_insert(
        &mut self,
        identity: PersistentCachePageIdentity,
        payload: PagePayload,
    ) -> Result<u64, PersistentCacheError> {
        validate_payload_identity(&identity, &payload)?;
        self.enqueue(BrowserCacheOperation::Insert(identity, payload))
    }

    pub fn poll_insert(
        &mut self,
        ticket: &u64,
    ) -> BrowserPersistentCachePoll<PersistentCacheInsert> {
        match self.poll_operation(ticket) {
            BrowserPersistentCachePoll::Pending => BrowserPersistentCachePoll::Pending,
            BrowserPersistentCachePoll::Failed(error) => BrowserPersistentCachePoll::Failed(error),
            BrowserPersistentCachePoll::Ready(BrowserCacheOperationValue::Insert(value)) => {
                if matches!(value, PersistentCacheInsert::Written { .. }) {
                    self.stats.writes = self.stats.writes.saturating_add(1);
                    if let PersistentCacheInsert::Written { evicted } = &value {
                        self.stats.evictions =
                            self.stats.evictions.saturating_add(evicted.len() as u64);
                    }
                }
                BrowserPersistentCachePoll::Ready(value)
            }
            BrowserPersistentCachePoll::Ready(_) => BrowserPersistentCachePoll::Failed(
                PersistentCacheError::BrowserOperationKindMismatch,
            ),
        }
    }

    pub fn begin_invalidate(
        &mut self,
        identity: PersistentCachePageIdentity,
    ) -> Result<u64, PersistentCacheError> {
        identity.key()?;
        self.enqueue(BrowserCacheOperation::Invalidate(identity))
    }

    pub fn poll_invalidate(&mut self, ticket: &u64) -> BrowserPersistentCachePoll<bool> {
        match self.poll_operation(ticket) {
            BrowserPersistentCachePoll::Pending => BrowserPersistentCachePoll::Pending,
            BrowserPersistentCachePoll::Failed(error) => BrowserPersistentCachePoll::Failed(error),
            BrowserPersistentCachePoll::Ready(BrowserCacheOperationValue::Invalidate(removed)) => {
                if removed {
                    self.stats.evictions = self.stats.evictions.saturating_add(1);
                }
                BrowserPersistentCachePoll::Ready(removed)
            }
            BrowserPersistentCachePoll::Ready(_) => BrowserPersistentCachePoll::Failed(
                PersistentCacheError::BrowserOperationKindMismatch,
            ),
        }
    }

    pub fn cancel(&mut self, ticket: &u64) {
        if self.active == Some(*ticket) {
            // Cache Storage promises cannot be aborted. Let the atomic operation
            // complete, then discard its result while preserving storage/index
            // consistency.
            self.cancelled.insert(*ticket);
            return;
        }
        self.pending.retain(|(queued, _)| queued != ticket);
        self.results.borrow_mut().remove(ticket);
    }

    fn enqueue(&mut self, operation: BrowserCacheOperation) -> Result<u64, PersistentCacheError> {
        self.reap_cancelled_active();
        if self.namespace_state.is_temporarily_bypassed() {
            return Err(PersistentCacheError::BrowserCacheTemporarilyBypassed);
        }
        if browser_cache_queue_is_full(
            self.pending.len(),
            self.active.is_some(),
            self.config.max_pending_operations,
        ) {
            return Err(PersistentCacheError::CacheServiceQueueFull);
        }
        let ticket = self.next_ticket;
        self.next_ticket = self.next_ticket.wrapping_add(1).max(1);
        self.pending.push_back((ticket, operation));
        self.pump();
        Ok(ticket)
    }

    fn poll_operation(
        &mut self,
        ticket: &u64,
    ) -> BrowserPersistentCachePoll<BrowserCacheOperationValue> {
        self.reap_cancelled_active();
        self.pump();
        let result = self.results.borrow_mut().remove(ticket);
        let Some(result) = result else {
            if self.active == Some(*ticket)
                || self.pending.iter().any(|(queued, _)| queued == ticket)
            {
                return BrowserPersistentCachePoll::Pending;
            }
            return BrowserPersistentCachePoll::Failed(PersistentCacheError::InvalidBrowserTicket(
                *ticket,
            ));
        };
        if self.active == Some(*ticket) {
            self.active = None;
        }
        let output = match result {
            Ok(result) => {
                self.stats.entries = result.entries;
                self.stats.bytes = result.bytes;
                BrowserPersistentCachePoll::Ready(result.value)
            }
            Err(error) => BrowserPersistentCachePoll::Failed(error),
        };
        self.pump();
        output
    }

    fn reap_cancelled_active(&mut self) {
        let Some(ticket) = self.active else {
            return;
        };
        if self.cancelled.contains(&ticket) && self.results.borrow().contains_key(&ticket) {
            self.results.borrow_mut().remove(&ticket);
            self.cancelled.remove(&ticket);
            self.active = None;
        }
    }

    fn pump(&mut self) {
        if self.active.is_some() {
            return;
        }
        while let Some((ticket, operation)) = self.pending.pop_front() {
            if self.cancelled.remove(&ticket) {
                continue;
            }
            if self.namespace_state.is_temporarily_bypassed() {
                self.results.borrow_mut().insert(
                    ticket,
                    Err(PersistentCacheError::BrowserCacheTemporarilyBypassed),
                );
                continue;
            }
            let permit = match BrowserPersistentCacheOperationPermit::acquire() {
                Ok(permit) => permit,
                Err(error) => {
                    self.results.borrow_mut().insert(ticket, Err(error));
                    continue;
                }
            };
            self.active = Some(ticket);
            let config = self.config.clone();
            let results = self.results.clone();
            let publication = BrowserCacheOperationPublication {
                ticket,
                results,
                claimed: std::rc::Rc::new(std::cell::Cell::new(false)),
                timed_out: std::rc::Rc::new(std::cell::Cell::new(false)),
                namespace_state: self.namespace_state.clone(),
            };
            wasm_bindgen_futures::spawn_local(async move {
                let _permit = permit;
                let result =
                    run_browser_cache_operation(config, operation, publication.clone()).await;
                publication.settle(result);
            });
            break;
        }
    }
}
enum BrowserPersistentTransportTicket<UpstreamTicket> {
    Lookup {
        ticket: u64,
        request: PageRequest,
        validation: PersistentCachePageValidation,
    },
    Upstream {
        ticket: UpstreamTicket,
        validation: PersistentCachePageValidation,
        bypass_store: bool,
    },
    Store {
        ticket: u64,
        payload: PagePayload,
    },
}

/// Browser cache-first wrapper with the same [`LodPageTransport`] contract as
/// the native persistent wrapper. Cache Storage persists across app/runtime
/// recreation and remains usable while the upstream URL is offline.
pub struct BrowserPersistentCachePageTransport<Upstream: LodPageTransport> {
    upstream: Upstream,
    cache: BrowserPersistentPageCache,
    identities: PersistentCachePageIdentities,
    tickets: BTreeMap<u64, BrowserPersistentTransportTicket<Upstream::Ticket>>,
    next_ticket: u64,
}

impl<Upstream: LodPageTransport> Drop for BrowserPersistentCachePageTransport<Upstream> {
    fn drop(&mut self) {
        let tickets = self.tickets.keys().copied().collect::<Vec<_>>();
        for ticket in tickets {
            <Self as LodPageTransport>::cancel(self, &ticket);
        }
    }
}

impl<Upstream: LodPageTransport> BrowserPersistentCachePageTransport<Upstream> {
    pub fn new(
        upstream: Upstream,
        cache: BrowserPersistentPageCache,
        identities: PersistentCachePageIdentities,
    ) -> Self {
        Self {
            upstream,
            cache,
            identities,
            tickets: BTreeMap::new(),
            next_ticket: 1,
        }
    }

    pub fn cache(&self) -> &BrowserPersistentPageCache {
        &self.cache
    }

    pub fn upstream(&self) -> &Upstream {
        &self.upstream
    }
}

impl<Upstream: LodPageTransport> LodPageTransport
    for BrowserPersistentCachePageTransport<Upstream>
{
    type Ticket = u64;
    type Error = PersistentCacheTransportError<Upstream::Error>;

    fn begin(&mut self, request: PageRequest) -> Result<Self::Ticket, Self::Error> {
        let validation = validated_transport_page(&self.identities, request)?;
        let state = match self.cache.begin_validated_lookup(validation.clone()) {
            Ok(cache_ticket) => BrowserPersistentTransportTicket::Lookup {
                ticket: cache_ticket,
                request,
                validation,
            },
            Err(error) if is_browser_cache_bypass_error(&error) => {
                let upstream_ticket = self
                    .upstream
                    .begin(request)
                    .map_err(PersistentCacheTransportError::Upstream)?;
                BrowserPersistentTransportTicket::Upstream {
                    ticket: upstream_ticket,
                    validation,
                    bypass_store: true,
                }
            }
            Err(error) => return Err(PersistentCacheTransportError::Cache(error)),
        };
        let ticket = self.next_ticket;
        self.next_ticket = self.next_ticket.wrapping_add(1).max(1);
        self.tickets.insert(ticket, state);
        Ok(ticket)
    }

    fn poll(&mut self, ticket: &Self::Ticket) -> PagePoll<Self::Error> {
        let Some(state) = self.tickets.remove(ticket) else {
            return PagePoll::Failed(PersistentCacheTransportError::InvalidTicket(*ticket));
        };
        match state {
            BrowserPersistentTransportTicket::Lookup {
                ticket: cache_ticket,
                request,
                validation,
            } => match self.cache.poll_lookup(&cache_ticket) {
                BrowserPersistentCachePoll::Pending => {
                    self.tickets.insert(
                        *ticket,
                        BrowserPersistentTransportTicket::Lookup {
                            ticket: cache_ticket,
                            request,
                            validation,
                        },
                    );
                    PagePoll::Pending
                }
                BrowserPersistentCachePoll::Ready(PersistentCacheLookup::Hit(payload)) => {
                    PagePoll::Ready(payload)
                }
                BrowserPersistentCachePoll::Ready(
                    PersistentCacheLookup::Miss | PersistentCacheLookup::CorruptionRecovered(_),
                ) => match self.upstream.begin(request) {
                    Ok(upstream_ticket) => {
                        self.tickets.insert(
                            *ticket,
                            BrowserPersistentTransportTicket::Upstream {
                                ticket: upstream_ticket,
                                validation,
                                bypass_store: false,
                            },
                        );
                        PagePoll::Pending
                    }
                    Err(error) => PagePoll::Failed(PersistentCacheTransportError::Upstream(error)),
                },
                BrowserPersistentCachePoll::Failed(error)
                    if is_browser_cache_bypass_error(&error) =>
                {
                    match self.upstream.begin(request) {
                        Ok(upstream_ticket) => {
                            self.tickets.insert(
                                *ticket,
                                BrowserPersistentTransportTicket::Upstream {
                                    ticket: upstream_ticket,
                                    validation,
                                    bypass_store: true,
                                },
                            );
                            PagePoll::Pending
                        }
                        Err(error) => {
                            PagePoll::Failed(PersistentCacheTransportError::Upstream(error))
                        }
                    }
                }
                BrowserPersistentCachePoll::Failed(error) => {
                    PagePoll::Failed(PersistentCacheTransportError::Cache(error))
                }
            },
            BrowserPersistentTransportTicket::Upstream {
                ticket: upstream_ticket,
                validation,
                bypass_store,
            } => match self.upstream.poll(&upstream_ticket) {
                PagePoll::Pending => {
                    self.tickets.insert(
                        *ticket,
                        BrowserPersistentTransportTicket::Upstream {
                            ticket: upstream_ticket,
                            validation,
                            bypass_store,
                        },
                    );
                    PagePoll::Pending
                }
                PagePoll::Ready(payload) => {
                    if let Err(error) = validation.validate(&payload) {
                        return PagePoll::Failed(PersistentCacheTransportError::Cache(error));
                    }
                    if bypass_store {
                        return PagePoll::Ready(payload);
                    }
                    match self
                        .cache
                        .begin_insert(validation.identity, payload.clone())
                    {
                        Ok(cache_ticket) => {
                            self.tickets.insert(
                                *ticket,
                                BrowserPersistentTransportTicket::Store {
                                    ticket: cache_ticket,
                                    payload,
                                },
                            );
                            PagePoll::Pending
                        }
                        Err(error) if is_browser_cache_bypass_error(&error) => {
                            PagePoll::Ready(payload)
                        }
                        Err(error) => PagePoll::Failed(PersistentCacheTransportError::Cache(error)),
                    }
                }
                PagePoll::Failed(error) => {
                    PagePoll::Failed(PersistentCacheTransportError::Upstream(error))
                }
            },
            BrowserPersistentTransportTicket::Store {
                ticket: cache_ticket,
                payload,
            } => match self.cache.poll_insert(&cache_ticket) {
                BrowserPersistentCachePoll::Pending => {
                    self.tickets.insert(
                        *ticket,
                        BrowserPersistentTransportTicket::Store {
                            ticket: cache_ticket,
                            payload,
                        },
                    );
                    PagePoll::Pending
                }
                BrowserPersistentCachePoll::Ready(_) => PagePoll::Ready(payload),
                BrowserPersistentCachePoll::Failed(error)
                    if is_browser_cache_bypass_error(&error) =>
                {
                    PagePoll::Ready(payload)
                }
                BrowserPersistentCachePoll::Failed(error) => {
                    PagePoll::Failed(PersistentCacheTransportError::Cache(error))
                }
            },
        }
    }

    fn cancel(&mut self, ticket: &Self::Ticket) {
        match self.tickets.remove(ticket) {
            Some(BrowserPersistentTransportTicket::Lookup { ticket, .. })
            | Some(BrowserPersistentTransportTicket::Store { ticket, .. }) => {
                self.cache.cancel(&ticket)
            }
            Some(BrowserPersistentTransportTicket::Upstream { ticket, .. }) => {
                self.upstream.cancel(&ticket)
            }
            None => {}
        }
    }
}

/// Browser cache-first wrapper borrowing one manager-owned Cache Storage queue.
/// All same-name package entities therefore serialize index reads and writes
/// through a single [`BrowserPersistentPageCache`] instance.
pub struct SharedBrowserPersistentCachePageTransport<Upstream: LodPageTransport> {
    upstream: Upstream,
    cache: std::rc::Rc<std::cell::RefCell<BrowserPersistentPageCache>>,
    identities: PersistentCachePageIdentities,
    tickets: BTreeMap<u64, BrowserPersistentTransportTicket<Upstream::Ticket>>,
    invalidations: BTreeMap<LodPageId, BrowserCacheInvalidation>,
    next_ticket: u64,
}

enum BrowserCacheInvalidation {
    Queued(PersistentCachePageIdentity),
    InFlight {
        identity: PersistentCachePageIdentity,
        ticket: u64,
    },
}

impl<Upstream: LodPageTransport> Drop for SharedBrowserPersistentCachePageTransport<Upstream> {
    fn drop(&mut self) {
        let tickets = self.tickets.keys().copied().collect::<Vec<_>>();
        for ticket in tickets {
            <Self as LodPageTransport>::cancel(self, &ticket);
        }
        let invalidations = self
            .invalidations
            .values()
            .filter_map(|state| match state {
                BrowserCacheInvalidation::Queued(_) => None,
                BrowserCacheInvalidation::InFlight { ticket, .. } => Some(*ticket),
            })
            .collect::<Vec<_>>();
        for ticket in invalidations {
            let _ = self.with_cache(|cache| cache.cancel(&ticket));
        }
    }
}

impl<Upstream: LodPageTransport> SharedBrowserPersistentCachePageTransport<Upstream> {
    pub fn new(
        upstream: Upstream,
        cache: std::rc::Rc<std::cell::RefCell<BrowserPersistentPageCache>>,
        identities: PersistentCachePageIdentities,
    ) -> Self {
        Self {
            upstream,
            cache,
            identities,
            tickets: BTreeMap::new(),
            invalidations: BTreeMap::new(),
            next_ticket: 1,
        }
    }

    pub fn shared_cache(&self) -> &std::rc::Rc<std::cell::RefCell<BrowserPersistentPageCache>> {
        &self.cache
    }

    fn with_cache<Value>(
        &self,
        operation: impl FnOnce(&mut BrowserPersistentPageCache) -> Value,
    ) -> Result<Value, PersistentCacheTransportError<Upstream::Error>> {
        let mut cache = self
            .cache
            .try_borrow_mut()
            .map_err(|_| PersistentCacheTransportError::SharedCacheUnavailable)?;
        Ok(operation(&mut cache))
    }

    /// Queues removal of a page whose downstream codec/preprocess validation
    /// failed. The operation shares the namespace's bounded serialized queue.
    pub fn invalidate_page(
        &mut self,
        page: LodPageId,
    ) -> Result<(), PersistentCacheTransportError<Upstream::Error>> {
        let _ = self.maintain_cache()?;
        if self.invalidations.contains_key(&page) {
            return Ok(());
        }
        let identity = self
            .identities
            .get(page)
            .cloned()
            .ok_or(PersistentCacheTransportError::MissingIdentity(page))?;
        self.invalidations
            .insert(page, BrowserCacheInvalidation::Queued(identity));
        self.maintain_cache().map(|_| ())
    }

    /// Polls queued invalidations without blocking the browser main thread.
    /// Returns true once no invalidation remains outstanding.
    pub fn maintain_cache(
        &mut self,
    ) -> Result<bool, PersistentCacheTransportError<Upstream::Error>> {
        let pages = self.invalidations.keys().copied().collect::<Vec<_>>();
        for page in pages {
            let Some(state) = self.invalidations.remove(&page) else {
                continue;
            };
            match state {
                BrowserCacheInvalidation::Queued(identity) => {
                    self.invalidations
                        .insert(page, BrowserCacheInvalidation::Queued(identity));
                }
                BrowserCacheInvalidation::InFlight { identity, ticket } => {
                    let poll = match self.with_cache(|cache| cache.poll_invalidate(&ticket)) {
                        Ok(poll) => poll,
                        Err(error) => {
                            self.invalidations.insert(
                                page,
                                BrowserCacheInvalidation::InFlight { identity, ticket },
                            );
                            return Err(error);
                        }
                    };
                    match poll {
                        BrowserPersistentCachePoll::Pending => {
                            self.invalidations.insert(
                                page,
                                BrowserCacheInvalidation::InFlight { identity, ticket },
                            );
                        }
                        BrowserPersistentCachePoll::Ready(_) => {}
                        BrowserPersistentCachePoll::Failed(error) => {
                            self.invalidations
                                .insert(page, BrowserCacheInvalidation::Queued(identity));
                            return Err(PersistentCacheTransportError::Cache(error));
                        }
                    }
                }
            }
        }

        let queued = self
            .invalidations
            .iter()
            .filter_map(|(&page, state)| match state {
                BrowserCacheInvalidation::Queued(identity) => Some((page, identity.clone())),
                BrowserCacheInvalidation::InFlight { .. } => None,
            })
            .collect::<Vec<_>>();
        for (page, identity) in queued {
            let operation = self.with_cache(|cache| cache.begin_invalidate(identity.clone()))?;
            let ticket = match operation {
                Ok(ticket) => ticket,
                Err(PersistentCacheError::CacheServiceQueueFull)
                | Err(PersistentCacheError::BrowserCacheTemporarilyBypassed) => continue,
                Err(error) => return Err(PersistentCacheTransportError::Cache(error)),
            };
            self.invalidations.insert(
                page,
                BrowserCacheInvalidation::InFlight { identity, ticket },
            );
        }
        Ok(self.invalidations.is_empty())
    }
}

impl<Upstream: LodPageTransport> LodPageTransport
    for SharedBrowserPersistentCachePageTransport<Upstream>
{
    type Ticket = u64;
    type Error = PersistentCacheTransportError<Upstream::Error>;

    fn begin(&mut self, request: PageRequest) -> Result<Self::Ticket, Self::Error> {
        let validation = validated_transport_page(&self.identities, request)?;
        let state = if self.invalidations.contains_key(&request.page_id) {
            let upstream_ticket = self
                .upstream
                .begin(request)
                .map_err(PersistentCacheTransportError::Upstream)?;
            BrowserPersistentTransportTicket::Upstream {
                ticket: upstream_ticket,
                validation,
                bypass_store: true,
            }
        } else {
            match self.with_cache(|cache| cache.begin_validated_lookup(validation.clone()))? {
                Ok(cache_ticket) => BrowserPersistentTransportTicket::Lookup {
                    ticket: cache_ticket,
                    request,
                    validation,
                },
                Err(error) if is_browser_cache_bypass_error(&error) => {
                    let upstream_ticket = self
                        .upstream
                        .begin(request)
                        .map_err(PersistentCacheTransportError::Upstream)?;
                    BrowserPersistentTransportTicket::Upstream {
                        ticket: upstream_ticket,
                        validation,
                        bypass_store: true,
                    }
                }
                Err(error) => return Err(PersistentCacheTransportError::Cache(error)),
            }
        };
        let ticket = self.next_ticket;
        self.next_ticket = self.next_ticket.wrapping_add(1).max(1);
        self.tickets.insert(ticket, state);
        Ok(ticket)
    }

    fn poll(&mut self, ticket: &Self::Ticket) -> PagePoll<Self::Error> {
        let Some(state) = self.tickets.remove(ticket) else {
            return PagePoll::Failed(PersistentCacheTransportError::InvalidTicket(*ticket));
        };
        match state {
            BrowserPersistentTransportTicket::Lookup {
                ticket: cache_ticket,
                request,
                validation,
            } => {
                let cache_poll = match self.with_cache(|cache| cache.poll_lookup(&cache_ticket)) {
                    Ok(poll) => poll,
                    Err(error) => return PagePoll::Failed(error),
                };
                match cache_poll {
                    BrowserPersistentCachePoll::Pending => {
                        self.tickets.insert(
                            *ticket,
                            BrowserPersistentTransportTicket::Lookup {
                                ticket: cache_ticket,
                                request,
                                validation,
                            },
                        );
                        PagePoll::Pending
                    }
                    BrowserPersistentCachePoll::Ready(PersistentCacheLookup::Hit(payload)) => {
                        PagePoll::Ready(payload)
                    }
                    BrowserPersistentCachePoll::Ready(
                        PersistentCacheLookup::Miss | PersistentCacheLookup::CorruptionRecovered(_),
                    ) => match self.upstream.begin(request) {
                        Ok(upstream_ticket) => {
                            self.tickets.insert(
                                *ticket,
                                BrowserPersistentTransportTicket::Upstream {
                                    ticket: upstream_ticket,
                                    validation,
                                    bypass_store: false,
                                },
                            );
                            PagePoll::Pending
                        }
                        Err(error) => {
                            PagePoll::Failed(PersistentCacheTransportError::Upstream(error))
                        }
                    },
                    BrowserPersistentCachePoll::Failed(error)
                        if is_browser_cache_bypass_error(&error) =>
                    {
                        match self.upstream.begin(request) {
                            Ok(upstream_ticket) => {
                                self.tickets.insert(
                                    *ticket,
                                    BrowserPersistentTransportTicket::Upstream {
                                        ticket: upstream_ticket,
                                        validation,
                                        bypass_store: true,
                                    },
                                );
                                PagePoll::Pending
                            }
                            Err(error) => {
                                PagePoll::Failed(PersistentCacheTransportError::Upstream(error))
                            }
                        }
                    }
                    BrowserPersistentCachePoll::Failed(error) => {
                        PagePoll::Failed(PersistentCacheTransportError::Cache(error))
                    }
                }
            }
            BrowserPersistentTransportTicket::Upstream {
                ticket: upstream_ticket,
                validation,
                bypass_store,
            } => match self.upstream.poll(&upstream_ticket) {
                PagePoll::Pending => {
                    self.tickets.insert(
                        *ticket,
                        BrowserPersistentTransportTicket::Upstream {
                            ticket: upstream_ticket,
                            validation,
                            bypass_store,
                        },
                    );
                    PagePoll::Pending
                }
                PagePoll::Failed(error) => {
                    PagePoll::Failed(PersistentCacheTransportError::Upstream(error))
                }
                PagePoll::Ready(payload) => {
                    if let Err(error) = validation.validate(&payload) {
                        return PagePoll::Failed(PersistentCacheTransportError::Cache(error));
                    }
                    if bypass_store {
                        return PagePoll::Ready(payload);
                    }
                    let cache_ticket = match self.with_cache(|cache| {
                        cache.begin_insert(validation.identity, payload.clone())
                    }) {
                        Ok(Ok(ticket)) => ticket,
                        Ok(Err(error)) if is_browser_cache_bypass_error(&error) => {
                            return PagePoll::Ready(payload);
                        }
                        Ok(Err(error)) => {
                            return PagePoll::Failed(PersistentCacheTransportError::Cache(error));
                        }
                        Err(error) => return PagePoll::Failed(error),
                    };
                    self.tickets.insert(
                        *ticket,
                        BrowserPersistentTransportTicket::Store {
                            ticket: cache_ticket,
                            payload,
                        },
                    );
                    PagePoll::Pending
                }
            },
            BrowserPersistentTransportTicket::Store {
                ticket: cache_ticket,
                payload,
            } => {
                let cache_poll = match self.with_cache(|cache| cache.poll_insert(&cache_ticket)) {
                    Ok(poll) => poll,
                    Err(error) => return PagePoll::Failed(error),
                };
                match cache_poll {
                    BrowserPersistentCachePoll::Pending => {
                        self.tickets.insert(
                            *ticket,
                            BrowserPersistentTransportTicket::Store {
                                ticket: cache_ticket,
                                payload,
                            },
                        );
                        PagePoll::Pending
                    }
                    BrowserPersistentCachePoll::Ready(_) => PagePoll::Ready(payload),
                    BrowserPersistentCachePoll::Failed(error)
                        if is_browser_cache_bypass_error(&error) =>
                    {
                        PagePoll::Ready(payload)
                    }
                    BrowserPersistentCachePoll::Failed(error) => {
                        PagePoll::Failed(PersistentCacheTransportError::Cache(error))
                    }
                }
            }
        }
    }

    fn cancel(&mut self, ticket: &Self::Ticket) {
        match self.tickets.remove(ticket) {
            Some(BrowserPersistentTransportTicket::Lookup { ticket, .. })
            | Some(BrowserPersistentTransportTicket::Store { ticket, .. }) => {
                let _ = self.with_cache(|cache| cache.cancel(&ticket));
            }
            Some(BrowserPersistentTransportTicket::Upstream { ticket, .. }) => {
                self.upstream.cancel(&ticket)
            }
            None => {}
        }
    }
}
async fn run_browser_cache_operation(
    config: BrowserPersistentCacheConfig,
    operation: BrowserCacheOperation,
    publication: BrowserCacheOperationPublication,
) -> Result<BrowserCacheOperationResult, PersistentCacheError> {
    use wasm_bindgen::{JsCast as _, JsValue, closure::Closure};

    let window = web_sys::window().ok_or(PersistentCacheError::BrowserStorageUnavailable)?;
    let navigator = js_sys::Reflect::get(window.as_ref(), &JsValue::from_str("navigator"))
        .map_err(|error| map_browser_coordination_error("navigator lookup failed", error))?;
    let locks = js_sys::Reflect::get(&navigator, &JsValue::from_str("locks"))
        .map_err(|error| map_browser_coordination_error("Web Locks lookup failed", error))?;
    if locks.is_null() || locks.is_undefined() {
        return Err(PersistentCacheError::BrowserCoordinationUnavailable(
            "Web Locks API is unavailable".to_owned(),
        ));
    }
    let request = js_sys::Reflect::get(&locks, &JsValue::from_str("request"))
        .map_err(|error| map_browser_coordination_error("Web Locks request lookup failed", error))?
        .dyn_into::<js_sys::Function>()
        .map_err(|error| {
            map_browser_coordination_error("Web Locks request is not callable", error)
        })?;
    let abort = web_sys::AbortController::new().map_err(|error| {
        map_browser_coordination_error("Web Lock AbortController creation failed", error)
    })?;
    let abort_for_timeout = abort.clone();
    let acquisition_timeout = std::rc::Rc::new(std::cell::RefCell::new(Some(
        BrowserTimeoutGuard::schedule(&window, BROWSER_CACHE_LOCK_TIMEOUT_MS, move || {
            abort_for_timeout.abort();
        })
        .map_err(|error| {
            map_browser_coordination_error("Web Lock timeout registration failed", error)
        })?,
    )));
    let options = js_sys::Object::new();
    js_sys::Reflect::set(
        &options,
        &JsValue::from_str("signal"),
        abort.signal().as_ref(),
    )
    .map_err(|error| map_browser_coordination_error("Web Lock AbortSignal setup failed", error))?;
    let result = std::rc::Rc::new(std::cell::RefCell::new(None));
    let callback_result = result.clone();
    let callback_config = config.clone();
    let callback_operation = operation.clone();
    let callback_acquisition_timeout = acquisition_timeout.clone();
    let callback_window = window.clone();
    let callback_publication = publication.clone();
    let callback = Closure::<dyn FnMut(JsValue) -> js_sys::Promise>::new(move |_lock| {
        // The acquisition deadline must stop at grant. An AbortSignal cannot
        // cancel the callback's Cache Storage promises after the lock is held.
        callback_acquisition_timeout.borrow_mut().take();
        let result = callback_result.clone();
        let config = callback_config.clone();
        let operation = callback_operation.clone();
        let window = callback_window.clone();
        let publication = callback_publication.clone();
        wasm_bindgen_futures::future_to_promise(async move {
            let timeout_publication = publication.clone();
            let operation_timeout = match BrowserTimeoutGuard::schedule(
                &window,
                BROWSER_CACHE_OPERATION_TIMEOUT_MS,
                move || {
                    timeout_publication.publish_timeout();
                },
            ) {
                Ok(timeout) => timeout,
                Err(error) => {
                    *result.borrow_mut() = Some(Err(map_browser_coordination_error(
                        "browser cache operation timeout registration failed",
                        error,
                    )));
                    return Ok(JsValue::UNDEFINED);
                }
            };
            let operation_result = run_browser_cache_operation_unlocked(config, operation).await;
            drop(operation_timeout);
            *result.borrow_mut() = Some(operation_result);
            Ok(JsValue::UNDEFINED)
        })
    });
    let lock_name = format!("bevy-gaussian-lod-cache::{}", config.cache_name);
    let promise = request
        .call3(
            &locks,
            &JsValue::from_str(&lock_name),
            options.as_ref(),
            callback.as_ref().unchecked_ref(),
        )
        .map_err(|error| map_browser_coordination_error("Web Locks request failed", error))?
        .dyn_into::<js_sys::Promise>()
        .map_err(|error| {
            map_browser_coordination_error("Web Locks request did not return a Promise", error)
        })?;
    let waited = wasm_bindgen_futures::JsFuture::from(promise).await;
    acquisition_timeout.borrow_mut().take();
    drop(callback);
    if let Err(error) = waited {
        return Err(map_browser_coordination_error(
            "Web Lock acquisition failed",
            error,
        ));
    }
    let completed = result.borrow_mut().take();
    completed.ok_or_else(|| {
        PersistentCacheError::BrowserCoordinationUnavailable(
            "Web Locks callback completed without a cache result".to_owned(),
        )
    })?
}

async fn run_browser_cache_operation_unlocked(
    config: BrowserPersistentCacheConfig,
    operation: BrowserCacheOperation,
) -> Result<BrowserCacheOperationResult, PersistentCacheError> {
    use wasm_bindgen::JsCast as _;

    #[cfg(test)]
    if let BrowserCacheOperation::TestGate(gate) = &operation {
        wasm_bindgen_futures::JsFuture::from(gate.clone())
            .await
            .map_err(map_browser_storage_error)?;
        return Ok(BrowserCacheOperationResult {
            value: BrowserCacheOperationValue::Lookup(PersistentCacheLookup::Miss),
            entries: 0,
            bytes: 0,
        });
    }

    let window = web_sys::window().ok_or(PersistentCacheError::BrowserStorageUnavailable)?;
    let origin = window
        .location()
        .origin()
        .map_err(map_browser_storage_error)?;
    let storage = window.caches().map_err(map_browser_storage_error)?;
    let (cache, mut index) = open_browser_cache_with_index(&storage, &origin, &config).await?;
    let value = match operation {
        BrowserCacheOperation::Lookup(identity) => {
            let key = identity.key()?;
            let file_bytes = record_file_bytes(identity.encoded_len)
                .ok_or(PersistentCacheError::ByteCountOverflow)?;
            let url = browser_record_url(&origin, key);
            let matched = wasm_bindgen_futures::JsFuture::from(cache.match_with_str(&url))
                .await
                .map_err(map_browser_storage_error)?;
            if matched.is_undefined() {
                if index.entries.remove(&key).is_some() {
                    write_browser_index(&cache, &origin, &index).await?;
                }
                BrowserCacheOperationValue::Lookup(PersistentCacheLookup::Miss)
            } else {
                let response = matched
                    .dyn_into::<web_sys::Response>()
                    .map_err(map_browser_storage_error)?;
                let record = browser_response_bytes(response, file_bytes).await;
                match record.and_then(|bytes| decode_record_bytes(&bytes, key)) {
                    Ok(payload) => {
                        let epoch = index.take_epoch();
                        index.entries.insert(
                            key,
                            BrowserCacheIndexEntry {
                                file_bytes,
                                last_used: epoch,
                            },
                        );
                        write_browser_index(&cache, &origin, &index).await?;
                        BrowserCacheOperationValue::Lookup(PersistentCacheLookup::Hit(payload))
                    }
                    Err(reason) => {
                        wasm_bindgen_futures::JsFuture::from(cache.delete_with_str(&url))
                            .await
                            .map_err(map_browser_storage_error)?;
                        index.entries.remove(&key);
                        write_browser_index(&cache, &origin, &index).await?;
                        BrowserCacheOperationValue::Lookup(
                            PersistentCacheLookup::CorruptionRecovered(PersistentCacheCorruption {
                                key,
                                reason,
                            }),
                        )
                    }
                }
            }
        }
        BrowserCacheOperation::Insert(identity, payload) => {
            validate_payload_identity(&identity, &payload)?;
            let key = identity.key()?;
            let file_bytes = record_file_bytes(identity.encoded_len)
                .ok_or(PersistentCacheError::ByteCountOverflow)?;
            if file_bytes > config.max_bytes {
                return Err(PersistentCacheError::PageExceedsBudget {
                    page: identity.page_id,
                    record_bytes: file_bytes,
                    max_bytes: config.max_bytes,
                });
            }
            let old = index.entries.remove(&key);
            let mut current_bytes = index.bytes()?;
            let mut evicted = Vec::new();
            let mut candidates = index
                .entries
                .iter()
                .map(|(&key, entry)| (entry.last_used, key, entry.file_bytes))
                .collect::<Vec<_>>();
            candidates.sort_by_key(|(last_used, key, _)| (*last_used, *key));
            let mut candidates = candidates.into_iter();
            while index.entries.len() as u64 + 1 > u64::from(config.max_entries)
                || current_bytes
                    .checked_add(file_bytes)
                    .is_none_or(|total| total > config.max_bytes)
            {
                let Some((_, victim, victim_bytes)) = candidates.next() else {
                    return Err(PersistentCacheError::BudgetCannotBeSatisfied);
                };
                index.entries.remove(&victim);
                current_bytes = current_bytes.saturating_sub(victim_bytes);
                evicted.push(victim);
            }
            let header = CacheRecordHeader {
                key,
                payload_checksum: payload.checksum,
                payload_len: identity.encoded_len,
            };
            let mut record = Vec::new();
            record
                .try_reserve_exact(CACHE_HEADER_LEN + payload.bytes.len())
                .map_err(|_| PersistentCacheError::IndexAllocationFailed(file_bytes))?;
            record.extend_from_slice(&header.encode());
            record.extend_from_slice(&payload.bytes);
            let response = web_sys::Response::new_with_opt_u8_array(Some(record.as_mut_slice()))
                .map_err(map_browser_storage_error)?;

            // Commit an incomplete marker before mutating page records. A tab
            // crash, quota failure, or rejected Cache operation can then leave
            // neither an unindexed page nor an unbounded orphan namespace:
            // startup treats this marker exactly like a corrupt index and
            // recreates the named cache from empty.
            write_browser_index_with_flags(&cache, &origin, &index, BROWSER_INDEX_DIRTY_FLAG)
                .await?;
            for victim in evicted.iter().copied() {
                let url = browser_record_url(&origin, victim);
                wasm_bindgen_futures::JsFuture::from(cache.delete_with_str(&url))
                    .await
                    .map_err(map_browser_storage_error)?;
            }
            let url = browser_record_url(&origin, key);
            wasm_bindgen_futures::JsFuture::from(cache.put_with_str(&url, &response))
                .await
                .map_err(map_browser_storage_error)?;
            let epoch = index.take_epoch();
            index.entries.insert(
                key,
                BrowserCacheIndexEntry {
                    file_bytes,
                    last_used: epoch,
                },
            );
            write_browser_index(&cache, &origin, &index).await?;
            let value = if old.is_some() && evicted.is_empty() {
                PersistentCacheInsert::AlreadyPresent
            } else {
                PersistentCacheInsert::Written { evicted }
            };
            BrowserCacheOperationValue::Insert(value)
        }
        BrowserCacheOperation::Invalidate(identity) => {
            let key = identity.key()?;
            let url = browser_record_url(&origin, key);
            let deleted = wasm_bindgen_futures::JsFuture::from(cache.delete_with_str(&url))
                .await
                .map_err(map_browser_storage_error)?
                .as_bool()
                .unwrap_or(false);
            let indexed = index.entries.remove(&key).is_some();
            if indexed {
                write_browser_index(&cache, &origin, &index).await?;
            }
            BrowserCacheOperationValue::Invalidate(deleted || indexed)
        }
        #[cfg(test)]
        BrowserCacheOperation::TestGate(_) => unreachable!("test gate returned before cache I/O"),
    };
    let entries = index
        .entries
        .len()
        .try_into()
        .map_err(|_| PersistentCacheError::EntryCountOverflow)?;
    let bytes = index.bytes()?;
    Ok(BrowserCacheOperationResult {
        value,
        entries,
        bytes,
    })
}

async fn open_browser_cache_with_index(
    storage: &web_sys::CacheStorage,
    origin: &str,
    config: &BrowserPersistentCacheConfig,
) -> Result<(web_sys::Cache, BrowserCacheIndex), PersistentCacheError> {
    use wasm_bindgen::JsCast as _;

    let cache_value = wasm_bindgen_futures::JsFuture::from(storage.open(&config.cache_name))
        .await
        .map_err(map_browser_storage_error)?;
    let cache = cache_value
        .dyn_into::<web_sys::Cache>()
        .map_err(map_browser_storage_error)?;
    let url = browser_index_url(origin);
    let matched = wasm_bindgen_futures::JsFuture::from(cache.match_with_str(&url))
        .await
        .map_err(map_browser_storage_error)?;
    if !matched.is_undefined() {
        let response = matched
            .dyn_into::<web_sys::Response>()
            .map_err(map_browser_storage_error)?;
        let maximum = BROWSER_INDEX_HEADER_LEN as u64
            + u64::from(config.max_entries) * BROWSER_INDEX_ENTRY_LEN as u64
            + 8;
        if let Ok(bytes) = browser_response_bytes(response, maximum).await
            && let Ok(mut index) = decode_browser_index(&bytes, config.max_entries)
        {
            enforce_browser_index_budget(&cache, origin, config, &mut index).await?;
            write_browser_index(&cache, origin, &index).await?;
            return Ok((cache, index));
        }
    }
    // Cache Storage has no bounded/paged key enumeration. If the authoritative
    // bounded index is absent or corrupt, atomically discard the entire named
    // namespace instead of materializing attacker-controlled orphan keys.
    wasm_bindgen_futures::JsFuture::from(storage.delete(&config.cache_name))
        .await
        .map_err(map_browser_storage_error)?;
    let cache_value = wasm_bindgen_futures::JsFuture::from(storage.open(&config.cache_name))
        .await
        .map_err(map_browser_storage_error)?;
    let cache = cache_value
        .dyn_into::<web_sys::Cache>()
        .map_err(map_browser_storage_error)?;
    let index = BrowserCacheIndex {
        next_epoch: 1,
        entries: BTreeMap::new(),
    };
    write_browser_index(&cache, origin, &index).await?;
    Ok((cache, index))
}

async fn enforce_browser_index_budget(
    cache: &web_sys::Cache,
    origin: &str,
    config: &BrowserPersistentCacheConfig,
    index: &mut BrowserCacheIndex,
) -> Result<(), PersistentCacheError> {
    let mut bytes = index.bytes()?;
    let mut candidates = index
        .entries
        .iter()
        .map(|(&key, entry)| (entry.last_used, key, entry.file_bytes))
        .collect::<Vec<_>>();
    candidates.sort_by_key(|(last_used, key, _)| (*last_used, *key));
    for (_, key, file_bytes) in candidates {
        if index.entries.len() <= config.max_entries as usize && bytes <= config.max_bytes {
            break;
        }
        wasm_bindgen_futures::JsFuture::from(
            cache.delete_with_str(&browser_record_url(origin, key)),
        )
        .await
        .map_err(map_browser_storage_error)?;
        index.entries.remove(&key);
        bytes = bytes.saturating_sub(file_bytes);
    }
    Ok(())
}

async fn write_browser_index(
    cache: &web_sys::Cache,
    origin: &str,
    index: &BrowserCacheIndex,
) -> Result<(), PersistentCacheError> {
    write_browser_index_with_flags(cache, origin, index, 0).await
}

async fn write_browser_index_with_flags(
    cache: &web_sys::Cache,
    origin: &str,
    index: &BrowserCacheIndex,
    flags: u16,
) -> Result<(), PersistentCacheError> {
    let mut bytes = encode_browser_index_with_flags(index, flags)?;
    let response = web_sys::Response::new_with_opt_u8_array(Some(bytes.as_mut_slice()))
        .map_err(map_browser_storage_error)?;
    wasm_bindgen_futures::JsFuture::from(cache.put_with_str(&browser_index_url(origin), &response))
        .await
        .map_err(map_browser_storage_error)?;
    Ok(())
}

async fn browser_response_bytes(
    response: web_sys::Response,
    maximum: u64,
) -> Result<Vec<u8>, PersistentCacheCorruptionReason> {
    use wasm_bindgen::JsCast as _;

    let stream = response
        .body()
        .ok_or(PersistentCacheCorruptionReason::TruncatedHeader)?;
    let reader = stream
        .get_reader()
        .dyn_into::<web_sys::ReadableStreamDefaultReader>()
        .map_err(|_| PersistentCacheCorruptionReason::TruncatedHeader)?;
    let mut bytes = Vec::new();
    loop {
        let result = wasm_bindgen_futures::JsFuture::from(reader.read())
            .await
            .map_err(|_| PersistentCacheCorruptionReason::TruncatedHeader)?;
        let done = js_sys::Reflect::get(&result, &wasm_bindgen::JsValue::from_str("done"))
            .map_err(|_| PersistentCacheCorruptionReason::TruncatedHeader)?
            .as_bool()
            .unwrap_or(false);
        if done {
            reader.release_lock();
            return Ok(bytes);
        }
        let value = js_sys::Reflect::get(&result, &wasm_bindgen::JsValue::from_str("value"))
            .map_err(|_| PersistentCacheCorruptionReason::TruncatedHeader)?;
        let chunk = js_sys::Uint8Array::new(&value);
        let start = bytes.len();
        let end = match bounded_cache_chunk_end(start, chunk.length() as usize, maximum) {
            Ok(end) => end,
            Err(error) => {
                let _ = reader.cancel();
                return Err(error);
            }
        };
        bytes.try_reserve_exact(end - start).map_err(|_| {
            PersistentCacheCorruptionReason::FileLengthMismatch {
                expected: maximum,
                actual: end as u64,
            }
        })?;
        bytes.resize(end, 0);
        chunk.copy_to(&mut bytes[start..end]);
    }
}

fn browser_record_url(origin: &str, key: PersistentCacheKey) -> String {
    format!("{origin}/__bgs_lod_cache_v1__/page/{}", key.file_name())
}

fn browser_index_url(origin: &str) -> String {
    format!("{origin}/__bgs_lod_cache_v1__/index")
}

fn browser_js_error_message(value: &wasm_bindgen::JsValue) -> String {
    use wasm_bindgen::JsCast as _;

    value
        .dyn_ref::<js_sys::Error>()
        .map(|error| String::from(error.message()))
        .or_else(|| value.as_string())
        .unwrap_or_else(|| format!("{value:?}"))
}

fn map_browser_coordination_error(
    context: &str,
    value: wasm_bindgen::JsValue,
) -> PersistentCacheError {
    PersistentCacheError::BrowserCoordinationUnavailable(format!(
        "{context}: {}",
        browser_js_error_message(&value)
    ))
}

fn map_browser_storage_error(value: wasm_bindgen::JsValue) -> PersistentCacheError {
    use wasm_bindgen::JsCast as _;

    let name = value
        .dyn_ref::<js_sys::Error>()
        .map(|error| String::from(error.name()))
        .unwrap_or_default();
    let message = browser_js_error_message(&value);
    if name == "QuotaExceededError" {
        PersistentCacheError::BrowserQuotaExceeded(message)
    } else {
        PersistentCacheError::BrowserStorage(message)
    }
}

fn decode_record_bytes(
    bytes: &[u8],
    expected_key: PersistentCacheKey,
) -> Result<PagePayload, PersistentCacheCorruptionReason> {
    if bytes.len() < CACHE_HEADER_LEN {
        return Err(PersistentCacheCorruptionReason::TruncatedHeader);
    }
    let header_bytes: &[u8; CACHE_HEADER_LEN] = bytes[..CACHE_HEADER_LEN]
        .try_into()
        .expect("checked fixed header length");
    let header = CacheRecordHeader::decode(header_bytes)?;
    if header.key != expected_key {
        return Err(PersistentCacheCorruptionReason::HeaderKeyMismatch);
    }
    let expected_len = record_file_bytes(header.payload_len)
        .ok_or(PersistentCacheCorruptionReason::RecordLengthOverflow)?;
    if bytes.len() as u64 != expected_len {
        return Err(PersistentCacheCorruptionReason::FileLengthMismatch {
            expected: expected_len,
            actual: bytes.len() as u64,
        });
    }
    let payload = bytes[CACHE_HEADER_LEN..].to_vec();
    let actual = page_checksum64(&payload);
    if actual != header.payload_checksum {
        return Err(PersistentCacheCorruptionReason::PayloadChecksumMismatch {
            expected: header.payload_checksum,
            actual,
        });
    }
    Ok(PagePayload {
        page_id: expected_key.page_id,
        bytes: payload,
        checksum: header.payload_checksum,
    })
}

#[cfg(test)]
mod browser_runtime_tests;

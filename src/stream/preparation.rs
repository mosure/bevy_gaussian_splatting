//! Explicitly polled record budgets for immutable package preparation.

use std::{
    future::poll_fn,
    sync::{
        Arc,
        atomic::{AtomicUsize, Ordering},
    },
    task::Poll,
};

/// Futures using this budget are polled by the package preparation owner. A
/// depleted budget deliberately does not self-wake: another application frame
/// (or another bounded native worker iteration) supplies the next allowance.
#[derive(Clone)]
pub(crate) struct PreparationBudget(Arc<AtomicUsize>);

impl PreparationBudget {
    pub(crate) fn new(records: usize) -> Self {
        Self(Arc::new(AtomicUsize::new(records)))
    }
    pub(crate) fn remaining(&self) -> usize {
        self.0.load(Ordering::Relaxed)
    }
    pub(crate) fn reset(&self, records: usize) {
        self.0.store(records, Ordering::Relaxed);
    }
    pub(crate) async fn record(&self) {
        poll_fn(|_| {
            match self
                .0
                .fetch_update(Ordering::Relaxed, Ordering::Relaxed, |remaining| {
                    if remaining == usize::MAX {
                        Some(remaining)
                    } else {
                        remaining.checked_sub(1)
                    }
                }) {
                Ok(_) => Poll::Ready(()),
                Err(_) => Poll::Pending,
            }
        })
        .await;
    }
}

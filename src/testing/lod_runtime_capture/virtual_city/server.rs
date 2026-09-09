use super::*;
use std::{
    collections::BTreeMap,
    io::BufReader,
    net::{TcpListener, TcpStream},
    sync::atomic::{AtomicBool, AtomicU8, AtomicUsize, Ordering},
    thread::{self, JoinHandle},
};

#[derive(Default, Serialize)]
pub(super) struct ServerStats {
    pub requests: u64,
    pub generated_gaussians: u64,
    pub encoded_bytes: u64,
    pub peak_active_handlers: usize,
    pub failures: u64,
    // Metadata is bounded by page count times the seven lifecycle phases.
    pub page_phase_requests: BTreeMap<String, u64>,
}

pub(super) struct Server {
    pub base_url: String,
    pub stats: Arc<Mutex<ServerStats>>,
    pub phase: Arc<AtomicU8>,
    stop: Arc<AtomicBool>,
    threads: Vec<JoinHandle<()>>,
}

impl Server {
    pub fn start(
        config: VirtualCityConfig,
        manifest: Arc<crate::GaussianLodManifest>,
    ) -> CaptureResult<Self> {
        let listener = TcpListener::bind("127.0.0.1:0")?;
        listener.set_nonblocking(true)?;
        let base_url = format!("http://{}/", listener.local_addr()?);
        let stats = Arc::new(Mutex::new(ServerStats::default()));
        let phase = Arc::new(AtomicU8::new(0));
        let stop = Arc::new(AtomicBool::new(false));
        let active = Arc::new(AtomicUsize::new(0));
        let mut threads = Vec::new();
        // Exactly two workers accept directly. There is no unbounded userspace
        // connection queue, job pool or decoded/encoded page cache.
        for _ in 0..2 {
            let listener = listener.try_clone()?;
            let (manifest, config) = (manifest.clone(), config.clone());
            let (stop, stats, phase, active) =
                (stop.clone(), stats.clone(), phase.clone(), active.clone());
            threads.push(thread::spawn(move || {
                while !stop.load(Ordering::Acquire) {
                    match listener.accept() {
                        Ok((stream, _)) => {
                            let count = active.fetch_add(1, Ordering::AcqRel) + 1;
                            {
                                let mut stats = stats.lock().unwrap();
                                stats.peak_active_handlers = stats.peak_active_handlers.max(count);
                            }
                            if serve(
                                stream,
                                &config,
                                &manifest,
                                &stats,
                                phase.load(Ordering::Acquire),
                            )
                            .is_err()
                            {
                                stats.lock().unwrap().failures += 1;
                            }
                            active.fetch_sub(1, Ordering::AcqRel);
                        }
                        Err(error) if error.kind() == std::io::ErrorKind::WouldBlock => {
                            thread::sleep(Duration::from_millis(2))
                        }
                        Err(_) => break,
                    }
                }
            }));
        }
        Ok(Self {
            base_url,
            stats,
            phase,
            stop,
            threads,
        })
    }
}

impl Drop for Server {
    fn drop(&mut self) {
        self.stop.store(true, Ordering::Release);
        for thread in self.threads.drain(..) {
            let _ = thread.join();
        }
    }
}

fn serve(
    mut stream: TcpStream,
    config: &VirtualCityConfig,
    manifest: &crate::GaussianLodManifest,
    stats: &Mutex<ServerStats>,
    phase: u8,
) -> CaptureResult<()> {
    stream.set_read_timeout(Some(Duration::from_secs(5)))?;
    stream.set_write_timeout(Some(Duration::from_secs(5)))?;
    let mut reader = BufReader::new(stream.try_clone()?);
    let mut header = Vec::new();
    loop {
        let mut byte = [0_u8];
        reader.read_exact(&mut byte)?;
        header.push(byte[0]);
        if header.ends_with(b"\r\n\r\n") {
            break;
        }
        if header.len() == 8192 {
            return Err("HTTP header exceeds 8KiB".into());
        }
    }
    let text = std::str::from_utf8(&header)?;
    let mut lines = text.split("\r\n");
    let words: Vec<_> = lines
        .next()
        .ok_or("missing request")?
        .split_whitespace()
        .collect();
    if words.len() != 3 || words[0] != "GET" {
        return Err("only GET is supported".into());
    }
    let id: u64 = words[1]
        .strip_prefix("/pages/")
        .and_then(|path| path.strip_suffix(".gspage"))
        .ok_or("unknown page path")?
        .parse()?;
    let index = usize::try_from(id.checked_sub(1).ok_or("zero page id")?)?;
    let node = manifest.nodes.get(index).ok_or("page out of range")?;
    let descriptor = manifest.pages.get(index).ok_or("descriptor out of range")?;
    let generated = data::page(config, node);
    generated.validate(descriptor)?;
    let encoded = encode_page(&generated)?;
    let expected = descriptor
        .storage
        .as_ref()
        .ok_or("storage missing")?
        .encoded_len;
    if encoded.len() as u64 != expected {
        return Err("regenerated length mismatch".into());
    }
    let mut range = None;
    for line in lines.filter(|line| !line.is_empty()) {
        if let Some((name, value)) = line.split_once(':')
            && name.eq_ignore_ascii_case("range")
        {
            if range.is_some() {
                return Err("duplicate range".into());
            }
            let (first, last) = value
                .trim()
                .strip_prefix("bytes=")
                .and_then(|value| value.split_once('-'))
                .ok_or("invalid range")?;
            let (first, last): (usize, usize) = (first.parse()?, last.parse()?);
            if first > last || last >= encoded.len() {
                return Err("out-of-bounds range".into());
            }
            range = Some((first, last));
        }
    }
    let (first, last) = range.unwrap_or((0, encoded.len() - 1));
    let status = if range.is_some() {
        "206 Partial Content"
    } else {
        "200 OK"
    };
    let content_range = if range.is_some() {
        format!("Content-Range: bytes {first}-{last}/{}\r\n", encoded.len())
    } else {
        String::new()
    };
    write!(
        stream,
        "HTTP/1.1 {status}\r\nContent-Type: application/octet-stream\r\nContent-Length: {}\r\nAccept-Ranges: bytes\r\nCache-Control: no-store\r\nETag: \"{}\"\r\n{content_range}Connection: close\r\n\r\n",
        last - first + 1,
        hash_bytes(&encoded)
    )?;
    stream.write_all(&encoded[first..=last])?;
    let mut stats = stats.lock().unwrap();
    stats.requests += 1;
    stats.generated_gaussians += generated.gaussians.len() as u64;
    stats.encoded_bytes += (last - first + 1) as u64;
    *stats
        .page_phase_requests
        .entry(format!("{phase}:{id}"))
        .or_default() += 1;
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn loopback_generates_real_page_bytes_on_demand_with_two_handler_bound() {
        let config = VirtualCityConfig {
            source_gaussians: 33,
            records_per_page: 16,
            grid_width: 4,
            ..Default::default()
        };
        let manifest = Arc::new(data::build(&config).unwrap());
        let server = Server::start(config.clone(), manifest.clone()).unwrap();
        assert_eq!(server.stats.lock().unwrap().generated_gaussians, 0);
        let address = server
            .base_url
            .trim_start_matches("http://")
            .trim_end_matches('/');
        let node = manifest.nodes.iter().find(|node| node.is_leaf()).unwrap();
        let mut stream = TcpStream::connect(address).unwrap();
        write!(
            stream,
            "GET /pages/{}.gspage HTTP/1.1\r\nHost: localhost\r\nRange: bytes=0-43\r\n\r\n",
            node.id.0
        )
        .unwrap();
        let mut response = Vec::new();
        stream.read_to_end(&mut response).unwrap();
        let split = response
            .windows(4)
            .position(|bytes| bytes == b"\r\n\r\n")
            .unwrap()
            + 4;
        assert!(response.starts_with(b"HTTP/1.1 206"));
        assert_eq!(
            &response[split..],
            &encode_page(&data::page(&config, node)).unwrap()[..44]
        );
        let stats = server.stats.lock().unwrap();
        assert_eq!(stats.requests, 1);
        assert_eq!(stats.generated_gaussians, node.source.count);
        assert!(stats.peak_active_handlers <= 2);
    }
}

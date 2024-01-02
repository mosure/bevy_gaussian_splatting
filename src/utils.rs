// #[cfg(target_arch = "wasm32")]
// pub use wasm_bindgen_rayon::init_thread_pool;

#[cfg(target_arch = "wasm32")]
use wasm_bindgen::prelude::*;
#[cfg(target_arch = "wasm32")]
use std::collections::HashMap;


pub fn setup_hooks() {
    #[cfg(debug_assertions)]
    #[cfg(target_arch = "wasm32")]
    {
        console_error_panic_hook::set_once();
    }
}


#[cfg(not(target_arch = "wasm32"))]
pub fn get_arg(n: usize) -> Option<String> {
    std::env::args().nth(n)
}

#[cfg(target_arch = "wasm32")]
pub fn get_arg(n: usize) -> Option<String> {
    let window = web_sys::window()?;
    let location = window.location();
    let search = location.search().ok()?;

    let args = search
        .trim_start_matches('?')
        .split('&')
        .map(|s| s.splitn(2, '=').collect::<Vec<_>>())
        .filter(|v| v.len() == 2)
        .map(|v| (v[0].to_string(), v[1].to_string()))
        .collect::<std::collections::HashMap<_, _>>();

    args.get(&format!("arg{}", n)).cloned()
}

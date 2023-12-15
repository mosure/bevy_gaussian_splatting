#[cfg(target_arch = "wasm32")]
pub use wasm_bindgen_rayon::init_thread_pool;


pub fn setup_hooks() {
    #[cfg(debug_assertions)]
    #[cfg(target_arch = "wasm32")]
    {
        console_error_panic_hook::set_once();
    }
}

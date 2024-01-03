// #[cfg(target_arch = "wasm32")]
// pub use wasm_bindgen_rayon::init_thread_pool;
use clap::Parser;


#[cfg(target_arch = "wasm32")]
use wasm_bindgen::prelude::*;
#[cfg(target_arch = "wasm32")]
use std::collections::HashMap;


#[cfg(not(target_arch = "wasm32"))]
#[derive(Debug, Parser)]
#[command(author, version, about, long_about = None)]
struct CliArgs {
    /// number of random gaussians to generate
    #[arg(short, long)]
    num_of_gaussians : Option<usize>,

    /// number of random particle behaviors to generate
    #[arg(short, long)]
    num_of_particle_behaviors : Option<usize>,
    
    /// .gcloud or .ply file to load
    #[arg(short, long)]
    asset_filename : Option<String>,
}

#[derive(Debug, Clone, Copy)]
pub enum MainArgs {
    NumOfGaussians=1,
    NumOfParticleBehaviors,
    AssetFilename,
}


pub fn setup_hooks() {
    #[cfg(debug_assertions)]
    #[cfg(target_arch = "wasm32")]
    {
        console_error_panic_hook::set_once();
    }
}


#[cfg(not(target_arch = "wasm32"))]
pub fn get_arg(arg: MainArgs) -> Option<String> {
    let args = CliArgs::parse();
    match arg {
        MainArgs::NumOfGaussians => args.num_of_gaussians.map(|n| n.to_string()),
        MainArgs::NumOfParticleBehaviors => args.num_of_particle_behaviors.map(|n| n.to_string()),
        MainArgs::AssetFilename => args.asset_filename,
    }
}

#[cfg(target_arch = "wasm32")]
pub fn get_arg(arg: MainArgs) -> Option<String> {
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

    let arg_value = args.get(&format!("arg{}", arg as u8)).cloned();
    match arg {
        MainArgs::NumOfGaussians | MainArgs::NumOfParticleBehaviors => arg_value.and_then(|a| a.parse::<usize>().ok().map(|_|a)),
        _ => arg_value,
    }
}

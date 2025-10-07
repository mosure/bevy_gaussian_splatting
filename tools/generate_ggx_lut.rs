use std::f32::consts::PI;
use std::path::Path;

const LUT_SIZE: u32 = 128;
const SAMPLE_COUNT: u32 = 256;

fn main() {
    println!("Generating GGX BRDF integration LUT...");
    println!("Resolution: {}x{}", LUT_SIZE, LUT_SIZE);
    println!("Samples per pixel: {}", SAMPLE_COUNT);

    let mut lut_data = vec![0u8; (LUT_SIZE * LUT_SIZE * 4) as usize];

    for y in 0..LUT_SIZE {
        for x in 0..LUT_SIZE {
            let roughness = (x as f32 + 0.5) / LUT_SIZE as f32;
            let ndotv = (y as f32 + 0.5) / LUT_SIZE as f32;

            let (scale, bias) = integrate_ggx_brdf(roughness, ndotv);

            let idx = ((y * LUT_SIZE + x) * 4) as usize;

            let scale_u16 = (scale * 65535.0).clamp(0.0, 65535.0) as u16;
            let bias_u16 = (bias * 65535.0).clamp(0.0, 65535.0) as u16;

            lut_data[idx] = (scale_u16 & 0xFF) as u8;
            lut_data[idx + 1] = ((scale_u16 >> 8) & 0xFF) as u8;
            lut_data[idx + 2] = (bias_u16 & 0xFF) as u8;
            lut_data[idx + 3] = ((bias_u16 >> 8) & 0xFF) as u8;
        }

        if y % 16 == 0 {
            println!("Progress: {}/{}", y, LUT_SIZE);
        }
    }

    let output_path = Path::new("assets/textures/ggx_energy_lut.png");
    if let Some(parent) = output_path.parent() {
        std::fs::create_dir_all(parent).expect("Failed to create output directory");
    }

    save_lut_as_png(&lut_data, output_path);

    println!("LUT generated successfully: {:?}", output_path);
}

fn integrate_ggx_brdf(perceptual_roughness: f32, ndotv: f32) -> (f32, f32) {
    let roughness = perceptual_roughness * perceptual_roughness;

    let v = Vec3::new(
        (1.0 - ndotv * ndotv).sqrt().max(0.0),
        0.0,
        ndotv,
    );

    let mut scale = 0.0;
    let mut bias = 0.0;

    for i in 0..SAMPLE_COUNT {
        let xi = hammersley(i, SAMPLE_COUNT);
        let h = importance_sample_ggx(xi, roughness);
        let l = 2.0 * h.dot(&v) * h - v;

        let ndotl = l.z.max(0.0);
        let ndoth = h.z.max(0.0);
        let vdoth = v.dot(&h).max(0.0);

        if ndotl > 0.0 {
            let g = geometry_smith(ndotv, ndotl, roughness);
            let g_vis = g * vdoth / (ndoth * ndotv).max(1e-6);
            let fc = (1.0 - vdoth).powf(5.0);

            scale += (1.0 - fc) * g_vis;
            bias += fc * g_vis;
        }
    }

    scale /= SAMPLE_COUNT as f32;
    bias /= SAMPLE_COUNT as f32;

    (scale, bias)
}

fn hammersley(i: u32, n: u32) -> Vec2 {
    Vec2::new(
        i as f32 / n as f32,
        radical_inverse_vdc(i),
    )
}

fn radical_inverse_vdc(mut bits: u32) -> f32 {
    bits = (bits << 16) | (bits >> 16);
    bits = ((bits & 0x55555555) << 1) | ((bits & 0xAAAAAAAA) >> 1);
    bits = ((bits & 0x33333333) << 2) | ((bits & 0xCCCCCCCC) >> 2);
    bits = ((bits & 0x0F0F0F0F) << 4) | ((bits & 0xF0F0F0F0) >> 4);
    bits = ((bits & 0x00FF00FF) << 8) | ((bits & 0xFF00FF00) >> 8);
    bits as f32 * 2.3283064365386963e-10
}

fn importance_sample_ggx(xi: Vec2, roughness: f32) -> Vec3 {
    let a = roughness * roughness;

    let phi = 2.0 * PI * xi.x;
    let cos_theta = ((1.0 - xi.y) / (1.0 + (a * a - 1.0) * xi.y)).sqrt();
    let sin_theta = (1.0 - cos_theta * cos_theta).sqrt();

    Vec3::new(
        phi.cos() * sin_theta,
        phi.sin() * sin_theta,
        cos_theta,
    )
}

fn geometry_schlick_ggx(ndotv: f32, roughness: f32) -> f32 {
    let a = roughness;
    let k = (a * a) / 2.0;

    let nom = ndotv;
    let denom = ndotv * (1.0 - k) + k;

    nom / denom.max(1e-6)
}

fn geometry_smith(ndotv: f32, ndotl: f32, roughness: f32) -> f32 {
    let ggx2 = geometry_schlick_ggx(ndotv, roughness);
    let ggx1 = geometry_schlick_ggx(ndotl, roughness);

    ggx1 * ggx2
}

fn save_lut_as_png(data: &[u8], path: &Path) {
    use std::fs::File;
    use std::io::BufWriter;

    let file = File::create(path).expect("Failed to create output file");
    let w = BufWriter::new(file);

    let mut encoder = png::Encoder::new(w, LUT_SIZE, LUT_SIZE);
    encoder.set_color(png::ColorType::Rgba);
    encoder.set_depth(png::BitDepth::Eight);

    let mut writer = encoder.write_header().expect("Failed to write PNG header");
    writer.write_image_data(data).expect("Failed to write PNG data");
}

#[derive(Debug, Clone, Copy)]
struct Vec2 {
    x: f32,
    y: f32,
}

impl Vec2 {
    fn new(x: f32, y: f32) -> Self {
        Self { x, y }
    }
}

#[derive(Debug, Clone, Copy)]
struct Vec3 {
    x: f32,
    y: f32,
    z: f32,
}

impl Vec3 {
    fn new(x: f32, y: f32, z: f32) -> Self {
        Self { x, y, z }
    }

    fn dot(&self, other: &Self) -> f32 {
        self.x * other.x + self.y * other.y + self.z * other.z
    }
}

impl std::ops::Sub for Vec3 {
    type Output = Self;

    fn sub(self, other: Self) -> Self {
        Self {
            x: self.x - other.x,
            y: self.y - other.y,
            z: self.z - other.z,
        }
    }
}

impl std::ops::Mul<f32> for Vec3 {
    type Output = Self;

    fn mul(self, scalar: f32) -> Self {
        Self {
            x: self.x * scalar,
            y: self.y * scalar,
            z: self.z * scalar,
        }
    }
}

impl std::ops::Mul<Vec3> for f32 {
    type Output = Vec3;

    fn mul(self, vec: Vec3) -> Vec3 {
        vec * self
    }
}

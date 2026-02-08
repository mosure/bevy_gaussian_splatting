use std::collections::{BTreeMap, HashMap};
use std::io::ErrorKind;

use base64::Engine as _;
use bevy::reflect::TypePath;
use bevy::{
    asset::{AssetLoader, LoadContext, io::Reader},
    prelude::*,
};
use gltf::{
    Accessor,
    accessor::{DataType, Dimensions, Item, Iter},
    buffer::Source,
};
use serde::Deserialize;
use serde_json::Value;

use crate::gaussian::{
    formats::planar_3d::{Gaussian3d, PlanarGaussian3d, PlanarGaussian3dHandle},
    settings::{CloudSettings, GaussianColorSpace, GaussianMode},
};
use crate::material::spherical_harmonics::{
    SH_CHANNELS, SH_COEFF_COUNT, SH_COEFF_COUNT_PER_CHANNEL,
};

const KHR_GAUSSIAN_SPLATTING_EXTENSION: &str = "KHR_gaussian_splatting";

const ATTR_POSITION: &str = "POSITION";
const ATTR_ROTATION: &str = "KHR_gaussian_splatting:ROTATION";
const ATTR_SCALE: &str = "KHR_gaussian_splatting:SCALE";
const ATTR_OPACITY: &str = "KHR_gaussian_splatting:OPACITY";
const ATTR_SH_PREFIX: &str = "KHR_gaussian_splatting:SH_DEGREE_";

#[derive(Clone, Copy, Debug, Default, Eq, Hash, PartialEq, Reflect)]
pub enum GaussianKernel {
    #[default]
    Ellipse,
}

#[derive(Clone, Copy, Debug, Default, Eq, Hash, PartialEq, Reflect)]
pub enum GaussianProjection {
    #[default]
    Perspective,
}

#[derive(Clone, Copy, Debug, Default, Eq, Hash, PartialEq, Reflect)]
pub enum GaussianSortingMethod {
    #[default]
    CameraDistance,
}

#[derive(Component, Clone, Debug, Default, Reflect)]
pub struct GaussianPrimitiveMetadata {
    pub kernel: GaussianKernel,
    pub projection: GaussianProjection,
    pub sorting_method: GaussianSortingMethod,
}

#[derive(Clone, Debug, Default, Reflect)]
pub struct CloudBundle {
    pub cloud: Handle<PlanarGaussian3d>,
    pub name: String,
    pub settings: CloudSettings,
    pub transform: Transform,
    pub metadata: GaussianPrimitiveMetadata,
}

#[derive(Asset, Clone, Debug, Default, Reflect)]
pub struct GaussianScene {
    pub bundles: Vec<CloudBundle>,
}

#[derive(Component, Clone, Debug, Default, Reflect)]
#[require(Transform, Visibility)]
pub struct GaussianSceneHandle(pub Handle<GaussianScene>);

#[derive(Component, Clone, Debug, Default, Reflect)]
pub struct GaussianSceneLoaded;

#[derive(Default)]
pub struct GaussianScenePlugin;

impl Plugin for GaussianScenePlugin {
    fn build(&self, app: &mut App) {
        app.register_type::<GaussianKernel>();
        app.register_type::<GaussianProjection>();
        app.register_type::<GaussianSortingMethod>();
        app.register_type::<GaussianPrimitiveMetadata>();
        app.register_type::<CloudBundle>();
        app.register_type::<GaussianScene>();
        app.register_type::<GaussianSceneHandle>();
        app.register_type::<GaussianSceneLoaded>();

        app.init_asset::<GaussianScene>();
        app.init_asset_loader::<GaussianSceneLoader>();

        app.add_systems(Update, (spawn_scene,));
    }
}

fn spawn_scene(
    mut commands: Commands,
    scene_handles: Query<(Entity, &GaussianSceneHandle), Without<GaussianSceneLoaded>>,
    asset_server: Res<AssetServer>,
    scenes: Res<Assets<GaussianScene>>,
) {
    for (entity, scene_handle) in scene_handles.iter() {
        if let Some(load_state) = asset_server.get_load_state(&scene_handle.0)
            && !load_state.is_loaded()
        {
            continue;
        }

        let Some(scene) = scenes.get(&scene_handle.0) else {
            continue;
        };

        let bundles = scene.bundles.clone();

        commands
            .entity(entity)
            .with_children(move |builder| {
                for bundle in bundles {
                    builder.spawn((
                        PlanarGaussian3dHandle(bundle.cloud.clone()),
                        Name::new(bundle.name.clone()),
                        bundle.settings.clone(),
                        bundle.transform,
                        bundle.metadata.clone(),
                    ));
                }
            })
            .insert(GaussianSceneLoaded);
    }
}

#[derive(Default, TypePath)]
pub struct GaussianSceneLoader;

impl AssetLoader for GaussianSceneLoader {
    type Asset = GaussianScene;
    type Settings = ();
    type Error = std::io::Error;

    async fn load(
        &self,
        reader: &mut dyn Reader,
        _: &Self::Settings,
        load_context: &mut LoadContext<'_>,
    ) -> Result<Self::Asset, Self::Error> {
        let mut bytes = Vec::new();
        reader.read_to_end(&mut bytes).await?;

        load_gltf_scene(&bytes, load_context).await
    }

    fn extensions(&self) -> &[&str] {
        &["gltf", "glb"]
    }
}

#[derive(Clone, Debug)]
struct GaussianPrimitiveSource {
    attributes: HashMap<String, usize>,
    metadata: GaussianPrimitiveMetadata,
    color_space: GaussianColorSpace,
}

#[derive(Debug, Default, Deserialize)]
#[serde(rename_all = "camelCase")]
struct RawRoot {
    #[serde(default, rename = "extensionsUsed")]
    extensions_used: Vec<String>,
    #[serde(default)]
    meshes: Vec<RawMesh>,
    #[serde(default)]
    nodes: Vec<RawNode>,
}

#[derive(Debug, Default, Deserialize)]
struct RawMesh {
    #[serde(default)]
    primitives: Vec<RawPrimitive>,
}

#[derive(Debug, Default, Deserialize)]
struct RawNode {
    #[serde(default)]
    name: Option<String>,
}

#[derive(Debug, Default, Deserialize)]
#[serde(rename_all = "camelCase")]
struct RawPrimitive {
    #[serde(default)]
    attributes: HashMap<String, usize>,
    #[serde(default)]
    mode: Option<u32>,
    #[serde(default)]
    extensions: HashMap<String, Value>,
}

#[derive(Debug, Deserialize)]
#[serde(rename_all = "camelCase")]
struct RawGaussianExtension {
    kernel: String,
    color_space: String,
    #[serde(default = "default_projection")]
    projection: String,
    #[serde(default = "default_sorting_method")]
    sorting_method: String,
}

fn default_projection() -> String {
    "perspective".to_owned()
}

fn default_sorting_method() -> String {
    "cameraDistance".to_owned()
}

async fn load_gltf_scene(
    bytes: &[u8],
    load_context: &mut LoadContext<'_>,
) -> Result<GaussianScene, std::io::Error> {
    let raw_root = parse_raw_root(bytes)?;

    let gltf = gltf::Gltf::from_slice_without_validation(bytes).map_err(|err| {
        std::io::Error::new(
            ErrorKind::InvalidData,
            format!("failed to parse glTF document: {err}"),
        )
    })?;

    let primitive_sources = collect_gaussian_primitives(&raw_root)?;
    if primitive_sources.is_empty() {
        return Err(std::io::Error::new(
            ErrorKind::InvalidData,
            "no KHR_gaussian_splatting primitives found",
        ));
    }
    ensure_gaussian_extension_used(&raw_root.extensions_used)?;

    let buffers = load_buffers(&gltf, load_context).await?;
    let scene = gltf.default_scene().or_else(|| gltf.scenes().next());
    let Some(scene) = scene else {
        return Err(std::io::Error::new(
            ErrorKind::InvalidData,
            "glTF does not contain any scenes",
        ));
    };

    let mut bundles = Vec::new();
    let mut bundle_index = 0usize;

    for node in scene.nodes() {
        collect_node_bundles(
            &node,
            Mat4::IDENTITY,
            &raw_root,
            &gltf.document,
            &buffers,
            &primitive_sources,
            load_context,
            &mut bundle_index,
            &mut bundles,
        )?;
    }

    if bundles.is_empty() {
        return Err(std::io::Error::new(
            ErrorKind::InvalidData,
            "KHR_gaussian_splatting scene contained no loadable gaussian primitives",
        ));
    }

    Ok(GaussianScene { bundles })
}

fn ensure_gaussian_extension_used(extensions_used: &[String]) -> Result<(), std::io::Error> {
    if extensions_used
        .iter()
        .any(|extension| extension == KHR_GAUSSIAN_SPLATTING_EXTENSION)
    {
        return Ok(());
    }

    Err(std::io::Error::new(
        ErrorKind::InvalidData,
        "KHR_gaussian_splatting primitives are present but the extension is missing from extensionsUsed",
    ))
}

fn parse_raw_root(bytes: &[u8]) -> Result<RawRoot, std::io::Error> {
    if bytes.starts_with(b"glTF") {
        let glb = gltf::binary::Glb::from_slice(bytes).map_err(|err| {
            std::io::Error::new(
                ErrorKind::InvalidData,
                format!("failed to parse GLB binary container: {err}"),
            )
        })?;

        serde_json::from_slice(glb.json.as_ref()).map_err(|err| {
            std::io::Error::new(
                ErrorKind::InvalidData,
                format!("failed to parse GLB JSON chunk: {err}"),
            )
        })
    } else {
        serde_json::from_slice(bytes).map_err(|err| {
            std::io::Error::new(
                ErrorKind::InvalidData,
                format!("failed to parse glTF JSON: {err}"),
            )
        })
    }
}

fn collect_gaussian_primitives(
    raw_root: &RawRoot,
) -> Result<HashMap<(usize, usize), GaussianPrimitiveSource>, std::io::Error> {
    let mut sources = HashMap::new();

    for (mesh_index, mesh) in raw_root.meshes.iter().enumerate() {
        for (primitive_index, primitive) in mesh.primitives.iter().enumerate() {
            let Some(extension_value) = primitive.extensions.get(KHR_GAUSSIAN_SPLATTING_EXTENSION)
            else {
                continue;
            };

            let mode = primitive.mode.unwrap_or(4);
            if mode != 0 {
                return Err(std::io::Error::new(
                    ErrorKind::InvalidData,
                    format!(
                        "mesh {mesh_index} primitive {primitive_index} has KHR_gaussian_splatting but mode={mode}; mode must be POINTS (0)"
                    ),
                ));
            }

            let extension: RawGaussianExtension =
                serde_json::from_value(extension_value.clone()).map_err(|err| {
                    std::io::Error::new(
                        ErrorKind::InvalidData,
                        format!(
                            "mesh {mesh_index} primitive {primitive_index} has invalid KHR_gaussian_splatting extension payload: {err}"
                        ),
                    )
                })?;

            let kernel = parse_kernel(&extension.kernel, mesh_index, primitive_index)?;
            let color_space =
                parse_color_space(&extension.color_space, mesh_index, primitive_index)?;
            let projection = parse_projection(&extension.projection, mesh_index, primitive_index)?;
            let sorting_method =
                parse_sorting_method(&extension.sorting_method, mesh_index, primitive_index)?;

            sources.insert(
                (mesh_index, primitive_index),
                GaussianPrimitiveSource {
                    attributes: primitive.attributes.clone(),
                    metadata: GaussianPrimitiveMetadata {
                        kernel,
                        projection,
                        sorting_method,
                    },
                    color_space,
                },
            );
        }
    }

    Ok(sources)
}

fn parse_kernel(
    value: &str,
    mesh_index: usize,
    primitive_index: usize,
) -> Result<GaussianKernel, std::io::Error> {
    match value {
        "ellipse" => Ok(GaussianKernel::Ellipse),
        _ => Err(std::io::Error::new(
            ErrorKind::InvalidData,
            format!(
                "mesh {mesh_index} primitive {primitive_index} uses unsupported KHR_gaussian_splatting kernel '{value}'; only 'ellipse' is currently supported"
            ),
        )),
    }
}

fn parse_color_space(
    value: &str,
    mesh_index: usize,
    primitive_index: usize,
) -> Result<GaussianColorSpace, std::io::Error> {
    match value {
        "srgb_rec709_display" => Ok(GaussianColorSpace::SrgbRec709Display),
        "lin_rec709_display" => Ok(GaussianColorSpace::LinRec709Display),
        _ => Err(std::io::Error::new(
            ErrorKind::InvalidData,
            format!(
                "mesh {mesh_index} primitive {primitive_index} uses unsupported KHR_gaussian_splatting colorSpace '{value}'"
            ),
        )),
    }
}

fn parse_projection(
    value: &str,
    mesh_index: usize,
    primitive_index: usize,
) -> Result<GaussianProjection, std::io::Error> {
    match value {
        "perspective" => Ok(GaussianProjection::Perspective),
        _ => Err(std::io::Error::new(
            ErrorKind::InvalidData,
            format!(
                "mesh {mesh_index} primitive {primitive_index} uses unsupported KHR_gaussian_splatting projection '{value}'; only 'perspective' is currently supported"
            ),
        )),
    }
}

fn parse_sorting_method(
    value: &str,
    mesh_index: usize,
    primitive_index: usize,
) -> Result<GaussianSortingMethod, std::io::Error> {
    match value {
        "cameraDistance" => Ok(GaussianSortingMethod::CameraDistance),
        _ => Err(std::io::Error::new(
            ErrorKind::InvalidData,
            format!(
                "mesh {mesh_index} primitive {primitive_index} uses unsupported KHR_gaussian_splatting sortingMethod '{value}'; only 'cameraDistance' is currently supported"
            ),
        )),
    }
}

async fn load_buffers(
    gltf: &gltf::Gltf,
    load_context: &mut LoadContext<'_>,
) -> Result<Vec<Vec<u8>>, std::io::Error> {
    let mut buffers = Vec::new();
    let mut blob = gltf.blob.clone();

    for buffer in gltf.buffers() {
        let mut data = match buffer.source() {
            Source::Bin => blob.take().ok_or_else(|| {
                std::io::Error::new(
                    ErrorKind::InvalidData,
                    "glTF buffer references BIN chunk but binary data is missing",
                )
            })?,
            Source::Uri(uri) => {
                if let Some(decoded) = decode_data_uri(uri) {
                    decoded?
                } else {
                    let path = load_context.path().resolve_embed(uri).map_err(|err| {
                        std::io::Error::new(
                            ErrorKind::InvalidData,
                            format!("failed to resolve external buffer URI '{uri}': {err}"),
                        )
                    })?;

                    load_context.read_asset_bytes(path).await.map_err(|err| {
                        std::io::Error::new(
                            ErrorKind::NotFound,
                            format!("failed to read external buffer '{uri}': {err}"),
                        )
                    })?
                }
            }
        };

        if data.len() < buffer.length() {
            return Err(std::io::Error::new(
                ErrorKind::InvalidData,
                format!(
                    "buffer {} length mismatch: expected at least {} bytes, got {} bytes",
                    buffer.index(),
                    buffer.length(),
                    data.len()
                ),
            ));
        }

        while data.len() % 4 != 0 {
            data.push(0);
        }

        buffers.push(data);
    }

    Ok(buffers)
}

fn decode_data_uri(uri: &str) -> Option<Result<Vec<u8>, std::io::Error>> {
    let rest = uri.strip_prefix("data:")?;

    let Some((metadata, payload)) = rest.split_once(',') else {
        return Some(Err(std::io::Error::new(
            ErrorKind::InvalidData,
            "malformed data URI; expected a ',' separator",
        )));
    };

    let is_base64 = metadata
        .split(';')
        .any(|part| part.eq_ignore_ascii_case("base64"));

    if is_base64 {
        return Some(
            base64::engine::general_purpose::STANDARD
                .decode(payload)
                .map_err(|err| {
                    std::io::Error::new(
                        ErrorKind::InvalidData,
                        format!("failed to decode base64 data URI: {err}"),
                    )
                }),
        );
    }

    Some(decode_percent_encoded_data_uri(payload))
}

fn decode_percent_encoded_data_uri(payload: &str) -> Result<Vec<u8>, std::io::Error> {
    let bytes = payload.as_bytes();
    let mut decoded = Vec::with_capacity(bytes.len());
    let mut index = 0usize;

    while index < bytes.len() {
        if bytes[index] == b'%' {
            if index + 2 >= bytes.len() {
                return Err(std::io::Error::new(
                    ErrorKind::InvalidData,
                    "malformed percent-encoded data URI payload",
                ));
            }

            let high = decode_hex(bytes[index + 1])?;
            let low = decode_hex(bytes[index + 2])?;
            decoded.push((high << 4) | low);
            index += 3;
            continue;
        }

        decoded.push(bytes[index]);
        index += 1;
    }

    Ok(decoded)
}

fn decode_hex(value: u8) -> Result<u8, std::io::Error> {
    match value {
        b'0'..=b'9' => Ok(value - b'0'),
        b'a'..=b'f' => Ok(value - b'a' + 10),
        b'A'..=b'F' => Ok(value - b'A' + 10),
        _ => Err(std::io::Error::new(
            ErrorKind::InvalidData,
            format!(
                "malformed percent-encoded data URI payload: invalid hex digit '{}'",
                value as char
            ),
        )),
    }
}

#[allow(clippy::too_many_arguments)]
fn collect_node_bundles(
    node: &gltf::Node<'_>,
    parent_transform: Mat4,
    raw_root: &RawRoot,
    document: &gltf::Document,
    buffers: &[Vec<u8>],
    primitive_sources: &HashMap<(usize, usize), GaussianPrimitiveSource>,
    load_context: &mut LoadContext<'_>,
    bundle_index: &mut usize,
    bundles: &mut Vec<CloudBundle>,
) -> Result<(), std::io::Error> {
    let local_transform = Mat4::from_cols_array_2d(&node.transform().matrix());
    let world_transform = parent_transform * local_transform;

    if let Some(mesh) = node.mesh() {
        for primitive in mesh.primitives() {
            let key = (mesh.index(), primitive.index());
            let Some(source) = primitive_sources.get(&key) else {
                continue;
            };

            let cloud = decode_gaussian_primitive(document, buffers, source)?;
            let cloud_handle =
                load_context.add_labeled_asset(format!("gltf_gaussian_{}", *bundle_index), cloud);

            let settings = CloudSettings {
                gaussian_mode: GaussianMode::Gaussian3d,
                color_space: source.color_space,
                ..default()
            };

            let node_name = raw_root
                .nodes
                .get(node.index())
                .and_then(|raw_node| raw_node.name.as_deref())
                .unwrap_or("gaussian_node");

            bundles.push(CloudBundle {
                cloud: cloud_handle,
                name: format!(
                    "{node_name}_mesh{}_primitive{}",
                    mesh.index(),
                    primitive.index()
                ),
                settings,
                transform: Transform::from_matrix(world_transform),
                metadata: source.metadata.clone(),
            });
            *bundle_index += 1;
        }
    }

    for child in node.children() {
        collect_node_bundles(
            &child,
            world_transform,
            raw_root,
            document,
            buffers,
            primitive_sources,
            load_context,
            bundle_index,
            bundles,
        )?;
    }

    Ok(())
}

fn decode_gaussian_primitive(
    document: &gltf::Document,
    buffers: &[Vec<u8>],
    source: &GaussianPrimitiveSource,
) -> Result<PlanarGaussian3d, std::io::Error> {
    let position_accessor = required_accessor(document, &source.attributes, ATTR_POSITION)?;
    let rotation_accessor = required_accessor(document, &source.attributes, ATTR_ROTATION)?;
    let scale_accessor = required_accessor(document, &source.attributes, ATTR_SCALE)?;
    let opacity_accessor = required_accessor(document, &source.attributes, ATTR_OPACITY)?;
    let sh_accessors = collect_sh_accessors(document, &source.attributes)?;

    let count = position_accessor.count();
    ensure_count(&rotation_accessor, count, ATTR_ROTATION)?;
    ensure_count(&scale_accessor, count, ATTR_SCALE)?;
    ensure_count(&opacity_accessor, count, ATTR_OPACITY)?;

    let positions = read_position_attribute(&position_accessor, buffers)?;
    let rotations = read_rotation_attribute(&rotation_accessor, buffers)?;
    let scales = read_scale_attribute(&scale_accessor, buffers)?;
    let opacities = read_opacity_attribute(&opacity_accessor, buffers)?;

    let mut sh_channels = Vec::with_capacity(sh_accessors.len());
    for (coefficient_index, accessor) in sh_accessors {
        ensure_count(
            &accessor,
            count,
            &format!("{ATTR_SH_PREFIX}{coefficient_index}"),
        )?;
        sh_channels.push((
            coefficient_index,
            read_sh_coefficient_attribute(&accessor, buffers)?,
        ));
    }

    let mut gaussians = Vec::with_capacity(count);
    for index in 0..count {
        let mut spherical_harmonic =
            crate::material::spherical_harmonics::SphericalHarmonicCoefficients::default();

        for (coefficient_index, values) in &sh_channels {
            let rgb = values[index];
            let base = coefficient_index * SH_CHANNELS;
            if (base + 2) < SH_COEFF_COUNT {
                spherical_harmonic.set(base, rgb[0]);
                spherical_harmonic.set(base + 1, rgb[1]);
                spherical_harmonic.set(base + 2, rgb[2]);
            }
        }

        gaussians.push(Gaussian3d {
            position_visibility: [
                positions[index][0],
                positions[index][1],
                positions[index][2],
                1.0,
            ]
            .into(),
            spherical_harmonic,
            rotation: rotations[index].into(),
            scale_opacity: [
                scales[index][0],
                scales[index][1],
                scales[index][2],
                opacities[index],
            ]
            .into(),
        });
    }

    Ok(gaussians.into())
}

fn required_accessor<'a>(
    document: &'a gltf::Document,
    attributes: &HashMap<String, usize>,
    semantic: &str,
) -> Result<Accessor<'a>, std::io::Error> {
    let index = attributes.get(semantic).ok_or_else(|| {
        std::io::Error::new(
            ErrorKind::InvalidData,
            format!("missing required attribute semantic '{semantic}'"),
        )
    })?;

    document.accessors().nth(*index).ok_or_else(|| {
        std::io::Error::new(
            ErrorKind::InvalidData,
            format!("attribute semantic '{semantic}' references missing accessor index {index}"),
        )
    })
}

fn collect_sh_accessors<'a>(
    document: &'a gltf::Document,
    attributes: &HashMap<String, usize>,
) -> Result<Vec<(usize, Accessor<'a>)>, std::io::Error> {
    let coefficient_map = collect_sh_coefficient_map(attributes)?;
    let mut accessors = Vec::with_capacity(coefficient_map.len());

    for (coefficient_index, accessor_index) in coefficient_map {
        let accessor = document.accessors().nth(accessor_index).ok_or_else(|| {
            std::io::Error::new(
                ErrorKind::InvalidData,
                format!("SH attribute references missing accessor index {accessor_index}"),
            )
        })?;

        accessors.push((coefficient_index, accessor));
    }

    Ok(accessors)
}

fn collect_sh_coefficient_map(
    attributes: &HashMap<String, usize>,
) -> Result<Vec<(usize, usize)>, std::io::Error> {
    let mut degrees: BTreeMap<usize, BTreeMap<usize, usize>> = BTreeMap::new();

    for (semantic, accessor_index) in attributes {
        let Some((degree, coefficient)) = parse_sh_semantic(semantic) else {
            continue;
        };

        degrees
            .entry(degree)
            .or_default()
            .insert(coefficient, *accessor_index);
    }

    let Some(degree_zero) = degrees.get(&0) else {
        return Err(std::io::Error::new(
            ErrorKind::InvalidData,
            "missing required spherical harmonics attribute 'KHR_gaussian_splatting:SH_DEGREE_0_COEF_0'",
        ));
    };
    if !degree_zero.contains_key(&0) {
        return Err(std::io::Error::new(
            ErrorKind::InvalidData,
            "missing required spherical harmonics attribute 'KHR_gaussian_splatting:SH_DEGREE_0_COEF_0'",
        ));
    }

    let max_degree = *degrees.keys().max().unwrap_or(&0);
    if max_degree > 3 {
        return Err(std::io::Error::new(
            ErrorKind::InvalidData,
            format!(
                "unsupported spherical harmonics degree {max_degree}; KHR_gaussian_splatting supports degrees up to 3"
            ),
        ));
    }

    let supported_degree = max_supported_sh_degree();
    if max_degree > supported_degree {
        return Err(std::io::Error::new(
            ErrorKind::InvalidData,
            format!(
                "asset requires spherical harmonics degree {max_degree}, but this build supports up to degree {supported_degree}"
            ),
        ));
    }

    for degree in 0..=max_degree {
        let expected_count = 2 * degree + 1;
        let Some(coefficients) = degrees.get(&degree) else {
            return Err(std::io::Error::new(
                ErrorKind::InvalidData,
                format!(
                    "spherical harmonics degree {degree} is required because higher degrees are present, but its coefficients are missing"
                ),
            ));
        };

        if coefficients.len() != expected_count {
            return Err(std::io::Error::new(
                ErrorKind::InvalidData,
                format!(
                    "spherical harmonics degree {degree} must define exactly {expected_count} coefficients"
                ),
            ));
        }

        for coefficient in 0..expected_count {
            if !coefficients.contains_key(&coefficient) {
                return Err(std::io::Error::new(
                    ErrorKind::InvalidData,
                    format!(
                        "spherical harmonics degree {degree} is partially defined; missing coefficient {coefficient}"
                    ),
                ));
            }
        }
    }

    let mut coefficient_map = Vec::new();
    for degree in 0..=max_degree {
        let coefficients = &degrees[&degree];
        for coefficient in 0..(2 * degree + 1) {
            let accessor_index = coefficients[&coefficient];
            coefficient_map.push((sh_coefficient_index(degree, coefficient), accessor_index));
        }
    }

    Ok(coefficient_map)
}

fn parse_sh_semantic(semantic: &str) -> Option<(usize, usize)> {
    let rest = semantic.strip_prefix(ATTR_SH_PREFIX)?;
    let (degree, coefficient) = rest.split_once("_COEF_")?;

    Some((degree.parse().ok()?, coefficient.parse().ok()?))
}

fn sh_coefficient_index(degree: usize, coefficient: usize) -> usize {
    degree * degree + coefficient
}

fn max_supported_sh_degree() -> usize {
    let mut degree = 0usize;
    loop {
        let coefficient_count = (degree + 1) * (degree + 1);
        if coefficient_count >= SH_COEFF_COUNT_PER_CHANNEL {
            return degree;
        }
        degree += 1;
    }
}

fn ensure_count(
    accessor: &Accessor<'_>,
    expected_count: usize,
    semantic: &str,
) -> Result<(), std::io::Error> {
    let count = accessor.count();
    if count == expected_count {
        return Ok(());
    }

    Err(std::io::Error::new(
        ErrorKind::InvalidData,
        format!("attribute semantic '{semantic}' has {count} entries; expected {expected_count}"),
    ))
}

fn read_position_attribute(
    accessor: &Accessor<'_>,
    buffers: &[Vec<u8>],
) -> Result<Vec<[f32; 3]>, std::io::Error> {
    if accessor.dimensions() != Dimensions::Vec3 {
        return Err(std::io::Error::new(
            ErrorKind::InvalidData,
            format!(
                "attribute semantic '{ATTR_POSITION}' must use accessor type VEC3, got {:?}",
                accessor.dimensions()
            ),
        ));
    }

    if accessor.data_type() != DataType::F32 {
        return Err(std::io::Error::new(
            ErrorKind::InvalidData,
            format!(
                "attribute semantic '{ATTR_POSITION}' must use float components, got {:?}",
                accessor.data_type()
            ),
        ));
    }

    read_items::<[f32; 3]>(accessor, buffers, ATTR_POSITION)
}

fn read_rotation_attribute(
    accessor: &Accessor<'_>,
    buffers: &[Vec<u8>],
) -> Result<Vec<[f32; 4]>, std::io::Error> {
    if accessor.dimensions() != Dimensions::Vec4 {
        return Err(std::io::Error::new(
            ErrorKind::InvalidData,
            format!(
                "attribute semantic '{ATTR_ROTATION}' must use accessor type VEC4, got {:?}",
                accessor.dimensions()
            ),
        ));
    }

    let normalized = accessor.normalized();
    let mut values = match accessor.data_type() {
        DataType::F32 => read_items::<[f32; 4]>(accessor, buffers, ATTR_ROTATION)?,
        DataType::I8 if normalized => read_items::<[i8; 4]>(accessor, buffers, ATTR_ROTATION)?
            .into_iter()
            .map(|v| {
                [
                    normalize_i8(v[0]),
                    normalize_i8(v[1]),
                    normalize_i8(v[2]),
                    normalize_i8(v[3]),
                ]
            })
            .collect(),
        DataType::I16 if normalized => read_items::<[i16; 4]>(accessor, buffers, ATTR_ROTATION)?
            .into_iter()
            .map(|v| {
                [
                    normalize_i16(v[0]),
                    normalize_i16(v[1]),
                    normalize_i16(v[2]),
                    normalize_i16(v[3]),
                ]
            })
            .collect(),
        _ => {
            return Err(std::io::Error::new(
                ErrorKind::InvalidData,
                format!(
                    "attribute semantic '{ATTR_ROTATION}' must use float or normalized signed integer components"
                ),
            ));
        }
    };

    for quaternion in &mut values {
        normalize_quaternion(quaternion)?;
    }

    Ok(values)
}

fn read_scale_attribute(
    accessor: &Accessor<'_>,
    buffers: &[Vec<u8>],
) -> Result<Vec<[f32; 3]>, std::io::Error> {
    if accessor.dimensions() != Dimensions::Vec3 {
        return Err(std::io::Error::new(
            ErrorKind::InvalidData,
            format!(
                "attribute semantic '{ATTR_SCALE}' must use accessor type VEC3, got {:?}",
                accessor.dimensions()
            ),
        ));
    }

    let normalized = accessor.normalized();
    let mut values = match accessor.data_type() {
        DataType::F32 => read_items::<[f32; 3]>(accessor, buffers, ATTR_SCALE)?,
        DataType::I8 => read_items::<[i8; 3]>(accessor, buffers, ATTR_SCALE)?
            .into_iter()
            .map(|v| {
                if normalized {
                    [normalize_i8(v[0]), normalize_i8(v[1]), normalize_i8(v[2])]
                } else {
                    [v[0] as f32, v[1] as f32, v[2] as f32]
                }
            })
            .collect(),
        DataType::I16 => read_items::<[i16; 3]>(accessor, buffers, ATTR_SCALE)?
            .into_iter()
            .map(|v| {
                if normalized {
                    [
                        normalize_i16(v[0]),
                        normalize_i16(v[1]),
                        normalize_i16(v[2]),
                    ]
                } else {
                    [v[0] as f32, v[1] as f32, v[2] as f32]
                }
            })
            .collect(),
        _ => {
            return Err(std::io::Error::new(
                ErrorKind::InvalidData,
                format!(
                    "attribute semantic '{ATTR_SCALE}' must use float or signed integer components"
                ),
            ));
        }
    };

    for scale in &mut values {
        scale[0] = scale[0].exp();
        scale[1] = scale[1].exp();
        scale[2] = scale[2].exp();
    }

    Ok(values)
}

fn read_opacity_attribute(
    accessor: &Accessor<'_>,
    buffers: &[Vec<u8>],
) -> Result<Vec<f32>, std::io::Error> {
    if accessor.dimensions() != Dimensions::Scalar {
        return Err(std::io::Error::new(
            ErrorKind::InvalidData,
            format!(
                "attribute semantic '{ATTR_OPACITY}' must use accessor type SCALAR, got {:?}",
                accessor.dimensions()
            ),
        ));
    }

    let normalized = accessor.normalized();
    let values: Vec<f32> = match accessor.data_type() {
        DataType::F32 => read_items::<f32>(accessor, buffers, ATTR_OPACITY)?,
        DataType::U8 if normalized => read_items::<u8>(accessor, buffers, ATTR_OPACITY)?
            .into_iter()
            .map(normalize_u8)
            .collect(),
        DataType::U16 if normalized => read_items::<u16>(accessor, buffers, ATTR_OPACITY)?
            .into_iter()
            .map(normalize_u16)
            .collect(),
        _ => {
            return Err(std::io::Error::new(
                ErrorKind::InvalidData,
                format!(
                    "attribute semantic '{ATTR_OPACITY}' must use float or normalized unsigned integer components"
                ),
            ));
        }
    };

    if values
        .iter()
        .any(|opacity| !opacity.is_finite() || *opacity < 0.0 || *opacity > 1.0)
    {
        return Err(std::io::Error::new(
            ErrorKind::InvalidData,
            format!(
                "attribute semantic '{ATTR_OPACITY}' contains out-of-range values; opacity must be in [0, 1]"
            ),
        ));
    }

    Ok(values)
}

fn read_sh_coefficient_attribute(
    accessor: &Accessor<'_>,
    buffers: &[Vec<u8>],
) -> Result<Vec<[f32; 3]>, std::io::Error> {
    if accessor.dimensions() != Dimensions::Vec3 {
        return Err(std::io::Error::new(
            ErrorKind::InvalidData,
            "spherical harmonics attributes must use accessor type VEC3",
        ));
    }

    if accessor.data_type() != DataType::F32 {
        return Err(std::io::Error::new(
            ErrorKind::InvalidData,
            "spherical harmonics attributes must use float components",
        ));
    }

    read_items::<[f32; 3]>(accessor, buffers, "KHR_gaussian_splatting:SH")
}

fn read_items<T: Item + Copy>(
    accessor: &Accessor<'_>,
    buffers: &[Vec<u8>],
    semantic: &str,
) -> Result<Vec<T>, std::io::Error> {
    let iter = Iter::<T>::new(accessor.clone(), |buffer| {
        buffers.get(buffer.index()).map(Vec::as_slice)
    })
    .ok_or_else(|| {
        std::io::Error::new(
            ErrorKind::InvalidData,
            format!(
                "failed to decode accessor data for attribute semantic '{semantic}' (accessor index {})",
                accessor.index()
            ),
        )
    })?;

    Ok(iter.collect())
}

fn normalize_quaternion(quaternion: &mut [f32; 4]) -> Result<(), std::io::Error> {
    let length_sq = quaternion
        .iter()
        .map(|component| component * component)
        .sum::<f32>();
    if length_sq <= f32::EPSILON {
        return Err(std::io::Error::new(
            ErrorKind::InvalidData,
            format!(
                "attribute semantic '{ATTR_ROTATION}' contains a zero-length quaternion, which is invalid"
            ),
        ));
    }

    let inv_length = length_sq.sqrt().recip();
    quaternion[0] *= inv_length;
    quaternion[1] *= inv_length;
    quaternion[2] *= inv_length;
    quaternion[3] *= inv_length;

    Ok(())
}

fn normalize_i8(value: i8) -> f32 {
    (value as f32 / 127.0).max(-1.0)
}

fn normalize_i16(value: i16) -> f32 {
    (value as f32 / 32767.0).max(-1.0)
}

fn normalize_u8(value: u8) -> f32 {
    value as f32 / 255.0
}

fn normalize_u16(value: u16) -> f32 {
    value as f32 / 65535.0
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn validates_complete_sh_coefficients_for_supported_degree() {
        let mut attributes = HashMap::new();
        attributes.insert(
            "KHR_gaussian_splatting:SH_DEGREE_0_COEF_0".to_owned(),
            0usize,
        );

        let supported_degree = max_supported_sh_degree().min(3);
        let mut index = 1usize;
        for degree in 1..=supported_degree {
            for coefficient in 0..(2 * degree + 1) {
                attributes.insert(
                    format!("KHR_gaussian_splatting:SH_DEGREE_{degree}_COEF_{coefficient}"),
                    index,
                );
                index += 1;
            }
        }

        let result = collect_sh_coefficient_map(&attributes).unwrap();
        assert_eq!(
            result.len(),
            (supported_degree + 1) * (supported_degree + 1)
        );
        assert_eq!(result[0].0, 0);
        assert_eq!(result.last().unwrap().0, result.len() - 1);
    }

    #[test]
    fn rejects_partial_sh_degree() {
        let mut attributes = HashMap::new();
        attributes.insert(
            "KHR_gaussian_splatting:SH_DEGREE_0_COEF_0".to_owned(),
            0usize,
        );
        attributes.insert(
            "KHR_gaussian_splatting:SH_DEGREE_1_COEF_0".to_owned(),
            1usize,
        );
        attributes.insert(
            "KHR_gaussian_splatting:SH_DEGREE_1_COEF_2".to_owned(),
            2usize,
        );

        let err = collect_sh_coefficient_map(&attributes).unwrap_err();
        let message = err.to_string();
        if max_supported_sh_degree() >= 1 {
            assert!(message.contains("must define exactly"));
        } else {
            assert!(message.contains("supports up to degree"));
        }
    }

    #[test]
    fn decodes_base64_data_uri() {
        let uri = "data:application/octet-stream;base64,AAECAwQF";
        let data = decode_data_uri(uri).unwrap().unwrap();
        assert_eq!(data, vec![0, 1, 2, 3, 4, 5]);
    }

    #[test]
    fn decodes_percent_encoded_data_uri() {
        let uri = "data:application/octet-stream,%00%01%02%03";
        let data = decode_data_uri(uri).unwrap().unwrap();
        assert_eq!(data, vec![0, 1, 2, 3]);
    }

    #[test]
    fn requires_khr_extension_in_extensions_used() {
        let err = ensure_gaussian_extension_used(&[]).unwrap_err();
        assert!(err.to_string().contains("extensionsUsed"));

        ensure_gaussian_extension_used(&[KHR_GAUSSIAN_SPLATTING_EXTENSION.to_owned()]).unwrap();
    }
}

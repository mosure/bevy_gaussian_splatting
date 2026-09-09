# Calibrated 3DGS camera paths

`camera::path::GaussianCameraPath::from_json` reads the standard 3DGS camera
array: `id`, optional `img_name`, `width`, `height`, `fx`, `fy`, `position`, and
row-major camera-to-world `rotation`. File order defines frame indices; authored
IDs need not be contiguous. Empty paths, duplicate IDs, invalid intrinsics and
non-orthonormal or reflected rotations are rejected before rendering.

The source camera axes are right/down/forward. Bevy uses right/up/back, so the
imported camera basis is `R * diag(1, -1, -1)` with the original translation.
Cloud coordinates are unchanged. For a source-camera point `(x,y,z)`, the
result projects to `u = width/2 + fx*x/z`, `v = height/2 + fy*y/z`.
The JSON does not describe distortion or principal-point offsets; the importer
uses a centered pinhole model. Rotation includes authored roll.

`frame.transform()` and `frame.projection(near, far)` can drive any
`GaussianCamera`. The custom `GaussianCameraIntrinsics` projection preserves
independent `fx`/`fy` through viewport changes, including sub-view crops. Pixel
intrinsics scale independently with output width and height. Match the source
aspect ratio to avoid stretching the resized image.

For a native viewer using a compatible SH0 package:

```sh
cargo run --release --no-default-features \
  --features 'planar lod_render sh0 viewer io_flexbuffers io_ply file_asset' \
  --bin bevy_gaussian_splatting -- \
  --input-lod /absolute/path/to/scene.gsplatlod \
  --camera-path 'assets/Jastrzębia_Góra_camera_path.json' \
  --camera-path-index 0 --camera-controller flycam --camera-speed 200 \
  --width 1280 --height 850 \
  --global-order --lod-gpu-traversal --lod-max-active-gaussians 65536
```

`--camera-controller flycam` selects the actual
[`bevy_flycam`](https://github.com/sburris0/bevy_flycam) plugin and is useful for
moving through large scenes. Its cursor starts released: press `Tab` to capture
or release it. While captured, mouse motion looks around, `WASD` moves and
`Space` / left `Shift` move up/down in the current camera basis. Movement speed
defaults to 12 world units/second; `--camera-speed 200` suits this large scene.
`--camera-sensitivity` defaults to 0.00012 in the plugin's units. Both settings
must be finite and positive (speed ≤1,000,000, sensitivity ≤1).

Add `--match-ground-plane` with flycam to estimate a navigation up direction from
the first cloud's geometry. The config/query field is `match_ground_plane=true`; it defaults
to false and requires `camera_controller=flycam`. The viewer samples at most 4,096
finite, visible, nontransparent means from an already loaded flat cloud or the
package's resident CPU staging slots. It does not reload the PLY, scan all source
records or read GPU memory. Resident sampling makes at most 16,384 address probes;
sparse startup allocations can yield fewer samples.

At most eight asynchronous fits are attempted, no more frequently than once per
two seconds, after adequate samples arrive. A fit needs at least 35% support,
broad two-dimensional spread and no similarly supported competing tilt. The
normal points toward the initial camera where possible; camera up only resolves
near-plane sign ambiguity. There is no assumed Y-up/Z-up orientation. A dominant
roof or facade can still be geometrically indistinguishable from terrain: this is
an optional navigation aid, not semantic ground detection. An unclear estimate
leaves the authored navigation orientation in place.

The accepted estimate is frozen for that scene identity and cloud transform.
Ground matching changes the navigation basis and levels roll only on actual
manual takeover. It never moves the source geometry or changes camera intrinsics;
held poses and `Home` retain their authored pose until input resumes.
`WASD` stays parallel to the estimated plane, `Space` / left `Shift` follows its
normal, and mouse yaw keeps that fixed up direction without accumulating roll.

For Poland, add `--camera-controller flycam --camera-speed 200 --match-ground-plane`
to the existing package launch. The corresponding query is
`?camera_controller=flycam&camera_speed=200&match_ground_plane=true`.

The same flat config/query fields are available:
`?camera_controller=flycam&camera_speed=200&camera_sensitivity=0.00012` or
`{"camera_controller":"flycam","camera_speed":200,"camera_sensitivity":0.00012}`
within the existing viewer configuration. `--camera-controller orbit` remains the
default for compatibility. Without ground matching, flycam leaves the rendered camera's calibrated
projection and authored roll intact and applies the plugin's motion in its local
basis; changing modes does not replace the camera with a generic perspective.

`--camera-path-index` holds that exact frame by default. Add
`--camera-path-fps 30` to advance through the remaining frames and hold the last
one. The imported pose, roll and independent focal lengths remain unchanged
until navigation begins. In orbit mode, left-drag orbits around a point ahead of
the camera, right-drag pans, and the mouse wheel zooms. Navigation stops path
playback; clicking or toggling cursor capture without movement does not. `Home` restores the initial
`--camera-path-index` pose and holds it. Navigation preserves the calibrated
projection and uses the camera's local axes, including authored roll. Its initial
orbit distance scales with the extent of the authored route. `F12` saves a
screenshot in both modes; `S` is an additional screenshot shortcut only in orbit
mode, since flycam reserves it for backward movement. `Esc` closes the viewer.

Camera input requires the pointer in the focused viewer window and yields to
editor UI input. Authored paths take precedence over automatic scene framing.
These are native viewer options; the reusable parser/projection also compiles
for WebGPU applications.

The Jastrzębia Góra file has 600 frames at 4946×3286 with `fx=4649.505859375`
and `fy=4627.30029296875`. Indices `0, 150, 300, 450, 599` provide a bounded
set of route checkpoints. The dataset is local input, not bundled crate data.
Importing its cameras establishes a coordinate contract, not rendered image
quality or large-scene performance. See the [capture guide](lod_capture.md)
and [current acceptance gates](lod_implementation_status.md).

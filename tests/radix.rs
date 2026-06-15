use bevy_gaussian_splatting::{RadixSortDepthBits, render::ShaderDefines};

#[derive(Clone, Copy, Debug)]
struct TestEntry {
    dist2: f32,
    key: u32,
}

#[test]
fn radix_depth_key_preserves_close_order_during_camera_motion() {
    let positions = [(-0.02_f32, 0.0_f32, 1.0_f32), (0.02, 0.0, 1.0)];
    let cameras = [(-0.01_f32, 0.0_f32, 0.0_f32), (0.01, 0.0, 0.0)];

    for radix_sort_depth_bits in [RadixSortDepthBits::Bits24, RadixSortDepthBits::Bits32] {
        let defines = ShaderDefines::for_radix_depth_bits(radix_sort_depth_bits);

        for camera in cameras {
            let mut entries = positions
                .iter()
                .map(|position| {
                    let dist2 = distance_squared(*position, camera);
                    TestEntry {
                        dist2,
                        key: radix_depth_key(dist2, &defines),
                    }
                })
                .collect::<Vec<_>>();

            entries.sort_by_key(|entry| entry.key);

            for pair in entries.windows(2) {
                assert!(
                    pair[0].dist2 >= pair[1].dist2,
                    "radix order must remain back-to-front while the camera moves; precision={radix_sort_depth_bits:?}, camera={camera:?}, sorted={entries:?}",
                );
            }
        }
    }
}

#[test]
fn radix_depth_bit_settings_select_expected_pass_count_and_shift() {
    let cases = [
        (RadixSortDepthBits::Bits16, 2, 16, 0),
        (RadixSortDepthBits::Bits24, 3, 8, 1),
        (RadixSortDepthBits::Bits32, 4, 0, 0),
    ];

    for (
        radix_sort_depth_bits,
        expected_digit_places,
        expected_key_shift,
        expected_initial_parity,
    ) in cases
    {
        let defines = ShaderDefines::for_radix_depth_bits(radix_sort_depth_bits);

        assert_eq!(defines.radix_digit_places, expected_digit_places);
        assert_eq!(defines.radix_key_shift, expected_key_shift);
        assert_eq!(defines.radix_initial_parity(), expected_initial_parity);
    }
}

#[test]
fn radix_initial_parity_finishes_in_sorted_entries_buffer() {
    for radix_sort_depth_bits in [
        RadixSortDepthBits::Bits16,
        RadixSortDepthBits::Bits24,
        RadixSortDepthBits::Bits32,
    ] {
        let defines = ShaderDefines::for_radix_depth_bits(radix_sort_depth_bits);
        let initial_parity = defines.radix_initial_parity();
        let final_pass_parity = (initial_parity + defines.radix_digit_places as usize - 1) % 2;

        // Bind group parity 1 writes entry_buffer_b -> sorted_entries, which is the
        // buffer consumed by rendering.
        assert_eq!(final_pass_parity, 1);
    }
}

#[test]
fn radix_16_bit_depth_key_can_collapse_close_depths() {
    let defines = ShaderDefines::for_radix_depth_bits(RadixSortDepthBits::Bits16);
    let near_key = radix_depth_key(
        distance_squared((-0.02, 0.0, 1.0), (-0.01, 0.0, 0.0)),
        &defines,
    );
    let far_key = radix_depth_key(
        distance_squared((0.02, 0.0, 1.0), (-0.01, 0.0, 0.0)),
        &defines,
    );

    assert_eq!(near_key, far_key);
}

fn distance_squared(position: (f32, f32, f32), camera: (f32, f32, f32)) -> f32 {
    let dx = position.0 - camera.0;
    let dy = position.1 - camera.1;
    let dz = position.2 - camera.2;
    dx * dx + dy * dy + dz * dz
}

fn radix_depth_key(dist2: f32, defines: &ShaderDefines) -> u32 {
    let key = 0xFFFF_FFFF_u32 - dist2.to_bits();
    key >> defines.radix_key_shift
}

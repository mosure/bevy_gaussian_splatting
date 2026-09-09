use bevy_gaussian_splatting::{RadixSortDepthBits, render::ShaderDefines};

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

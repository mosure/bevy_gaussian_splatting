#define_import_path bevy_gaussian_splatting::sort


@group(0) @binding(0) var<storage, read> points: array<GaussianInput>;

@group(0) @binding(1) var<storage, write> sorted_points: array<GaussianInput>;


fn sort() {
    let num_points = points.length();

    for (let i = 0; i < num_points; i++) {
        sorted_points[i] = points[i];
    }

    for (let i = 0; i < num_points; i++) {
        let min_index = i;
        let min_value = sorted_points[i].position.z;

        for (let j = i + 1; j < num_points; j++) {
            if (sorted_points[j].position.z < min_value) {
                min_index = j;
                min_value = sorted_points[j].position.z;
            }
        }

        let temp = sorted_points[i];
        sorted_points[i] = sorted_points[min_index];
        sorted_points[min_index] = temp;
    }

    workgroupBarrier();
}

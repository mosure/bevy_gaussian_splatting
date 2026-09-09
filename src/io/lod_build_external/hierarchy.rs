//! Independent source-derived cohorts, bounded by admitted worker scratch.

use super::*;

pub(super) fn admit_workers(
    levels: &[u64],
    config: ExternalLodBuildConfig,
    spatial_scratch: u64,
) -> Result<(u32, u64), ExternalLodBuildError> {
    let Some(&first_internal_nodes) = levels.get(1) else {
        return Ok((0, 0));
    };
    let cohorts = balanced_group_count(
        first_internal_nodes,
        u64::from(config.settings.branching_factor),
        false,
    );
    // In addition to the conservative spatial/risk-aware scratch, account for
    // retained per-node rung metadata, one child's morph boundaries, and page
    // encode/decode temporaries. This also covers streamed rungs whose page
    // capacity may exceed the fixed risk-aware source cap.
    let records = u64::from(config.settings.leaf_capacity);
    let per_record = (size_of::<crate::gaussian::formats::planar_3d_lod::MomentMergeResult>()
        + size_of::<LodSourceRange>()
        + size_of::<std::ops::Range<usize>>()
        + size_of::<u16>()
        + size_of::<u64>() * 2) as u64;
    let rung_bytes = records
        .checked_mul(u64::from(config.settings.branching_factor))
        .and_then(|count| count.checked_mul(per_record));
    let page_bytes = records
        .checked_mul(size_of::<Gaussian3d>() as u64)
        .and_then(|bytes| bytes.checked_add(PAGE_CONTAINER_HEADER_BYTES))
        .and_then(|bytes| bytes.checked_mul(4));
    let per_worker = rung_bytes
        .and_then(|rungs| page_bytes?.checked_add(rungs))
        .and_then(|bytes| bytes.checked_add(spatial_scratch))
        .and_then(|bytes| bytes.checked_add(config.limits.run_buffer_bytes as u64))
        .ok_or_else(|| {
            ExternalLodBuildError::InvalidConfig("hierarchy worker scratch overflow".into())
        })?;
    let workers = (config.limits.hierarchy_workers as u64)
        .min(cohorts)
        .min(config.limits.max_hierarchy_working_bytes / per_worker);
    if workers == 0 {
        return Err(ExternalLodBuildError::LimitExceeded {
            field: "minimum hierarchy cohort working bytes",
            actual: per_worker,
            limit: config.limits.max_hierarchy_working_bytes,
        });
    }
    Ok((workers as u32, per_worker * workers))
}

#[derive(Default)]
pub(super) struct InternalLevel {
    pub drafts: Vec<NodeDraft>,
    pub descriptors: Vec<LodPageDescriptor>,
    pub summaries: Vec<ReductionSummary>,
    pub stored_gaussians: u64,
    pub maximum_encoded_page_bytes: u64,
    pub maximum_representative_source_records: u64,
    pub maximum_risk_aware_source_records: u64,
    pub maximum_risk_aware_host_bytes: u64,
    pub touching_pairs: u64,
    pub measured_pairs: u64,
    pub unmeasured_pairs: u64,
    pub cross_cohort_pairs: u64,
    pub concurrent_cohorts: u32,
}

impl InternalLevel {
    fn append(&mut self, mut other: Self) -> Result<(), ExternalLodBuildError> {
        self.drafts.append(&mut other.drafts);
        self.descriptors.append(&mut other.descriptors);
        self.summaries.append(&mut other.summaries);
        for (total, value) in [
            (&mut self.stored_gaussians, other.stored_gaussians),
            (&mut self.touching_pairs, other.touching_pairs),
            (&mut self.measured_pairs, other.measured_pairs),
            (&mut self.unmeasured_pairs, other.unmeasured_pairs),
        ] {
            *total = total.checked_add(value).ok_or_else(|| {
                ExternalLodBuildError::InvalidConfig("hierarchy cohort counter overflow".into())
            })?;
        }
        self.maximum_encoded_page_bytes = self
            .maximum_encoded_page_bytes
            .max(other.maximum_encoded_page_bytes);
        self.maximum_representative_source_records = self
            .maximum_representative_source_records
            .max(other.maximum_representative_source_records);
        self.maximum_risk_aware_source_records = self
            .maximum_risk_aware_source_records
            .max(other.maximum_risk_aware_source_records);
        self.maximum_risk_aware_host_bytes = self
            .maximum_risk_aware_host_bytes
            .max(other.maximum_risk_aware_host_bytes);
        Ok(())
    }
}

/// Children and output groups are balanced identically to the serial writer.
/// Returning only finalized metadata permits dynamic work scheduling without
/// retaining every source-bearing cohort while waiting for an earlier result.
pub(super) fn build_internal_level(
    path: &Path,
    source_count: u64,
    current: &[ReductionSummary],
    draft_start: usize,
    pages_directory: &Path,
    config: ExternalLodBuildConfig,
    worker_limit: usize,
) -> Result<InternalLevel, ExternalLodBuildError> {
    let group_count = balanced_group_count(
        current.len() as u64,
        u64::from(config.settings.branching_factor),
        false,
    );
    let (base, remainder) = balanced_group_sizes(current.len() as u64, group_count);
    let cohort_count = balanced_group_count(
        group_count,
        u64::from(config.settings.branching_factor),
        false,
    );
    let (cohort_base, cohort_remainder) = balanced_group_sizes(group_count, cohort_count);
    let group_start = |index: u64| index * base + index.min(remainder);
    let (cohorts, intervals) = run_bounded_indexed_tasks(
        cohort_count as usize,
        worker_limit.min(cohort_count as usize),
        |cohort_index| {
            let cohort_index = cohort_index as u64;
            let first_group = cohort_index * cohort_base + cohort_index.min(cohort_remainder);
            let end_group = first_group + cohort_base + u64::from(cohort_index < cohort_remainder);
            let child_start = group_start(first_group) as usize;
            let child_end = group_start(end_group) as usize;
            let children = &current[child_start..child_end];
            let source_start = children[0].source.start;
            let source_end = children.last().unwrap().source.end().ok_or_else(|| {
                ExternalLodBuildError::InvalidConfig("cohort source range overflow".into())
            })?;
            let source = LodSourceRange {
                start: source_start,
                count: source_end - source_start,
            };
            let mut reader =
                RunReader::open_range(path, config.limits.run_buffer_bytes, source_count, source)?;
            let mut pending = Vec::with_capacity((end_group - first_group) as usize);
            let mut output = InternalLevel::default();
            for group in first_group..end_group {
                let children =
                    &current[group_start(group) as usize..group_start(group + 1) as usize];
                let node = build_node(&mut reader, children, config)?;
                output.maximum_representative_source_records = output
                    .maximum_representative_source_records
                    .max(node.rung.maximum_partition_records);
                output.maximum_risk_aware_source_records = output
                    .maximum_risk_aware_source_records
                    .max(node.rung.risk_aware_source_records);
                output.maximum_risk_aware_host_bytes = output
                    .maximum_risk_aware_host_bytes
                    .max(node.rung.risk_aware_host_bytes);
                pending.push(node);
            }
            reader.finish()?;
            let report = finalize_spatial_sibling_cohort(
                &mut pending,
                pages_directory,
                config,
                draft_start + first_group as usize,
                &mut output.drafts,
                &mut output.descriptors,
                &mut output.summaries,
                &mut output.stored_gaussians,
                &mut output.maximum_encoded_page_bytes,
            )?;
            output.touching_pairs = u64::from(report.touching_node_pairs);
            output.measured_pairs = u64::from(report.overlapping_node_pairs);
            output.unmeasured_pairs = u64::from(report.unmeasured_touching_node_pairs);
            Ok(output)
        },
    )?;
    let mut result = InternalLevel::default();
    for cohort in cohorts {
        result.append(cohort)?;
    }
    let mut source_offset = 0u64;
    for node in &result.drafts {
        if node.source.start != source_offset {
            return Err(ExternalLodBuildError::Validation(
                "hierarchy level is not a contiguous canonical source partition".into(),
            ));
        }
        source_offset = node.source.end().ok_or_else(|| {
            ExternalLodBuildError::InvalidConfig("hierarchy source range overflow".into())
        })?;
    }
    if source_offset != source_count || result.drafts.len() as u64 != group_count {
        return Err(ExternalLodBuildError::Validation(
            "hierarchy level did not consume its canonical source and child partition".into(),
        ));
    }
    result.cross_cohort_pairs = spatial_cross_cohort_pair_upper_bound_for_level(
        group_count,
        cohort_count,
        cohort_base,
        cohort_remainder,
    )?;
    result.concurrent_cohorts = parallel_interval_stats(&intervals).2;
    Ok(result)
}

fn build_node(
    reader: &mut RunReader,
    children: &[ReductionSummary],
    config: ExternalLodBuildConfig,
) -> Result<PendingInternalNode, ExternalLodBuildError> {
    let mut bounds = children[0].bounds;
    let mut authored_source_bounds = children[0].authored_source_bounds;
    let mut inherited_error = LodError::ZERO;
    let mut high_fidelity_certificate = 1.0_f32;
    let mut child_representation_count = 0_u64;
    for child in children {
        bounds = bounds.union(child.bounds);
        authored_source_bounds = authored_source_bounds.union(child.authored_source_bounds);
        inherited_error = inherited_error.max(child.reduction_error);
        high_fidelity_certificate = high_fidelity_certificate.min(child.high_fidelity_certificate);
        child_representation_count = child_representation_count
            .checked_add(u64::from(child.representation_count))
            .ok_or_else(|| {
                ExternalLodBuildError::InvalidConfig("child representation count overflow".into())
            })?;
    }
    let first = &children[0];
    let last = children.last().unwrap();
    let source_end = last
        .source
        .end()
        .ok_or_else(|| ExternalLodBuildError::InvalidConfig("node source range overflow".into()))?;
    let source = LodSourceRange {
        start: first.source.start,
        count: source_end - first.source.start,
    };
    let representative_count = child_representation_count
        .div_ceil(u64::from(config.settings.branching_factor))
        .max(1);
    let representative_count = u32::try_from(representative_count).map_err(|_| {
        ExternalLodBuildError::InvalidConfig(
            "external rung representation count exceeds u32".into(),
        )
    })?;
    if representative_count > config.settings.leaf_capacity {
        return Err(ExternalLodBuildError::Validation(format!(
            "external rung requires {representative_count} records above leaf capacity {}",
            config.settings.leaf_capacity
        )));
    }
    let rung = build_external_progressive_rung(
        reader,
        source,
        representative_count,
        config.settings.support_sigma,
    )?;
    let child_representation_capacity =
        usize::try_from(child_representation_count).map_err(|_| {
            ExternalLodBuildError::InvalidConfig("child representation count exceeds usize".into())
        })?;
    let mut child_representation_source_ends = Vec::new();
    reserve_exact(
        &mut child_representation_source_ends,
        child_representation_capacity,
        "morph child representation boundaries",
    )?;
    for child in children {
        if let Some(source_ends) = &child.representation_source_ends {
            child_representation_source_ends.extend(source_ends.iter().copied());
        } else {
            for offset in 1..=child.source.count {
                child_representation_source_ends.push(
                    child.source.start.checked_add(offset).ok_or_else(|| {
                        ExternalLodBuildError::InvalidConfig(
                            "leaf morph source boundary overflow".into(),
                        )
                    })?,
                );
            }
        }
    }
    if child_representation_source_ends.len() != child_representation_capacity {
        return Err(ExternalLodBuildError::Validation(
            "morph child boundary count disagrees with child representations".into(),
        ));
    }
    let morph_child_run_lengths = monotone_morph_run_lengths(
        source,
        &rung.source_ranges,
        &child_representation_source_ends,
    )?;
    let representation_source_ends = rung
        .source_ranges
        .iter()
        .map(|range| range.end().unwrap())
        .collect::<Vec<_>>();
    let morton = LodMortonRange {
        min: first.morton.min,
        max: last.morton.max,
    };
    let first_child = children[0].draft_index;
    if children
        .iter()
        .enumerate()
        .any(|(offset, child)| child.draft_index != first_child + offset)
    {
        return Err(ExternalLodBuildError::Validation(
            "hierarchy child drafts are not contiguous".into(),
        ));
    }
    let node = PendingInternalNode {
        children: (first_child, children.len()),
        source,
        morton,
        inherited_bounds: bounds,
        authored_source_bounds,
        inherited_error,
        inherited_high_fidelity_certificate: high_fidelity_certificate,
        rung,
        morph_child_run_lengths,
        representation_source_ends,
    };
    Ok(node)
}

#pragma once

#include <cuda_runtime.h>

#include <cstddef>
#include <cstdint>

namespace gpufastq {

// Delta-encode a monotonic offset vector into same-length uint32 deltas.
// The first output element is copied from the first offset, which is expected
// to be 0 for FASTQ line offsets.
void delta_encode_offsets_to_lengths(const uint64_t *d_offsets,
                                     uint32_t *d_lengths, size_t count,
                                     cudaStream_t stream = 0);

// Decode same-length uint32 deltas back into uint64 offsets via inclusive sum.
void delta_decode_lengths_to_offsets(const uint32_t *d_lengths,
                                     uint64_t *d_offsets, size_t count,
                                     cudaStream_t stream = 0);

} // namespace gpufastq

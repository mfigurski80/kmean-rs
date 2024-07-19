use crate::memory::*;
use crate::{KMeans, KMeansConfig, KMeansState};
use std::simd::{LaneCount, Simd, SupportedLaneCount};

#[inline(always)]
pub fn calculate<T, const LANES: usize>(kmean: &KMeans<T, LANES>, state: &mut KMeansState<T>, _config: &KMeansConfig<'_, T>, computed: Vec<T>)
where
    T: Primitive,
    LaneCount<LANES>: SupportedLaneCount,
    Simd<T, LANES>: SupportedSimdArray<T, LANES>,
{
  
  computed.chunks_exact(kmean.p_sample_dims)
    .enumerate()
    .for_each(|(ci, c)| {
      if ci > state.k {
        panic!("Initialized with more centroids than k");
      }
      state.set_centroid_from_iter(ci, c.iter().cloned());
    });
}


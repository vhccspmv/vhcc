
#include "mem.h"
#include "mmio.h"
#include "timer.h"
#include "vhcc_matrix.h"
#include <cstring>
#include <string>
#include <cmath>
#include <cstdint>
#include <getopt.h>
#include "omp.h"
#include "immintrin.h"
#include "mkl.h"
#include "SparseMatrixReader.h"

int MAX_ITER = 1;

typedef int index_t;
typedef double value_t;

#define LEN16 16
#define LEN8 8


typedef __attribute__((aligned(64))) union zmmi {
	__m512i reg;
	unsigned int elems[LEN16];
} zmmi_t;
typedef __attribute__((aligned(64))) union zmmd {
	__m512d reg;
	__m512i regi32;
	double elems[LEN8];
} zmmd_t;


void compute_spmv(int n_threads, int num_vectors,
									int threads_per_core,
									int num_panels,
									panel_info_t *restrict panel_info,
									thr_info_t   *restrict thr_info,
									index_t *restrict veceor_ptr,
									uint8_t *restrict scan_mask,
									index_t *restrict row_arr,
									index_t *restrict col_arr,
									value_t *restrict vals_arr,
									value_t *restrict input,
									value_t *restrict result)
{
#ifdef __MIC__
#pragma omp parallel default(shared) num_threads(n_threads)
	{

		int id = omp_get_thread_num();

		int core_id = id / threads_per_core;
		int local_thr_id = id % threads_per_core;
		
		int panel_id = thr_info[id].panel_id;
		
		value_t *tmp_result = panel_info[panel_id].tmp_result;
		
		index_t start_vec = thr_info[id].start_vec;
		index_t end_vec   = thr_info[id].end_vec;
		
		zmmi_t row, col, wrmask;
		zmmd_t res, tmp;
		__mmask8 mask1, mask2, mask3, maskwr;
		
		index_t veceor_idx = thr_info[id].vbase;
		index_t scan_idx   = thr_info[id].sbase;
		index_t ridx       = thr_info[id].rbase;
		index_t vec_idx    = start_vec * LEN8;
		
		value_t nrval = 0;
		index_t eor_vec = veceor_ptr[veceor_idx++];
		res.elems[:] = 0;
		for (index_t v = start_vec; v < end_vec; ++v) {
			
			col.elems[0:LEN8] = col_arr[vec_idx:LEN8];
			
			__assume_aligned(&vals_arr[vec_idx], 64);
			
			res.elems[0:LEN8] += vals_arr[vec_idx:LEN8] * 
				input[col.elems[0:LEN8]];
			vec_idx += LEN8;
			
			nrval = 0;
			if (v == eor_vec) {
				mask1 = (__mmask8)scan_mask[scan_idx++];
				mask2 = (__mmask8)scan_mask[scan_idx++];
				mask3 = (__mmask8)scan_mask[scan_idx++];
				maskwr = (__mmask8)scan_mask[scan_idx++];
				
				res.reg = _mm512_mask_add_pd(res.reg, mask1, res.reg, _mm512_swizzle_pd(res.reg, _MM_SWIZ_REG_CDAB));
				res.reg = _mm512_mask_add_pd(res.reg, mask2, res.reg, _mm512_swizzle_pd(res.reg, _MM_SWIZ_REG_BBBB));
				tmp.regi32 = _mm512_permute4f128_epi32(res.regi32, _MM_PERM_BBBA);
				res.reg = _mm512_mask_add_pd(res.reg, mask3, res.reg, _mm512_swizzle_pd(tmp.reg, _MM_SWIZ_REG_BBBB));
				
				if ((maskwr & 0x80) == 0)
					nrval = res.elems[LEN8-1];

				int bcnt = _mm_countbits_32(maskwr);
				int a = -1;
				int x = maskwr;
				for (int i = 0; i < bcnt; ++i) {
					int y = _mm_tzcnti_32(a, x);
					index_t r = row_arr[ridx+i];
					tmp_result[r] += res.elems[y];
					a = y;
				}
				ridx += bcnt;

				eor_vec = veceor_ptr[veceor_idx++];
				
			} else {

				res.reg = _mm512_add_pd(res.reg, _mm512_swizzle_pd(res.reg, _MM_SWIZ_REG_CDAB));
				res.reg = _mm512_add_pd(res.reg, _mm512_swizzle_pd(res.reg, _MM_SWIZ_REG_BBBB));
				nrval = res.elems[LEN8-1] + res.elems[3];
			}
			
			res.elems[:] = 0;
			res.elems[0] = nrval;
		}
		
#pragma omp barrier

		index_t nridx = thr_info[id].last_row;
		nrval = tmp_result[thr_info[id].overflow_row];

#pragma omp atomic update
		tmp_result[nridx] += nrval;
	
#pragma omp barrier
		
		index_t merge_start = thr_info[id].merge_start;
		index_t merge_end   = thr_info[id].merge_end;
		index_t blk_size    = 512;
		
		for (index_t i = merge_start; i < merge_end; i += blk_size) {
			index_t blk_end = i + blk_size > merge_end ? merge_end : i + blk_size;
			for (int c = 0; c < num_panels; ++c) {
				for (index_t b = i; b < blk_end; b += LEN8) {
					result[b:LEN8] += panel_info[c].tmp_result[b:LEN8];
				}
			}
		}
	}
#endif
}

void compute_spmv1(int n_threads, int num_vectors,
									 thr_info_t *restrict thr_info,
									 index_t *restrict veceor_ptr,
									 uint8_t *restrict scan_mask,
									 index_t *restrict row_arr,
									 index_t *restrict col_arr,
									 value_t *restrict vals_arr,
									 value_t *restrict input,
									 value_t *restrict result)
{
#ifdef __MIC__
#pragma omp parallel default(shared) num_threads(n_threads)
	{

		int id = omp_get_thread_num();
		index_t start_vec = thr_info[id].start_vec;
		index_t end_vec   = thr_info[id].end_vec;

		zmmi_t row, col, wrmask;
		zmmd_t res, tmp;
		__mmask8 mask1, mask2, mask3, maskwr;

		index_t cidx       = thr_info[id].vbase;
		index_t veceor_idx = thr_info[id].vbase;
		index_t scan_idx   = thr_info[id].vbase * 4;
		index_t ridx       = thr_info[id].rbase;
		index_t vec_idx    = start_vec * LEN8;

		value_t nrval = 0;
		index_t eor_vec = veceor_ptr[veceor_idx++];
		res.elems[:] = 0;
		for (index_t v = start_vec; v < end_vec; ++v) {
			
			col.elems[0:LEN8] = col_arr[vec_idx:LEN8];

			__assume_aligned(&vals_arr[vec_idx], 64);
			res.elems[0:LEN8] += vals_arr[vec_idx:LEN8] * input[col.elems[0:LEN8]];
			vec_idx += LEN8;

			nrval = 0;
			if (v == eor_vec) {
				mask1 = (__mmask8)scan_mask[scan_idx++];
				mask2 = (__mmask8)scan_mask[scan_idx++];
				mask3 = (__mmask8)scan_mask[scan_idx++];
				maskwr = (__mmask8)scan_mask[scan_idx++];
				
				res.reg = _mm512_mask_add_pd(res.reg, mask1, res.reg, _mm512_swizzle_pd(res.reg, _MM_SWIZ_REG_CDAB));
				res.reg = _mm512_mask_add_pd(res.reg, mask2, res.reg, _mm512_swizzle_pd(res.reg, _MM_SWIZ_REG_BBBB));
				tmp.regi32 = _mm512_permute4f128_epi32(res.regi32, _MM_PERM_BBBA);
				res.reg = _mm512_mask_add_pd(res.reg, mask3, res.reg, _mm512_swizzle_pd(tmp.reg, _MM_SWIZ_REG_BBBB));

				if ((maskwr & 0x80) == 0)
					nrval = res.elems[LEN8-1];

				int bcnt = _mm_countbits_32(maskwr);
				int a = -1;
				int x = maskwr;
				for (int i = 0; i < bcnt; ++i) {
					int y = _mm_tzcnti_32(a, x);
					index_t r = row_arr[ridx+i];
					result[r] += res.elems[y];
					a = y;
				}
				ridx += bcnt;

				eor_vec = veceor_ptr[veceor_idx++];
				
			} else {
				
				res.reg = _mm512_add_pd(res.reg, _mm512_swizzle_pd(res.reg, _MM_SWIZ_REG_CDAB));
				res.reg = _mm512_add_pd(res.reg, _mm512_swizzle_pd(res.reg, _MM_SWIZ_REG_BBBB));
				nrval = res.elems[LEN8-1] + res.elems[3];
			}

			res.elems[:] = 0;
			res.elems[0] = nrval;
		}

#pragma omp barrier

		index_t nridx = thr_info[id].last_row;
		nrval = result[thr_info[id].overflow_row];
#pragma omp atomic update
		result[nridx] += nrval;
	}
#endif
}


int main(int argc, char *argv[])
{
	int c = 0;
	char *filename = NULL;
	int num_threads = 240;
	int tile_row = -1;
	int tile_col = -1;
	int num_panel = 1;
	int threads_per_core = MAX_THREADS_PER_CORE;
	bool set_iter = false;
	bool use_binary_input = false;
	while ((c = getopt(argc, argv, "f:r:c:t:i:b:p:y:")) != -1) {
		switch(c) {
		case 'f':
			filename = optarg;
			break;
		case 'b':
			filename = optarg;
			use_binary_input = true;
			break;
		case 'r':
			tile_row = atoi(optarg);
			break;
		case 'c':
			tile_col = atoi(optarg);
			break;
		case 't':
			num_threads = atoi(optarg);
			break;
		case 'p':
			num_panel = atoi(optarg);
			break;
		case 'y':
			threads_per_core = atoi(optarg);
			break;
		case 'i':
			MAX_ITER = atoi(optarg);
			set_iter = true;
			break;
		default:
			exit(1);
		}
	}
	if (filename == NULL) {
		printf("Error: specify matrix filename.\n");
		exit(1);
	}
	if (num_panel > 60) {
		printf("Error. Expect number of panels to be between 1 and 60 inclusive\n");
		exit(1);
	}
	
	kmp_set_defaults("KMP_AFFINITY=compact");

	int m, n, nnz;
	int *row_idx, *col_idx;
	double *tvals;
 
	if(use_binary_input) {
		if(!SparseMatrixReader::ReadEncodedData(filename, &m, &n, &nnz, &row_idx, &col_idx, &tvals)) {
			return EXIT_FAILURE;
		}
	} else {
		if (!SparseMatrixReader::ReadRawData(filename, &m, &n, &nnz, &row_idx, &col_idx, &tvals)) {
			printf("Reading input matrix fails!\n");
			return -1;
		} 
	}

  value_t *vals = (value_t *)MALLOC(nnz * sizeof(value_t));
	for (index_t i = 0; i < nnz; ++i)
		vals[i] = tvals[i];

	if (!set_iter) {
		MAX_ITER = 2000*1000000/(2*nnz);
	}
	printf("Iterations: %u\n", MAX_ITER);

  vhcc_matrix<index_t, value_t> mat(m, n, nnz, row_idx, col_idx, vals);
  mat.convert(num_threads, threads_per_core, num_panel, tile_row, tile_col);

  index_t n_rows = m;
	index_t n_cols = n;
	index_t padrows            = mat.get_pad_rows();
	index_t padcols            = mat.get_pad_cols();
	index_t num_vectors        = mat.get_num_vectors();
	int         num_panels     = mat.get_num_panels();
	panel_info_t *panel_info   = mat.get_panel_info();
	thr_info_t *thr_info       = mat.get_thr_info();
	index_t     thr_info_size  = mat.get_thr_info_size();
	index_t    *veceor_ptr     = mat.get_veceor_ptr();
	index_t     veceor_size    = mat.get_veceor_size();
	uint8_t    *scan_mask      = mat.get_scan_mask();
	index_t     scan_mask_size = mat.get_scan_mask_size();
	index_t    *row_arr        = mat.get_row_arr();
	index_t     row_arr_size   = mat.get_row_arr_size();
	index_t    *col_arr        = mat.get_col_arr();
	index_t     col_arr_size   = mat.get_col_arr_size();
	value_t    *vals_arr       = mat.get_vals_arr();
	index_t     vals_arr_size  = mat.get_vals_arr_size();

  value_t *input = (value_t *)MALLOC(padcols * sizeof(value_t));
  value_t *result = (value_t *)MALLOC(padrows * sizeof(value_t));
	for (int i = 0; i < padcols; ++i) {
		input[i] = 1;
	}
	for (int i = 0; i < padrows; ++i) {
		result[i] = 0;
	}

	cputimer timer;
	double elapsed;

	if (num_panels > 1) {
		// warm up
		for (int i = 0; i < 1; ++i) {
			compute_spmv(num_threads, num_vectors, threads_per_core, num_panels, panel_info, thr_info, veceor_ptr, scan_mask, row_arr, col_arr, vals_arr, input, result);
		}

		for (int i = 0; i < padrows; ++i) result[i] = 0;
		for (int i = 0; i < num_panels; ++i) {
			memset(panel_info[i].tmp_result, 0, padrows * sizeof(value_t));
		}

		double tstart = dsecnd();
		for (int i = 0; i < MAX_ITER; ++i) {
			compute_spmv(num_threads, num_vectors, threads_per_core, num_panels, panel_info, thr_info, veceor_ptr, scan_mask, row_arr, col_arr, vals_arr, input, result);
		}
		elapsed = (dsecnd() - tstart) * 1000;
	} else {
		// warm up
		for (int i = 0; i < 1; ++i) {
			compute_spmv1(num_threads, num_vectors, thr_info, veceor_ptr, scan_mask, row_arr, col_arr, vals_arr, input, result);
		}

		for (int i = 0; i < padrows; ++i) result[i] = 0;
		for (int i = 0; i < num_panels; ++i) {
			memset(panel_info[i].tmp_result, 0, padrows * sizeof(value_t));
		}

		double tstart = dsecnd();
		for (int i = 0; i < MAX_ITER; ++i) {
			compute_spmv1(num_threads, num_vectors, thr_info, veceor_ptr, scan_mask, row_arr, col_arr, vals_arr, input, result);
		}
		elapsed = (dsecnd() - tstart) * 1000;
	}

	elapsed = elapsed / MAX_ITER;
	printf("Gflops: %f\n", double(2.0*nnz)/elapsed/1000000);

	FREE(input);
	FREE(result);

	return 0;
}
  

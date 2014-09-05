
ICC = icpc
ICCFLAGS = -O3 -std=c++0x -vec-report0 -restrict
DEFINES = 
INCLUDE_DIRS = -I/opt/intel/mkl/include  -I$(HOME)/local/include/poski -I$(HOME)/local/build_oski/include/oski/  -L$(HOME)/local/lib/
MKL_ROOT = /opt/intel/mkl
MKL_DIRS = -L$(MKL_ROOT)/lib/intel64
MKL_LIBS_SEQ = -lmkl_intel_lp64 -lmkl_core -lmkl_sequential -lpthread -lm
MKL_LIBS_THR = -lmkl_intel_lp64 -lmkl_core -lmkl_intel_thread -lpthread -lm
MMIC = -mmic
OUTPATH = ./


phi_spmv: $(wildcard *.h) $(wildcard *.cpp)
	$(ICC) $(ICCFLAGS) $(DEFINES) $(MMIC) -c -o $(OUTPATH)/mmio.o mmio.cpp
	$(ICC) $(ICCFLAGS) $(DEFINES) $(MMIC) -c -o $(OUTPATH)/MatrixDataConverter.o MatrixDataConverter.cpp
	$(ICC) $(ICCFLAGS) $(DEFINES) $(MMIC) -c -o $(OUTPATH)/SparseMatrixReader.o SparseMatrixReader.cpp
	$(ICC) $(ICCFLAGS) $(DEFINES) $(INCLUDE_DIRS) \
		$(MMIC) -openmp -o $(OUTPATH)/phi_spmv $(OUTPATH)/mmio.o $(OUTPATH)/SparseMatrixReader.o $(OUTPATH)/MatrixDataConverter.o phi_spmv.cpp $(LIBS) \
		-Wl,--start-group $(MKL_ROOT)/lib/mic/libmkl_intel_lp64.a $(MKL_ROOT)/lib/mic/libmkl_intel_thread.a $(MKL_ROOT)/lib/mic/libmkl_core.a -Wl,--end-group



clean:
	rm -rf ./*.o phi_spmv

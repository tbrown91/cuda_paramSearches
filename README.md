# cuda_paramSearches
Parameter searches for transcription model in CUDA/C

compile as follows:

nvcc cuda_paramSearchSeparate.cu -o cuda_paramSearchSeparateGAL1 -O3

run with the command:

./cuda_paramSearchSeparateGAL1 data_files.txt output_file.txt

Data should be in the form 50 rows by 50 columns where rows correspond to nuclear counts and columns cytoplasmic counts per cell

default: lib/sqa.so
	python3 main.py

lib/sqa.so: simulatedQA.cu
	nvcc --compiler-options -fPIC -shared simulatedQA.cu -o lib/sqa.so
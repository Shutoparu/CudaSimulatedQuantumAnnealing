default: lib/sqa.so
	python3 main.py

lib/sqa.so: simulatedQA_binary.c
	nvcc --compiler-options -fPIC -shared simulatedQA_binary.c -o lib/sqa.so
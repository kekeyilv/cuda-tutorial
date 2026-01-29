build/vecadd: src/vecadd.cu build
	nvcc src/vecadd.cu -o build/vecadd
	
build:
	mkdir build

run: build/vecadd
	python3 run.py

clean:
	rm -rf ./build
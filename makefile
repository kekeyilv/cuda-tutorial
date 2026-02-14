ifndef NAME
$(error Do not make in the root of the project)
endif

SRCS += ../framework/framework.cu

build/$(NAME): $(SRCS) ../framework/framework.cuh build
	nvcc $(SRCS) -I ../framework -o build/$(NAME)
	
build:
	mkdir build

run: build/$(NAME)
	python3 run.py

clean:
	rm -rf ./build
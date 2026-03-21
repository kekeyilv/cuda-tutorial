ifndef NAME
$(error Do not make in the root of the project)
endif

SRCS += ../framework/framework.cu

build/$(NAME): $(SRCS) ../framework/framework.cuh build
	nvcc $(SRCS) -I ../framework -lcurand -o build/$(NAME)
	
build:
	mkdir build

run: build/$(NAME)
	cd ../ && python3 benchmark.py $(NAME)

clean:
	rm -rf ./build
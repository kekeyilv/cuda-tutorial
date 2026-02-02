ifndef NAME
$(error Do not make in the root of the project)
endif

build/$(NAME): $(SRCS) build
	nvcc $(SRCS) -o build/$(NAME)
	
build:
	mkdir build

run: build/$(NAME)
	python3 run.py

clean:
	rm -rf ./build
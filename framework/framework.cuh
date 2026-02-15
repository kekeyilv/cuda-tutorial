#ifndef FRAMEWORK_CUH
#define FRAMEWORK_CUH

#include <cuda_fp16.h>
#include <cuda_runtime.h>

#include <chrono>
#include <functional>
#include <iomanip>
#include <iostream>
#include <random>
#include <string>
#include <vector>
using time_point = std::chrono::steady_clock::time_point;

template <typename... Types>
struct TypeList {};

template <std::size_t Index, typename List>
struct type_at;

template <std::size_t Index, typename Head, typename... Tail>
struct type_at<Index, TypeList<Head, Tail...>> {
    using type = typename type_at<Index - 1, TypeList<Tail...>>::type;
};

template <typename Head, typename... Tail>
struct type_at<0, TypeList<Head, Tail...>> {
    using type = Head;
};

template <typename T>
struct CudaArg {
    ~CudaArg() {};
    T hostArg, kernelArg;
    size_t size;
    CudaArg<T>* clone() const {
        return new CudaArg<T>{
            .hostArg = hostArg,
            .kernelArg = kernelArg,
            .size = size,
        };
    }
    void toHost() { hostArg = kernelArg; }
    bool isEqualWith(const CudaArg<T>& b, float tolerance) const {
        return this->hostArg == b.hostArg && this->kernelArg == b.kernelArg;
    }
};

template <typename T>
struct CudaArg<T*> {
    ~CudaArg() {
        delete[] hostArg;
        cudaFree(kernelArg);
    };
    T *hostArg, *kernelArg;
    size_t size;
    CudaArg<T*>* clone() const {
        CudaArg<T*>* result = new CudaArg<T*>{
            .hostArg = new T[size],
            .size = size,
        };
        memcpy(result->hostArg, hostArg, size * sizeof(T));
        cudaMalloc(&result->kernelArg, size * sizeof(T));
        cudaMemcpy(result->kernelArg, kernelArg, size * sizeof(T),
                   cudaMemcpyDeviceToDevice);
        return result;
    }
    void toHost() {
        cudaMemcpy(hostArg, kernelArg, size * sizeof(T),
                   cudaMemcpyDeviceToHost);
    }
    bool isEqualWith(const CudaArg<T*>& b, float tolerance) const {
        for (int i = 0; i < size; i++) {
            if (fabs(toFloat(hostArg[i]) - toFloat(b.hostArg[i])) > tolerance) {
                printf("%d %f %f\n", i, toFloat(hostArg[i]),
                       toFloat(b.hostArg[i]));
                return false;
            }
        }
        return true;
    }

   private:
    static float toFloat(T value) { return 0.0f; }
};

template <>
float CudaArg<float*>::toFloat(float value);

template <>
float CudaArg<half*>::toFloat(half value);

template <typename T>
class CudaArgInitializer {
   public:
    virtual void init(CudaArg<T>& arg) const = 0;
};

template <typename T = float>
class CudaNewArray : public CudaArgInitializer<T*> {
   public:
    CudaNewArray(size_t array_size) { this->array_size = array_size; }

    void init(CudaArg<T*>& arg) const override {
        arg.size = array_size;
        arg.hostArg = new T[array_size];
        cudaMalloc(&arg.kernelArg, array_size * sizeof(T));
    }

   private:
    size_t array_size;
};

template <typename T = float>
class CudaRandomArray : public CudaArgInitializer<T*> {
   public:
    CudaRandomArray(size_t array_size, float rand_min, float rand_max) {
        this->array_size = array_size;
        this->rand_min = rand_min;
        this->rand_max = rand_max;
    }
    void init(CudaArg<T*>& arg) const override {
        arg.size = array_size;
        arg.hostArg = new T[array_size];
        cudaMalloc(&arg.kernelArg, array_size * sizeof(T));
        int64_t seed =
            std::chrono::steady_clock::now().time_since_epoch().count();
        std::mt19937 engine(seed);
        std::uniform_real_distribution<float> dist(rand_min, rand_max);
        for (size_t i = 0; i < array_size; i++) {
            arg.hostArg[i] = fromFloat(dist(engine));
        }
        cudaMemcpy(arg.kernelArg, arg.hostArg, array_size * sizeof(T),
                   cudaMemcpyHostToDevice);
    }

   private:
    static T fromFloat(float value) { return (T)value; }
    size_t array_size;
    float rand_min, rand_max;
};

template <>
float CudaRandomArray<float>::fromFloat(float value);

template <>
half CudaRandomArray<half>::fromFloat(float value);

template <typename T>
class CudaSetValue : public CudaArgInitializer<T> {
   public:
    CudaSetValue(T value) { this->value = value; }
    void init(CudaArg<T>& arg) const override {
        arg.hostArg = arg.kernelArg = value;
    }

   private:
    T value;
};

template <typename... Args>
class CudaTask {
   public:
    // Returns the running time in ms
    virtual float run(CudaArg<Args>&... args) = 0;
    virtual bool onHost() = 0;
    std::string name;
};

template <typename... Args>
class CudaHostTask : public CudaTask<Args...> {
   public:
    CudaHostTask(std::string name, void (*func)(Args...)) {
        this->name = name;
        this->func = func;
    }

    float run(CudaArg<Args>&... args) override {
        time_point start_h = std::chrono::steady_clock::now();
        func(args.hostArg...);
        time_point end_h = std::chrono::steady_clock::now();
        int64_t duration =
            std::chrono::duration_cast<std::chrono::microseconds>(end_h -
                                                                  start_h)
                .count();
        return duration / 1000.0f;
    }

    bool onHost() override { return true; }

   private:
    std::function<void(Args...)> func;
};

template <typename... Args>
class CudaKernelTask : public CudaTask<Args...> {
   public:
    CudaKernelTask(std::string name, dim3 grid_size, dim3 block_size,
                   size_t shared_mem_size, void (*func)(Args...)) {
        this->name = name;
        this->func = func;
        this->grid_size = grid_size;
        this->block_size = block_size;
        this->shared_mem_size = shared_mem_size;
    }

    float run(CudaArg<Args>&... args) override {
        float elapsed_time = 0;
        cudaEvent_t start_d, end_d;
        cudaEventCreate(&start_d);
        cudaEventCreate(&end_d);
        cudaEventRecord(start_d);
        func<<<grid_size, block_size, shared_mem_size>>>(args.kernelArg...);
        cudaEventRecord(end_d);
        cudaEventSynchronize(end_d);
        cudaEventElapsedTime(&elapsed_time, start_d, end_d);
        return elapsed_time;
    }

    bool onHost() override { return false; }

   private:
    void (*func)(Args...);
    dim3 grid_size, block_size;
    size_t shared_mem_size;
};

template <typename... Args>
class CudaApp {
   public:
    CudaApp() {};
    CudaApp* initArgs(const CudaArgInitializer<Args>&... inits) {
        initArgsImpl(std::index_sequence_for<Args...>{}, inits...);
        return this;
    }
    CudaApp* addTask(CudaTask<Args...>* task) {
        this->tasks.push_back(task);
        return this;
    }
    template <size_t ResultIndex>
    void run(float tolerance) {
        std::cout << std::setw(15) << "Name";
        std::cout << std::setw(15) << "Time";
        std::cout << std::setw(10) << "Result" << std::endl;
        std::cout << "-------------------------------------------" << std::endl;
        using ResultType =
            typename type_at<ResultIndex, TypeList<Args...>>::type;
        std::vector<CudaArg<ResultType>*> results;
        CudaArg<ResultType>& resultArg = std::get<ResultIndex>(args);

        for (auto task : this->tasks) {
            auto func = [task](CudaArg<Args>&... args) {
                return task->run(args...);
            };
            float time = std::apply(func, this->args);
            if (!task->onHost()) {
                resultArg.toHost();
            }
            int result_type = 0;
            for (; result_type < results.size(); result_type++) {
                if (results[result_type]->isEqualWith(resultArg, tolerance)) {
                    break;
                }
            }
            if (result_type == results.size()) {
                results.emplace_back(resultArg.clone());
            }
            std::cout << std::setw(15) << task->name;
            std::cout << std::setw(13) << std::fixed << std::setprecision(2)
                      << time << "ms";
            std::cout << std::setw(10) << (char)('A' + result_type)
                      << std::endl;
        }

        for (auto result : results) {
            delete result;
        }
        std::cout << std::endl;
    }

   private:
    template <size_t... Indexes>
    void initArgsImpl(std::index_sequence<Indexes...>,
                      const CudaArgInitializer<Args>&... inits) {
        (..., inits.init(std::get<Indexes>(args)));
    }

    std::tuple<CudaArg<Args>...> args;
    std::vector<CudaTask<Args...>*> tasks;
};

#endif
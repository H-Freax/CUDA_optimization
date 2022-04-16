# CUDA_optimization
# CUDA优化入门实战

注：本markdown代码部分格式高亮基于c++

## 新建一个CUDA项目（first_cuda.cu）

首先需要引用所需的库

要使用到runtime API的时候需要引用cuda_runtime.h

```cmake
#include <stdio.h>
#include <cuda_runtime.h>
```

编写主函数部分

判断CUDA是否可用

```cmake
int main(){
    if(!InitCUDA()){
        return 0;
    }
    
    printf("CUDA initialized.\n");

    return 0;
}
```

判断CUDA是否可用

```c++
bool InitCUDA(){
    int count;
    
    cudaGetDeviceCount(&count); //获取可用的device数目
    if(count == 0){
        fprintf(stderr, "There is no device.\n");
        return false;
    }

    
    int i;	
    for(int i = 0; i<count;i++){
        cudaDeviceProp prop;
        if(cudaGetDeviceProperties(&prop, i) == cudaSuccess){
            if(prop.major >= 1){
                break;
            }
        }
    }
    
    if(i == count){
        fprintf(stderr, "There is no device supporting CUDA 1.x.\n");
        return false;
    }

    cudaSetDevice(i);

    return true;
}
```

首先使用cudaGetDeviceCount函数获取支持CUDA的device数目，如果没有的话会返回1，device 0 会是一个模拟的device但是不支持CUDA 1.0以上的功能。

需要对每个device使用cudaGetDeviceProperties函数取得device的相关资料，例如device的名称、memory的大小，最大的thread数目等等。同时通过该函数判断该device所支持的CUDA版本（prop.majar和prop.minor分别代表装备支持的版本号，例如1.0即prop.major =1，prop.minor为0

如果找到了支持CUDA 1.0版本以上的device以后，可以使用cudaSetDevice将该device设为目前需要使用的device



完成以上内容以后就可以使用nvcc来complie这个文件了。

- nvcc是CUDA的compile工具，会将.cu拆解出GPU或者host上需要执行的部分。在GPU上执行的部分会采用NVIDIA提供的compiler编译成中介码，host上执行的部分会通过系统上的C++ compiler编译。

## 用CUDA做个简单的加法(cuda_sum1.cu)

该部分在first_cuda.cu的基础上进行

目标是计算一堆数字的平方和

首先把开头部分改成

```c++
#include<stdio.h>
#include<stdlib.h>
#include<cuda_runtime.h>

#define DATA_SIZE 1048576

int data[DATA_SIZE];
```

编写生成随机数的函数

```c++
void GenerateNumbers(int *number, int size){
    for(int i = 0; i < size; i++){
        number[i] = rand() % 10;
    }
}
```

CUDA执行计算前需要把数据从主记忆体复制到显示记忆体中，才能让显示晶片使用，因此需要获取一块大小适当的显示记忆体来复制数据。在main函数中加入以下部分

```c++
   GenerateNumbers(data,DATASIZE);
    int* gpudata, *result;

    cudaMalloc((void**) &gpudata,sizeof(int) * DATA_SIZE);
    cudaMalloc((void**) &result,sizeof(int));
    //从主记忆体复制到显示记忆体，所以使用 cudaMemcpyHostToDevice。
    //如果是从显示记忆体复制到主记忆体，则使用 cudaMemcpyDeviceToHost
    cudaMemcpy(gpudata, data,sizeof(int) * DATA_SIZE,
        cudaMemcpyHostToDevice);
```

首先引用GenerateNumbers函数产生数据，使用cudaMalloc取得大小合适的记忆体，gpudata变量用来存储数据，result用来存储计算结果，通过cudaMemcpy将数据复制到显示记忆体中。

cudaMalloc喝cudaMemcpy的用法和一般的malloc以及memcpy类似，不过cudaMemcpy多出一个参数，指示复制记忆体的方向，如果是从主记忆体复制到显示记忆体，使用 cudaMemcpyHostToDevice，如果是从显示记忆体复制到主记忆体，则使用 cudaMemcpyDeviceToHost。

接下来编写在显示晶片上的函数，在CUDA中，在函数前面加上\__global__表示这个函数是在显示晶片上执行的。

```c++
__global__ static void sumOfSquares(int *num, int* result)
{
    int sum = 0;
    int i;
    for(i = 0; i < DATA_SIZE; i++) {
        sum += num[i] * num[i];
    }

    *result = sum;
}
```

在显示晶片上执行程序有一定的限制条件，例如不能传回值。

接下来是让CUDA执行函数，在CUDA中执行函数需要使用以下的语法

```c++
函数名称<<<block 数目, thread 数目, shared memory 大小>>>(参数);
```

执行完毕后需要将结果从显示晶片复制回主记忆体，在main函数中加入以下部分

```c++
    sumOfSquares<<<1, 1, 0>>>(gpudata, result);

    int sum;
    cudaMemcpy(&sum, result, sizeof(int), cudaMemcpyDeviceToHost);
    cudaFree(gpudata);
    cudaFree(result);

    printf("sum: %d\n", sum);
```

编写的部分设置了只适用一个thread，所以block数目跟thread数目都为1，没有使用到shared memory，设为0。

如果要验证结果是否正确，可以加入CPU的执行代码进行验证

```c++
    sum = 0;
    for(int i = 0; i < DATA_SIZE; i++) {
        sum += data[i] * data[i];
    }
    printf("CPU sum: %d\n", sum);
```

CUDA提供了clock函数可以取得timestamp，适合用来判断程序执行的时间，单位为GPU执行单元的时间，可以在优化程序时间时提供参考，如果需要记录时间的话，需要更改sumOfSquares为以下内容

```c++
__global__ static void sumOfSquares(int *num, int* result, clock_t* time){
    int sum = 0;
    int i;
    clock_t start = clock();
    for(i =0; i<DATA_SIZE; i++){
        sum+= num[i]*num[i];
    }

    *result = sum;
    *time = clock()-start;
}
```

main函数的部分修改如下：

```c++
    int* gpudata, *result;
    clock_t* time;
    cudaMalloc((void**) &gpudata, sizeof(int) * DATA_SIZE);
    cudaMalloc((void**) &result, sizeof(int));
    cudaMalloc((void**) &time, sizeof(clock_t));
    cudaMemcpy(gpudata, data, sizeof(int) * DATA_SIZE,
        cudaMemcpyHostToDevice);

    sumOfSquares<<<1, 1, 0>>>(gpudata, result, time);

    int sum;
    clock_t time_used;
    cudaMemcpy(&sum, result, sizeof(int), cudaMemcpyDeviceToHost);
    cudaMemcpy(&time_used, time, sizeof(clock_t),
        cudaMemcpyDeviceToHost);
    cudaFree(gpudata);
    cudaFree(result);

    printf("sum: %d time: %d\n", sum, time_used);

```

编译后运行就可以查看所花费的时间了。

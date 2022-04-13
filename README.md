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

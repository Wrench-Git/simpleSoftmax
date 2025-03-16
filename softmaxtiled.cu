#include "cx.h"
#include "cxtimers.h"
#include <random>
#include <cmath>
#include <limits>

int getSMCount(int deviceId) {
	int deviceCount;
    	cudaGetDeviceCount(&deviceCount); // 获取设备数量
	
	if (deviceCount == 0 || deviceId >= deviceCount) return 0;
	cudaSetDevice(deviceId);
	cudaDeviceProp deviceProp;
	cudaGetDeviceProperties(&deviceProp, deviceId); 
	return deviceProp.multiProcessorCount;
}

int hostsoftmax(float * C, float* A, int Ay, int Ax)
{
	for(int i=0;i<Ay; i++){
	    float max_in_this_line = A[Ax*i];
	    for(int j=0; j<Ax; j++){
	        max_in_this_line = A[Ax*i+j]>max_in_this_line ? A[Ax*i+j] : max_in_this_line;
	    }
	    float sum_in_this_line = 0.0;
	    for(int j=0; j<Ax; j++){
	        C[Ax*i+j] = exp(A[Ax*i+j]-max_in_this_line);
		sum_in_this_line += C[Ax*i+j];
	    }
	    for(int j=0; j<Ax; j++){
	    	C[Ax*i+j] /= sum_in_this_line;
	    }
	}
	return 0;
}

bool areFloatEqual(float a, float b, float episilon = std::numeric_limits<float>::epsilon()){
	return std::fabs(a-b) < episilon;
}

bool checkAccuracy(float * A, float * Ref, int size){
	for(int i=0;i<size;i++){
		if(!areFloatEqual(A[i],Ref[i],0.0001f)) return false;
	}
	return true;
}

#define LOAD_FLOAT4(A,B) *reinterpret_cast<float4*>(&A) = *reinterpret_cast<const float4*>(&B)
#define LOAD_FLOAT2(A,B) *reinterpret_cast<float2*>(&A) = *reinterpret_cast<const float2*>(&B)
#define ROUND_UP_TO_32(A) ((A+31) & ~31)

__device__ float warpReduceMax(float val) {
	for (int offset = 16; offset > 0; offset /= 2) {
		float tmp = __shfl_xor_sync(0xFFFFFFFF, val, offset);
		val = max(val, tmp);
	}
	return val;
}

__device__ float warpReduceSum(float val) {
	for (int offset = 16; offset > 0; offset /= 2) {
		val +=__shfl_xor_sync(0xFFFFFFFF, val, offset);
	}
	return val;
}

template <int TS> __global__ void softmaxtiled0(r_Ptr<float> C, cr_Ptr<float> A, int Ay,int Ax)
{
	float sum = 0.0;
	float PrefixMax = -FLT_MAX;
	int warpIdx = threadIdx.x/warpSize;
	int thIdx = threadIdx.x - warpSize*warpIdx;
	// every warp deals with one row.
	int row = (blockDim.x*blockIdx.x + threadIdx.x)/warpSize;
	if( row >= Ay) return;
	int rowOffset = row * Ax;
	for(int col = 0; col < Ax; col += warpSize){
		// first assume the col num is a multiple of 32.
		float element = A[rowOffset + col + thIdx];
		float warpMax = warpReduceMax(element);
		element = exp(element-warpMax);
		float warpSum = warpReduceSum(element);
		if(col == 0) PrefixMax = warpMax;
		float maxdiff = warpMax - PrefixMax;
		if(maxdiff > 0){
			sum = sum * exp(-maxdiff) + warpSum;
			PrefixMax = warpMax;
		}
		else sum += warpSum * exp(maxdiff);
	}
        for(int col = 0; col < Ax; col+= warpSize){
		C[rowOffset + col + thIdx] = exp(A[rowOffset + col + thIdx]-PrefixMax)/sum;
	}
}

int main(int argc,char *argv[])
{
	int kernel_index = (argc > 1) ? atoi(argv[1]) : 0;
	int Arow = (argc > 2) ? atoi(argv[2]) : 1 << 10; // default 2^10
	int Acol = (argc > 3) ? atoi(argv[3]) : Arow;
	int Crow = Arow;
	int Ccol = Acol;
	uint tilex = (argc > 5) ? atoi(argv[5]) : 32;
	int nacc = (argc > 6) ? atoi(argv[6]) : 100;   // for timing

	thrust::host_vector<float>       A(Arow*Acol);
	thrust::host_vector<float>       C(Crow*Ccol);
	thrust::host_vector<float>       Ref(Crow*Ccol);
	thrust::device_vector<float> dev_A(Arow*Acol);
	thrust::device_vector<float> dev_C(Crow*Ccol);

	// initialise x with random numbers and copy to dx.
	std::default_random_engine gen(12345678);
	std::uniform_real_distribution<float> fran(0.0,1.0);
	for(int k = 0; k<Arow*Acol; k++) A[k] = fran(gen);
	hostsoftmax(Ref.data(),A.data(),Arow,Acol);

	dev_A = A;  // H2D copy
	
	double t3 = 0.0;
	int SMs = getSMCount(0);
	if(kernel_index == 0){
		// warpSoftmax kernel 0
		unsigned int threads_per_block = max(ROUND_UP_TO_32(Arow/(4*SMs)),128);
		dim3 threads ={threads_per_block,1,1}; // force square
		unsigned int blocks_per_grid = (Arow+threads_per_block-1) / threads_per_block * 32;
		dim3 blocks ={blocks_per_grid,1,1};
		std::cout << threads_per_block << "  " << blocks_per_grid << std::endl;
		cx::timer tim;
		for(int k=0;k<nacc;k++){
			softmaxtiled0<32><<<blocks,threads>>>(dev_C.data().get(),dev_A.data().get(),Arow,Acol);
		}
		cudaDeviceSynchronize();
		t3 = tim.lap_ms()/(double)(nacc);
	} else {
		std::cout << "Unsupported kernel!" << std::endl;
	}
	C = dev_C; // D2H copy

	double flops = 4.0*(double)Arow*(double)Acol; // 1 sub, 1 exp, 1 sum , 1 div
	double gflops = flops/(t3*1000000.0);
	double gbytes = gflops*6.0; // i.e 12 bytes per term
	bool accuracy = checkAccuracy(C.data(), Ref.data(),Ccol*Crow);
	printf("A %d x %d, accuracy %s, gpu time %.3f ms, GFlops %.3f, GBytes %.3f (gputiled)\n",Arow,Acol,accuracy?"true":"false",t3,gflops,gbytes);

	return 0;
}

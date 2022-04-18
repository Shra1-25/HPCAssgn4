#include<stdio.h>
#include<cuda.h>

#define BLOCKSIZE 16
#define EPS 1.0e-15

cudaDeviceProp deviceProp;	


double *host_mat,*host_vec,*host_res,*cpu_result;
double *device_mat,*device_vec,*device_res;
int     lenV ,matR , matC;
int     size = BLOCKSIZE;

double calculate_gbs(float &Tsec)
{
        float bw=(1.0e-9 * (( size*size + size )/Tsec));
	return bw;
}

void cpu_multiply()
{
	cpu_result = (double *)malloc(matR*sizeof(double));
	if(cpu_result==NULL)
                {
                        printf("Not enough memory");
                        exit(-1);
                }

	int i,j;
	for(i=0;i<matR;i++)
	{cpu_result[i]=0;
	for(j=0;j<matC;j++)
	cpu_result[i]+=host_mat[i*lenV+j]*host_vec[j];
	}
}

void device_free(double * arr[],int len)
{
        for(int i=0;i<len;i++)
                cudaFree(arr[i]);
        
}

/* function to calculate relative error*/
void relative_error(double* device,double* host,int size)
{
        double relativeError=0.0,maxError=0.0;
        int flag=0;
        int i;

        for( i = 0; i < size; ++i) 
        {
               
                relativeError = fabs((host[i] - device[i]) )/ max(fabs(host[i]), fabs(device[i]));
                
                if (relativeError > EPS && relativeError != 0.0e+00 )
                {       
                        maxError = max(maxError, relativeError);
                        flag = 1;                        
                }

        }
        if( flag == 1)
        {
                printf(" \n Verification failed with error %e on machine with precision %e", maxError, EPS);
        }
        
}

void fill_matrix(double* vec,int size)
{
        int ind;
        for(ind=0;ind<size;ind++)
                vec[ind]=drand48();
}


__global__ void MatVectMultiplication(double *device_mat, double *device_vec,int matR, int lenV,double *device_res)
  {
        int thrdx = blockIdx.x*blockDim.x + threadIdx.x;
        int thrdy = blockIdx.y*blockDim.y + threadIdx.y;
        int t_idx=thrdx+gridDim.x*BLOCKSIZE*thrdy;


        if(t_idx<matR)
	{
                int i;int m=t_idx*lenV;
                device_res[t_idx]=0.00;
                for(i=0;i<lenV;i++)
                device_res[t_idx]+=device_mat[m+i]*device_vec[i];
	}

     __syncthreads();

  }//end of MatVect device function



void MatVectMul()
{
        int max=BLOCKSIZE*BLOCKSIZE;
        int BlocksPerGrid= matR/max+1;
        dim3 dimBlock(BLOCKSIZE,BLOCKSIZE);
        if(matR%max==0)BlocksPerGrid--;
        dim3 dimGrid(1,BlocksPerGrid);
        
        MatVectMultiplication<<<dimGrid,dimBlock>>>(device_mat,device_vec,matR,lenV,device_res);

}


double simulation()
{
       	lenV = matC = matR = size;
       	
	float elapsedTime,Tsec;
	cudaEvent_t start,end;


	host_mat =new double[matR*matC];
	host_vec = new double[lenV];
	host_res = new double[matR];

	
        if(host_mat==NULL || host_vec == NULL || host_res == NULL)
        {
                printf("Not enough memory\n");
                exit(-1);
        }

	fill_matrix(host_mat,matR*matC);
	fill_matrix(host_vec,lenV);

 	
        cudaEventCreate (&start);
        cudaEventCreate (&end);

	cudaMalloc( (void**)&device_mat, matR*matC* sizeof(double));
	cudaMalloc( (void**)&device_vec, lenV* sizeof(double));
	cudaMalloc( (void**)&device_res, matR* sizeof(double));

	cudaMemcpy((void*)device_mat, (void*)host_mat, matR*matC*sizeof(double) ,cudaMemcpyHostToDevice);
	cudaMemcpy((void*)device_vec, (void*)host_vec,lenV*sizeof(double),cudaMemcpyHostToDevice);

	cudaEventRecord (start, 0);
	
	MatVectMul();
	
	cudaEventRecord (end, 0);
	cudaEventSynchronize (end);
	cudaEventElapsedTime ( &elapsedTime, start, end);

	Tsec= 1.0e-3*elapsedTime;
 	
        double ret = calculate_gbs(Tsec);
	
	
  	cudaMemcpy((void*)host_res, (void*)device_res,matR*sizeof(double),cudaMemcpyDeviceToHost);

	cpu_multiply();
  	relative_error(cpu_result,host_res,size);
   	/*free the memory from GPU */
	double *array[3];
        array[0]=device_mat;
        array[1]=device_vec;
        array[2]=device_res;
        device_free(array,3);

	//free host memory----------
        free(host_mat);
        free(host_vec);
        free(host_res);
        free(cpu_result);

	return ret;
}// end of main


int main()
{       
       
        cudaSetDevice(0);
      
        int device;
        // Current Device Detection
        cudaGetDevice(&device);
        cudaGetDeviceProperties(&deviceProp,device);
        printf("Using device %d: %s \n", device, deviceProp.name);

        printf("Size \t \t Bandwith\n");
        for(size = 16 ;size <= 8192*2;size *=2)
        {
                double gbs = simulation();
                printf("%d \t \t %f\n", size, gbs);
        }


}
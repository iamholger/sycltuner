#include "omp.h"
#include <cstdio>
#include <iostream>
#include <numeric>
#include <random>
#include <CL/sycl.hpp>
using namespace sycl;

#include <chrono>


#ifdef NOGPU
static queue Q(cpu_selector{});
#else
static queue Q(gpu_selector{});
#endif

void flux(
 const double * __restrict__ Q, // Q[5+0],
 int                                          normal,
 double * __restrict__ F // F[5],
) 
{
  constexpr double gamma = 1.4;
  const double irho = 1./Q[0];
  #if Dimensions==3
  const double p = (gamma-1) * (Q[4] - 0.5*irho*(Q[1]*Q[1]+Q[2]*Q[2]+Q[3]*Q[3]));
  #else
  const double p = (gamma-1) * (Q[4] - 0.5*irho*(Q[1]*Q[1]+Q[2]*Q[2]));
  #endif

  const double coeff = irho*Q[normal+1];
  F[0] = coeff*Q[0];
  F[1] = coeff*Q[1];
  F[2] = coeff*Q[2];
  F[3] = coeff*Q[3];
  F[4] = coeff*Q[4];
  F[normal+1] += p;
  F[4]        += coeff*p;
}


double maxEigenvalue(
  const double * __restrict__ Q,
  int                                          normal
) {
  constexpr double gamma = 1.4;
  const double irho = 1./Q[0];
  #if Dimensions==3
  const double p = (gamma-1) * (Q[4] - 0.5*irho*(Q[1]*Q[1]+Q[2]*Q[2]+Q[3]*Q[3]));
  #else
  const double p = (gamma-1) * (Q[4] - 0.5*irho*(Q[1]*Q[1]+Q[2]*Q[2]));
  #endif

  const double u_n = Q[normal + 1] * irho;
  const double c   = std::sqrt(gamma * p * irho);

  double result = std::max( std::abs(u_n - c), std::abs(u_n + c) );
  return result;
}


// 2D
std::tuple<int, int> glob2loc(const int g, const int J)
{
    return std::make_tuple(g/J,g%J);
}

// 3D
std::tuple<int, int, int> glob2loc(const int g, const int J, const int K)
{
    int i = g/J/K;
    return std::make_tuple(i, (g-i*J*K)/J, g%K);
}

template<
    int numVPAIP,
    int unknowns,
    int aux
    >
void defaultcomputeparallel(const int haloSize, const int sourcePatchSize, const int destPatchSize, double * Qin, double * Qout)
{
  const size_t NPT=20000;
#pragma omp parallel for collapse(4)
    for (int pidx=0;pidx<NPT;pidx++)
    for (int y=0; y < numVPAIP; y++)
    for (int x=0; x < numVPAIP; x++)
    {
      for (int i=0; i<unknowns+aux; i++)
      {
        double *reconstructedPatch = Qin + sourcePatchSize*pidx;
        int sourceIndex      = (y+1)*(numVPAIP+ 3*haloSize) + x - y;
        int destinationIndex = y*numVPAIP + x;
        Qout[pidx*destPatchSize + destinationIndex*(unknowns+aux)+i] =  reconstructedPatch[sourceIndex*(unknowns+aux)+i];
      }
    }
}

template<
    int numVPAIP,
    int unknowns,
    int aux,
    int ncollapse
    >
void computeparallelgpudist(const int haloSize, const int sourcePatchSize, const int destPatchSize, double * Qin, double * Qout)
{
    const size_t NPT=20000;

#pragma omp target map(from:Qout[0:NPT*destPatchSize]) 
  {
#pragma omp teams distribute
    for (int pidx=0;pidx<NPT;pidx++)
    {
      #pragma omp parallel for collapse(ncollapse)
      for (int y=0; y < numVPAIP; y++)
      for (int x=0; x < numVPAIP; x++)
      for (int i=0; i<unknowns+aux; i++)
      {
        double *reconstructedPatch = Qin + sourcePatchSize*pidx;
        int sourceIndex      = (y+1)*(numVPAIP+ 3*haloSize) + x - y;
        int destinationIndex = y*numVPAIP + x;
        Qout[pidx*destPatchSize + destinationIndex*(unknowns+aux)+i] =  reconstructedPatch[sourceIndex*(unknowns+aux)+i];
      }
    }
  }
}

// We like this one as it is fast on cpu and gpu
template<
    size_t NPT,
    int numVPAIP,
    int unknowns,
    int aux
    >
void fcompute3(queue& Q, const int haloSize, const int sourcePatchSize, const int destPatchSize, double * Qin, double * Qout, bool skipSourceTerm, const size_t GX, const size_t GY, const size_t GZ)
{

  Q.submit([&](handler &cgh)
  {
    cgh.parallel_for(nd_range<3>{{NPT, numVPAIP, numVPAIP}, {GX, GY, GZ}}, [=](nd_item<3> item)
    {
        const size_t pidx=item.get_global_id(0);//[0];
        double *reconstructedPatch = Qin + sourcePatchSize*pidx;
        const size_t x=item.get_global_id(1);
        const size_t y=item.get_global_id(2);
        int sourceIndex      = (y+1)*(numVPAIP+ 3*haloSize) + x - y;
        int destinationIndex = y*numVPAIP + x;
        for (int i=0; i<unknowns+aux; i++)
        { 
          Qout[pidx*destPatchSize + destinationIndex*(unknowns+aux)+i] = reconstructedPatch[sourceIndex*(unknowns+aux)+i];
        }
    });
  }).wait();
}

template<
    int numVPAIP,
    int unknowns,
    int aux
    >
void strider(queue& Q, const int haloSize, const int sourcePatchSize, const int destPatchSize, double * Qin, double * Qout, bool skipSourceTerm)
{
  const size_t NPT=20000;

  Q.submit([&](handler &cgh)
  {
    cgh.parallel_for(nd_range<3>{{NPT, numVPAIP, unknowns+aux}, {1,20,1}}, [=](nd_item<3> item)
    //cgh.parallel_for(nd_range<3>{{NPT, numVPAIP, unknowns+aux}, {1, numVPAIP, unknowns+aux}}, [=](nd_item<3> item)
    {
        const size_t pidx=item.get_global_id(0);//[0];
        double *reconstructedPatch = Qin + sourcePatchSize*pidx;
        const size_t x=item.get_global_id(1);
        const size_t i=item.get_global_id(2);
        for (int y=0; y < numVPAIP; y++)
        {
          int sourceIndex      = (y+1)*(numVPAIP+ 3*haloSize) + x - y;
          int destinationIndex = y*numVPAIP + x;
          Qout[pidx*destPatchSize + destinationIndex*(unknowns+aux)+i] = reconstructedPatch[sourceIndex*(unknowns+aux)+i];
        }
    });
  }).wait();
}

template<
    size_t NPT,
    int numVPAIP,
    int unknowns,
    int aux
    >
void strider2(queue& Q, const int haloSize, const int sourcePatchSize, const int destPatchSize, double * Qin, double * Qout, bool skipSourceTerm, const size_t GX, const size_t GY, const size_t GZ)
{

  Q.submit([&](handler &cgh)
  {
    cgh.parallel_for(nd_range<3>{{NPT, (unknowns+aux)*numVPAIP, numVPAIP}, {GX, GY, GZ}}, [=](nd_item<3> item)
    {
        const size_t pidx=item.get_global_id(0);
        double *reconstructedPatch = Qin + sourcePatchSize*pidx;
        const size_t y=item.get_global_id(2);
        auto [x,i] = glob2loc(item.get_global_id(1), (unknowns+aux)*numVPAIP);
        int sourceIndex      = (y+1)*(numVPAIP+ 3*haloSize) + x - y;
        int destinationIndex = y*numVPAIP + x;
        Qout[pidx*destPatchSize + destinationIndex*(unknowns+aux)+i] = reconstructedPatch[sourceIndex*(unknowns+aux)+i];
    });
  }).wait();
}

template<
    int numVPAIP,
    int unknowns,
    int aux
    >
void strider4(queue& Q, const int haloSize, const int sourcePatchSize, const int destPatchSize, double * Qin, double * Qout, bool skipSourceTerm)
{
  const size_t NPT=20000;

  Q.submit([&](handler &cgh)
  {
    cgh.parallel_for(nd_range<3>{{NPT, numVPAIP, (unknowns+aux)*numVPAIP}, {1, 1, 100}}, [=](nd_item<3> item)
    {
        const size_t pidx=item.get_global_id(0);
        double *reconstructedPatch = Qin + sourcePatchSize*pidx;
        const size_t y=item.get_global_id(1);
        auto [x,i] = glob2loc(item.get_global_id(2), (unknowns+aux)*numVPAIP);
        int sourceIndex      = (y+1)*(numVPAIP+ 3*haloSize) + x - y;
        int destinationIndex = y*numVPAIP + x;
        Qout[pidx*destPatchSize + destinationIndex*(unknowns+aux)+i] = reconstructedPatch[sourceIndex*(unknowns+aux)+i];
    });
  }).wait();
}


//template<
    //int numVPAIP,
    //int unknowns,
    //int aux
    //>
//void strider2(queue& Q, const int haloSize, const int sourcePatchSize, const int destPatchSize, double * Qin, double * Qout, bool skipSourceTerm)
//{
  //const size_t NPT=20000;

  //Q.submit([&](handler &cgh)
  //{
    //cgh.parallel_for(nd_range<3>{{NPT, numVPAIP, (unknowns+aux)*numVPAIP}, {1, 20, 10}}, [=](nd_item<3> item)
    //{
        //const size_t pidx=item.get_global_id(0);//[0];
        //double *reconstructedPatch = Qin + sourcePatchSize*pidx;
        //const size_t x=item.get_global_id(1);
        //auto [i,y] = glob2loc(item.get_global_id(2), numVPAIP);
        //int sourceIndex      = (y+1)*(numVPAIP+ 3*haloSize) + x - y;
        //int destinationIndex = y*numVPAIP + x;
        //Qout[pidx*destPatchSize + destinationIndex*(unknowns+aux)+i] = reconstructedPatch[sourceIndex*(unknowns+aux)+i];
    //});
  //}).wait();
//}

template<
    int numVPAIP,
    int unknowns,
    int aux
    >
void strider3(queue& Q, const int haloSize, const int sourcePatchSize, const int destPatchSize, double * Qin, double * Qout, bool skipSourceTerm)
{
  const size_t NPT=20000;

  Q.submit([&](handler &cgh)
  {
    cgh.parallel_for(nd_range<2>{{NPT, numVPAIP*(unknowns+aux)*numVPAIP}, {1,100}}, [=](nd_item<2> item)
    {
        const size_t pidx=item.get_global_id(0);//[0];
        double *reconstructedPatch = Qin + sourcePatchSize*pidx;
        auto [i,y,x] = glob2loc(item.get_global_id(1),numVPAIP,numVPAIP);
        int sourceIndex      = (y+1)*(numVPAIP+ 3*haloSize) + x - y;
        int destinationIndex = y*numVPAIP + x;
        Qout[pidx*destPatchSize + destinationIndex*(unknowns+aux)+i] = reconstructedPatch[sourceIndex*(unknowns+aux)+i];
    });
  }).wait();
}





template<
    int numVPAIP,
    int unknowns,
    int aux
    >
void hcompute(queue& Q, const int haloSize, const int sourcePatchSize, const int destPatchSize, double * Qin, double * Qout, bool wtf)
{
  const size_t NPT=20000;

  Q.submit([&](handler &cgh)
  {
    //cgh.parallel_for_work_group(range<3>{NPT, numVPAIP, numVPAIP}, {1,1,1}, [=](group<3> grp)
    cgh.parallel_for_work_group(range<3>{NPT, numVPAIP, numVPAIP},  [=](group<3> grp)
    {
      const size_t pidx=grp[0];
      double *reconstructedPatch = Qin + sourcePatchSize*pidx;
      grp.parallel_for_work_item([&](auto idx)
      {
          const size_t y=idx.get_global_id(1);
          const size_t x=idx.get_global_id(2);
          int sourceIndex      = (y+1)*(numVPAIP+ 3*haloSize) + x - y;
          int destinationIndex = y*numVPAIP + x;
          for (int i=0; i<unknowns+aux; i++)
          { 
            Qout[pidx*destPatchSize + destinationIndex*(unknowns+aux)+i] =  reconstructedPatch[sourceIndex*(unknowns+aux)+i];
          }
      });
    });
  }).wait();
}



//int main(int argc, char* argv[])
//{
    //std::cout << "  Using SYCL device: " << Q.get_device().get_info<sycl::info::device::name>() << std::endl;

    //Q.submit([&](handler &cgh)
    //{ 
       //cgh.single_task([=]() {});
    //});

    //// https://stackoverflow.com/questions/2704521/generate-random-double-numbers-in-c
    //std::uniform_real_distribution<double> unif(0, 10);
    //std::default_random_engine re(time(NULL));

    //const size_t NPT=20000;
    //const int numVPAIP = 20;
    //const int unknowns=5;
    //const int srcPS = (numVPAIP+2)*(numVPAIP+2)*unknowns;
    //const int destPS = numVPAIP*numVPAIP*unknowns;
    
    //auto Xin  = malloc_shared<double>(srcPS*NPT, Q);
    //for (int i=0;i<srcPS*NPT;i++) Xin[i] = unif(re);
    //auto Xout = malloc_shared<double>(destPS*NPT, Q);
    ////defaultcomputeparallel<numVPAIP,unknowns,0>(1, srcPS, destPS ,Xin, Xout);
    ////std::cout << "sum should be: " << std::accumulate(Xout, Xout + destPS*NPT, 0)  << "\n";

    ////for (int i=0;i<destPS*NPT;i++) Xout[i] = 0;
    ////computeparallelgpudist<numVPAIP,unknowns,0,2>(1, srcPS, destPS ,Xin, Xout);
    ////std::cout << "sum is gpu2: " << std::accumulate(Xout, Xout + destPS*NPT, 0)  << "\n";
    
    ////for (int i=0;i<destPS*NPT;i++) Xout[i] = 0;
    ////computeparallelgpudist<numVPAIP,unknowns,0,3>(1, srcPS, destPS ,Xin, Xout);
    ////std::cout << "sum is gpu3: " << std::accumulate(Xout, Xout + destPS*NPT, 0)  << "\n";
    
    //for (int i=0;i<destPS*NPT;i++) Xout[i] = 0;
    //fcompute3<numVPAIP,unknowns,0>(Q, 1, srcPS, destPS ,Xin, Xout, true);
    //std::cout << "sum: " << std::accumulate(Xout, Xout + destPS*NPT, 0) << "\n";

    //for (int i=0;i<destPS*NPT;i++) Xout[i] = 0;
    //hcompute<numVPAIP,unknowns,0>(Q, 1, srcPS, destPS ,Xin, Xout, true);
    //std::cout << "sum: " << std::accumulate(Xout, Xout + destPS*NPT, 0) << "\n";

    //for (int i=0;i<destPS*NPT;i++) Xout[i] = 0;
    //strider<numVPAIP,unknowns,0>(Q, 1, srcPS, destPS ,Xin, Xout, true);
    //std::cout << "sum: " << std::accumulate(Xout, Xout + destPS*NPT, 0) << "\n";

    //for (int i=0;i<destPS*NPT;i++) Xout[i] = 0;
    //strider2<numVPAIP,unknowns,0>(Q, 1, srcPS, destPS ,Xin, Xout, true);
    //std::cout << "sum: " << std::accumulate(Xout, Xout + destPS*NPT, 0) << "\n";
    
    //for (int i=0;i<destPS*NPT;i++) Xout[i] = 0;
    //strider3<numVPAIP,unknowns,0>(Q, 1, srcPS, destPS ,Xin, Xout, true);
    //std::cout << "sum: " << std::accumulate(Xout, Xout + destPS*NPT, 0) << "\n";
    
    //for (int i=0;i<destPS*NPT;i++) Xout[i] = 0;
    //strider4<numVPAIP,unknowns,0>(Q, 1, srcPS, destPS ,Xin, Xout, true);
    //std::cout << "sum: " << std::accumulate(Xout, Xout + destPS*NPT, 0) << "\n";


    //auto start = std::chrono::steady_clock::now();   
    //for (int i=0;i<std::atoi(argv[1]);i++) 
      //defaultcomputeparallel<numVPAIP,unknowns,0>(1, srcPS, destPS ,Xin, Xout);
    //auto end = std::chrono::steady_clock::now();
    //std::cout << "OMP PARALLEL FOR on CPU (defaultcomputeparallel):   " <<std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count()   << " ms" << std::endl;
    
    //start = std::chrono::steady_clock::now();   
    //for (int i=0;i<std::atoi(argv[1]);i++) 
      //fcompute3<numVPAIP,unknowns,0>(Q, 1, srcPS, destPS ,Xin, Xout, true);
    //end = std::chrono::steady_clock::now();
    //std::cout << "NDRANGE<3> with loop (fcompute3):                   " << std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count()   << " ms" << std::endl;
    
    //start = std::chrono::steady_clock::now();   
    //for (int i=0;i<std::atoi(argv[1]);i++) 
      //strider<numVPAIP,unknowns,0>(Q, 1, srcPS, destPS ,Xin, Xout, true);
    //end = std::chrono::steady_clock::now();
    //std::cout << "NDRANGE<3> with loop but swap dimensions (strider): " << std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count()   << " ms" << std::endl;
    
    //start = std::chrono::steady_clock::now();   
    //for (int i=0;i<std::atoi(argv[1]);i++) 
      //strider2<numVPAIP,unknowns,0>(Q, 1, srcPS, destPS ,Xin, Xout, true);
    //end = std::chrono::steady_clock::now();
    //std::cout << "NDRANGE<3> without loop (strider2) :                " << std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count()   << " ms" << std::endl;
    
    //start = std::chrono::steady_clock::now();   
    //for (int i=0;i<std::atoi(argv[1]);i++) 
      //strider4<numVPAIP,unknowns,0>(Q, 1, srcPS, destPS ,Xin, Xout, true);
    //end = std::chrono::steady_clock::now();
    //std::cout << "NDRANGE<3> without loop (strider4) :                " << std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count()   << " ms" << std::endl;
    
    //start = std::chrono::steady_clock::now();   
    //for (int i=0;i<std::atoi(argv[1]);i++) 
      //strider3<numVPAIP,unknowns,0>(Q, 1, srcPS, destPS ,Xin, Xout, true);
    //end = std::chrono::steady_clock::now();
    //std::cout << "NDANGE<2> without loop (strider3)                   " << std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count()   << " ms" << std::endl;
    
    //start = std::chrono::steady_clock::now();   
    //for (int i=0;i<std::atoi(argv[1]);i++) 
      //hcompute<numVPAIP,unknowns,0>(Q, 1, srcPS, destPS ,Xin, Xout, true);
    //end = std::chrono::steady_clock::now();
    //std::cout << "Hierarchical parallelism (hcompute):                " << std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count()   << " ms" << std::endl;
    
    ////start = std::chrono::steady_clock::now();   
    ////for (int i=0;i<std::atoi(argv[1]);i++) 
      ////computeparallelgpudist<numVPAIP,unknowns,0,2>(1, srcPS, destPS ,Xin, Xout);
    ////end = std::chrono::steady_clock::now();
    ////std::cout << "OpenMP  collapse(2)  (computeparallelgpudist) :     " << std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count()   << " ms" << std::endl;
    
    ////start = std::chrono::steady_clock::now();   
    ////for (int i=0;i<std::atoi(argv[1]);i++) 
      ////computeparallelgpudist<numVPAIP,unknowns,0,3>(1, srcPS, destPS ,Xin, Xout);
    ////end = std::chrono::steady_clock::now();
    ////std::cout << "OpenMP  collapse(3)  (computeparallelgpudist) :     " << std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count()   << " ms" << std::endl;

    //free(Xin, Q);
    //free(Xout, Q);

    //return 0;
//}

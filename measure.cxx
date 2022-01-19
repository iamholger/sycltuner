#include <chrono>
#include <cmath>
#include <vector>
#include <iostream>
#include <fstream>

#include "comparisonlib.h"

struct Record
{
    double x; double xsq; int n; const size_t GX; const size_t GY; const size_t GZ;

    Record(const size_t gx, const size_t gy, const size_t gz) : x(0), xsq(0), n(0), GX(gx), GY(gy), GZ(gz) {};

    constexpr double mean() const {return x/n;}

    // Weighted variance defined as
    // sig2 = ( sum(wx**2) * sum(w) - sum(wx)**2 ) / ( sum(w)**2 - sum(w**2) )
    // see http://en.wikipedia.org/wiki/Weighted_mean --- all weights == 1
    constexpr double stddev() const {
        return sqrt( (n*xsq - x*x)/double(n*n -n)) / n;
    }

    // TODO can we do this with variadic templates?
    void measure(queue& Q, const int haloSize, const int sourcePatchSize, const int destPatchSize, double * Qin, double * Qout, bool skipSourceTerm,
        void (*fn)(queue& Q, const int, const int, const int, double *, double *, bool, const size_t, const size_t, const size_t))
    {
      // Call once to get rid of jit cost in measurement  
        fn(Q, haloSize, sourcePatchSize, destPatchSize, Qin, Qout, skipSourceTerm, GX, GY, GZ);
        auto start = std::chrono::steady_clock::now();   
        fn(Q, haloSize, sourcePatchSize, destPatchSize, Qin, Qout, skipSourceTerm, GX, GY, GZ);
        auto end = std::chrono::steady_clock::now();

        double measurement = std::chrono::duration_cast<std::chrono::microseconds>(end - start).count();

        x   += measurement;
        xsq += measurement*measurement;
        n++;
    }

    std::string const toCSV()
    {
      std::ostringstream s;
      s << GX << "," << GY << "," << GZ << "," << this->mean() << "," << this->stddev();
      return s.str();
    }
};

struct Tuner
{
    std::vector<std::vector<int> > sizes_;
    std::vector<Record > records_;
    int winner_;
    double besttime_;
    const int maxwgsize_;
    const int lx_; const int ly_; const int lz_;

    std::vector<int> factors(const int x)
    {
        std::vector<int> fcts;
        for (int i=1;i<x+1;i++) if (x%i==0) fcts.push_back(i);
        return fcts;
    }

    Tuner(const int wgmax, const int lx, const int ly, const int lz) : winner_(-1), besttime_(1e99), maxwgsize_(wgmax), lx_(lx), ly_(ly), lz_(lz)
    {
        auto FX = factors(lx);
        auto FY = factors(ly);
        auto FZ = factors(lz);

        for (auto fx : FX)
        for (auto fy : FY)
        for (auto fz : FZ) if (fx*fy*fz <=maxwgsize_) sizes_.push_back({fx,fy,fz});
    };


    friend std::ostream& operator<<(std::ostream& os, const Tuner& tr)
    {
      auto win = tr.records_[tr.winner_];
      return os << "Fastest time: " << tr.besttime_ << " +/- " << win.stddev() << " microseconds with WG sizes {" << win.GX << ","  << win.GY << "," << win.GZ << "}\n";
    }

    void tune(queue& Q, const int haloSize, const int sourcePatchSize, const int destPatchSize, double * Qin, double * Qout, bool skipSourceTerm,
        void (*fn)(queue& Q, const int, const int, const int, double *, double *, bool, const size_t, const size_t, const size_t),
        double tol=0.1, int mineval=6)
    {
        double relerr;
        std::cerr << "Got " << sizes_.size() << " variations\n";

        for (int i=0;i<sizes_.size();i++)
        {
            relerr=1;
            auto s = sizes_[i];
            Record r(s[0], s[1], s[2]);
            while (relerr>tol)
            {
               r.measure(Q, haloSize, sourcePatchSize, destPatchSize, Qin, Qout, skipSourceTerm, fn);
               if (r.n >= mineval) relerr = r.stddev() / r.mean();
            }
            records_.push_back(r);

            if (r.mean() < besttime_)
            {
                besttime_ = r.mean();
                winner_ = i;
            }
        }
    }

    void dump(const std::string name, std::ofstream& ofs)
    {
      for (auto r : records_)
      {
        ofs << r.toCSV() << "," << name << "\n";
      }
    }
};


int main(int argc, char* argv[])
{
    std::cerr << "  Using SYCL device: " << Q.get_device().get_info<sycl::info::device::name>() << std::endl;

    Q.submit([&](handler &cgh)
    { 
       cgh.single_task([=]() {});
    });

    // https://stackoverflow.com/questions/2704521/generate-random-double-numbers-in-c
    std::uniform_real_distribution<double> unif(0, 10);
    std::default_random_engine re(time(NULL));

    const size_t NPT=20000;
    const int numVPAIP = 20;
    const int unknowns=5;
    const int aux=0;
    const int srcPS = (numVPAIP+2)*(numVPAIP+2)*unknowns;
    const int destPS = numVPAIP*numVPAIP*unknowns;
    const int max_local_wg_size(std::atoi(argv[1]));
    
    auto Xin  = malloc_shared<double>(srcPS*NPT, Q);
    for (int i=0;i<srcPS*NPT;i++) Xin[i] = unif(re);
    auto Xout = malloc_shared<double>(destPS*NPT, Q);

    std::ofstream ofs("data.csv", std::ofstream::out);
    ofs << "GX,GY,GZ,mu,sigma,name\n";


    Tuner t_fcompute(max_local_wg_size, NPT,numVPAIP,numVPAIP);
    t_fcompute.tune(Q, 1, srcPS, destPS ,Xin, Xout, true, &fcompute3<NPT,numVPAIP,unknowns,aux>);
    std::cerr << t_fcompute << "\n";
    t_fcompute.dump("fcompute3", ofs);
    
    Tuner t_strider2(max_local_wg_size, NPT, (unknowns+aux)*numVPAIP, numVPAIP);
    t_strider2.tune(Q, 1, srcPS, destPS ,Xin, Xout, true, &strider2<NPT,numVPAIP,unknowns,aux>);
    std::cerr << t_strider2 << "\n";
    t_fcompute.dump("strider2", ofs);

    ofs.close();

    return 0;
}

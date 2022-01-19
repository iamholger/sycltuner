# sycltuner

The code runs through all possible kernel shapes for a fixed
problem size and architecture. The measurement of the excution time
is repeaded until sufficiently precise (10% relative error by default).

dpcpp measure.cxx -I. 

./a.out 1024

The command line argument is the maximum local work group size, i.e. the product of GX,GY,GZ if
the WG sizes are specified as {GX,GY,GZ}.

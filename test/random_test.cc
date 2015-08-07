// Copyright (c) 2015 by Contributors
// random test code

#include <iostream>
#include <random>
#include "../src/utils/random.h"

//using namspeace mxnet;
using namespace std;

int main(int argc, char *argv[]) {
  mxnet::utils::RandomSampler rnd;
  rnd.Seed(atoi(argv[1]));
  for (int n = 0; n < 10; n++) {
    std::cout << rnd.NextDouble() << " ";
  }
  return 0;
}

/*
This is a relatively pedestrian translation in C++ of the Python implementation
of the Inductive VennABERS Predictor (IVAP).

It was developed with the aid of Gemini, as an exploration of the use of LLM code assistants.
It relies on pybind11 (pip install pybind11) for C++ bindings.
Also, the fmt library (apt install libfmt-dev) is required only when generating the STANDALONE version.

/usr/bin/g++ -std=c++20 -fdiagnostics-color=always -g $(python3 -m pybind11 --includes) VennABERSlib.cpp -shared -fPIC  -o VennABERSlib.so

Paolo Toccaceli, 2024-02

*/

#include <tuple>
#include <vector>
#include <map>
#include <algorithm>
#include <cmath>
#include <numeric>
#include <ranges>


using T = double;

std::tuple<std::vector<T>, std::vector<int>, std::vector<int>> 
unique_with_counts_and_inverse(const std::vector<T>& x) {

  // Find unique elements and their counts
  std::map<T, int> counts;

  for (const T& value : x) {
    if (auto search=counts.find(value); search != counts.end()) {  
      counts[value]++;
    } else {
      counts[value] = 1;
    }
  }

  std::vector<T> unique;
  std::vector<int> counts_vec;
  unique.reserve(counts.size());
  counts_vec.reserve(counts.size());

  std::vector<int> inverse(x.size());

  for (const auto& [val,count] : counts) {
    unique.push_back(val);
    counts_vec.push_back(count);
  }

  // Find inverse indices
  for (int i = 0; i < x.size(); ++i) {
    inverse[i] = std::lower_bound(unique.begin(), unique.end(), x[i]) - unique.begin();
  }

  return std::make_tuple(std::move(unique), std::move(counts_vec), std::move(inverse));
}

// Assuming coordinates are defined as pairs of doubles
using Point = std::pair<double, double>;

Point top(const std::vector<Point>& S) {
    return S[S.size()-1];
}

Point nextToTop(const std::vector<Point>& S) {
    return S[S.size()-2];
}


bool nonLeftTurn(const Point& a, const Point& b, const Point& c) {
  double d1x = b.first - a.first, d1y = b.second - a.second;
  double d2x = c.first - b.first, d2y = c.second - b.second;
  return (d1x * d2y - d1y * d2x) <= 0.0;
}

bool nonRightTurn(const Point& a, const Point& b, const Point& c) {
  return nonLeftTurn(c, b, a); // Equivalent due to reversed order
}

double slope(const Point& a, const Point& b) {
  if (a.first == b.first) {
    return std::numeric_limits<double>::infinity(); // Handle vertical lines
  }
  return (b.second - a.second) / (b.first - a.first);
}

bool notBelow(const Point& t, const Point& p1, const Point& p2) {
  double m = slope(p1, p2);
  double b = p2.second - m * p2.first;
  return t.second >= m * t.first + b;
}


// We shift all the indices for P and make it a vector
std::vector<Point> algorithm1(const std::vector<Point>& P, int kPrime) {
  std::vector<Point> S;

  // Initialize stack with first two points
  S.push_back(P[-1+1]);
  S.push_back(P[0+1]);

  // Iterate from 1 to kPrime (inclusive)
  for (int i = 1; i <= kPrime; ++i) {
    while (S.size() > 1 && nonLeftTurn(nextToTop(S), top(S), P[i+1])) {
      S.pop_back();
    }
    S.push_back(P[i+1]);
  }

  return S;
}

std::vector<double> algorithm2(std::vector<Point>& P, const std::vector<Point>& S, int kPrime) {
  std::vector<Point> Sprime(S.rbegin(), S.rend()); // Reverse stack
  std::vector<double> F1(kPrime + 1);

  F1[0] = 0.0;
  // Iterate from 1 to kPrime (inclusive)
  for (int i = 1+1; i <= kPrime+1; ++i) {    // i shifted by 1
    F1[i - 1] = slope(top(Sprime), nextToTop(Sprime));
    P[i - 1].first  = P[i - 2].first  + P[i].first  - P[i - 1].first;
    P[i - 1].second = P[i - 2].second + P[i].second - P[i - 1].second;

    if (notBelow(P[i - 1], top(Sprime), nextToTop(Sprime))) {
      continue;
    }

    Sprime.pop_back();
    while (Sprime.size() > 1 && nonLeftTurn(P[i - 1], top(Sprime), nextToTop(Sprime))) {
      Sprime.pop_back();
    }
    Sprime.push_back(P[i - 1]);
  }
  return F1;
}

std::vector<Point> algorithm3(const std::vector<Point>& P, int kPrime) {
  std::vector<Point> S;
  S.push_back(P[kPrime + 1]); // Push P[kPrime+1] onto the stack (correct starting point)
  S.push_back(P[kPrime + 0]);

  // Iterate from k'-1 to 0 (inclusive) in descending order
  for (int i = kPrime-1; i >= 0; --i) {
    while (S.size() > 1 && nonRightTurn(nextToTop(S), top(S), P[i])) {
      S.pop_back();
    }
    S.push_back(P[i]);
  }
  return S;
}


std::vector<double> algorithm4(std::vector<Point>& P, const std::vector<Point>& S, int kPrime) {
  std::vector<Point> Sprime(S.rbegin(), S.rend()); // Reverse stack
  std::vector<double> F0(kPrime + 1);

  // Iterate from k' to 1 (inclusive) in descending order
  for (int i = kPrime; i >= 1; --i) {
    F0[i] = slope(top(Sprime), nextToTop(Sprime));
    P[i].first = P[i - 1].first + P[i+1].first - P[i].first;
    P[i].second = P[i - 1].second + P[i+1].second - P[i].second;

    if (notBelow(P[i], top(Sprime), nextToTop(Sprime))) {
      continue;
    }

    Sprime.pop_back();
    while (Sprime.size() > 1 && nonRightTurn(P[i], top(Sprime), nextToTop(Sprime))) {
      Sprime.pop_back();
    }
    Sprime.push_back(P[i]);
  }
  return F0;
}


bool compare_on_xs(const std::pair<double, double>& a, const std::pair<double, double>& b) {
  return a.first < b.first;
}

void sort_both_on_xs(std::vector<double>& xs, std::vector<double>& ys) {
  // Combine xs and ys into a vector of pairs
  std::vector<std::pair<double, double>> pairs(xs.size());
  for (size_t i = 0; i < xs.size(); ++i) {
    pairs[i] = {xs[i], ys[i]};
  }

  // Sort the pairs based on xs
  std::sort(pairs.begin(), pairs.end(), compare_on_xs);

  // Extract back to separate vectors (optional)
  // Similar to approach 1, this step is optional.
  for (size_t i = 0; i < xs.size(); ++i) {
    xs[i] = pairs[i].first;
    ys[i] = pairs[i].second;
  }
}


std::tuple<std::vector<double>, std::vector<double>, std::vector<double>> 
prepareData( std::vector<double> xs, std::vector<double> ys) {
  // Sort and extract x and y coordinates
  // std::ranges::sort(std::views::zip(xs, ys));
  sort_both_on_xs(xs, ys);

  // Find unique points and calculate weights and cumulative sums
  std::vector<double> ptsUnique;
  std::vector<int> ptsInverse;
  std::vector<int> w;

  std::tie(ptsUnique,w,ptsInverse) = unique_with_counts_and_inverse(xs);

  std::vector<double> a(ptsUnique.size());
  for (size_t i = 0; i < ptsInverse.size(); ++i) {
    a[ptsInverse[i]] += ys[i];
  }

  std::vector<double> yCsd(a.size());
  std::partial_sum(a.begin(), a.end(), yCsd.begin()); // Cumulative sum of yCsd

  std::vector<double> xPrime(ptsUnique.size());
  std::partial_sum(w.begin(), w.end(), xPrime.begin()); // Cumulative sum of w

  return std::make_tuple(yCsd, xPrime, ptsUnique);
}


std::pair< std::vector<double>, std::vector<double> > 
getFVal(const std::vector<double>& F0, const std::vector<double>& F1,
        const std::vector<double>& ptsUnique, const std::vector<double>& testObjects) {
  std::vector<size_t> pos0(testObjects.size());
  std::vector<size_t> pos1(testObjects.size());

  // Use std::lower_bound for left-side search
  std::transform(testObjects.begin(), testObjects.end(), pos0.begin(),
                 [&ptsUnique](double testObject) {
                   return std::lower_bound(ptsUnique.begin(), ptsUnique.end(), testObject) - ptsUnique.begin();
                 });

  // Use std::upper_bound for right-side search, adjusting index for excluding the last element
  std::transform(testObjects.begin(), testObjects.end(), pos1.begin(),
                 [&ptsUnique](double testObject) {
                   return std::upper_bound(ptsUnique.begin(), ptsUnique.end() - 1, testObject) - ptsUnique.begin()+1;
                 });

  // Extract and return values from F0 and F1 based on positions
  std::vector<double> F0_vals(pos0.size());
  std::vector<double> F1_vals(pos0.size());
  for (size_t i = 0; i < pos0.size(); ++i) {
    F0_vals[i] = F0[pos0[i]];
    F1_vals[i] = F1[pos1[i]];
  }

  return std::make_pair(F0_vals, F1_vals);
}


std::pair<std::vector<double>, std::vector<double>> 
computeF(const std::vector<double>& xPrime, const std::vector<double>& yCsd) {

  auto kPrime = xPrime.size();
  
  // Create P vector with Point values
  std::vector<Point> P(kPrime+2);

  P[-1+1] = Point(-1.0, -1.0); // Dummy point for index -1
  P[0+1] = Point(0.0, 0.0);
  for (int i = 0; i < kPrime; ++i) {
    P[i + 2] = Point(xPrime[i], yCsd[i]);
  }

  // Compute F1 using algorithms 1 and 2
  std::vector<Point> S = algorithm1(P, kPrime);
  std::vector<double> F1 = algorithm2(P, S, kPrime);

  // Update P with dummy point and modified P[kPrime + 1]
  P[0] = Point(0.0, 0.0);
  for (int i = 0; i < kPrime; ++i) {
    P[i + 1] = Point(xPrime[i], yCsd[i]);
  }
  // P[kPrime + 1] = P[kPrime] + Point(1.0, 0.0); // Adjust based on paper's correction
  P[kPrime + 1].first = P[kPrime].first + 1.0;
  P[kPrime + 1].second = P[kPrime].second;

  // Compute F0 using algorithms 3 and 4
  S = algorithm3(P, kPrime);
  std::vector<double> F0 = algorithm4(P, S, kPrime);

  return std::make_pair(F0, F1);
}

std::pair<std::vector<double>, std::vector<double>> 
ScoresToMultiProbs(const std::vector<double>& xs, const std::vector<double>& ys, const std::vector<double>& testObjects) {

  std::vector<double> yCsd, xPrime, ptsUnique;
  std::tie(yCsd,xPrime,ptsUnique) = prepareData(xs, ys);

  std::vector<double> F0, F1;
  std::tie(F0,F1) = computeF(xPrime,yCsd);
    
  std::vector<double> p0, p1;
  std::tie(p0, p1) = getFVal(F0,F1,ptsUnique,testObjects);
  return std::make_pair(p0, p1);
}

#ifdef _STANDALONE_
#include <fmt/core.h>
#include <fmt/ranges.h>
int main(int argc, char *argv[])
{
  std::vector<double> xs{2.0, 3.0, 4.0, 5.0, 7.0, 8.0};
  std::vector<double> ys{0.0, 1.0, 0.0, 0.0, 1.0, 1.0};

  std::vector<double> testObjs{1.0, 2.5, 7.2};

  std::vector<double> p0,p1;
  std::tie(p0, p1) = ScoresToMultiProbs(xs, ys, testObjs);


  fmt::print("p0: {}\n", fmt::join(p0, ", "));
  fmt::print("p1: {}\n", fmt::join(p1, ", "));
}
#endif

#define _PYBIND11_VA

#ifdef _PYBIND11_VA
// /usr/bin/g++-11 -std=c++17 -fdiagnostics-color=always -g VennABERS.cpp -shared -fPIC -L/usr/local/lib -I/home/paolo/miniconda3/envs/VennABERS/include -I/home/paolo/miniconda3/envs/VennABERS/include/python3.11 -I/home/paolo/miniconda3/envs/VennABERS/lib/python3.11/site-packages/pybind11/include -lfmt -o /home/paolo/workspace/VennABERS/VennABERS.so

#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <pybind11/stl.h>

namespace py = pybind11;

#define PASS_LISTS

py::tuple
// Access the NumPy arrays as C++ arrays
#ifdef PASS_NUMPY_ARRAYS
train( py::array_t<double>& xs_,  py::array_t<double>& ys_) {
  // Can I just do this? Or do I have to do the following?
  // std::vector<double> xs(xs_.data(), xs_.data()+xs_.size());
  // std::vector<double> ys(ys_.data(), ys_.data()+ys_.size());

  double *xs_buf = static_cast<double *>(xs_.request().ptr);
  double *ys_buf = static_cast<double *>(ys_.request().ptr);

  std::vector<double> xs(xs_buf, xs_buf + xs_.request().shape[0]);
  std::vector<double> ys(ys_buf, ys_buf + ys_.request().shape[0]);
#else
#ifdef PASS_LISTS
train(std::vector<double> xs, std::vector<double> ys) {
#endif
#endif
  std::vector<double> yCsd, xPrime, ptsUnique;
  std::tie(yCsd,xPrime,ptsUnique) = prepareData(xs, ys);

  std::vector<double> F0, F1;
  std::tie(F0,F1) = computeF(xPrime,yCsd);

  // Consider passing arrays as explained in
  // https://stackoverflow.com/questions/44659924/returning-numpy-arrays-via-pybind11
  // or refactoring into a class
  // https://developer.lsst.io/v/DM-9089/coding/python_wrappers_for_cpp_with_pybind11.html
  return py::make_tuple(F0,F1,ptsUnique); 
}

py::tuple
predict(std::vector<double> F0, std::vector<double> F1, std::vector<double> ptsUnique, std::vector<double> testObjects) {
  std::vector<double> p0, p1;
  std::tie(p0, p1) = getFVal(F0,F1,ptsUnique,testObjects);
  return py::make_tuple(p0, p1);
}

PYBIND11_MODULE(VennABERSlib, m) {
    m.def("train", &train, "Train IVAP model");
    m.def("predict", &predict, "Predict using IVAP model");
}

#endif // _PYBIND11_VA
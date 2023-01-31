#include <RcppArmadillo.h>
#include <complex>
using namespace std;
using namespace arma;

// [[Rcpp::depends(RcppArmadillo)]]
// [[Rcpp::export]]
arma::vec Psi_func(arma::vec x, double a, double b, double gamma) {
  arma::vec logf = log(arma::normcdf(a/sqrt(x) + b*arma::sqrt(x)));
  arma::vec logd = (gamma - 1.0)*log(x) - x - lgamma(gamma);
  return exp(logf + logd);
}

// [[Rcpp::export]]
arma::vec Psi_func_small_gamma(arma::vec x, double a, double b, double gamma) {
  arma::vec logf = log(arma::normcdf(a/sqrt(x) + b*arma::sqrt(x)));
  arma::vec logd = (gamma - 1.0)*log(x) - x - lgamma(gamma);
  return 10.0*(exp(logf + logd) - exp((gamma-1.0)*log(x) - lgamma(gamma)));
}

// [[Rcpp::export]]
double Psi_small_gamma(double a, double b, double gamma) {
  double lower_int = 0.0000000000001;
  arma::vec x =  linspace(lower_int, 5, 100000);
  arma::vec y = Psi_func_small_gamma(x, a, b, gamma);
  mat integral = arma::trapz(x, y);
  double res = integral(0, 0)/10.0 + exp(gamma*log(5) - log(gamma) - lgamma(gamma));

  return res;
}

// [[Rcpp::export]]
double Psi_big_gamma(double a, double b, double gamma) {
  double lower_int = R::qgamma(0.0001, gamma, 1, true, false);
  double upper_int = R::qgamma(0.9999, gamma, 1, true, false);
  arma::vec x =  linspace(lower_int, upper_int, 100000);
  arma::vec y = Psi_func(x, a, b, gamma);
  mat integral = arma::trapz(x, y);
  double res = integral(0, 0);
  return res;
}

// [[Rcpp::export]]
double Psi(double a, double b, double gamma) {
  double res;
  if(gamma < 0.6 & a > 0) {
    res = Psi_small_gamma(a, b, gamma);
  }else{
    res = Psi_big_gamma(a, b, gamma);
  }

  return res;
}

// [[Rcpp::export]]
double MCPsi(double a, double b, double gamma) {
  vec x = Rcpp::rgamma(100000, gamma, 1);
  vec y = arma::normcdf(a/sqrt(x) + b*arma::sqrt(x));
  double res = mean(y);
  return res;
}

// [[Rcpp::export]]
double VGprice(double s0,
               double K,
               double t,
               double r,
               double sigma,
               double nu) {
  double theta = 0;
  double d;
  double gamma;
  double alpha;
  double s;
  double c1;
  double c2;
  double C;
  double sigma2 = pow(sigma, 2);
  double price;

  C = -theta/sigma2;
  s = sigma/sqrt(1.0 + pow(theta/sigma, 2)*(nu/2.0));

  alpha = C*s;
  c1 = nu*pow(alpha + s, 2)/2.0;
  c2 = nu*pow(alpha, 2)/2.0;
  d = (1.0/s)*(log(s0/K) + r*t + (t/nu)*log((1.0-c1)/(1.0-c2)));

  double Psi1 = Psi(d*sqrt((1.0-c1)/nu), (alpha + s)*sqrt(nu/(1.0-c1)), t/nu);
  double Psi2 = Psi(d*sqrt((1.0-c2)/nu), (alpha * s)*sqrt(nu/(1.0-c2)), t/nu);

  price = s0*Psi1 - K*exp(-r*t)*Psi2;
  return price;
}


// [[Rcpp::export]]
vec VGprice_vec(double s0,
               double K,
               double t,
               double r,
               vec sigma,
               vec nu) {
  int n = arma::size(sigma)(0);
  vec price(n);

  for(int i = 0; i < n; i++) {
    price[i] = VGprice(s0, K, t, r, sigma[i], nu[i]);
  }

  return price;
}






// [[Rcpp::export]]
double MCVGprice(double s0,
                 double K,
                 double t,
                 double r,
                 double sigma,
                 double nu) {
  double theta = 0;
  double d;
  double gamma;
  double alpha;
  double s;
  double c1;
  double c2;
  double C;
  double sigma2 = pow(sigma, 2);
  double price;

  C = -theta/sigma2;
  s = sigma/sqrt(1.0 + pow(theta/sigma, 2)*(nu/2.0));

  alpha = C*s;
  c1 = nu*pow(alpha + s, 2)/2.0;
  c2 = nu*pow(alpha, 2)/2.0;
  d = (1.0/s)*(log(s0/K) + r*t + (t/nu)*log((1.0-c1)/(1.0-c2)));

  double Psi1 = MCPsi(d*sqrt((1.0-c1)/nu), (alpha + s)*sqrt(nu/(1.0-c1)), t/nu);
  double Psi2 = MCPsi(d*sqrt((1.0-c2)/nu), (alpha * s)*sqrt(nu/(1.0-c2)), t/nu);

  price = s0*Psi1 - K*exp(-r*t)*Psi2;
  return price;
}


// [[Rcpp::export]]
double BSprice(double s0,
               double K,
               double t,
               double r,
               double sigma) {

  double d1;
  double d2;
  double price;

  d1 = (log(s0/K) + (r + (pow(sigma, 2))/2)*t)/(sigma*sqrt(t));
  d2 = d1 - sigma*sqrt(t);
  price = s0*arma::normcdf(d1) - K*exp(-r*t)*arma::normcdf(d2);
  return price;
}


// [[Rcpp::export]]
vec BSprice_vec(double s0,
                double K,
                double t,
                double r,
                vec sigma) {
  int n = arma::size(sigma)(0);
  vec price(n);

  for(int i = 0; i < n; i++) {
    price[i] = BSprice(s0, K, t, r, sigma[i]);
  }

  return price;
}

//[[Rcpp::export]]
double VGmse(vec params, double s0,
             vec market_price,
             vec d,
             vec K,
             vec t,
             vec r) {
  int n = arma::size(market_price)(0);
  vec pred_price(n);
  double mse;

  double sigma = params(0);
  double nu = params(1);

  for(int i = 0; i < n; i++) {
    pred_price[i] = VGprice(s0-d[i], K[i], t[i], r[i], sigma, nu);
  }

  mse = mean(pow(pred_price - market_price, 2));
  return mse;
}

// [[Rcpp::export]]
double BSmse(double sigma, double s0,
             vec market_price,
             vec d,
             vec K,
             vec t,
             vec r) {
  int n = arma::size(market_price)(0);
  vec pred_price(n);
  double mse;

  for(int i = 0; i < n; i++) {
    pred_price[i] = BSprice(s0-d[i], K[i], t[i], r[i], sigma);
  }

  mse = mean(pow(pred_price - market_price, 2));
  return mse;
}

// [[Rcpp::export]]
complex<double> fHeston(complex<double> s, double St, double r, double t,
                        double sigma, double kappa,
                        double theta, double volvol, double rho) {
  complex<double> i(0, 1);
  complex<double> prod =  rho * sigma * i * s;
  complex<double> d1 = pow(prod - kappa, 2);
  complex<double> d2 = (sigma*2) * (i*s + s*2.0);
  complex<double> d = sqrt(d1 + d2);

  complex<double> g1 = kappa - prod - d;
  complex<double> g2 = kappa - prod + d;
  complex<double> g = g1/g2;

  complex<double> exp1 = exp(log(St) * i*s) * exp(i * s*r* t);
  complex<double> exp2 = 1.0 - g * exp(-d *t);
  complex<double> exp3 = 1.0 - g;
  complex<double>mainExp1 = exp1*pow(exp2/exp3, -2*theta*kappa/(pow(sigma, 2)));

  complex<double>exp4 = theta * kappa * t/(pow(sigma, 2));
  complex<double>exp5 = volvol/(pow(sigma, 2));
  complex<double>exp6 = (1.0 - exp(-d * t))/(1.0 - g * exp(-d * t));
  complex<double>mainExp2 = exp((exp4 * g1) + (exp5 *g1 * exp6));
  return mainExp1 * mainExp2;
}

// [[Rcpp::export]]
double Hprice(double s0,
              double K,
              double t,
              double r,
              double sigma,
              double kappa,
              double theta,
              double volvol,
              double rho) {
  complex<double> P(0, 0);
  int iterations = 1000;
  int maxNumber = 100;
  complex<double> i(0, 1);

  double fiter = iterations;
  double fmaxNumber = maxNumber;
  double ds = fmaxNumber/fiter;

  double element1 = 0.5 * (s0 - K * exp(-r * t));
  complex<double> numerator1;
  complex<double> numerator2;
  complex<double> denominator;
  double s1;
  complex<double> s2;

  for(int j = 1; j < iterations; j++) {
    s1 = ds * (2.0*j + 1.0)/2.0;
    s2 = s1 - i;
    numerator1 = fHeston(s2,  s0, r, t, sigma, kappa, theta, volvol, rho);
    numerator2 = K * fHeston(s1,  s0, r, t, sigma, kappa, theta, volvol, rho);
    denominator = exp(log(K) * i * s1) *i *s1;
    P = P + ds * (numerator1 - numerator2)/denominator;
  }

  complex<double> element2 = P/datum::pi;
  return real(element1 + element2);

}

//[[Rcpp::export]]
double Hmse(vec params,
            double s0,
            vec market_price,
            vec d,
            vec K,
            vec t,
            vec r) {

  int n = arma::size(market_price)(0);
  vec pred_price(n);
  double mse;

  double sigma = params(0);
  double kappa = params(1);
  double theta = params(2);
  double volvol = params(3);
  double rho = params(4);

  for(int i = 0; i < n; i++) {
    pred_price[i] = Hprice(s0-d[i], K[i], t[i], r[i], sigma, kappa, theta, volvol, rho);
  }

  mse = mean(pow(pred_price - market_price, 2));
  return mse;
}






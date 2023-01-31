library(tidyverse)
library(lubridate)
Rcpp::sourceCpp("pricingFormulas.cpp")


### generates fake call market data #### 

data = expand.grid(S0 = 100, K = c(90, 95, 105, 110, 115), t = c(1, 2, 3, 4))
libor = data.frame(t = c(1, 2, 3, 4), r = c(0.1, 1, 3, 5)/100)

data = data %>% left_join(libor, by="t")
data
data$market_price = NA

true_sigma = 0.38
true_nu = 0.9
n = nrow(data)

for(i in 1:n) {
  data$market_price[i] = VGprice(
    s0 = data$S0[i], K = data$K[i], 
    t = data$t[i], r = data$r[i],
    sigma = true_sigma, nu = true_nu
  )
}

# add noise in the VG price
data$market_price = data$market_price + rnorm(n=nrow(data), sd = 1)
data %>% glimpse()

#### BS calibration ####
s0 = data$S0 %>% unique()
K = data$K
t = data$t
market_price = data$market_price
r = data$r
d = rep(0, n) # no dividends


bs_opt = optim(par = 0.1,
               lower = 0.001, upper = 2,
               fn = BSmse, method = "L-BFGS-B",
               market_price = market_price, 
               s0 = s0, K = K, t = t, r = r, d = d)
bs_opt
sigma_bs_est = bs_opt$par
sigma_bs_est


#### VG estimate ####

# function to avoid NA's during the opt
VGmse_alt = function(params, ...) {
  res = VGmse(params, ...)
  if(is.na(res)) {
    return(10000 + sum(params))
  }else{
    return(res)
  }
}

# calibrate vg for a different set o nu initial values
calibrate_vg_model = function(nu_grid, sigma_grid, market_price, s0, K, t, r, d) {
  res = lapply(nu_grid, FUN = function(x, ...){
    print(x)
    tryCatch({
      vg_opt = optim(
        par = c(sigma_grid, x),
        lower = c(0.0001, 0.001), upper = c(3, 4),
        fn = VGmse_alt, method = "L-BFGS-B",
        market_price = market_price, s0 = s0, K = K, t = t, r = r, d = d
      )
      data.frame(sigma = vg_opt$par[1], nu = vg_opt$par[2], mse = vg_opt$value[1])
    }, error = function(cond) {
      return(data.frame(sigma = NA, nu = NA, mse = NA))
    })
    
  }) %>%
    do.call(rbind, .)
  
  res$init = nu_grid
  res %>% as_tibble() %>% arrange(mse)
}

vg_opt = calibrate_vg_model(
  nu_grid = c(0.001, 0.1, 0.5, 1, 2, 3),
  sigma_grid = bs_opt$par,
  market_price = market_price, s0 = s0, K = K, t = t, r = r, d = d
)

vg_opt %>% head(1) # best sigma, nu

#### Heston model ####

Hmse_alt = function(params, ...) {
  res = Hmse(params, ...)
  if(is.na(res)) {
    return(1000000 + sum(params^2))
  }else{
    return(res)
  }
}

calibrate_heston_model = function(param_grid, market_price, s0, K, t, r, d) {
  res = lapply(param_grid, FUN = function(x, ...){
    print(x)
    tryCatch({
      heston_opt = optim(
        par = x,
        lower = c(0.001, 0.0001, 0.0001, 0.001, -0.999),
        upper = c(1.2, 30, 30, 1.2, 0.999),
        fn = Hmse_alt, method = "L-BFGS-B",
        market_price = market_price, s0 = s0, K = K, t = t, r = r, d = d
      )
      
      data.frame(
        sigma = heston_opt$par[1],
        kappa = heston_opt$par[2],
        theta = heston_opt$par[3],
        volvol = heston_opt$par[4],
        rho = heston_opt$par[5],
        mse = heston_opt$value[1],
        sigma_init = x[1],
        kappa_init = x[2],
        theta_init = x[3],
        volvol_init = x[4],
        rho_init = x[5]
      )
    }, error = function(cond) {
      return(
        data.frame(
          sigma = NA,
          kappa = NA,
          theta = NA,
          volvol = NA,
          rho = NA,
          mse = NA,
          sigma_init = x[1],
          kappa_init = x[2],
          theta_init = x[3],
          volvol_init = x[4],
          rho_init = x[5]
        )
      )
    })
    
  }) %>%
    do.call(rbind, .)
  res %>% as_tibble() %>% arrange(mse)
}


set.seed(10)
h_init_grid = expand.grid(bs_opt$par,
                          kappa = runif(5, 0, 10),
                          theta = runif(5, 0, 10),
                          volvol = c(0.01, 0.1),
                          rho = c(-0.9, -0.8, -0.3, 0, 0.3, 0.8, 0.9))

h_init = list()
for(i in 1:nrow(h_init_grid)) {
  h_init[[i]] = h_init_grid[i,] %>% as.numeric()
}

length(h_init)

heston_opt = calibrate_heston_model(
  param_grid = h_init, market_price = market_price,
  s0 = s0, K = K, t = t, r = r, d = d
)

heston_opt

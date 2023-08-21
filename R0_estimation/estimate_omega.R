library(stats)

# The following two functions are copied directly 
# from EpiEstim (https://github.com/mrc-ide/EpiEstim as described in Cori et al., 2013)*:

vnapply <- function(X, FUN, ...) {
  vapply(X, FUN, numeric(1), ...)
}

discr_si <- function(k, mu, sigma) 
{
  if (sigma < 0) {
    stop("sigma must be >=0.")
  }
  if (mu <= 1) {
    stop("mu must be >1")
  }
  if (any(k < 0)) {
    stop("all values in k must be >=0.")
  }
  
  a <- ((mu - 1) / sigma)^2
  b <- sigma^2 / (mu - 1)
  
  cdf_gamma <- function(k, a, b) stats::pgamma(k, shape = a, scale = b)
  
  res <- k * cdf_gamma(k, a, b) + 
    (k - 2) * cdf_gamma(k - 2, a, b) - 2 * (k - 1) * cdf_gamma(k - 1, a, b)
  res <- res + a * b * (2 * cdf_gamma(k - 1, a + 1, b) - 
                          cdf_gamma(k - 2, a + 1, b) - cdf_gamma(k, a + 1, b))
  res <- vnapply(res, function(e) max(0, e))
  
  return(res)
}

discr_si(1:30, mu = 4.7, sigma = 2.9) 

# * Citation, as per https://github.com/mrc-ide/EpiEstim
# @misc{Cori2021, author={Cori, A and Kamvar, ZN and Stockwin, J and Jombart, T and Dahlqwist, E and FitzJohn, R and Thompson, R},
# year={2021},
# title={{EpiEstim v2.2-3: A tool to estimate time varying instantaneous reproduction number during epidemics}},
# publisher={GitHub}, journal={GitHub repository},
# howpublished = {\url{https://github.com/mrc-ide/EpiEstim}}, commit={c18949d93fe4dcc384cbcae7567a788622efc781},
# }
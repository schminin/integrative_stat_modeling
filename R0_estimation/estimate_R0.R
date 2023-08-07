# load libraries
library(R0)

# load data
df <- read.csv("../data/preprocessed/canada.csv")
df$date <- as.Date(df$date)

# estimate R0 for Toronto, Highland Creek
pop <- 13000
estimate.R(epid = df$cases_reported_date,
           t = df$date, 
           pop.size = pop, 
           methods = c('AR'))
# EG, 'ML', 'AR', 'TD', 'SB'
# several methods are available: https://cran.r-project.org/web/packages/R0/R0.pdf

# fx <- function(x) {(x >= 0 & x < 9/20) * ((100/27) * x^2) +


gx <- function(x) {ifelse((x >= 0 & x < 9/40), ((160/9) * x + 2), ifelse((x >= 9/40 & x < 23/40), ((-80/7) * x + (60/7)), ifelse((x >= 23/40 & x < 17/20), 2, ifelse((x >= 17/20 & x <= 1), ((-20/3) * x + (23/3)), NA))))}
fx <- function(x) {ifelse((x >= 0 & x < 9/20), ((100/27) * x^2), ifelse((x > 9/20 & x < 7/10), (9/8), ifelse((x >= 7/10 + 0.000000001 & x <= 1), ( 2 - (25/9) * (1-x) * (1-x)), NA)))}
f = Vectorize(fx)
g = Vectorize(gx)
plot(f, 0, 1, col="red", xlim = c(0,1), ylim = c(0,6))
plot(g, 0, 1, col="blue", add = TRUE)


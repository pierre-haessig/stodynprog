# Fit a time series model to Searev data
#Pierre Haessig — April 2013

library(forecast)

####
# 0) Preamble:
options(max.print = 100)
# String concatenation
'%+%' <- function(a,b) {paste(a,b,sep='')}

# Simulation parameters:
damp = 4.e6 # N/(rad/s)
torque.max = 2e6 # N.m
power.max = 1.1e6 # W

# Torque command law ("PTO strategy")
torque.law <- function(speed) {
  tor = speed * damp
  # 1) Max torque limitation:
  if (tor > torque.max) {
    tor = torque.max
  }
  if (tor < -torque.max) {
    tor = -torque.max
  } 
  # 2) Max power limitation:
  if (tor*speed > power.max) {
    tor = power.max/speed
  }
  return(tor)
}
torque.law <- Vectorize(torque.law)


ts = 0.1 # [s]
Hs = 3. # [m]
Tp = 9. # [s]

# read Searev data file:
fname = 'data/Em_1.txt'
data = read.table(fname, skip=4)

# split columns:
elev   = ts(data$V2, deltat=ts, start=0)
angle  = ts(data$V3, deltat=ts, start=0)
speed  = ts(data$V4, deltat=ts, start=0)
torque = ts(data$V5, deltat=ts, start=0)
power=speed*torque/1e6 # [MW]

N = length(speed)

# Threshold at speed > 0.5 rad/s
speed_ns = speed
speed_ns[abs(speed )>0.5] = NA
angle_ns = angle
angle_ns[abs(speed )>0.5] = NA


# Regenerate the time vector because there are some irregularities:
t = seq(N)*ts

plot(speed, col='#007700', main='Searev speed', ylab='speed (rad/s)')

### Fit various ARMA model:
ar2 = arima(speed, order=c(2,0,0), include.mean=FALSE)
#summary(ar2)
arma21 = arima(speed, order=c(2,0,1), include.mean=FALSE)
#summary(arma21)
ar3 = arima(speed, order=c(3,0,0), include.mean=FALSE)
#summary(ar3)
arma31 = arima(speed, order=c(3,0,1), include.mean=FALSE)
#summary(arma31)

### Plot the ACF and compare with models:
lag.max=200

#X11()
acf(speed, lag.max=lag.max, main='ACF of speed and AR models')
acf.ar2 = ARMAacf(ar2$model$phi, ar2$model$theta, lag.max)
acf.ar3 = ARMAacf(ar3$model$phi, ar3$model$theta, lag.max)
lines((0:lag.max)*ts, acf.ar2, col='#007777')
lines((0:lag.max)*ts, acf.ar3, col='#7700AA')

res2 = ar2$residuals
res3 = ar3$residuals
res31 = arma31$residuals

### Predictions:

# horizon:
h = 50 # (s)
h.int = as.integer(h/ts)

speed.ar2 = predict(ar2, h.int)$pred
speed.ar2.se = predict(ar2, h.int)$se
speed.ar2.sup = speed.ar2 + 1.96 * speed.ar2.se
speed.ar2.inf = speed.ar2 - 1.96 * speed.ar2.se

speed.ar3 = predict(ar3, h.int)$pred
speed.ar3.se = predict(ar3, h.int)$se
speed.ar3.sup = speed.ar3 + 1.96 * speed.ar3.se
speed.ar3.inf = speed.ar3 - 1.96 * speed.ar3.se

# Plot the prediction
#X11()
# 1) past data
plot(window(speed, start=950), col='black',
     xlim=c(950,1000+h), ylim=c(-0.6, 0.6),
     main='prediction of the speed', ylab='speed (rad/s)')
# 2) confidence intervals
polygon(c(index(speed.ar3),rev(index(speed.ar3))),
        c(speed.ar3.sup, rev(speed.ar3.inf)), col="#FFDDFF", border=NA)
polygon(c(index(speed.ar2),rev(index(speed.ar2))),
        c(speed.ar2.sup, rev(speed.ar2.inf)), col="#DDFFFF", border=NA)
# 3) prediction
lines(speed.ar2, col='#007777')
lines(speed.ar3, col='#7700AA')


### Fit a state space model:

# Capture the non-linear aspect of the torque in regressor:
nltor = torque/damp - speed # (homogeneous to a speed)

a.reg = ts.intersect(angle, lag(speed, -1), lag(angle,-1))
s.reg = ts.intersect(speed, lag(speed, -1), lag(angle,-1), lag(nltor, -1))

a.fit = lm(a.reg[,1] ~ a.reg[,2:3]-1)
s.fit = lm(s.reg[,1] ~ s.reg[,2:3]-1)

a.res = ts(a.fit$residuals, deltat=ts, start=0)
s.res = ts(s.fit$residuals, deltat=ts, start=0)

# Transition matrix:
trans.mat = rbind(coef(s.fit), coef(a.fit))

# Make a prediction:
x = rbind(speed[N], angle[N])

speed.pred = ts(rep(0,h.int), freq=10, start=999.9)

for(i in 1:h.int) {
  x = trans.mat %*% x
  speed.pred[i] = x[1]  
}
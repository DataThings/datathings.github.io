luxweatherdata <- read.csv("/Users/assaad/work/github/datathings.github.io/static/datasets/weatherlux.csv")
luxts = ts(as.vector(t(luxweatherdata['temperature'])), start=c(1947, 1), end=c(2016, 12), frequency = 12)
decomposition = stl(luxts, t.window = 12*30, s.window="periodic")
plot(decomposition)
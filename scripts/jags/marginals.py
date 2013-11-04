from math import sqrt

count = 0
total = 0.0
total_sq = 0.0
for line in open('CODAchain1.txt'):
  (sampn, value) = line = line.split()
  value = float(value)
  total += value
  total_sq += value*value
  count += 1

mean = total / count
var = (total_sq - total**2/count) / count

mvar = var / count
mstd = sqrt(mvar)
errbar = mstd * 2.0

print "%s < %s < %s" % (mean-errbar, mean, mean+errbar)

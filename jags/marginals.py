
values = []
for line in list(open('CODAchain1.txt')):
  (sampn, value) = line.split()
  values.append(float(value))

def mean(vals):
  return sum(vals) / float(len(vals))

print mean(values)

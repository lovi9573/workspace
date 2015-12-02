
def d_new(d_old, d_size, acc, s):
	return (d_old*d_size+acc*s)/(d_size + s)

def acc_new(m,d):
	return m*d + (1-m)*(1-d)

def sim(d_0, m_0, s_0, kd, km):
	d = [d_0]
	m = [m_0]
	acc = [m[0]*d[0]]
	s = [s_0]
	for i in range(15):
		print "=========== Iteration {} ===========".format(i)
		print "Dataset Size: {}\nDataset Accuracy: {}\nModel Capacity: {}\nPrediction Accuracy: {}".format(s[i],d[i],m[i],acc[i])
		d.append(d_new(d[i],s[i],acc[i],kd*s[i]))
		m.append(km*1 + (1-km)*m[-1])
		s.append(s[i] + kd*s[i])
		acc.append(acc_new(m[i+1],d[i+1]))
	return acc[-1]

if __name__ == "__main__":
	res = {}
	for kdb in range(1,10):
		kd = float(kdb)
		for kmb in range(5):
			km = float(kmb)/10+0.05
			res[(kd,km)] = sim(1,0.75,100,kd,km)	
	for k,v in res.iteritems():
		print k,v


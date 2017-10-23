import random
import sys

def randstr(len):
	str = ''
	for i in range(0,len):
		str = str + random.choice('0123456789abcdefghijklmnopqrstuvwxyz')
	
	return str
	
if __name__ == '__main__':
	if len(sys.argv) == 2:
		print randstr(int(sys.argv[1]))
		
	else:
		print 'Usage: python randomstr.py String_Length'
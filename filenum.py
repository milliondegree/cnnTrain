import os
import sys

if __name__ == '__main__':
	if len(sys.argv) == 2:
		dir = sys.argv[1]
		print len(os.listdir(dir))
		
	else:
		print 'Usage: python filenum.py Directory'
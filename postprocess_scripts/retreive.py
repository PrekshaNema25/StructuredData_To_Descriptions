import sys
import os
import re

def split_file(filename):

	with open(filename, "rb") as f:

		f1 = open(filename + "_plabels","wb")
		f2 = open(filename + "_tlabels", "wb")

		count = 0
		for lines in f:
		#	print count
			lines = lines.strip("\n")
			if (count % 2 == 0):
				#x = re.sub("<eos>", "", lines)
				x = lines.split("<eos>")[0]
				x = re.sub("Predicted summary ::", "",x)
				x = re.sub("<pad>", "",x)
				x = " ".join(x.split())

			else:
				x1 = lines.split("<eos>")[0]
				x1 = re.sub("True labels ::","",x1)
				x1 = re.sub("<pad>", "",x1)
				x1 = " ".join(x1.split())
				if not(x1 is " " or x1 is ""):
					f1.write(x + " <eos>\n")
					f2.write(x1 + " <eos>\n")

 				else:
					print count
					print (x)
			count = count + 1

def main():

	split_file(sys.argv[1])


if __name__ == '__main__':
	main()

import re
import sys
import os


def convert_indices_aw(content_file, title_file, as_file):

	title = open(title_file, "r")
	content = open(content_file, "r")
	attn = open(as_file, "r")

	write_file = open(title_file + "_copy", "w")

	count = 0 
	for (line_t, line_c, line_a) in zip (title, content, attn):


		words_t = line_t.split()
		words_c = line_c.split()
		words_a = line_a.split()
		temp = []
		for (i, word)  in enumerate(words_t):

			if (word == "<unk>" and len(words_c) > int(words_a[i]) ):
				temp.append(words_c[int(words_a[i])])

			else:
				temp.append(word)


		temp.append("\n")
		write_file.write(" ".join(temp))

def convert_indices(content_file, title_file, len_vocab):

	title = open(title_file, "r")
	content = open(content_file, "r")
	write_file = open(title_file + "_c", "w")
	for (line_t, line_c) in zip(title, content):

		words_t = line_t.split()
		words_c = line_c.split()
		temp = []
		print ("P")
		for word in words_t:
			if (word.isdigit() and int(word) >= len_vocab):
				if (int(word)-len_vocab >= len(words_c)):
					temp.append("<unk>")
				else:
					temp.append(words_c[int(word) - len_vocab])
			else:
				temp.append(word)
		temp.append("\n")

		write_file.write(" ".join(temp))

def main():
	#convert_indices(sys.argv[1], sys.argv[2], int(sys.argv[3]))
	convert_indices_aw(sys.argv[1], sys.argv[2], sys.argv[3])
	print "Done with the postprocessing"


if __name__ == '__main__':
	main()

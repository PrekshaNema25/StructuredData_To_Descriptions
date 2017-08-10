f1 = open('train_content')
f2 = open('train_summary')
f3 = open('train_field')

words = {}

for lines in f1:
	w = lines.split()
	for i in w:
		if i in words:
			words[i] += 1
		else:
			words[i] = 1


for lines in f2:
        w = lines.split()
        for i in w:
                if i in words:
                        words[i] += 1
                else:
                        words[i] = 1


for lines in f3:
        w = lines.split()
        for i in w:
                if i in words:
                        words[i] += 1
                else:
                        words[i] = 1



values = []
for i in words:
	values.append(words[i])

values.sort()

print values[-20000]

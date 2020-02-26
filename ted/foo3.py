from collections import Counter
import re
words = re.findall(r'\w+', open('hamlet.txt').read().lower())
print(Counter(words).most_common(10))
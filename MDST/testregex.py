import re

# regex to identify highways
patt = re.compile(r"I-\d+|HWY|Highway|Interstate", flags=re.IGNORECASE)

assert(re.search(patt, "I-225"))
assert(re.search(patt, "I-225 W"))
assert(re.search(patt, "asbasdfI-225 W"))
assert(re.search(patt, "hwy 12"))
assert(re.search(patt, "Chicago I-80 S"))
assert(re.search(patt, "asdfbasdf") == None)
assert(re.search(patt, "W 404 main st") == None)
assert(re.search(patt, "CR-23") == None)

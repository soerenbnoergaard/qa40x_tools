import sys
import requests

outfile = sys.argv[1]
num_repeats = 30

def write(s):
    with open(outfile, "a") as f:
        f.write(s + "\n")

def measure_rms():
    requests.post("http://localhost:9402/Acquisition")
    return float(requests.get("http://localhost:9402/RmsDbv/20/20000").json()["Left"])

print(f"{num_repeats=}")

while True:
    r = input("Variable value: ").strip()
    if r == "":
        break
    for _ in range(num_repeats):
        s = "{},{}".format(r, measure_rms())
        print(s)
        write(s)

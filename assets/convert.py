import os

no = ["convert.py", "tabletop.png", "resize.py", "README.md", "sounds"]

filename_map = {
    "h": "Hearts",
    "d": "Diamonds",
    "c": "Clubs",
    "s": "Spades",
}

value_map = {
    1: "A",
    11: "J",
    12: "Q",
    13: "K",
}

os.system(f"convert Card-Back-01.png -resize 80x111\> Back.png")


# images = os.listdir()


# for i in images:
#     if i.startswith('.'):
#         continue

#     if i not in no:
#         value = int(i[1:-4])
#         if value == 1 or value > 10:
#             value = value_map[value]
#         filename = filename_map[i[0:1]] + str(value) + ".png"
#         os.system(f"convert {i} -resize 80x111\> {filename}")


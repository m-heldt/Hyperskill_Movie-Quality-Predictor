# the list with classes; please, do not modify it
groups = ['1A', '1B', '1C', '2A', '2B', '2C', '3A', '3B', '3C']

# your code here
classes_numbers = int(input())
group_out = {}
for group in groups:
    group_out[group] = None

for group in group_out.keys():
    if classes_numbers == 0:
        break
    group_out[group] = int(input())
    classes_numbers -= 1

print(group_out)

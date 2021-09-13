from ChessModelTools_v7_ResNet import Tools


tools = Tools()

classes = tools.load("data/classes.pkl")
dist = {}

for item in classes.keys():
    dist[item] = 0


for i in range(7):
    x, y = tools.retrieve_MySql_table(i, conv=True)
    for label in y:
        if label in dist.keys():
            dist[label] += 1
        else:
            dist[label] = 1

sort_dist = [i[0] for i in sorted(dist.items(), key=lambda x: x[1])]
for item in sort_dist:
    print(item)

# tools.split_classes(sort_dist)


# tools.save(sort_dist, "data/Classes/Distributed.pkl")
# print(len(sort_dist))
# print("Done!")
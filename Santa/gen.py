from haversine import haversine
import pandas
import numpy
from sklearn.cluster import KMeans

sleigh_size = 1000
north_pole = (90.0, 0.0)

df = pandas.read_csv('gifts.csv')
rows = df.get_values()
unvisited = set()
for i in range(0, len(rows)):
    unvisited.add(i)
total_weight = 0
for row in rows:
    total_weight += row[3]

def add_dest(file_descriptor, data, giftindex, trip, uv):
    file_descriptor.write("{},{}\n".format(data[giftindex][0], trip))
    uv.remove(giftindex)
 
def greedy_simple(data):
    with open('submission.csv', 'w') as f:
        trip_id = 0
        f.write('GiftId,TripId\n')
        while len(data) > 0:
            trip_id += 1
            print "trip: {}".format(trip_id)
            sleigh = 0
            dest = None
            destindex = list(unvisited)[0]
            weight = data[destindex][3]
            while sleigh + weight <= sleigh_size:
                print "loop started, sleigh: {}".format(sleigh)
                dest = data[destindex]
                add_dest(f, data, destindex, trip_id, unvisited)
                sleigh += weight

                mindist = None
                minindex = None
                for i in unvisited:
                    dist = haversine((dest[1], dest[2]), (data[i][1], data[i][2]))
                    if mindist == None or dist < mindist:
                        mindist = dist
                        minindex = i
                weight = data[minindex][3] 
                destindex = minindex
          
greedy_simple(rows)

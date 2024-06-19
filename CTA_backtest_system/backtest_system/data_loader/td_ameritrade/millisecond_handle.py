from datetime import datetime
from os import times
import pandas as pd

dt_obj = datetime.strptime('2000-01-01 00:00:00', '%Y-%m-%d %H:%M:%S')
millisec = int(dt_obj.timestamp() * 1000)

Y2000 = int(datetime.strptime('2000-01-01 00:00:00', '%Y-%m-%d %H:%M:%S').timestamp() * 1000)
#946656000000
Y2001 = int(datetime.strptime('2001-01-01 00:00:00', '%Y-%m-%d %H:%M:%S').timestamp() * 1000)
Y2002 = int(datetime.strptime('2002-01-01 00:00:00', '%Y-%m-%d %H:%M:%S').timestamp() * 1000)
Y2003 = int(datetime.strptime('2003-01-01 00:00:00', '%Y-%m-%d %H:%M:%S').timestamp() * 1000)

# print(Y2000)
# print(Y2001)
# print(Y2002)
# print(Y2003)



# print(Y2001-Y2000)
# print(Y2002-Y2001)
# print(Y2003-Y2002)


# quaater

Jan_2021 = int(datetime.strptime('2021-01-01 00:00:00', '%Y-%m-%d %H:%M:%S').timestamp() * 1000)
Apr_2021 = int(datetime.strptime('2021-04-01 00:00:00', '%Y-%m-%d %H:%M:%S').timestamp() * 1000)
Jul_2021 = int(datetime.strptime('2021-07-01 00:00:00', '%Y-%m-%d %H:%M:%S').timestamp() * 1000)
Oct_2021 = int(datetime.strptime('2021-10-01 00:00:00', '%Y-%m-%d %H:%M:%S').timestamp() * 1000)

Jan_2022 = int(datetime.strptime('2022-01-01 00:00:00', '%Y-%m-%d %H:%M:%S').timestamp() * 1000)
Apr_2022 = int(datetime.strptime('2022-04-01 00:00:00', '%Y-%m-%d %H:%M:%S').timestamp() * 1000)
Jul_2022 = int(datetime.strptime('2022-07-01 00:00:00', '%Y-%m-%d %H:%M:%S').timestamp() * 1000)
Oct_2022 = int(datetime.strptime('2022-10-01 00:00:00', '%Y-%m-%d %H:%M:%S').timestamp() * 1000)

Q1_22 = Apr_2022 - Jan_2022
Q2_22 = Jul_2022 - Apr_2022


end_1 = int(datetime.strptime('2022-09-21 23:59:00', '%Y-%m-%d %H:%M:%S').timestamp() * 1000)
start_1 = int(datetime.strptime('2022-08-08 08:00:00', '%Y-%m-%d %H:%M:%S').timestamp() * 1000)



print("-----------")
print(end_1)
print("-----------")
print(Y2000)
print("----delta-------")
print(end_1 - start_1)

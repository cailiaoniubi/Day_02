#Excercise Day 2

# improt packages to be used
import seaborn

import matplotlib.pyplot as plt

# 1. load the seaborn dateset
df = seaborn.load_dataset("tips")

# 2. convert short names to long names and store the dataset in the same variable
shorts = ["Sun","Sat","Thur","Fri"]
longs = ["Sunday","Saturday","Thursday","Friday"]
df["day"] = df["day"].replace(shorts, longs)



# df["day"]= df["day"].replace("Sun","Sunday")
# df["day"]= df["day"].replace("Sat","Saturday")
# df["day"]= df["day"].replace("Thur","Thursday")
# df["day"]= df["day"].replace("Fri","Friday")
#
#
# df3 = df["day"].replace["sun","sat","Thur","Fri"].["Sunday","Saturday","Thursday","Friday"]


# df["days"] = df["day"].replace["sun","sat","Thur","Fri"].["Sunday","Saturday","Thursday","Friday"]


# 3. creat a scatterplot with tip
# set the global options
seaborn.set(style="white")
plt.show()
fig, axes = plt.subplots()

# now do the plotting
seaborn.scatterplot(x="tip",y="total_bill",hue="day",data = df)
df.info()
# catplot(x="tip",y="total_bill",hue="day",data = df, col="sex")
# facet_grid= catplot(x="tip",y="total_bill",hue="day",data = df, col="sex")

# seaborn.catplot(x="tip",y="total_bill",hue="day",data = df, col="sex", kind = "strip")
# facet_grid.xlabel("Tip")
# facet_grid.ylabel("Total Bill")

seabron.catplot(x="tip",y="total_bill",hue="day",data = df, col="sex")



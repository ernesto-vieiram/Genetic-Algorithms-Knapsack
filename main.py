import knapsack

#Create new GA session with simulation parameters
session = knapsack.Session(50, 25, 0.6, 0.00067, 400)
#Generate random set of items and load them to "test.csv"
session.generator(1500, 19000, 14000, "test.csv")
#Load items from the csv document
session.read_task("test.csv")

#Initialise Generation 0
session.init_population()
#Configure data gathering and plotting parameters
session.stats = knapsack.Stats(session.task, session.iterations, 50)
#Execute simulation
result = session.execute()
print(result)

#Load and plot yielded data
session.stats.plot_fitness("KNAPSACK")
session.stats.plot_scatter()
session.show_stats()




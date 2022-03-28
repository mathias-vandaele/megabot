from neural_intelligence.batches_generator import calculate_local_top, calculate_sell_index
from matplotlib import pyplot as plt


clean_data = calculate_sell_index('BTCUSDT', 60)

fig, ax = plt.subplots()
ax.scatter(clean_data[:,14].flatten(), clean_data[:,4].flatten(), c = clean_data[:,12].flatten(), s = 0.4)
plt.show()
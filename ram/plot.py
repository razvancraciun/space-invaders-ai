from matplotlib import pyplot as plt


f = open('log.txt', 'r')
nums = f.read().split()
f.close()
nums = [int(num[:-3]) for num in nums[1:] ]
plt.plot(nums)
plt.xlabel('Games played')
plt.ylabel('Score')
plt.show()

means = [ 0 if i < 100 else sum(nums[i-100:i]) / 100 for i in range(len(nums))]
plt.plot(means)
plt.xlabel('Games played')
plt.ylabel('Avg last 100 games')
plt.show()

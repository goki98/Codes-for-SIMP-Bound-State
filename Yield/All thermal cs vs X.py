import pandas as pd
import matplotlib.pyplot as plt
from labellines import labelLines
# Step 1: Read the CSV file
df = pd.read_csv("thermal_avg_all_CS2.csv")

# Step 2: Check the data
print(df.shape)      # (rows, columns)
print(df.columns)    # list of column names
# sigma v
plt.figure(figsize=(8,8))
plt.plot(df['x'], df['cs2'],'r',label='2')
plt.plot(df['x'], df['cs1'],'g',label='1')
plt.plot(df['x'], df['cs10_t'],'b',label='6')
plt.plot(df['x'], df['cs12_t'],'y',label='8')
plt.xlabel(r"$x = m/T$")
plt.ylabel(r'$<\sigma v>$')
plt.yscale("log")
plt.xscale("log")
labelLines(plt.gca().get_lines(), align=True,xvals=[80, 80,40,80], fontsize=16)
#plt.ylim(1e30, 1e37)
#plt.ylim(1e10, 1e22)
plt.tick_params(axis='both', which='both', direction='in', top=True, right=True)
plt.minorticks_on()

plt.grid(True, which="both", ls="--", alpha=0.5)
plt.legend()
plt.show()

# sigama v^2 plots
plt.figure(figsize=(8,8))
plt.plot(df['x'], df['cs9_t'],'r',label='5')
plt.plot(df['x'], df['cs11_t'],'g',label='7')

plt.xlabel(r"$x = m/T$")
plt.ylabel(r'$<\sigma v^2>$')
plt.yscale("log")
plt.xscale("log")
#plt.ylim(1e30, 1e37)
#plt.ylim(1e10, 1e22)
plt.tick_params(axis='both', which='both', direction='in', top=True, right=True)
plt.minorticks_on()
labelLines(plt.gca().get_lines(), align=True,xvals=[80, 80], fontsize=16)
plt.grid(True, which="both", ls="--", alpha=0.5)
plt.legend()
plt.show()
# sigama v^3 plots
plt.figure(figsize=(8,8))
plt.plot(df['x'], df['cs3'],'r',label='3')
plt.plot(df['x'], df['cs4_t'],'g',label='4')

plt.xlabel(r"$x = m/T$")
plt.ylabel(r'$<\sigma v^3>$')
plt.yscale("log")
plt.xscale("log")
#plt.ylim(1e30, 1e37)
#plt.ylim(1e10, 1e22)
# Add ticks on all sides
plt.tick_params(axis='both', which='both', direction='in', top=True, right=True)
plt.minorticks_on()
# Label directly along the lines
labelLines(plt.gca().get_lines(), align=True,xvals=[80, 80], fontsize=16)

plt.grid(True, which="both", ls="--", alpha=0.5)

plt.show()
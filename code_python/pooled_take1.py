#collapse-hide
a_ = 2
b_ = -0.4
x_ = np.linspace(0, 10, 31)
year_ = np.arange(2021-len(x_), 2021)
y_ = a_ + b_ * x_ 

fig, ax = plt.subplots()
ax.plot(x_, y_, "o-")
ax.text(
    0.93, 0.9, r"$y_i = a + bx_i + \mathcal{N}(0,1)$", ha='right', va='top', transform=ax.transAxes, fontsize=18
)

ax.set_xticks(x_[::3])
ax.set_xticklabels(year_[::3])
ax.set_yticks([])
ax.set_xlabel("Year")
ax.set_ylabel("Quantity of interest");

year_
x_
coords = {"year": year_}
with pm.Model(coords=coords) as linreg_model:
    x = pm.Data("x", x_, dims="year")
    
    a = pm.Normal("a", 0, 3)
    b = pm.Normal("b", 0, 2)
    sigma = pm.HalfNormal("sigma", 2)
    
    y = pm.Normal("y", a + b * x, sigma, observed=y_, dims="year")
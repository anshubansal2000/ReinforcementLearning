"""Regenerate publication graphics from the saved experiment results."""
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch, Circle
from matplotlib.colors import LogNorm
import numpy as np

ROOT = Path(__file__).resolve().parent
OUT = ROOT / "blog" / "images"
OUT.mkdir(exist_ok=True)
DATA = np.load(ROOT / "results" / "values.npz")
BG, INK, MUTED = "#f5f3ee", "#172b38", "#586873"
BLUE, ORANGE, GREEN = "#126c91", "#c56732", "#237b65"
plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 12,
                     "text.color": INK, "axes.labelcolor": INK,
                     "svg.fonttype": "none"})

def canvas(title, subtitle, height=7):
    fig = plt.figure(figsize=(14, height), facecolor=BG)
    fig.text(.045, .94, title, fontsize=25, weight="bold", va="top")
    fig.text(.045, .87, subtitle, fontsize=12, color=MUTED, va="top")
    return fig

def save(fig, name):
    fig.savefig(OUT / f"{name}.png", dpi=180, facecolor=BG)
    fig.savefig(OUT / f"{name}.svg", facecolor=BG)
    plt.close(fig)

def panel(fig, bounds):
    ax = fig.add_axes(bounds)
    ax.set(xlim=(0, 1), ylim=(0, 1))
    ax.axis("off")
    return ax

def card(ax, x, y, w, h, color="white"):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.012,rounding_size=0.025",
                              facecolor=color, edgecolor="none"))

def arrow(ax, a, b, color=BLUE):
    ax.add_patch(FancyArrowPatch(a, b, arrowstyle="-|>", mutation_scale=20,
                               linewidth=2.5, color=color))

fig = canvas("A simple world. Consequences that compound.",
             "25 states  /  4 actions  /  stochastic movement  /  rewards over time")
ax = fig.add_axes([.05, .17, .38, .61])
ax.set(xlim=(-.1, 5.1), ylim=(5.1, -.1), aspect="equal")
ax.axis("off")
for r in range(5):
    for c in range(5):
        special = (r,c) in [(1,3),(2,2)]
        color = "#f0d4c4" if special else "#d8ebe3" if (r,c)==(4,4) else "white"
        ax.add_patch(FancyBboxPatch((c+.04,r+.04), .92,.92,boxstyle="round,pad=0,rounding_size=0.1",
                                  facecolor=color, edgecolor="#d9dedc"))
        if special:
            ax.add_patch(Circle((c+.5,r+.43),.19,facecolor=ORANGE))
            ax.text(c+.5,r+.78,"CRATER",ha="center",fontsize=8,color=ORANGE,weight="bold")
        elif (r,c)==(0,0):
            ax.text(c+.5,r+.5,"START",ha="center",va="center",weight="bold",fontsize=11,color=BLUE)
        elif (r,c)==(4,4):
            ax.text(c+.5,r+.5,"GOAL",ha="center",va="center",weight="bold",fontsize=11,color=GREEN)
        else:
            ax.text(c+.5,r+.5,str(r*5+c),ha="center",va="center",color="#86949b",fontsize=11)
right=panel(fig,[.48,.16,.47,.64])
card(right,0,.51,1,.46)
right.text(.05,.88,"REQUEST RIGHT",fontsize=15,weight="bold")
right.text(.05,.75,"An action is a request, not a guarantee.",color=MUTED,fontsize=12)
arrow(right,(.3,.58),(.64,.58))
right.text(.70,.58,"80% right",va="center",color=BLUE,weight="bold")
right.text(.05,.64,"10% up",color=ORANGE,fontsize=11)
right.text(.05,.54,"10% down",color=ORANGE,fontsize=11)
for y,label,value,color in [(.35,"Ordinary move","−1",INK),(.19,"Goal reached","+10",GREEN),(.03,"Crater entered","−10",ORANGE)]:
    right.text(.05,y,label,fontsize=15)
    right.text(.94,y,value,fontsize=20,weight="bold",color=color,ha="right")
fig.text(.05,.07,"Goal and crater entry end the episode. Arrival rewards replace the ordinary movement cost.",color=MUTED,fontsize=11)
save(fig,"rover-world")

fig=canvas("Same experience. Different moments of learning.",
           "MC waits for the outcome. TD updates after each transition.",height=7.5)
ax=panel(fig,[.045,.12,.91,.67])
for y,name,color,description in [(.63,"MONTE CARLO",ORANGE,"Complete the episode → compute returns → update visited states"),
                                  (.16,"TD(0)",BLUE,"Take one step → combine reward + next-state estimate → update")]:
    card(ax,0,y-.10,1,.39)
    ax.text(.025,y+.205,name,color=color,weight="bold",fontsize=15)
    ax.text(.025,y-.035,description,fontsize=12,color=MUTED)
    for i,label in enumerate(["State A","State B","State C","Terminal"]):
        x=.13+i*.235
        ax.text(x,y+.085,label,ha="center",va="center",fontsize=13,weight="bold")
        if i<3:
            arrow(ax,(x+.066,y+.085),(x+.168,y+.085),color)
    if name=="MONTE CARLO":
        ax.text(.835,y+.20,"Learn after finish",color=color,fontsize=11,ha="center")
    else:
        for x in [.247,.482,.717]:
            ax.text(x,y+.19,"update",ha="center",fontsize=10,color=color)
fig.text(.05,.055,"TD bootstraps from an estimate that may be wrong. MC uses full sampled returns that may be noisy.",fontsize=11,color=MUTED)
save(fig,"mc-vs-td")

fig=canvas("Removing slip changes the value of the map.",
           "Optimal discounted returns from DP  •  gamma = 0.95  •  shared color scale",height=7.3)
for bounds,key,title,start in [([.06,.19,.37,.58],"dp","Dusty terrain · 10% each perpendicular slip","Start: −1.416"),
                                ([.57,.19,.37,.58],"no_slip","Reliable terrain · no slip","Start: +0.950")]:
    ax=fig.add_axes(bounds)
    values=DATA[key].reshape(5,5)
    masked=np.ma.array(values, mask=np.isin(np.arange(25).reshape(5,5),[8,12,24]))
    cmap=plt.get_cmap("YlGnBu").copy();cmap.set_bad("#deded7")
    im=ax.imshow(masked,cmap=cmap,vmin=-3,vmax=10)
    for r in range(5):
        for c in range(5):
            s=r*5+c
            label="GOAL" if s==24 else "CRATER" if s in [8,12] else f"{values[r,c]:.2f}"
            ax.text(c,r,label,ha="center",va="center",fontsize=9 if s in [8,12,24] else 12,
                    color="white" if values[r,c]>5.5 else INK,weight="bold")
    ax.set_xticks([]);ax.set_yticks([])
    for spine in ax.spines.values():spine.set_visible(False)
    ax.set_title(title,fontsize=12,pad=15)
    ax.set_xlabel(start,fontsize=17,weight="bold",labelpad=12)
cax=fig.add_axes([.474,.23,.014,.47]);fig.colorbar(im,cax=cax)
fig.text(.05,.06,"Terminal states have zero remaining value; entry rewards are paid on arrival.",fontsize=11,color=MUTED)
save(fig,"terrain-values")

fig=canvas("20,000 episodes can still leave blind spots.",
           "Visit counts under the fixed rover policy  •  seed 42  •  logarithmic color scale",height=7.3)
ax=fig.add_axes([.06,.16,.43,.62])
visits=DATA["visits"].reshape(5,5)
masked=np.ma.masked_equal(visits,0)
cmap=plt.get_cmap("YlGnBu").copy();cmap.set_bad("#deded7")
im=ax.imshow(masked,cmap=cmap,norm=LogNorm(vmin=10,vmax=25000))
for r in range(5):
    for c in range(5):
        s=r*5+c
        label="GOAL" if s==24 else "CRATER" if s in [8,12] else f"{visits[r,c]:,}"
        ax.text(c,r,label,ha="center",va="center",fontsize=10,
                color="white" if visits[r,c]>1800 else INK,weight="bold")
ax.add_patch(FancyBboxPatch((2.51,1.51),.98,.98,boxstyle="round,pad=0,rounding_size=0.04",
                          fill=False,edgecolor=ORANGE,linewidth=3))
ax.set_xticks([]);ax.set_yticks([])
for spine in ax.spines.values():spine.set_visible(False)
right=panel(fig,[.56,.2,.39,.56]);card(right,0,0,1,1)
right.text(.07,.84,"STATE 13",fontsize=14,color=ORANGE,weight="bold")
right.text(.07,.60,"16 visits",fontsize=32,weight="bold")
for y,label,value in [(.43,"DP reference",DATA["dp"][13]),(.29,"MC estimate",DATA["mc"][13]),(.15,"TD estimate",DATA["td"][13])]:
    right.text(.07,y,label,fontsize=14,color=MUTED)
    right.text(.91,y,f"{value:.2f}",ha="right",fontsize=18,weight="bold")
fig.text(.05,.06,"A good start-state estimate does not imply accurate values everywhere. Terminal states are not updated.",fontsize=11,color=MUTED)
save(fig,"coverage-blind-spot")
fig=canvas("The reinforcement-learning loop", "The policy chooses an action. The environment returns what actually happened.",height=6)
ax=panel(fig,[.05,.17,.90,.60])
card(ax,.025,.2,.32,.55)
card(ax,.66,.2,.32,.55)
ax.text(.185,.56,"AGENT",ha="center",fontsize=24,weight="bold",color=BLUE)
ax.text(.185,.40,"Chooses using a policy",ha="center",fontsize=12,color=MUTED)
ax.text(.82,.56,"ENVIRONMENT",ha="center",fontsize=21,weight="bold",color=GREEN)
ax.text(.82,.40,"Applies the world's rules",ha="center",fontsize=12,color=MUTED)
arrow(ax,(.36,.67),(.64,.67),BLUE)
ax.text(.50,.76,"Action",ha="center",fontsize=14,weight="bold",color=BLUE)
arrow(ax,(.64,.29),(.36,.29),GREEN)
ax.text(.50,.16,"Next observation + reward",ha="center",fontsize=12,color=GREEN)
fig.text(.05,.08,"In our rover, the observation is its cell ID. MC and TD use sampled transitions to update value estimates.",fontsize=11,color=MUTED)
save(fig,"rl-loop")
print("Created five graphics in PNG and SVG formats")

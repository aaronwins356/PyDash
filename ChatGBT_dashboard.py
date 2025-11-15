# dashboard.py
import streamlit as st
import numpy as np
import matplotlib.pyplot as plt

# ---------- PAGE CONFIG ----------
st.set_page_config(
    page_title="RL Visual Guide",
    layout="wide",
)

# ---------- GLOBAL STYLE ----------
st.markdown(
    """
    <style>
        @import url('https://fonts.googleapis.com/css2?family=Playfair+Display:wght@500;700&family=Inter:wght@300;400;600&display=swap');

        html, body, .stApp {
            background-color: #000000 !important;
            color: #FFFFFF !important;
        }

        /* Main content container full-width, but centered text blocks inside */
        .block-container {
            padding-top: 1.5rem;
            padding-bottom: 3rem;
            max-width: 1400px;
        }

        * {
            font-family: 'Inter', sans-serif;
        }

        .title-slide {
            text-align: center;
            margin-top: 1.5rem;
            margin-bottom: 2.5rem;
        }

        .title-slide h1 {
            font-family: 'Playfair Display', serif;
            font-size: 3.0rem;
            letter-spacing: 0.05em;
            margin-bottom: 0.4rem;
        }

        .title-slide h3 {
            font-weight: 300;
            font-size: 1.1rem;
            color: #bbbbbb;
        }

        .section {
            margin-top: 2.5rem;
            margin-bottom: 2.5rem;
        }

        .section-title {
            font-family: 'Playfair Display', serif;
            font-size: 2.0rem;
            margin-bottom: 0.8rem;
            text-align: left;
        }

        .section-subtitle {
            font-size: 1.0rem;
            color: #aaaaaa;
            margin-bottom: 1.2rem;
        }

        .card {
            border-radius: 14px;
            border: 1px solid #333333;
            padding: 1.2rem 1.4rem;
            background: #050505;
        }

        .card h3 {
            margin-top: 0;
            margin-bottom: 0.5rem;
        }

        .slide-divider {
            border-top: 1px solid #222222;
            margin: 1.5rem 0 1.5rem 0;
        }

        .big-divider {
            border-top: 1px solid #333333;
            margin: 3rem 0 1.5rem 0;
        }

        .center-note {
            text-align: center;
            font-size: 0.9rem;
            color: #bbbbbb;
            margin-top: 0.3rem;
        }

        .pill {
            border-radius: 999px;
            padding: 0.1rem 0.7rem;
            font-size: 0.75rem;
            border: 1px solid #555555;
            display: inline-block;
            margin-bottom: 0.35rem;
        }

        .ppo-pill { border-color: #4aa8ff; color: #4aa8ff; }
        .sac-pill { border-color: #3ad96b; color: #3ad96b; }
        .dqn-pill { border-color: #ffc857; color: #ffc857; }

        .summary-box {
            border-radius: 18px;
            border: 2px solid #e5c15a;
            padding: 1.8rem 2.5rem;
            background: #050505;
            max-width: 900px;
            margin: 0 auto;
        }

        .summary-box p {
            margin-bottom: 0.4rem;
        }

        .summary-title {
            font-family: 'Playfair Display', serif;
            font-size: 2.1rem;
            text-align: center;
            margin-bottom: 1.4rem;
        }

        .algo-title {
            font-family: 'Playfair Display', serif;
            font-size: 1.7rem;
            margin-bottom: 0.4rem;
        }

        .small-caption {
            font-size: 0.8rem;
            color: #aaaaaa;
            text-align: center;
            margin-top: 0.4rem;
        }

        .arrow-down {
            text-align: center;
            font-size: 1.2rem;
            color: #777777;
            margin-top: 0.6rem;
        }

        ul {
            margin-bottom: 0.3rem;
        }
    </style>
    """,
    unsafe_allow_html=True,
)

plt.style.use("dark_background")
np.random.seed(0)


# ---------- PLOTTING HELPERS ----------

def fig_dark(figsize=(6, 4)):
    return plt.subplots(figsize=figsize, dpi=160)


def plot_rl_loop():
    fig, ax = fig_dark((5, 3))
    ax.axis("off")

    # Boxes
    ax.add_patch(plt.Rectangle((0.1, 0.35), 0.25, 0.3,
                               edgecolor="#4aa8ff", facecolor="none", linewidth=2))
    ax.text(0.225, 0.5, "Agent", ha="center", va="center", fontsize=12)

    ax.add_patch(plt.Rectangle((0.65, 0.35), 0.25, 0.3,
                               edgecolor="#3ad96b", facecolor="none", linewidth=2))
    ax.text(0.775, 0.5, "Environment", ha="center", va="center", fontsize=12)

    # Arrows
    ax.annotate("action",
                xy=(0.35, 0.55), xytext=(0.65, 0.55),
                arrowprops=dict(arrowstyle="->", color="white", lw=1.5),
                ha="center", va="center", fontsize=10)

    ax.annotate("state, reward",
                xy=(0.65, 0.42), xytext=(0.35, 0.42),
                arrowprops=dict(arrowstyle="->", color="white", lw=1.5),
                ha="center", va="center", fontsize=10)

    return fig


def plot_learning_curve():
    steps = np.arange(0, 200)
    trend = 1 - np.exp(-steps / 80)
    noise = np.random.normal(scale=0.07, size=len(steps))
    rewards = trend + noise * (1.1 - trend)

    fig, ax = fig_dark((6, 3))
    ax.plot(steps, rewards, color="#4be0ff", linewidth=1.8)
    ax.set_xlabel("Training Steps")
    ax.set_ylabel("Average Reward")
    ax.set_title("Noisy Learning Curve")
    ax.grid(alpha=0.25)
    return fig


def plot_explore_exploit():
    steps = np.arange(0, 200)
    exploit = 0.4 + 0.7 * (1 - np.exp(-steps / 45))
    explore = 0.3 + 0.9 * (1 - np.exp(-steps / 110))

    fig, ax = fig_dark((6, 3))
    ax.plot(steps, exploit, label="Greedy / low exploration",
            color="#ffb347", linewidth=1.8)
    ax.plot(steps, explore, label="Balanced exploration",
            color="#3ad96b", linewidth=1.8)
    ax.set_xlabel("Training Steps")
    ax.set_ylabel("Cumulative Reward")
    ax.set_title("Exploration vs Exploitation")
    ax.legend()
    ax.grid(alpha=0.25)
    return fig


def plot_policy_shift():
    x = np.linspace(-2.5, 2.5, 400)
    pi_old = 1/np.sqrt(2*np.pi*0.6**2) * np.exp(-(x - 0.0)**2/(2*0.6**2))
    pi_new = 1/np.sqrt(2*np.pi*0.6**2) * np.exp(-(x - 0.7)**2/(2*0.6**2))

    fig, ax = fig_dark((6, 3.2))
    ax.plot(x, pi_old, "b--", label=r"$\pi_{\mathrm{old}}$", linewidth=1.7)
    ax.plot(x, pi_new, "r-", label=r"$\pi_{\mathrm{new}}$", linewidth=1.7)
    ax.fill_between(x, pi_old, pi_new, where=pi_new > pi_old,
                    color="red", alpha=0.18)
    ax.set_xlabel("Action Space")
    ax.set_ylabel("Probability")
    ax.set_title("Policy Shift (Policy Gradient)")
    ax.legend()
    ax.grid(alpha=0.25)
    return fig


def plot_value_selection():
    actions = ["A1", "A2", "A3", "A4"]
    q_vals = [2.1, 3.7, 1.1, 2.9]

    fig, ax = fig_dark((5.2, 3.2))
    colors = ["#4aa8ff", "#ffc857", "#ff6b6b", "#3ad96b"]
    bars = ax.bar(actions, q_vals, color=colors)
    best_idx = int(np.argmax(q_vals))
    bars[best_idx].set_edgecolor("#ffffff")
    bars[best_idx].set_linewidth(2)
    ax.set_ylabel("Q-Value")
    ax.set_title("Value-Based Selection (DQN-style)")
    ax.grid(axis="y", alpha=0.25)
    return fig


def plot_ppo_clipping():
    r = np.linspace(0.6, 2.0, 400)
    A = 1.0
    unclipped = r * A
    eps = 0.2
    clipped = np.clip(r, 1 - eps, 1 + eps) * A

    fig, ax = fig_dark((5.5, 3.3))
    ax.plot(r, unclipped, "--", color="#4aa8ff", label="Unclipped objective", linewidth=1.8)
    ax.plot(r, clipped, "-", color="#ff4b4b", label="PPO clipped objective", linewidth=1.8)
    ax.axvline(1 - eps, color="#ffc857", linestyle=":", linewidth=1.5)
    ax.axvline(1 + eps, color="#ffc857", linestyle=":", linewidth=1.5)
    ax.text(1 - eps, 0.6, " 1 - ε", color="#ffc857", rotation=90, va="bottom")
    ax.text(1 + eps, 0.6, " 1 + ε", color="#ffc857", rotation=90, va="bottom")
    ax.set_xlabel("Probability Ratio r = π_new / π_old")
    ax.set_ylabel("Scaled Objective (A > 0)")
    ax.set_title("PPO Clipping")
    ax.legend()
    ax.grid(alpha=0.25)
    return fig


def plot_entropy_distributions():
    x = np.linspace(-3, 3, 400)
    low = 1/np.sqrt(2*np.pi*0.55**2) * np.exp(-(x)**2/(2*0.55**2))
    high = 1/np.sqrt(2*np.pi*1.6**2) * np.exp(-(x)**2/(2*1.6**2))

    fig, ax = fig_dark((5.5, 3.3))
    ax.plot(x, low, color="#ff4b4b", label="Low Entropy (confident)", linewidth=1.8)
    ax.plot(x, high, color="#3ad96b", label="High Entropy (uncertain)", linewidth=1.8)
    ax.fill_between(x, low, alpha=0.15, color="#ff4b4b")
    ax.fill_between(x, high, alpha=0.15, color="#3ad96b")
    ax.set_xlabel("Action")
    ax.set_ylabel("Probability")
    ax.set_title("Entropy Regularization (SAC)")
    ax.legend()
    ax.grid(alpha=0.25)
    return fig


def plot_dqn_temporal_difference():
    t = np.arange(0, 10)
    true_values = 3 + 2 * np.sin(t / 2)
    approx = true_values + np.random.normal(scale=0.5, size=len(t))
    target = true_values + np.random.normal(scale=0.3, size=len(t))

    fig, ax = fig_dark((6, 3.2))
    ax.plot(t, true_values, label="True value", color="#3ad96b", linewidth=1.8)
    ax.plot(t, approx, "--", label="Q(s,a) prediction", color="#ffb347", linewidth=1.8)
    ax.plot(t, target, ":", label="Target value", color="#4aa8ff", linewidth=1.8)
    ax.set_xlabel("Time / Update")
    ax.set_ylabel("Value")
    ax.set_title("Temporal-Difference Targets (DQN)")
    ax.legend()
    ax.grid(alpha=0.25)
    return fig


# ---------- SLIDE 1: TITLE ----------
st.markdown(
    """
    <div class="title-slide">
        <h1>Reinforcement Learning: Visual Guide</h1>
        <h3>RL, PPO, SAC, and DQN — a cinematic walkthrough for your YouTube video</h3>
    </div>
    """,
    unsafe_allow_html=True,
)

st.markdown('<div class="big-divider"></div>', unsafe_allow_html=True)

# ---------- SLIDE 2: WHAT IS RL ----------
st.markdown('<div class="section">', unsafe_allow_html=True)
st.markdown('<div class="section-title">What is Reinforcement Learning?</div>', unsafe_allow_html=True)
st.markdown(
    '<div class="section-subtitle">An agent learns by trial and error to maximize long-term reward.</div>',
    unsafe_allow_html=True,
)

c1, c2 = st.columns([1.2, 1.0])

with c1:
    st.markdown(
        """
        - An **agent** interacts with an **environment**.
        - At each step it:
          - observes a **state**,
          - chooses an **action**,
          - receives a **reward**,
          - moves to a new state.
        - The goal: learn a **policy** that maximizes *cumulative discounted reward*.
        - Perfect for problems where:
          - there are no labeled “correct” actions,
          - actions have **consequences over time**,
          - and we can experiment to learn.
        """)
with c2:
    st.pyplot(plot_rl_loop(), use_container_width=True)
    st.markdown('<div class="center-note">The RL feedback loop: actions change future states and rewards.</div>',
                unsafe_allow_html=True)
st.markdown('</div>', unsafe_allow_html=True)

st.markdown('<div class="big-divider"></div>', unsafe_allow_html=True)

# ---------- SLIDE 3: CORE CONCEPTS ----------
st.markdown('<div class="section">', unsafe_allow_html=True)
st.markdown('<div class="section-title">Core Concepts</div>', unsafe_allow_html=True)

core1, core2, core3 = st.columns(3)

with core1:
    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.markdown("### Agent")
    st.markdown(
        """
        - The **decision-maker** in RL  
        - Receives observations / states  
        - Chooses actions via a **policy**  
        - Updates its policy from experience  
        - Often a neural network mapping states → actions
        """)
    st.markdown('</div>', unsafe_allow_html=True)

with core2:
    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.markdown("### Exploration")
    st.markdown(
        """
        - Trying actions whose outcomes are uncertain  
        - Prevents getting stuck in mediocre behavior  
        - Trades off with **exploitation** (using best-known action)  
        - Implemented via randomness, entropy bonuses, or ε-greedy rules
        """)
    st.markdown('</div>', unsafe_allow_html=True)

with core3:
    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.markdown("### Entropy")
    st.markdown(
        """
        - Measures **uncertainty** of a probability distribution  
        - High entropy → spread-out, exploratory policy  
        - Low entropy → sharp, confident policy  
        - SAC maximizes reward **plus** entropy to keep exploring
        """)
    st.markdown('</div>', unsafe_allow_html=True)

g1, g2 = st.columns(2)
with g1:
    st.pyplot(plot_entropy_distributions(), use_container_width=True)
with g2:
    st.pyplot(plot_explore_exploit(), use_container_width=True)

st.markdown('</div>', unsafe_allow_html=True)

st.markdown('<div class="big-divider"></div>', unsafe_allow_html=True)

# ---------- SLIDE 4: TANGENT CONCEPTS ----------
st.markdown('<div class="section">', unsafe_allow_html=True)
st.markdown('<div class="section-title">Tangent Concepts that Make RL Click</div>', unsafe_allow_html=True)

t1, t2 = st.columns(2)

with t1:
    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.markdown("#### Markov Decision Processes (MDPs)")
    st.markdown(
        """
        RL problems are often modeled as an MDP *(S, A, P, R, γ)*:

        - **S**: states  
        - **A**: actions  
        - **P**: transition dynamics  
        - **R**: reward function  
        - **γ**: discount factor  

        *Markov* → the future depends on the **current** state and action,
        not the entire history.
        """
    )
    st.markdown("#### Policies vs Value Functions")
    st.markdown(
        """
        - **Policy** π(a|s): how the agent acts  
        - **Value** V(s), Q(s,a): how good states / actions are  

        - Policy-based: learn π directly  
        - Value-based: learn Q and act greedily  
        - Actor-critic: learn **both**
        """)
    st.markdown('</div>', unsafe_allow_html=True)

with t2:
    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.markdown("#### On-policy vs Off-policy")
    st.markdown(
        """
        - **On-policy**: learn from data generated by the current policy  
          - Example: **PPO**  
        - **Off-policy**: learn from any behavior policy / replay buffer  
          - Examples: **DQN**, **SAC**

        Off-policy methods can reuse old data → more sample efficient.
        """
    )
    st.markdown("#### Advantages & Bellman Equation")
    st.markdown(
        """
        - **Advantage**: A(s,a) = Q(s,a) − V(s)  
          - How much better an action is than average.  

        - **Bellman equation** (for Q-learning):  
          Q(s,a) ≈ E[r + γ maxₐ′ Q(s′,a′)]

        DQN reduces the gap between current Q(s,a) and this target.
        """)
    st.markdown('</div>', unsafe_allow_html=True)

ld1, ld2 = st.columns(2)
with ld1:
    st.pyplot(plot_learning_curve(), use_container_width=True)
with ld2:
    st.pyplot(plot_dqn_temporal_difference(), use_container_width=True)

st.markdown('</div>', unsafe_allow_html=True)

st.markdown('<div class="big-divider"></div>', unsafe_allow_html=True)

# ---------- SLIDE 5: POLICY VS VALUE OPTIMIZATION ----------
st.markdown('<div class="section">', unsafe_allow_html=True)
st.markdown('<div class="section-title">Policy Optimization vs Value Optimization</div>', unsafe_allow_html=True)

pv1, pv2 = st.columns(2)
with pv1:
    st.pyplot(plot_policy_shift(), use_container_width=True)
    st.markdown(
        """
        **Policy-gradient approach**

        - Directly adjust policy parameters θ  
        - Learn a distribution π(a|s) over actions  
        - Very natural for **continuous action** spaces  
        - Used by PPO, SAC and other actor-critic methods
        """
    )
with pv2:
    st.pyplot(plot_value_selection(), use_container_width=True)
    st.markdown(
        """
        **Value-based approach**

        - Learn Q(s,a): how good is action *a* in state *s*  
        - Agent picks action with highest Q(s,a)  
        - Naturally suited to **discrete actions**  
        - Used by DQN and its many variants
        """
    )
st.markdown('</div>', unsafe_allow_html=True)

st.markdown('<div class="big-divider"></div>', unsafe_allow_html=True)

# ---------- SLIDE 6: PPO ----------
st.markdown('<div class="section">', unsafe_allow_html=True)
st.markdown('<span class="ppo-pill pill">PPO</span>', unsafe_allow_html=True)
st.markdown('<div class="algo-title">Proximal Policy Optimization</div>', unsafe_allow_html=True)

a1, a2 = st.columns([1.1, 1.0])
with a1:
    st.markdown(
        """
        - On-policy **actor-critic** algorithm  
        - Uses a **clipped surrogate objective** to keep updates safe  
        - Multiple epochs on the same batch of data  
        - Works well in many environments (games, simulators)  
        - Simple to implement with standard optimizers  
        """)
with a2:
    st.pyplot(plot_ppo_clipping(), use_container_width=True)
    st.markdown('<div class="small-caption">Clipping prevents destructive jumps in action probabilities.</div>',
                unsafe_allow_html=True)

st.markdown('</div>', unsafe_allow_html=True)

st.markdown('<div class="big-divider"></div>', unsafe_allow_html=True)

# ---------- SLIDE 7: SAC ----------
st.markdown('<div class="section">', unsafe_allow_html=True)
st.markdown('<span class="sac-pill pill">SAC</span>', unsafe_allow_html=True)
st.markdown('<div class="algo-title">Soft Actor-Critic</div>', unsafe_allow_html=True)

s1, s2 = st.columns([1.1, 1.0])
with s1:
    st.markdown(
        """
        - Off-policy **maximum entropy** RL  
        - Policy outputs mean & std of a Gaussian over actions  
        - Twin Q-networks reduce overestimation  
        - Objective: maximize reward **plus** entropy  
        - Strong baseline for continuous control & robotics
        """)
with s2:
    st.pyplot(plot_entropy_distributions(), use_container_width=True)
    st.markdown('<div class="small-caption">Entropy bonus keeps the policy broad and exploratory.</div>',
                unsafe_allow_html=True)

st.markdown('</div>', unsafe_allow_html=True)

st.markdown('<div class="big-divider"></div>', unsafe_allow_html=True)

# ---------- SLIDE 8: DQN ----------
st.markdown('<div class="section">', unsafe_allow_html=True)
st.markdown('<span class="dqn-pill pill">DQN</span>', unsafe_allow_html=True)
st.markdown('<div class="algo-title">Deep Q-Network</div>', unsafe_allow_html=True)

d1, d2 = st.columns([1.1, 1.0])
with d1:
    st.markdown(
        """
        - Value-based deep RL for **discrete actions**  
        - Learns Q(s,a) using the **Bellman update**  
        - Uses **experience replay** to break correlation  
        - Uses a **target network** for stability  
        - Famous for mastering Atari games from pixels
        """)
with d2:
    st.pyplot(plot_value_selection(), use_container_width=True)
    st.markdown('<div class="small-caption">Highest Q-value action is selected at each step.</div>',
                unsafe_allow_html=True)

st.markdown('</div>', unsafe_allow_html=True)

st.markdown('<div class="big-divider"></div>', unsafe_allow_html=True)

# ---------- FINAL SLIDE: SUMMARY ----------
st.markdown('<div class="section">', unsafe_allow_html=True)
st.markdown('<div class="summary-title">The Landscape of Modern RL</div>', unsafe_allow_html=True)

st.markdown(
    """
    <div class="summary-box">
        <p style="text-align:center;">
            <span style="color:#4aa8ff;">PPO</span> — stable on-policy learning via clipped policy gradients.<br>
            <span style="color:#3ad96b;">SAC</span> — entropy-regularized, off-policy control for continuous actions.<br>
            <span style="color:#ffc857;">DQN</span> — deep value learning for discrete action spaces.
        </p>
        <p style="text-align:center; margin-top:0.8rem;">
            Each algorithm tackles exploration, stability, and delayed reward in a different way,<br>
            but all aim for the same goal: <b>intelligent behavior through trial and error.</b>
        </p>
        <p style="text-align:center; margin-top:0.9rem; font-size:0.9rem; color:#bbbbbb;">
            Next steps for your series: Actor-Critic methods, TRPO, TD3, distributional RL, and transformer-based agents.
        </p>
    </div>
    """,
    unsafe_allow_html=True,
)

st.markdown('</div>', unsafe_allow_html=True)

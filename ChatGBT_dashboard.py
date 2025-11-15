import streamlit as st
import numpy as np
import matplotlib.pyplot as plt

# ---------- PAGE CONFIG ----------
st.set_page_config(
    page_title="RL Mind-Map Dashboard",
    layout="wide",
)

# ---------- GLOBAL STYLE ----------
st.markdown(
    """
    <style>
        @import url('https://fonts.googleapis.com/css2?family=Playfair+Display:wght@500;700&family=Permanent+Marker&display=swap');

        html, body, [class*="css"] {
            background-color: #000000 !important;
            color: #FFFFFF !important;
        }

        .main-title {
            font-family: 'Playfair Display', serif;
            font-size: 3.0rem;
            text-align: center;
            margin-top: 0.5rem;
            margin-bottom: 0.2rem;
        }

        .subtitle {
            font-family: 'Playfair Display', serif;
            font-size: 1.2rem;
            text-align: center;
            color: #dddddd;
            margin-bottom: 2rem;
        }

        .section-title {
            font-family: 'Playfair Display', serif;
            font-size: 1.6rem;
            margin-top: 2rem;
            margin-bottom: 0.5rem;
        }

        .script-title {
            font-family: 'Permanent Marker', cursive;
            font-size: 2.0rem;
            text-align: center;
            margin-top: 3rem;
            margin-bottom: 1rem;
        }

        .card {
            border-radius: 12px;
            border: 1px solid #444444;
            padding: 1.0rem 1.3rem;
            margin-bottom: 1.5rem;
            background: #050505;
        }

        .highlight-card {
            border-radius: 16px;
            border: 2px solid #e5c15a;
            padding: 1.5rem 2rem;
            margin: 2rem auto;
            max-width: 900px;
            background: rgba(10, 10, 10, 0.9);
        }

        .pill {
            border-radius: 999px;
            padding: 0.2rem 0.8rem;
            font-size: 0.8rem;
            border: 1px solid #555555;
            display: inline-block;
            margin-bottom: 0.4rem;
        }

        .algo-pill-ppo { border-color: #4aa8ff; }
        .algo-pill-sac { border-color: #67e37f; }
        .algo-pill-dqn { border-color: #ffc857; }

        .small-caption {
            font-size: 0.8rem;
            color: #bbbbbb;
            text-align: center;
            margin-top: 0.4rem;
        }

        .arrow-down {
            font-size: 1.4rem;
            text-align: center;
            margin-top: 0.6rem;
            margin-bottom: 0.3rem;
        }

        hr {
            border: none;
            border-top: 1px solid #333333;
            margin: 2.5rem 0 1.5rem 0;
        }
    </style>
    """,
    unsafe_allow_html=True,
)

plt.style.use("dark_background")
np.random.seed(0)


# ---------- PLOTTING HELPERS ----------

def plot_rl_loop():
    fig, ax = plt.subplots(figsize=(4, 2.4))
    ax.axis("off")

    # Agent box
    ax.add_patch(plt.Rectangle((0.1, 0.3), 0.25, 0.4,
                               edgecolor="#4aa8ff", facecolor="none", linewidth=2))
    ax.text(0.225, 0.5, "Agent", ha="center", va="center", fontsize=11)

    # Env box
    ax.add_patch(plt.Rectangle((0.65, 0.3), 0.25, 0.4,
                               edgecolor="#67e37f", facecolor="none", linewidth=2))
    ax.text(0.775, 0.5, "Environment", ha="center", va="center", fontsize=11)

    # Action arrow
    ax.annotate("action",
                xy=(0.35, 0.55), xytext=(0.65, 0.55),
                arrowprops=dict(arrowstyle="->", color="white"),
                ha="center", va="center", fontsize=9)

    # State/reward arrow
    ax.annotate("state, reward",
                xy=(0.65, 0.42), xytext=(0.35, 0.42),
                arrowprops=dict(arrowstyle="->", color="white"),
                ha="center", va="center", fontsize=9)

    return fig


def plot_learning_curve():
    steps = np.arange(0, 200)
    true_trend = 1 - np.exp(-steps / 70)
    noise = np.random.normal(scale=0.08, size=len(steps))
    rewards = true_trend + noise * (1.1 - true_trend)

    fig, ax = plt.subplots(figsize=(5, 3))
    ax.plot(steps, rewards, linewidth=1.5)
    ax.set_xlabel("Training Steps")
    ax.set_ylabel("Average Reward")
    ax.set_title("Noisy Learning Curve")
    ax.grid(alpha=0.2)
    return fig


def plot_explore_exploit():
    steps = np.arange(0, 200)
    exploit = 0.4 + 0.7 * (1 - np.exp(-steps / 50))
    explore = 0.2 + 0.9 * (1 - np.exp(-steps / 110))

    fig, ax = plt.subplots(figsize=(5, 3))
    ax.plot(steps, exploit, label="Greedy / low exploration")
    ax.plot(steps, explore, label="Balanced exploration")
    ax.set_xlabel("Training Steps")
    ax.set_ylabel("Cumulative Reward")
    ax.set_title("Exploration vs Exploitation")
    ax.legend()
    ax.grid(alpha=0.2)
    return fig


def plot_policy_shift():
    x = np.linspace(-2.5, 2.5, 400)
    pi_old = 1/np.sqrt(2*np.pi*0.6**2) * np.exp(-(x - 0.0)**2/(2*0.6**2))
    pi_new = 1/np.sqrt(2*np.pi*0.6**2) * np.exp(-(x - 0.6)**2/(2*0.6**2))

    fig, ax = plt.subplots(figsize=(5, 3))
    ax.plot(x, pi_old, "b--", label=r"$\pi_{\mathrm{old}}$")
    ax.plot(x, pi_new, "r-", label=r"$\pi_{\mathrm{new}}$")
    ax.fill_between(x, pi_old, pi_new, where=pi_new>pi_old,
                    color="red", alpha=0.15)
    ax.set_xlabel("Action Space")
    ax.set_ylabel("Probability")
    ax.set_title("Policy Shift (Policy Gradient)")
    ax.legend()
    ax.grid(alpha=0.2)
    return fig


def plot_value_selection():
    actions = ["A1", "A2", "A3", "A4"]
    q_vals = [2.1, 3.7, 1.0, 2.9]

    fig, ax = plt.subplots(figsize=(4.5, 3))
    bars = ax.bar(actions, q_vals)
    best_idx = int(np.argmax(q_vals))
    bars[best_idx].set_edgecolor("#ffc857")
    bars[best_idx].set_linewidth(2)
    ax.set_ylabel("Q-Value")
    ax.set_title("Value-Based Selection (DQN-style)")
    ax.grid(axis="y", alpha=0.2)
    return fig


def plot_ppo_clipping():
    r = np.linspace(0.6, 2.0, 400)
    A = 1.0
    unclipped = r * A
    eps = 0.2
    clipped = np.clip(r, 1 - eps, 1 + eps) * A

    fig, ax = plt.subplots(figsize=(4.5, 3))
    ax.plot(r, unclipped, "--", label="Unclipped objective")
    ax.plot(r, clipped, "r", label="PPO clipped objective")
    ax.axvline(1 - eps, color="orange", linestyle=":")
    ax.axvline(1 + eps, color="orange", linestyle=":")
    ax.text(1 - eps, 0.5, "1 - ε", color="orange", rotation=90, va="bottom")
    ax.text(1 + eps, 0.5, "1 + ε", color="orange", rotation=90, va="bottom")
    ax.set_xlabel("Probability Ratio r = π_new / π_old")
    ax.set_ylabel("Scaled Objective (A > 0)")
    ax.set_title("PPO Clipping")
    ax.legend()
    ax.grid(alpha=0.2)
    return fig


def plot_entropy_distributions():
    x = np.linspace(-3, 3, 400)
    low = 1/np.sqrt(2*np.pi*0.6**2) * np.exp(-(x)**2/(2*0.6**2))
    high = 1/np.sqrt(2*np.pi*1.5**2) * np.exp(-(x)**2/(2*1.5**2))

    fig, ax = plt.subplots(figsize=(4.5, 3))
    ax.plot(x, low, "r", label="Low Entropy (confident)")
    ax.plot(x, high, "g", label="High Entropy (uncertain)")
    ax.fill_between(x, low, alpha=0.1, color="red")
    ax.fill_between(x, high, alpha=0.1, color="green")
    ax.set_xlabel("Action")
    ax.set_ylabel("Probability")
    ax.set_title("Entropy Regularization (SAC)")
    ax.legend()
    ax.grid(alpha=0.2)
    return fig


def plot_dqn_temporal_difference():
    t = np.arange(0, 10)
    true_values = 3 + 2*np.sin(t/2)
    approx = true_values + np.random.normal(scale=0.4, size=len(t))
    target = true_values + np.random.normal(scale=0.2, size=len(t))

    fig, ax = plt.subplots(figsize=(4.8, 3))
    ax.plot(t, true_values, label="True value", linewidth=1.5)
    ax.plot(t, approx, "--", label="Q(s,a) prediction")
    ax.plot(t, target, ":", label="Target value", linewidth=1.5)
    ax.set_xlabel("Time / Update")
    ax.set_ylabel("Value")
    ax.set_title("Temporal-Difference Targets (DQN)")
    ax.legend()
    ax.grid(alpha=0.2)
    return fig


# ---------- TITLE ----------
st.markdown('<div class="main-title">Reinforcement Learning Mind-Map</div>', unsafe_allow_html=True)
st.markdown('<div class="subtitle">RL, PPO, SAC, and DQN — a visual overview for teaching and storytelling</div>', unsafe_allow_html=True)

# ---------- INTRO: WHAT IS RL ----------
st.markdown('<div class="section-title">What is Reinforcement Learning and why is it useful?</div>', unsafe_allow_html=True)

col_intro_left, col_intro_right = st.columns([1.2, 1.0])

with col_intro_left:
    st.markdown(
        """
        In **Reinforcement Learning (RL)**, an *agent* learns to make a sequence of decisions
        by interacting with an **environment**.  
        
        The agent:
        - observes a **state** of the world,
        - chooses an **action**,
        - receives a **reward** signal,
        - and transitions to a new state.

        Over time, the goal is to learn a **policy** – a mapping from states to actions –
        that maximizes **long-term cumulative reward**.

        RL is useful when:
        - we don’t have labeled examples of “correct” actions,
        - actions have **consequences over time**,
        - and we can try things, observe outcomes, and improve.
        
        **Examples:**
        - Game-playing agents (Atari, Go, Dota)
        - Robotics and control
        - Recommendation systems and advertising
        - Finance and trading strategies
        """
    )

with col_intro_right:
    fig_loop = plot_rl_loop()
    st.pyplot(fig_loop)
    st.caption("The RL feedback loop: the agent learns from delayed consequences of its own actions.")

st.markdown("---")

# ---------- DEFINITIONS: AGENT / ENTROPY / EXPLORATION ----------
st.markdown('<div class="section-title">Core Definitions</div>', unsafe_allow_html=True)

d1, d2, d3 = st.columns(3)

with d1:
    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.markdown("### Agent")
    st.markdown(
        """
        The **agent** is the decision-maker.

        - Receives observations or states from the environment  
        - Chooses actions using a **policy**  
        - Updates its policy based on rewards and experience  

        In code, the agent is usually a neural network that outputs
        action probabilities or value estimates.
        """
    )
    st.markdown('</div>', unsafe_allow_html=True)

with d2:
    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.markdown("### Entropy")
    st.markdown(
        """
        **Entropy** measures how *uncertain* or *spread out* a probability distribution is.

        - **High entropy** → the policy is unsure and explores many actions  
        - **Low entropy** → the policy is confident and chooses a few actions often  

        Algorithms like **SAC** explicitly **maximize entropy** to encourage exploration:
        they prefer policies that earn reward *and* stay suitably uncertain.
        """
    )
    st.markdown('</div>', unsafe_allow_html=True)

with d3:
    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.markdown("### Exploration")
    st.markdown(
        """
        **Exploration** is the process of trying actions whose outcomes are not yet well known.

        - Prevents the agent from getting stuck in mediocre behaviors  
        - Trades off against **exploitation**, where we choose the best-known action  
        - Often implemented via randomness, entropy bonuses, or ε-greedy strategies  

        Good RL algorithms carefully balance **explore vs exploit** over time.
        """
    )
    st.markdown('</div>', unsafe_allow_html=True)

g1, g2 = st.columns(2)
with g1:
    st.pyplot(plot_entropy_distributions())
    st.caption("High entropy spreads probability mass across actions → better early exploration.")
with g2:
    st.pyplot(plot_explore_exploit())
    st.caption("Greedy policies learn fast but can plateau; exploration can lead to better long-term reward.")

st.markdown("---")

# ---------- TANGENT CONCEPTS ----------
st.markdown('<div class="section-title">Tangent Concepts that Make RL Click</div>', unsafe_allow_html=True)

tc1, tc2 = st.columns(2)

with tc1:
    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.markdown("#### Markov Decision Processes (MDPs)")
    st.markdown(
        """
        RL problems are often modeled as **MDPs**:  
        \nA tuple *(S, A, P, R, γ)* where:
        - **S** – set of states  
        - **A** – set of actions  
        - **P** – transition dynamics  
        - **R** – reward function  
        - **γ** – discount factor for future rewards  

        *Markov* means: the future depends on the current state and action,
        not on the entire past history.
        """
    )

    st.markdown("#### Policies vs Value Functions")
    st.markdown(
        """
        - A **policy** π(a|s) tells us **what to do**.  
        - A **value function** V(s) or Q(s,a) tells us **how good it is**.  

        - **Policy-based** methods: directly learn π.  
        - **Value-based** methods: learn Q, then choose greedy actions.  
        - **Actor-critic**: learn **both**.
        """
    )
    st.markdown('</div>', unsafe_allow_html=True)

with tc2:
    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.markdown("#### On-policy vs Off-policy")
    st.markdown(
        """
        - **On-policy**: learn from data generated by the *current* policy  
          - Example: **PPO**  
        - **Off-policy**: learn from data generated by *any* behavior policy  
          - Examples: **DQN**, **SAC**

        Off-policy methods can re-use old data (via replay buffers),
        which can be much more **sample-efficient**.
        """
    )

    st.markdown("#### Advantages and Bellman Equations")
    st.markdown(
        """
        The **advantage** function  
        \n*A(s,a) = Q(s,a) − V(s)*  
        tells us how much better an action is than average.

        The **Bellman equation** links values recursively:  
        \n*Q(s,a) = E[r + γ max<sub>a'</sub> Q(s',a')]*  

        DQN and many value-based methods learn by reducing the gap between
        current Q(s,a) estimates and these **targets**.
        """
    )
    st.markdown('</div>', unsafe_allow_html=True)

lc1, lc2 = st.columns(2)
with lc1:
    st.pyplot(plot_learning_curve())
with lc2:
    st.pyplot(plot_dqn_temporal_difference())

st.markdown("---")

# ---------- POLICY VS VALUE OPTIMIZATION ----------
st.markdown('<div class="section-title">Policy Optimization vs Value Optimization</div>', unsafe_allow_html=True)

upper_box = st.container()
with upper_box:
    col_p, col_v = st.columns(2)
    with col_p:
        st.pyplot(plot_policy_shift())
        st.markdown(
            """
            **Policy-gradient approach**

            - Directly adjusts the policy parameters  
            - Learns **probability distributions** over actions  
            - Ideal for **continuous action spaces**  
            - Examples: PPO, SAC
            """
        )

    with col_v:
        st.pyplot(plot_value_selection())
        st.markdown(
            """
            **Value-based approach**

            - Learns **Q-values**: how good is action *a* in state *s*?  
            - Agent chooses action with largest Q(s,a)  
            - Naturally suited to **discrete actions**  
            - Examples: DQN and its variants
            """
        )

st.markdown("---")

# ---------- THREE PILLARS SECTIONS ----------
st.markdown('<div class="section-title">Three Pillars of Modern Deep RL</div>', unsafe_allow_html=True)

top_cards = st.columns(3)

with top_cards[0]:
    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.markdown('<span class="pill algo-pill-ppo">PPO</span>', unsafe_allow_html=True)
    st.markdown("### Proximal Policy Optimization")
    st.markdown(
        """
        - On-policy **actor-critic** method  
        - Uses a **clipped surrogate objective** to prevent destructive updates  
        - Multiple epochs on the same mini-batches  
        - Widely used in practice (games, simulators)
        """
    )
    st.markdown('</div>', unsafe_allow_html=True)

with top_cards[1]:
    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.markdown('<span class="pill algo-pill-sac">SAC</span>', unsafe_allow_html=True)
    st.markdown("### Soft Actor-Critic")
    st.markdown(
        """
        - Off-policy **maximum entropy** RL  
        - Stochastic Gaussian policy (mean & std)  
        - Twin Q-networks reduce overestimation  
        - Very strong for continuous control and robotics
        """
    )
    st.markdown('</div>', unsafe_allow_html=True)

with top_cards[2]:
    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.markdown('<span class="pill algo-pill-dqn">DQN</span>', unsafe_allow_html=True)
    st.markdown("### Deep Q-Network")
    st.markdown(
        """
        - Value-based method for **discrete actions**  
        - Uses **experience replay** and a **target network**  
        - Learned to play Atari directly from pixels  
        - Foundation for many modern Q-learning variants
        """
    )
    st.markdown('</div>', unsafe_allow_html=True)

st.markdown('<div class="arrow-down">↓</div>', unsafe_allow_html=True)

algo_row = st.columns(3)

with algo_row[0]:
    st.pyplot(plot_ppo_clipping())
    st.caption("PPO clips probability ratios to keep policy updates in a safe range.")

with algo_row[1]:
    st.pyplot(plot_entropy_distributions())
    st.caption("SAC trades reward for entropy: it prefers policies that are both good and uncertain.")

with algo_row[2]:
    st.pyplot(plot_value_selection())
    st.caption("DQN estimates Q(s,a) and chooses the action with highest predicted value.")

st.markdown("---")

# ---------- CONCLUSION ----------
st.markdown('<div class="script-title">The Landscape of Modern RL</div>', unsafe_allow_html=True)

st.markdown(
    """
    <div class="highlight-card">
        <p style="text-align:center;">
            <span style="color:#4aa8ff;">PPO</span> for stable on-policy learning,<br>
            <span style="color:#67e37f;">SAC</span> for sample-efficient continuous control,<br>
            <span style="color:#ffc857;">DQN</span> for powerful value-based learning in discrete action spaces.
        </p>
        <p style="text-align:center; margin-top:0.7rem;">
            Each algorithm tackles RL's core challenges — exploration, instability, and delayed reward —<br>
            but all aim for the same outcome: <b>intelligent behavior through trial and error.</b>
        </p>
        <p style="text-align:center; margin-top:0.7rem; font-size:0.9rem; color:#bbbbbb;">
            → Natural next topics: Actor-Critic methods, TRPO, TD3, distributional RL, and transformer-based agents.
        </p>
    </div>
    """,
    unsafe_allow_html=True,
)

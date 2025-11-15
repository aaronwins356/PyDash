"""
Comprehensive Reinforcement Learning Dashboard
==============================================
A professional, dark-mode visual storyboard for deeply understanding modern RL algorithms:
PPO, SAC, and DQN with extensive educational content and high-quality visualizations.

This dashboard covers:
- What is RL and why it matters
- Core concepts: Agent, Entropy, Exploration
- Deep dives into PPO, SAC, and DQN
- Tangent concepts: MDPs, Value Functions, Advantage Estimation, etc.
- High-quality visualizations throughout
"""

import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.colors import LinearSegmentedColormap

# ============================================================================
# PAGE CONFIGURATION
# ============================================================================

st.set_page_config(
    page_title="Comprehensive RL Dashboard",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# Set matplotlib defaults for high quality
plt.rcParams['figure.dpi'] = 150
plt.rcParams['savefig.dpi'] = 150
plt.rcParams['figure.facecolor'] = 'black'
plt.rcParams['axes.facecolor'] = 'black'

# ============================================================================
# CUSTOM CSS - ENHANCED DARK MODE THEME
# ============================================================================

st.markdown("""
<style>
    /* Global dark background */
    .stApp {
        background-color: #000000;
        color: #FFFFFF;
    }
    
    [data-testid="stSidebar"] {
        background-color: #1a1a1a;
    }
    
    .stMarkdown, p, span, div, h1, h2, h3, h4, h5, h6 {
        color: #FFFFFF !important;
    }
    
    /* Handwritten-style font for titles */
    .handwritten-title {
        font-family: 'Brush Script MT', 'Comic Sans MS', cursive;
        font-size: 48px;
        font-weight: bold;
        color: #FFFFFF;
        margin: 20px 0;
        text-align: center;
        text-shadow: 2px 2px 4px rgba(255,255,255,0.15);
    }
    
    .handwritten-section {
        font-family: 'Brush Script MT', 'Comic Sans MS', cursive;
        font-size: 36px;
        font-weight: bold;
        color: #FFFFFF;
        margin: 30px 0 20px 0;
        text-align: center;
    }
    
    .handwritten-subtitle {
        font-family: 'Brush Script MT', 'Comic Sans MS', cursive;
        font-size: 26px;
        color: #FFFFFF;
        margin-bottom: 15px;
        text-align: center;
    }
    
    /* Content blocks */
    .intro-section {
        background-color: #0a0a0a;
        border: 3px solid #FFFFFF;
        border-radius: 15px;
        padding: 35px;
        margin: 25px 0;
        box-shadow: 0 0 25px rgba(255,255,255,0.1);
    }
    
    .definition-block {
        background-color: #1a1a1a;
        border: 2px solid #FFFFFF;
        border-radius: 12px;
        padding: 25px;
        margin: 10px;
        min-height: 420px;
        box-shadow: 0 0 15px rgba(255,255,255,0.05);
    }
    
    .algo-block {
        background-color: #1a1a1a;
        border: 2px solid #FFFFFF;
        border-radius: 12px;
        padding: 28px;
        margin: 10px;
        min-height: 550px;
        box-shadow: 0 0 15px rgba(255,255,255,0.05);
    }
    
    .info-box {
        background-color: #0a1a2a;
        border: 2px solid #4DABF7;
        border-left: 5px solid #4DABF7;
        border-radius: 8px;
        padding: 20px;
        margin: 15px 0;
    }
    
    .mini-card {
        background-color: #2a2a2a;
        border-left: 4px solid #4DABF7;
        padding: 15px;
        margin: 12px 0;
        border-radius: 5px;
    }
    
    .floating-concept {
        background: linear-gradient(135deg, #1a1a1a 0%, #2a2a2a 100%);
        border: 1px solid #555;
        border-radius: 8px;
        padding: 18px;
        margin: 12px 0;
    }
    
    .conclusion-panel {
        background: linear-gradient(135deg, #0a0a0a 0%, #1a1a1a 100%);
        border: 3px solid #FFFFFF;
        border-radius: 15px;
        padding: 40px;
        margin: 30px 0;
        box-shadow: 0 0 30px rgba(255,255,255,0.15);
    }
    
    /* Colors */
    .red-text { color: #FF6B6B !important; }
    .green-text { color: #51CF66 !important; }
    .yellow-text { color: #FFD700 !important; }
    .blue-text { color: #4DABF7 !important; }
    .orange-text { color: #FF9F43 !important; }
    .purple-text { color: #B794F6 !important; }
    .cyan-text { color: #20C997 !important; }
    
    ul { color: #FFFFFF; }
    li { color: #FFFFFF; margin: 8px 0; line-height: 1.7; }
    
    .arrow-down {
        text-align: center;
        font-size: 55px;
        color: #FFFFFF;
        margin: 25px 0;
    }
    
    .divider-line {
        height: 2px;
        background: linear-gradient(to right, transparent, #FFFFFF, transparent);
        margin: 35px 0;
    }
</style>
""", unsafe_allow_html=True)

# ============================================================================
# VISUALIZATION FUNCTIONS
# ============================================================================

def create_rl_loop_diagram():
    """Enhanced RL loop diagram."""
    fig, ax = plt.subplots(figsize=(9, 5.5), facecolor='black')
    ax.set_facecolor('black')
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 6.5)
    ax.axis('off')
    
    # Agent box
    agent_box = patches.FancyBboxPatch((0.5, 2.5), 2.8, 1.8, linewidth=3, 
                                       edgecolor='#4DABF7', facecolor='#1a1a1a',
                                       boxstyle="round,pad=0.1")
    ax.add_patch(agent_box)
    ax.text(1.9, 3.4, 'AGENT', ha='center', va='center', color='#4DABF7', 
            fontsize=15, weight='bold')
    ax.text(1.9, 2.9, 'π(a|s)', ha='center', color='white', 
            fontsize=11, style='italic', alpha=0.8)
    
    # Environment box
    env_box = patches.FancyBboxPatch((6.7, 2.5), 2.8, 1.8, linewidth=3, 
                                     edgecolor='#51CF66', facecolor='#1a1a1a',
                                     boxstyle="round,pad=0.1")
    ax.add_patch(env_box)
    ax.text(8.1, 3.4, 'ENVIRONMENT', ha='center', va='center', color='#51CF66', 
            fontsize=13, weight='bold')
    
    # Action arrow
    ax.annotate('', xy=(6.7, 3.7), xytext=(3.3, 3.7),
                arrowprops=dict(arrowstyle='->', color='#FFD700', lw=3.5,
                               connectionstyle="arc3,rad=0.3"))
    ax.text(5, 4.6, 'action (a)', ha='center', color='#FFD700', 
            fontsize=12, weight='bold')
    
    # State + Reward arrow
    ax.annotate('', xy=(3.3, 2.9), xytext=(6.7, 2.9),
                arrowprops=dict(arrowstyle='->', color='#FF6B6B', lw=3.5,
                               connectionstyle="arc3,rad=0.3"))
    ax.text(5, 1.8, 'state (s), reward (r)', ha='center', color='#FF6B6B', 
            fontsize=12, weight='bold')
    
    # Goal label
    ax.text(5, 6, 'Goal: Maximize cumulative reward Σ γᵗ·r_t', ha='center', 
            color='white', fontsize=13, weight='bold',
            bbox=dict(boxstyle='round', facecolor='#2a2a2a', edgecolor='white', pad=0.6))
    
    plt.tight_layout()
    return fig

def create_mdp_diagram():
    """MDP state transition diagram."""
    fig, ax = plt.subplots(figsize=(8, 5), facecolor='black')
    ax.set_facecolor('black')
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 6)
    ax.axis('off')
    
    # States
    states = [(2, 4), (5, 4), (8, 4), (5, 2)]
    colors = ['#4DABF7', '#51CF66', '#FFD700', '#FF6B6B']
    labels = ['s₁', 's₂', 's₃', 's₄']
    
    for i, ((x, y), color, label) in enumerate(zip(states, colors, labels)):
        circle = patches.Circle((x, y), 0.5, linewidth=2, 
                                edgecolor=color, facecolor='#1a1a1a')
        ax.add_patch(circle)
        ax.text(x, y, label, ha='center', va='center', color='white', 
                fontsize=13, weight='bold')
    
    # Transitions
    transitions = [(0, 1, '+2'), (1, 2, '+5'), (1, 3, '-1'), (3, 1, '+1')]
    for start, end, reward in transitions:
        start_pos = states[start]
        end_pos = states[end]
        ax.annotate('', xy=end_pos, xytext=start_pos,
                    arrowprops=dict(arrowstyle='->', color='white', lw=1.5,
                                   connectionstyle="arc3,rad=0.3"))
        mid_x = (start_pos[0] + end_pos[0]) / 2
        mid_y = (start_pos[1] + end_pos[1]) / 2
        ax.text(mid_x, mid_y + 0.3, reward, ha='center', 
                color='#FFD700', fontsize=9, weight='bold',
                bbox=dict(boxstyle='round', facecolor='#2a2a2a', alpha=0.8))
    
    ax.text(5, 5.5, 'Markov Decision Process', ha='center', color='white', 
            fontsize=14, weight='bold')
    
    plt.tight_layout()
    return fig

def create_entropy_comparison():
    """High vs low entropy distributions."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4.5), facecolor='black')
    
    x = np.linspace(-4, 4, 250)
    
    # Low entropy
    ax1.set_facecolor('black')
    y_low = np.exp(-3.5*x**2) / np.sqrt(np.pi/3.5)
    ax1.plot(x, y_low, 'r-', linewidth=3.5, label='Low Entropy')
    ax1.fill_between(x, y_low, alpha=0.35, color='red')
    ax1.axvline(x=0, color='#FFD700', linestyle='--', linewidth=2.5, alpha=0.8)
    ax1.set_xlabel('Action Space', color='white', fontsize=12, weight='bold')
    ax1.set_ylabel('Probability Density', color='white', fontsize=12, weight='bold')
    ax1.set_title('Low Entropy → Exploitation', color='#FF6B6B', 
                  fontsize=14, weight='bold')
    ax1.tick_params(colors='white', labelsize=10)
    for spine in ['bottom', 'left']:
        ax1.spines[spine].set_color('white')
    ax1.spines['top'].set_visible(False)
    ax1.spines['right'].set_visible(False)
    ax1.grid(True, alpha=0.25, color='white')
    ax1.text(0, max(y_low)*0.6, 'Peaked\\nDeterministic', ha='center', 
             color='white', fontsize=11, weight='bold',
             bbox=dict(boxstyle='round', facecolor='#2a2a2a', alpha=0.9, pad=0.5))
    
    # High entropy
    ax2.set_facecolor('black')
    y_high = np.exp(-0.35*x**2) / np.sqrt(np.pi/0.35)
    ax2.plot(x, y_high, 'g-', linewidth=3.5, label='High Entropy')
    ax2.fill_between(x, y_high, alpha=0.35, color='green')
    ax2.set_xlabel('Action Space', color='white', fontsize=12, weight='bold')
    ax2.set_ylabel('Probability Density', color='white', fontsize=12, weight='bold')
    ax2.set_title('High Entropy → Exploration', color='#51CF66', 
                  fontsize=14, weight='bold')
    ax2.tick_params(colors='white', labelsize=10)
    for spine in ['bottom', 'left']:
        ax2.spines[spine].set_color('white')
    ax2.spines['top'].set_visible(False)
    ax2.spines['right'].set_visible(False)
    ax2.grid(True, alpha=0.25, color='white')
    ax2.text(0, max(y_high)*0.6, 'Flat\\nStochastic', ha='center', 
             color='white', fontsize=11, weight='bold',
             bbox=dict(boxstyle='round', facecolor='#2a2a2a', alpha=0.9, pad=0.5))
    
    plt.tight_layout()
    return fig

def create_exploration_curves():
    """Exploration vs exploitation reward progression."""
    fig, ax = plt.subplots(figsize=(9, 5.5), facecolor='black')
    ax.set_facecolor('black')
    
    episodes = np.linspace(0, 100, 120)
    
    # Pure exploitation - gets stuck
    np.random.seed(42)
    exploit = 5 + 2.5*np.tanh((episodes-8)/12) + np.random.randn(120)*0.4
    exploit = np.clip(exploit, 0, 7.5)
    
    # Balanced - finds better solutions
    balanced = 3 + 0.055*episodes + np.sin(episodes/8)*0.8 + np.random.randn(120)*0.6
    balanced = np.clip(balanced, 0, 13)
    
    ax.plot(episodes, exploit, 'r-', linewidth=3, label='Pure Exploitation (stuck)', alpha=0.85)
    ax.plot(episodes, balanced, 'g-', linewidth=3, label='Balanced Explore/Exploit (optimal)', alpha=0.85)
    
    ax.fill_between(episodes, exploit, alpha=0.2, color='red')
    ax.fill_between(episodes, balanced, alpha=0.2, color='green')
    
    # Mark phases
    ax.axvspan(0, 35, alpha=0.08, color='yellow')
    ax.text(17, 12, 'Exploration Phase', ha='center', color='#FFD700', 
            fontsize=11, weight='bold', alpha=0.9)
    ax.axvspan(35, 100, alpha=0.08, color='blue')
    ax.text(67, 12, 'Exploitation Phase', ha='center', color='#4DABF7', 
            fontsize=11, weight='bold', alpha=0.9)
    
    ax.set_xlabel('Episode', color='white', fontsize=13, weight='bold')
    ax.set_ylabel('Average Reward', color='white', fontsize=13, weight='bold')
    ax.set_title('Exploration vs Exploitation Trade-off', color='white', 
                 fontsize=14, weight='bold')
    ax.legend(facecolor='#1a1a1a', edgecolor='white', labelcolor='white', fontsize=11)
    ax.tick_params(colors='white', labelsize=10)
    for spine in ['bottom', 'left']:
        ax.spines[spine].set_color('white')
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.grid(True, alpha=0.3, color='white')
    
    plt.tight_layout()
    return fig

def create_ppo_clipping_detailed():
    """Enhanced PPO clipping with multiple views."""
    fig = plt.figure(figsize=(13, 5.5), facecolor='black')
    gs = fig.add_gridspec(1, 2, wspace=0.3)
    
    # Left: Clipping curve
    ax1 = fig.add_subplot(gs[0, 0])
    ax1.set_facecolor('black')
    
    r = np.linspace(0.3, 2.2, 250)
    epsilon = 0.2
    advantage = 1.8
    
    unclipped = r * advantage
    clipped = np.minimum(r * advantage, np.clip(r, 1-epsilon, 1+epsilon) * advantage)
    
    ax1.plot(r, unclipped, 'b--', linewidth=3, label='Unclipped Objective', alpha=0.75)
    ax1.plot(r, clipped, 'r-', linewidth=3.5, label='PPO Clipped')
    ax1.axvline(x=1.0, color='gray', linestyle=':', linewidth=2.5, alpha=0.7, label='Old Policy (r=1)')
    ax1.axvline(x=1-epsilon, color='orange', linestyle='--', linewidth=2, alpha=0.8)
    ax1.axvline(x=1+epsilon, color='orange', linestyle='--', linewidth=2, alpha=0.8)
    ax1.fill_between(r, clipped, unclipped, where=(r > 1+epsilon), alpha=0.25, color='red')
    
    ax1.set_xlabel('Policy Ratio r(θ) = π_new/π_old', color='white', fontsize=12, weight='bold')
    ax1.set_ylabel('Objective Value', color='white', fontsize=12, weight='bold')
    ax1.set_title('PPO Clipping Mechanism', color='white', fontsize=14, weight='bold')
    ax1.legend(facecolor='#1a1a1a', edgecolor='white', labelcolor='white', fontsize=10)
    ax1.tick_params(colors='white', labelsize=10)
    for spine in ['bottom', 'left']:
        ax1.spines[spine].set_color('white')
    ax1.spines['top'].set_visible(False)
    ax1.spines['right'].set_visible(False)
    ax1.grid(True, alpha=0.25, color='white')
    
    # Right: Policy shift
    ax2 = fig.add_subplot(gs[0, 1])
    ax2.set_facecolor('black')
    
    actions = np.linspace(-2.5, 2.5, 200)
    old_policy = np.exp(-((actions-0)**2)/0.6) / np.sqrt(np.pi*0.6)
    new_policy = np.exp(-((actions-0.7)**2)/0.6) / np.sqrt(np.pi*0.6)
    no_clip = np.exp(-((actions-1.4)**2)/0.6) / np.sqrt(np.pi*0.6)
    
    ax2.plot(actions, old_policy, 'b--', linewidth=2.5, label='π_old', alpha=0.75)
    ax2.plot(actions, new_policy, 'g-', linewidth=3, label='π_new (clipped)')
    ax2.plot(actions, no_clip, 'r:', linewidth=2.5, label='unclipped (too aggressive)', alpha=0.75)
    
    ax2.fill_between(actions, old_policy, alpha=0.18, color='blue')
    ax2.fill_between(actions, new_policy, alpha=0.18, color='green')
    
    ax2.set_xlabel('Action Space', color='white', fontsize=12, weight='bold')
    ax2.set_ylabel('Probability π(a|s)', color='white', fontsize=12, weight='bold')
    ax2.set_title('Policy Update with Clipping', color='white', fontsize=14, weight='bold')
    ax2.legend(facecolor='#1a1a1a', edgecolor='white', labelcolor='white', fontsize=10)
    ax2.tick_params(colors='white', labelsize=10)
    for spine in ['bottom', 'left']:
        ax2.spines[spine].set_color('white')
    ax2.spines['top'].set_visible(False)
    ax2.spines['right'].set_visible(False)
    ax2.grid(True, alpha=0.25, color='white')
    
    plt.tight_layout()
    return fig

def create_sac_visualization():
    """SAC twin Q-networks and entropy bonus."""
    fig = plt.figure(figsize=(13, 5.5), facecolor='black')
    gs = fig.add_gridspec(1, 2, wspace=0.3)
    
    # Left: Twin Q-networks
    ax1 = fig.add_subplot(gs[0, 0])
    ax1.set_facecolor('black')
    
    states = np.arange(10)
    q1 = np.array([2.2, 3.7, 2.9, 4.4, 4.1, 5.3, 4.9, 5.7, 5.3, 6.2])
    q2 = np.array([2.4, 3.3, 3.1, 4.7, 3.8, 5.0, 5.1, 5.4, 5.5, 5.9])
    min_q = np.minimum(q1, q2)
    
    width = 0.25
    ax1.bar(states - width, q1, width, label='Q₁(s,a)', color='#4DABF7', alpha=0.85, edgecolor='white', linewidth=1.5)
    ax1.bar(states, q2, width, label='Q₂(s,a)', color='#51CF66', alpha=0.85, edgecolor='white', linewidth=1.5)
    ax1.bar(states + width, min_q, width, label='min(Q₁, Q₂)', color='#FFD700', alpha=0.85, edgecolor='white', linewidth=1.5)
    
    ax1.set_xlabel('State Index', color='white', fontsize=12, weight='bold')
    ax1.set_ylabel('Q-Value Estimate', color='white', fontsize=12, weight='bold')
    ax1.set_title('Twin Q-Networks (Reduces Overestimation)', color='white', fontsize=14, weight='bold')
    ax1.legend(facecolor='#1a1a1a', edgecolor='white', labelcolor='white', fontsize=10)
    ax1.tick_params(colors='white', labelsize=10)
    for spine in ['bottom', 'left']:
        ax1.spines[spine].set_color('white')
    ax1.spines['top'].set_visible(False)
    ax1.spines['right'].set_visible(False)
    ax1.grid(True, alpha=0.25, color='white', axis='y')
    
    # Right: Entropy bonus
    ax2 = fig.add_subplot(gs[0, 1])
    ax2.set_facecolor('black')
    
    alpha_vals = np.linspace(0, 1.2, 150)
    base_reward = 10
    entropy_bonus = 3 * alpha_vals
    total_value = base_reward + entropy_bonus
    
    ax2.plot(alpha_vals, [base_reward]*len(alpha_vals), 'b--', linewidth=2.5, 
             label='Reward Only', alpha=0.75)
    ax2.plot(alpha_vals, total_value, 'g-', linewidth=3.5, label='Reward + α·Entropy')
    ax2.fill_between(alpha_vals, base_reward, total_value, alpha=0.3, color='green',
                     label='Entropy Bonus')
    
    ax2.set_xlabel('Temperature α', color='white', fontsize=12, weight='bold')
    ax2.set_ylabel('Total Value', color='white', fontsize=12, weight='bold')
    ax2.set_title('Soft Value Function: V = Q + α·H(π)', color='white', fontsize=14, weight='bold')
    ax2.legend(facecolor='#1a1a1a', edgecolor='white', labelcolor='white', fontsize=10)
    ax2.tick_params(colors='white', labelsize=10)
    for spine in ['bottom', 'left']:
        ax2.spines[spine].set_color('white')
    ax2.spines['top'].set_visible(False)
    ax2.spines['right'].set_visible(False)
    ax2.grid(True, alpha=0.25, color='white')
    
    plt.tight_layout()
    return fig

def create_dqn_components():
    """DQN Bellman equation and replay buffer visualization."""
    fig = plt.figure(figsize=(13, 5.5), facecolor='black')
    gs = fig.add_gridspec(1, 2, wspace=0.3)
    
    # Left: Bellman backup
    ax1 = fig.add_subplot(gs[0, 0])
    ax1.set_facecolor('black')
    ax1.set_xlim(0, 10)
    ax1.set_ylim(0, 7.5)
    ax1.axis('off')
    
    # Current state
    s_box = patches.FancyBboxPatch((1, 4.5), 2, 1.5, linewidth=3, 
                                   edgecolor='#4DABF7', facecolor='#1a1a1a',
                                   boxstyle="round,pad=0.1")
    ax1.add_patch(s_box)
    ax1.text(2, 5.25, 's', ha='center', va='center', color='white', fontsize=16, weight='bold')
    ax1.text(2, 3.7, 'Q(s,a₁)=2.3\\nQ(s,a₂)=3.8\\nQ(s,a₃)=1.9', ha='center', 
            color='#4DABF7', fontsize=9.5)
    
    # Next state
    sp_box = patches.FancyBboxPatch((7, 4.5), 2, 1.5, linewidth=3, 
                                    edgecolor='#51CF66', facecolor='#1a1a1a',
                                    boxstyle="round,pad=0.1")
    ax1.add_patch(sp_box)
    ax1.text(8, 5.25, "s'", ha='center', va='center', color='white', fontsize=16, weight='bold')
    ax1.text(8, 3.7, "Q(s',a₁)=3.4\\nQ(s',a₂)=4.3\\nQ(s',a₃)=3.1", ha='center', 
            color='#51CF66', fontsize=9.5)
    
    # Transition
    ax1.annotate('', xy=(7, 5.25), xytext=(3, 5.25),
                arrowprops=dict(arrowstyle='->', color='white', lw=3.5))
    ax1.text(5, 6.1, 'Take action a₂', ha='center', color='#FFD700', fontsize=12, weight='bold')
    ax1.text(5, 5.25, 'Reward r = +5', ha='center', color='#FF6B6B', fontsize=12, weight='bold')
    
    # Bellman equation
    eq_text = 'Q(s,a₂) ← r + γ·max Q(s\',a\') = 5 + 0.99·4.3 ≈ 9.26'
    ax1.text(5, 2, eq_text, ha='center', color='white', fontsize=11,
            bbox=dict(boxstyle='round', facecolor='#2a2a2a', edgecolor='#FFD700', linewidth=2.5, pad=0.9))
    
    ax1.text(8, 2.8, 'Target\\nNetwork', ha='center', color='#51CF66', 
            fontsize=10, style='italic')
    
    ax1.text(5, 7.2, 'DQN Bellman Backup', ha='center', color='white', 
            fontsize=15, weight='bold')
    
    # Right: Q-value bars
    ax2 = fig.add_subplot(gs[0, 1])
    ax2.set_facecolor('black')
    
    actions = ['Up', 'Down', 'Left', 'Right', 'Stay']
    q_vals = [3.2, 4.7, 2.1, 3.8, 2.9]
    colors_list = ['#FF6B6B', '#51CF66', '#4DABF7', '#FFD700', '#B794F6']
    
    bars = ax2.bar(actions, q_vals, color=colors_list, alpha=0.8, edgecolor='white', linewidth=2)
    
    # Highlight max
    max_idx = q_vals.index(max(q_vals))
    bars[max_idx].set_edgecolor('#51CF66')
    bars[max_idx].set_linewidth(4)
    
    ax2.set_ylabel('Q-Value Q(s,a)', color='white', fontsize=12, weight='bold')
    ax2.set_title('Value-Based Action Selection', color='white', fontsize=14, weight='bold')
    ax2.tick_params(colors='white', labelsize=10)
    for spine in ['bottom', 'left']:
        ax2.spines[spine].set_color('white')
    ax2.spines['top'].set_visible(False)
    ax2.spines['right'].set_visible(False)
    ax2.grid(True, alpha=0.25, color='white', axis='y')
    ax2.text(max_idx, q_vals[max_idx] + 0.2, 'Best Action', ha='center', 
            color='#51CF66', fontsize=10, weight='bold')
    
    plt.tight_layout()
    return fig

def create_advantage_diagram():
    """Advantage function A(s,a) = Q(s,a) - V(s)."""
    fig, ax = plt.subplots(figsize=(8, 5), facecolor='black')
    ax.set_facecolor('black')
    
    actions = ['a₁', 'a₂', 'a₃', 'a₄', 'a₅']
    q_values = [3.2, 5.1, 2.8, 4.6, 3.9]
    v_state = np.mean(q_values)
    advantages = [q - v_state for q in q_values]
    
    colors = ['#FF6B6B' if a < 0 else '#51CF66' for a in advantages]
    bars = ax.bar(actions, advantages, color=colors, alpha=0.8, edgecolor='white', linewidth=2)
    
    ax.axhline(y=0, color='white', linestyle='-', linewidth=2, alpha=0.7)
    ax.text(len(actions)-0.5, 0.1, 'V(s) baseline', ha='right', color='white', 
            fontsize=10, style='italic')
    
    ax.set_ylabel('Advantage A(s,a)', color='white', fontsize=12, weight='bold')
    ax.set_title('Advantage = Q(s,a) - V(s)', color='white', fontsize=14, weight='bold')
    ax.tick_params(colors='white', labelsize=10)
    for spine in ['bottom', 'left']:
        ax.spines[spine].set_color('white')
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.grid(True, alpha=0.25, color='white', axis='y')
    
    # Label positive/negative
    ax.text(0.5, 1.2, 'Better than\\naverage', ha='center', color='#51CF66', 
            fontsize=9, weight='bold')
    ax.text(2, -0.6, 'Worse than\\naverage', ha='center', color='#FF6B6B', 
            fontsize=9, weight='bold')
    
    plt.tight_layout()
    return fig

def create_value_vs_policy():
    """Policy-based vs value-based comparison."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4.5), facecolor='black')
    
    # Left: Policy-based
    ax1.set_facecolor('black')
    actions = np.linspace(-2, 2, 150)
    policy_dist = np.exp(-((actions-0.5)**2)/0.7) / np.sqrt(np.pi*0.7)
    
    ax1.plot(actions, policy_dist, 'g-', linewidth=3, label='π(a|s)')
    ax1.fill_between(actions, policy_dist, alpha=0.3, color='green')
    ax1.set_xlabel('Continuous Action Space', color='white', fontsize=11, weight='bold')
    ax1.set_ylabel('Probability', color='white', fontsize=11, weight='bold')
    ax1.set_title('Policy-Based: Learn π(a|s)', color='#51CF66', fontsize=13, weight='bold')
    ax1.tick_params(colors='white', labelsize=10)
    for spine in ['bottom', 'left']:
        ax1.spines[spine].set_color('white')
    ax1.spines['top'].set_visible(False)
    ax1.spines['right'].set_visible(False)
    ax1.grid(True, alpha=0.25, color='white')
    
    # Right: Value-based
    ax2.set_facecolor('black')
    discrete_actions = ['a₁', 'a₂', 'a₃', 'a₄']
    q_vals = [2.3, 4.1, 3.2, 2.8]
    bars = ax2.bar(discrete_actions, q_vals, color='#4DABF7', alpha=0.8, edgecolor='white', linewidth=2)
    
    max_idx = q_vals.index(max(q_vals))
    bars[max_idx].set_edgecolor('#FFD700')
    bars[max_idx].set_linewidth(3.5)
    
    ax2.set_ylabel('Q-Value', color='white', fontsize=11, weight='bold')
    ax2.set_title('Value-Based: Learn Q(s,a)', color='#4DABF7', fontsize=13, weight='bold')
    ax2.tick_params(colors='white', labelsize=10)
    for spine in ['bottom', 'left']:
        ax2.spines[spine].set_color('white')
    ax2.spines['top'].set_visible(False)
    ax2.spines['right'].set_visible(False)
    ax2.grid(True, alpha=0.25, color='white', axis='y')
    ax2.text(max_idx, q_vals[max_idx] + 0.2, 'Choose this', ha='center', 
            color='#FFD700', fontsize=10, weight='bold')
    
    plt.tight_layout()
    return fig

def create_reward_progression():
    """Enhanced noisy learning curve with moving average."""
    fig, ax = plt.subplots(figsize=(9, 5), facecolor='black')
    ax.set_facecolor('black')
    
    steps = np.linspace(0, 100, 400)
    np.random.seed(45)
    
    # Base signal
    base = 2 + 0.08*steps + 3*np.sin(steps/15) * np.exp(-steps/80)
    noise = np.random.randn(400) * (1.5 - 0.01*steps)
    rewards = base + noise
    
    # Moving average
    window = 20
    moving_avg = np.convolve(rewards, np.ones(window)/window, mode='same')
    
    ax.plot(steps, rewards, 'b-', linewidth=0.8, alpha=0.4, label='Episode Reward')
    ax.plot(steps, moving_avg, 'r-', linewidth=3, label='Moving Average')
    ax.fill_between(steps, moving_avg, alpha=0.2, color='red')
    
    ax.set_xlabel('Training Episodes', color='white', fontsize=12, weight='bold')
    ax.set_ylabel('Total Reward', color='white', fontsize=12, weight='bold')
    ax.set_title('Noisy Learning Curve (Typical RL Training)', color='white', 
                 fontsize=14, weight='bold')
    ax.legend(facecolor='#1a1a1a', edgecolor='white', labelcolor='white', fontsize=11)
    ax.tick_params(colors='white', labelsize=10)
    for spine in ['bottom', 'left']:
        ax.spines[spine].set_color('white')
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.grid(True, alpha=0.3, color='white')
    
    plt.tight_layout()
    return fig


# ============================================================================
# MAIN DASHBOARD
# ============================================================================

def main():
    """Main dashboard application with comprehensive RL education."""
    
    # Title
    st.markdown('<p class="handwritten-title">🧠 The Comprehensive Guide to Modern Reinforcement Learning</p>', 
                unsafe_allow_html=True)
    
    st.markdown('<div class="divider-line"></div>', unsafe_allow_html=True)
    
    # Intro Section
    st.markdown('<p class="handwritten-section">What is Reinforcement Learning?</p>', 
                unsafe_allow_html=True)
    
    st.markdown('<div class="intro-section">', unsafe_allow_html=True)
    
    intro_col1, intro_col2 = st.columns([3, 2])
    
    with intro_col1:
        st.markdown("""
        <p style="font-size: 16px; line-height: 1.8;">
        <strong>Reinforcement Learning (RL)</strong> is a machine learning paradigm where an 
        <span class="blue-text">agent</span> learns to make decisions by interacting with an 
        <span class="green-text">environment</span>. The agent receives 
        <span class="yellow-text">rewards</span> for its actions and learns to maximize 
        cumulative reward through trial and error.
        </p>
        
        <p style="font-size: 15px; margin-top: 20px;"><strong class="yellow-text">Why RL Matters:</strong></p>
        <ul style="font-size: 14px;">
            <li><strong>Autonomous Decision Making:</strong> Agents learn without explicit supervision</li>
            <li><strong>Sequential Decisions:</strong> Handles long-term consequences</li>
            <li><strong>Adaptability:</strong> Learns optimal behavior through experience</li>
        </ul>
        
        <p style="font-size: 15px; margin-top: 20px;"><strong class="green-text">Real-World Applications:</strong></p>
        <ul style="font-size: 14px;">
            <li><span class="blue-text">🤖 Robotics:</span> Manipulation, locomotion, navigation</li>
            <li><span class="yellow-text">💰 Finance:</span> Portfolio optimization, trading</li>
            <li><span class="green-text">🎮 Gaming:</span> AlphaGo, OpenAI Five, Atari</li>
            <li><span class="purple-text">📱 Recommendation:</span> Personalized content</li>
            <li><span class="orange-text">🚗 Vehicles:</span> Self-driving decisions</li>
        </ul>
        """, unsafe_allow_html=True)
    
    with intro_col2:
        fig = create_rl_loop_diagram()
        st.pyplot(fig, use_container_width=True)
        plt.close(fig)
        
        st.markdown("""
        <div style="margin-top: 20px; padding: 15px; background-color: #2a2a2a; border-radius: 8px;">
            <p style="font-size: 14px;"><strong class="yellow-text">Why Now?</strong></p>
            <p style="font-size: 13px; margin-top: 10px;">
            Modern RL success stems from:
            </p>
            <ul style="font-size: 12px;">
                <li><strong>Deep Learning:</strong> Neural function approximators</li>
                <li><strong>Compute:</strong> GPUs enable massive simulation</li>
                <li><strong>Algorithms:</strong> PPO, SAC, DQN solve stability</li>
            </ul>
        </div>
        """, unsafe_allow_html=True)
    
    st.markdown('</div>', unsafe_allow_html=True)
    st.markdown('<div class="arrow-down">↓</div>', unsafe_allow_html=True)
    
    # Core Concepts Section
    st.markdown('<p class="handwritten-section">Core Concepts</p>', unsafe_allow_html=True)
    
    def_col1, def_col2, def_col3 = st.columns(3)
    
    with def_col1:
        st.markdown('<div class="definition-block">', unsafe_allow_html=True)
        st.markdown('<p class="handwritten-subtitle">The Agent</p>', unsafe_allow_html=True)
        st.markdown("""
        <p style="font-size: 14px; line-height: 1.7;">
        <strong>What it is:</strong> The <span class="blue-text">agent</span> is the learner 
        that observes, acts, and learns from feedback.
        </p>
        <p style="font-size: 14px; margin-top: 12px;">
        <strong>How it behaves:</strong> Follows policy π(a|s) mapping states to actions.
        </p>
        <p style="font-size: 14px; margin-top: 12px;">
        <strong>Why central:</strong> Goal is learning optimal policy π* maximizing reward.
        </p>
        """, unsafe_allow_html=True)
        st.markdown('</div>', unsafe_allow_html=True)
    
    with def_col2:
        st.markdown('<div class="definition-block">', unsafe_allow_html=True)
        st.markdown('<p class="handwritten-subtitle">Entropy</p>', unsafe_allow_html=True)
        st.markdown("""
        <p style="font-size: 14px; line-height: 1.7;">
        <strong>What it means:</strong> <span class="green-text">Entropy</span> measures 
        randomness in policy. High = exploration, low = exploitation.
        </p>
        <p style="font-size: 14px; margin-top: 12px;">
        <strong>SAC Connection:</strong> Explicitly maximizes entropy for exploration.
        </p>
        """, unsafe_allow_html=True)
        
        fig = create_entropy_comparison()
        st.pyplot(fig, use_container_width=True)
        plt.close(fig)
        st.markdown('</div>', unsafe_allow_html=True)
    
    with def_col3:
        st.markdown('<div class="definition-block">', unsafe_allow_html=True)
        st.markdown('<p class="handwritten-subtitle">Exploration</p>', unsafe_allow_html=True)
        st.markdown("""
        <p style="font-size: 14px; line-height: 1.7;">
        <strong>What it is:</strong> <span class="yellow-text">Exploration</span> tries 
        new actions to discover better strategies.
        </p>
        <p style="font-size: 14px; margin-top: 12px;">
        <strong>Why important:</strong> Without it, agents get stuck in local optima.
        </p>
        """, unsafe_allow_html=True)
        
        fig = create_exploration_curves()
        st.pyplot(fig, use_container_width=True)
        plt.close(fig)
        st.markdown('</div>', unsafe_allow_html=True)
    
    st.markdown('<div class="arrow-down">↓</div>', unsafe_allow_html=True)
    
    # Background Concepts
    st.markdown('<p class="handwritten-section">Essential Background</p>', unsafe_allow_html=True)
    
    concept_col1, concept_col2 = st.columns(2)
    
    with concept_col1:
        st.markdown("""
        <div class="floating-concept">
            <p style="font-size: 16px;"><strong class="blue-text">Markov Decision Processes</strong></p>
            <p style="font-size: 13px;">Mathematical framework: (S, A, P, R, γ)</p>
            <ul style="font-size: 12px;">
                <li>S: State space</li>
                <li>A: Action space</li>
                <li>P: Transition P(s'|s,a)</li>
                <li>R: Reward R(s,a,s')</li>
                <li>γ: Discount factor</li>
            </ul>
        </div>
        """, unsafe_allow_html=True)
        
        fig = create_mdp_diagram()
        st.pyplot(fig, use_container_width=True)
        plt.close(fig)
        
        st.markdown("""
        <div class="floating-concept">
            <p style="font-size: 16px;"><strong class="green-text">Policies vs Values</strong></p>
            <ul style="font-size: 12px;">
                <li><strong>Policy π(a|s):</strong> What to do</li>
                <li><strong>Value V(s):</strong> How good is state</li>
                <li><strong>Q-value Q(s,a):</strong> How good is (s,a)</li>
            </ul>
        </div>
        """, unsafe_allow_html=True)
        
        fig = create_value_vs_policy()
        st.pyplot(fig, use_container_width=True)
        plt.close(fig)
    
    with concept_col2:
        st.markdown("""
        <div class="floating-concept">
            <p style="font-size: 16px;"><strong class="orange-text">On-Policy vs Off-Policy</strong></p>
            <ul style="font-size: 12px;">
                <li><strong class="green-text">On-Policy (PPO):</strong> Learn from current policy</li>
                <li><strong class="blue-text">Off-Policy (SAC, DQN):</strong> Learn from any data</li>
            </ul>
        </div>
        
        <div class="floating-concept">
            <p style="font-size: 16px;"><strong class="cyan-text">Advantage Estimation</strong></p>
            <p style="font-size: 12px;">A(s,a) = Q(s,a) - V(s)</p>
            <p style="font-size: 12px;">Measures how much better action is vs average.</p>
        </div>
        """, unsafe_allow_html=True)
        
        fig = create_advantage_diagram()
        st.pyplot(fig, use_container_width=True)
        plt.close(fig)
        
        st.markdown("""
        <div class="floating-concept">
            <p style="font-size: 16px;"><strong class="yellow-text">Bellman Equations</strong></p>
            <p style="font-size: 12px; font-family: monospace;">
            V(s) = 𝔼[r + γ·V(s')]<br>
            Q(s,a) = 𝔼[r + γ·max Q(s',a')]
            </p>
        </div>
        """, unsafe_allow_html=True)
    
    st.markdown('<div class="arrow-down">↓</div>', unsafe_allow_html=True)
    
    # Algorithm Deep Dives
    st.markdown('<p class="handwritten-section">The Three Pillars</p>', unsafe_allow_html=True)
    
    algo_col1, algo_col2, algo_col3 = st.columns(3)
    
    # PPO
    with algo_col1:
        st.markdown('<div class="algo-block">', unsafe_allow_html=True)
        st.markdown('<p class="handwritten-subtitle" style="color: #4DABF7;">PPO</p>', unsafe_allow_html=True)
        st.markdown("""
        <p style="font-size: 14px;"><strong>Proximal Policy Optimization</strong></p>
        <p style="font-size: 13px;">On-policy method with clipped objective for stable training.</p>
        <p style="font-size: 13px; margin-top: 10px;"><strong class="yellow-text">Innovations:</strong></p>
        <ul style="font-size: 12px;">
            <li>Clipped objective limits updates</li>
            <li>Multiple epochs on data</li>
            <li>Entropy bonus for exploration</li>
        </ul>
        """, unsafe_allow_html=True)
        
        fig = create_ppo_clipping_detailed()
        st.pyplot(fig, use_container_width=True)
        plt.close(fig)
        
        st.markdown("""
        <p style="font-size: 12px; margin-top: 10px;"><strong class="green-text">When to use:</strong></p>
        <ul style="font-size: 11px;">
            <li>Robotics, control, games</li>
            <li>Need stability</li>
            <li>Continuous/discrete actions</li>
        </ul>
        """, unsafe_allow_html=True)
        st.markdown('</div>', unsafe_allow_html=True)
    
    # SAC
    with algo_col2:
        st.markdown('<div class="algo-block">', unsafe_allow_html=True)
        st.markdown('<p class="handwritten-subtitle" style="color: #51CF66;">SAC</p>', unsafe_allow_html=True)
        st.markdown("""
        <p style="font-size: 14px;"><strong>Soft Actor-Critic</strong></p>
        <p style="font-size: 13px;">Off-policy with entropy maximization for exploration.</p>
        <p style="font-size: 13px; margin-top: 10px;"><strong class="yellow-text">Innovations:</strong></p>
        <ul style="font-size: 12px;">
            <li>Entropy maximization</li>
            <li>Twin Q-networks</li>
            <li>Automatic temperature tuning</li>
        </ul>
        """, unsafe_allow_html=True)
        
        fig = create_sac_visualization()
        st.pyplot(fig, use_container_width=True)
        plt.close(fig)
        
        st.markdown("""
        <p style="font-size: 12px; margin-top: 10px;"><strong class="green-text">When to use:</strong></p>
        <ul style="font-size: 11px;">
            <li>Continuous actions</li>
            <li>High sample efficiency needed</li>
            <li>Complex environments</li>
        </ul>
        """, unsafe_allow_html=True)
        st.markdown('</div>', unsafe_allow_html=True)
    
    # DQN
    with algo_col3:
        st.markdown('<div class="algo-block">', unsafe_allow_html=True)
        st.markdown('<p class="handwritten-subtitle" style="color: #FFD700;">DQN</p>', unsafe_allow_html=True)
        st.markdown("""
        <p style="font-size: 14px;"><strong>Deep Q-Network</strong></p>
        <p style="font-size: 13px;">Value-based method with deep neural networks.</p>
        <p style="font-size: 13px; margin-top: 10px;"><strong class="yellow-text">Innovations:</strong></p>
        <ul style="font-size: 12px;">
            <li>Experience replay</li>
            <li>Target network</li>
            <li>Deep Q-function approximation</li>
        </ul>
        """, unsafe_allow_html=True)
        
        fig = create_dqn_components()
        st.pyplot(fig, use_container_width=True)
        plt.close(fig)
        
        st.markdown("""
        <p style="font-size: 12px; margin-top: 10px;"><strong class="green-text">When to use:</strong></p>
        <ul style="font-size: 11px;">
            <li>Discrete actions only</li>
            <li>Atari, grid worlds</li>
            <li>Large state spaces</li>
        </ul>
        """, unsafe_allow_html=True)
        st.markdown('</div>', unsafe_allow_html=True)
    
    st.markdown('<div class="arrow-down">↓</div>', unsafe_allow_html=True)
    
    # Learning Dynamics
    st.markdown('<p class="handwritten-section">Learning Dynamics</p>', unsafe_allow_html=True)
    
    vis_col1, vis_col2 = st.columns(2)
    
    with vis_col1:
        st.markdown('<div class="info-box">', unsafe_allow_html=True)
        st.markdown('<p style="font-size: 16px;"><strong>Training Curves</strong></p>', unsafe_allow_html=True)
        st.markdown('<p style="font-size: 13px;">RL training is noisy due to stochastic environments, non-stationary data, and exploration.</p>', unsafe_allow_html=True)
        st.markdown('</div>', unsafe_allow_html=True)
        
        fig = create_reward_progression()
        st.pyplot(fig, use_container_width=True)
        plt.close(fig)
    
    with vis_col2:
        st.markdown('<div class="info-box">', unsafe_allow_html=True)
        st.markdown('<p style="font-size: 16px;"><strong>Challenges</strong></p>', unsafe_allow_html=True)
        st.markdown("""
        <ul style="font-size: 12px;">
            <li><strong class="red-text">Credit Assignment:</strong> Which action caused reward?</li>
            <li><strong class="yellow-text">Exploration:</strong> Finding better policies</li>
            <li><strong class="blue-text">Sample Efficiency:</strong> Needs lots of data</li>
            <li><strong class="purple-text">Generalization:</strong> Overfitting to training</li>
        </ul>
        """, unsafe_allow_html=True)
        st.markdown('</div>', unsafe_allow_html=True)
    
    st.markdown('<div class="divider-line"></div>', unsafe_allow_html=True)
    
    # Conclusion
    st.markdown('<p class="handwritten-section">The Landscape</p>', unsafe_allow_html=True)
    
    st.markdown('<div class="conclusion-panel">', unsafe_allow_html=True)
    st.markdown("""
    <p style="font-size: 16px; text-align: center; line-height: 1.8;">
    Modern RL offers diverse tools: <span class="blue-text">PPO</span> for stability, 
    <span class="green-text">SAC</span> for efficiency, <span class="yellow-text">DQN</span> for discrete actions.
    </p>
    
    <div style="margin: 25px 0; padding: 20px; background-color: #2a2a2a; border-radius: 10px;">
        <p style="text-align: center; font-size: 15px;"><strong>Algorithm Comparison</strong></p>
        <p style="font-size: 13px; margin-top: 15px;">
        • <strong class="blue-text">PPO:</strong> On-Policy, Both actions, Stable<br>
        • <strong class="green-text">SAC:</strong> Off-Policy, Continuous, High efficiency<br>
        • <strong class="yellow-text">DQN:</strong> Off-Policy, Discrete, Value-based
        </p>
    </div>
    
    <p style="font-size: 15px; text-align: center; margin-top: 25px;"><strong class="yellow-text">Future Frontiers:</strong></p>
    <p style="font-size: 13px; text-align: center;">
    Model-Based RL • Multi-Agent RL • Meta-RL • Offline RL • Transformers • Distributional RL
    </p>
    """, unsafe_allow_html=True)
    st.markdown('</div>', unsafe_allow_html=True)
    
    # Footer
    st.markdown('<div class="divider-line"></div>', unsafe_allow_html=True)
    st.markdown("""
    <div style="text-align: center; margin-top: 30px; padding: 20px; opacity: 0.7;">
        <p style="font-size: 14px;">🎬 <strong>Comprehensive RL Dashboard</strong></p>
        <p style="font-size: 12px;">Built with Streamlit | Visualizing PPO, SAC, DQN</p>
    </div>
    """, unsafe_allow_html=True)

if __name__ == "__main__":
    main()

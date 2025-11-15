"""
Reinforcement Learning Mind-Map Dashboard
==========================================
A professional, dark-mode visual storyboard for understanding modern RL algorithms:
PPO, SAC, and DQN.
"""

import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches

# ============================================================================
# PAGE CONFIGURATION
# ============================================================================

st.set_page_config(
    page_title="RL Mind-Map Dashboard",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# ============================================================================
# CUSTOM CSS - DARK MODE THEME
# ============================================================================

st.markdown("""
<style>
    /* Global dark background */
    .stApp {
        background-color: #000000;
        color: #FFFFFF;
    }
    
    /* Sidebar styling */
    [data-testid="stSidebar"] {
        background-color: #1a1a1a;
    }
    
    /* All text white by default */
    .stMarkdown, p, span, div, h1, h2, h3, h4, h5, h6 {
        color: #FFFFFF !important;
    }
    
    /* Handwritten-style font for section titles */
    .handwritten-title {
        font-family: 'Comic Sans MS', 'Brush Script MT', cursive;
        font-size: 28px;
        font-weight: bold;
        color: #FFFFFF;
        margin-bottom: 10px;
        text-align: center;
    }
    
    .handwritten-subtitle {
        font-family: 'Comic Sans MS', 'Brush Script MT', cursive;
        font-size: 20px;
        color: #FFFFFF;
        margin-bottom: 8px;
    }
    
    /* Concept blocks */
    .concept-block {
        background-color: #1a1a1a;
        border: 2px solid #FFFFFF;
        border-radius: 10px;
        padding: 20px;
        margin: 10px;
        min-height: 280px;
    }
    
    /* Yellow highlight box */
    .highlight-box {
        background-color: #1a1a1a;
        border: 3px solid #FFD700;
        border-radius: 10px;
        padding: 25px;
        margin: 20px 0;
    }
    
    /* Algorithm blocks */
    .algo-block {
        background-color: #1a1a1a;
        border: 2px solid #FFFFFF;
        border-radius: 10px;
        padding: 20px;
        margin: 10px;
        min-height: 400px;
    }
    
    /* Accent colors */
    .red-text { color: #FF6B6B !important; }
    .green-text { color: #51CF66 !important; }
    .yellow-text { color: #FFD700 !important; }
    .blue-text { color: #4DABF7 !important; }
    
    /* Bullets styling */
    ul {
        color: #FFFFFF;
    }
    
    li {
        color: #FFFFFF;
        margin: 5px 0;
    }
    
    /* Mini-cards */
    .mini-card {
        background-color: #2a2a2a;
        border-left: 4px solid #4DABF7;
        padding: 10px;
        margin: 10px 0;
        border-radius: 5px;
    }
    
    /* Arrow styling */
    .arrow-down {
        text-align: center;
        font-size: 40px;
        color: #FFFFFF;
        margin: 10px 0;
    }
    
    .arrow-right {
        display: inline-block;
        font-size: 30px;
        color: #FFFFFF;
    }
</style>
""", unsafe_allow_html=True)

# ============================================================================
# HELPER FUNCTIONS FOR DIAGRAMS
# ============================================================================

def create_agent_environment_diagram():
    """Create simple Agent-Environment loop diagram."""
    fig, ax = plt.subplots(figsize=(6, 3), facecolor='black')
    ax.set_facecolor('black')
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 4)
    ax.axis('off')
    
    # Agent box
    agent_box = patches.Rectangle((1, 1.5), 2, 1, linewidth=2, 
                                   edgecolor='white', facecolor='#1a1a1a')
    ax.add_patch(agent_box)
    ax.text(2, 2, 'Agent', ha='center', va='center', color='white', fontsize=12, weight='bold')
    
    # Environment box
    env_box = patches.Rectangle((7, 1.5), 2, 1, linewidth=2, 
                                 edgecolor='white', facecolor='#1a1a1a')
    ax.add_patch(env_box)
    ax.text(8, 2, 'Env', ha='center', va='center', color='white', fontsize=12, weight='bold')
    
    # Action arrow
    ax.annotate('', xy=(7, 2.2), xytext=(3, 2.2),
                arrowprops=dict(arrowstyle='->', color='#51CF66', lw=2))
    ax.text(5, 2.5, 'action', ha='center', color='#51CF66', fontsize=10)
    
    # State arrow
    ax.annotate('', xy=(3, 1.8), xytext=(7, 1.8),
                arrowprops=dict(arrowstyle='->', color='#4DABF7', lw=2))
    ax.text(5, 1.4, 'state, reward', ha='center', color='#4DABF7', fontsize=10)
    
    plt.tight_layout()
    return fig

def create_policy_gradient_diagram():
    """Create policy gradient intuition diagram."""
    fig, ax = plt.subplots(figsize=(5, 3), facecolor='black')
    ax.set_facecolor('black')
    
    # Sample policy distribution
    x = np.linspace(-2, 2, 100)
    y_old = np.exp(-x**2) / np.sqrt(np.pi)
    y_new = np.exp(-(x-0.5)**2) / np.sqrt(np.pi)
    
    ax.plot(x, y_old, 'b--', linewidth=2, label='π_old', alpha=0.7)
    ax.plot(x, y_new, 'r-', linewidth=2, label='π_new')
    ax.fill_between(x, y_old, alpha=0.1, color='blue')
    ax.fill_between(x, y_new, alpha=0.1, color='red')
    
    ax.set_xlabel('Action Space', color='white', fontsize=10)
    ax.set_ylabel('Probability', color='white', fontsize=10)
    ax.set_title('Policy Shift', color='white', fontsize=11, weight='bold')
    ax.legend(facecolor='#1a1a1a', edgecolor='white', labelcolor='white')
    ax.tick_params(colors='white')
    ax.spines['bottom'].set_color('white')
    ax.spines['left'].set_color('white')
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.grid(True, alpha=0.2, color='white')
    
    plt.tight_layout()
    return fig

def create_value_function_diagram():
    """Create Q-learning intuition diagram."""
    fig, ax = plt.subplots(figsize=(5, 3), facecolor='black')
    ax.set_facecolor('black')
    
    # Q-values for different actions
    actions = ['A1', 'A2', 'A3', 'A4']
    q_values = [2.5, 3.8, 1.2, 2.9]
    colors = ['#FF6B6B', '#51CF66', '#4DABF7', '#FFD700']
    
    bars = ax.bar(actions, q_values, color=colors, alpha=0.7, edgecolor='white', linewidth=1.5)
    
    # Highlight best action
    max_idx = q_values.index(max(q_values))
    bars[max_idx].set_edgecolor('#51CF66')
    bars[max_idx].set_linewidth(3)
    
    ax.set_ylabel('Q-Value', color='white', fontsize=10)
    ax.set_title('Value-Based Selection', color='white', fontsize=11, weight='bold')
    ax.tick_params(colors='white')
    ax.spines['bottom'].set_color('white')
    ax.spines['left'].set_color('white')
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.grid(True, alpha=0.2, color='white', axis='y')
    
    plt.tight_layout()
    return fig

def create_ppo_diagram():
    """Create PPO clipping diagram."""
    fig, ax = plt.subplots(figsize=(5, 3.5), facecolor='black')
    ax.set_facecolor('black')
    
    r = np.linspace(0.5, 2.0, 100)
    epsilon = 0.2
    advantage = 1.0
    
    unclipped = r * advantage
    clipped = np.minimum(r * advantage, np.clip(r, 1-epsilon, 1+epsilon) * advantage)
    
    ax.plot(r, unclipped, 'b--', linewidth=2, label='Unclipped', alpha=0.7)
    ax.plot(r, clipped, 'r-', linewidth=2.5, label='PPO Clipped')
    ax.axvline(x=1.0, color='gray', linestyle=':', linewidth=1.5, alpha=0.5)
    ax.axvline(x=1-epsilon, color='orange', linestyle='--', linewidth=1, alpha=0.5)
    ax.axvline(x=1+epsilon, color='orange', linestyle='--', linewidth=1, alpha=0.5)
    
    ax.set_xlabel('Ratio r(θ)', color='white', fontsize=10)
    ax.set_ylabel('Objective', color='white', fontsize=10)
    ax.set_title('PPO Clipping', color='white', fontsize=11, weight='bold')
    ax.legend(facecolor='#1a1a1a', edgecolor='white', labelcolor='white')
    ax.tick_params(colors='white')
    ax.spines['bottom'].set_color('white')
    ax.spines['left'].set_color('white')
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.grid(True, alpha=0.2, color='white')
    
    plt.tight_layout()
    return fig

def create_sac_diagram():
    """Create SAC entropy diagram."""
    fig, ax = plt.subplots(figsize=(5, 3.5), facecolor='black')
    ax.set_facecolor('black')
    
    x = np.linspace(-3, 3, 100)
    
    # Low entropy (narrow)
    y_low = np.exp(-2*x**2) / np.sqrt(np.pi/2)
    # High entropy (wide)
    y_high = np.exp(-0.5*x**2) / np.sqrt(2*np.pi)
    
    ax.plot(x, y_low, 'r-', linewidth=2, label='Low Entropy', alpha=0.7)
    ax.plot(x, y_high, 'g-', linewidth=2, label='High Entropy')
    ax.fill_between(x, y_low, alpha=0.1, color='red')
    ax.fill_between(x, y_high, alpha=0.1, color='green')
    
    ax.set_xlabel('Action', color='white', fontsize=10)
    ax.set_ylabel('Probability', color='white', fontsize=10)
    ax.set_title('Entropy Regularization', color='white', fontsize=11, weight='bold')
    ax.legend(facecolor='#1a1a1a', edgecolor='white', labelcolor='white')
    ax.tick_params(colors='white')
    ax.spines['bottom'].set_color('white')
    ax.spines['left'].set_color('white')
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.grid(True, alpha=0.2, color='white')
    
    plt.tight_layout()
    return fig

def create_dqn_diagram():
    """Create DQN Bellman diagram."""
    fig, ax = plt.subplots(figsize=(5, 3.5), facecolor='black')
    ax.set_facecolor('black')
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 4)
    ax.axis('off')
    
    # State boxes
    s_box = patches.Rectangle((1, 2), 1.5, 0.8, linewidth=2, 
                               edgecolor='#4DABF7', facecolor='#1a1a1a')
    ax.add_patch(s_box)
    ax.text(1.75, 2.4, 's', ha='center', va='center', color='white', fontsize=12, weight='bold')
    
    sp_box = patches.Rectangle((7, 2), 1.5, 0.8, linewidth=2, 
                                edgecolor='#51CF66', facecolor='#1a1a1a')
    ax.add_patch(sp_box)
    ax.text(7.75, 2.4, "s'", ha='center', va='center', color='white', fontsize=12, weight='bold')
    
    # Arrow with reward
    ax.annotate('', xy=(7, 2.4), xytext=(2.5, 2.4),
                arrowprops=dict(arrowstyle='->', color='white', lw=2))
    ax.text(4.75, 2.8, 'action, reward', ha='center', color='#FFD700', fontsize=10)
    
    # Q-values
    ax.text(1.75, 1.2, 'Q(s,a)', ha='center', color='#4DABF7', fontsize=11)
    ax.text(7.75, 1.2, "max Q(s',a')", ha='center', color='#51CF66', fontsize=11)
    
    # Bellman equation
    ax.text(5, 0.5, 'Q(s,a) ← r + γ max Q(s\',a\')', ha='center', 
            color='white', fontsize=10, style='italic',
            bbox=dict(boxstyle='round', facecolor='#2a2a2a', edgecolor='#FFD700'))
    
    plt.tight_layout()
    return fig

# ============================================================================
# MAIN DASHBOARD
# ============================================================================

def main():
    # Main title with handwritten style
    st.markdown('<p class="handwritten-title">🧠 The Landscape of Modern Reinforcement Learning</p>', 
                unsafe_allow_html=True)
    
    st.markdown("---")
    
    # ========================================================================
    # TOP ROW - Three Blocks
    # ========================================================================
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.markdown('<div class="concept-block">', unsafe_allow_html=True)
        st.markdown('<p class="handwritten-subtitle">The RL Problem</p>', unsafe_allow_html=True)
        
        fig = create_agent_environment_diagram()
        st.pyplot(fig, use_container_width=True)
        plt.close(fig)
        
        st.markdown("""
        <p style="font-size: 14px; margin-top: 15px;">
        An agent learns to maximize rewards by taking actions in an environment.
        </p>
        <ul style="font-size: 13px;">
            <li><span class="blue-text">States</span>: What the agent observes</li>
            <li><span class="green-text">Actions</span>: What the agent does</li>
            <li><span class="yellow-text">Rewards</span>: Feedback signal</li>
        </ul>
        """, unsafe_allow_html=True)
        st.markdown('</div>', unsafe_allow_html=True)
    
    with col2:
        st.markdown('<div class="concept-block">', unsafe_allow_html=True)
        st.markdown('<p class="handwritten-subtitle">Why RL is Hard</p>', unsafe_allow_html=True)
        
        st.markdown("""
        <ul style="font-size: 13px; margin-top: 20px;">
            <li><span class="red-text">Credit Assignment</span>: Which action caused the reward?</li>
            <li><span class="yellow-text">Exploration</span>: Try new things vs. exploit known good actions</li>
            <li><span class="blue-text">Instability</span>: Learning can diverge or collapse</li>
            <li><span class="green-text">Sample Efficiency</span>: Need lots of data</li>
            <li><span class="red-text">Non-stationarity</span>: Policy changes → data distribution changes</li>
        </ul>
        """, unsafe_allow_html=True)
        
        # Small illustrative plot
        fig, ax = plt.subplots(figsize=(4, 2), facecolor='black')
        ax.set_facecolor('black')
        x = np.linspace(0, 10, 100)
        y = np.sin(x) * np.exp(-x/10) + np.random.randn(100) * 0.1
        ax.plot(x, y, color='#FF6B6B', linewidth=2, alpha=0.7)
        ax.set_xlabel('Training Steps', color='white', fontsize=9)
        ax.set_ylabel('Reward', color='white', fontsize=9)
        ax.set_title('Noisy Learning Curve', color='white', fontsize=10)
        ax.tick_params(colors='white', labelsize=8)
        ax.spines['bottom'].set_color('white')
        ax.spines['left'].set_color('white')
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.grid(True, alpha=0.2, color='white')
        plt.tight_layout()
        st.pyplot(fig, use_container_width=True)
        plt.close(fig)
        
        st.markdown('</div>', unsafe_allow_html=True)
    
    with col3:
        st.markdown('<div class="concept-block">', unsafe_allow_html=True)
        st.markdown('<p class="handwritten-subtitle">Three Pillars of Modern RL</p>', unsafe_allow_html=True)
        
        st.markdown("""
        <div class="mini-card">
            <strong class="blue-text">PPO</strong> (Proximal Policy Optimization)<br>
            <span style="font-size: 12px;">Stable policy gradients via clipping.</span>
        </div>
        
        <div class="mini-card">
            <strong class="green-text">SAC</strong> (Soft Actor-Critic)<br>
            <span style="font-size: 12px;">Maximum entropy for exploration.</span>
        </div>
        
        <div class="mini-card">
            <strong class="yellow-text">DQN</strong> (Deep Q-Network)<br>
            <span style="font-size: 12px;">Value-based learning for discrete actions.</span>
        </div>
        """, unsafe_allow_html=True)
        st.markdown('</div>', unsafe_allow_html=True)
    
    # Arrow down
    st.markdown('<div class="arrow-down">↓</div>', unsafe_allow_html=True)
    
    # ========================================================================
    # MIDDLE SECTION - Highlight Box
    # ========================================================================
    
    st.markdown('<div class="highlight-box">', unsafe_allow_html=True)
    st.markdown('<p class="handwritten-subtitle" style="text-align: center; color: #FFD700;">Policy Optimization vs Value Optimization</p>', 
                unsafe_allow_html=True)
    
    col_left, col_right = st.columns(2)
    
    with col_left:
        st.markdown('<p style="text-align: center; font-weight: bold; color: #4DABF7;">Policy Gradient Approach</p>', 
                    unsafe_allow_html=True)
        
        fig = create_policy_gradient_diagram()
        st.pyplot(fig, use_container_width=True)
        plt.close(fig)
        
        st.markdown("""
        <ul style="font-size: 12px;">
            <li>Directly optimize policy π(a|s)</li>
            <li>Learn probability ratios</li>
            <li>Good for continuous actions</li>
            <li><span class="green-text">Examples: PPO, SAC</span></li>
        </ul>
        """, unsafe_allow_html=True)
    
    with col_right:
        st.markdown('<p style="text-align: center; font-weight: bold; color: #51CF66;">Value-Based Approach</p>', 
                    unsafe_allow_html=True)
        
        fig = create_value_function_diagram()
        st.pyplot(fig, use_container_width=True)
        plt.close(fig)
        
        st.markdown("""
        <ul style="font-size: 12px;">
            <li>Learn action values Q(s,a)</li>
            <li>Select action with max Q-value</li>
            <li>Good for discrete actions</li>
            <li><span class="yellow-text">Examples: DQN, DDQN</span></li>
        </ul>
        """, unsafe_allow_html=True)
    
    st.markdown('</div>', unsafe_allow_html=True)
    
    # Arrow down
    st.markdown('<div class="arrow-down">↓</div>', unsafe_allow_html=True)
    
    # ========================================================================
    # LOWER SECTION - Three Algorithm Blocks
    # ========================================================================
    
    algo_col1, algo_col2, algo_col3 = st.columns(3)
    
    with algo_col1:
        st.markdown('<div class="algo-block">', unsafe_allow_html=True)
        st.markdown('<p class="handwritten-subtitle" style="color: #4DABF7;">PPO — Stable Policy Gradients</p>', 
                    unsafe_allow_html=True)
        
        fig = create_ppo_diagram()
        st.pyplot(fig, use_container_width=True)
        plt.close(fig)
        
        st.markdown("""
        <p style="font-size: 13px; margin-top: 10px;">
        <strong>Key Innovation:</strong> Clipped surrogate objective
        </p>
        <ul style="font-size: 12px;">
            <li><span class="blue-text">Clipping</span>: Limits policy change to safe range</li>
            <li><span class="green-text">Advantages</span>: A(s,a) = Q(s,a) - V(s)</li>
            <li><span class="yellow-text">Entropy bonus</span>: Encourages exploration</li>
            <li>Multiple epochs on same data</li>
        </ul>
        <p style="font-size: 11px; font-style: italic; margin-top: 10px;">
        L<sup>CLIP</sup>(θ) = 𝔼[min(r·A, clip(r, 1-ε, 1+ε)·A)]
        </p>
        """, unsafe_allow_html=True)
        
        st.markdown('<div class="arrow-down">↓</div>', unsafe_allow_html=True)
        st.markdown('</div>', unsafe_allow_html=True)
    
    with algo_col2:
        st.markdown('<div class="algo-block">', unsafe_allow_html=True)
        st.markdown('<p class="handwritten-subtitle" style="color: #51CF66;">SAC — Maximum Entropy RL</p>', 
                    unsafe_allow_html=True)
        
        fig = create_sac_diagram()
        st.pyplot(fig, use_container_width=True)
        plt.close(fig)
        
        st.markdown("""
        <p style="font-size: 13px; margin-top: 10px;">
        <strong>Key Innovation:</strong> Entropy regularization
        </p>
        <ul style="font-size: 12px;">
            <li><span class="green-text">Gaussian policy</span>: Outputs mean & std</li>
            <li><span class="blue-text">Twin Q-networks</span>: Reduces overestimation</li>
            <li><span class="yellow-text">Entropy term</span>: α·H(π(·|s))</li>
            <li>Automatic temperature tuning</li>
        </ul>
        <p style="font-size: 11px; font-style: italic; margin-top: 10px;">
        J(π) = 𝔼[∑(r + α·H(π(·|s<sub>t</sub>)))]
        </p>
        """, unsafe_allow_html=True)
        
        st.markdown('<div class="arrow-down">↓</div>', unsafe_allow_html=True)
        st.markdown('</div>', unsafe_allow_html=True)
    
    with algo_col3:
        st.markdown('<div class="algo-block">', unsafe_allow_html=True)
        st.markdown('<p class="handwritten-subtitle" style="color: #FFD700;">DQN — Deep Value Learning</p>', 
                    unsafe_allow_html=True)
        
        fig = create_dqn_diagram()
        st.pyplot(fig, use_container_width=True)
        plt.close(fig)
        
        st.markdown("""
        <p style="font-size: 13px; margin-top: 10px;">
        <strong>Key Innovation:</strong> Neural network Q-function
        </p>
        <ul style="font-size: 12px;">
            <li><span class="yellow-text">Bellman update</span>: Bootstrap from next state</li>
            <li><span class="blue-text">Target network</span>: Stabilizes learning</li>
            <li><span class="red-text">Replay buffer</span>: Breaks correlation</li>
            <li>ε-greedy exploration</li>
        </ul>
        <p style="font-size: 11px; font-style: italic; margin-top: 10px;">
        L = 𝔼[(r + γ max<sub>a'</sub> Q<sub>target</sub>(s',a') - Q(s,a))<sup>2</sup>]
        </p>
        """, unsafe_allow_html=True)
        
        st.markdown('<div class="arrow-down">↓</div>', unsafe_allow_html=True)
        st.markdown('</div>', unsafe_allow_html=True)
    
    # ========================================================================
    # BOTTOM SECTION - Conclusion
    # ========================================================================
    
    st.markdown("---")
    st.markdown('<p class="handwritten-title" style="font-size: 32px;">The Landscape of Modern RL</p>', 
                unsafe_allow_html=True)
    
    conclusion_col1, conclusion_col2, conclusion_col3 = st.columns([1, 2, 1])
    
    with conclusion_col2:
        st.markdown("""
        <div style="text-align: center; padding: 20px; background-color: #1a1a1a; border-radius: 10px; border: 2px solid white;">
            <p style="font-size: 16px; margin: 10px 0;">
            <span class="blue-text">PPO</span> for stable on-policy learning<br>
            <span class="green-text">SAC</span> for sample-efficient continuous control<br>
            <span class="yellow-text">DQN</span> for discrete action spaces
            </p>
            <p style="font-size: 14px; margin-top: 20px; font-style: italic;">
            Each algorithm tackles RL's challenges differently,<br>
            but all aim for the same goal: <strong>intelligent behavior through trial and error.</strong>
            </p>
            <div style="margin-top: 20px; font-size: 30px;">
                → <span style="font-size: 14px; vertical-align: middle;">Future episodes: Actor-Critic methods, TRPO, TD3...</span>
            </div>
        </div>
        """, unsafe_allow_html=True)
    
    st.markdown("---")
    
    # Footer
    st.markdown("""
    <div style="text-align: center; margin-top: 30px; padding: 20px; font-size: 12px; opacity: 0.7;">
        <p>🎬 Storyboard-style RL Dashboard | Built with Streamlit</p>
        <p>Visualizing the core concepts of Proximal Policy Optimization, Soft Actor-Critic, and Deep Q-Networks</p>
    </div>
    """, unsafe_allow_html=True)

# ============================================================================
# ENTRY POINT
# ============================================================================

if __name__ == "__main__":
    main()
